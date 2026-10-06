"""Paired significance tool on synthetic scored rows."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.paired_significance import (
    bootstrap_ci,
    compare,
    load_scored,
    load_scored_many,
    main,
    pair_rows,
    paired_gate,
    sign_test_p,
    tail_gate,
)


def _row(sample_id: str, rate: float, text: str = "a", scenario: str = "noise",
         group: str = "degraded", language: str = "en") -> dict:
    return {
        "sample_id": sample_id,
        "error_rate": rate,
        "prediction_normalized": text,
        "scenario": scenario,
        "condition_group": group,
        "language": language,
    }


def _table(rows: list[dict]) -> dict:
    return {row["sample_id"]: row for row in rows}


class SignTestTest(unittest.TestCase):
    def test_no_changes_is_p_one(self) -> None:
        self.assertEqual(sign_test_p(0, 0), 1.0)

    def test_balanced_is_not_significant(self) -> None:
        self.assertEqual(sign_test_p(6, 5), 1.0)

    def test_all_better_is_significant(self) -> None:
        self.assertAlmostEqual(sign_test_p(10, 0), 2 / 1024)


class CompareTest(unittest.TestCase):
    def test_identical_inputs_have_zero_delta_and_noise_verdict(self) -> None:
        rows = [_row(f"s{i}", 0.1 * (i % 3)) for i in range(30)]
        report = compare(_table(rows), _table(rows), iterations=200)
        overall = report["overall"]
        self.assertEqual(overall["mean_delta"], 0.0)
        self.assertEqual(overall["changed_predictions"], 0)
        self.assertEqual(overall["sign_test_p"], 1.0)
        self.assertEqual(overall["verdict"], "indistinguishable_from_noise")

    def test_uniform_improvement_is_significant(self) -> None:
        base = [_row(f"s{i}", 0.5, text="old") for i in range(40)]
        cand = [_row(f"s{i}", 0.4, text="new") for i in range(40)]
        report = compare(_table(base), _table(cand), iterations=200)
        overall = report["overall"]
        self.assertAlmostEqual(overall["mean_delta"], -0.1)
        self.assertEqual(overall["better"], 40)
        self.assertEqual(overall["changed_predictions"], 40)
        self.assertEqual(overall["verdict"], "significant_improvement")

    def test_few_mixed_changes_stay_indistinguishable(self) -> None:
        base = [_row(f"s{i}", 0.2) for i in range(200)]
        cand = [dict(row) for row in base]
        for index in range(6):
            cand[index]["error_rate"] = 0.1
            cand[index]["prediction_normalized"] = "better"
        for index in range(6, 11):
            cand[index]["error_rate"] = 0.3
            cand[index]["prediction_normalized"] = "worse"
        report = compare(_table(base), _table(cand), iterations=300)
        overall = report["overall"]
        self.assertEqual((overall["better"], overall["worse"]), (6, 5))
        self.assertEqual(overall["verdict"], "indistinguishable_from_noise")

    def test_groups_are_reported(self) -> None:
        base = [_row("a", 0.3, scenario="noise"), _row("b", 0.3, scenario="echo", language="zh")]
        cand = [_row("a", 0.3, scenario="noise"), _row("b", 0.3, scenario="echo", language="zh")]
        report = compare(_table(base), _table(cand), iterations=100)
        self.assertEqual(set(report["by_language_scenario"]), {"en|noise", "zh|echo"})
        self.assertEqual(set(report["by_condition_group"]), {"degraded"})

    def test_same_seed_is_reproducible(self) -> None:
        deltas = [0.1, -0.2, 0.0, 0.05, -0.03, 0.2]
        self.assertEqual(
            bootstrap_ci(deltas, iterations=200, seed=7),
            bootstrap_ci(deltas, iterations=200, seed=7),
        )


class InputValidationTest(unittest.TestCase):
    def test_mismatched_sample_ids_raise(self) -> None:
        with self.assertRaises(ValueError):
            pair_rows(_table([_row("a", 0.1)]), _table([_row("b", 0.1)]))

    def test_metadata_mismatch_raise(self) -> None:
        baseline = _table([_row("a", 0.1, scenario="noise")])
        candidate = _table([_row("a", 0.1, scenario="echo")])
        with self.assertRaisesRegex(ValueError, "metadata mismatch.*scenario"):
            pair_rows(baseline, candidate)

    def test_duplicate_and_missing_ids_raise(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dup = Path(tmp) / "dup.jsonl"
            dup.write_text(
                json.dumps(_row("a", 0.1)) + "\n" + json.dumps(_row("a", 0.2)) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_scored(dup)
            missing = Path(tmp) / "missing.jsonl"
            missing.write_text(json.dumps({"error_rate": 0.1}) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_scored(missing)

    def test_cli_writes_report_and_does_not_touch_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.jsonl"
            cand = Path(tmp) / "cand.jsonl"
            rows = [_row(f"s{i}", 0.2) for i in range(20)]
            text = "".join(json.dumps(row) + "\n" for row in rows)
            base.write_text(text, encoding="utf-8")
            cand.write_text(text, encoding="utf-8")
            out = Path(tmp) / "out" / "report.json"
            code = main(
                ["--baseline", str(base), "--candidate", str(cand),
                 "--output", str(out), "--bootstrap", "100"]
            )
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(out.read_text())["overall"]["samples"], 20)
            self.assertEqual(base.read_text(encoding="utf-8"), text)

    def test_cli_rejects_mismatch_with_exit_code_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "base.jsonl"
            cand = Path(tmp) / "cand.jsonl"
            base.write_text(json.dumps(_row("a", 0.1)) + "\n", encoding="utf-8")
            cand.write_text(json.dumps(_row("b", 0.1)) + "\n", encoding="utf-8")
            self.assertEqual(
                main(["--baseline", str(base), "--candidate", str(cand), "--bootstrap", "100"]),
                2,
            )


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class V31ExtensionsTest(unittest.TestCase):
    """docs/qwen3-asr/29_rl_v31_scale_design.md: merged verdict sets and gate."""

    def test_language_condition_groups(self) -> None:
        rows = [
            _row("a", 0.3, language="en", group="degraded"),
            _row("b", 0.3, language="zh", group="degraded"),
            _row("c", 0.0, language="zh", group="clean", scenario="clean"),
        ]
        report = compare(_table(rows), _table(rows), iterations=100)
        self.assertEqual(
            set(report["by_language_condition"]), {"en|degraded", "zh|degraded", "zh|clean"}
        )

    def test_load_many_merges_and_rejects_cross_file_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            one = _write(Path(tmp) / "one.jsonl", [_row("a", 0.1), _row("b", 0.2)])
            two = _write(Path(tmp) / "two.jsonl", [_row("c", 0.3)])
            self.assertEqual(set(load_scored_many([one, two])), {"a", "b", "c"})
            dup = _write(Path(tmp) / "dup.jsonl", [_row("b", 0.5)])
            with self.assertRaisesRegex(ValueError, "appears in both"):
                load_scored_many([one, dup])

    def _report(self, degraded_delta: float, clean_delta: float, n: int = 60) -> dict:
        base, cand = [], []
        for i in range(n):
            base.append(_row(f"d{i}", 0.3, text="x"))
            cand.append(_row(f"d{i}", 0.3 + degraded_delta, text="y" if degraded_delta else "x"))
            base.append(_row(f"c{i}", 0.02, group="clean", scenario="clean"))
            cand.append(_row(f"c{i}", 0.02 + clean_delta, group="clean", scenario="clean"))
        return compare(_table(base), _table(cand), iterations=200)

    def test_gate_passes_on_significant_degraded_gain_and_noninferior_clean(self) -> None:
        gate = paired_gate(self._report(-0.05, 0.0))
        self.assertEqual(gate["status"], "PASSED", gate)
        self.assertEqual(gate["failures"], [])

    def test_gate_fails_when_degraded_is_not_significant(self) -> None:
        gate = paired_gate(self._report(0.0, 0.0))
        self.assertEqual(gate["status"], "FAILED")
        self.assertTrue(any("degraded" in f for f in gate["failures"]))

    def test_gate_fails_on_clean_regression_beyond_margin(self) -> None:
        gate = paired_gate(self._report(-0.05, 0.01))
        self.assertEqual(gate["status"], "FAILED")
        self.assertTrue(any("clean" in f for f in gate["failures"]))
        self.assertEqual(paired_gate(self._report(-0.05, 0.01), clean_margin=0.02)["status"], "PASSED")

    def test_gate_fails_when_a_group_is_missing(self) -> None:
        rows = [_row(f"d{i}", 0.3) for i in range(10)]
        gate = paired_gate(compare(_table(rows), _table(rows), iterations=100))
        self.assertEqual(gate["status"], "FAILED")
        self.assertIn("no clean rows", gate["failures"])

    def test_cli_accepts_several_files_and_writes_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp)
            b1 = _write(t / "b1.jsonl", [_row(f"d{i}", 0.3, text="x") for i in range(40)])
            b2 = _write(t / "b2.jsonl", [_row(f"c{i}", 0.02, group="clean", scenario="clean") for i in range(40)])
            c1 = _write(t / "c1.jsonl", [_row(f"d{i}", 0.2, text="y") for i in range(40)])
            c2 = _write(t / "c2.jsonl", [_row(f"c{i}", 0.02, group="clean", scenario="clean") for i in range(40)])
            out = t / "report.json"
            code = main([
                "--baseline", str(b1), str(b2), "--candidate", str(c1), str(c2),
                "--gate", "--output", str(out), "--bootstrap", "100",
            ])
            self.assertEqual(code, 0)
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["overall"]["samples"], 80)
            self.assertEqual(report["gate"]["status"], "PASSED")
            self.assertEqual(len(report["inputs"]["baseline"]), 2)

    def test_tail_gate_rejects_new_severe_output_and_contract_drift(self) -> None:
        base = _row("a", 0.1)
        cand = _row("a", 2.5, text="loop")
        cand.update({"too_long": True, "hallucination_like": True})
        report = tail_gate(
            [(dict(base, decoding={"max_new_tokens": 512, "do_sample": False, "num_beams": 1, "num_return_sequences": 1}),
              dict(cand, decoding={"max_new_tokens": 512, "do_sample": False, "num_beams": 1, "num_return_sequences": 1}))],
            required_max_new_tokens=512,
        )
        self.assertEqual(report["status"], "FAILED")
        self.assertEqual(report["severe_regressions"], 1)

    def test_tail_gate_passes_matching_rows(self) -> None:
        contract = {"max_new_tokens": 512, "do_sample": False, "num_beams": 1, "num_return_sequences": 1}
        row = dict(_row("a", 0.1), decoding=contract)
        self.assertEqual(tail_gate([(row, dict(row))], required_max_new_tokens=512)["status"], "PASSED")


if __name__ == "__main__":
    unittest.main()
