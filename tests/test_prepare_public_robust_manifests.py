from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import prepare_public_robust_manifests as tool


ATOMIC = [
    "distortion",
    "dropout",
    "echo",
    "far_field",
    "noise",
    "obstructed",
    "recording",
]


def test_config() -> dict:
    return {
        "project": {"seed": 17, "data_root": ".", "output_dir": "."},
        "sources": {
            "robust": {
                "dataset_id": "example/robust",
                "revision": "a" * 40,
                "license": "apache-2.0",
                "expected_splits": 54,
                "expected_compound_splits": 47,
                "atomic_splits": ATOMIC,
                "required_source_fields": ["answer", "name"],
            }
        },
        "selection": {
            "robust_train": {"atomic_per_split": 2, "compound_total": 94},
            "robust_validation": {"atomic_per_split": 1, "compound_total": 47},
            "min_duration_s": 0.5,
            "max_duration_s": 30.0,
            "expected_train_rows": 2,
            "expected_validation_rows": 1,
            "expected_bench_rows": 1,
        },
        "curriculum": {
            "target_rows": 2,
            "maximum_error_rate": 0.7,
            "cumulative_thresholds": [0.3, 0.5, 0.7],
        },
        "canary": {"expected_rows": 8, "robust_rows": 6, "clean_rows": 2},
        "manifest": {
            "required_fields": [
                "sample_id",
                "audio",
                "answer",
                "language",
                "scenario",
                "condition_group",
                "audio_origin",
                "source_dataset",
                "source_revision",
                "source_split",
                "source_index",
                "source_utterance_id",
                "duration_s",
                "license",
                "seed",
                "audio_sha256",
            ]
        },
    }


def canonical_row(index: int, *, role: str = "train", language: str = "en") -> dict:
    return {
        "sample_id": f"sample-{role}-{index}",
        "audio": f"audio/{role}-{index}.wav",
        "answer": "same reference text",
        "language": language,
        "scenario": "noise",
        "condition_group": "atomic",
        "audio_origin": "synthetic",
        "source_dataset": "example/robust",
        "source_revision": "a" * 40,
        "source_split": "noise",
        "source_index": index,
        "source_utterance_id": f"source-{role}-{index}",
        "duration_s": 2.0,
        "license": "apache-2.0",
        "seed": 17,
        "audio_sha256": f"{index:064x}",
    }


class QuotaPlanTest(unittest.TestCase):
    def test_fixed_train_and_validation_compound_remainders(self) -> None:
        compounds = [f"compound_{index:02d}" for index in range(47)]
        split_names = [*ATOMIC, *compounds]

        train = tool.plan_scenario_quotas(split_names, ATOMIC, 16000, 48000)
        self.assertEqual(sum(train.values()), 160000)
        self.assertTrue(all(train[name] == 16000 for name in ATOMIC))
        self.assertTrue(all(train[name] == 1022 for name in compounds[:13]))
        self.assertTrue(all(train[name] == 1021 for name in compounds[13:]))

        validation = tool.plan_scenario_quotas(split_names, ATOMIC, 800, 2400)
        self.assertEqual(sum(validation.values()), 8000)
        self.assertTrue(all(validation[name] == 52 for name in compounds[:3]))
        self.assertTrue(all(validation[name] == 51 for name in compounds[3:]))

    def test_language_quota_is_balanced_and_deterministic(self) -> None:
        quotas = tool.plan_language_quotas({"b": 5, "a": 5, "c": 4})
        self.assertEqual(quotas[("a", "en")], 3)
        self.assertEqual(quotas[("b", "zh")], 3)
        self.assertEqual(quotas[("c", "en")], 2)
        self.assertEqual(sum(quotas.values()), 14)


class SelectionTest(unittest.TestCase):
    def test_stratified_selection_is_order_independent_and_excludes_sources(self) -> None:
        rows = []
        for index in range(12):
            row = canonical_row(index, language="en" if index % 2 == 0 else "zh")
            row["scenario"] = "noise" if index < 8 else "echo"
            rows.append(row)
        quotas = {
            ("noise", "en"): 2,
            ("noise", "zh"): 2,
            ("echo", "en"): 1,
            ("echo", "zh"): 1,
        }
        first = tool.select_stratified(rows, quotas, seed=9, namespace="fixture")
        second = tool.select_stratified(list(reversed(rows)), quotas, seed=9, namespace="fixture")
        self.assertEqual(
            [row["sample_id"] for row in first],
            [row["sample_id"] for row in second],
        )

        excluded = {first[0]["source_utterance_id"]}
        replacement = tool.select_stratified(
            rows,
            quotas,
            seed=9,
            namespace="fixture",
            excluded_source_ids=excluded,
        )
        self.assertNotIn(excluded.pop(), tool.selected_source_ids(replacement))

    def test_smoke_selection_has_128_train_rows_and_all_54_scenarios(self) -> None:
        config = test_config()
        config["smoke"] = {
            "robust_per_language_per_split": 1,
            "clean_per_language": 10,
            "expected_train_rows": 128,
            "bench_per_language_origin": 1,
            "expected_bench_rows": 4,
        }
        compounds = [f"compound_{index:02d}" for index in range(47)]
        robust = []
        index = 0
        for scenario in [*ATOMIC, *compounds]:
            for language in tool.LANGUAGES:
                row = canonical_row(index, language=language)
                row["scenario"] = scenario
                row["source_split"] = scenario
                robust.append(row)
                index += 1
        english = [canonical_row(1000 + offset, language="en") for offset in range(10)]
        chinese = [canonical_row(2000 + offset, language="zh") for offset in range(10)]
        for row in [*english, *chinese]:
            row["condition_group"] = "clean"
            row["scenario"] = "clean"
            row["audio_origin"] = "clean"
        bench = []
        for offset, (language, origin) in enumerate(
            (('en', 'real'), ('en', 'synthetic'), ('zh', 'real'), ('zh', 'synthetic'))
        ):
            row = canonical_row(3000 + offset, role="test", language=language)
            row["audio_origin"] = origin
            bench.append(row)

        selected = tool.build_smoke_selection(
            config, robust, english, chinese, bench
        )
        self.assertEqual(len(selected["train"]), 128)
        self.assertEqual(len(selected["test"]), 4)
        self.assertEqual(
            len({row["scenario"] for row in selected["train"] if row["scenario"] != "clean"}),
            54,
        )

    def test_validation_canary_is_balanced_deterministic_and_derived(self) -> None:
        config = test_config()
        validation = []
        index = 0
        for scenario in ("echo", "noise", "recording"):
            for language in tool.LANGUAGES:
                for _ in range(2):
                    row = canonical_row(index, role="validation", language=language)
                    row["scenario"] = scenario
                    row["source_split"] = scenario
                    validation.append(row)
                    index += 1
        for language in tool.LANGUAGES:
            row = canonical_row(index, role="validation", language=language)
            row["condition_group"] = "clean"
            row["scenario"] = "clean"
            row["source_split"] = "clean"
            row["audio_origin"] = "clean"
            validation.append(row)
            index += 1

        first = tool.build_validation_canary(config, validation)
        second = tool.build_validation_canary(config, list(reversed(validation)))
        self.assertEqual(
            [row["sample_id"] for row in first],
            [row["sample_id"] for row in second],
        )
        self.assertEqual(len(first), 8)
        self.assertEqual(
            sum(row["condition_group"] == "clean" for row in first), 2
        )
        self.assertEqual(Counter(row["language"] for row in first), {"en": 4, "zh": 4})
        self.assertLessEqual(
            {row["sample_id"] for row in first},
            {row["sample_id"] for row in validation},
        )


class CurriculumTest(unittest.TestCase):
    def test_evaluator_error_rate_supports_chinese_curriculum(self) -> None:
        row = canonical_row(1, language="zh")
        selected = tool.build_curriculum_rows(
            [row],
            [{"sample_id": row["sample_id"], "prediction": "文本", "error_rate": 0.2, "metric": "cer"}],
            target_rows=1,
            maximum_error_rate=0.7,
            seed=1,
        )
        self.assertEqual(selected[0]["base_error_rate"], 0.2)
        self.assertEqual(selected[0]["base_metric"], "cer")

    def test_curriculum_is_stratified_deterministic_and_cumulative(self) -> None:
        train = []
        scored = []
        rates = (0.1, 0.4, 0.6, 0.8)
        for index in range(48):
            language = "en" if index % 2 == 0 else "zh"
            row = canonical_row(index, language=language)
            row["scenario"] = f"scenario-{index % 3}"
            train.append(row)
            scored.append(
                {
                    "sample_id": row["sample_id"],
                    "base_prediction": f"prediction-{index}",
                    "base_error_rate": rates[index % len(rates)],
                    "base_metric": "wer" if language == "en" else "cer",
                }
            )

        first = tool.build_curriculum_rows(
            train,
            scored,
            target_rows=24,
            maximum_error_rate=0.7,
            seed=19,
        )
        second = tool.build_curriculum_rows(
            list(reversed(train)),
            list(reversed(scored)),
            target_rows=24,
            maximum_error_rate=0.7,
            seed=19,
        )
        self.assertEqual(
            [row["sample_id"] for row in first],
            [row["sample_id"] for row in second],
        )
        self.assertTrue(all(row["base_error_rate"] < 0.7 for row in first))
        views = tool.curriculum_views(first, [0.3, 0.5, 0.7])
        ids = [{row["sample_id"] for row in views[value]} for value in (0.3, 0.5, 0.7)]
        self.assertLessEqual(ids[0], ids[1])
        self.assertLessEqual(ids[1], ids[2])
        self.assertEqual(len(ids[2]), 24)

    def test_clean_scores_are_not_admitted(self) -> None:
        robust = canonical_row(1)
        clean = canonical_row(2)
        clean["condition_group"] = "clean"
        clean["scenario"] = "clean"
        scored = [
            {"sample_id": robust["sample_id"], "base_error_rate": 0.2, "base_metric": "wer"},
            {"sample_id": clean["sample_id"], "base_error_rate": 0.1, "base_metric": "wer"},
        ]
        selected = tool.build_curriculum_rows(
            [robust, clean], scored, target_rows=1, maximum_error_rate=0.7, seed=1
        )
        self.assertEqual(selected[0]["sample_id"], robust["sample_id"])


class ValidationTest(unittest.TestCase):
    def test_missing_audio_can_be_ignored_or_checked(self) -> None:
        config = test_config()
        row = canonical_row(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ignored = tool.validate_rows(
                {"train": [row]},
                config,
                data_root=root,
                audio_mode="ignore",
                check_counts=False,
            )
            self.assertTrue(ignored["hard_checks_pass"])
            checked = tool.validate_rows(
                {"train": [row]},
                config,
                data_root=root,
                audio_mode="exists",
                check_counts=False,
            )
            self.assertFalse(checked["hard_checks_pass"])
            self.assertTrue(any("audio does not exist" in error for error in checked["errors"]))

    def test_source_leakage_is_hard_but_transcript_overlap_is_report_only(self) -> None:
        config = test_config()
        train = canonical_row(1, role="train")
        validation = canonical_row(2, role="validation")
        report = tool.validate_rows(
            {"train": [train], "validation": [validation]},
            config,
            data_root=Path("."),
            audio_mode="ignore",
            check_counts=False,
        )
        self.assertTrue(report["hard_checks_pass"])
        self.assertEqual(report["normalized_transcript_overlap_report_only"]["train:validation"], 1)

        validation["source_utterance_id"] = train["source_utterance_id"]
        leaked = tool.validate_rows(
            {"train": [train], "validation": [validation]},
            config,
            data_root=Path("."),
            audio_mode="ignore",
            check_counts=False,
        )
        self.assertFalse(leaked["hard_checks_pass"])
        self.assertTrue(any("source_utterance_id" in error for error in leaked["errors"]))

    def test_curriculum_may_share_train_sample_ids(self) -> None:
        config = test_config()
        train = canonical_row(1)
        curriculum = dict(train)
        curriculum.update(
            {"base_prediction": "text", "base_error_rate": 0.2, "base_metric": "wer"}
        )
        report = tool.validate_rows(
            {"train": [train], "curriculum": [curriculum]},
            config,
            data_root=Path("."),
            audio_mode="ignore",
            check_counts=False,
        )
        self.assertTrue(report["hard_checks_pass"], report["errors"])


class ProbeTest(unittest.TestCase):
    def test_probe_passes_revision_and_never_loads_rows(self) -> None:
        captured = {}

        class FakeSplit:
            num_examples = 100

        def fake_loader(dataset_id: str, **kwargs: object) -> SimpleNamespace:
            captured["dataset_id"] = dataset_id
            captured.update(kwargs)
            info = SimpleNamespace(
                splits={f"split-{index}": FakeSplit() for index in range(54)},
                features={"answer": object(), "name": object()},
                download_size=123,
                dataset_size=456,
            )
            return SimpleNamespace(info=info)

        source = test_config()["sources"]["robust"]
        report = tool.probe_dataset_source("robust", source, loader=fake_loader)
        self.assertTrue(report["passed"])
        self.assertFalse(report["audio_downloaded"])
        self.assertEqual(captured["revision"], "a" * 40)
        self.assertEqual(captured["dataset_id"], "example/robust")

    def test_probe_quota_capacity_uses_train_plus_validation(self) -> None:
        config = test_config()
        compounds = [f"compound_{index:02d}" for index in range(47)]
        report = {
            "split_rows": {name: 3 for name in [*ATOMIC, *compounds]},
            "errors": [],
            "passed": True,
        }
        tool.add_robust_quota_capacity_check(report, config)
        self.assertTrue(report["quota_total_capacity_passed"])
        report["split_rows"][ATOMIC[0]] = 2
        report["errors"] = []
        report["passed"] = True
        tool.add_robust_quota_capacity_check(report, config)
        self.assertFalse(report["quota_total_capacity_passed"])
        self.assertFalse(report["passed"])


class CliTest(unittest.TestCase):
    def test_top_level_and_subcommand_help(self) -> None:
        script = ROOT / "scripts" / "prepare_public_robust_manifests.py"
        for arguments in (["--help"], ["curriculum", "--help"], ["probe", "--help"]):
            completed = subprocess.run(
                [sys.executable, str(script), *arguments],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("usage:", completed.stdout)

    def test_curriculum_and_validate_cli_fixture(self) -> None:
        script = ROOT / "scripts" / "prepare_public_robust_manifests.py"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.json"
            train_path = root / "train.jsonl"
            scored_path = root / "scored.jsonl"
            output_path = root / "curriculum.jsonl"
            tool.write_json(config_path, test_config())
            train = [canonical_row(index, language="en" if index % 2 == 0 else "zh") for index in range(4)]
            scored = [
                {
                    "sample_id": row["sample_id"],
                    "base_prediction": "fixture prediction",
                    "base_error_rate": 0.1 + index * 0.1,
                    "base_metric": "wer" if row["language"] == "en" else "cer",
                }
                for index, row in enumerate(train)
            ]
            tool.write_jsonl(train_path, train)
            tool.write_jsonl(scored_path, scored)

            built = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "curriculum",
                    "--config",
                    str(config_path),
                    "--train",
                    str(train_path),
                    "--scored",
                    str(scored_path),
                    "--output",
                    str(output_path),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(built.returncode, 0, built.stderr)
            self.assertEqual(len(tool.read_jsonl(output_path)), 2)
            self.assertTrue((root / "curriculum.lt_0_30.jsonl").exists())
            self.assertTrue((root / "curriculum.lt_0_50.jsonl").exists())
            self.assertTrue((root / "curriculum.lt_0_70.jsonl").exists())

            validated = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "validate",
                    "--config",
                    str(config_path),
                    "--train",
                    str(train_path),
                    "--curriculum",
                    str(output_path),
                    "--audio-mode",
                    "ignore",
                    "--skip-counts",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(validated.returncode, 0, validated.stderr)


if __name__ == "__main__":
    unittest.main()
