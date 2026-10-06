"""v26 drops noise and recording from the degraded training epoch."""

import unittest

from train.rl_sample_strategy import (
    DEGRADED_BALANCED_SCENARIOS,
    FORMAL_RL_SCENARIOS,
    REGRESSED_SCENARIOS,
    degraded_balanced_indices,
    degraded_balanced_summary,
    degraded_skip_regressed_indices,
    validate_degraded_manifest,
)


def _row(scenario: str, group: str = "degraded", language: str = "en") -> dict[str, str]:
    return {"scenario": scenario, "condition_group": group, "language": language}


class SampleStrategyTest(unittest.TestCase):
    def test_noise_and_recording_are_the_regressed_set(self) -> None:
        self.assertEqual(REGRESSED_SCENARIOS, frozenset({"noise", "recording"}))

    def test_shipped_selector_keeps_other_degraded_rows_in_manifest_order(self) -> None:
        rows = [
            _row("noise"),
            _row("distortion"),
            _row("clean", group="clean"),
            _row("recording", language="zh"),
            _row("dropout"),
            _row("echo"),
            _row("far_field"),
            _row("obstructed", language="zh"),
            _row("Noise"),
            _row("noise_like"),
        ]
        self.assertEqual(degraded_skip_regressed_indices(rows), [1, 4, 5, 6, 7, 9])

    def test_clean_scenario_without_group_stays_out(self) -> None:
        rows = [
            {"scenario": "clean", "language": "en"},
            _row("dropout"),
        ]
        self.assertEqual(degraded_skip_regressed_indices(rows), [1])

    def test_empty_selection_raises(self) -> None:
        rows = [_row("noise"), _row("recording"), _row("clean", group="clean")]
        with self.assertRaises(ValueError):
            degraded_skip_regressed_indices(rows)

    def test_degraded_balanced_is_deterministic_and_excludes_clean(self) -> None:
        rows = []
        for language in ("en", "zh"):
            for scenario in DEGRADED_BALANCED_SCENARIOS:
                for index in range(10):
                    rows.append(_row(scenario, language=language))
        rows.extend([_row("clean", group="clean"), _row("clean", group="clean", language="zh")])

        first = degraded_balanced_indices(
            rows,
            epoch=0,
            seed=42,
            virtual_epoch_rows=200,
            max_repeat=4,
        )
        repeat = degraded_balanced_indices(
            rows,
            epoch=0,
            seed=42,
            virtual_epoch_rows=200,
            max_repeat=4,
        )
        next_epoch = degraded_balanced_indices(
            rows,
            epoch=1,
            seed=42,
            virtual_epoch_rows=200,
            max_repeat=4,
        )
        self.assertEqual(first, repeat)
        self.assertNotEqual(first, next_epoch)
        summary = degraded_balanced_summary(rows, first, virtual_epoch_rows=200)
        self.assertEqual(summary["selected_rows"], 200)
        self.assertEqual(summary["clean_rows"], 0)
        self.assertLessEqual(summary["max_repeat"], 4)
        self.assertEqual(summary["backfill_rows"], 0)
        self.assertEqual(
            set(summary["cell_counts"]),
            {f"{language}|{scenario}" for language in ("en", "zh") for scenario in DEGRADED_BALANCED_SCENARIOS},
        )
        self.assertEqual(summary["target_cell_counts"]["en|noise"], 20)

    def test_degraded_balanced_caps_rare_cell_repetition_and_backfills(self) -> None:
        rows = []
        for language in ("en", "zh"):
            for scenario in DEGRADED_BALANCED_SCENARIOS:
                count = 1 if language == "zh" and scenario == "noise" else 10
                for _ in range(count):
                    rows.append(_row(scenario, language=language))
        selected = degraded_balanced_indices(
            rows,
            epoch=0,
            seed=42,
            virtual_epoch_rows=200,
            max_repeat=4,
        )
        summary = degraded_balanced_summary(rows, selected, virtual_epoch_rows=200)
        self.assertEqual(summary["selected_rows"], 200)
        self.assertEqual(summary["clean_rows"], 0)
        self.assertLessEqual(summary["max_repeat"], 4)
        self.assertGreater(summary["cell_counts"]["zh|noise"], 0)
        self.assertEqual(summary["backfill_by_language"]["zh"], 16)

    def test_degraded_balanced_requires_all_cells(self) -> None:
        rows = [_row("noise", language="en"), _row("noise", language="zh")]
        with self.assertRaisesRegex(ValueError, "missing"):
            degraded_balanced_indices(rows, virtual_epoch_rows=20)

    def test_formal_manifest_contract_checks_size_and_cells(self) -> None:
        rows = [
            _row(scenario, language=language)
            for language in ("en", "zh")
            for scenario in FORMAL_RL_SCENARIOS
            for _ in range(2)
        ]
        summary = validate_degraded_manifest(rows, min_rows=32, min_cell_rows=2)
        self.assertEqual(summary["degraded_rows"], 32)
        self.assertEqual(summary["clean_rows"], 0)
        self.assertEqual(summary["cell_counts"]["zh|noise"], 2)

    def test_formal_manifest_contract_rejects_short_pool_and_clean_rows_do_not_count(self) -> None:
        rows = [
            _row(scenario, language=language)
            for language in ("en", "zh")
            for scenario in FORMAL_RL_SCENARIOS
        ]
        rows.append(_row("clean", group="clean"))
        with self.assertRaisesRegex(ValueError, "requires at least"):
            validate_degraded_manifest(rows, min_rows=33, min_cell_rows=1)

    def test_formal_manifest_contract_rejects_low_cell(self) -> None:
        rows = [
            _row(scenario, language=language)
            for language in ("en", "zh")
            for scenario in FORMAL_RL_SCENARIOS
            for _ in range(2)
        ]
        rows = [row for row in rows if not (row["language"] == "zh" and row["scenario"] == "noise")]
        with self.assertRaisesRegex(ValueError, "cells below"):
            validate_degraded_manifest(rows, min_rows=1, min_cell_rows=2)

    def test_formal_manifest_contract_accepts_mixed_and_can_bound_skew(self) -> None:
        rows = [
            _row(scenario, language=language)
            for language in ("en", "zh")
            for scenario in FORMAL_RL_SCENARIOS
        ]
        summary = validate_degraded_manifest(
            rows,
            min_rows=len(rows),
            min_cell_rows=1,
            max_cell_fraction=0.20,
        )
        self.assertEqual(summary["cell_counts"]["en|mixed"], 1)
        self.assertEqual(summary["max_cell_fraction"], 0.20)

    def test_formal_manifest_contract_rejects_skewed_cell(self) -> None:
        rows = [
            _row(scenario, language=language)
            for language in ("en", "zh")
            for scenario in FORMAL_RL_SCENARIOS
        ]
        rows.extend(_row("mixed", language="en") for _ in range(20))
        with self.assertRaisesRegex(ValueError, "max fraction"):
            validate_degraded_manifest(
                rows,
                min_rows=len(rows),
                min_cell_rows=1,
                max_cell_fraction=0.20,
            )

    def test_smoke_contract_can_skip_full_cell_coverage(self) -> None:
        summary = validate_degraded_manifest(
            [_row("distortion", language="en")],
            min_rows=1,
            min_cell_rows=1,
            require_all_cells=False,
        )
        self.assertFalse(summary["require_all_cells"])


if __name__ == "__main__":
    unittest.main()
