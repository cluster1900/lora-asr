"""v26 drops noise and recording from the degraded training epoch."""

import unittest

from train.rl_sample_strategy import (
    REGRESSED_SCENARIOS,
    degraded_skip_regressed_indices,
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


if __name__ == "__main__":
    unittest.main()
