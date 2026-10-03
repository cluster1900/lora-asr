"""Winner-advantage modes used by rl_pilot_v12 and rl_pilot_v13."""

import unittest

from train.rl_pair_advantage import winner_advantage


class WinnerAdvantageTest(unittest.TestCase):
    def test_unit_ignores_the_gap(self) -> None:
        self.assertEqual(winner_advantage(0.80, 0.875, "unit"), 1.0)

    def test_raw_gap_is_the_reward_difference(self) -> None:
        self.assertAlmostEqual(winner_advantage(0.80, 0.875, "raw_gap"), 0.075)

    def test_raw_gap_cannot_go_negative(self) -> None:
        self.assertEqual(winner_advantage(0.90, 0.80, "raw_gap"), 0.0)

    def test_fixed_uses_the_configured_constant(self) -> None:
        self.assertEqual(winner_advantage(0.10, 0.90, "fixed", 0.10), 0.10)

    def test_fixed_rejects_values_outside_the_open_unit_interval(self) -> None:
        with self.assertRaises(ValueError):
            winner_advantage(0.10, 0.20, "fixed", 0.0)
        with self.assertRaises(ValueError):
            winner_advantage(0.10, 0.20, "fixed", 1.5)

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            winner_advantage(0.10, 0.20, "scaled")

    def test_capped_gap_keeps_small_gaps_and_limits_large_ones(self) -> None:
        self.assertAlmostEqual(
            winner_advantage(0.80, 0.83, "capped_gap", cap=0.05), 0.03
        )
        self.assertAlmostEqual(
            winner_advantage(0.80, 0.95, "capped_gap", 0.10, 0.05), 0.05
        )
        self.assertEqual(winner_advantage(0.90, 0.80, "capped_gap", cap=0.05), 0.0)
        self.assertAlmostEqual(
            winner_advantage(0.80, 0.95, "raw_gap", 0.10, 0.05), 0.15
        )

    def test_capped_gap_rejects_cap_outside_the_open_unit_interval(self) -> None:
        with self.assertRaises(ValueError):
            winner_advantage(0.10, 0.20, "capped_gap", cap=0.0)
        with self.assertRaises(ValueError):
            winner_advantage(0.10, 0.20, "capped_gap", cap=1.5)


if __name__ == "__main__":
    unittest.main()
