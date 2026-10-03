"""Driver follow-up and config contract for the v17 local-unit run."""

import unittest
from pathlib import Path

from train.rl_local_winner import local_winner_index
from train.rl_pair_advantage import winner_advantage
from train.rl_v17_decision import followup_action

CONFIG = Path("configs/train/qwen3_asr_rl_v17.yaml")


class V17FollowupTest(unittest.TestCase):
    def test_passed_gate_exports(self) -> None:
        action, _reason = followup_action(
            "CHUNK_DONE", 40, 48, 0.0021, -0.0001, "PASSED", 0.0, 30.0
        )
        self.assertEqual(action, "passed")

    def test_robust_ceiling_stops_even_when_it_fell(self) -> None:
        action, reason = followup_action(
            "CHUNK_DONE", 8, 48, 0.0004, 0.0005, "FAILED", 0.0008, 12.0
        )
        self.assertEqual(action, "stop")
        self.assertIn("robust", reason)

    def test_negative_reward_stops(self) -> None:
        action, _reason = followup_action(
            "BLOCKED_TRANSFER", 8, 48, -0.0001, 0.0001, "FAILED", 0.0003, 12.0
        )
        self.assertEqual(action, "stop")

    def test_kl_stop_does_not_continue(self) -> None:
        action, _reason = followup_action(
            "STOPPED_KL", 12, 48, 0.0004, 0.0001, "FAILED", 0.0002, 16.0
        )
        self.assertEqual(action, "stop")

    def test_step4_without_prior_continues(self) -> None:
        action, _reason = followup_action(
            "CHUNK_DONE", 4, 48, 0.0, 0.000321, "FAILED", None, 7.43
        )
        self.assertEqual(action, "continue")

    def test_v16_step8_falling_robust_continues(self) -> None:
        action, reason = followup_action(
            "BLOCKED_TRANSFER",
            8,
            48,
            0.0004,
            0.000164,
            "FAILED",
            0.000321,
            11.724898,
        )
        self.assertEqual(action, "continue")
        self.assertIn("robust", reason)

    def test_robust_rise_stops(self) -> None:
        action, reason = followup_action(
            "BLOCKED_TRANSFER", 8, 48, 0.0004, 0.000247, "FAILED", 0.000164, 12.0
        )
        self.assertEqual(action, "stop")
        self.assertIn("rose", reason)

    def test_flat_robust_continues(self) -> None:
        action, _reason = followup_action(
            "BLOCKED_TRANSFER", 12, 48, 0.0006, 0.000164, "FAILED", 0.000164, 16.0
        )
        self.assertEqual(action, "continue")

    def test_missing_prior_at_step8_stops(self) -> None:
        action, reason = followup_action(
            "BLOCKED_TRANSFER", 8, 48, 0.0004, 0.000164, "FAILED", None, 12.0
        )
        self.assertEqual(action, "stop")
        self.assertIn("prior", reason)

    def test_step12_search_above_floor_continues(self) -> None:
        action, _reason = followup_action(
            "BLOCKED_SEARCH", 12, 48, 0.0006, 0.0001, "FAILED", 0.000164, 16.0
        )
        self.assertEqual(action, "continue")

    def test_step12_search_below_floor_stops(self) -> None:
        action, reason = followup_action(
            "BLOCKED_SEARCH", 12, 48, 0.0006, 0.0001, "FAILED", 0.000164, 8.49
        )
        self.assertEqual(action, "stop")
        self.assertIn("search", reason)

    def test_search_floor_boundary_continues(self) -> None:
        action, _reason = followup_action(
            "BLOCKED_SEARCH", 12, 48, 0.0006, 0.0001, "FAILED", 0.000164, 8.50
        )
        self.assertEqual(action, "continue")

    def test_horizon_stops_a_still_short_run(self) -> None:
        action, reason = followup_action(
            "CHUNK_DONE", 48, 48, 0.0019, 0.0001, "FAILED", 0.0002, 40.0
        )
        self.assertEqual(action, "stop")
        self.assertIn("horizon", reason)

    def test_step24_short_of_old_horizon_continues(self) -> None:
        action, _reason = followup_action(
            "BLOCKED_TRANSFER", 24, 48, 0.0012, -0.0001, "FAILED", 0.0, 28.0
        )
        self.assertEqual(action, "continue")

    def test_unknown_status_stops(self) -> None:
        action, _reason = followup_action(
            "TRAINING", 8, 48, 0.0004, 0.0, "FAILED", 0.0001, 12.0
        )
        self.assertEqual(action, "stop")

    def test_search_above_floor_still_stops_when_robust_rises(self) -> None:
        action, reason = followup_action(
            "BLOCKED_SEARCH", 12, 48, 0.0006, 0.0003, "FAILED", 0.000164, 16.0
        )
        self.assertEqual(action, "stop")
        self.assertIn("rose", reason)

    def test_local_winner_then_unit_advantage_is_one(self) -> None:
        texts = [
            "Ayr has won eight of the last nine meetings in this series.",
            "Iowa has won eight of the last nine meetings in this series.",
            "In the difficult moments we recognize our thirst for fulfillment.",
        ]
        rewards = [0.40, 0.46, 0.95]
        index, status = local_winner_index(texts, rewards, language="en", max_relative=0.35)
        self.assertEqual(index, 1)
        self.assertEqual(status, "update")
        self.assertEqual(winner_advantage(rewards[0], rewards[index], "unit"), 1.0)

    def test_shipped_config_is_unit_on_local_corrections(self) -> None:
        text = CONFIG.read_text(encoding="utf-8")
        self.assertIn("learning_rate: 1.0e-5", text)
        self.assertIn("mode: unit", text)
        self.assertNotIn("mode: raw_gap", text)
        self.assertIn("local_max_relative: 0.35", text)
        self.assertIn("beta: 0.04", text)
        self.assertIn("max_raw_kl: 5.0e-4", text)
        self.assertIn("max_steps: 48", text)


if __name__ == "__main__":
    unittest.main()
