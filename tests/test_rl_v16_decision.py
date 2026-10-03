"""Driver follow-up for the v16 local-correction run."""

import unittest

from train.rl_v16_decision import followup_action


class V16FollowupTest(unittest.TestCase):
    def test_passed_gate_exports(self) -> None:
        action, _reason = followup_action(
            "CHUNK_DONE", 16, 24, 0.003, 0.0, "PASSED"
        )
        self.assertEqual(action, "passed")

    def test_robust_ceiling_stops(self) -> None:
        action, _reason = followup_action(
            "CHUNK_DONE", 4, 24, 0.0002, 0.0005, "FAILED"
        )
        self.assertEqual(action, "stop")

    def test_negative_reward_stops(self) -> None:
        action, _reason = followup_action(
            "BLOCKED_TRANSFER", 8, 24, -0.0001, 0.0, "FAILED"
        )
        self.assertEqual(action, "stop")

    def test_kl_stop_does_not_continue(self) -> None:
        action, _reason = followup_action(
            "STOPPED_KL", 6, 24, 0.0002, 0.0, "FAILED"
        )
        self.assertEqual(action, "stop")

    def test_search_block_does_not_continue(self) -> None:
        action, _reason = followup_action(
            "BLOCKED_SEARCH", 8, 24, 0.0002, 0.0, "FAILED"
        )
        self.assertEqual(action, "stop")

    def test_step4_small_robust_increase_continues(self) -> None:
        action, _reason = followup_action(
            "CHUNK_DONE", 4, 24, 0.0002, 0.0002, "FAILED"
        )
        self.assertEqual(action, "continue")

    def test_step8_robust_above_dpo_stops(self) -> None:
        action, reason = followup_action(
            "BLOCKED_TRANSFER", 8, 24, 0.0005, 0.000247, "FAILED"
        )
        self.assertEqual(action, "stop")
        self.assertIn("robust", reason)

    def test_step8_non_positive_robust_continues_through_transfer(self) -> None:
        action, _reason = followup_action(
            "BLOCKED_TRANSFER", 8, 24, 0.0004, 0.0, "FAILED"
        )
        self.assertEqual(action, "continue")

    def test_horizon_stops_a_short_positive_run(self) -> None:
        action, _reason = followup_action(
            "CHUNK_DONE", 24, 24, 0.0019, 0.0, "FAILED"
        )
        self.assertEqual(action, "stop")

    def test_mid_horizon_positive_run_continues(self) -> None:
        action, _reason = followup_action(
            "BLOCKED_TRANSFER", 12, 24, 0.0011, -0.0001, "FAILED"
        )
        self.assertEqual(action, "continue")

    def test_unknown_status_stops(self) -> None:
        action, _reason = followup_action(
            "TRAINING", 8, 24, 0.0004, 0.0, "FAILED"
        )
        self.assertEqual(action, "stop")


if __name__ == "__main__":
    unittest.main()
