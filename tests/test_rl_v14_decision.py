"""Driver follow-up for the v14 continuation and the optional v15 learning rate."""

import unittest

from train.rl_v14_decision import followup_action


class FollowupActionTest(unittest.TestCase):
    def test_passed_gate_exports(self) -> None:
        action, _reason = followup_action(
            "CHUNK_DONE", 8, 12, 0.003, 0.0, "PASSED", True
        )
        self.assertEqual(action, "passed")

    def test_robust_stop_blocks_the_switch(self) -> None:
        action, _reason = followup_action(
            "BLOCKED_TRANSFER", 8, 12, 0.0002, 0.0005, "FAILED", True
        )
        self.assertEqual(action, "stop")

    def test_negative_reward_does_not_raise_the_learning_rate(self) -> None:
        action, _reason = followup_action(
            "BLOCKED_TRANSFER", 8, 12, -0.0001, 0.0, "FAILED", True
        )
        self.assertEqual(action, "stop")

    def test_kl_stop_does_not_switch(self) -> None:
        action, _reason = followup_action(
            "STOPPED_KL", 6, 12, 0.0002, 0.0, "FAILED", True
        )
        self.assertEqual(action, "stop")

    def test_search_block_does_not_switch(self) -> None:
        action, _reason = followup_action(
            "BLOCKED_SEARCH", 8, 12, 0.0002, 0.0, "FAILED", True
        )
        self.assertEqual(action, "stop")

    def test_transfer_block_with_a_small_positive_reward_switches(self) -> None:
        action, reason = followup_action(
            "BLOCKED_TRANSFER", 8, 12, 0.0002, 5.9e-05, "FAILED", True
        )
        self.assertEqual(action, "switch")
        self.assertIn("blocked transfer", reason)

    def test_transfer_block_without_permission_stops(self) -> None:
        action, _reason = followup_action(
            "BLOCKED_TRANSFER", 8, 12, 0.0002, 0.0, "FAILED", False
        )
        self.assertEqual(action, "stop")

    def test_transfer_block_cleared_by_the_gate_continues(self) -> None:
        action, _reason = followup_action(
            "BLOCKED_TRANSFER", 8, 12, 0.001, 0.0, "FAILED", True
        )
        self.assertEqual(action, "continue")

    def test_chunk_done_below_the_horizon_continues(self) -> None:
        action, _reason = followup_action(
            "CHUNK_DONE", 4, 12, 0.0002, 0.0, "FAILED", False
        )
        self.assertEqual(action, "continue")

    def test_chunk_done_at_the_horizon_stops(self) -> None:
        action, _reason = followup_action(
            "CHUNK_DONE", 12, 12, 0.0015, 0.0, "FAILED", True
        )
        self.assertEqual(action, "stop")

    def test_unknown_status_stops(self) -> None:
        action, _reason = followup_action(
            "TRAINING", 8, 12, 0.0002, 0.0, "FAILED", True
        )
        self.assertEqual(action, "stop")


if __name__ == "__main__":
    unittest.main()
