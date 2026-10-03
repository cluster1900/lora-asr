"""v18 keeps raw-gap local corrections and the non-rising Robust follow-up."""

import unittest
from pathlib import Path

from train.rl_v17_decision import followup_action

CONFIG = Path("configs/train/qwen3_asr_rl_v18.yaml")


class V18ContractTest(unittest.TestCase):
    def test_shipped_config_resumes_raw_gap(self) -> None:
        text = CONFIG.read_text(encoding="utf-8")
        self.assertIn("learning_rate: 1.0e-5", text)
        self.assertIn("mode: raw_gap", text)
        self.assertNotIn("mode: unit", text)
        self.assertIn("local_max_relative: 0.35", text)
        self.assertIn("beta: 0.04", text)
        self.assertIn("max_raw_kl: 5.0e-4", text)
        self.assertIn("max_steps: 48", text)

    def test_v16_step8_numbers_continue(self) -> None:
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

    def test_later_rise_above_v16_step8_stops(self) -> None:
        action, reason = followup_action(
            "BLOCKED_SEARCH",
            12,
            48,
            0.0006,
            0.0002,
            "FAILED",
            0.000164,
            16.0,
        )
        self.assertEqual(action, "stop")
        self.assertIn("rose", reason)


if __name__ == "__main__":
    unittest.main()
