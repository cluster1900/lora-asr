"""v20 lowers the learning rate and keeps the raw reward gap."""

import unittest
from pathlib import Path

from train.rl_v17_decision import followup_action

CONFIG = Path("configs/train/qwen3_asr_rl_v20.yaml")


def _scalar(text: str, key: str) -> str:
    """Return one uncommented ``key: value`` scalar from the shipped config."""
    found = []
    for line in text.splitlines():
        body = line.split("#", 1)[0].strip()
        prefix = f"{key}:"
        if body.startswith(prefix):
            found.append(body[len(prefix):].strip())
    if len(found) != 1:
        raise AssertionError(f"{key} appears {len(found)} times")
    return found[0]


class V20ContractTest(unittest.TestCase):
    def test_shipped_config_halves_the_learning_rate(self) -> None:
        text = CONFIG.read_text(encoding="utf-8")
        self.assertAlmostEqual(float(_scalar(text, "learning_rate")), 5.0e-6)
        self.assertEqual(int(_scalar(text, "max_steps")), 80)
        self.assertAlmostEqual(float(_scalar(text, "max_raw_kl")), 5.0e-4)
        self.assertEqual(_scalar(text, "mode"), "raw_gap")
        self.assertAlmostEqual(float(_scalar(text, "local_max_relative")), 0.35)
        self.assertAlmostEqual(float(_scalar(text, "beta")), 0.04)
        self.assertNotIn("1.0e-5", text)
        self.assertNotIn("mode: unit", text)
        self.assertNotIn("mode: capped_gap", text)

    def test_positive_step4_continues(self) -> None:
        action, reason = followup_action(
            "CHUNK_DONE",
            4,
            80,
            0.0002,
            0.0001,
            "FAILED",
            None,
            7.0,
        )
        self.assertEqual(action, "continue")
        self.assertIn("robust", reason)

    def test_negative_reward_stops(self) -> None:
        action, reason = followup_action(
            "CHUNK_DONE",
            4,
            80,
            -0.0001,
            0.0001,
            "FAILED",
            None,
            7.0,
        )
        self.assertEqual(action, "stop")
        self.assertIn("fell below", reason)


if __name__ == "__main__":
    unittest.main()
