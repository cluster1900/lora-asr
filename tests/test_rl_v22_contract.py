"""v22 resumes the v21 step-4 weights and keeps that run's levers."""

import unittest
from pathlib import Path

from train.rl_v16_decision import followup_action

CONFIG = Path("configs/train/qwen3_asr_rl_v22.yaml")
DRIVER = Path("scripts/run_rl_pilot_v22.sh")


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


class V22ContractTest(unittest.TestCase):
    def test_shipped_config_matches_the_v21_levers(self) -> None:
        text = CONFIG.read_text(encoding="utf-8")
        self.assertAlmostEqual(float(_scalar(text, "learning_rate")), 1.0e-5)
        self.assertEqual(int(_scalar(text, "max_steps")), 24)
        self.assertAlmostEqual(float(_scalar(text, "max_raw_kl")), 5.0e-4)
        self.assertEqual(_scalar(text, "mode"), "raw_gap")
        self.assertAlmostEqual(float(_scalar(text, "local_max_relative")), 0.35)
        self.assertAlmostEqual(float(_scalar(text, "beta")), 0.04)
        self.assertEqual(_scalar(text, "train_audio_projections"), "false")
        self.assertEqual(int(_scalar(text, "expected_total_targets")), 196)
        self.assertNotIn("5.0e-6", text)
        self.assertNotIn("mode: unit", text)
        self.assertNotIn("mode: capped_gap", text)
        self.assertNotIn("train_audio_projections: true", text)

    def test_driver_resumes_v21_step4_only(self) -> None:
        text = DRIVER.read_text(encoding="utf-8")
        self.assertIn(
            "/data/mega-asr/runs/rl_pilot_v21/checkpoints/step_4",
            text,
        )
        self.assertIn('rm -rf "$V22_RUN"', text)
        self.assertNotIn('rm -rf "$V21_RUN"', text)

    def test_negative_step8_stops(self) -> None:
        action, reason = followup_action(
            "CHUNK_DONE",
            8,
            24,
            -0.0002,
            -0.00009,
            "FAILED",
        )
        self.assertEqual(action, "stop")
        self.assertIn("fell below", reason)

    def test_non_negative_step8_below_dpo_continues(self) -> None:
        action, reason = followup_action(
            "CHUNK_DONE",
            8,
            24,
            0.0001,
            -0.00009,
            "FAILED",
        )
        self.assertEqual(action, "continue")
        self.assertIn("local-correction", reason)

    def test_step8_robust_back_above_dpo_stops(self) -> None:
        action, reason = followup_action(
            "BLOCKED_TRANSFER",
            8,
            24,
            0.0001,
            0.000014,
            "FAILED",
        )
        self.assertEqual(action, "stop")
        self.assertIn("robust still above", reason)


if __name__ == "__main__":
    unittest.main()
