"""v23 resumes v22 step 8 and applies a lower learning rate after resume."""

import unittest
from pathlib import Path

from train.rl_v16_decision import followup_action

CONFIG = Path("configs/train/qwen3_asr_rl_v23.yaml")
DRIVER = Path("scripts/run_rl_pilot_v23.sh")


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


class V23ContractTest(unittest.TestCase):
    def test_shipped_config_halves_the_resumed_rate(self) -> None:
        text = CONFIG.read_text(encoding="utf-8")
        self.assertAlmostEqual(float(_scalar(text, "learning_rate")), 5.0e-6)
        self.assertEqual(_scalar(text, "apply_learning_rate_on_resume"), "true")
        self.assertEqual(int(_scalar(text, "max_steps")), 24)
        self.assertAlmostEqual(float(_scalar(text, "max_raw_kl")), 5.0e-4)
        self.assertEqual(_scalar(text, "mode"), "raw_gap")
        self.assertAlmostEqual(float(_scalar(text, "local_max_relative")), 0.35)
        self.assertAlmostEqual(float(_scalar(text, "beta")), 0.04)
        self.assertEqual(_scalar(text, "train_audio_projections"), "false")
        self.assertEqual(int(_scalar(text, "expected_total_targets")), 196)
        self.assertNotIn("learning_rate: 1.0e-5", text)
        self.assertNotIn("mode: unit", text)
        self.assertNotIn("mode: capped_gap", text)
        self.assertNotIn("train_audio_projections: true", text)
        self.assertNotIn("apply_learning_rate_on_resume: false", text)

    def test_driver_resumes_v22_step8_only(self) -> None:
        text = DRIVER.read_text(encoding="utf-8")
        self.assertIn(
            "/data/mega-asr/runs/rl_pilot_v22/checkpoints/step_8",
            text,
        )
        self.assertIn('rm -rf "$V23_RUN"', text)
        self.assertNotIn('rm -rf "$V22_RUN"', text)

    def test_negative_step12_stops(self) -> None:
        action, reason = followup_action(
            "STOPPED_KL",
            12,
            24,
            -0.0004,
            -0.000143,
            "FAILED",
        )
        self.assertEqual(action, "stop")
        self.assertIn("fell below", reason)

    def test_non_negative_step12_below_dpo_continues(self) -> None:
        action, reason = followup_action(
            "CHUNK_DONE",
            12,
            24,
            0.0001,
            -0.000143,
            "FAILED",
        )
        self.assertEqual(action, "continue")
        self.assertIn("local-correction", reason)

    def test_step12_robust_back_above_dpo_stops(self) -> None:
        action, reason = followup_action(
            "BLOCKED_TRANSFER",
            12,
            24,
            0.0001,
            0.000014,
            "FAILED",
        )
        self.assertEqual(action, "stop")
        self.assertIn("robust still above", reason)

    def test_kl_stop_with_non_negative_reward_stops(self) -> None:
        action, reason = followup_action(
            "STOPPED_KL",
            12,
            24,
            0.0001,
            -0.000143,
            "FAILED",
        )
        self.assertEqual(action, "stop")
        self.assertIn("STOPPED_KL", reason)
