"""v28 resumes the v27 step-4 checkpoint and overwrites the learning rate."""

import unittest
from pathlib import Path

from train.rl_v16_decision import followup_action


CONFIG = Path("configs/train/qwen3_asr_rl_v28.yaml")
DRIVER = Path("scripts/run_rl_pilot_v28.sh")


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


class V28ContractTest(unittest.TestCase):
    def test_shipped_config_halves_the_resumed_rate(self) -> None:
        text = CONFIG.read_text(encoding="utf-8")
        self.assertAlmostEqual(float(_scalar(text, "learning_rate")), 5.0e-6)
        self.assertEqual(_scalar(text, "apply_learning_rate_on_resume"), "true")
        self.assertEqual(int(_scalar(text, "max_steps")), 24)
        self.assertEqual(_scalar(text, "loss_reduction"), "sequence_sum")
        self.assertEqual(_scalar(text, "sample_strategy"), "degraded")
        self.assertEqual(_scalar(text, "include_reference_candidate"), "true")
        self.assertEqual(_scalar(text, "policy_token_mask"), "signed_edits")
        self.assertEqual(_scalar(text, "mode"), "raw_gap")
        self.assertAlmostEqual(float(_scalar(text, "local_max_relative")), 0.35)
        self.assertAlmostEqual(float(_scalar(text, "beta")), 0.04)
        self.assertAlmostEqual(float(_scalar(text, "max_raw_kl")), 5.0e-4)
        self.assertEqual(_scalar(text, "train_audio_projections"), "true")
        self.assertEqual(int(_scalar(text, "expected_total_targets")), 199)
        for forbidden in (
            "learning_rate: 1.0e-5",
            "learning_rate: 2.0e-5",
            "mode: unit",
            "mode: capped_gap",
            "mode: fixed",
            "train_audio_projections: false",
            "policy_token_mask: all",
            "policy_token_mask: changes_only",
            "degraded_skip_regressed",
            "include_reference_candidate: false",
            "apply_learning_rate_on_resume: false",
        ):
            self.assertNotIn(forbidden, text)

    def test_driver_resumes_step4_and_protects_v27(self) -> None:
        text = DRIVER.read_text(encoding="utf-8")
        self.assertIn(
            'V27_STEP4="/data/mega-asr/runs/rl_pilot_v27/checkpoints/step_4"',
            text,
        )
        self.assertIn('resume="$V27_STEP4"', text)
        self.assertNotIn("checkpoints/step_7", text)
        self.assertIn('rm -rf "$V28_RUN"', text)
        self.assertNotIn('rm -rf "$V27_RUN"', text)
        self.assertNotIn("/data/mega-asr/runs/rl_pilot_v27/checkpoints/step_7", text)
        self.assertIn("apply_learning_rate_on_resume: true", text)
        self.assertIn("learning_rate: 5.0e-6", text)
        self.assertIn("append_reference_candidate", text)
        self.assertIn("apply_configured_learning_rate", text)
        self.assertIn("--sample-strategy degraded \\", text)
        self.assertNotIn("--sample-strategy degraded_skip_regressed", text)
        self.assertIn("len(selected) != 2000", text)
        self.assertIn("16.727412", text)
        self.assertIn(
            'for name in ("v16", "v20", "v21", "v22", "v23", "v25", "v26", "v27")',
            text,
        )
        self.assertIn('v24.get("action") != "continue"', text)
        for name in (
            "rl_pilot_v10",
            "rl_pilot_v27",
            "dpo_pilot_v2/merged_base",
        ):
            self.assertIn(name, text)

    def test_step8_transfer_block_can_continue(self) -> None:
        action, reason = followup_action(
            "BLOCKED_TRANSFER",
            8,
            24,
            0.0001,
            -7e-6,
            "FAILED",
        )
        self.assertEqual(action, "continue")
        self.assertIn("another chunk", reason)

    def test_negative_reward_or_kl_or_robust_stops(self) -> None:
        negative, negative_reason = followup_action(
            "CHUNK_DONE",
            8,
            24,
            -0.0001,
            -7e-6,
            "FAILED",
        )
        self.assertEqual(negative, "stop")
        self.assertIn("fell below", negative_reason)
        robust, robust_reason = followup_action(
            "BLOCKED_TRANSFER",
            8,
            24,
            0.0002,
            1.4e-5,
            "FAILED",
        )
        self.assertEqual(robust, "stop")
        self.assertIn("above DPO", robust_reason)
        kl, kl_reason = followup_action(
            "STOPPED_KL",
            8,
            24,
            0.0002,
            -7e-6,
            "FAILED",
        )
        self.assertEqual(kl, "stop")
        self.assertIn("STOPPED_KL", kl_reason)


if __name__ == "__main__":
    unittest.main()
