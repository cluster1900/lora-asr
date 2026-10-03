"""v24 keeps v16's rate and advantage and masks unchanged policy tokens."""

import unittest
from pathlib import Path

from train.rl_policy_mask import changed_token_mask
from train.rl_v16_decision import followup_action

CONFIG = Path("configs/train/qwen3_asr_rl_v24.yaml")
DRIVER = Path("scripts/run_rl_pilot_v24.sh")


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


class V24ContractTest(unittest.TestCase):
    def test_shipped_config_masks_changed_tokens_only(self) -> None:
        text = CONFIG.read_text(encoding="utf-8")
        self.assertAlmostEqual(float(_scalar(text, "learning_rate")), 1.0e-5)
        self.assertEqual(int(_scalar(text, "max_steps")), 24)
        self.assertEqual(_scalar(text, "loss_reduction"), "sequence_sum")
        self.assertEqual(_scalar(text, "policy_token_mask"), "changes_only")
        self.assertEqual(_scalar(text, "mode"), "raw_gap")
        self.assertAlmostEqual(float(_scalar(text, "local_max_relative")), 0.35)
        self.assertAlmostEqual(float(_scalar(text, "beta")), 0.04)
        self.assertAlmostEqual(float(_scalar(text, "max_raw_kl")), 5.0e-4)
        self.assertEqual(_scalar(text, "train_audio_projections"), "true")
        self.assertEqual(int(_scalar(text, "expected_total_targets")), 199)
        for forbidden in (
            "5.0e-6",
            "2.0e-5",
            "mode: unit",
            "mode: capped_gap",
            "mode: fixed",
            "train_audio_projections: false",
            "policy_token_mask: all",
        ):
            self.assertNotIn(forbidden, text)

    def test_changed_token_mask_is_the_shipped_lever(self) -> None:
        self.assertEqual(
            changed_token_mask([10, 11, 12, 13], [10, 99, 12, 13]),
            [0.0, 1.0, 0.0, 0.0],
        )

    def test_driver_protects_older_runs(self) -> None:
        text = DRIVER.read_text(encoding="utf-8")
        for name in (
            "rl_pilot_v10",
            "rl_pilot_v11",
            "rl_pilot_v12",
            "rl_pilot_v16",
            "rl_pilot_v20",
            "rl_pilot_v21",
            "rl_pilot_v22",
            "rl_pilot_v23",
            "dpo_pilot_v2/merged_base",
        ):
            self.assertIn(name, text)
        self.assertIn('rm -rf "$V24_RUN"', text)
        self.assertNotIn('rm -rf "$V23_RUN"', text)
        self.assertNotIn('rm -rf "$V22_RUN"', text)
        self.assertNotIn("/data/mega-asr/runs/rl_pilot_v22/checkpoints/step_8", text)
        self.assertNotIn("/data/mega-asr/runs/rl_pilot_v23/checkpoints", text)
        self.assertIn("changed_token_mask", text)
        self.assertIn("policy_token_mask: changes_only", text)

    def test_negative_step4_stops(self) -> None:
        action, reason = followup_action(
            "CHUNK_DONE",
            4,
            24,
            -0.0002,
            -0.000090,
            "FAILED",
        )
        self.assertEqual(action, "stop")
        self.assertIn("fell below", reason)

    def test_step8_robust_above_dpo_stops(self) -> None:
        action, reason = followup_action(
            "CHUNK_DONE",
            8,
            24,
            0.0004,
            0.000164,
            "FAILED",
        )
        self.assertEqual(action, "stop")
        self.assertIn("robust still above", reason)

    def test_step8_below_dpo_continues(self) -> None:
        action, reason = followup_action(
            "CHUNK_DONE",
            8,
            24,
            0.0004,
            -0.000090,
            "FAILED",
        )
        self.assertEqual(action, "continue")
        self.assertIn("local-correction", reason)

    def test_kl_stop_does_not_continue(self) -> None:
        action, reason = followup_action(
            "STOPPED_KL",
            8,
            24,
            0.0004,
            -0.000090,
            "FAILED",
        )
        self.assertEqual(action, "stop")
        self.assertIn("STOPPED_KL", reason)


if __name__ == "__main__":
    unittest.main()
