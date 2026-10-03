"""v26 keeps v25's update and drops noise and recording from the epoch."""

import unittest
from pathlib import Path

from train.rl_sample_strategy import degraded_skip_regressed_indices
from train.rl_v16_decision import followup_action

CONFIG = Path("configs/train/qwen3_asr_rl_v26.yaml")
DRIVER = Path("scripts/run_rl_pilot_v26.sh")


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


class V26ContractTest(unittest.TestCase):
    def test_shipped_config_skips_regressed_scenarios(self) -> None:
        text = CONFIG.read_text(encoding="utf-8")
        self.assertAlmostEqual(float(_scalar(text, "learning_rate")), 1.0e-5)
        self.assertEqual(int(_scalar(text, "max_steps")), 24)
        self.assertEqual(_scalar(text, "loss_reduction"), "sequence_sum")
        self.assertEqual(_scalar(text, "sample_strategy"), "degraded_skip_regressed")
        self.assertEqual(_scalar(text, "policy_token_mask"), "signed_edits")
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
            "policy_token_mask: changes_only",
        ):
            self.assertNotIn(forbidden, text)

    def test_selector_drops_the_two_regressed_scenarios(self) -> None:
        rows = [
            {"scenario": "noise", "condition_group": "degraded"},
            {"scenario": "recording", "condition_group": "degraded"},
            {"scenario": "distortion", "condition_group": "degraded"},
            {"scenario": "clean", "condition_group": "clean"},
        ]
        self.assertEqual(degraded_skip_regressed_indices(rows), [2])

    def test_driver_protects_v25_and_uses_the_new_strategy(self) -> None:
        text = DRIVER.read_text(encoding="utf-8")
        for name in (
            "rl_pilot_v10",
            "rl_pilot_v16",
            "rl_pilot_v24",
            "rl_pilot_v25",
            "dpo_pilot_v2/merged_base",
        ):
            self.assertIn(name, text)
        self.assertIn('rm -rf "$V26_RUN"', text)
        self.assertNotIn('rm -rf "$V25_RUN"', text)
        self.assertNotIn('rm -rf "$V24_RUN"', text)
        self.assertNotIn("/data/mega-asr/runs/rl_pilot_v25/checkpoints", text)
        self.assertIn("degraded_skip_regressed_indices", text)
        self.assertIn(
            'choices=["balanced", "standard", "degraded", "degraded_skip_regressed"]',
            text,
        )
        self.assertIn("--sample-strategy degraded_skip_regressed", text)
        self.assertIn('for name in ("v16", "v20", "v21", "v22", "v23", "v25")', text)
        self.assertIn('v24.get("action") != "continue"', text)
        self.assertIn("len(selected) != 1483", text)

    def test_negative_step4_stops(self) -> None:
        action, reason = followup_action(
            "CHUNK_DONE",
            4,
            24,
            -0.0003,
            0.000209,
            "FAILED",
        )
        self.assertEqual(action, "stop")
        self.assertIn("fell below", reason)


if __name__ == "__main__":
    unittest.main()
