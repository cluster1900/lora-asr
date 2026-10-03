"""v25 keeps v16's rate and puts the negated gap on deleted greedy tokens."""

import unittest
from pathlib import Path

from train.rl_policy_mask import signed_edit_update
from train.rl_v16_decision import followup_action

CONFIG = Path("configs/train/qwen3_asr_rl_v25.yaml")
DRIVER = Path("scripts/run_rl_pilot_v25.sh")


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


class V25ContractTest(unittest.TestCase):
    def test_shipped_config_signs_deleted_tokens(self) -> None:
        text = CONFIG.read_text(encoding="utf-8")
        self.assertAlmostEqual(float(_scalar(text, "learning_rate")), 1.0e-5)
        self.assertEqual(int(_scalar(text, "max_steps")), 24)
        self.assertEqual(_scalar(text, "loss_reduction"), "sequence_sum")
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

    def test_deletion_only_and_identical_pairs_use_the_shipped_helper(self) -> None:
        deletion = signed_edit_update([1, 2, 3], [1, 3], 0.25)
        self.assertEqual(deletion.action, "update")
        self.assertEqual(deletion.winner_token_advantage, (0.0, 0.0))
        self.assertEqual(deletion.anchor_token_advantage, (0.0, -0.25, 0.0))
        same = signed_edit_update([10, 11, 12], [10, 11, 12], 0.25)
        self.assertEqual(same.action, "skip")
        self.assertEqual(same.winner_token_advantage, (0.0, 0.0, 0.0))
        self.assertEqual(same.anchor_token_advantage, (0.0, 0.0, 0.0))

    def test_driver_protects_older_runs_and_the_v24_decision(self) -> None:
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
            "rl_pilot_v24",
            "dpo_pilot_v2/merged_base",
        ):
            self.assertIn(name, text)
        self.assertIn('rm -rf "$V25_RUN"', text)
        self.assertNotIn('rm -rf "$V24_RUN"', text)
        self.assertNotIn('rm -rf "$V23_RUN"', text)
        self.assertNotIn("/data/mega-asr/runs/rl_pilot_v22/checkpoints/step_8", text)
        self.assertNotIn("/data/mega-asr/runs/rl_pilot_v23/checkpoints", text)
        self.assertNotIn("/data/mega-asr/runs/rl_pilot_v24/checkpoints", text)
        self.assertIn("signed_edit_update", text)
        self.assertIn("signed_edit_loss_args(", text)
        self.assertIn("policy_token_mask: signed_edits", text)
        self.assertIn('v24.get("action") != "continue"', text)

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


if __name__ == "__main__":
    unittest.main()
