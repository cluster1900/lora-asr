"""v27 keeps v25's update and adds the reference only inside the local budget."""

import unittest
from pathlib import Path

from train.rl_local_winner import local_winner_index
from train.rl_reference_candidate import append_reference_candidate
from train.rl_v16_decision import followup_action

CONFIG = Path("configs/train/qwen3_asr_rl_v27.yaml")
DRIVER = Path("scripts/run_rl_pilot_v27.sh")


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


class V27ContractTest(unittest.TestCase):
    def test_shipped_config_uses_the_full_degraded_epoch(self) -> None:
        text = CONFIG.read_text(encoding="utf-8")
        self.assertAlmostEqual(float(_scalar(text, "learning_rate")), 1.0e-5)
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
            "5.0e-6",
            "2.0e-5",
            "mode: unit",
            "mode: capped_gap",
            "mode: fixed",
            "train_audio_projections: false",
            "policy_token_mask: all",
            "policy_token_mask: changes_only",
            "degraded_skip_regressed",
            "include_reference_candidate: false",
        ):
            self.assertNotIn(forbidden, text)

    def test_close_reference_is_selected_and_a_far_one_is_not(self) -> None:
        texts, rewards = append_reference_candidate(
            ["the cat sat", "the cat sit"],
            [0.40, 0.41],
            "the cat sits",
            1.0,
        )
        winner, status = local_winner_index(
            texts,
            rewards,
            language="en",
            min_improvement=0.02,
            max_relative=0.35,
        )
        self.assertEqual((status, texts[winner]), ("update", "the cat sits"))
        far_texts, far_rewards = append_reference_candidate(
            ["the cat sat", "the cat sit"],
            [0.40, 0.41],
            "one two three four five six seven eight nine ten",
            1.0,
        )
        far_winner, far_status = local_winner_index(
            far_texts,
            far_rewards,
            language="en",
            min_improvement=0.02,
            max_relative=0.35,
        )
        self.assertIsNone(far_winner)
        self.assertEqual(far_status, "no_improvement")

    def test_driver_protects_v26_and_uses_the_reference_candidate(self) -> None:
        text = DRIVER.read_text(encoding="utf-8")
        for name in (
            "rl_pilot_v10",
            "rl_pilot_v16",
            "rl_pilot_v24",
            "rl_pilot_v25",
            "rl_pilot_v26",
            "dpo_pilot_v2/merged_base",
        ):
            self.assertIn(name, text)
        self.assertIn('rm -rf "$V27_RUN"', text)
        self.assertNotIn('rm -rf "$V26_RUN"', text)
        self.assertNotIn('rm -rf "$V25_RUN"', text)
        self.assertNotIn("/data/mega-asr/runs/rl_pilot_v26/checkpoints", text)
        self.assertNotIn("/data/mega-asr/runs/rl_pilot_v25/checkpoints", text)
        self.assertIn("append_reference_candidate", text)
        self.assertIn("reference_text=reference_text", text)
        self.assertIn("compute_sequence_reward", text)
        self.assertIn("--sample-strategy degraded \\", text)
        self.assertNotIn("--sample-strategy degraded_skip_regressed", text)
        self.assertIn(
            'for name in ("v16", "v20", "v21", "v22", "v23", "v25", "v26")',
            text,
        )
        self.assertIn('v24.get("action") != "continue"', text)
        self.assertIn("len(selected) != 2000", text)

    def test_negative_step4_stops(self) -> None:
        action, reason = followup_action(
            "CHUNK_DONE",
            4,
            24,
            -0.0005,
            0.000359,
            "FAILED",
        )
        self.assertEqual(action, "stop")
        self.assertIn("fell below", reason)


if __name__ == "__main__":
    unittest.main()
