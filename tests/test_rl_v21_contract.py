"""v21 trains the v16 recipe without the three audio-tower projections."""

import unittest
from pathlib import Path

from train.rl_lora_targets import filter_lora_targets
from train.rl_v16_decision import followup_action

CONFIG = Path("configs/train/qwen3_asr_rl_v21.yaml")


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


class V21ContractTest(unittest.TestCase):
    def test_shipped_config_drops_audio_projections(self) -> None:
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

    def test_filter_drops_only_audio_tower_names(self) -> None:
        names = [
            "audio_tower.conv_out",
            "audio_tower.proj1",
            "audio_tower.proj2",
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.mlp.down_proj",
        ]
        self.assertEqual(
            filter_lora_targets(names, False),
            [
                "model.layers.0.self_attn.q_proj",
                "model.layers.0.mlp.down_proj",
            ],
        )
        self.assertEqual(filter_lora_targets(names, True), names)
        self.assertEqual(len(names), 5)

    def test_negative_step4_stops(self) -> None:
        action, reason = followup_action(
            "CHUNK_DONE",
            4,
            24,
            -0.0003,
            0.000202,
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


if __name__ == "__main__":
    unittest.main()
