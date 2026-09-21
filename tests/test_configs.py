from __future__ import annotations

import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class ConfigContractTest(unittest.TestCase):
    """Ensure YAML configurations strictly implement execution contract 08."""

    def test_data_config_loads_and_matches_contract(self) -> None:
        path = ROOT / "configs/data/public_robust_v100.yaml"
        self.assertTrue(path.is_file(), f"Missing config file: {path}")

        with path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)

        self.assertEqual(cfg["seed"], 20260722)
        self.assertEqual(cfg["audio_normalization"]["sample_rate"], 16000)
        self.assertEqual(cfg["audio_normalization"]["channels"], 1)
        self.assertEqual(cfg["audio_normalization"]["min_duration_s"], 0.5)
        self.assertEqual(cfg["audio_normalization"]["max_duration_s"], 30.0)

        # Sources & Pinned revisions
        sources = cfg["sources"]
        self.assertEqual(
            sources["voices_in_the_wild"]["revision"],
            "a8a35d3319737190d6fd3d39157b258eaab35980",
        )
        self.assertEqual(
            sources["librispeech_asr"]["revision"],
            "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
        )
        self.assertEqual(
            sources["aishell1"]["revision"],
            "c6dde006238091dc7c81cf5888208140bba33cba",
        )
        self.assertEqual(
            sources["bench_test"]["revision"],
            "788f5d72c6b0e9091b5c2e432370923b6f9f0660",
        )

        # Role quotas matching Table 3.2
        roles = cfg["roles"]
        self.assertEqual(roles["sft_train"]["quotas"]["robust_degraded"], 120000)
        self.assertEqual(roles["sft_train"]["quotas"]["english_clean"], 16000)
        self.assertEqual(roles["sft_train"]["quotas"]["chinese_clean"], 16000)

        self.assertEqual(roles["dpo_train_pool"]["target_pairs"]["robust_degraded"], 16000)
        self.assertEqual(roles["dpo_train_pool"]["target_pairs"]["english_clean"], 2000)
        self.assertEqual(roles["dpo_train_pool"]["target_pairs"]["chinese_clean"], 2000)
        self.assertEqual(roles["dpo_train_pool"]["source_candidates"]["english_clean"], 10000)

        self.assertEqual(roles["dpo_val_pool"]["target_pairs"]["robust_degraded"], 1600)
        self.assertEqual(roles["dpo_val_pool"]["target_pairs"]["english_clean"], 200)
        self.assertEqual(roles["dpo_val_pool"]["target_pairs"]["chinese_clean"], 200)

        self.assertEqual(roles["rl_train_pool"]["quotas"]["robust_degraded"], 16000)
        self.assertEqual(roles["rl_train_pool"]["quotas"]["english_clean"], 1000)
        self.assertEqual(roles["rl_train_pool"]["quotas"]["chinese_clean"], 1000)

        self.assertEqual(roles["rl_val_pool"]["quotas"]["robust_degraded"], 1600)
        self.assertEqual(roles["rl_val_pool"]["quotas"]["english_clean"], 200)
        self.assertEqual(roles["rl_val_pool"]["quotas"]["chinese_clean"], 200)

        self.assertEqual(roles["validation"]["quotas"]["robust_degraded"], 8000)
        self.assertEqual(roles["validation"]["quotas"]["english_clean"], 1000)
        self.assertEqual(roles["validation"]["quotas"]["chinese_clean"], 1000)

        self.assertEqual(roles["bench_test"]["quotas"]["robust_degraded"], 5000)

    def test_reward_config_loads_and_matches_contract(self) -> None:
        path = ROOT / "configs/train/reward_config.yaml"
        self.assertTrue(path.is_file(), f"Missing config file: {path}")

        with path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)

        comps = cfg["components"]
        self.assertIn("asr", comps)
        self.assertEqual(comps["empty"]["penalty"], -0.25)
        self.assertEqual(comps["repeat"]["penalty"], -0.25)
        self.assertEqual(comps["too_long"]["penalty"], -0.15)
        self.assertEqual(comps["too_long"]["length_ratio_threshold"], 1.5)
        self.assertEqual(comps["hallucination"]["penalty"], -0.25)
        self.assertEqual(comps["hallucination"]["error_rate_threshold"], 0.8)

        self.assertEqual(cfg["clip"]["min"], -1.0)
        self.assertEqual(cfg["clip"]["max"], 1.0)

        grpo = cfg["grpo"]
        self.assertEqual(grpo["group_size"], 4)
        self.assertEqual(grpo["sampling"]["temperature"], 0.7)
        self.assertEqual(grpo["sampling"]["top_p"], 0.9)
        self.assertEqual(grpo["advantage"]["zero_variance_action"], "zero_advantage")
        self.assertEqual(grpo["advantage"]["batch_zero_variance_threshold"], 0.30)

    def test_train_config_loads_and_targets_199_linears(self) -> None:
        path = ROOT / "configs/train/qwen3_asr_v100.yaml"
        self.assertTrue(path.is_file(), f"Missing config file: {path}")

        with path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)

        # Model & Runtime
        self.assertEqual(cfg["model"]["model_id"], "Qwen/Qwen3-ASR-1.7B")
        self.assertEqual(
            cfg["model"]["model_revision"],
            "7278e1e70fe206f11671096ffdd38061171dd6e5",
        )
        self.assertEqual(cfg["model"]["dtype"], "float16")
        self.assertEqual(cfg["model"]["attention_implementation"], "eager")
        self.assertFalse(cfg["model"]["flash_attention_2"])

        runtime = cfg["runtime"]
        self.assertEqual(runtime["world_size"], 4)
        self.assertEqual(runtime["micro_batch_size"], 1)
        self.assertEqual(runtime["gradient_accumulation_steps"], 16)
        self.assertEqual(runtime["global_batch_size"], 64)
        self.assertTrue(runtime["gradient_checkpointing"])
        self.assertTrue(runtime["enable_input_require_grads"])

        # LoRA Target Calculation: exactly 199 targets
        lora = cfg["lora"]
        self.assertEqual(lora["r"], 16)
        self.assertEqual(lora["lora_alpha"], 32)
        proj_count = len(lora["projection_targets"])
        self.assertEqual(proj_count, 3)

        attn_count = lora["decoder_layers_count"] * len(lora["decoder_attention_names"])
        mlp_count = lora["decoder_layers_count"] * len(lora["decoder_mlp_names"])
        total_calculated = proj_count + attn_count + mlp_count
        self.assertEqual(total_calculated, 199)
        self.assertEqual(lora["expected_total_targets"], 199)

        # Lifecycle & Gates
        self.assertEqual(cfg["lifecycle"]["strategy"], "merge_and_unload")
        self.assertEqual(
            cfg["gates"]["dpo_pilot"]["clean_error_rate_increase_vs_base_max"], 0.025
        )
        self.assertEqual(
            cfg["gates"]["rl_pilot"]["clean_error_rate_increase_vs_base_max"], 0.025
        )
        self.assertEqual(
            cfg["gates"]["release"]["global_clean_error_increase_vs_base_max"], 0.025
        )


if __name__ == "__main__":
    unittest.main()
