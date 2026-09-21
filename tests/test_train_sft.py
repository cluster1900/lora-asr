#!/usr/bin/env python3
"""Unit tests for train/train_sft.py."""

from __future__ import annotations

import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.train_sft import (
    EXPECTED_LORA_TARGET_COUNT,
    LORA_TARGET_REGEX,
    ASRDataset,
    compute_target_map_hash,
    parse_args,
    verify_dataset_gate_for_training,
)


class TrainSFTTest(unittest.TestCase):
    """Test suite for SFT training runner components."""

    def test_lora_regex_matches_exact_199_canonical_targets(self) -> None:
        """Verify regex strictly accepts the 199 targets and rejects all others."""
        canonical_targets = [
            "audio_tower.conv_out",
            "audio_tower.proj1",
            "audio_tower.proj2",
        ]
        # 28 decoder layers with 4 attn + 3 mlp
        attn_names = ["q_proj", "k_proj", "v_proj", "o_proj"]
        mlp_names = ["gate_proj", "up_proj", "down_proj"]

        for layer_idx in range(28):
            for attn in attn_names:
                canonical_targets.append(f"model.layers.{layer_idx}.self_attn.{attn}")
            for mlp in mlp_names:
                canonical_targets.append(f"model.layers.{layer_idx}.mlp.{mlp}")

        self.assertEqual(len(canonical_targets), EXPECTED_LORA_TARGET_COUNT)

        # Verify every canonical target matches the regex
        for target in canonical_targets:
            self.assertTrue(
                re.search(LORA_TARGET_REGEX, target) is not None,
                f"Target {target} should match LORA_TARGET_REGEX",
            )

        # Verify forbidden modules are strictly rejected
        forbidden_targets = [
            "audio_tower.layers.0.self_attn.q_proj",
            "audio_tower.layers.0.mlp.fc1",
            "audio_tower.conv2d",
            "model.embed_tokens",
            "model.norm",
            "lm_head",
            "model.layers.28.self_attn.q_proj",  # only 0..27 exist
        ]
        for forbidden in forbidden_targets:
            self.assertIsNone(
                re.search(LORA_TARGET_REGEX, forbidden),
                f"Target {forbidden} must NOT match LORA_TARGET_REGEX",
            )

    def test_target_map_hash_is_deterministic(self) -> None:
        targets_a = ["model.layers.0.self_attn.q_proj", "audio_tower.proj1"]
        targets_b = ["audio_tower.proj1", "model.layers.0.self_attn.q_proj"]
        self.assertEqual(compute_target_map_hash(targets_a), compute_target_map_hash(targets_b))

    def test_parse_args_defaults(self) -> None:
        args = parse_args(["--manifest", "dummy.jsonl", "--output-dir", "dummy_run"])
        self.assertEqual(args.manifest, "dummy.jsonl")
        self.assertEqual(args.output_dir, "dummy_run")
        self.assertIsNone(args.max_steps)
        self.assertIsNone(args.save_steps)
        self.assertFalse(args.single_gpu)
        self.assertFalse(args.export_merged)
        self.assertIsNone(args.micro_batch_size)
        self.assertIsNone(args.gradient_accumulation_steps)

    def test_resume_from_checkpoint_provenance_validation(self) -> None:
        from train.train_sft import resume_from_checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir)
            meta = {
                "global_step": 10,
                "epoch": 0,
                "dataset_index": 640,
                "world_size": 4,
                "manifest_sha256": "correct_hash",
                "model_revision": "correct_rev",
                "seed": 20260722,
                "target_map_hash": "dummy_hash",
            }
            with open(ckpt_dir / "training_state.json", "w") as f:
                json.dump(meta, f)

            # 1. Manifest mismatch
            with self.assertRaises(ValueError) as ctx:
                resume_from_checkpoint(
                    ckpt_dir, None, None, None, rank=0,
                    current_manifest_sha256="wrong_hash",
                    current_world_size=4,
                    current_model_revision="correct_rev",
                )
            self.assertIn("Manifest SHA256 mismatch", str(ctx.exception))

            # 2. World size mismatch
            with self.assertRaises(ValueError) as ctx:
                resume_from_checkpoint(
                    ckpt_dir, None, None, None, rank=0,
                    current_manifest_sha256="correct_hash",
                    current_world_size=1,
                    current_model_revision="correct_rev",
                )
            self.assertIn("World size mismatch", str(ctx.exception))

            # 3. Model revision mismatch
            with self.assertRaises(ValueError) as ctx:
                resume_from_checkpoint(
                    ckpt_dir, None, None, None, rank=0,
                    current_manifest_sha256="correct_hash",
                    current_world_size=4,
                    current_model_revision="wrong_rev",
                )
            self.assertIn("Model revision mismatch", str(ctx.exception))

            # 4. Seed mismatch
            with self.assertRaises(ValueError) as ctx:
                resume_from_checkpoint(
                    ckpt_dir, None, None, None, rank=0,
                    current_manifest_sha256="correct_hash",
                    current_world_size=4,
                    current_model_revision="correct_rev",
                    current_seed=9999,
                )
            self.assertIn("Seed mismatch", str(ctx.exception))

            # 5. Gradient accumulation mismatch
            with self.assertRaises(ValueError) as ctx:
                meta_with_accum = dict(meta, gradient_accumulation_steps=16)
                with open(ckpt_dir / "training_state.json", "w") as f:
                    json.dump(meta_with_accum, f)
                resume_from_checkpoint(
                    ckpt_dir, None, None, None, rank=0,
                    current_manifest_sha256="correct_hash",
                    current_world_size=4,
                    current_model_revision="correct_rev",
                    current_gradient_accumulation_steps=8,
                )
            self.assertIn("Gradient accumulation steps mismatch", str(ctx.exception))

            # 6. Target map hash mismatch
            with self.assertRaises(ValueError) as ctx:
                resume_from_checkpoint(
                    ckpt_dir, None, None, None, rank=0,
                    current_manifest_sha256="correct_hash",
                    current_world_size=4,
                    current_model_revision="correct_rev",
                    current_target_map_hash="different_target_hash",
                )
            self.assertIn("Target map hash mismatch", str(ctx.exception))

            # 7. require_states=True when optimizer.pt is missing
            with self.assertRaises(FileNotFoundError) as ctx:
                resume_from_checkpoint(
                    ckpt_dir, None, None, None, rank=0,
                    current_manifest_sha256="correct_hash",
                    current_world_size=4,
                    current_model_revision="correct_rev",
                    require_states=True,
                )
            self.assertIn("optimizer state file missing", str(ctx.exception))

    def test_asr_dataset_loads_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_file = Path(tmpdir) / "test_manifest.jsonl"
            with open(manifest_file, "w", encoding="utf-8") as f:
                f.write(json.dumps({"sample_id": "s1", "audio": "a1.wav", "text": "hello"}) + "\n")
                f.write(json.dumps({"sample_id": "s2", "audio": "a2.wav", "answer": "world"}) + "\n")
                f.write("invalid json line\n")
                f.write(json.dumps({"sample_id": "s3"}) + "\n")  # missing audio/text

            dataset = ASRDataset(manifest_file)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]["sample_id"], "s1")
            self.assertEqual(dataset[1]["sample_id"], "s2")

    def test_parse_args_allow_subset(self) -> None:
        args_default = parse_args(["--manifest", "dummy.jsonl", "--output-dir", "dummy_run"])
        self.assertFalse(args_default.allow_subset)

        args_subset = parse_args(["--manifest", "dummy.jsonl", "--output-dir", "dummy_run", "--allow-subset"])
        self.assertTrue(args_subset.allow_subset)

    def test_verify_dataset_gate_enforcement(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            gate_file = tmp_path / "DATASET_COMPLETE.json"
            full_manifest = tmp_path / "sft_train.jsonl"
            pilot_manifest = tmp_path / "pilot_sft.jsonl"
            smoke_manifest = tmp_path / "smoke.jsonl"

            # 1. Gate not found -> returns NOT_FOUND
            ok, status = verify_dataset_gate_for_training(gate_file, full_manifest)
            self.assertTrue(ok)
            self.assertEqual(status, "NOT_FOUND")

            # 2. Gate PASSED -> always succeeds
            gate_file.write_text(json.dumps({"status": "PASSED"}), encoding="utf-8")
            ok, status = verify_dataset_gate_for_training(gate_file, full_manifest)
            self.assertTrue(ok)
            self.assertEqual(status, "PASSED")

            # 3. Gate FAILED -> raises unless ignore_dataset_gate=True
            gate_file.write_text(json.dumps({"status": "FAILED"}), encoding="utf-8")
            with self.assertRaises(RuntimeError) as ctx:
                verify_dataset_gate_for_training(gate_file, full_manifest)
            self.assertIn("Dataset gate check FAILED", str(ctx.exception))

            ok, status = verify_dataset_gate_for_training(gate_file, full_manifest, ignore_dataset_gate=True)
            self.assertTrue(ok)

            # 4. Gate NON_STRICT_SUBSET
            gate_file.write_text(json.dumps({"status": "NON_STRICT_SUBSET"}), encoding="utf-8")

            # 4a. Full training without --allow-subset MUST FAIL
            with self.assertRaises(RuntimeError) as ctx:
                verify_dataset_gate_for_training(gate_file, full_manifest, allow_subset=False)
            self.assertIn("Formal full training requires PASSED gate status", str(ctx.exception))

            # 4b. Full training with --allow-subset succeeds
            ok, status = verify_dataset_gate_for_training(gate_file, full_manifest, allow_subset=True)
            self.assertTrue(ok)
            self.assertEqual(status, "NON_STRICT_SUBSET")

            # 4c. Pilot or Smoke manifest succeeds even without --allow-subset
            ok, status = verify_dataset_gate_for_training(gate_file, pilot_manifest, allow_subset=False)
            self.assertTrue(ok)
            self.assertEqual(status, "NON_STRICT_SUBSET")

            ok, status = verify_dataset_gate_for_training(gate_file, smoke_manifest, allow_subset=False)
            self.assertTrue(ok)
            self.assertEqual(status, "NON_STRICT_SUBSET")

    def test_get_environment_info_contains_all_provenance_fields(self) -> None:
        from train.train_sft import get_environment_info
        info = get_environment_info(
            device_name="cuda:0",
            world_size=4,
            seed=20260722,
            model_id="Qwen/Qwen3-ASR-1.7B",
            model_revision="test_rev",
            dtype="float16",
            attention="eager",
        )
        required_fields = [
            "timestamp", "platform", "python_version", "torch_version", "transformers_version",
            "peft_version", "qwen_asr_version", "datasets_version", "git_commit",
            "cuda_available", "cuda_version", "gpu_count", "gpu_names",
            "device", "world_size", "seed", "model_id", "model_revision", "dtype", "attention",
        ]
        for field in required_fields:
            self.assertIn(field, info, f"Missing required environment provenance field: {field}")
        self.assertEqual(info["model_id"], "Qwen/Qwen3-ASR-1.7B")
        self.assertEqual(info["model_revision"], "test_rev")
        self.assertEqual(info["dtype"], "float16")
        self.assertEqual(info["attention"], "eager")

    def test_save_checkpoint_records_scheduler_null_when_scheduler_none(self) -> None:
        from unittest.mock import MagicMock
        from train.train_sft import save_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "step_10"
            mock_thinker = MagicMock()
            mock_opt = MagicMock()
            mock_opt.state_dict.return_value = {"state": {}}

            save_checkpoint(
                ckpt_dir=ckpt_dir,
                peft_thinker=mock_thinker,
                processor=None,
                optimizer=mock_opt,
                scheduler=None,
                global_step=10,
                epoch=0,
                dataset_index=640,
                world_size=4,
                rank=0,
                manifest_sha256="fake_sha",
                model_revision="fake_rev",
                seed=20260722,
                target_map_hash="fake_target_hash",
            )

            state_path = ckpt_dir / "training_state.json"
            self.assertTrue(state_path.is_file())
            state_data = json.loads(state_path.read_text(encoding="utf-8"))
            self.assertIn("scheduler", state_data)
            self.assertIsNone(state_data["scheduler"])
            self.assertFalse((ckpt_dir / "scheduler.pt").exists())

    def test_build_epoch_sample_indices_balanced(self) -> None:
        from unittest.mock import MagicMock
        from train.train_sft import build_epoch_sample_indices

        mock_dataset = MagicMock()
        # 50 degraded, 10 en clean, 10 zh clean, 10 other
        samples = []
        for i in range(50):
            samples.append({"condition_group": "degraded", "language": "en", "sample_id": f"deg_{i}"})
        for i in range(10):
            samples.append({"condition_group": "clean", "language": "en", "sample_id": f"en_{i}"})
        for i in range(10):
            samples.append({"condition_group": "clean", "language": "zh", "sample_id": f"zh_{i}"})
        for i in range(10):
            samples.append({"condition_group": "other", "language": "en", "sample_id": f"oth_{i}"})

        mock_dataset.samples = samples
        mock_dataset.__len__.return_value = len(samples)

        # 1. Test balanced strategy: 50 deg + 25 en + 25 zh = 100 samples
        balanced_indices = build_epoch_sample_indices(mock_dataset, strategy="balanced", epoch=0, seed=20260722)
        self.assertEqual(len(balanced_indices), 100)
        sampled_groups = [samples[idx]["condition_group"] for idx in balanced_indices]
        sampled_langs = [samples[idx]["language"] for idx in balanced_indices]

        deg_count = sum(1 for g in sampled_groups if g == "degraded")
        en_clean_count = sum(1 for g, l in zip(sampled_groups, sampled_langs) if g == "clean" and l == "en")
        zh_clean_count = sum(1 for g, l in zip(sampled_groups, sampled_langs) if g == "clean" and l == "zh")

        self.assertEqual(deg_count, 50)
        self.assertEqual(en_clean_count, 25)
        self.assertEqual(zh_clean_count, 25)

        # Determinism check across epochs
        idx_ep0 = build_epoch_sample_indices(mock_dataset, strategy="balanced", epoch=0, seed=20260722)
        idx_ep0_repeat = build_epoch_sample_indices(mock_dataset, strategy="balanced", epoch=0, seed=20260722)
        idx_ep1 = build_epoch_sample_indices(mock_dataset, strategy="balanced", epoch=1, seed=20260722)
        self.assertEqual(idx_ep0, idx_ep0_repeat)
        self.assertNotEqual(idx_ep0, idx_ep1)

    def test_parse_args_controlled_options(self) -> None:
        args = parse_args([
            "--manifest", "dummy.jsonl",
            "--output-dir", "dummy_run",
            "--lr-scheduler", "linear",
            "--sample-strategy", "balanced",
            "--val-manifest", "val.jsonl",
            "--eval-steps", "50",
            "--learning-rate", "1e-5",
            "--warmup-steps", "50",
        ])
        self.assertEqual(args.lr_scheduler, "linear")
        self.assertEqual(args.sample_strategy, "balanced")
        self.assertEqual(args.val_manifest, "val.jsonl")
        self.assertEqual(args.eval_steps, 50)
        self.assertEqual(args.learning_rate, 1e-5)
        self.assertEqual(args.warmup_steps, 50)


    def test_evaluate_validation_loss_handles_empty_dataset(self) -> None:
        from unittest.mock import MagicMock
        from train.train_sft import evaluate_validation_loss
        mock_model = MagicMock()
        mock_ds = MagicMock()
        mock_ds.__len__.return_value = 0
        loss = evaluate_validation_loss(mock_model, mock_ds, None, "cpu", max_eval_samples=16)
        self.assertEqual(loss, 0.0)


if __name__ == "__main__":
    unittest.main()

