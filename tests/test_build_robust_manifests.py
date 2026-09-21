#!/usr/bin/env python3
"""Unit tests for E1 dataset builder and gate generation."""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import wave
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_robust_manifests import (
    RobustDatasetBuilder,
    compute_sha256,
    deterministic_bucket,
    inspect_wav_audio,
)


def create_synthetic_wav(
    file_path: Path,
    framerate: int = 16000,
    nchannels: int = 1,
    sampwidth: int = 2,
    duration_s: float = 1.0,
) -> None:
    """Create a minimal valid synthetic WAV file."""
    nframes = int(framerate * duration_s)
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        # write silence (0 bytes)
        wf.writeframes(b"\x00" * (nframes * nchannels * sampwidth))


class BuildRobustManifestsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.audio_dir = self.root / "audio"
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir = self.root / "manifests"
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

        with open("configs/data/public_robust_v100.yaml", "r", encoding="utf-8") as f:
            self.base_config = yaml.safe_load(f)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_inspect_wav_audio_valid_and_invalid(self) -> None:
        valid_wav = self.audio_dir / "valid.wav"
        create_synthetic_wav(valid_wav, 16000, 1, 2, 2.0)
        is_val, reason, duration, sha = inspect_wav_audio(valid_wav)
        self.assertTrue(is_val)
        self.assertEqual(reason, "")
        self.assertAlmostEqual(duration, 2.0, places=2)
        self.assertIsNotNone(sha)

        # Invalid channels (stereo)
        stereo_wav = self.audio_dir / "stereo.wav"
        create_synthetic_wav(stereo_wav, 16000, 2, 2, 2.0)
        is_val, reason, _, _ = inspect_wav_audio(stereo_wav)
        self.assertFalse(is_val)
        self.assertIn("channel", reason.lower())

        # Invalid sample rate (8000)
        rate_wav = self.audio_dir / "rate8k.wav"
        create_synthetic_wav(rate_wav, 8000, 1, 2, 2.0)
        is_val, reason, _, _ = inspect_wav_audio(rate_wav)
        self.assertFalse(is_val)
        self.assertIn("16000", reason)

        # Too short (<0.5s)
        short_wav = self.audio_dir / "short.wav"
        create_synthetic_wav(short_wav, 16000, 1, 2, 0.2)
        is_val, reason, _, _ = inspect_wav_audio(short_wav)
        self.assertFalse(is_val)
        self.assertIn("out of range", reason)

    def test_deterministic_bucket(self) -> None:
        val1 = deterministic_bucket("utt_001", 20260722)
        val2 = deterministic_bucket("utt_001", 20260722)
        val3 = deterministic_bucket("utt_002", 20260722)
        self.assertEqual(val1, val2)
        self.assertNotEqual(val1, val3)
        self.assertTrue(0.0 <= val1 <= 1.0)

    def test_partitioning_and_isolation(self) -> None:
        builder = RobustDatasetBuilder(
            self.base_config, output_dir=self.manifest_dir, data_dir=self.audio_dir
        )

        samples_by_source = {
            "librispeech_asr": [
                {
                    "sample_id": f"libri_tr_{i}",
                    "audio": f"dummy_tr_{i}.wav",
                    "text": f"hello world {i}",
                    "language": "en",
                    "scenario": "clean",
                    "condition_group": "clean",
                    "audio_origin": "real",
                    "source_dataset": "openslr/librispeech_asr",
                    "source_revision": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
                    "source_split": "train.100",
                    "source_index": i,
                    "source_utterance_id": f"libri_tr_utt_{i}",
                    "duration_s": 2.5,
                    "license": "CC-BY-4.0",
                }
                for i in range(100)
            ]
            + [
                {
                    "sample_id": f"libri_val_{i}",
                    "audio": f"dummy_val_{i}.wav",
                    "text": f"validation sample {i}",
                    "language": "en",
                    "scenario": "clean",
                    "condition_group": "clean",
                    "audio_origin": "real",
                    "source_dataset": "openslr/librispeech_asr",
                    "source_revision": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
                    "source_split": "validation",
                    "source_index": i,
                    "source_utterance_id": f"libri_val_utt_{i}",
                    "duration_s": 3.0,
                    "license": "CC-BY-4.0",
                }
                for i in range(50)
            ],
            "aishell1": [
                {
                    "sample_id": f"aishell_tr_{i}",
                    "audio": f"dummy_zh_tr_{i}.wav",
                    "text": f"中文训练测试 {i}",
                    "language": "zh",
                    "scenario": "clean",
                    "condition_group": "clean",
                    "audio_origin": "real",
                    "source_dataset": "knoveleng/aishell1-mandarin",
                    "source_revision": "c6dde006238091dc7c81cf5888208140bba33cba",
                    "source_split": "train",
                    "source_index": i,
                    "source_utterance_id": f"aishell_tr_utt_{i}",
                    "duration_s": 4.0,
                    "license": "Apache-2.0",
                }
                for i in range(100)
            ]
            + [
                {
                    "sample_id": f"aishell_val_{i}",
                    "audio": f"dummy_zh_val_{i}.wav",
                    "text": f"中文验证测试 {i}",
                    "language": "zh",
                    "scenario": "clean",
                    "condition_group": "clean",
                    "audio_origin": "real",
                    "source_dataset": "knoveleng/aishell1-mandarin",
                    "source_revision": "c6dde006238091dc7c81cf5888208140bba33cba",
                    "source_split": "validation",
                    "source_index": i,
                    "source_utterance_id": f"aishell_val_utt_{i}",
                    "duration_s": 3.5,
                    "license": "Apache-2.0",
                }
                for i in range(50)
            ],
            "voices_in_the_wild": [
                {
                    "sample_id": f"vitw_{i}",
                    "audio": f"dummy_vitw_{i}.wav",
                    "text": f"robust noise sample {i}",
                    "language": "en" if i % 2 == 0 else "zh",
                    "scenario": "noise",
                    "condition_group": "degraded",
                    "audio_origin": "real",
                    "source_dataset": "zhifeixie/Voices-in-the-Wild-2M",
                    "source_revision": "a8a35d3319737190d6fd3d39157b258eaab35980",
                    "source_split": "train",
                    "source_index": i,
                    "source_utterance_id": f"vitw_utt_{i}",
                    "duration_s": 5.0,
                    "license": "Open",
                }
                for i in range(200)
            ],
            "bench_test": [
                {
                    "sample_id": f"bench_{i}",
                    "audio": f"dummy_bench_{i}.wav",
                    "text": f"bench sample {i}",
                    "language": "en",
                    "scenario": "reverb",
                    "condition_group": "degraded",
                    "audio_origin": "real",
                    "source_dataset": "zhifeixie/Voices-in-the-Wild-Bench",
                    "source_revision": "788f5d72c6b0e9091b5c2e432370923b6f9f0660",
                    "source_split": "test",
                    "source_index": i,
                    "source_utterance_id": f"bench_utt_{i}",
                    "duration_s": 2.0,
                    "license": "Bench",
                }
                for i in range(20)
            ],
        }

        role_pools, rejects = builder.partition_samples(samples_by_source, verify_audio_file=False)

        self.assertEqual(len(rejects), 0)
        self.assertEqual(len(role_pools["bench_test"]), 20)

        # Check LibriSpeech train.100 strictly went to train roles
        train_roles = ["sft_train", "dpo_train_pool", "rl_train_pool"]
        val_roles = ["validation", "dpo_val_pool", "rl_val_pool"]

        for r in train_roles:
            for s in role_pools[r]:
                if s["source_dataset"] == "openslr/librispeech_asr":
                    self.assertEqual(s["source_split"], "train.100")
                elif s["source_dataset"] == "knoveleng/aishell1-mandarin":
                    self.assertEqual(s["source_split"], "train")

        for r in val_roles:
            for s in role_pools[r]:
                if s["source_dataset"] == "openslr/librispeech_asr":
                    self.assertEqual(s["source_split"], "validation")
                elif s["source_dataset"] == "knoveleng/aishell1-mandarin":
                    self.assertEqual(s["source_split"], "validation")

        # Verify cross-role leakage
        no_leak, errors = builder.verify_leakage(role_pools)
        self.assertTrue(no_leak, f"Leakage errors: {errors}")

    def test_leakage_detection_catches_overlaps(self) -> None:
        builder = RobustDatasetBuilder(
            self.base_config, output_dir=self.manifest_dir, data_dir=self.audio_dir
        )
        leaky_pools = {
            "sft_train": [
                {
                    "sample_id": "s1",
                    "source_utterance_id": "utt_shared_001",
                }
            ],
            "validation": [
                {
                    "sample_id": "s2",
                    "source_utterance_id": "utt_shared_001",  # Leak!
                }
            ],
        }
        no_leak, errors = builder.verify_leakage(leaky_pools)
        self.assertFalse(no_leak)
        self.assertTrue(any("Leakage detected" in e for e in errors))

    def test_write_manifests_and_gates(self) -> None:
        builder = RobustDatasetBuilder(
            self.base_config, output_dir=self.manifest_dir, data_dir=self.audio_dir
        )

        role_pools = {role: [] for role in self.base_config["roles"]}
        role_pools["sft_train"] = [
            {
                "sample_id": "sft_0",
                "audio": "a0.wav",
                "text": "hello",
                "language": "en",
                "scenario": "clean",
                "condition_group": "clean",
                "audio_origin": "real",
                "source_dataset": "openslr/librispeech_asr",
                "source_revision": "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
                "source_split": "train.100",
                "source_index": 0,
                "source_utterance_id": "u0",
                "duration_s": 2.0,
                "license": "CC-BY-4.0",
                "seed": 20260722,
                "audio_sha256": "fakehash",
            }
        ]
        rejects = [{"sample_id": "rej_1", "reason": "empty text"}]
        pilot_subsets = builder.build_pilot_subsets(role_pools)

        gate_res = builder.write_manifests_and_gates(
            role_pools, rejects, pilot_subsets, strict_quotas=False
        )

        # Empty roles must NOT pass the gate
        self.assertEqual(gate_res["status"], "FAILED")
        self.assertEqual(gate_res["leakage_check"], "PASSED")

        # Verify files exist on disk
        self.assertTrue((self.manifest_dir / "sft_train.jsonl").exists())
        self.assertTrue((self.manifest_dir / "sft_train_COMPLETE.json").exists())
        self.assertTrue((self.manifest_dir / "rejects.jsonl").exists())
        self.assertTrue((self.manifest_dir / "manifest_sha256.json").exists())
        self.assertTrue((self.manifest_dir / "RAW_COMPLETE.json").exists())
        self.assertTrue((self.manifest_dir / "PROCESSED_COMPLETE.json").exists())
        self.assertTrue((self.manifest_dir / "DATASET_COMPLETE.json").exists())

        # Verify manifest_sha256 matches actual hash
        with open(self.manifest_dir / "manifest_sha256.json", "r") as f:
            manifest_hashes = json.load(f)
        sft_manifest_hash = compute_sha256(self.manifest_dir / "sft_train.jsonl")
        self.assertEqual(manifest_hashes["sft_train.jsonl"], sft_manifest_hash)

    def test_pairwise_21_roles_isolation(self) -> None:
        """Verify that leakage between any two of the 7 roles is strictly detected."""
        builder = RobustDatasetBuilder(
            self.base_config, output_dir=self.manifest_dir, data_dir=self.audio_dir
        )
        roles = list(self.base_config["roles"].keys())
        self.assertEqual(len(roles), 7)

        # Test sample_id leakage between dpo_train_pool and bench_test
        pools = {r: [] for r in roles}
        pools["dpo_train_pool"].append({"sample_id": "overlap_id", "source_utterance_id": "u1", "audio_sha256": "h1"})
        pools["bench_test"].append({"sample_id": "overlap_id", "source_utterance_id": "u2", "audio_sha256": "h2"})
        ok, errs = builder.verify_leakage(pools)
        self.assertFalse(ok)
        self.assertTrue(any("Sample ID overlap between dpo_train_pool and bench_test" in e for e in errs))

        # Test audio_sha256 leakage between rl_train_pool and validation
        pools2 = {r: [] for r in roles}
        pools2["rl_train_pool"].append({"sample_id": "s1", "source_utterance_id": "u1", "audio_sha256": "shared_hash"})
        pools2["validation"].append({"sample_id": "s2", "source_utterance_id": "u2", "audio_sha256": "shared_hash"})
        ok2, errs2 = builder.verify_leakage(pools2)
        self.assertFalse(ok2)
        self.assertTrue(any("Audio sha256 overlap" in e and "validation" in e and "rl_train_pool" in e for e in errs2))

    def test_smoke_subset_raises_if_sft_train_insufficient(self) -> None:
        """Verify build_smoke_subset fails hard instead of backfilling from other roles."""
        builder = RobustDatasetBuilder(
            self.base_config, output_dir=self.manifest_dir, data_dir=self.audio_dir
        )
        pools = {r: [] for r in self.base_config["roles"]}
        pools["sft_train"] = [{"sample_id": "s1", "condition_group": "clean", "language": "en"}]
        # Put 100 samples in bench_test to simulate contamination temptation
        pools["bench_test"] = [{"sample_id": f"b{i}", "condition_group": "clean", "language": "en"} for i in range(100)]

        with self.assertRaises(ValueError) as ctx:
            builder.build_smoke_subset(pools)
        self.assertIn("insufficient samples", str(ctx.exception))
        self.assertIn("Backfilling from other roles", str(ctx.exception))

    def test_pilot_subsets_dpo_and_rl_have_balanced_categories(self) -> None:
        """Verify pilot subsets for dpo and rl contain balanced degraded, en clean, and zh clean."""
        builder = RobustDatasetBuilder(
            self.base_config, output_dir=self.manifest_dir, data_dir=self.audio_dir
        )
        pools = {r: [] for r in self.base_config["roles"]}

        # Fill dpo_train_pool
        pools["dpo_train_pool"] = [
            {"sample_id": f"dpo_deg_{i}", "condition_group": "degraded", "language": "en"} for i in range(10)
        ] + [
            {"sample_id": f"dpo_en_{i}", "condition_group": "clean", "language": "en"} for i in range(10)
        ] + [
            {"sample_id": f"dpo_zh_{i}", "condition_group": "clean", "language": "zh"} for i in range(10)
        ]

        # Fill rl_train_pool
        pools["rl_train_pool"] = [
            {"sample_id": f"rl_deg_{i}", "condition_group": "degraded", "language": "en"} for i in range(10)
        ] + [
            {"sample_id": f"rl_en_{i}", "condition_group": "clean", "language": "en"} for i in range(10)
        ] + [
            {"sample_id": f"rl_zh_{i}", "condition_group": "clean", "language": "zh"} for i in range(10)
        ]

        pilots = builder.build_pilot_subsets(pools, strict=False)
        self.assertIn("dpo", pilots)
        self.assertIn("rl", pilots)

        dpo_deg = [r for r in pilots["dpo"] if r.get("condition_group") == "degraded"]
        dpo_en = [r for r in pilots["dpo"] if r.get("condition_group") == "clean" and r.get("language") == "en"]
        dpo_zh = [r for r in pilots["dpo"] if r.get("condition_group") == "clean" and r.get("language") == "zh"]
        self.assertTrue(len(dpo_deg) > 0)
        self.assertTrue(len(dpo_en) > 0)
        self.assertTrue(len(dpo_zh) > 0)

        rl_deg = [r for r in pilots["rl"] if r.get("condition_group") == "degraded"]
        rl_en = [r for r in pilots["rl"] if r.get("condition_group") == "clean" and r.get("language") == "en"]
        rl_zh = [r for r in pilots["rl"] if r.get("condition_group") == "clean" and r.get("language") == "zh"]
        self.assertTrue(len(rl_deg) > 0)
        self.assertTrue(len(rl_en) > 0)
        self.assertTrue(len(rl_zh) > 0)

    def test_verify_dataset_integrity_strict_quotas_enforcement(self) -> None:
        """Verify verify_dataset_integrity blocks NON_STRICT_SUBSET when strict_quotas=True."""
        builder = RobustDatasetBuilder(
            self.base_config, output_dir=self.manifest_dir, data_dir=self.audio_dir
        )
        gate_file = self.manifest_dir / "DATASET_COMPLETE.json"
        gate_file.write_text(json.dumps({"status": "NON_STRICT_SUBSET"}), encoding="utf-8")

        # Non-strict mode: doesn't error out on gate status alone
        # (might fail on missing manifests, but shouldn't fail on gate status)
        _, errs_non_strict = builder.verify_dataset_integrity(check_audio=False, strict_quotas=False)
        self.assertFalse(any("expected PASSED for strict quotas" in e for e in errs_non_strict))

        # Strict mode: must record strict quota failure
        _, errs_strict = builder.verify_dataset_integrity(check_audio=False, strict_quotas=True)
        self.assertTrue(any("expected PASSED for strict quotas" in e for e in errs_strict))


if __name__ == "__main__":
    unittest.main()
