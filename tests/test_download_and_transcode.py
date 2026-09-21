#!/usr/bin/env python3
"""Unit tests for dataset downloading, transcoding, and smoke manifest generation."""

from __future__ import annotations

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

from scripts.download_sources import PINNED_SOURCES, RawDataDownloader
from scripts.build_robust_manifests import (
    RobustDatasetBuilder,
    compute_sha256,
    inspect_wav_audio,
    transcode_to_wav,
)


def create_sample_wav(
    file_path: Path,
    framerate: int = 44100,
    nchannels: int = 2,
    sampwidth: int = 2,
    duration_s: float = 1.0,
) -> None:
    """Create a sample WAV file with non-standard properties (e.g. stereo, 44.1kHz)."""
    nframes = int(framerate * duration_s)
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(b"\x00" * (nframes * nchannels * sampwidth))


class DownloadAndTranscodeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.raw_dir = self.root / "raw"
        self.audio_dir = self.root / "audio"
        self.manifest_dir = self.root / "manifests"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

        with open("configs/data/public_robust_v100.yaml", "r", encoding="utf-8") as f:
            self.base_config = yaml.safe_load(f)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_pinned_sources_revisions_match_contract(self) -> None:
        self.assertEqual(
            PINNED_SOURCES["voices_in_the_wild"]["revision"],
            "a8a35d3319737190d6fd3d39157b258eaab35980",
        )
        self.assertEqual(
            PINNED_SOURCES["librispeech_asr"]["revision"],
            "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
        )
        self.assertEqual(
            PINNED_SOURCES["aishell1"]["revision"],
            "c6dde006238091dc7c81cf5888208140bba33cba",
        )
        self.assertEqual(
            PINNED_SOURCES["bench_test"]["revision"],
            "788f5d72c6b0e9091b5c2e432370923b6f9f0660",
        )

    def test_downloader_mock_mode_produces_raw_complete(self) -> None:
        downloader = RawDataDownloader(
            Path("configs/data/public_robust_v100.yaml"),
            raw_dir=self.raw_dir,
        )
        # Override manifest root for test isolation
        downloader.manifest_root = self.manifest_dir

        report = downloader.run_download(mock=True)
        self.assertEqual(report["status"], "COMPLETE")
        self.assertEqual(report["mode"], "mock")

        raw_gate_file = self.manifest_dir / "RAW_COMPLETE.json"
        self.assertTrue(raw_gate_file.exists())
        with open(raw_gate_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data["status"], "COMPLETE")
        self.assertIn("voices_in_the_wild", data["sources"])

    def test_transcode_to_wav_converts_stereo_44k_to_mono_16k(self) -> None:
        raw_stereo = self.root / "raw_stereo_44k.wav"
        create_sample_wav(raw_stereo, framerate=44100, nchannels=2, sampwidth=2, duration_s=1.5)

        target_wav = self.audio_dir / "normalized_16k.wav"
        success, err = transcode_to_wav(raw_stereo, target_wav, sample_rate=16000)
        self.assertTrue(success, f"Transcode failed: {err}")
        self.assertTrue(target_wav.exists())

        # Inspect using WAV validator
        is_valid, reason, duration, sha = inspect_wav_audio(target_wav)
        self.assertTrue(is_valid, f"Transcoded audio invalid: {reason}")
        self.assertAlmostEqual(duration, 1.5, places=2)
        self.assertIsNotNone(sha)

    def test_build_smoke_subset_structure(self) -> None:
        builder = RobustDatasetBuilder(
            self.base_config, output_dir=self.manifest_dir, data_dir=self.audio_dir
        )

        # Create mock role pools with clean en, clean zh, and degraded
        role_pools = {role: [] for role in self.base_config["roles"]}
        for i in range(50):
            role_pools["sft_train"].append({
                "sample_id": f"sft_en_{i}",
                "condition_group": "clean",
                "language": "en",
                "text": f"clean english {i}",
            })
            role_pools["sft_train"].append({
                "sample_id": f"sft_zh_{i}",
                "condition_group": "clean",
                "language": "zh",
                "text": f"中文干净语音 {i}",
            })
        for i in range(100):
            role_pools["sft_train"].append({
                "sample_id": f"sft_deg_{i}",
                "condition_group": "degraded",
                "language": "en" if i % 2 == 0 else "zh",
                "text": f"robust noise {i}",
            })

        smoke = builder.build_smoke_subset(role_pools)
        self.assertEqual(len(smoke), 128)

        en_count = sum(1 for s in smoke if s["condition_group"] == "clean" and s["language"] == "en")
        zh_count = sum(1 for s in smoke if s["condition_group"] == "clean" and s["language"] == "zh")
        deg_count = sum(1 for s in smoke if s["condition_group"] == "degraded")

        self.assertEqual(en_count, 32)
        self.assertEqual(zh_count, 32)
        self.assertEqual(deg_count, 64)

    def test_smoke_manifest_persisted_and_checksummed(self) -> None:
        builder = RobustDatasetBuilder(
            self.base_config, output_dir=self.manifest_dir, data_dir=self.audio_dir
        )

        role_pools = {role: [] for role in self.base_config["roles"]}
        # Minimal samples
        sample = {
            "sample_id": "s0",
            "audio": "a0.wav",
            "text": "test",
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
            "audio_sha256": "dummy",
        }
        role_pools["sft_train"].append(sample)
        pilot_subsets = builder.build_pilot_subsets(role_pools)

        gate_res = builder.write_manifests_and_gates(
            role_pools, [], pilot_subsets, strict_quotas=False, smoke_subset=[sample]
        )

        smoke_file = self.manifest_dir / "smoke.jsonl"
        self.assertTrue(smoke_file.exists())

        with open(self.manifest_dir / "manifest_sha256.json", "r") as f:
            hashes = json.load(f)
        self.assertIn("smoke.jsonl", hashes)
        self.assertEqual(hashes["smoke.jsonl"], compute_sha256(smoke_file))


if __name__ == "__main__":
    unittest.main()
