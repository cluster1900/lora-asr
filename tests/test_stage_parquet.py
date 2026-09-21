#!/usr/bin/env python3
"""Unit tests for parquet dataset streaming, audio transcoding, and language detection."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.stage_parquet_sources import (
    PINNED_DATASETS,
    detect_language,
    parse_bench_subset,
    transcode_bytes_to_wav,
)


def create_raw_wav_bytes(
    framerate: int = 44100, nchannels: int = 2, sampwidth: int = 2, duration_s: float = 1.0
) -> bytes:
    """Create in-memory raw WAV bytes with non-standard properties."""
    buf = io.BytesIO()
    nframes = int(framerate * duration_s)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(nchannels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(b"\x00" * (nframes * nchannels * sampwidth))
    return buf.getvalue()


class StageParquetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_pinned_datasets_dict(self) -> None:
        self.assertIn("voices_in_the_wild", PINNED_DATASETS)
        self.assertIn("librispeech_asr", PINNED_DATASETS)
        self.assertIn("aishell1", PINNED_DATASETS)
        self.assertIn("bench_test", PINNED_DATASETS)

    def test_detect_language(self) -> None:
        self.assertEqual(detect_language("hello world this is an english test"), "en")
        self.assertEqual(detect_language("这是一个中文测试语音"), "zh")
        self.assertEqual(detect_language("Qwen3-ASR 中文识别测试"), "zh")

    def test_parse_bench_subset(self) -> None:
        origin, lang, scen = parse_bench_subset("real-zh-distortion")
        self.assertEqual(origin, "real")
        self.assertEqual(lang, "zh")
        self.assertEqual(scen, "distortion")

        origin, lang, scen = parse_bench_subset("synthetic-en-far_field")
        self.assertEqual(origin, "synthetic")
        self.assertEqual(lang, "en")
        self.assertEqual(scen, "far_field")

    def test_transcode_bytes_to_wav(self) -> None:
        raw_bytes = create_raw_wav_bytes(44100, 2, 2, 1.2)
        out_wav = self.root / "transcoded_sample.wav"

        success, err, dur, sha = transcode_bytes_to_wav(
            raw_bytes, out_wav, target_sr=16000, min_dur=0.5, max_dur=30.0
        )

        self.assertTrue(success, f"Transcode failed: {err}")
        self.assertTrue(out_wav.exists())
        self.assertAlmostEqual(dur, 1.2, places=1)
        self.assertIsNotNone(sha)

        with wave.open(str(out_wav), "rb") as wf:
            self.assertEqual(wf.getnchannels(), 1)
            self.assertEqual(wf.getframerate(), 16000)
            self.assertEqual(wf.getsampwidth(), 2)


if __name__ == "__main__":
    unittest.main()
