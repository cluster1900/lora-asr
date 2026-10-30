#!/usr/bin/env python3
"""Build the three-row Phase 0 smoke manifest.

The clean clip is the public Qwen3-ASR Chinese example. The degraded clip is
the same waveform plus seeded Gaussian noise. The third row points at a file
that is never created, so inference can record a failure without stopping.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

OFFICIAL_ZH_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_zh.wav"
OFFICIAL_ZH_TEXT = "甚至出现交易几乎停滞的情况。"
CLEAN_ID = "phase0_zh_clean"
NOISE_ID = "phase0_zh_noise"
MISSING_ID = "phase0_zh_missing"
SOURCE_ID = "qwen3_asr_repo_asr_zh"
SAMPLE_RATE = 16000
NOISE_SNR_DB = 5.0
NOISE_SEED = 42


def to_mono_16k(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Mix channels and linearly resample to 16 kHz."""
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio = np.clip(audio, -1.0, 1.0)
    if sample_rate == SAMPLE_RATE:
        return audio
    if sample_rate <= 0 or audio.size == 0:
        raise ValueError(f"cannot resample audio with rate {sample_rate} and {audio.size} samples")
    new_length = int(round(audio.size * SAMPLE_RATE / sample_rate))
    old_axis = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    new_axis = np.linspace(0.0, 1.0, num=new_length, endpoint=False)
    return np.interp(new_axis, old_axis, audio).astype(np.float32)


def add_gaussian_noise(samples: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """Add white noise at a fixed SNR. The same seed reproduces the waveform."""
    audio = np.asarray(samples, dtype=np.float32)
    signal_power = float(np.mean(np.square(audio)))
    if signal_power <= 0.0:
        raise ValueError("cannot add noise to a silent waveform")
    generator = np.random.default_rng(seed)
    noise = generator.normal(0.0, 1.0, size=audio.shape).astype(np.float32)
    noise_power = float(np.mean(np.square(noise)))
    scale = np.sqrt(signal_power / (noise_power * (10.0 ** (snr_db / 10.0))))
    return np.clip(audio + noise * np.float32(scale), -1.0, 1.0)


def _row(sample_id: str, audio: str, scenario: str, augmentation: dict[str, object] | None) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": "qwen3-asr-manifest-v1",
        "sample_id": sample_id,
        "source_utterance_id": SOURCE_ID,
        "audio": audio,
        "text": OFFICIAL_ZH_TEXT,
        "language": "zh",
        "scenario": scenario,
        "split": "smoke",
    }
    if augmentation is not None:
        row["augmentation"] = augmentation
    return row


def write_smoke_manifest(manifest_dir: Path, clean: np.ndarray, noisy: np.ndarray) -> Path:
    """Write wavs and smoke.jsonl. The missing row has no wav on purpose."""
    wav_dir = manifest_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    sf.write(wav_dir / "asr_zh.wav", clean, SAMPLE_RATE)
    sf.write(wav_dir / "asr_zh_noise.wav", noisy, SAMPLE_RATE)
    rows = [
        _row(CLEAN_ID, "wavs/asr_zh.wav", "clean", None),
        _row(
            NOISE_ID,
            "wavs/asr_zh_noise.wav",
            "noise",
            {"type": "gaussian_noise", "snr_db": NOISE_SNR_DB, "seed": NOISE_SEED, "version": "phase0-noise-v1"},
        ),
        _row(MISSING_ID, "wavs/does_not_exist.wav", "dropout", {"type": "missing_file", "version": "phase0-noise-v1"}),
    ]
    manifest_path = manifest_dir / "smoke.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return manifest_path


def download_official_zh(destination: Path) -> None:
    """Download the public Chinese example clip."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(OFFICIAL_ZH_URL, destination)


def prepare(manifest_dir: Path, download_dir: Path) -> Path:
    """Download, normalize, degrade, and write the smoke manifest."""
    raw_path = download_dir / "asr_zh_raw.wav"
    download_official_zh(raw_path)
    samples, sample_rate = sf.read(raw_path, always_2d=False)
    clean = to_mono_16k(np.asarray(samples), int(sample_rate))
    noisy = add_gaussian_noise(clean, NOISE_SNR_DB, NOISE_SEED)
    return write_smoke_manifest(manifest_dir, clean, noisy)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="准备 Phase 0 的三条中文 smoke 音频")
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--download-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = prepare(args.manifest_dir, args.download_dir)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
