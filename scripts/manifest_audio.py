"""Deterministic 16 kHz degradations for the five Phase 1 scenarios."""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.manifest_contract import scenario_seed
from scripts.prepare_phase0_audio import add_gaussian_noise, to_mono_16k


def load_mono_16k(path: str, sample_rate_reader: Any) -> np.ndarray:
    """Read a source file and return mono float32 audio at 16 kHz."""
    samples, sample_rate = sample_rate_reader(path)
    audio = to_mono_16k(np.asarray(samples), int(sample_rate))
    if audio.size == 0 or float(np.max(np.abs(audio))) == 0.0:
        raise ValueError("silent_audio")
    return np.asarray(audio, dtype=np.float32)


def render_scenarios(
    clean: np.ndarray,
    seed: int,
    source_id: str,
    augmentation: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Return the five waveforms. Noise uses a source-specific seed."""
    noise_seed = scenario_seed(seed, source_id, "noise")
    return {
        "clean": clean,
        "noise": add_gaussian_noise(clean, float(augmentation["noise"]["snr_db"]), noise_seed),
        "reverb": apply_reverb(clean, augmentation["reverb"]),
        "far_field": apply_far_field(clean, augmentation["far_field"]),
        "dropout": apply_dropout(clean, augmentation["dropout"]),
    }


def apply_reverb(clean: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    """Add a fixed decaying comb. No random tap positions."""
    delay = max(1, int(round(16000 * float(settings["delay_ms"]) / 1000.0)))
    mixed = np.array(clean, dtype=np.float32, copy=True)
    for repeat in range(1, int(settings["repeats"]) + 1):
        offset = delay * repeat
        if offset >= clean.size:
            break
        mixed[offset:] += clean[:-offset] * np.float32(float(settings["decay"]) ** repeat)
    return _limit(mixed)


def apply_far_field(clean: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    """Attenuate a moving-average lowpass. The window is fixed by config."""
    window = int(settings["lowpass_window"])
    kernel = np.ones(window, dtype=np.float32) / np.float32(window)
    smoothed = np.convolve(clean, kernel, mode="same").astype(np.float32)
    return _limit(smoothed * np.float32(float(settings["gain"])))


def apply_dropout(clean: np.ndarray, settings: dict[str, Any]) -> np.ndarray:
    """Zero one interior span. The endpoints stay intact."""
    audio = np.array(clean, dtype=np.float32, copy=True)
    start = int(audio.size * float(settings["start_ratio"]))
    stop = int(audio.size * (float(settings["start_ratio"]) + float(settings["width_ratio"])))
    stop = min(audio.size, max(start + 1, stop))
    audio[start:stop] = 0.0
    return audio


def _limit(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio)))
    if peak > 1.0:
        audio = audio / np.float32(peak)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)
