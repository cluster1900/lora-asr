"""Shared Phase 1 manifest schema, config, split, and status rules."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

import yaml

DATA_SCHEMA = "qwen3-asr-data-v1"
MANIFEST_ROW_SCHEMA = "qwen3-asr-manifest-v2"
MANIFEST_SET_SCHEMA = "qwen3-asr-manifest-set-v2"
REJECTED_SCHEMA = "qwen3-asr-rejected-v1"
SHA_SCHEMA = "qwen3-asr-sha256-v1"
PROCESSOR_VERSION = "phase1-aug-v1"
SCENARIOS = ("clean", "noise", "reverb", "far_field", "dropout")
SPLITS = ("train", "val", "test")
SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,80}$")
OFFICIAL_SPLIT_TO_MANIFEST = {"train": "train", "dev": "val", "test": "test"}
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")
STATUS_CODE = {"passed": 0, "failed": 1, "insufficient_data": 3, "data_failed": 4}


def sha256_file(path: Path) -> str:
    """Hash file bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scenario_seed(seed: int, source_id: str, scenario: str) -> int:
    """Derive a NumPy seed without using Python's salted hash."""
    digest = hashlib.sha256(f"{seed}:{source_id}:{scenario}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def assign_split(seed: int, source_id: str, val_fraction: float, test_fraction: float) -> str:
    """Map one source id onto train, val, or test."""
    digest = hashlib.sha256(f"{seed}:{source_id}:split".encode("utf-8")).digest()
    unit = int.from_bytes(digest[:8], "big") / 2**64
    if unit < test_fraction:
        return "test"
    if unit < test_fraction + val_fraction:
        return "val"
    return "train"


def split_mode(rows: list[dict[str, Any]]) -> str:
    """Return hash or official. A mixture is a source-contract error."""
    touched = 0
    for row in rows:
        if "speaker_id" in row or "official_split" in row:
            touched += 1
    if touched == 0:
        return "hash"
    if touched != len(rows):
        raise ValueError("mixed_split_mode")
    return "official"


def sample_id_for(source_id: str, scenario: str) -> str:
    """Build the global sample id used in every split file."""
    return f"{source_id}__{scenario}"


def text_problem(text: object) -> str | None:
    """Return a reason when a transcript cannot enter the manifest."""
    if not isinstance(text, str) or text.strip() == "":
        return "empty_text"
    if "language Chinese" in text or "<asr_text>" in text or "<|" in text:
        return "text_has_prompt_marker"
    if LATIN_RE.search(text):
        return "text_has_latin"
    if CJK_RE.search(text) is None:
        return "text_not_chinese"
    return None


def safe_relative_audio(audio: object) -> bool:
    """Accept a manifest-relative POSIX path and reject traversal."""
    if not isinstance(audio, str) or audio == "" or audio.startswith(("/", "\\")):
        return False
    parts = Path(audio).parts
    return ".." not in parts and "\\" not in audio


def load_data_config(path: Path) -> dict[str, Any]:
    """Parse the Phase 1 data YAML and reject unknown fields."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    _reject_unknown(
        raw,
        {
            "schema_version",
            "seed",
            "source_version",
            "audited",
            "audio_processor_version",
            "language",
            "sample_rate",
            "duration_seconds",
            "splits",
            "quota",
            "scenarios",
            "augmentation",
        },
        "config",
    )
    if raw.get("schema_version") != DATA_SCHEMA:
        raise ValueError(f"schema_version must be {DATA_SCHEMA}")
    seed = _as_int(raw.get("seed"), "seed")
    source_version = raw.get("source_version")
    if not isinstance(source_version, str) or source_version.strip() == "":
        raise ValueError("source_version must be a non-empty string")
    audited = _as_bool(raw.get("audited"), "audited")
    if raw.get("audio_processor_version") != PROCESSOR_VERSION:
        raise ValueError(f"audio_processor_version must be {PROCESSOR_VERSION}")
    if raw.get("language") != "zh":
        raise ValueError("language must be zh")
    sample_rate = _as_int(raw.get("sample_rate"), "sample_rate")
    if sample_rate != 16000:
        raise ValueError("sample_rate must be 16000")
    duration = _section(raw.get("duration_seconds"), "duration_seconds", {"min", "max"})
    minimum = _as_float(duration.get("min"), "duration_seconds.min")
    maximum = _as_float(duration.get("max"), "duration_seconds.max")
    if not 0 < minimum < maximum:
        raise ValueError("duration bounds must satisfy 0 < min < max")
    splits = _section(raw.get("splits"), "splits", {"val", "test"})
    val_fraction = _as_float(splits.get("val"), "splits.val")
    test_fraction = _as_float(splits.get("test"), "splits.test")
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("val and test fractions must be non-negative and sum to less than 1")
    quota = _section(raw.get("quota"), "quota", {"min_val_rows_per_scenario"})
    minimum_rows = _as_int(quota.get("min_val_rows_per_scenario"), "quota.min_val_rows_per_scenario")
    if minimum_rows < 1:
        raise ValueError("quota.min_val_rows_per_scenario must be at least 1")
    scenarios = raw.get("scenarios")
    if list(scenarios or []) != list(SCENARIOS):
        raise ValueError(f"scenarios must be {list(SCENARIOS)}")
    augmentation = _augmentation(raw.get("augmentation"))
    return {
        "schema_version": DATA_SCHEMA,
        "seed": seed,
        "source_version": source_version,
        "audited": audited,
        "audio_processor_version": PROCESSOR_VERSION,
        "language": "zh",
        "sample_rate": sample_rate,
        "duration_seconds": {"min": minimum, "max": maximum},
        "min_frames": round(minimum * sample_rate),
        "max_frames": round(maximum * sample_rate),
        "splits": {"val": val_fraction, "test": test_fraction},
        "min_val_rows_per_scenario": minimum_rows,
        "augmentation": augmentation,
        "config_sha256": sha256_file(path),
    }


def frame_bounds(config: dict[str, Any]) -> tuple[int, int]:
    """Return inclusive PCM frame limits at 16 kHz."""
    return int(config["min_frames"]), int(config["max_frames"])


def _augmentation(raw: object) -> dict[str, Any]:
    section = _section(raw, "augmentation", {"noise", "reverb", "far_field", "dropout"})
    noise = _section(section.get("noise"), "augmentation.noise", {"snr_db"})
    reverb = _section(section.get("reverb"), "augmentation.reverb", {"delay_ms", "decay", "repeats"})
    far_field = _section(section.get("far_field"), "augmentation.far_field", {"gain", "lowpass_window"})
    dropout = _section(section.get("dropout"), "augmentation.dropout", {"start_ratio", "width_ratio"})
    snr_db = _as_float(noise.get("snr_db"), "augmentation.noise.snr_db")
    delay_ms = _as_float(reverb.get("delay_ms"), "augmentation.reverb.delay_ms")
    decay = _as_float(reverb.get("decay"), "augmentation.reverb.decay")
    repeats = _as_int(reverb.get("repeats"), "augmentation.reverb.repeats")
    gain = _as_float(far_field.get("gain"), "augmentation.far_field.gain")
    window = _as_int(far_field.get("lowpass_window"), "augmentation.far_field.lowpass_window")
    start_ratio = _as_float(dropout.get("start_ratio"), "augmentation.dropout.start_ratio")
    width_ratio = _as_float(dropout.get("width_ratio"), "augmentation.dropout.width_ratio")
    if delay_ms <= 0 or not 0 < decay < 1 or repeats < 1:
        raise ValueError("reverb settings are out of range")
    if not 0 < gain < 1 or window < 2:
        raise ValueError("far_field settings are out of range")
    if start_ratio < 0 or width_ratio <= 0 or start_ratio + width_ratio >= 1:
        raise ValueError("dropout span must leave some audio")
    return {
        "noise": {"snr_db": snr_db},
        "reverb": {"delay_ms": delay_ms, "decay": decay, "repeats": repeats},
        "far_field": {"gain": gain, "lowpass_window": window},
        "dropout": {"start_ratio": start_ratio, "width_ratio": width_ratio},
    }


def _section(raw: object, name: str, allowed: set[str]) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be a mapping")
    _reject_unknown(raw, allowed, name)
    return raw


def _reject_unknown(section: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {unknown}")


def _as_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _as_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _as_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError(f"{field} must be finite")
    return number
