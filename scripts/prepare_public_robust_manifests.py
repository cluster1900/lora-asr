#!/usr/bin/env python3
"""Prepare and validate the fixed public-data manifests for fast A2S training.

The tool deliberately separates remote metadata inspection from audio
materialization. ``probe`` only asks ``datasets`` for builder metadata; ``build``
and ``smoke`` consume local candidate JSONL files produced by the Colab staging
step and never call the Hub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


LANGUAGES = ("en", "zh")
CONDITION_GROUPS = {"clean", "atomic", "compound"}
AUDIO_ORIGINS = {"clean", "real", "synthetic"}
DERIVED_ROLES = {"canary", "curriculum"}
PINNED_REVISION_LENGTH = 40
STAGE_SOURCE_NAMES = ("robust", "english_clean", "chinese_clean", "robust_test")


class ConfigError(RuntimeError):
    """Raised when the fixed data config is incomplete or inconsistent."""


class ManifestError(RuntimeError):
    """Raised when candidate or canonical manifest data fails a hard gate."""


def load_config(path: Path) -> dict[str, Any]:
    """Load JSON with stdlib or YAML with an explicit PyYAML dependency."""
    if not path.exists():
        raise ConfigError(f"config does not exist: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigError(
                "YAML config requires PyYAML; install it with `pip install PyYAML` "
                "or pass an equivalent JSON config."
            ) from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping")
    return data


def require_mapping(value: Any, name: str) -> dict[str, Any]:
    """Return a mutable mapping or raise a readable config error."""
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def config_section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Read one required top-level config section."""
    return require_mapping(config.get(name), name)


def assert_pinned_revision(revision: Any, source_name: str) -> str:
    """Require a full hexadecimal Hub commit rather than a moving branch."""
    value = str(revision or "")
    if len(value) != PINNED_REVISION_LENGTH or any(
        char not in "0123456789abcdef" for char in value.lower()
    ):
        raise ConfigError(
            f"sources.{source_name}.revision must be a 40-character commit hash"
        )
    return value


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield JSON objects with source line numbers."""
    if not path.exists():
        raise ManifestError(f"JSONL does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ManifestError(f"{path}:{line_number}: row must be a JSON object")
            yield line_number, row


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into memory."""
    return [row for _, row in iter_jsonl(path)]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomically write JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Atomically write formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(dict(data), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


class StageWriter:
    """Append candidate rows with bounded durable-checkpoint overhead."""

    def __init__(self, path: Path, checkpoint_rows: int) -> None:
        """Configure append output and the maximum unsynced row count."""
        self.path = path
        self.checkpoint_rows = max(1, int(checkpoint_rows))
        self.handle: Any = None
        self.pending = 0

    def __enter__(self) -> "StageWriter":
        """Open the candidate file in append mode for resumable staging."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a", encoding="utf-8")
        return self

    def append(self, row: Mapping[str, Any]) -> None:
        """Append one candidate row and checkpoint at the configured interval."""
        self.handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
        self.pending += 1
        if self.pending >= self.checkpoint_rows:
            self.sync()

    def sync(self) -> None:
        """Flush pending rows through the operating system to durable storage."""
        if self.handle is None or not self.pending:
            return
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.pending = 0

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """Persist the final partial checkpoint and close the file."""
        if self.handle is not None:
            self.sync()
            self.handle.close()


def file_sha256(path: Path) -> str:
    """Hash a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, namespace: str, identity: Any) -> str:
    """Return a cross-process deterministic ordering key."""
    payload = f"{seed}\0{namespace}\0{identity}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_order(
    rows: Iterable[dict[str, Any]], seed: int, namespace: str
) -> list[dict[str, Any]]:
    """Order rows reproducibly by sample identity."""
    return sorted(
        rows,
        key=lambda row: (
            stable_key(seed, namespace, row.get("sample_id", "")),
            str(row.get("sample_id", "")),
        ),
    )


def distribute_total(names: Sequence[str], total: int) -> dict[str, int]:
    """Distribute a total over sorted names, assigning remainder to the front."""
    ordered = sorted(str(name) for name in names)
    if not ordered:
        if total:
            raise ConfigError("cannot distribute a non-zero total over no names")
        return {}
    if len(set(ordered)) != len(ordered):
        raise ConfigError("quota names must be unique")
    if total < 0:
        raise ConfigError("quota total must be non-negative")
    base, remainder = divmod(total, len(ordered))
    return {
        name: base + int(index < remainder)
        for index, name in enumerate(ordered)
    }


def plan_scenario_quotas(
    split_names: Sequence[str],
    atomic_splits: Sequence[str],
    atomic_per_split: int,
    compound_total: int,
    expected_compound_splits: int = 47,
) -> dict[str, int]:
    """Build the fixed atomic plus sorted-remainder compound quota map."""
    all_splits = {str(name) for name in split_names}
    atomics = {str(name) for name in atomic_splits}
    missing = sorted(atomics - all_splits)
    if missing:
        raise ConfigError(f"missing atomic splits: {', '.join(missing)}")
    compounds = sorted(all_splits - atomics)
    if len(compounds) != expected_compound_splits:
        raise ConfigError(
            f"compound split count {len(compounds)} != expected {expected_compound_splits}"
        )
    quotas = {name: int(atomic_per_split) for name in sorted(atomics)}
    quotas.update(distribute_total(compounds, int(compound_total)))
    return quotas


def plan_language_quotas(scenario_quotas: Mapping[str, int]) -> dict[tuple[str, str], int]:
    """Split each scenario as evenly as possible across English and Chinese."""
    result: dict[tuple[str, str], int] = {}
    for index, scenario in enumerate(sorted(scenario_quotas)):
        total = int(scenario_quotas[scenario])
        half, remainder = divmod(total, 2)
        en_extra = int(bool(remainder) and index % 2 == 0)
        result[(scenario, "en")] = half + en_extra
        result[(scenario, "zh")] = total - half - en_extra
    return result


def infer_language(text: str) -> str | None:
    """Infer a conservative en/zh label from a gold transcript."""
    chinese = 0
    latin = 0
    for char in text:
        codepoint = ord(char)
        if (
            0x3400 <= codepoint <= 0x4DBF
            or 0x4E00 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
        ):
            chinese += 1
        elif "LATIN" in unicodedata.name(char, ""):
            latin += 1
    if chinese and chinese >= latin:
        return "zh"
    if latin and not chinese:
        return "en"
    return None


def source_identity(row: Mapping[str, Any]) -> str:
    """Return the required source utterance identity."""
    value = row.get("source_utterance_id") or row.get("name")
    if value not in (None, ""):
        return str(value)
    file_name = row.get("file_name")
    if file_name:
        return Path(str(file_name)).stem
    raise ManifestError("source identity missing (need source_utterance_id, name, or file_name)")


def audio_path_value(row: Mapping[str, Any]) -> str:
    """Extract a materialized audio path from common candidate representations."""
    value = row.get("audio")
    if isinstance(value, str) and value:
        return value
    if isinstance(value, Mapping) and value.get("path"):
        return str(value["path"])
    if row.get("file_name"):
        return str(row["file_name"])
    raise ManifestError("materialized audio path missing")


def finite_float(value: Any, field: str) -> float:
    """Parse a finite numeric value."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ManifestError(f"{field} must be finite")
    return parsed


def safe_path_component(value: Any) -> str:
    """Make a stable, portable directory component."""
    cleaned = "".join(
        char if char.isalnum() or char in {"-", "_", "."} else "_"
        for char in str(value)
    ).strip("._")
    return cleaned or "unknown"


def stage_source_identity(
    row: Mapping[str, Any],
    *,
    source_name: str,
    dataset_id: str,
    source_split: str,
    fallback_index: int,
) -> str:
    """Use the pre-augmentation index for robust cross-scenario isolation."""
    source_index = row.get("index")
    if source_name in {"robust", "robust_test"} and source_index not in (None, ""):
        return f"{dataset_id}:{source_index}"
    audio = row.get("audio")
    streamed_audio_path = audio.get("path") if isinstance(audio, Mapping) else None
    identity = (
        row.get("source_utterance_id")
        or row.get("id")
        or row.get("name")
        or row.get("file_name")
        or row.get("audio_path")
        or streamed_audio_path
    )
    if identity in (None, ""):
        identity = f"{source_split}:{fallback_index}"
    return f"{dataset_id}:{identity}"


def robust_partition_role(identity: str, seed: int, validation_percent: int) -> str:
    """Assign one base utterance to a fixed train or validation partition."""
    if not 1 <= validation_percent <= 50:
        raise ConfigError("staging.robust_validation_percent must be in [1, 50]")
    bucket = int(stable_key(seed, "robust-partition", identity)[:8], 16) % 100
    return "validation" if bucket < validation_percent else "train"


def stage_descriptor(
    row: Mapping[str, Any],
    *,
    source_name: str,
    source_config: Mapping[str, Any],
    source_split: str,
    seed: int,
    validation_percent: int,
    fallback_index: int,
    mode: str,
) -> dict[str, Any]:
    """Create candidate metadata before downloading or copying audio bytes."""
    answer = str(row.get("answer") or row.get("text") or "").strip()
    if not answer:
        raise ManifestError("answer is empty")
    if source_name == "english_clean":
        language = "en"
    elif source_name == "chinese_clean":
        language = "zh"
    else:
        language = str(row.get("language") or "").lower()
        if language not in LANGUAGES:
            language = infer_language(answer) or ""
    if language not in LANGUAGES:
        raise ManifestError("language is mixed or could not be inferred")

    dataset_id = str(source_config.get("dataset_id") or "")
    revision = assert_pinned_revision(source_config.get("revision"), source_name)
    identity = stage_source_identity(
        row,
        source_name=source_name,
        dataset_id=dataset_id,
        source_split=source_split,
        fallback_index=fallback_index,
    )
    source_index = row.get("index")
    if source_index in (None, ""):
        source_index = row.get("id") or identity
    if source_name == "robust":
        scenario = source_split
        origin = "synthetic"
        role = (
            robust_partition_role(identity, seed, validation_percent)
            if mode == "full"
            else "train"
        )
    elif source_name == "robust_test":
        scenario = source_split.removeprefix("real_").removeprefix("syn_")
        origin = "real" if source_split.startswith("real_") else "synthetic"
        role = "test"
    else:
        scenario = "clean"
        origin = "clean"
        role = (
            "validation"
            if mode == "full" and source_split == str(source_config["validation_split"])
            else "train"
        )
    descriptor: dict[str, Any] = {
        "sample_id": f"{dataset_id}:{source_split}:{source_index}",
        "answer": answer,
        "language": language,
        "scenario": scenario,
        "audio_origin": origin,
        "selection_role": role,
        "source_dataset": dataset_id,
        "source_revision": revision,
        "source_split": source_split,
        "source_index": source_index,
        "source_utterance_id": identity,
        "license": str(source_config.get("license") or ""),
        "seed": seed,
    }
    for source_field, output_field in (
        ("name", "source_name"),
        ("file_name", "file_name"),
        ("speaker_id", "speaker_id"),
    ):
        if row.get(source_field) not in (None, ""):
            descriptor[output_field] = row[source_field]
    return descriptor


def _audio_suffix(audio: Any, row: Mapping[str, Any]) -> str:
    """Infer a short source extension, falling back to a WAV container."""
    candidates: list[Any] = []
    if isinstance(audio, Mapping):
        candidates.append(audio.get("path"))
    candidates.extend((row.get("file_name"), row.get("audio_path")))
    for candidate in candidates:
        if candidate:
            suffix = Path(str(candidate)).suffix.lower()
            if suffix and len(suffix) <= 6:
                return suffix
    return ".wav"


def _audio_duration(path: Path) -> float:
    """Decode audio metadata and return duration in seconds."""
    try:
        import soundfile as sf  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ConfigError(
            "audio staging requires soundfile; install requirements-colab.txt"
        ) from exc
    try:
        return float(sf.info(str(path)).duration)
    except Exception as exc:
        raise ManifestError(f"audio cannot be decoded: {path}: {exc}") from exc


def materialize_stage_audio(
    row: Mapping[str, Any],
    descriptor: Mapping[str, Any],
    *,
    source_name: str,
    data_root: Path,
) -> tuple[str, float, str, int]:
    """Materialize one streaming Audio value and return canonical metadata."""
    audio = row.get("audio")
    suffix = _audio_suffix(audio, row)
    identity_hash = hashlib.sha256(str(descriptor["sample_id"]).encode("utf-8")).hexdigest()[:24]
    relative = (
        Path(safe_path_component(source_name))
        / safe_path_component(descriptor["source_split"])
        / f"{identity_hash}{suffix}"
    )
    destination = data_root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")

    if not destination.exists():
        try:
            if isinstance(audio, Mapping) and audio.get("bytes") is not None:
                payload = audio["bytes"]
                if isinstance(payload, memoryview):
                    payload = payload.tobytes()
                if not isinstance(payload, (bytes, bytearray)):
                    raise ManifestError("audio.bytes is not bytes")
                temporary.write_bytes(bytes(payload))
            elif isinstance(audio, Mapping) and audio.get("array") is not None:
                try:
                    import soundfile as sf  # type: ignore[import-not-found]
                except ImportError as exc:
                    raise ConfigError(
                        "decoded audio staging requires soundfile"
                    ) from exc
                sampling_rate = int(audio.get("sampling_rate") or 16000)
                sf.write(str(temporary), audio["array"], sampling_rate, format="WAV")
            else:
                source_value = audio.get("path") if isinstance(audio, Mapping) else audio
                if not source_value:
                    source_value = row.get("audio_path")
                source_path = Path(str(source_value or "")).expanduser()
                if not source_path.is_file():
                    raise ManifestError("streamed audio has neither bytes nor a readable path")
                shutil.copyfile(source_path, temporary)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()

    duration = _audio_duration(destination)
    return relative.as_posix(), duration, file_sha256(destination), destination.stat().st_size


def stage_bucket(row: Mapping[str, Any], source_name: str, mode: str) -> tuple[str, ...]:
    """Map a candidate row to the quota bucket used during staging."""
    if source_name == "robust":
        role = str(row.get("selection_role") or "train") if mode == "full" else "smoke"
        return role, str(row["scenario"]), str(row["language"])
    if source_name in {"english_clean", "chinese_clean"}:
        role = str(row.get("selection_role") or "train") if mode == "full" else "train"
        return role, str(row["language"])
    if mode == "full":
        return (str(row["source_split"]),)
    return str(row["audio_origin"]), str(row["language"])


def load_valid_stage_rows(path: Path, data_root: Path) -> tuple[list[dict[str, Any]], int]:
    """Repair a candidate manifest and retain only complete, hash-valid rows."""
    if not path.exists():
        return [], 0
    valid: list[dict[str, Any]] = []
    seen: set[str] = set()
    dropped = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                sample_id = str(row["sample_id"])
                audio_path = Path(str(row["audio"]))
                resolved = audio_path if audio_path.is_absolute() else data_root / audio_path
                if sample_id in seen or not resolved.is_file():
                    raise ValueError("duplicate sample or missing audio")
                if str(row.get("audio_sha256") or "") != file_sha256(resolved):
                    resolved.unlink(missing_ok=True)
                    raise ValueError("audio hash mismatch")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
                dropped += 1
                continue
            seen.add(sample_id)
            valid.append(row)
    write_jsonl(path, valid)
    return valid, dropped


def stage_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    source_name: str,
    source_config: Mapping[str, Any],
    source_split: str,
    mode: str,
    seed: int,
    validation_percent: int,
    targets: Mapping[tuple[str, ...], int],
    active_buckets: set[tuple[str, ...]],
    candidate_rows: list[dict[str, Any]],
    candidate_path: Path,
    data_root: Path,
    minimum_duration: float,
    maximum_duration: float,
    checkpoint_rows: int,
    rejects: list[dict[str, Any]],
) -> int:
    """Consume one source split until its active staging buckets are full."""
    counts = Counter(stage_bucket(row, source_name, mode) for row in candidate_rows)
    if all(counts[key] >= int(targets[key]) for key in active_buckets):
        return 0
    known_ids = {str(row["sample_id"]) for row in candidate_rows}
    materialized = 0
    with StageWriter(candidate_path, checkpoint_rows) as writer:
        for index, source_row in enumerate(rows):
            try:
                descriptor = stage_descriptor(
                    source_row,
                    source_name=source_name,
                    source_config=source_config,
                    source_split=source_split,
                    seed=seed,
                    validation_percent=validation_percent,
                    fallback_index=index,
                    mode=mode,
                )
                bucket = stage_bucket(descriptor, source_name, mode)
                if bucket not in active_buckets or counts[bucket] >= int(targets[bucket]):
                    continue
                if str(descriptor["sample_id"]) in known_ids:
                    continue
                audio, duration, audio_hash, audio_size = materialize_stage_audio(
                    source_row,
                    descriptor,
                    source_name=source_name,
                    data_root=data_root,
                )
                if not minimum_duration <= duration <= maximum_duration:
                    (data_root / audio).unlink(missing_ok=True)
                    raise ManifestError(
                        f"duration_s {duration} outside [{minimum_duration}, {maximum_duration}]"
                    )
                candidate = dict(descriptor)
                candidate.update(
                    {
                        "audio": audio,
                        "duration_s": duration,
                        "audio_sha256": audio_hash,
                        "audio_size_bytes": audio_size,
                    }
                )
                writer.append(candidate)
                candidate_rows.append(candidate)
                known_ids.add(str(candidate["sample_id"]))
                counts[bucket] += 1
                materialized += 1
                if all(counts[key] >= int(targets[key]) for key in active_buckets):
                    break
            except (ConfigError, ManifestError, KeyError, TypeError, ValueError, OSError) as exc:
                rejects.append(
                    {
                        "source": source_name,
                        "source_split": source_split,
                        "source_index": source_row.get("index", index),
                        "reason": str(exc),
                    }
                )
    return materialized


def normalize_candidate(
    row: Mapping[str, Any],
    *,
    source_config: Mapping[str, Any],
    source_kind: str,
    atomic_splits: Sequence[str],
    seed: int,
    fallback_index: int,
) -> dict[str, Any]:
    """Convert one staged source row into the canonical project schema."""
    answer = str(row.get("answer") or row.get("text") or "").strip()
    if not answer:
        raise ManifestError("answer is empty")
    language = str(row.get("language") or "").lower()
    if language not in LANGUAGES:
        language = infer_language(answer) or ""
    if language not in LANGUAGES:
        raise ManifestError("language is mixed or could not be inferred")

    source_split = str(
        row.get("source_split") or row.get("subset") or row.get("split") or ""
    )
    scenario = str(row.get("scenario") or source_split or "clean")
    if not source_split:
        source_split = scenario
    if source_kind == "clean":
        condition_group = "clean"
        scenario = "clean"
    elif source_kind == "robust":
        condition_group = "atomic" if scenario in set(atomic_splits) else "compound"
    else:
        condition_group = str(row.get("condition_group") or "atomic")

    source_dataset = str(
        row.get("source_dataset") or source_config.get("dataset_id") or ""
    )
    source_revision = str(
        row.get("source_revision") or source_config.get("revision") or ""
    )
    source_index = row.get("source_index", row.get("index", fallback_index))
    identity = source_identity(row)
    sample_id = str(
        row.get("sample_id")
        or f"{source_dataset}:{source_split}:{source_index}"
    )
    duration = finite_float(
        row.get("duration_s", row.get("duration")), "duration_s"
    )
    default_origin = "clean" if source_kind == "clean" else "synthetic"
    origin = str(row.get("audio_origin") or row.get("origin") or default_origin).lower()
    if origin == "syn":
        origin = "synthetic"

    canonical: dict[str, Any] = {
        "sample_id": sample_id,
        "audio": audio_path_value(row),
        "answer": answer,
        "language": language,
        "scenario": scenario,
        "condition_group": condition_group,
        "audio_origin": origin,
        "source_dataset": source_dataset,
        "source_revision": source_revision,
        "source_split": source_split,
        "source_index": source_index,
        "source_utterance_id": identity,
        "speaker_id": row.get("speaker_id"),
        "duration_s": duration,
        "license": str(row.get("license") or source_config.get("license") or ""),
        "seed": int(row.get("seed", seed)),
        "audio_sha256": str(row.get("audio_sha256") or ""),
    }
    source_name = row.get("source_name", row.get("name"))
    if source_name not in (None, ""):
        canonical["source_name"] = str(source_name)
    if row.get("benchmark_id") not in (None, ""):
        canonical["benchmark_id"] = str(row["benchmark_id"])
    if row.get("selection_role") not in (None, ""):
        canonical["selection_role"] = str(row["selection_role"])
    return canonical


def load_candidates(
    path: Path,
    *,
    source_config: Mapping[str, Any],
    source_kind: str,
    atomic_splits: Sequence[str],
    seed: int,
    min_duration_s: float,
    max_duration_s: float,
    rejects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize staged candidates while recording row-level rejects."""
    rows: list[dict[str, Any]] = []
    for line_number, source_row in iter_jsonl(path):
        try:
            row = normalize_candidate(
                source_row,
                source_config=source_config,
                source_kind=source_kind,
                atomic_splits=atomic_splits,
                seed=seed,
                fallback_index=line_number - 1,
            )
            duration = float(row["duration_s"])
            if not min_duration_s <= duration <= max_duration_s:
                raise ManifestError(
                    f"duration_s {duration} outside [{min_duration_s}, {max_duration_s}]"
                )
            rows.append(row)
        except (ManifestError, TypeError, ValueError) as exc:
            rejects.append(
                {
                    "source_jsonl": str(path),
                    "source_line": line_number,
                    "source_locator": source_row.get("sample_id")
                    or source_row.get("name")
                    or source_row.get("file_name")
                    or source_row.get("index"),
                    "reason": str(exc),
                }
            )
    return rows


def select_stratified(
    rows: Sequence[dict[str, Any]],
    quotas: Mapping[tuple[str, str], int],
    *,
    seed: int,
    namespace: str,
    excluded_source_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select exact per-scenario/language quotas in deterministic order."""
    excluded = excluded_source_ids or set()
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("source_utterance_id", "")) in excluded:
            continue
        buckets[(str(row.get("scenario", "")), str(row.get("language", "")))].append(row)

    selected: list[dict[str, Any]] = []
    shortages: list[str] = []
    for key in sorted(quotas):
        required = int(quotas[key])
        ordered = stable_order(buckets.get(key, []), seed, f"{namespace}:{key[0]}:{key[1]}")
        if len(ordered) < required:
            shortages.append(f"{key[0]}/{key[1]}={len(ordered)}/{required}")
            continue
        selected.extend(ordered[:required])
    if shortages:
        raise ManifestError("candidate quota shortage: " + ", ".join(shortages[:20]))
    return stable_order(selected, seed, namespace)


def select_clean(
    rows: Sequence[dict[str, Any]],
    *,
    language: str,
    count: int,
    seed: int,
    namespace: str,
    excluded_source_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Select one clean-language quota."""
    excluded = excluded_source_ids or set()
    candidates = [
        row
        for row in rows
        if row.get("language") == language
        and str(row.get("source_utterance_id", "")) not in excluded
    ]
    ordered = stable_order(candidates, seed, namespace)
    if len(ordered) < count:
        raise ManifestError(
            f"clean {language} candidate quota shortage: {len(ordered)}/{count}"
        )
    return ordered[:count]


def select_bench_smoke(
    rows: Sequence[dict[str, Any]], seed: int, per_stratum: int
) -> list[dict[str, Any]]:
    """Select a tiny en/zh by real/synthetic benchmark probe."""
    quotas = {
        (language, origin): per_stratum
        for language in LANGUAGES
        for origin in ("real", "synthetic")
    }
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[(str(row.get("language")), str(row.get("audio_origin")))].append(row)
    selected: list[dict[str, Any]] = []
    for key, required in sorted(quotas.items()):
        ordered = stable_order(buckets.get(key, []), seed, f"bench-smoke:{key}")
        if len(ordered) < required:
            raise ManifestError(
                f"bench smoke quota shortage for {key[0]}/{key[1]}: "
                f"{len(ordered)}/{required}"
            )
        selected.extend(ordered[:required])
    return stable_order(selected, seed, "bench-smoke")


def selected_source_ids(rows: Iterable[Mapping[str, Any]]) -> set[str]:
    """Collect non-empty source utterance IDs."""
    return {
        str(row["source_utterance_id"])
        for row in rows
        if row.get("source_utterance_id") not in (None, "")
    }


def _selection_settings(config: Mapping[str, Any]) -> tuple[int, float, float]:
    """Return the shared seed and accepted duration range."""
    project = config_section(config, "project")
    selection = config_section(config, "selection")
    return (
        int(project["seed"]),
        float(selection["min_duration_s"]),
        float(selection["max_duration_s"]),
    )


def load_all_candidate_inputs(
    config: Mapping[str, Any], args: argparse.Namespace
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Load the four local candidate files used by build and smoke."""
    sources = config_section(config, "sources")
    robust_config = require_mapping(sources.get("robust"), "sources.robust")
    atomic_splits = [str(item) for item in robust_config.get("atomic_splits", [])]
    seed, minimum, maximum = _selection_settings(config)
    rejects: list[dict[str, Any]] = []
    robust = load_candidates(
        Path(args.robust_candidates).expanduser(),
        source_config=robust_config,
        source_kind="robust",
        atomic_splits=atomic_splits,
        seed=seed,
        min_duration_s=minimum,
        max_duration_s=maximum,
        rejects=rejects,
    )
    english = load_candidates(
        Path(args.english_clean_candidates).expanduser(),
        source_config=require_mapping(sources.get("english_clean"), "sources.english_clean"),
        source_kind="clean",
        atomic_splits=atomic_splits,
        seed=seed,
        min_duration_s=minimum,
        max_duration_s=maximum,
        rejects=rejects,
    )
    chinese = load_candidates(
        Path(args.chinese_clean_candidates).expanduser(),
        source_config=require_mapping(sources.get("chinese_clean"), "sources.chinese_clean"),
        source_kind="clean",
        atomic_splits=atomic_splits,
        seed=seed,
        min_duration_s=minimum,
        max_duration_s=maximum,
        rejects=rejects,
    )
    bench = load_candidates(
        Path(args.bench_candidates).expanduser(),
        source_config=require_mapping(sources.get("robust_test"), "sources.robust_test"),
        source_kind="bench",
        atomic_splits=atomic_splits,
        seed=seed,
        min_duration_s=minimum,
        max_duration_s=maximum,
        rejects=rejects,
    )
    return robust, english, chinese, bench, rejects


def build_full_selection(
    config: Mapping[str, Any],
    robust_rows: Sequence[dict[str, Any]],
    english_rows: Sequence[dict[str, Any]],
    chinese_rows: Sequence[dict[str, Any]],
    bench_rows: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build the exact 200k/10k/5k deterministic selection."""
    seed = int(config_section(config, "project")["seed"])
    sources = config_section(config, "sources")
    robust_config = require_mapping(sources.get("robust"), "sources.robust")
    selection = config_section(config, "selection")
    train_plan = require_mapping(selection.get("robust_train"), "selection.robust_train")
    val_plan = require_mapping(
        selection.get("robust_validation"), "selection.robust_validation"
    )
    atomic_splits = [str(item) for item in robust_config["atomic_splits"]]
    split_names = sorted({str(row["scenario"]) for row in robust_rows})
    expected_compounds = int(robust_config["expected_compound_splits"])
    if len(split_names) != int(robust_config["expected_splits"]):
        raise ManifestError(
            f"robust candidate split count {len(split_names)} != "
            f"expected {robust_config['expected_splits']}"
        )

    train_scenarios = plan_scenario_quotas(
        split_names,
        atomic_splits,
        int(train_plan["atomic_per_split"]),
        int(train_plan["compound_total"]),
        expected_compounds,
    )
    val_scenarios = plan_scenario_quotas(
        split_names,
        atomic_splits,
        int(val_plan["atomic_per_split"]),
        int(val_plan["compound_total"]),
        expected_compounds,
    )
    robust_roles = {str(row.get("selection_role")) for row in robust_rows if row.get("selection_role")}
    if robust_roles:
        if robust_roles != {"train", "validation"} or any(
            not row.get("selection_role") for row in robust_rows
        ):
            raise ManifestError("staged robust candidates require complete train/validation roles")
        robust_train_pool = [row for row in robust_rows if row["selection_role"] == "train"]
        robust_validation_pool = [
            row for row in robust_rows if row["selection_role"] == "validation"
        ]
    else:
        robust_train_pool = list(robust_rows)
        robust_validation_pool = list(robust_rows)
    robust_train = select_stratified(
        robust_train_pool,
        plan_language_quotas(train_scenarios),
        seed=seed,
        namespace="robust-train",
    )
    train_sources = selected_source_ids(robust_train)
    robust_validation = select_stratified(
        robust_validation_pool,
        plan_language_quotas(val_scenarios),
        seed=seed,
        namespace="robust-validation",
        excluded_source_ids=train_sources,
    )

    clean_train_count = int(selection["clean_train_per_language"])
    clean_val_count = int(selection["clean_validation_per_language"])
    english_roles = {str(row.get("selection_role")) for row in english_rows if row.get("selection_role")}
    chinese_roles = {str(row.get("selection_role")) for row in chinese_rows if row.get("selection_role")}
    if english_roles or chinese_roles:
        if english_roles != {"train", "validation"} or chinese_roles != {
            "train",
            "validation",
        }:
            raise ManifestError("staged clean candidates require train and validation roles")
        if any(not row.get("selection_role") for row in [*english_rows, *chinese_rows]):
            raise ManifestError("staged clean candidate role is missing")
        english_train_pool = [row for row in english_rows if row["selection_role"] == "train"]
        english_validation_pool = [
            row for row in english_rows if row["selection_role"] == "validation"
        ]
        chinese_train_pool = [row for row in chinese_rows if row["selection_role"] == "train"]
        chinese_validation_pool = [
            row for row in chinese_rows if row["selection_role"] == "validation"
        ]
    else:
        english_train_pool = english_validation_pool = list(english_rows)
        chinese_train_pool = chinese_validation_pool = list(chinese_rows)
    en_train = select_clean(
        english_train_pool,
        language="en",
        count=clean_train_count,
        seed=seed,
        namespace="english-clean-train",
    )
    zh_train = select_clean(
        chinese_train_pool,
        language="zh",
        count=clean_train_count,
        seed=seed,
        namespace="chinese-clean-train",
    )
    en_validation = select_clean(
        english_validation_pool,
        language="en",
        count=clean_val_count,
        seed=seed,
        namespace="english-clean-validation",
        excluded_source_ids=selected_source_ids(en_train),
    )
    zh_validation = select_clean(
        chinese_validation_pool,
        language="zh",
        count=clean_val_count,
        seed=seed,
        namespace="chinese-clean-validation",
        excluded_source_ids=selected_source_ids(zh_train),
    )

    expected_bench = int(selection["expected_bench_rows"])
    if len(bench_rows) != expected_bench:
        raise ManifestError(
            f"Bench must be used in full: got {len(bench_rows)} rows, expected {expected_bench}"
        )
    train = stable_order(
        [*robust_train, *en_train, *zh_train], seed, "full-train"
    )
    validation = stable_order(
        [*robust_validation, *en_validation, *zh_validation],
        seed,
        "full-validation",
    )
    test = stable_order(bench_rows, seed, "full-bench")
    canary = build_validation_canary(config, validation)
    return {
        "train": train,
        "validation": validation,
        "canary": canary,
        "test": test,
    }


def build_validation_canary(
    config: Mapping[str, Any], validation_rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Derive the fixed 512-row phase canary from the 10k validation set."""
    seed = int(config_section(config, "project")["seed"])
    settings = config_section(config, "canary")
    robust_target = int(settings["robust_rows"])
    clean_target = int(settings["clean_rows"])
    if robust_target + clean_target != int(settings["expected_rows"]):
        raise ConfigError("canary robust_rows plus clean_rows must equal expected_rows")

    robust = [
        row for row in validation_rows if row.get("condition_group") in {"atomic", "compound"}
    ]
    clean = [row for row in validation_rows if row.get("condition_group") == "clean"]
    scenarios = sorted({str(row["scenario"]) for row in robust})
    robust_scenario_quotas = distribute_total(scenarios, robust_target)
    robust_selected = select_stratified(
        robust,
        plan_language_quotas(robust_scenario_quotas),
        seed=seed,
        namespace="validation-canary-robust",
    )
    clean_language_quotas = distribute_total(LANGUAGES, clean_target)
    clean_selected: list[dict[str, Any]] = []
    for language in LANGUAGES:
        clean_selected.extend(
            select_clean(
                clean,
                language=language,
                count=clean_language_quotas[language],
                seed=seed,
                namespace=f"validation-canary-clean-{language}",
            )
        )
    canary = stable_order(
        [*robust_selected, *clean_selected], seed, "validation-canary"
    )
    if len(canary) != int(settings["expected_rows"]):
        raise ManifestError(
            f"canary rows {len(canary)} != expected {settings['expected_rows']}"
        )
    return canary


def build_smoke_selection(
    config: Mapping[str, Any],
    robust_rows: Sequence[dict[str, Any]],
    english_rows: Sequence[dict[str, Any]],
    chinese_rows: Sequence[dict[str, Any]],
    bench_rows: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Build a 128-row train fixture plus a four-row Bench probe."""
    seed = int(config_section(config, "project")["seed"])
    sources = config_section(config, "sources")
    robust_config = require_mapping(sources.get("robust"), "sources.robust")
    smoke = config_section(config, "smoke")
    split_names = sorted({str(row["scenario"]) for row in robust_rows})
    if len(split_names) != int(robust_config["expected_splits"]):
        raise ManifestError(
            f"smoke needs all {robust_config['expected_splits']} robust splits; "
            f"got {len(split_names)}"
        )
    per_language = int(smoke["robust_per_language_per_split"])
    quotas = {
        (scenario, language): per_language
        for scenario in split_names
        for language in LANGUAGES
    }
    robust = select_stratified(
        robust_rows, quotas, seed=seed, namespace="smoke-robust"
    )
    clean_count = int(smoke["clean_per_language"])
    english = select_clean(
        english_rows,
        language="en",
        count=clean_count,
        seed=seed,
        namespace="smoke-clean-en",
    )
    chinese = select_clean(
        chinese_rows,
        language="zh",
        count=clean_count,
        seed=seed,
        namespace="smoke-clean-zh",
    )
    train = stable_order([*robust, *english, *chinese], seed, "smoke-train")
    if len(train) != int(smoke["expected_train_rows"]):
        raise ManifestError(
            f"smoke train rows {len(train)} != expected {smoke['expected_train_rows']}"
        )
    test = select_bench_smoke(
        bench_rows, seed, int(smoke["bench_per_language_origin"])
    )
    if len(test) != int(smoke["expected_bench_rows"]):
        raise ManifestError(
            f"smoke Bench rows {len(test)} != expected {smoke['expected_bench_rows']}"
        )
    return {"train": train, "test": test}


def normalized_transcript(text: Any) -> str:
    """Normalize transcript text only for overlap reporting."""
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    return "".join(char for char in normalized if char.isalnum())


def _audio_gate(path: Path, mode: str) -> str | None:
    """Return an audio validation error or ``None`` for the requested mode."""
    if mode == "ignore":
        return None
    if not path.exists() or not path.is_file():
        return f"audio does not exist: {path}"
    if mode == "decode":
        try:
            import soundfile  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ManifestError(
                "--audio-mode decode requires soundfile; install it with "
                "`pip install soundfile`"
            ) from exc
        try:
            info = soundfile.info(str(path))
        except Exception as exc:  # library-specific exception types vary
            return f"audio decode failed for {path}: {exc}"
        if info.frames <= 0 or info.samplerate <= 0:
            return f"audio metadata invalid: {path}"
    return None


def validate_rows(
    manifests: Mapping[str, Sequence[dict[str, Any]]],
    config: Mapping[str, Any],
    *,
    data_root: Path,
    audio_mode: str = "ignore",
    allow_absolute_audio: bool = False,
    check_counts: bool = True,
) -> dict[str, Any]:
    """Validate schema, paths, counts and hard cross-split leakage gates."""
    required = [str(item) for item in config_section(config, "manifest")["required_fields"]]
    selection = config_section(config, "selection")
    minimum = float(selection["min_duration_s"])
    maximum = float(selection["max_duration_s"])
    errors: list[str] = []
    counts: dict[str, int] = {}
    field_values: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    sample_owner: dict[str, str] = {}
    role_sample_ids: dict[str, set[str]] = defaultdict(set)
    transcript_sets: dict[str, set[str]] = defaultdict(set)

    for role, rows in manifests.items():
        counts[role] = len(rows)
        for index, row in enumerate(rows):
            location = f"{role}[{index}]"
            missing = [field for field in required if row.get(field) in (None, "")]
            if missing:
                errors.append(f"{location}: missing fields: {', '.join(missing)}")
                continue
            if row.get("language") not in LANGUAGES:
                errors.append(f"{location}: invalid language {row.get('language')!r}")
            if row.get("condition_group") not in CONDITION_GROUPS:
                errors.append(
                    f"{location}: invalid condition_group {row.get('condition_group')!r}"
                )
            if row.get("audio_origin") not in AUDIO_ORIGINS:
                errors.append(f"{location}: invalid audio_origin {row.get('audio_origin')!r}")
            try:
                duration = finite_float(row.get("duration_s"), "duration_s")
                if not minimum <= duration <= maximum:
                    errors.append(f"{location}: duration_s {duration} outside allowed range")
            except ManifestError as exc:
                errors.append(f"{location}: {exc}")
            revision = str(row.get("source_revision", ""))
            if len(revision) != PINNED_REVISION_LENGTH or any(
                char not in "0123456789abcdef" for char in revision.lower()
            ):
                errors.append(f"{location}: source_revision is not a pinned commit")

            audio_value = str(row.get("audio", ""))
            audio = Path(audio_value).expanduser()
            if audio.is_absolute() and not allow_absolute_audio:
                errors.append(f"{location}: audio path must be relative to data_root")
            resolved_audio = audio if audio.is_absolute() else data_root / audio
            audio_error = _audio_gate(resolved_audio, audio_mode)
            if audio_error:
                errors.append(f"{location}: {audio_error}")

            sample_id = str(row.get("sample_id", ""))
            if sample_id in role_sample_ids[role]:
                errors.append(f"duplicate sample_id within {role}: {sample_id!r}")
            role_sample_ids[role].add(sample_id)
            if role not in DERIVED_ROLES:
                previous = sample_owner.get(sample_id)
                if previous is not None:
                    errors.append(f"sample_id overlap: {sample_id!r} in {previous} and {role}")
                else:
                    sample_owner[sample_id] = role
            for field in (
                "source_utterance_id",
                "source_name",
                "benchmark_id",
                "audio",
                "audio_sha256",
            ):
                value = row.get(field)
                if value not in (None, ""):
                    field_values[field][role].add(str(value))
            transcript = normalized_transcript(row.get("answer"))
            if transcript:
                transcript_sets[role].add(transcript)

    leakage_roles = [role for role in ("train", "validation", "test") if role in manifests]
    leakage: dict[str, dict[str, int]] = {}
    for field, by_role in sorted(field_values.items()):
        for left_index, left in enumerate(leakage_roles):
            for right in leakage_roles[left_index + 1 :]:
                overlap = by_role[left] & by_role[right]
                key = f"{left}:{right}:{field}"
                leakage[key] = {
                    "count": len(overlap),
                    "examples": sorted(overlap)[:10],
                }
                if overlap:
                    errors.append(f"hard overlap {key}: {len(overlap)}")

    transcript_overlap: dict[str, int] = {}
    for left_index, left in enumerate(leakage_roles):
        for right in leakage_roles[left_index + 1 :]:
            transcript_overlap[f"{left}:{right}"] = len(
                transcript_sets[left] & transcript_sets[right]
            )

    if check_counts:
        expected = {
            "train": int(selection["expected_train_rows"]),
            "validation": int(selection["expected_validation_rows"]),
            "test": int(selection["expected_bench_rows"]),
        }
        curriculum = config_section(config, "curriculum")
        expected["curriculum"] = int(curriculum["target_rows"])
        expected["canary"] = int(config_section(config, "canary")["expected_rows"])
        for role, actual in counts.items():
            if role in expected and actual != expected[role]:
                errors.append(f"{role} rows {actual} != expected {expected[role]}")

    if "curriculum" in manifests:
        train_ids = {str(row["sample_id"]) for row in manifests.get("train", [])}
        maximum_error = float(config_section(config, "curriculum")["maximum_error_rate"])
        for index, row in enumerate(manifests["curriculum"]):
            sample_id = str(row.get("sample_id", ""))
            if train_ids and sample_id not in train_ids:
                errors.append(f"curriculum[{index}]: sample is not in train: {sample_id}")
            try:
                error_rate = finite_float(row.get("base_error_rate"), "base_error_rate")
                if error_rate < 0 or error_rate >= maximum_error:
                    errors.append(
                        f"curriculum[{index}]: base_error_rate {error_rate} outside "
                        f"[0, {maximum_error})"
                    )
            except ManifestError as exc:
                errors.append(f"curriculum[{index}]: {exc}")
            expected_metric = "wer" if row.get("language") == "en" else "cer"
            if row.get("base_metric") != expected_metric:
                errors.append(
                    f"curriculum[{index}]: base_metric must be {expected_metric}"
                )

    if "canary" in manifests:
        validation_ids = {
            str(row["sample_id"]) for row in manifests.get("validation", [])
        }
        if validation_ids:
            for index, row in enumerate(manifests["canary"]):
                sample_id = str(row.get("sample_id", ""))
                if sample_id not in validation_ids:
                    errors.append(f"canary[{index}]: sample is not in validation: {sample_id}")

    return {
        "hard_checks_pass": not errors,
        "errors": errors[:200],
        "error_count": len(errors),
        "row_counts": counts,
        "hard_overlap": leakage,
        "normalized_transcript_overlap_report_only": transcript_overlap,
        "audio_mode": audio_mode,
        "data_root": str(data_root),
    }


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return compact scenario/language/group statistics."""
    return {
        "rows": len(rows),
        "language": dict(sorted(Counter(str(row.get("language")) for row in rows).items())),
        "condition_group": dict(
            sorted(Counter(str(row.get("condition_group")) for row in rows).items())
        ),
        "scenario": dict(sorted(Counter(str(row.get("scenario")) for row in rows).items())),
        "audio_origin": dict(
            sorted(Counter(str(row.get("audio_origin")) for row in rows).items())
        ),
    }


def ensure_overwrite_allowed(paths: Iterable[Path], force: bool) -> None:
    """Avoid silently replacing a reproducible manifest run."""
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        preview = "\n".join(str(path) for path in existing)
        raise ManifestError(f"refusing to overwrite outputs; pass --force:\n{preview}")


def build_curriculum_rows(
    train_rows: Sequence[dict[str, Any]],
    scored_rows: Sequence[dict[str, Any]],
    *,
    target_rows: int,
    maximum_error_rate: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Select a deterministic language/scenario round-robin A2S curriculum."""
    train_by_id: dict[str, dict[str, Any]] = {}
    for row in train_rows:
        sample_id = str(row.get("sample_id", ""))
        if not sample_id or sample_id in train_by_id:
            raise ManifestError(f"train sample_id is empty or duplicated: {sample_id!r}")
        train_by_id[sample_id] = row

    scored_by_id: dict[str, dict[str, Any]] = {}
    for score in scored_rows:
        sample_id = str(score.get("sample_id", ""))
        if not sample_id or sample_id in scored_by_id:
            raise ManifestError(f"scored sample_id is empty or duplicated: {sample_id!r}")
        if sample_id not in train_by_id:
            raise ManifestError(f"scored sample is not in train: {sample_id}")
        train_row = train_by_id[sample_id]
        if train_row.get("condition_group") not in {"atomic", "compound"}:
            continue
        raw_error = score.get("base_error_rate")
        if raw_error is None:
            raw_error = score.get("error_rate")
        if raw_error is None:
            raw_error = score.get("wer") if train_row.get("language") == "en" else score.get("cer")
        error_rate = finite_float(raw_error, "base_error_rate")
        if error_rate < 0:
            raise ManifestError(f"base_error_rate must be non-negative: {sample_id}")
        if error_rate >= maximum_error_rate:
            continue
        expected_metric = "wer" if train_row.get("language") == "en" else "cer"
        metric = str(score.get("base_metric") or score.get("metric") or expected_metric).lower()
        if metric != expected_metric:
            raise ManifestError(
                f"{sample_id}: base_metric {metric!r} does not match {expected_metric!r}"
            )
        enriched = dict(train_row)
        enriched["base_prediction"] = str(
            score.get("base_prediction") or score.get("prediction") or ""
        )
        enriched["base_error_rate"] = error_rate
        enriched["base_metric"] = metric
        scored_by_id[sample_id] = enriched

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored_by_id.values():
        buckets[(str(row["language"]), str(row["scenario"]))].append(row)
    for key in buckets:
        buckets[key] = stable_order(
            buckets[key], seed, f"curriculum:{key[0]}:{key[1]}"
        )

    selected: list[dict[str, Any]] = []
    offsets = {key: 0 for key in buckets}
    keys = sorted(buckets)
    while len(selected) < target_rows:
        made_progress = False
        for key in keys:
            offset = offsets[key]
            if offset >= len(buckets[key]):
                continue
            selected.append(buckets[key][offset])
            offsets[key] += 1
            made_progress = True
            if len(selected) == target_rows:
                break
        if not made_progress:
            raise ManifestError(
                f"only {len(selected)} scored train rows have base_error_rate "
                f"< {maximum_error_rate}; need {target_rows}"
            )
    return selected


def threshold_slug(threshold: float) -> str:
    """Format 0.30 as the stable filename fragment 0_30."""
    return f"{threshold:.2f}".replace(".", "_")


def curriculum_views(
    rows: Sequence[dict[str, Any]], thresholds: Sequence[float]
) -> dict[float, list[dict[str, Any]]]:
    """Build cumulative threshold views while preserving selected order."""
    ordered_thresholds = sorted(float(value) for value in thresholds)
    views = {
        threshold: [
            row for row in rows if float(row["base_error_rate"]) < threshold
        ]
        for threshold in ordered_thresholds
    }
    previous: set[str] = set()
    for threshold in ordered_thresholds:
        current = {str(row["sample_id"]) for row in views[threshold]}
        if not previous <= current:
            raise ManifestError("curriculum threshold views are not cumulative")
        previous = current
    return views


def _feature_names(features: Any) -> list[str]:
    """Normalize datasets feature containers into sorted field names."""
    if features is None:
        return []
    if hasattr(features, "keys"):
        return sorted(str(key) for key in features.keys())
    if isinstance(features, Mapping):
        return sorted(str(key) for key in features)
    return []


def probe_dataset_source(
    source_name: str,
    source_config: Mapping[str, Any],
    *,
    loader: Callable[..., Any] | None = None,
    config_info_loader: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Inspect one pinned Hub dataset builder without loading any examples."""
    dataset_id = str(source_config.get("dataset_id") or "")
    revision = assert_pinned_revision(source_config.get("revision"), source_name)
    if not dataset_id:
        raise ConfigError(f"sources.{source_name}.dataset_id is required")
    if loader is None:
        try:
            from datasets import load_dataset_builder  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigError(
                "metadata probe requires `datasets`; install it with "
                "`pip install datasets`"
            ) from exc
        loader = load_dataset_builder
    kwargs: dict[str, Any] = {"revision": revision}
    if source_config.get("config"):
        kwargs["name"] = source_config["config"]
    builder = loader(dataset_id, **kwargs)
    info = builder.info
    split_info = info.splits or {}
    if not split_info:
        if config_info_loader is None:
            try:
                from datasets import get_dataset_config_info  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ConfigError(
                    "metadata split discovery requires `datasets`"
                ) from exc
            config_info_loader = get_dataset_config_info
        fallback_kwargs: dict[str, Any] = {"revision": revision}
        if source_config.get("config"):
            fallback_kwargs["config_name"] = source_config["config"]
        fallback_info = config_info_loader(dataset_id, **fallback_kwargs)
        split_info = fallback_info.splits or {}
        if not _feature_names(info.features):
            info = fallback_info
    split_rows = {
        str(name): int(getattr(value, "num_examples", 0) or 0)
        for name, value in split_info.items()
    }
    split_rows_known = bool(split_rows) and all(value > 0 for value in split_rows.values())
    features = _feature_names(info.features)
    errors: list[str] = []
    if source_config.get("expected_splits") is not None:
        expected = int(source_config["expected_splits"])
        if len(split_rows) != expected:
            errors.append(f"split count {len(split_rows)} != expected {expected}")
    required_fields = [
        str(item) for item in source_config.get("required_source_fields", [])
    ]
    missing_fields = sorted(set(required_fields) - set(features))
    if missing_fields:
        errors.append("missing source fields: " + ", ".join(missing_fields))
    if source_config.get("expected_rows") is not None:
        expected_rows = int(source_config["expected_rows"])
        if not split_rows_known:
            errors.append("split row counts are unavailable")
        else:
            actual_rows = sum(split_rows.values())
            if actual_rows != expected_rows:
                errors.append(f"rows {actual_rows} != expected {expected_rows}")
    download_size = int(getattr(info, "download_size", 0) or 0)
    dataset_size = int(getattr(info, "dataset_size", 0) or 0)
    if source_config.get("max_download_bytes") is not None and download_size:
        maximum_download = int(source_config["max_download_bytes"])
        if download_size > maximum_download:
            errors.append(
                f"download size {download_size} exceeds budget {maximum_download}"
            )
    if source_config.get("max_dataset_bytes") is not None and dataset_size:
        maximum_dataset = int(source_config["max_dataset_bytes"])
        if dataset_size > maximum_dataset:
            errors.append(f"dataset size {dataset_size} exceeds budget {maximum_dataset}")
    return {
        "source": source_name,
        "dataset_id": dataset_id,
        "requested_revision": revision,
        "audio_downloaded": False,
        "features": features,
        "split_count": len(split_rows),
        "split_rows": dict(sorted(split_rows.items())),
        "split_rows_known": split_rows_known,
        "download_size_bytes": download_size,
        "dataset_size_bytes": dataset_size,
        "quota_language_check": "deferred_to_materialized_candidate_build",
        "errors": errors,
        "passed": not errors,
    }


def add_robust_quota_capacity_check(
    report: dict[str, Any], config: Mapping[str, Any]
) -> None:
    """Check total per-split capacity before any audio is materialized."""
    sources = config_section(config, "sources")
    source = require_mapping(sources.get("robust"), "sources.robust")
    selection = config_section(config, "selection")
    train_plan = require_mapping(selection.get("robust_train"), "selection.robust_train")
    val_plan = require_mapping(
        selection.get("robust_validation"), "selection.robust_validation"
    )
    split_rows = {str(key): int(value) for key, value in report["split_rows"].items()}
    split_names = sorted(split_rows)
    train = plan_scenario_quotas(
        split_names,
        [str(item) for item in source["atomic_splits"]],
        int(train_plan["atomic_per_split"]),
        int(train_plan["compound_total"]),
        int(source["expected_compound_splits"]),
    )
    validation = plan_scenario_quotas(
        split_names,
        [str(item) for item in source["atomic_splits"]],
        int(val_plan["atomic_per_split"]),
        int(val_plan["compound_total"]),
        int(source["expected_compound_splits"]),
    )
    required = {name: train[name] + validation[name] for name in split_names}
    if not report.get("split_rows_known"):
        report["quota_required_rows_by_split"] = required
        report["quota_total_capacity_passed"] = "deferred_to_staging"
        report["quota_shortages"] = {}
        report["quota_language_check"] = "deferred_to_materialized_candidate_build"
        return
    shortages = {
        name: {"available": split_rows[name], "required": required[name]}
        for name in split_names
        if split_rows[name] < required[name]
    }
    report["quota_required_rows_by_split"] = required
    report["quota_total_capacity_passed"] = not shortages
    report["quota_shortages"] = shortages
    report["quota_language_check"] = "deferred_to_materialized_candidate_build"
    if shortages:
        report["errors"].append(
            f"{len(shortages)} splits do not have enough rows for train plus validation"
        )
        report["passed"] = False


def stream_hub_split(
    source_name: str,
    source_config: Mapping[str, Any],
    split: str,
    *,
    seed: int,
    shuffle_buffer_rows: int,
    loader: Callable[..., Any] | None = None,
) -> Iterable[Mapping[str, Any]]:
    """Open one pinned Hub split as a deterministic decode-free stream."""
    if loader is None:
        try:
            from datasets import Audio, load_dataset  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ConfigError(
                "audio staging requires `datasets`; install requirements-colab.txt"
            ) from exc
        loader = load_dataset
    else:
        Audio = None  # type: ignore[misc,assignment]
    kwargs: dict[str, Any] = {
        "split": split,
        "revision": assert_pinned_revision(source_config.get("revision"), source_name),
        "streaming": True,
    }
    if source_config.get("config"):
        kwargs["name"] = source_config["config"]
    stream = loader(str(source_config["dataset_id"]), **kwargs)
    if Audio is not None and hasattr(stream, "cast_column"):
        stream = stream.cast_column("audio", Audio(decode=False))
    if hasattr(stream, "shuffle") and shuffle_buffer_rows > 1:
        split_seed = int(stable_key(seed, f"stage:{source_name}", split)[:8], 16)
        stream = stream.shuffle(seed=split_seed, buffer_size=shuffle_buffer_rows)
    return stream


def stage_candidate_paths(
    staging: Mapping[str, Any], candidate_dir: Path
) -> dict[str, Path]:
    """Resolve every required staging artifact below one candidate directory."""
    outputs = require_mapping(staging.get("outputs"), "staging.outputs")
    required = (*STAGE_SOURCE_NAMES, "rejects", "report")
    missing = [name for name in required if not outputs.get(name)]
    if missing:
        raise ConfigError("missing staging outputs: " + ", ".join(missing))
    return {name: candidate_dir / str(outputs[name]) for name in required}


def robust_stage_targets(
    config: Mapping[str, Any], split_names: Sequence[str], mode: str
) -> dict[tuple[str, ...], int]:
    """Build robust quotas by role, scenario, and language for smoke/full mode."""
    source = require_mapping(config_section(config, "sources").get("robust"), "sources.robust")
    if mode == "smoke":
        per_language = int(config_section(config, "smoke")["robust_per_language_per_split"])
        return {
            ("smoke", scenario, language): per_language
            for scenario in sorted(split_names)
            for language in LANGUAGES
        }
    selection = config_section(config, "selection")
    train_plan = require_mapping(selection.get("robust_train"), "selection.robust_train")
    validation_plan = require_mapping(
        selection.get("robust_validation"), "selection.robust_validation"
    )
    atomic = [str(item) for item in source["atomic_splits"]]
    expected_compounds = int(source["expected_compound_splits"])
    train = plan_language_quotas(
        plan_scenario_quotas(
            split_names,
            atomic,
            int(train_plan["atomic_per_split"]),
            int(train_plan["compound_total"]),
            expected_compounds,
        )
    )
    validation = plan_language_quotas(
        plan_scenario_quotas(
            split_names,
            atomic,
            int(validation_plan["atomic_per_split"]),
            int(validation_plan["compound_total"]),
            expected_compounds,
        )
    )
    targets = {
        ("train", scenario, language): count
        for (scenario, language), count in train.items()
    }
    targets.update(
        {
            ("validation", scenario, language): count
            for (scenario, language), count in validation.items()
        }
    )
    return targets


def clean_stage_targets(
    config: Mapping[str, Any], source_name: str, mode: str
) -> dict[tuple[str, ...], int]:
    """Build official-split clean retention quotas for one language source."""
    language = "en" if source_name == "english_clean" else "zh"
    if mode == "smoke":
        return {("train", language): int(config_section(config, "smoke")["clean_per_language"])}
    selection = config_section(config, "selection")
    return {
        ("train", language): int(selection["clean_train_per_language"]),
        ("validation", language): int(selection["clean_validation_per_language"]),
    }


def bench_stage_targets(
    config: Mapping[str, Any], split_rows: Mapping[str, int], mode: str
) -> dict[tuple[str, ...], int]:
    """Build full split or smoke origin-language quotas for the Bench source."""
    if mode == "full":
        return {(str(split),): int(count) for split, count in split_rows.items()}
    per_stratum = int(config_section(config, "smoke")["bench_per_language_origin"])
    return {
        (origin, language): per_stratum
        for origin in ("real", "synthetic")
        for language in LANGUAGES
    }


def _source_stage_plan(
    config: Mapping[str, Any],
    source_name: str,
    source_config: Mapping[str, Any],
    split_rows: Mapping[str, int],
    mode: str,
) -> tuple[list[str], dict[tuple[str, ...], int], Callable[[str], set[tuple[str, ...]]]]:
    """Return source splits, quotas, and the buckets active for each split."""
    if source_name == "robust":
        splits = sorted(split_rows)
        targets = robust_stage_targets(config, splits, mode)

        def active(split: str) -> set[tuple[str, ...]]:
            """Select robust role/scenario/language buckets for one split."""
            return {key for key in targets if len(key) == 3 and key[1] == split}

        return splits, targets, active
    if source_name in {"english_clean", "chinese_clean"}:
        if mode == "smoke":
            splits = [str(source_config["train_split"])]
        else:
            splits = [
                str(source_config["train_split"]),
                str(source_config["validation_split"]),
            ]
        targets = clean_stage_targets(config, source_name, mode)
        language = "en" if source_name == "english_clean" else "zh"

        def active(split: str) -> set[tuple[str, ...]]:
            """Select the train or validation clean bucket for one split."""
            role = "validation" if mode == "full" and split == str(source_config["validation_split"]) else "train"
            return {(role, language)}

        return splits, targets, active
    splits = sorted(split_rows)
    targets = bench_stage_targets(config, split_rows, mode)

    def active(split: str) -> set[tuple[str, ...]]:
        """Select one full Bench split or both languages of its origin."""
        if mode == "full":
            return {(split,)}
        origin = "real" if split.startswith("real_") else "synthetic"
        return {(origin, language) for language in LANGUAGES}

    return splits, targets, active


def command_stage(args: argparse.Namespace) -> dict[str, Any]:
    """Stream pinned Hub audio into resumable, quota-bounded candidates."""
    config = load_config(Path(args.config).expanduser())
    staging = config_section(config, "staging")
    sources = config_section(config, "sources")
    seed, minimum, maximum = _selection_settings(config)
    candidate_dir = Path(args.candidate_dir or staging["candidate_dir"]).expanduser()
    data_root = Path(args.data_root or config_section(config, "project")["data_root"]).expanduser()
    paths = stage_candidate_paths(staging, candidate_dir)
    checkpoint_rows = int(staging["checkpoint_rows"])
    shuffle_buffer = int(
        staging["smoke_shuffle_buffer_rows"]
        if args.mode == "smoke"
        else staging["shuffle_buffer_rows"]
    )
    validation_percent = int(staging["robust_validation_percent"])
    rejects: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "mode": args.mode,
        "candidate_dir": str(candidate_dir),
        "data_root": str(data_root),
        "seed": seed,
        "sources": {},
        "passed": False,
    }

    for source_name in STAGE_SOURCE_NAMES:
        source = require_mapping(sources.get(source_name), f"sources.{source_name}")
        metadata = probe_dataset_source(source_name, source)
        if source_name == "robust" and metadata["passed"]:
            add_robust_quota_capacity_check(metadata, config)
        if not metadata["passed"]:
            report["sources"][source_name] = {"metadata": metadata, "passed": False}
            write_json(paths["report"], report)
            raise ManifestError(f"{source_name} metadata probe failed")

        candidate_path = paths[source_name]
        # Resume only from rows whose audio still exists and matches its hash;
        # damaged rows are removed so the stream can refill their quota.
        candidate_rows, dropped = load_valid_stage_rows(candidate_path, data_root)
        resumed_rows = len(candidate_rows)
        splits, targets, active_for_split = _source_stage_plan(
            config, source_name, source, metadata["split_rows"], args.mode
        )
        materialized = 0
        rejected_before = len(rejects)
        for split in splits:
            active = active_for_split(split)
            counts = Counter(stage_bucket(row, source_name, args.mode) for row in candidate_rows)
            if all(counts[key] >= int(targets[key]) for key in active):
                continue
            stream = stream_hub_split(
                source_name,
                source,
                split,
                seed=seed,
                shuffle_buffer_rows=shuffle_buffer,
            )
            materialized += stage_rows(
                stream,
                source_name=source_name,
                source_config=source,
                source_split=split,
                mode=args.mode,
                seed=seed,
                validation_percent=validation_percent,
                targets=targets,
                active_buckets=active,
                candidate_rows=candidate_rows,
                candidate_path=candidate_path,
                data_root=data_root,
                minimum_duration=minimum,
                maximum_duration=maximum,
                checkpoint_rows=checkpoint_rows,
                rejects=rejects,
            )
        counts = Counter(stage_bucket(row, source_name, args.mode) for row in candidate_rows)
        shortages = {
            "/".join(key): {"actual": counts[key], "required": int(required)}
            for key, required in sorted(targets.items())
            if counts[key] < int(required)
        }
        report["sources"][source_name] = {
            "metadata": metadata,
            "candidate_path": str(candidate_path),
            "requested_rows": sum(int(value) for value in targets.values()),
            "candidate_rows": len(candidate_rows),
            "resumed_rows": resumed_rows,
            "materialized_rows": materialized,
            "dropped_resume_rows": dropped,
            "rejected_rows": len(rejects) - rejected_before,
            "shortages": shortages,
            "manifest_sha256": file_sha256(candidate_path),
            "passed": not shortages,
        }

    write_jsonl(paths["rejects"], rejects)
    report["rejects"] = str(paths["rejects"])
    report["rejected_rows"] = len(rejects)
    report["passed"] = all(item["passed"] for item in report["sources"].values())
    write_json(paths["report"], report)
    if not report["passed"]:
        raise ManifestError("staging quota shortage; inspect stage_report.json")
    return report


def output_directory(config: Mapping[str, Any], override: str | None) -> Path:
    """Resolve a CLI output override or the configured manifest directory."""
    value = override or str(config_section(config, "project")["output_dir"])
    return Path(value).expanduser()


def command_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Inspect pinned dataset schemas and capacities without downloading audio."""
    config = load_config(Path(args.config).expanduser())
    sources = config_section(config, "sources")
    names = sorted(sources) if args.source == "all" else [args.source]
    reports = []
    for name in names:
        report = probe_dataset_source(
            name, require_mapping(sources.get(name), f"sources.{name}")
        )
        if name == "robust" and report["passed"]:
            add_robust_quota_capacity_check(report, config)
        reports.append(report)
    result = {
        "passed": all(report["passed"] for report in reports),
        "audio_downloaded": False,
        "sources": reports,
    }
    if args.output:
        write_json(Path(args.output).expanduser(), result)
    if not result["passed"]:
        raise ManifestError("metadata probe failed; inspect the probe report")
    return result


def _candidate_args(parser: argparse.ArgumentParser) -> None:
    """Attach the four required local candidate inputs to a subcommand."""
    parser.add_argument("--robust-candidates", required=True)
    parser.add_argument("--english-clean-candidates", required=True)
    parser.add_argument("--chinese-clean-candidates", required=True)
    parser.add_argument("--bench-candidates", required=True)


def command_smoke(args: argparse.Namespace) -> dict[str, Any]:
    """Build and validate the fixed 128-row training plus 4-row Bench smoke set."""
    config = load_config(Path(args.config).expanduser())
    robust, english, chinese, bench, rejects = load_all_candidate_inputs(config, args)
    selected = build_smoke_selection(config, robust, english, chinese, bench)
    output_dir = output_directory(config, args.output_dir)
    train_path = output_dir / "public_robust_smoke_128.jsonl"
    bench_path = output_dir / "vitw_bench_smoke_4.jsonl"
    report_path = output_dir / "public_robust_smoke_validation.json"
    rejects_path = output_dir / "public_robust_smoke_rejects.jsonl"
    ensure_overwrite_allowed(
        [train_path, bench_path, report_path, rejects_path], args.force
    )
    write_jsonl(train_path, selected["train"])
    write_jsonl(bench_path, selected["test"])
    write_jsonl(rejects_path, rejects)
    data_root = Path(args.data_root or config_section(config, "project")["data_root"])
    report = validate_rows(
        selected,
        config,
        data_root=data_root,
        audio_mode=args.audio_mode,
        allow_absolute_audio=args.allow_absolute_audio,
        check_counts=False,
    )
    report["expected_train_rows"] = int(config_section(config, "smoke")["expected_train_rows"])
    report["expected_test_rows"] = int(config_section(config, "smoke")["expected_bench_rows"])
    write_json(report_path, report)
    if not report["hard_checks_pass"]:
        raise ManifestError("smoke validation failed; inspect the validation report")
    return {
        "train": str(train_path),
        "test": str(bench_path),
        "rejects": len(rejects),
        "validation": str(report_path),
    }


def command_build(args: argparse.Namespace) -> dict[str, Any]:
    """Build canonical train, validation, canary, and Bench manifests."""
    config = load_config(Path(args.config).expanduser())
    robust, english, chinese, bench, rejects = load_all_candidate_inputs(config, args)
    selected = build_full_selection(config, robust, english, chinese, bench)
    output_dir = output_directory(config, args.output_dir)
    manifest_config = config_section(config, "manifest")
    outputs = require_mapping(manifest_config.get("outputs"), "manifest.outputs")
    paths = {name: output_dir / str(filename) for name, filename in outputs.items()}
    required_paths = [
        paths["train"],
        paths["validation"],
        paths["validation_canary"],
        paths["robust_test"],
        paths["stats"],
        paths["validation_report"],
        paths["rejects"],
        paths["sources"],
    ]
    ensure_overwrite_allowed(required_paths, args.force)
    write_jsonl(paths["train"], selected["train"])
    write_jsonl(paths["validation"], selected["validation"])
    write_jsonl(paths["validation_canary"], selected["canary"])
    write_jsonl(paths["robust_test"], selected["test"])
    write_jsonl(paths["rejects"], rejects)

    data_root = Path(args.data_root or config_section(config, "project")["data_root"])
    report = validate_rows(
        selected,
        config,
        data_root=data_root,
        audio_mode=args.audio_mode,
        allow_absolute_audio=args.allow_absolute_audio,
        check_counts=True,
    )
    write_json(paths["validation_report"], report)
    stats = {
        "seed": int(config_section(config, "project")["seed"]),
        "train": summarize_rows(selected["train"]),
        "validation": summarize_rows(selected["validation"]),
        "canary": summarize_rows(selected["canary"]),
        "test": summarize_rows(selected["test"]),
        "rejects": len(rejects),
        "manifest_sha256": {
            "train": file_sha256(paths["train"]),
            "validation": file_sha256(paths["validation"]),
            "canary": file_sha256(paths["validation_canary"]),
            "test": file_sha256(paths["robust_test"]),
        },
    }
    write_json(paths["stats"], stats)
    write_json(
        paths["sources"],
        {
            "seed": int(config_section(config, "project")["seed"]),
            "sources": config_section(config, "sources"),
            "config_sha256": file_sha256(Path(args.config).expanduser()),
        },
    )
    if not report["hard_checks_pass"]:
        raise ManifestError("full manifest validation failed; inspect validation report")
    return {
        "train": str(paths["train"]),
        "validation": str(paths["validation"]),
        "canary": str(paths["validation_canary"]),
        "test": str(paths["robust_test"]),
        "stats": str(paths["stats"]),
        "rejects": len(rejects),
    }


def command_validate(args: argparse.Namespace) -> dict[str, Any]:
    """Run schema, audio, count, and leakage gates on supplied manifests."""
    config = load_config(Path(args.config).expanduser())
    manifests: dict[str, list[dict[str, Any]]] = {}
    for role in ("train", "validation", "canary", "test", "curriculum"):
        value = getattr(args, role)
        if value:
            manifests[role] = read_jsonl(Path(value).expanduser())
    if not manifests:
        raise ManifestError("validate requires at least one manifest path")
    data_root = Path(args.data_root or config_section(config, "project")["data_root"])
    report = validate_rows(
        manifests,
        config,
        data_root=data_root,
        audio_mode=args.audio_mode,
        allow_absolute_audio=args.allow_absolute_audio,
        check_counts=True,
    )
    if args.output:
        write_json(Path(args.output).expanduser(), report)
    if not report["hard_checks_pass"]:
        raise ManifestError("manifest validation failed")
    return report


def command_curriculum(args: argparse.Namespace) -> dict[str, Any]:
    """Join base scores and emit deterministic 30k cumulative curriculum views."""
    config = load_config(Path(args.config).expanduser())
    settings = config_section(config, "curriculum")
    seed = int(config_section(config, "project")["seed"])
    train_rows = read_jsonl(Path(args.train).expanduser())
    scored_rows = read_jsonl(Path(args.scored).expanduser())
    target = int(settings["target_rows"])
    maximum = float(settings["maximum_error_rate"])
    rows = build_curriculum_rows(
        train_rows,
        scored_rows,
        target_rows=target,
        maximum_error_rate=maximum,
        seed=seed,
    )
    thresholds = [float(value) for value in settings["cumulative_thresholds"]]
    views = curriculum_views(rows, thresholds)
    output = Path(args.output).expanduser()
    paths = [output, *[
        output.with_name(f"{output.stem}.lt_{threshold_slug(value)}{output.suffix}")
        for value in thresholds
    ]]
    if args.report:
        paths.append(Path(args.report).expanduser())
    ensure_overwrite_allowed(paths, args.force)
    write_jsonl(output, rows)
    view_paths: dict[str, str] = {}
    previous_ids: set[str] = set()
    for threshold in sorted(views):
        view_path = output.with_name(
            f"{output.stem}.lt_{threshold_slug(threshold)}{output.suffix}"
        )
        write_jsonl(view_path, views[threshold])
        current_ids = {str(row["sample_id"]) for row in views[threshold]}
        if not previous_ids <= current_ids:
            raise ManifestError("written curriculum views are not cumulative")
        previous_ids = current_ids
        view_paths[f"lt_{threshold:.2f}"] = str(view_path)
    result = {
        "output": str(output),
        "rows": len(rows),
        "maximum_error_rate": maximum,
        "manifest_sha256": file_sha256(output),
        "views": {
            key: {"path": path, "rows": len(views[float(key[3:])])}
            for key, path in view_paths.items()
        },
        "strata": {
            f"{language}:{scenario}": count
            for (language, scenario), count in sorted(
                Counter((str(row["language"]), str(row["scenario"])) for row in rows).items()
            )
        },
    }
    if args.report:
        write_json(Path(args.report).expanduser(), result)
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the probe/stage/smoke/build/validate/curriculum CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    probe = subparsers.add_parser("probe", help="inspect pinned Hub metadata only")
    probe.add_argument("--config", required=True)
    probe.add_argument(
        "--source",
        default="all",
        choices=("all", "robust", "robust_test", "english_clean", "chinese_clean"),
    )
    probe.add_argument("--output", default="")
    probe.set_defaults(handler=command_probe)

    stage = subparsers.add_parser(
        "stage", help="stream pinned Hub audio into resumable local candidates"
    )
    stage.add_argument("--config", required=True)
    stage.add_argument("--mode", required=True, choices=("smoke", "full"))
    stage.add_argument("--candidate-dir", default="")
    stage.add_argument("--data-root", default="")
    stage.set_defaults(handler=command_stage)

    smoke = subparsers.add_parser("smoke", help="build the 128-row local fixture")
    smoke.add_argument("--config", required=True)
    _candidate_args(smoke)
    smoke.add_argument("--output-dir", default="")
    smoke.add_argument("--data-root", default="")
    smoke.add_argument("--audio-mode", choices=("ignore", "exists", "decode"), default="ignore")
    smoke.add_argument("--allow-absolute-audio", action="store_true")
    smoke.add_argument("--force", action="store_true")
    smoke.set_defaults(handler=command_smoke)

    build = subparsers.add_parser("build", help="build fixed 200k/10k/5k manifests")
    build.add_argument("--config", required=True)
    _candidate_args(build)
    build.add_argument("--output-dir", default="")
    build.add_argument("--data-root", default="")
    build.add_argument("--audio-mode", choices=("exists", "decode"), default="exists")
    build.add_argument("--allow-absolute-audio", action="store_true")
    build.add_argument("--force", action="store_true")
    build.set_defaults(handler=command_build)

    validate = subparsers.add_parser("validate", help="validate canonical manifests")
    validate.add_argument("--config", required=True)
    validate.add_argument("--train", default="")
    validate.add_argument("--validation", default="")
    validate.add_argument("--canary", default="")
    validate.add_argument("--test", default="")
    validate.add_argument("--curriculum", default="")
    validate.add_argument("--data-root", default="")
    validate.add_argument("--audio-mode", choices=("exists", "decode"), default="exists")
    validate.add_argument("--allow-absolute-audio", action="store_true")
    validate.add_argument("--output", default="")
    validate.set_defaults(handler=command_validate)

    curriculum = subparsers.add_parser(
        "curriculum", help="build deterministic 30k base-error views"
    )
    curriculum.add_argument("--config", required=True)
    curriculum.add_argument("--train", required=True)
    curriculum.add_argument("--scored", required=True)
    curriculum.add_argument("--output", required=True)
    curriculum.add_argument("--report", default="")
    curriculum.add_argument("--force", action="store_true")
    curriculum.set_defaults(handler=command_curriculum)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one data command and convert expected failures to exit code 2."""
    args = build_parser().parse_args(argv)
    try:
        result = args.handler(args)
    except (ConfigError, ManifestError, KeyError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
