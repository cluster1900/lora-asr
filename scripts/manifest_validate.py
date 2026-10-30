#!/usr/bin/env python3
"""Check a Phase 1 manifest directory against the data contract.

Exit 0 is the only status that may be handed to inference or training.
Missing --check-audio exits 2 so an unread audio tree cannot pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.manifest_contract import (  # noqa: E402
    MANIFEST_ROW_SCHEMA,
    MANIFEST_SET_SCHEMA,
    OFFICIAL_SPLIT_TO_MANIFEST,
    PROCESSOR_VERSION,
    REJECTED_SCHEMA,
    SCENARIOS,
    SPLITS,
    STATUS_CODE,
    frame_bounds,
    load_data_config,
    safe_relative_audio,
    sample_id_for,
    sha256_file,
    text_problem,
)

ROW_FIELDS = {
    "schema_version",
    "sample_id",
    "source_utterance_id",
    "audio",
    "text",
    "language",
    "scenario",
    "split",
    "augmentation",
    "speaker_id",
    "official_split",
    "source_package",
}


def inspect_manifest(manifest_dir: Path, config: dict[str, Any], check_audio: bool) -> dict[str, Any]:
    """Recompute status, counts, shortages, and quality from the files on disk."""
    problems: list[str] = []
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        path = manifest_dir / f"{split}.jsonl"
        if not path.is_file():
            problems.append(f"missing_{split}_jsonl")
            continue
        rows.extend(_read_rows(path, split, problems))
    rejected = _read_rejected(manifest_dir / "rejected.jsonl", problems)
    _check_identity(rows, problems)
    durations = _check_audio(manifest_dir, rows, config, check_audio, problems) if check_audio else []
    quality = _quality(rows, problems)
    counts = _counts(rows, len(rejected))
    shortages = _shortages(counts, config)
    status = _status(rows, problems, shortages)
    return {
        "schema_version": MANIFEST_SET_SCHEMA,
        "status": status,
        "seed": config["seed"],
        "source_version": config["source_version"],
        "audited": config["audited"],
        "audio_processor_version": config["audio_processor_version"],
        "config_sha256": config["config_sha256"],
        "counts": counts,
        "quality": quality,
        "duration_seconds": _duration_summary(durations),
        "shortages": shortages,
        "problems": problems,
        "split_audit": _split_audit(rows, rejected),
    }


def validate_manifest_dir(
    manifest_dir: Path,
    config: dict[str, Any],
    check_audio: bool,
) -> tuple[dict[str, Any], int]:
    """Compare the stored summary with a fresh inspection and return an exit code."""
    if not check_audio:
        report = inspect_manifest(manifest_dir, config, False)
        report["status"] = "failed"
        report["problems"] = [*report["problems"], "check_audio_required"]
        return report, 2
    report = inspect_manifest(manifest_dir, config, True)
    stored_path = manifest_dir / "manifest.json"
    if not stored_path.is_file():
        report["status"] = "failed"
        report["problems"] = [*report["problems"], "missing_manifest_json"]
        return report, 1
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    if not isinstance(stored, dict):
        report["status"] = "failed"
        report["problems"] = [*report["problems"], "manifest_json_not_object"]
        return report, 1
    for key in ("status", "counts", "quality", "shortages", "duration_seconds", "seed", "config_sha256", "split_audit"):
        if stored.get(key) != report[key]:
            report["status"] = "failed"
            report["problems"] = [*report["problems"], f"manifest_mismatch:{key}"]
    return report, STATUS_CODE[report["status"]]


def _read_rows(path: Path, split: str, problems: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"invalid_json:{path.name}:{line_number}")
            continue
        if not isinstance(row, dict):
            problems.append(f"row_not_object:{path.name}:{line_number}")
            continue
        if row.get("split") != split:
            problems.append(f"split_mismatch:{row.get('sample_id')}")
        rows.append(row)
    return rows


def _read_rejected(path: Path, problems: list[str]) -> list[dict[str, Any]]:
    if not path.is_file():
        problems.append("missing_rejected_jsonl")
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            problems.append(f"invalid_rejected_json:{line_number}")
            continue
        if not isinstance(row, dict) or row.get("schema_version") != REJECTED_SCHEMA or not row.get("reason"):
            problems.append(f"bad_rejected_row:{line_number}")
            continue
        rows.append(row)
    return rows


def _check_identity(rows: list[dict[str, Any]], problems: list[str]) -> None:
    seen_ids: set[str] = set()
    seen_audio: set[str] = set()
    source_splits: dict[str, set[str]] = defaultdict(set)
    texts: dict[str, set[str]] = {split: set() for split in SPLITS}
    for row in rows:
        sample_id = row.get("sample_id")
        source_id = row.get("source_utterance_id")
        scenario = row.get("scenario")
        split = row.get("split")
        if row.get("schema_version") != MANIFEST_ROW_SCHEMA or set(row) - ROW_FIELDS:
            problems.append(f"bad_row_schema:{sample_id}")
        if not isinstance(sample_id, str) or sample_id in seen_ids:
            problems.append(f"duplicate_or_bad_sample_id:{sample_id}")
        elif isinstance(sample_id, str):
            seen_ids.add(sample_id)
        if scenario not in SCENARIOS or split not in SPLITS:
            problems.append(f"bad_enum:{sample_id}")
        elif isinstance(source_id, str) and sample_id != sample_id_for(source_id, scenario):
            problems.append(f"sample_id_mismatch:{sample_id}")
        if row.get("language") != "zh":
            problems.append(f"language:{sample_id}")
        problem = text_problem(row.get("text"))
        if problem:
            problems.append(f"{problem}:{sample_id}")
        audio = row.get("audio")
        if not safe_relative_audio(audio):
            problems.append(f"unsafe_audio_path:{sample_id}")
        elif audio in seen_audio:
            problems.append(f"duplicate_audio_path:{audio}")
        elif isinstance(audio, str):
            seen_audio.add(audio)
        _check_augmentation(row, problems)
        if isinstance(source_id, str) and isinstance(split, str):
            source_splits[source_id].add(split)
            if isinstance(row.get("text"), str) and split in texts:
                texts[split].add(row["text"])
        _check_official_row(row, problems)
    for source_id, splits in sorted(source_splits.items()):
        if len(splits) > 1:
            problems.append(f"source_leakage:{source_id}")
    _check_speaker_identity(rows, problems)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = texts[left] & texts[right]
        if overlap:
            problems.append(f"text_leakage:{left}:{right}")


def _check_official_row(row: dict[str, Any], problems: list[str]) -> None:
    sample_id = row.get("sample_id")
    has_speaker = "speaker_id" in row
    has_official = "official_split" in row
    if has_speaker != has_official:
        problems.append(f"incomplete_official_identity:{sample_id}")
        return
    if not has_official:
        if "source_package" in row:
            problems.append(f"source_package_without_official_split:{sample_id}")
        return
    official_split = row.get("official_split")
    mapped = OFFICIAL_SPLIT_TO_MANIFEST.get(official_split) if isinstance(official_split, str) else None
    if mapped != row.get("split"):
        problems.append(f"official_split_mismatch:{sample_id}")
    speaker_id = row.get("speaker_id")
    if not isinstance(speaker_id, str) or speaker_id == "":
        problems.append(f"bad_speaker_id:{sample_id}")


def _check_speaker_identity(rows: list[dict[str, Any]], problems: list[str]) -> None:
    official_rows = [row for row in rows if "official_split" in row or "speaker_id" in row]
    if official_rows and len(official_rows) != len(rows):
        problems.append("mixed_split_mode")
    identities: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    speakers: dict[str, set[str]] = {split: set() for split in SPLITS}
    for row in rows:
        source_id = row.get("source_utterance_id")
        split = row.get("split")
        if not isinstance(source_id, str):
            continue
        speaker = row.get("speaker_id") if isinstance(row.get("speaker_id"), str) else ""
        official = row.get("official_split") if isinstance(row.get("official_split"), str) else ""
        package = row.get("source_package") if isinstance(row.get("source_package"), str) else ""
        identities[source_id].add((speaker, official, package))
        if speaker and isinstance(split, str) and split in speakers:
            speakers[split].add(speaker)
    for source_id, seen in sorted(identities.items()):
        if len(seen) > 1:
            problems.append(f"speaker_identity_mismatch:{source_id}")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        for speaker in sorted(speakers[left] & speakers[right]):
            problems.append(f"speaker_leakage:{speaker}")


def _split_audit(rows: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> dict[str, Any]:
    speakers: dict[str, set[str]] = {split: set() for split in SPLITS}
    packages: set[str] = set()
    official = False
    for row in rows:
        if "official_split" in row or "speaker_id" in row:
            official = True
        speaker = row.get("speaker_id")
        split = row.get("split")
        if isinstance(speaker, str) and isinstance(split, str) and split in speakers:
            speakers[split].add(speaker)
        package = row.get("source_package")
        if isinstance(package, str):
            packages.add(package)
    reasons: dict[str, int] = {}
    for row in rejected:
        reason = row.get("reason")
        if isinstance(reason, str):
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "mode": "official" if official else "hash",
        "speakers": {split: sorted(speakers[split]) for split in SPLITS},
        "speaker_intersection": {
            "train_val": sorted(speakers["train"] & speakers["val"]),
            "train_test": sorted(speakers["train"] & speakers["test"]),
            "val_test": sorted(speakers["val"] & speakers["test"]),
        },
        "source_packages": sorted(packages),
        "rejected_reasons": dict(sorted(reasons.items())),
    }


def _check_augmentation(row: dict[str, Any], problems: list[str]) -> None:
    scenario = row.get("scenario")
    augmentation = row.get("augmentation", None)
    if scenario == "clean":
        if "augmentation" in row:
            problems.append(f"clean_has_augmentation:{row.get('sample_id')}")
        return
    if not isinstance(augmentation, dict):
        problems.append(f"missing_augmentation:{row.get('sample_id')}")
        return
    if augmentation.get("processor_version") != PROCESSOR_VERSION:
        problems.append(f"processor_version:{row.get('sample_id')}")
    if "audio" in augmentation or "text" in augmentation:
        problems.append(f"augmentation_overwrites_row:{row.get('sample_id')}")


def _check_audio(
    manifest_dir: Path,
    rows: list[dict[str, Any]],
    config: dict[str, Any],
    check_audio: bool,
    problems: list[str],
) -> list[int]:
    del check_audio
    minimum, maximum = frame_bounds(config)
    durations: list[int] = []
    hash_sources: dict[str, set[str]] = defaultdict(set)
    hashes_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    for row in rows:
        audio = row.get("audio")
        if not isinstance(audio, str) or not safe_relative_audio(audio):
            continue
        path = manifest_dir / audio
        if not path.is_file():
            problems.append(f"missing_audio:{row.get('sample_id')}")
            continue
        try:
            info = sf.info(path)
        except Exception:
            problems.append(f"audio_unreadable:{row.get('sample_id')}")
            continue
        if info.samplerate != 16000 or info.channels != 1 or info.subtype != "PCM_16":
            problems.append(f"audio_format:{row.get('sample_id')}")
            continue
        if info.frames < minimum or info.frames > maximum:
            problems.append(f"duration_out_of_range:{row.get('sample_id')}")
            continue
        durations.append(int(info.frames))
        digest = sha256_file(path)
        source_id = str(row.get("source_utterance_id"))
        hash_sources[digest].add(source_id)
        split = str(row.get("split"))
        if split in hashes_by_split:
            hashes_by_split[split].add(digest)
    for digest, sources in sorted(hash_sources.items()):
        if len(sources) > 1:
            problems.append(f"audio_hash_conflict:{digest[:12]}")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if hashes_by_split[left] & hashes_by_split[right]:
            problems.append(f"audio_leakage:{left}:{right}")
    return durations


def _quality(rows: list[dict[str, Any]], problems: list[str]) -> dict[str, int]:
    joined = "\n".join(problems)
    return {
        "empty_text": joined.count("empty_text:"),
        "bad_audio": sum(joined.count(token) for token in ("missing_audio:", "audio_unreadable:", "audio_format:")),
        "duplicate_ids": joined.count("duplicate_or_bad_sample_id:"),
        "audio_hash_conflicts": joined.count("audio_hash_conflict:"),
        "source_leakage": joined.count("source_leakage:"),
        "speaker_leakage": joined.count("speaker_leakage:"),
        "accepted_rows": len(rows),
    }


def _counts(rows: list[dict[str, Any]], rejected: int) -> dict[str, Any]:
    by_split: dict[str, dict[str, int]] = {split: {scenario: 0 for scenario in SCENARIOS} for split in SPLITS}
    sources: dict[str, set[str]] = {split: set() for split in SPLITS}
    for row in rows:
        split = str(row.get("split"))
        scenario = str(row.get("scenario"))
        if split in by_split and scenario in by_split[split]:
            by_split[split][scenario] += 1
            if isinstance(row.get("source_utterance_id"), str):
                sources[split].add(row["source_utterance_id"])
    return {
        "rows": len(rows),
        "rejected": rejected,
        "sources": {split: len(sources[split]) for split in SPLITS},
        "by_split_scenario": by_split,
    }


def _shortages(counts: dict[str, Any], config: dict[str, Any]) -> list[dict[str, Any]]:
    shortages = []
    table = counts["by_split_scenario"]
    minimum = int(config["min_val_rows_per_scenario"])
    for scenario in SCENARIOS:
        val_rows = int(table["val"][scenario])
        if val_rows < minimum:
            shortages.append({"split": "val", "scenario": scenario, "rows": val_rows, "minimum": minimum})
        train_rows = int(table["train"][scenario])
        if train_rows < 1:
            shortages.append({"split": "train", "scenario": scenario, "rows": train_rows, "minimum": 1})
    if not config["audited"]:
        shortages.append({"reason": "source_not_audited", "source_version": config["source_version"]})
    return shortages


def _status(rows: list[dict[str, Any]], problems: list[str], shortages: list[dict[str, Any]]) -> str:
    if problems:
        return "failed"
    if not rows:
        return "data_failed"
    if shortages:
        return "insufficient_data"
    return "passed"


def _duration_summary(frames: list[int]) -> dict[str, float | None]:
    if not frames:
        return {"min": None, "p50": None, "max": None}
    ordered = sorted(frames)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        median_frames = ordered[middle]
    else:
        median_frames = (ordered[middle - 1] + ordered[middle]) / 2
    return {
        "min": ordered[0] / 16000,
        "p50": median_frames / 16000,
        "max": ordered[-1] / 16000,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验 Phase 1 的 train/val/test manifest")
    parser.add_argument("--manifest", required=True, type=Path, help="含三个 JSONL 的目录")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--check-audio", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_data_config(args.config)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    report, code = validate_manifest_dir(args.manifest, config, args.check_audio)
    print(json.dumps({"status": report["status"], "exit_code": code}, ensure_ascii=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
