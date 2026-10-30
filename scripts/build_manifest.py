#!/usr/bin/env python3
"""Build a five-scenario Phase 1 manifest from a source JSONL.

The same config, source bytes, and seed reproduce the JSONL rows and WAV
bytes. sha256.json records a clock time that is not part of that identity.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.jsonl_io import read_jsonl, write_jsonl  # noqa: E402
from scripts.manifest_audio import load_mono_16k, render_scenarios
from scripts.manifest_contract import (
    MANIFEST_ROW_SCHEMA,
    PROCESSOR_VERSION,
    REJECTED_SCHEMA,
    SCENARIOS,
    SHA_SCHEMA,
    SOURCE_ID_RE,
    SPLITS,
    STATUS_CODE,
    OFFICIAL_SPLIT_TO_MANIFEST,
    assign_split,
    frame_bounds,
    load_data_config,
    sample_id_for,
    sha256_file,
    split_mode,
    text_problem,
    safe_relative_audio,
)
from scripts.manifest_validate import inspect_manifest

HASH_FIELDS = {"source_utterance_id", "audio", "text", "language"}
OFFICIAL_FIELDS = HASH_FIELDS | {"speaker_id", "official_split", "source_package"}
PRODUCTS = (
    "train.jsonl",
    "val.jsonl",
    "test.jsonl",
    "rejected.jsonl",
    "manifest.json",
    "sha256.json",
)
KNOWN_AUDIO_REASONS = {"silent_audio", "duration_out_of_range"}


def build_manifest(
    config: dict[str, Any],
    sources_path: Path,
    output_dir: Path,
    git_sha: str | None,
) -> dict[str, Any]:
    """Materialize WAV files and the manifest directory. Return the summary."""
    raw_rows = read_jsonl(sources_path)
    mode = split_mode(raw_rows)
    _reset_output(output_dir)
    accepted, rejected = _load_sources(raw_rows, sources_path.parent, config, mode)
    if mode == "official":
        accepted = _drop_speaker_cross_split(accepted, rejected, config)
    accepted = _drop_cross_split_text(accepted, rejected, config)
    rendered = _render_all(accepted, output_dir, config, rejected)
    rendered = _drop_duplicate_audio(rendered, output_dir, rejected, config)
    _write_splits(output_dir, rendered)
    write_jsonl(output_dir / "rejected.jsonl", rejected)
    report = inspect_manifest(output_dir, config, True)
    report["generator_git_sha"] = git_sha
    report["source_sha256"] = sha256_file(sources_path)
    (output_dir / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_sha_index(output_dir)
    confirmed, _code = _confirm(output_dir, config)
    return confirmed


def _confirm(output_dir: Path, config: dict[str, Any]) -> tuple[dict[str, Any], int]:
    from scripts.manifest_validate import validate_manifest_dir

    return validate_manifest_dir(output_dir, config, True)


def _load_sources(
    raw_rows: list[dict[str, Any]],
    root: Path,
    config: dict[str, Any],
    mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(raw_rows, start=1):
        item, reason = _classify_source(row, root, seen, mode)
        if item is None:
            rejected.append(_rejected(config, f"source-line-{index}", row.get("source_utterance_id"), reason or "bad_source", row.get("audio")))
            continue
        seen.add(item["source_id"])
        if mode == "hash":
            item["split"] = assign_split(
                int(config["seed"]),
                item["source_id"],
                float(config["splits"]["val"]),
                float(config["splits"]["test"]),
            )
        accepted.append(item)
    accepted.sort(key=lambda item: item["source_id"])
    return accepted, rejected


def _classify_source(
    row: dict[str, Any],
    root: Path,
    seen: set[str],
    mode: str,
) -> tuple[dict[str, Any] | None, str | None]:
    allowed = OFFICIAL_FIELDS if mode == "official" else HASH_FIELDS
    if set(row) - allowed:
        return None, "unknown_field"
    source_id = row.get("source_utterance_id")
    if not isinstance(source_id, str) or SOURCE_ID_RE.fullmatch(source_id) is None:
        return None, "bad_source_id"
    if source_id in seen:
        return None, "duplicate_source"
    if row.get("language", "zh") != "zh":
        return None, "language"
    problem = text_problem(row.get("text"))
    if problem:
        return None, problem
    audio = row.get("audio")
    if not safe_relative_audio(audio):
        return None, "unsafe_audio_path"
    audio_path = root / str(audio)
    if not audio_path.is_file():
        return None, "missing_audio"
    item = {
        "source_id": source_id,
        "text": row["text"],
        "audio": str(audio),
        "audio_path": audio_path,
    }
    if mode == "official":
        identity, reason = _official_identity(row)
        if identity is None:
            return None, reason
        item.update(identity)
    return item, None


def _official_identity(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    speaker_id = row.get("speaker_id")
    official_split = row.get("official_split")
    if "speaker_id" not in row or "official_split" not in row:
        return None, "incomplete_official_identity"
    if not isinstance(speaker_id, str) or SOURCE_ID_RE.fullmatch(speaker_id) is None:
        return None, "bad_speaker_id"
    manifest_split = OFFICIAL_SPLIT_TO_MANIFEST.get(official_split) if isinstance(official_split, str) else None
    if manifest_split is None:
        return None, "bad_official_split"
    identity: dict[str, Any] = {
        "speaker_id": speaker_id,
        "official_split": official_split,
        "split": manifest_split,
    }
    if "source_package" in row:
        package = row.get("source_package")
        if not isinstance(package, str) or SOURCE_ID_RE.fullmatch(package) is None:
            return None, "bad_source_package"
        identity["source_package"] = package
    return identity, None


def _drop_speaker_cross_split(
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    speaker_split: dict[str, str] = {}
    for item in accepted:
        previous = speaker_split.get(item["speaker_id"])
        if previous is not None and previous != item["split"]:
            rejected.append(_rejected(config, item["source_id"], item["source_id"], "speaker_cross_split", item["audio"]))
            continue
        speaker_split[item["speaker_id"]] = item["split"]
        kept.append(item)
    return kept


def _drop_cross_split_text(
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    text_split: dict[str, str] = {}
    for item in accepted:
        previous = text_split.get(item["text"])
        if previous is not None and previous != item["split"]:
            rejected.append(_rejected(config, item["source_id"], item["source_id"], "text_cross_split", item["audio"]))
            continue
        text_split[item["text"]] = item["split"]
        kept.append(item)
    return kept


def _render_all(
    accepted: list[dict[str, Any]],
    output_dir: Path,
    config: dict[str, Any],
    rejected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rendered = []
    minimum, maximum = frame_bounds(config)
    for item in accepted:
        try:
            clean = load_mono_16k(str(item["audio_path"]), _read_audio)
            if clean.size < minimum or clean.size > maximum:
                raise ValueError("duration_out_of_range")
            waves = render_scenarios(clean, int(config["seed"]), item["source_id"], config["augmentation"])
            rows = [_write_wave(output_dir, item, scenario, waves[scenario], config) for scenario in SCENARIOS]
        except Exception as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            reason = str(exc) if str(exc) in KNOWN_AUDIO_REASONS else "audio_unreadable"
            _delete_source_wavs(output_dir, item["source_id"])
            rejected.append(_rejected(config, item["source_id"], item["source_id"], reason, item["audio"]))
            continue
        rendered.append({**item, "rows": rows, "clean_sha256": sha256_file(output_dir / rows[0]["audio"])})
    return rendered


def _write_wave(
    output_dir: Path,
    item: dict[str, Any],
    scenario: str,
    audio: np.ndarray,
    config: dict[str, Any],
) -> dict[str, Any]:
    sample_id = sample_id_for(item["source_id"], scenario)
    relative = f"wavs/{sample_id}.wav"
    destination = output_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, np.clip(audio, -1.0, 1.0), 16000, format="WAV", subtype="PCM_16")
    row: dict[str, Any] = {
        "schema_version": MANIFEST_ROW_SCHEMA,
        "sample_id": sample_id,
        "source_utterance_id": item["source_id"],
        "audio": relative,
        "text": item["text"],
        "language": "zh",
        "scenario": scenario,
        "split": item["split"],
    }
    if "speaker_id" in item:
        row["speaker_id"] = item["speaker_id"]
        row["official_split"] = item["official_split"]
    if "source_package" in item:
        row["source_package"] = item["source_package"]
    if scenario != "clean":
        row["augmentation"] = _augmentation_record(scenario, item["source_id"], config)
    return row


def _augmentation_record(scenario: str, source_id: str, config: dict[str, Any]) -> dict[str, Any]:
    settings = config["augmentation"][scenario]
    record: dict[str, Any] = {"type": scenario, "processor_version": PROCESSOR_VERSION, **settings}
    if scenario == "noise":
        from scripts.manifest_contract import scenario_seed

        record["seed"] = scenario_seed(int(config["seed"]), source_id, scenario)
    return record


def _drop_duplicate_audio(
    rendered: list[dict[str, Any]],
    output_dir: Path,
    rejected: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in rendered:
        grouped.setdefault(item["clean_sha256"], []).append(item)
    kept = []
    for group in grouped.values():
        group.sort(key=lambda item: item["source_id"])
        kept.append(group[0])
        for item in group[1:]:
            _delete_source_wavs(output_dir, item["source_id"])
            rejected.append(
                _rejected(config, item["source_id"], item["source_id"], "audio_hash_conflict", item["audio"])
            )
    return kept


def _write_splits(output_dir: Path, rendered: list[dict[str, Any]]) -> None:
    grouped = {split: [] for split in SPLITS}
    for item in rendered:
        grouped[item["split"]].extend(item["rows"])
    for split, rows in grouped.items():
        rows.sort(key=lambda row: row["sample_id"])
        write_jsonl(output_dir / f"{split}.jsonl", rows)


def _rejected(
    config: dict[str, Any],
    sample_id: str,
    source_id: object,
    reason: str,
    audio: object,
) -> dict[str, Any]:
    return {
        "schema_version": REJECTED_SCHEMA,
        "sample_id": sample_id,
        "source_utterance_id": source_id if isinstance(source_id, str) else None,
        "reason": reason,
        "path": audio if isinstance(audio, str) else "",
        "processor_version": config["audio_processor_version"],
    }


def _delete_source_wavs(output_dir: Path, source_id: str) -> None:
    wav_dir = output_dir / "wavs"
    for scenario in SCENARIOS:
        path = wav_dir / f"{sample_id_for(source_id, scenario)}.wav"
        if path.exists():
            path.unlink()


def _reset_output(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in PRODUCTS:
        path = output_dir / name
        if path.exists():
            path.unlink()
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    for path in wav_dir.glob("*.wav"):
        path.unlink()


def _write_sha_index(output_dir: Path) -> None:
    files = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name == "sha256.json":
            continue
        files.append(
            {
                "name": path.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    payload = {
        "schema_version": SHA_SCHEMA,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "files": files,
    }
    (output_dir / "sha256.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_audio(path: str) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(path, always_2d=False)
    return np.asarray(samples), int(sample_rate)


def _git_sha(cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    sha = result.stdout.strip()
    return sha or None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="从源 JSONL 构建 Phase 1 五类场景 manifest")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--sources", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--git-sha", default=None)
    args = parser.parse_args(argv)
    try:
        config = load_data_config(args.config)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.seed != config["seed"]:
        print(f"seed {args.seed} does not match config seed {config['seed']}", file=sys.stderr)
        return 2
    if not args.sources.is_file():
        print(f"missing sources {args.sources}", file=sys.stderr)
        return 2
    git_sha = args.git_sha if args.git_sha is not None else _git_sha(Path.cwd())
    try:
        report = build_manifest(config, args.sources, args.output_dir, git_sha)
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({"status": report["status"], "rows": report["counts"]["rows"], "rejected": report["counts"]["rejected"]}, ensure_ascii=False))
    return STATUS_CODE[report["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
