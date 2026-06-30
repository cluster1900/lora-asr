#!/usr/bin/env python3
"""Create v6A hard-profile train/val data from existing clean bootstrap audio.

The script intentionally uses this project's existing LoRA MVP clean audio as
the source and does not touch the fixed MVP 150 held-out test set.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from array import array
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from create_mvp_eval_audio import (  # noqa: E402
    DEGRADERS,
    PROFILE_SETTINGS,
    degradation_quality,
    project_root,
    read_pcm16_mono,
    reference_word_count,
    resolve_path,
    summarize_quality,
    text_length_bucket,
    write_pcm16,
)


DEFAULT_CONFIG = "configs/data/v6a_hard_profile.yaml"
DEFAULT_SCENARIOS = (
    "clean",
    "noise",
    "reverb",
    "noise_reverb",
    "far_field",
    "dropout",
    "far_field_noise",
)
COMPOSITE_STAGES = {
    "noise_reverb": ("reverb", "noise"),
    "far_field_noise": ("far_field", "noise"),
}
ALLOWED_SCENARIOS = {"clean", *DEGRADERS.keys(), *COMPOSITE_STAGES.keys()}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and skip blank lines."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file."""
    try:
        import yaml

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ModuleNotFoundError:
        return load_simple_yaml(path)


def parse_simple_scalar(value: str) -> Any:
    """Parse a scalar for the small project config fallback."""
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("'\"")


def load_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse the simple nested YAML shape used by the v6A data config.

    This keeps the data script runnable in a fresh local checkout even when
    PyYAML is not installed. It intentionally supports only maps and lists.
    """
    data: dict[str, Any] = {}
    current_section: str | None = None
    current_nested_list: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()

        if indent == 0:
            key, sep, value = stripped.partition(":")
            if sep != ":":
                raise ValueError(f"Unsupported YAML line: {raw_line}")
            if value.strip():
                data[key] = parse_simple_scalar(value.strip())
                current_section = None
            else:
                data[key] = {}
                current_section = key
            current_nested_list = None
            continue

        if current_section is None:
            raise ValueError(f"YAML line without section: {raw_line}")

        if indent == 2 and stripped.startswith("- "):
            if data[current_section] == {}:
                data[current_section] = []
            if not isinstance(data[current_section], list):
                raise ValueError(f"Section is not a list: {current_section}")
            data[current_section].append(parse_simple_scalar(stripped[2:].strip()))
            continue

        if indent == 2:
            key, sep, value = stripped.partition(":")
            if sep != ":" or not isinstance(data[current_section], dict):
                raise ValueError(f"Unsupported YAML line: {raw_line}")
            if value.strip():
                data[current_section][key] = parse_simple_scalar(value.strip())
                current_nested_list = None
            else:
                data[current_section][key] = []
                current_nested_list = key
            continue

        if indent == 4 and stripped.startswith("- "):
            if current_nested_list is None or not isinstance(data[current_section], dict):
                raise ValueError(f"Nested YAML list without key: {raw_line}")
            data[current_section][current_nested_list].append(parse_simple_scalar(stripped[2:].strip()))
            continue

        raise ValueError(f"Unsupported YAML line: {raw_line}")

    return data


def cfg_get(config: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    """Read a nested config value."""
    current: Any = config
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def parse_scenarios(value: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Parse and validate scenario names."""
    if value is None:
        scenarios = DEFAULT_SCENARIOS
    elif isinstance(value, str):
        scenarios = tuple(item.strip() for item in value.split(",") if item.strip())
    else:
        scenarios = tuple(str(item).strip() for item in value if str(item).strip())

    if not scenarios:
        raise ValueError("At least one scenario is required.")
    unknown = [scenario for scenario in scenarios if scenario not in ALLOWED_SCENARIOS]
    if unknown:
        raise ValueError(f"Unknown scenarios: {', '.join(unknown)}")
    return scenarios


def manifest_value(root: Path, path: Path, absolute_paths: bool) -> str:
    """Format paths for manifest/stats rows.

    Repo-local paths stay relative by default. Paths outside the repo, such as
    smoke outputs under /tmp, are written as absolute paths so validation still
    works.
    """
    resolved = path.resolve()
    if absolute_paths:
        return str(resolved)
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def resolve_source_audio(root: Path, manifest_path: Path, audio_root: Path | None, value: str) -> Path:
    """Resolve a source audio path from a manifest row."""
    audio_path = Path(value).expanduser()
    if audio_path.is_absolute():
        return audio_path

    candidates: list[Path] = []
    if audio_root is not None:
        candidates.append(audio_root / audio_path)
    candidates.append(manifest_path.parent / audio_path)
    candidates.append(root / audio_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def get_answer(row: dict[str, Any]) -> str:
    """Extract reference text from a manifest row."""
    for key in ("answer", "text", "transcript", "reference"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError("source row is missing answer/text/transcript/reference")


def clean_quality(samples: array) -> dict[str, float]:
    """Return quality-like fields for clean copies."""
    if not samples:
        clipping_ratio = 0.0
    else:
        clipping_ratio = sum(1 for sample in samples if abs(sample) >= 32760) / len(samples)
    return {
        "snr_db": 99.0,
        "rms_ratio": 1.0,
        "active_near_silence_ratio": 0.0,
        "clipping_ratio": round(clipping_ratio, 6),
    }


def apply_scenario(
    *,
    scenario: str,
    params: Any,
    clean_samples: array,
    seed: int,
    settings: dict[str, object],
) -> array:
    """Apply a single or composite degradation scenario."""
    if scenario == "clean":
        return array("h", clean_samples)
    if scenario in DEGRADERS:
        return DEGRADERS[scenario](params, clean_samples, random.Random(seed), settings)

    stages = COMPOSITE_STAGES[scenario]
    current = array("h", clean_samples)
    for stage_index, stage in enumerate(stages, start=1):
        stage_seed = seed + stage_index * 1_000_003
        current = DEGRADERS[stage](params, current, random.Random(stage_seed), settings)
    return current


def source_rows(
    *,
    manifest_path: Path,
    source_scenario: str,
    limit: int | None,
) -> list[dict[str, Any]]:
    """Load and filter source clean rows."""
    rows = [
        row
        for row in read_jsonl(manifest_path)
        if str(row.get("scenario", "")).strip() == source_scenario
    ]
    if not rows:
        raise ValueError(f"No source rows with scenario={source_scenario} in {manifest_path}")
    if limit is not None:
        rows = rows[:limit]
    return rows


def assert_overwrite_allowed(paths: list[Path], force: bool) -> None:
    """Refuse to overwrite generated data unless --force is provided."""
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        existing_text = "\n".join(str(path) for path in existing)
        raise SystemExit(f"Refusing to overwrite existing files. Use --force.\n{existing_text}")


def guard_output_dir(output_dir: Path, source_audio_root: Path | None) -> None:
    """Avoid deleting known input directories by mistake."""
    resolved = output_dir.resolve()
    forbidden = {
        project_root().resolve(),
        (project_root() / "data").resolve(),
        (project_root() / "data/lora_mvp/audio").resolve(),
        (project_root() / "data/mvp_eval/audio").resolve(),
    }
    if source_audio_root is not None:
        forbidden.add(source_audio_root.resolve())
    if resolved in forbidden:
        raise ValueError(f"Refusing to use protected output directory: {output_dir}")


def make_row(
    *,
    root: Path,
    split: str,
    scenario: str,
    base_id: str,
    utterance_id: str,
    output_audio: Path,
    source_audio: Path,
    source_manifest: Path,
    source_row: dict[str, Any],
    text: str,
    source_index: int,
    source_count: int,
    variant_index: int,
    seed: int,
    sample_rate: int,
    duration_seconds: float,
    profile: str,
    quality: dict[str, float],
    absolute_paths: bool,
) -> dict[str, Any]:
    """Build one v6A manifest row."""
    source_base_id = str(source_row.get("base_utterance_id", f"{split}_source_{source_index:04d}"))
    bucket = source_row.get("text_length_bucket") or text_length_bucket(source_index, source_count)
    word_count = source_row.get("reference_word_count") or reference_word_count(text)

    return {
        "audio": manifest_value(root, output_audio, absolute_paths),
        "answer": text,
        "language": source_row.get("language", "en"),
        "scenario": scenario,
        "split": split,
        "source": "lora_mvp_clean_copy" if scenario == "clean" else f"lora_mvp_clean_plus_{scenario}",
        "is_degraded": scenario != "clean",
        "utterance_id": utterance_id,
        "base_utterance_id": base_id,
        "source_base_utterance_id": source_base_id,
        "source_utterance_id": source_row.get("utterance_id"),
        "source_audio": manifest_value(root, source_audio, absolute_paths),
        "source_manifest": manifest_value(root, source_manifest, absolute_paths),
        "variant_index": variant_index,
        "degradation": "none" if scenario == "clean" else scenario,
        "profile": "clean" if scenario == "clean" else profile,
        "source_profile": source_row.get("profile"),
        "seed": seed,
        "sample_rate": sample_rate,
        "duration_seconds": round(duration_seconds, 4),
        "text_length_bucket": str(bucket),
        "reference_word_count": int(word_count),
        "approx_snr_db": quality["snr_db"],
        "rms_ratio": quality["rms_ratio"],
        "active_near_silence_ratio": quality["active_near_silence_ratio"],
        "clipping_ratio": quality["clipping_ratio"],
        "generation_stage": "v6A_hard_profile",
    }


def build_split(
    *,
    root: Path,
    split: str,
    rows: list[dict[str, Any]],
    source_manifest: Path,
    source_audio_root: Path | None,
    output_dir: Path,
    scenarios: tuple[str, ...],
    variants_per_utterance: int,
    profile: str,
    seed: int,
    absolute_paths: bool,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, float]]], list[float]]:
    """Build one train/val split."""
    settings = PROFILE_SETTINGS[profile]
    manifest_rows: list[dict[str, Any]] = []
    quality_by_scenario: dict[str, list[dict[str, float]]] = defaultdict(list)
    durations: list[float] = []
    split_offset = 0 if split == "train" else 1_000_000

    for source_index, source_row in enumerate(rows, start=1):
        source_audio_value = source_row.get("audio") or source_row.get("audio_path")
        if not source_audio_value:
            raise ValueError(f"{source_manifest} row {source_index} missing audio/audio_path")
        source_audio = resolve_source_audio(root, source_manifest, source_audio_root, str(source_audio_value))
        if not source_audio.exists():
            raise FileNotFoundError(f"Missing source audio: {source_audio}")

        params, clean_samples = read_pcm16_mono(source_audio)
        duration_seconds = params.nframes / params.framerate
        durations.append(round(duration_seconds, 4))
        text = get_answer(source_row)
        base_id = f"v6a_{split}_{source_index:04d}"

        for variant_index in range(1, variants_per_utterance + 1):
            for scenario_index, scenario in enumerate(scenarios):
                scenario_seed = (
                    seed
                    + split_offset
                    + source_index * 10_000
                    + variant_index * 100
                    + scenario_index
                )
                samples = apply_scenario(
                    scenario=scenario,
                    params=params,
                    clean_samples=clean_samples,
                    seed=scenario_seed,
                    settings=settings,
                )
                quality = clean_quality(clean_samples) if scenario == "clean" else degradation_quality(clean_samples, samples)
                quality_by_scenario[scenario].append(quality)

                utterance_id = f"{base_id}_{scenario}_v{variant_index:02d}"
                output_audio = output_dir / split / scenario / f"{utterance_id}.wav"
                write_pcm16(output_audio, params, samples)
                manifest_rows.append(
                    make_row(
                        root=root,
                        split=split,
                        scenario=scenario,
                        base_id=base_id,
                        utterance_id=utterance_id,
                        output_audio=output_audio,
                        source_audio=source_audio,
                        source_manifest=source_manifest,
                        source_row=source_row,
                        text=text,
                        source_index=source_index,
                        source_count=len(rows),
                        variant_index=variant_index,
                        seed=scenario_seed,
                        sample_rate=params.framerate,
                        duration_seconds=duration_seconds,
                        profile=profile,
                        quality=quality,
                        absolute_paths=absolute_paths,
                    )
                )

    return manifest_rows, dict(quality_by_scenario), durations


def scenario_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count scenarios for a row list."""
    return dict(sorted(Counter(str(row["scenario"]) for row in rows).items()))


def validate_rows(
    *,
    root: Path,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
    forbidden_fragments: list[str],
) -> dict[str, Any]:
    """Validate generated manifests and return validation stats."""
    all_rows = train_rows + val_rows
    missing_audio: list[str] = []
    forbidden_hits: list[str] = []

    for row in all_rows:
        audio_path = Path(str(row["audio"])).expanduser()
        resolved = audio_path if audio_path.is_absolute() else root / audio_path
        if not resolved.exists():
            missing_audio.append(str(row["audio"]))

        searchable = "\n".join(
            str(row.get(key, ""))
            for key in ("audio", "source_audio", "source_manifest")
        )
        for fragment in forbidden_fragments:
            if fragment and fragment in searchable:
                forbidden_hits.append(f"{row['utterance_id']}:{fragment}")

    train_ids = {str(row["base_utterance_id"]) for row in train_rows}
    val_ids = {str(row["base_utterance_id"]) for row in val_rows}
    overlap = sorted(train_ids & val_ids)
    train_source_ids = {str(row["source_base_utterance_id"]) for row in train_rows}
    val_source_ids = {str(row["source_base_utterance_id"]) for row in val_rows}
    source_overlap = sorted(train_source_ids & val_source_ids)

    if missing_audio:
        raise RuntimeError(f"Generated rows with missing audio: {missing_audio[:5]}")
    if forbidden_hits:
        raise RuntimeError(f"Forbidden path fragments found: {forbidden_hits[:5]}")
    if overlap:
        raise RuntimeError(f"train/val base_utterance_id overlap: {overlap[:5]}")
    if source_overlap:
        raise RuntimeError(f"train/val source_base_utterance_id overlap: {source_overlap[:5]}")

    return {
        "missing_audio": 0,
        "forbidden_path_hits": 0,
        "train_val_base_utterance_overlap": 0,
        "train_val_source_base_utterance_overlap": 0,
    }


def duration_summary(values: list[float]) -> dict[str, float]:
    """Summarize source clean durations."""
    if not values:
        return {"avg": 0.0, "min": 0.0, "max": 0.0}
    return {
        "avg": round(sum(values) / len(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }


def check_expected_rows(
    *,
    config: dict[str, Any],
    options: argparse.Namespace,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare configured expected row counts when running full mode."""
    skip = options.max_train_base_utterances is not None or options.max_val_base_utterances is not None
    expected_train = cfg_get(config, ("quality_checks", "expected_train_rows"))
    expected_val = cfg_get(config, ("quality_checks", "expected_val_rows"))
    result = {
        "expected_train_rows": expected_train,
        "expected_val_rows": expected_val,
        "checked": not skip,
    }
    if skip:
        return result
    if expected_train is not None and len(train_rows) != int(expected_train):
        raise RuntimeError(f"train row count {len(train_rows)} != expected {expected_train}")
    if expected_val is not None and len(val_rows) != int(expected_val):
        raise RuntimeError(f"val row count {len(val_rows)} != expected {expected_val}")
    return result


def build_dataset(options: argparse.Namespace) -> dict[str, Any]:
    """Build the v6A dataset."""
    root = project_root()
    config = load_yaml(options.config_path)
    train_manifest_in = resolve_path(root, options.source_train_manifest)
    val_manifest_in = resolve_path(root, options.source_val_manifest)
    output_dir = resolve_path(root, options.output_dir)
    train_manifest_out = resolve_path(root, options.train_manifest)
    val_manifest_out = resolve_path(root, options.val_manifest)
    stats_path = resolve_path(root, options.stats)
    source_audio_root = resolve_path(root, options.source_audio_root) if options.source_audio_root else None

    guard_output_dir(output_dir, source_audio_root if source_audio_root != root else None)
    assert_overwrite_allowed([output_dir, train_manifest_out, val_manifest_out, stats_path], options.force)
    if output_dir.exists() and options.force:
        shutil.rmtree(output_dir)

    train_source_rows = source_rows(
        manifest_path=train_manifest_in,
        source_scenario=options.source_scenario,
        limit=options.max_train_base_utterances,
    )
    val_source_rows = source_rows(
        manifest_path=val_manifest_in,
        source_scenario=options.source_scenario,
        limit=options.max_val_base_utterances,
    )

    train_rows, train_quality, train_durations = build_split(
        root=root,
        split="train",
        rows=train_source_rows,
        source_manifest=train_manifest_in,
        source_audio_root=source_audio_root,
        output_dir=output_dir,
        scenarios=options.scenarios,
        variants_per_utterance=options.variants_per_utterance,
        profile=options.profile,
        seed=options.seed,
        absolute_paths=options.absolute_paths,
    )
    val_rows, val_quality, val_durations = build_split(
        root=root,
        split="val",
        rows=val_source_rows,
        source_manifest=val_manifest_in,
        source_audio_root=source_audio_root,
        output_dir=output_dir,
        scenarios=options.scenarios,
        variants_per_utterance=options.variants_per_utterance,
        profile=options.profile,
        seed=options.seed,
        absolute_paths=options.absolute_paths,
    )

    forbidden_fragments = list(options.forbidden_path_fragments)
    validation = validate_rows(
        root=root,
        train_rows=train_rows,
        val_rows=val_rows,
        forbidden_fragments=forbidden_fragments,
    )
    row_expectations = check_expected_rows(
        config=config,
        options=options,
        train_rows=train_rows,
        val_rows=val_rows,
    )

    write_jsonl(train_manifest_out, train_rows)
    write_jsonl(val_manifest_out, val_rows)

    stats = {
        "dataset": options.dataset_name,
        "purpose": options.purpose,
        "generation_stage": "v6A_hard_profile",
        "seed": options.seed,
        "profile": options.profile,
        "variants_per_utterance": options.variants_per_utterance,
        "scenarios": list(options.scenarios),
        "source_scenario": options.source_scenario,
        "source_manifest": {
            "train": manifest_value(root, train_manifest_in, options.absolute_paths),
            "val": manifest_value(root, val_manifest_in, options.absolute_paths),
        },
        "output": {
            "audio_dir": manifest_value(root, output_dir, options.absolute_paths),
            "train_manifest": manifest_value(root, train_manifest_out, options.absolute_paths),
            "val_manifest": manifest_value(root, val_manifest_out, options.absolute_paths),
            "stats": manifest_value(root, stats_path, options.absolute_paths),
        },
        "source_clean_rows": {
            "train": len(train_source_rows),
            "val": len(val_source_rows),
        },
        "rows": {
            "train": len(train_rows),
            "val": len(val_rows),
            "total": len(train_rows) + len(val_rows),
        },
        "scenario_counts": {
            "train": scenario_counts(train_rows),
            "val": scenario_counts(val_rows),
        },
        "source_clean_duration_seconds": {
            "train": duration_summary(train_durations),
            "val": duration_summary(val_durations),
        },
        "degradation_stats": {
            "train": {
                scenario: summarize_quality(values)
                for scenario, values in sorted(train_quality.items())
            },
            "val": {
                scenario: summarize_quality(values)
                for scenario, values in sorted(val_quality.items())
            },
        },
        "validation": validation,
        "row_expectations": row_expectations,
        "forbidden_path_fragments": forbidden_fragments,
        "next_steps": {
            "difficulty_notebook": cfg_get(config, ("next_steps", "difficulty_notebook")),
            "train_notebook": cfg_get(config, ("next_steps", "train_notebook")),
        },
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def build_options(args: argparse.Namespace) -> argparse.Namespace:
    """Merge CLI overrides with the YAML config."""
    root = project_root()
    config_path = resolve_path(root, args.config)
    config = load_yaml(config_path)

    scenarios = parse_scenarios(args.scenarios if args.scenarios is not None else cfg_get(config, ("scenarios",)))
    profile = args.profile or cfg_get(config, ("dataset", "profile"), "hard")
    if profile not in PROFILE_SETTINGS:
        raise ValueError(f"Unknown profile: {profile}")

    absolute_paths = (
        args.absolute_paths
        if args.absolute_paths is not None
        else bool(cfg_get(config, ("dataset", "absolute_paths"), False))
    )

    return argparse.Namespace(
        config_path=config_path,
        dataset_name=cfg_get(config, ("dataset", "name"), "v6a_hard_profile"),
        purpose=cfg_get(config, ("dataset", "purpose"), "v6a_hard_profile_data_alignment"),
        seed=int(args.seed if args.seed is not None else cfg_get(config, ("dataset", "seed"), 20260702)),
        profile=profile,
        variants_per_utterance=int(
            args.variants_per_utterance
            if args.variants_per_utterance is not None
            else cfg_get(config, ("dataset", "variants_per_utterance"), 2)
        ),
        absolute_paths=absolute_paths,
        source_train_manifest=args.source_train_manifest or cfg_get(config, ("source", "train_manifest")),
        source_val_manifest=args.source_val_manifest or cfg_get(config, ("source", "val_manifest")),
        source_scenario=args.source_scenario or cfg_get(config, ("source", "source_scenario"), "clean"),
        source_audio_root=args.source_audio_root or cfg_get(config, ("source", "audio_root"), "."),
        forbidden_path_fragments=(
            args.forbidden_path_fragments.split(",")
            if args.forbidden_path_fragments
            else cfg_get(config, ("source", "forbidden_path_fragments"), [])
        ),
        output_dir=args.output_dir or cfg_get(config, ("output", "audio_dir")),
        train_manifest=args.train_manifest or cfg_get(config, ("output", "train_manifest")),
        val_manifest=args.val_manifest or cfg_get(config, ("output", "val_manifest")),
        stats=args.stats or cfg_get(config, ("output", "stats")),
        scenarios=scenarios,
        max_train_base_utterances=args.max_train_base_utterances,
        max_val_base_utterances=args.max_val_base_utterances,
        force=args.force,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--source-train-manifest", default=None)
    parser.add_argument("--source-val-manifest", default=None)
    parser.add_argument("--source-scenario", default=None)
    parser.add_argument("--source-audio-root", default=None)
    parser.add_argument("--forbidden-path-fragments", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-manifest", default=None)
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--stats", default=None)
    parser.add_argument("--scenarios", default=None, help="Comma-separated scenario list.")
    parser.add_argument("--profile", default=None, choices=sorted(PROFILE_SETTINGS))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--variants-per-utterance", type=int, default=None)
    parser.add_argument("--max-train-base-utterances", type=int, default=None)
    parser.add_argument("--max-val-base-utterances", type=int, default=None)
    parser.add_argument("--absolute-paths", action="store_true")
    parser.set_defaults(absolute_paths=None)
    parser.add_argument("--force", action="store_true", help="Overwrite generated output files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    options = build_options(args)
    stats = build_dataset(options)
    print("Generated v6A hard-profile dataset:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
