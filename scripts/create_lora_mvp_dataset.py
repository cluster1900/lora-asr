#!/usr/bin/env python3
"""Generate LoRA MVP bootstrap train/val audio and JSONL manifests.

This script creates a small, reproducible bootstrap dataset for the first
formal LoRA MVP loop. It intentionally does not touch the fixed MVP 150
held-out test set.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from array import array
from collections import Counter
from pathlib import Path
from typing import Any

from create_mvp_eval_audio import (
    DEGRADERS,
    PROFILE_SETTINGS,
    audio_value,
    degradation_quality,
    project_root,
    read_pcm16_mono,
    reference_word_count,
    resolve_path,
    summarize_quality,
    synthesize_clean_audio,
    text_length_bucket,
    write_pcm16,
)


DEFAULT_SCENARIOS = ("clean", "noise", "reverb")

BOOTSTRAP_TEXTS = [
    "Please confirm the meeting room before the weekly review begins.",
    "The technician checked the microphone cable after the recording sounded faint.",
    "Send the updated schedule to the team before the afternoon call.",
    "The customer repeated the invoice number because the line was noisy.",
    "We need a clear transcript of the support conversation by Friday.",
    "The speaker moved closer to the laptop and started the presentation again.",
    "Please remind everyone to mute background music during the interview.",
    "The manager asked for a short summary of the delayed shipment.",
    "Record the next sentence in a quiet office with the door closed.",
    "The service desk received a voicemail about the broken printer.",
    "A short pause appeared before the caller gave the confirmation code.",
    "The analyst compared yesterday's recording with today's cleaner sample.",
    "Please keep the phone steady while reading the account details.",
    "The training script should save the checkpoint after each evaluation block.",
    "Background noise from the hallway made the final instruction difficult.",
    "The meeting started late because the conference speaker was not connected.",
    "Please review the transcript and mark every missing appointment time.",
    "The operator asked the caller to repeat the address one more time.",
    "We should test the adapter on clean speech before checking noisy clips.",
    "The audio file includes a product name, a deadline, and a room number.",
    "During the morning operations meeting, the dispatcher explained that several radio messages were difficult to hear because rain was hitting the vehicle roof.",
    "The support engineer saved a new diagnostic recording after the customer moved away from a loud air conditioner near the office window.",
    "When the classroom microphone was placed behind the projector, the teacher's voice sounded distant and several words disappeared under the fan noise.",
    "The evaluation report should separate clean recordings from noisy and reverberant clips so that regression can be measured without hiding scenario failures.",
    "A reliable recognizer should preserve names, numbers, dates, and short function words even when the room echo overlaps the beginning of the next phrase.",
    "The training manifest records the random seed, the scenario label, the source utterance, and the generated audio path for every synthetic example.",
    "Before changing the LoRA target modules, we need one controlled experiment that compares the adapter with the same baseline predictions.",
    "The caller gave a detailed explanation of the payment issue, but the recording contained keyboard clicks and several people talking nearby.",
    "If the adapter improves noisy speech but damages clean speech, the next stage should measure that regression before building a router.",
    "The validation split must contain different source utterances from the training split, even when both splits use similar degradation parameters.",
    "A long voicemail about delivery status, approval history, and meeting availability is useful for finding omissions in noisy transcription results.",
    "The first formal LoRA run should produce a checkpoint, a loss log, a configuration snapshot, predictions, and scenario level metrics.",
    "Please compare the reverberant recordings carefully because room reflections can make the model replace simple words with unrelated phrases.",
    "The project notebook should run in Google Colab with Drive paths while the same scripts remain usable from a local checkout.",
    "After the adapter is trained, the held out test set should remain unchanged so that every future experiment has the same reference point.",
    "The researcher warned that synthetic audio can reveal broken training code, but real recordings are still required before making product claims.",
]


def cycle_texts(count: int, start_index: int = 0) -> list[str]:
    """Return `count` deterministic texts, cycling with suffixes when needed."""
    texts: list[str] = []
    for offset in range(count):
        index = start_index + offset
        base = BOOTSTRAP_TEXTS[index % len(BOOTSTRAP_TEXTS)]
        cycle = index // len(BOOTSTRAP_TEXTS)
        if cycle:
            texts.append(f"{base} This is bootstrap variation {cycle + 1}.")
        else:
            texts.append(base)
    return texts


def parse_scenarios(value: str) -> tuple[str, ...]:
    """Parse and validate comma-separated scenario names."""
    scenarios = tuple(item.strip() for item in value.split(",") if item.strip())
    if not scenarios:
        raise ValueError("At least one scenario is required.")

    allowed = {"clean", *DEGRADERS.keys()}
    unknown = [scenario for scenario in scenarios if scenario not in allowed]
    if unknown:
        raise ValueError(f"Unknown scenarios: {', '.join(unknown)}")
    return scenarios


def assert_overwrite_allowed(paths: list[Path], force: bool) -> None:
    """Refuse to overwrite generated data unless --force is provided."""
    existing = [path for path in paths if path.exists()]
    if existing and not force:
        existing_text = "\n".join(str(path) for path in existing)
        raise SystemExit(f"Refusing to overwrite existing files. Use --force.\n{existing_text}")


def row_audio_path(root: Path, audio_path: Path, absolute_paths: bool) -> str:
    """Format audio path for manifest rows."""
    return audio_value(root, audio_path, absolute_paths)


def make_row(
    *,
    root: Path,
    split: str,
    scenario: str,
    audio_path: Path,
    text: str,
    index: int,
    items_in_split: int,
    source: str,
    is_degraded: bool,
    profile: str,
    seed: int,
    sample_rate: int,
    quality: dict[str, float] | None,
    absolute_paths: bool,
) -> dict[str, Any]:
    """Build a manifest row with the standard ASR fields plus metadata."""
    base_id = f"{split}_utt_{index:04d}"
    row: dict[str, Any] = {
        "audio": row_audio_path(root, audio_path, absolute_paths),
        "answer": text,
        "language": "en",
        "scenario": scenario,
        "split": split,
        "source": source,
        "is_degraded": is_degraded,
        "utterance_id": f"{base_id}_{scenario}",
        "base_utterance_id": base_id,
        "degradation": "none" if scenario == "clean" else scenario,
        "profile": "clean" if scenario == "clean" else profile,
        "seed": seed,
        "sample_rate": sample_rate,
        "text_length_bucket": text_length_bucket(index, items_in_split),
        "reference_word_count": reference_word_count(text),
    }
    if quality:
        row.update({
            "approx_snr_db": quality["snr_db"],
            "rms_ratio": quality["rms_ratio"],
            "active_near_silence_ratio": quality["active_near_silence_ratio"],
        })
    return row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write rows as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def split_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Return scenario counts per split."""
    counts: dict[str, Counter[str]] = {}
    for row in rows:
        split = str(row["split"])
        counts.setdefault(split, Counter())[str(row["scenario"])] += 1
    return {split: dict(sorted(counter.items())) for split, counter in sorted(counts.items())}


def validate_splits(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> None:
    """Check train/val base utterance separation."""
    train_ids = {str(row["base_utterance_id"]) for row in train_rows}
    val_ids = {str(row["base_utterance_id"]) for row in val_rows}
    overlap = train_ids & val_ids
    if overlap:
        preview = ", ".join(sorted(overlap)[:5])
        raise RuntimeError(f"train/val base_utterance_id overlap: {preview}")


def build_split(
    *,
    root: Path,
    output_dir: Path,
    split: str,
    items_per_scenario: int,
    scenarios: tuple[str, ...],
    profile: str,
    seed: int,
    sample_rate: int,
    voice: str,
    text_start_index: int,
    absolute_paths: bool,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, float]]], list[float]]:
    """Generate one split and return manifest rows plus stats components."""
    settings = PROFILE_SETTINGS[profile]
    split_dir = output_dir / split
    texts = cycle_texts(items_per_scenario, start_index=text_start_index)
    rows: list[dict[str, Any]] = []
    durations: list[float] = []
    quality_by_scenario: dict[str, list[dict[str, float]]] = {scenario: [] for scenario in scenarios if scenario != "clean"}

    for index, text in enumerate(texts, start=1):
        clean_path = split_dir / "clean" / f"clean_{index:04d}.wav"
        synthesize_clean_audio(text, clean_path, sample_rate, voice)
        params, clean_samples = read_pcm16_mono(clean_path)
        durations.append(round(params.nframes / params.framerate, 4))

        if "clean" in scenarios:
            rows.append(
                make_row(
                    root=root,
                    split=split,
                    scenario="clean",
                    audio_path=clean_path,
                    text=text,
                    index=index,
                    items_in_split=items_per_scenario,
                    source="macos_say",
                    is_degraded=False,
                    profile=profile,
                    seed=seed + index,
                    sample_rate=params.framerate,
                    quality=None,
                    absolute_paths=absolute_paths,
                )
            )

        for scenario in scenarios:
            if scenario == "clean":
                continue
            scenario_seed = seed + index * 100 + sorted(DEGRADERS).index(scenario)
            rng = random.Random(scenario_seed)
            degraded_samples: array = DEGRADERS[scenario](params, clean_samples, rng, settings)
            quality = degradation_quality(clean_samples, degraded_samples)
            quality_by_scenario[scenario].append(quality)
            degraded_path = split_dir / scenario / f"{scenario}_{index:04d}.wav"
            write_pcm16(degraded_path, params, degraded_samples)
            rows.append(
                make_row(
                    root=root,
                    split=split,
                    scenario=scenario,
                    audio_path=degraded_path,
                    text=text,
                    index=index,
                    items_in_split=items_per_scenario,
                    source=f"macos_say_plus_{scenario}",
                    is_degraded=True,
                    profile=profile,
                    seed=scenario_seed,
                    sample_rate=params.framerate,
                    quality=quality,
                    absolute_paths=absolute_paths,
                )
            )

    return rows, quality_by_scenario, durations


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    """Generate train/val audio, manifests and stats."""
    root = project_root()
    output_dir = resolve_path(root, args.output_dir)
    train_manifest = resolve_path(root, args.train_manifest)
    val_manifest = resolve_path(root, args.val_manifest)
    stats_path = resolve_path(root, args.stats)
    scenarios = parse_scenarios(args.scenarios)

    assert_overwrite_allowed([output_dir, train_manifest, val_manifest, stats_path], args.force)
    if output_dir.exists() and args.force:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows, train_quality, train_durations = build_split(
        root=root,
        output_dir=output_dir,
        split="train",
        items_per_scenario=args.train_items_per_scenario,
        scenarios=scenarios,
        profile=args.profile,
        seed=args.seed,
        sample_rate=args.sample_rate,
        voice=args.voice,
        text_start_index=0,
        absolute_paths=args.absolute_paths,
    )
    val_rows, val_quality, val_durations = build_split(
        root=root,
        output_dir=output_dir,
        split="val",
        items_per_scenario=args.val_items_per_scenario,
        scenarios=scenarios,
        profile=args.profile,
        seed=args.seed + 1_000_000,
        sample_rate=args.sample_rate,
        voice=args.voice,
        text_start_index=args.train_items_per_scenario,
        absolute_paths=args.absolute_paths,
    )
    validate_splits(train_rows, val_rows)

    write_jsonl(train_manifest, train_rows)
    write_jsonl(val_manifest, val_rows)

    all_rows = train_rows + val_rows
    quality_summary = {
        "train": {scenario: summarize_quality(values) for scenario, values in sorted(train_quality.items())},
        "val": {scenario: summarize_quality(values) for scenario, values in sorted(val_quality.items())},
    }
    stats = {
        "dataset": "lora_mvp_bootstrap",
        "profile": args.profile,
        "seed": args.seed,
        "voice": args.voice,
        "sample_rate": args.sample_rate,
        "scenarios": list(scenarios),
        "train_manifest": audio_value(root, train_manifest, args.absolute_paths),
        "val_manifest": audio_value(root, val_manifest, args.absolute_paths),
        "audio_dir": audio_value(root, output_dir, args.absolute_paths),
        "held_out_test_manifest": args.held_out_test_manifest,
        "rows": {
            "train": len(train_rows),
            "val": len(val_rows),
            "total": len(all_rows),
        },
        "scenario_counts": split_counts(all_rows),
        "train_val_base_utterance_overlap": 0,
        "duration_seconds": {
            "train_avg_clean": round(sum(train_durations) / len(train_durations), 4) if train_durations else 0.0,
            "val_avg_clean": round(sum(val_durations) / len(val_durations), 4) if val_durations else 0.0,
            "train_min_clean": min(train_durations) if train_durations else 0.0,
            "train_max_clean": max(train_durations) if train_durations else 0.0,
            "val_min_clean": min(val_durations) if val_durations else 0.0,
            "val_max_clean": max(val_durations) if val_durations else 0.0,
        },
        "degradation_stats": quality_summary,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-items-per-scenario", type=int, default=120)
    parser.add_argument("--val-items-per-scenario", type=int, default=30)
    parser.add_argument("--scenarios", default=",".join(DEFAULT_SCENARIOS))
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--voice", default="Alex", help="macOS `say` voice name.")
    parser.add_argument("--profile", default="medium", choices=sorted(PROFILE_SETTINGS))
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--output-dir", default="data/lora_mvp/audio")
    parser.add_argument("--train-manifest", default="data/jsonl/lora_mvp_train.local.jsonl")
    parser.add_argument("--val-manifest", default="data/jsonl/lora_mvp_val.local.jsonl")
    parser.add_argument("--stats", default="data/jsonl/lora_mvp_stats.local.json")
    parser.add_argument("--held-out-test-manifest", default="data/jsonl/baseline_mvp_150.local.jsonl")
    parser.add_argument("--absolute-paths", action="store_true")
    parser.add_argument("--force", action="store_true", help="Overwrite generated audio/manifest/stats.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_dataset(args)
    print("Generated LoRA MVP bootstrap dataset:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
