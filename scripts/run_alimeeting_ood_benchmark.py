#!/usr/bin/env python3
"""Phase 6 Track 2 read-only AliMeeting Eval paired near/far 4-stage OOD benchmark.

Evaluates the four frozen stages (`base`, `sft`, `dpo`, `rl`) on `ood_eval.jsonl`
under the exact frozen greedy `decode.eval` contract (`zh-cer-v1` CER + locked
`jieba==0.42.1` Chinese WER), computing near-vs-far degradation gaps, per-meeting
slices, and 2,000-resample paired source-cluster bootstrap 95% confidence intervals.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.eval_cer import evaluate, write_scenario_csv  # noqa: E402
from evaluation.jsonl_io import read_jsonl  # noqa: E402
from inference.config import load_infer_config  # noqa: E402
from scripts.prepare_alimeeting_ood_eval import OOD_SCENARIOS  # noqa: E402
from scripts.run_phase5_test_benchmark import (  # noqa: E402
    EXPECTED_EVAL_DECODE_SHA256,
    STAGES,
    load_sha_index,
    mean_greedy_reward_from_scored,
    sync_phase5_files,
    validate_published_releases,
    zero_lora_parameters,
)
from train.rl_schema import load_rl_config  # noqa: E402
from train.sft_checkpoint import atomic_write_json, atomic_write_text, sha256_file, utc_now  # noqa: E402
from train.sft_release import _match_training_runtime, completed_hypothesis_rows  # noqa: E402
from train.train_sft import RunStop, validate_rows  # noqa: E402

OOD_BENCHMARK_SCHEMA = "qwen3-asr-ood-benchmark-v1"


def verify_ood_manifest_and_audio(ood_manifest: Path, sha_index_path: Path) -> dict[str, Any]:
    """Verify read-only flags, 1:1 near/far pairing, and SHA-256 of ood_eval.jsonl + all WAVs."""
    if not ood_manifest.is_file():
        raise ValueError(f"OOD manifest is missing: {ood_manifest}")
    manifest_dir = ood_manifest.parent
    manifest_meta_path = manifest_dir / "manifest.json"
    if not manifest_meta_path.is_file():
        raise ValueError(f"OOD manifest.json is missing: {manifest_meta_path}")
    manifest_meta = json.loads(manifest_meta_path.read_text(encoding="utf-8"))
    if manifest_meta.get("trainable") is not False:
        raise ValueError("refusing OOD benchmark: manifest.json must have trainable=false")
    if manifest_meta.get("usage_policy") != "read_only_ood_evaluation":
        raise ValueError("refusing OOD benchmark: usage_policy must be 'read_only_ood_evaluation'")
    if manifest_meta.get("license") != "CC-BY-SA-4.0":
        raise ValueError(f"unexpected OOD manifest license: {manifest_meta.get('license')!r}")
    if manifest_meta.get("status") != "passed":
        raise ValueError(f"OOD manifest status is not 'passed': {manifest_meta.get('status')!r}")

    sha_map = load_sha_index(sha_index_path)
    actual_manifest_sha = sha256_file(ood_manifest)
    if sha_map.get(ood_manifest.name) != actual_manifest_sha:
        raise ValueError(
            f"{ood_manifest.name} sha256 mismatch against {sha_index_path}: "
            f"{actual_manifest_sha} != {sha_map.get(ood_manifest.name)}"
        )
    actual_meta_sha = sha256_file(manifest_meta_path)
    if "manifest.json" in sha_map and sha_map["manifest.json"] != actual_meta_sha:
        raise ValueError("manifest.json sha256 mismatch against sha256.json")

    rows = read_jsonl(ood_manifest)
    if not rows:
        raise ValueError("ood_eval.jsonl is empty")
    validate_rows(rows, "ood_eval")

    by_source: dict[str, dict[str, dict[str, Any]]] = {}
    meetings: set[str] = set()
    speakers: set[str] = set()
    verified_wavs = 0
    for row in rows:
        if row.get("split") != "ood_eval":
            raise ValueError(f"row {row.get('sample_id')} has invalid split {row.get('split')!r}")
        if row.get("license") != "CC-BY-SA-4.0":
            raise ValueError(f"row {row.get('sample_id')} has invalid license {row.get('license')!r}")
        sc = str(row.get("scenario"))
        if sc not in OOD_SCENARIOS:
            raise ValueError(f"row {row.get('sample_id')} has unsupported OOD scenario {sc!r}")
        src_id = str(row["source_utterance_id"])
        src_bucket = by_source.setdefault(src_id, {})
        if sc in src_bucket:
            raise ValueError(f"duplicate scenario {sc} for source {src_id}")
        src_bucket[sc] = row

        if isinstance(row.get("meeting_id"), str):
            meetings.add(row["meeting_id"])
        if isinstance(row.get("speaker_id"), str):
            speakers.add(row["speaker_id"])

        audio_rel = str(row["audio"])
        if audio_rel not in sha_map:
            raise ValueError(f"audio {audio_rel} missing from {sha_index_path}")
        audio_path = manifest_dir / audio_rel
        if not audio_path.is_file():
            raise ValueError(f"OOD audio file is missing: {audio_path}")
        if sha256_file(audio_path) != sha_map[audio_rel]:
            raise ValueError(f"OOD audio sha256 mismatch for {audio_rel}")
        verified_wavs += 1

    for src_id, sc_map in sorted(by_source.items()):
        if set(sc_map) != set(OOD_SCENARIOS):
            raise ValueError(f"source {src_id} does not have 1:1 near_field/far_field pair: {sorted(sc_map)}")
        if sc_map["near_field"]["text"] != sc_map["far_field"]["text"]:
            raise ValueError(f"source {src_id} has mismatched near/far reference text")

    return {
        "ood_manifest_sha256": actual_manifest_sha,
        "manifest_meta_sha256": actual_meta_sha,
        "dataset_name": str(manifest_meta.get("dataset_name", "alimeeting_eval_openslr119")),
        "license": "CC-BY-SA-4.0",
        "usage_policy": "read_only_ood_evaluation",
        "trainable": False,
        "ood_rows": len(rows),
        "paired_sources": len(by_source),
        "meetings": len(meetings),
        "speakers": len(speakers),
        "verified_wav_files": verified_wavs,
    }


def _compute_meeting_breakdown(
    manifest_rows: list[dict[str, Any]],
    scored_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    meta_by_id = {str(r["sample_id"]): r for r in manifest_rows}
    acc: dict[str, dict[str, dict[str, int]]] = {}
    for srow in scored_rows:
        sid = str(srow["sample_id"])
        mrow = meta_by_id[sid]
        meeting_id = str(mrow.get("meeting_id", "unknown"))
        sc = str(srow["scenario"])
        cell = acc.setdefault(meeting_id, {}).setdefault(
            sc,
            {"samples": 0, "edit_count": 0, "ref_chars": 0, "word_edit_count": 0, "ref_words": 0},
        )
        cell["samples"] += 1
        cell["edit_count"] += int(srow["edit_count"])
        cell["ref_chars"] += int(srow["reference_char_count"])
        cell["word_edit_count"] += int(srow.get("word_edit_count", 0))
        cell["ref_words"] += int(srow.get("reference_word_count", 0))

    result: dict[str, dict[str, Any]] = {}
    for meeting_id in sorted(acc):
        m_out: dict[str, Any] = {}
        for sc in OOD_SCENARIOS:
            c = acc[meeting_id].get(
                sc,
                {"samples": 0, "edit_count": 0, "ref_chars": 0, "word_edit_count": 0, "ref_words": 0},
            )
            m_out[sc] = {
                "samples": c["samples"],
                "edit_count": c["edit_count"],
                "reference_char_count": c["ref_chars"],
                "micro_cer": (c["edit_count"] / c["ref_chars"]) if c["ref_chars"] > 0 else 0.0,
                "word_edit_count": c["word_edit_count"],
                "reference_word_count": c["ref_words"],
                "micro_wer": (c["word_edit_count"] / c["ref_words"]) if c["ref_words"] > 0 else 0.0,
            }
        result[meeting_id] = m_out
    return result


def paired_cluster_bootstrap_ci(
    manifest_rows: list[dict[str, Any]],
    scored_a: list[dict[str, Any]],
    scored_b: list[dict[str, Any]],
    *,
    num_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired source-cluster bootstrap 95% CI for (stage_b - stage_a) micro CER and micro WER."""
    import numpy as np

    src_by_sample = {str(r["sample_id"]): str(r["source_utterance_id"]) for r in manifest_rows}
    map_a = {str(r["sample_id"]): r for r in scored_a}
    map_b = {str(r["sample_id"]): r for r in scored_b}

    sources = sorted(set(src_by_sample.values()))
    n_src = len(sources)
    src_index = {s: idx for idx, s in enumerate(sources)}

    # Shape: [n_src, 3] where columns 0=overall, 1=near_field, 2=far_field
    cer_ed_a = np.zeros((n_src, 3), dtype=np.float64)
    cer_ed_b = np.zeros((n_src, 3), dtype=np.float64)
    cer_ref = np.zeros((n_src, 3), dtype=np.float64)
    wer_ed_a = np.zeros((n_src, 3), dtype=np.float64)
    wer_ed_b = np.zeros((n_src, 3), dtype=np.float64)
    wer_ref = np.zeros((n_src, 3), dtype=np.float64)

    for sample_id, src_id in src_by_sample.items():
        ra = map_a[sample_id]
        rb = map_b[sample_id]
        idx = src_index[src_id]
        sc_col = 1 if ra["scenario"] == "near_field" else 2
        for col in (0, sc_col):
            cer_ed_a[idx, col] += float(ra["edit_count"])
            cer_ed_b[idx, col] += float(rb["edit_count"])
            cer_ref[idx, col] += float(ra["reference_char_count"])
            wer_ed_a[idx, col] += float(ra["word_edit_count"])
            wer_ed_b[idx, col] += float(rb["word_edit_count"])
            wer_ref[idx, col] += float(ra["reference_word_count"])

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n_src, size=(num_resamples, n_src))

    slices = ("overall", "near_field", "far_field")
    out: dict[str, Any] = {"resamples": num_resamples, "clusters": n_src}
    for col, name in enumerate(slices):
        ref_c_sum = cer_ref[:, col][boot_indices].sum(axis=1)
        da_c = cer_ed_a[:, col][boot_indices].sum(axis=1) / np.maximum(ref_c_sum, 1.0)
        db_c = cer_ed_b[:, col][boot_indices].sum(axis=1) / np.maximum(ref_c_sum, 1.0)
        delta_cer = db_c - da_c

        ref_w_sum = wer_ref[:, col][boot_indices].sum(axis=1)
        da_w = wer_ed_a[:, col][boot_indices].sum(axis=1) / np.maximum(ref_w_sum, 1.0)
        db_w = wer_ed_b[:, col][boot_indices].sum(axis=1) / np.maximum(ref_w_sum, 1.0)
        delta_wer = db_w - da_w

        point_cer = float((cer_ed_b[:, col].sum() - cer_ed_a[:, col].sum()) / max(cer_ref[:, col].sum(), 1.0))
        point_wer = float((wer_ed_b[:, col].sum() - wer_ed_a[:, col].sum()) / max(wer_ref[:, col].sum(), 1.0))
        out[name] = {
            "edit_delta": int(round(float(cer_ed_b[:, col].sum() - cer_ed_a[:, col].sum()))),
            "micro_cer_delta": point_cer,
            "micro_cer_ci95": [float(np.percentile(delta_cer, 2.5)), float(np.percentile(delta_cer, 97.5))],
            "word_edit_delta": int(round(float(wer_ed_b[:, col].sum() - wer_ed_a[:, col].sum()))),
            "micro_wer_delta": point_wer,
            "micro_wer_ci95": [float(np.percentile(delta_wer, 2.5)), float(np.percentile(delta_wer, 97.5))],
        }
    return out


def ood_stage_summary_entry(
    stage: str,
    release_meta: dict[str, Any] | None,
    manifest_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    scored_rows: list[dict[str, Any]],
    reward_scoring_cfg: dict[str, Any],
    *,
    predictions_sha256: str,
    metrics_sha256: str,
    scored_sha256: str,
) -> dict[str, Any]:
    by_sc = metrics["by_scenario"]
    near_sc = by_sc["near_field"]
    far_sc = by_sc["far_field"]
    return {
        "stage": stage,
        "release_id": "base_qwen3_asr_06b" if release_meta is None else str(release_meta["release_id"]),
        "source_run_id": None if release_meta is None else release_meta.get("source_run_id"),
        "source_step": 0 if release_meta is None else int(release_meta["source_step"]),
        "adapter_sha256": "base_zero_lora" if release_meta is None else str(release_meta["adapter_sha256"]),
        "samples": int(metrics["samples"]),
        "edit_count": int(metrics["edit_count"]),
        "reference_char_count": int(metrics["reference_char_count"]),
        "micro_cer": float(metrics["micro_cer"]),
        "macro_cer": float(metrics["macro_cer"]),
        "word_edit_count": int(metrics["word_edit_count"]),
        "reference_word_count": int(metrics["reference_word_count"]),
        "micro_wer": float(metrics["micro_wer"]),
        "macro_wer": float(metrics["macro_wer"]),
        "far_minus_near_micro_cer": float(far_sc["micro_cer"]) - float(near_sc["micro_cer"]),
        "far_minus_near_micro_wer": float(far_sc["micro_wer"]) - float(near_sc["micro_wer"]),
        "far_minus_near_edits": int(far_sc["edit_count"]) - int(near_sc["edit_count"]),
        "mean_greedy_reward": mean_greedy_reward_from_scored(scored_rows, reward_scoring_cfg),
        "empty_rate": float(metrics["empty_rate"]),
        "repeated_rate": float(metrics["repeated_rate"]),
        "long_rate": float(metrics["long_rate"]),
        "truncation_rate": float(metrics["truncation_rate"]),
        "inference_error_rate": float(metrics["inference_error_rate"]),
        "valid_output_rate": float(metrics["valid_output_rate"]),
        "by_scenario": {
            sc: {
                "samples": int(by_sc[sc]["samples"]),
                "edit_count": int(by_sc[sc]["edit_count"]),
                "reference_char_count": int(by_sc[sc]["reference_char_count"]),
                "micro_cer": float(by_sc[sc]["micro_cer"]),
                "macro_cer": float(by_sc[sc]["macro_cer"]),
                "word_edit_count": int(by_sc[sc]["word_edit_count"]),
                "reference_word_count": int(by_sc[sc]["reference_word_count"]),
                "micro_wer": float(by_sc[sc]["micro_wer"]),
                "macro_wer": float(by_sc[sc]["macro_wer"]),
                "empty_rate": float(by_sc[sc]["empty_rate"]),
                "repeated_rate": float(by_sc[sc]["repeated_rate"]),
                "long_rate": float(by_sc[sc]["long_rate"]),
                "truncation_rate": float(by_sc[sc]["truncation_rate"]),
            }
            for sc in OOD_SCENARIOS
        },
        "by_meeting": _compute_meeting_breakdown(manifest_rows, scored_rows),
        "predictions_sha256": predictions_sha256,
        "metrics_sha256": metrics_sha256,
        "scored_sha256": scored_sha256,
    }


def build_ood_benchmark_summary(
    *,
    run_id: str,
    train_config: dict[str, Any],
    infer_config: dict[str, Any],
    ood_verification: dict[str, Any],
    manifest_rows: list[dict[str, Any]],
    stage_entries: dict[str, dict[str, Any]],
    scored_by_stage: dict[str, list[dict[str, Any]]],
    bootstrap_resamples: int = 2000,
) -> dict[str, Any]:
    base_entry = stage_entries["base"]
    comparisons: dict[str, dict[str, Any]] = {}
    prev_stage = "base"
    for stage in STAGES:
        cur = stage_entries[stage]
        prev = stage_entries[prev_stage]
        comparisons[stage] = {
            "previous_stage": prev_stage,
            "total_edit_delta_vs_base": int(cur["edit_count"]) - int(base_entry["edit_count"]),
            "total_edit_delta_vs_prev": int(cur["edit_count"]) - int(prev["edit_count"]),
            "micro_cer_delta_vs_base": float(cur["micro_cer"]) - float(base_entry["micro_cer"]),
            "micro_cer_delta_vs_prev": float(cur["micro_cer"]) - float(prev["micro_cer"]),
            "total_word_edit_delta_vs_base": int(cur["word_edit_count"]) - int(base_entry["word_edit_count"]),
            "total_word_edit_delta_vs_prev": int(cur["word_edit_count"]) - int(prev["word_edit_count"]),
            "micro_wer_delta_vs_base": float(cur["micro_wer"]) - float(base_entry["micro_wer"]),
            "micro_wer_delta_vs_prev": float(cur["micro_wer"]) - float(prev["micro_wer"]),
            "near_field_edit_delta_vs_base": (
                int(cur["by_scenario"]["near_field"]["edit_count"])
                - int(base_entry["by_scenario"]["near_field"]["edit_count"])
            ),
            "far_field_edit_delta_vs_base": (
                int(cur["by_scenario"]["far_field"]["edit_count"])
                - int(base_entry["by_scenario"]["far_field"]["edit_count"])
            ),
            "far_minus_near_cer_gap_delta_vs_base": (
                float(cur["far_minus_near_micro_cer"]) - float(base_entry["far_minus_near_micro_cer"])
            ),
            "reward_delta_vs_base": float(cur["mean_greedy_reward"]) - float(base_entry["mean_greedy_reward"]),
            "reward_delta_vs_prev": float(cur["mean_greedy_reward"]) - float(prev["mean_greedy_reward"]),
        }
        prev_stage = stage

    pair_specs = (
        ("sft_vs_base", "base", "sft"),
        ("dpo_vs_base", "base", "dpo"),
        ("rl_vs_base", "base", "rl"),
        ("dpo_vs_sft", "sft", "dpo"),
        ("rl_vs_dpo", "dpo", "rl"),
    )
    bootstrap_ci = {
        label: paired_cluster_bootstrap_ci(
            manifest_rows,
            scored_by_stage[st_a],
            scored_by_stage[st_b],
            num_resamples=bootstrap_resamples,
            seed=int(train_config["seed"]),
        )
        for label, st_a, st_b in pair_specs
    }

    return {
        "schema_version": OOD_BENCHMARK_SCHEMA,
        "run_id": run_id,
        "split": "ood_eval",
        "seed": int(train_config["seed"]),
        "base_model": str(train_config["model"]["name"]),
        "base_revision": str(train_config["model"]["revision"]),
        "processor_sha256": str(train_config["model"]["processor_sha256"]),
        "eval_decode_sha256": str(train_config["decode_eval_sha256"]),
        "normalization_version": str(infer_config["scoring"]["normalization"]),
        "wer_tokenizer_info": {
            "name": infer_config["scoring"]["wer_tokenizer"],
            "version": infer_config["scoring"]["wer_tokenizer_version"],
            "dictionary_sha256": infer_config["scoring"]["wer_dictionary_sha256"],
            "hmm": bool(infer_config["scoring"]["wer_hmm"]),
        },
        "ood_manifest_verification": ood_verification,
        "stages": {stage: stage_entries[stage] for stage in STAGES},
        "comparisons": comparisons,
        "paired_cluster_bootstrap_95ci": bootstrap_ci,
        "generated_at": utc_now(),
    }


def format_ood_report(summary: dict[str, Any]) -> str:
    ver = summary["ood_manifest_verification"]
    lines = [
        "# Phase 6 Track 2: AliMeeting Eval Real-World OOD Four-Stage Benchmark Report",
        "",
        f"- **run_id**: `{summary['run_id']}`",
        f"- **dataset**: `{ver['dataset_name']}` (`license={ver['license']}`, `usage_policy={ver['usage_policy']}`, `trainable={ver['trainable']}`)",
        f"- **split**: `{summary['split']}` (`{ver['ood_rows']}` rows = `{ver['paired_sources']}` paired utterances × 2 scenarios across `{ver['meetings']}` meetings, `{ver['speakers']}` speakers)",
        f"- **ood_manifest_sha256**: `{ver['ood_manifest_sha256']}`",
        f"- **base_model**: `{summary['base_model']}` (`revision={summary['base_revision']}`)",
        f"- **eval_decode_sha256**: `{summary['eval_decode_sha256']}`",
        f"- **wer_tokenizer**: `{summary['wer_tokenizer_info']['name']}=={summary['wer_tokenizer_info']['version']}` (`hmm={summary['wer_tokenizer_info']['hmm']}`)",
        "",
        "## 1. Overall OOD Trajectory (CER + Locked `jieba` WER)",
        "",
        "| Stage | Release ID | Char Edits | Δ vs Base | Micro CER | Word Edits | Δ Word vs Base | Micro WER | Far−Near CER Gap | Mean Greedy Reward |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage in STAGES:
        st = summary["stages"][stage]
        cmp_row = summary["comparisons"][stage]
        lines.append(
            f"| `{stage.upper()}` | `{st['release_id']}` | `{st['edit_count']}` | "
            f"`{cmp_row['total_edit_delta_vs_base']:+d}` | `{st['micro_cer']:.6f}` | "
            f"`{st['word_edit_count']}` | `{cmp_row['total_word_edit_delta_vs_base']:+d}` | "
            f"`{st['micro_wer']:.6f}` | `{st['far_minus_near_micro_cer']:.6f}` | "
            f"`{st['mean_greedy_reward']:.6f}` |"
        )

    lines.extend(
        [
            "",
            "## 2. Near-Field Headset vs Far-Field 8-Ch Array (`ch0`) Breakdown",
            "",
            "| Scenario | Base Edits (CER / WER) | SFT Edits (CER / WER) | DPO Edits (CER / WER) | RL Edits (CER / WER) | DPO Δ vs Base | RL Δ vs Base |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for sc in OOD_SCENARIOS:
        b = summary["stages"]["base"]["by_scenario"][sc]
        s = summary["stages"]["sft"]["by_scenario"][sc]
        d = summary["stages"]["dpo"]["by_scenario"][sc]
        r = summary["stages"]["rl"]["by_scenario"][sc]
        lines.append(
            f"| `{sc}` | `{b['edit_count']}` (`{b['micro_cer']:.6f}` / `{b['micro_wer']:.6f}`) | "
            f"`{s['edit_count']}` (`{s['micro_cer']:.6f}` / `{s['micro_wer']:.6f}`) | "
            f"`{d['edit_count']}` (`{d['micro_cer']:.6f}` / `{d['micro_wer']:.6f}`) | "
            f"`{r['edit_count']}` (`{r['micro_cer']:.6f}` / `{r['micro_wer']:.6f}`) | "
            f"`{int(d['edit_count']) - int(b['edit_count']):+d}` | "
            f"`{int(r['edit_count']) - int(b['edit_count']):+d}` |"
        )

    lines.extend(
        [
            "",
            "## 3. Paired Source-Cluster Bootstrap 95% CIs (`B=2000`)",
            "",
            "| Comparison | Overall Δ CER [95% CI] | Near-Field Δ CER [95% CI] | Far-Field Δ CER [95% CI] | Overall Δ WER [95% CI] |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, ci_entry in summary["paired_cluster_bootstrap_95ci"].items():
        ov = ci_entry["overall"]
        nf = ci_entry["near_field"]
        ff = ci_entry["far_field"]
        lines.append(
            f"| `{label}` | `{ov['micro_cer_delta']:+.6f}` [`{ov['micro_cer_ci95'][0]:+.6f}`, `{ov['micro_cer_ci95'][1]:+.6f}`] | "
            f"`{nf['micro_cer_delta']:+.6f}` [`{nf['micro_cer_ci95'][0]:+.6f}`, `{nf['micro_cer_ci95'][1]:+.6f}`] | "
            f"`{ff['micro_cer_delta']:+.6f}` [`{ff['micro_cer_ci95'][0]:+.6f}`, `{ff['micro_cer_ci95'][1]:+.6f}`] | "
            f"`{ov['micro_wer_delta']:+.6f}` [`{ov['micro_wer_ci95'][0]:+.6f}`, `{ov['micro_wer_ci95'][1]:+.6f}`] |"
        )
    lines.append("")
    return "\n".join(lines)


def run_alimeeting_ood_benchmark(
    *,
    train_config_path: Path,
    infer_config_path: Path,
    ood_manifest: Path,
    manifest_sha_index: Path,
    sft_release_dir: Path,
    dpo_release_dir: Path,
    rl_release_dir: Path,
    output_dir: Path,
    run_id: str,
    seed: int,
    drive_root: Path | None = None,
    enforce_pinned_hashes: bool = True,
    bootstrap_resamples: int = 2000,
    decode_fn: Callable[..., list[dict[str, Any]]] | None = None,
    model_loader: Callable[[dict[str, Any]], tuple[Any, Any, dict[str, Any]]] | None = None,
    adapter_loader: Callable[[Any, Path], None] | None = None,
) -> dict[str, Any]:
    train_cfg = load_rl_config(train_config_path)
    infer_cfg = load_infer_config(infer_config_path)

    if int(seed) != int(train_cfg["seed"]) or int(seed) != int(infer_cfg["seed"]):
        raise ValueError(
            f"seed mismatch: CLI={seed}, train_config={train_cfg['seed']}, infer_config={infer_cfg['seed']}"
        )
    if train_cfg["model"]["name"] != infer_cfg["model"]["name"]:
        raise ValueError("model.name mismatch between train_config and infer_config")
    if train_cfg["model"]["revision"] != infer_cfg["model"]["revision"]:
        raise ValueError("model.revision mismatch between train_config and infer_config")
    if train_cfg["decode_eval_sha256"] != infer_cfg["decode_eval_sha256"]:
        raise ValueError("decode_eval_sha256 mismatch between train_config and infer_config")
    if enforce_pinned_hashes and train_cfg["decode_eval_sha256"] != EXPECTED_EVAL_DECODE_SHA256:
        raise ValueError(
            f"decode_eval_sha256 {train_cfg['decode_eval_sha256']} != {EXPECTED_EVAL_DECODE_SHA256}"
        )
    if infer_cfg["scoring"].get("wer_tokenizer") != "jieba":
        raise ValueError("infer_config must enable locked jieba WER (wer_tokenizer: 'jieba')")

    ood_info = verify_ood_manifest_and_audio(ood_manifest, manifest_sha_index)
    releases = validate_published_releases(
        sft_release_dir,
        dpo_release_dir,
        rl_release_dir,
        train_cfg,
        enforce_pinned_ids=enforce_pinned_hashes,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    drive_run_dir = (drive_root / "runs" / run_id) if (drive_root is not None and drive_root.is_dir()) else None
    atomic_write_json(
        output_dir / "status.json",
        {"status": "running", "reason": "alimeeting_ood_eval", "time": utc_now(), "run_id": run_id},
    )

    rows = read_jsonl(ood_manifest)
    manifest_sha = sha256_file(ood_manifest)

    if model_loader is None or adapter_loader is None or decode_fn is None:
        from train.train_sft import decode_eval_rows, load_adapter, load_sft_model

        if model_loader is None:
            model_loader = load_sft_model
        if adapter_loader is None:
            adapter_loader = load_adapter
        if decode_fn is None:
            decode_fn = decode_eval_rows

    _match_training_runtime(train_cfg)
    model, processor, _trainable = model_loader(train_cfg)

    stage_entries: dict[str, dict[str, Any]] = {}
    scored_by_stage: dict[str, list[dict[str, Any]]] = {}
    stage_configs: list[tuple[str, dict[str, Any] | None, Path | None]] = [
        ("base", None, None),
        ("sft", releases["sft"], sft_release_dir / "adapter" / "adapter.safetensors"),
        ("dpo", releases["dpo"], dpo_release_dir / "adapter" / "adapter.safetensors"),
        ("rl", releases["rl"], rl_release_dir / "adapter" / "adapter.safetensors"),
    ]

    for stage, rel_meta, adapter_path in stage_configs:
        pred_rel = f"predictions/{stage}.jsonl"
        pred_path = output_dir / pred_rel
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        done = completed_hypothesis_rows(pred_path, rows)
        print(f"OOD_STAGE_START stage={stage} resume={done}/{len(rows)}", flush=True)

        if done < len(rows):
            _match_training_runtime(train_cfg)
            if stage == "base":
                zero_lora_parameters(model)
                adapter_sha = "base_zero_lora"
                step_num = 0
            else:
                assert rel_meta is not None and adapter_path is not None
                adapter_loader(model, adapter_path)
                if sha256_file(adapter_path) != rel_meta["adapter_sha256"]:
                    raise ValueError(f"{stage} adapter file hash changed after load")
                adapter_sha = str(rel_meta["adapter_sha256"])
                step_num = int(rel_meta["source_step"])

            written = done
            with pred_path.open("a" if done else "w", encoding="utf-8") as handle:

                def on_row(prediction: dict[str, Any]) -> None:
                    nonlocal written
                    handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                    handle.flush()
                    written += 1

                def on_progress(finished: int, _total: int) -> None:
                    absolute = done + finished
                    print(f"OOD_PROGRESS stage={stage} done={absolute}/{len(rows)}", flush=True)
                    if absolute % 250 == 0 or absolute == len(rows):
                        sync_phase5_files(output_dir, drive_run_dir, [pred_rel])

                decode_fn(
                    model,
                    processor,
                    rows[done:],
                    ood_manifest,
                    train_cfg,
                    step_num,
                    adapter_sha,
                    on_progress=on_progress,
                    on_row=on_row,
                )
            if written != len(rows):
                raise ValueError(f"stage {stage} wrote {written} predictions, expected {len(rows)}")

        predictions = read_jsonl(pred_path)
        predictions_sha = sha256_file(pred_path)
        scored, metrics = evaluate(
            rows,
            predictions,
            manifest_sha256=manifest_sha,
            predictions_sha256=predictions_sha,
            decode_eval_sha256=infer_cfg["decode_eval_sha256"],
            wer_tokenizer=infer_cfg["scoring"]["wer_tokenizer"],
            wer_tokenizer_version=infer_cfg["scoring"].get("wer_tokenizer_version"),
            wer_dictionary_sha256=infer_cfg["scoring"].get("wer_dictionary_sha256"),
            wer_hmm=bool(infer_cfg["scoring"].get("wer_hmm", False)),
        )
        scored_by_stage[stage] = scored
        eval_dir = output_dir / "eval" / stage
        eval_dir.mkdir(parents=True, exist_ok=True)
        scored_path = eval_dir / "scored.jsonl"
        metrics_path = eval_dir / "metrics.json"
        csv_path = eval_dir / "by_scenario.csv"
        with scored_path.open("w", encoding="utf-8") as handle:
            for item in scored:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        atomic_write_json(metrics_path, metrics)
        write_scenario_csv(csv_path, metrics["by_scenario"])

        stage_entries[stage] = ood_stage_summary_entry(
            stage,
            rel_meta,
            rows,
            metrics,
            scored,
            train_cfg["scoring"],
            predictions_sha256=predictions_sha,
            metrics_sha256=sha256_file(metrics_path),
            scored_sha256=sha256_file(scored_path),
        )
        sync_phase5_files(
            output_dir,
            drive_run_dir,
            [
                pred_rel,
                f"eval/{stage}/scored.jsonl",
                f"eval/{stage}/metrics.json",
                f"eval/{stage}/by_scenario.csv",
            ],
        )
        print(
            f"OOD_STAGE_DONE stage={stage} edits={stage_entries[stage]['edit_count']} "
            f"micro_cer={stage_entries[stage]['micro_cer']:.8f} "
            f"micro_wer={stage_entries[stage]['micro_wer']:.8f} "
            f"near_cer={stage_entries[stage]['by_scenario']['near_field']['micro_cer']:.8f} "
            f"far_cer={stage_entries[stage]['by_scenario']['far_field']['micro_cer']:.8f}",
            flush=True,
        )

    summary = build_ood_benchmark_summary(
        run_id=run_id,
        train_config=train_cfg,
        infer_config=infer_cfg,
        ood_verification=ood_info,
        manifest_rows=rows,
        stage_entries=stage_entries,
        scored_by_stage=scored_by_stage,
        bootstrap_resamples=bootstrap_resamples,
    )
    atomic_write_json(output_dir / "benchmark_summary.json", summary)
    atomic_write_text(output_dir / "run_report.md", format_ood_report(summary))
    atomic_write_json(
        output_dir / "status.json",
        {
            "status": "completed",
            "reason": "all_ood_stages_evaluated",
            "time": utc_now(),
            "run_id": run_id,
            "stages_completed": list(STAGES),
        },
    )
    sync_status = sync_phase5_files(
        output_dir,
        drive_run_dir,
        ["benchmark_summary.json", "run_report.md", "status.json"],
    )
    print(
        f"OOD_BENCHMARK_DONE run_id={run_id} sync={sync_status} "
        f"base_edits={stage_entries['base']['edit_count']} "
        f"sft_edits={stage_entries['sft']['edit_count']} "
        f"dpo_edits={stage_entries['dpo']['edit_count']} "
        f"rl_edits={stage_entries['rl']['edit_count']}",
        flush=True,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 6 Track 2 AliMeeting Eval 四阶段只读 OOD 评测")
    parser.add_argument("--train-config", default=Path("configs/train/rl_pilot_aligned.yaml"), type=Path)
    parser.add_argument("--infer-config", default=Path("configs/inference/base_eval_wer.yaml"), type=Path)
    parser.add_argument("--ood-manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha-index", required=True, type=Path)
    parser.add_argument("--sft-release-dir", required=True, type=Path)
    parser.add_argument("--dpo-release-dir", required=True, type=Path)
    parser.add_argument("--rl-release-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--drive-root",
        default=Path(os.environ.get("QWEN3_ASR_DRIVE_ROOT", "/content/drive/MyDrive/qwen3-asr")),
        type=Path,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_alimeeting_ood_benchmark(
            train_config_path=args.train_config,
            infer_config_path=args.infer_config,
            ood_manifest=args.ood_manifest,
            manifest_sha_index=args.manifest_sha_index,
            sft_release_dir=args.sft_release_dir,
            dpo_release_dir=args.dpo_release_dir,
            rl_release_dir=args.rl_release_dir,
            output_dir=args.output_dir,
            run_id=args.run_id,
            seed=args.seed,
            drive_root=args.drive_root,
        )
    except (OSError, ValueError, json.JSONDecodeError, RunStop) as exc:
        print(f"OOD_BENCHMARK_FAIL {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
