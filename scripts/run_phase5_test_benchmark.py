#!/usr/bin/env python3
"""Phase 5 one-shot sealed test.jsonl benchmark across Base, SFT, DPO, and RL.

This script evaluates the four frozen stages (`base`, `sft`, `dpo`, `rl`) on the
sealed 2,500-row `test.jsonl` manifest using the exact same greedy `decode.eval`
contract and `zh-cer-v1` scorer. Before decoding any row, it verifies:
1. `test.jsonl` SHA-256 and row schema;
2. All 2,500 test WAV files against `manifest/sha256.json` (deterministically
   rendering missing degraded scenarios from the 500 `clean` test WAVs if needed);
3. That SFT, DPO, and RL releases are `published` (`publishable=True`), have
   matching base model/processor/decode hashes, and form an unbroken lineage
   (`base -> sft -> dpo -> rl`).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.eval_cer import evaluate  # noqa: E402
from evaluation.jsonl_io import read_jsonl  # noqa: E402
from scripts.manifest_audio import load_mono_16k, render_scenarios  # noqa: E402
from scripts.manifest_contract import SCENARIOS, load_data_config, scenario_seed  # noqa: E402
from train.dpo_release import load_dpo_release  # noqa: E402
from train.rl_contract import mean_greedy_reward  # noqa: E402
from train.rl_release import load_rl_release  # noqa: E402
from train.rl_schema import load_rl_config  # noqa: E402
from train.sft_checkpoint import atomic_write_json, atomic_write_text, sha256_file, utc_now  # noqa: E402
from train.sft_contract import DEGRADED_SCENARIOS  # noqa: E402
from train.sft_release import _match_training_runtime, completed_hypothesis_rows, load_release  # noqa: E402
from train.train_sft import RunStop, validate_rows, write_scenario_csv  # noqa: E402

PHASE5_BENCHMARK_SCHEMA = "qwen3-asr-phase5-benchmark-v1"
EXPECTED_TEST_MANIFEST_SHA256 = "611051822cd6a3a11a1c4df63b2434b18ee43475a7fe7569aeb13426fc4cc588"
EXPECTED_EVAL_DECODE_SHA256 = "4a4fee2b67884438faa196909749fcc8dcc6126b175554ad19992bf9b4f2f642"
STAGES = ("base", "sft", "dpo", "rl")
EXPECTED_RELEASES: dict[str, tuple[str, str]] = {
    "sft": (
        "sft_06b_zh_20261015T234939_step000030",
        "a185a61f54b6fd74bd5e3bfa1173e15450cfbb772731567671a928aad805096a",
    ),
    "dpo": (
        "dpo_06b_zh_20261022T133007_step000040",
        "1d28d3119385c178c4b1b9a1e52a68d134f33a9bfd3de36659290a184ba5c51a",
    ),
    "rl": (
        "rl_06b_zh_20261023T042034_step000005",
        "79df11247728f149df4999447bbcd05ae66ae5dd1c5646f48d7a9515a94ff3a2",
    ),
}


def load_sha_index(sha_index_path: Path) -> dict[str, str]:
    payload = json.loads(sha_index_path.read_text(encoding="utf-8"))
    files = payload.get("files")
    if isinstance(files, list):
        mapping: dict[str, str] = {}
        for item in files:
            if isinstance(item, dict) and isinstance(item.get("name"), str) and isinstance(item.get("sha256"), str):
                mapping[item["name"]] = item["sha256"]
        if mapping:
            return mapping
    if isinstance(files, dict):
        return {str(k): str(v) for k, v in files.items()}
    raise ValueError(f"invalid sha256 index structure in {sha_index_path}")


def _reconcile_noise_ulp(
    clean_wave: Any,
    seed: int,
    source_id: str,
    snr_db: float,
    target_path: Path,
    expected_sha256: str,
    wav_writer: Callable[..., None],
    *,
    max_ulp: int = 8,
) -> bool:
    """Reconcile cross-CPU (arm64 NEON vs x86_64 AVX-512) float32 power-sum ULP drift on noise WAVs."""
    import numpy as np

    audio = np.asarray(clean_wave, dtype=np.float32)
    signal_power = float(np.mean(np.square(audio)))
    if signal_power <= 0.0:
        return False
    noise_seed = scenario_seed(seed, source_id, "noise")
    generator = np.random.default_rng(noise_seed)
    noise = generator.normal(0.0, 1.0, size=audio.shape).astype(np.float32)
    noise_power = float(np.mean(np.square(noise)))
    scale0 = np.float32(np.sqrt(signal_power / (noise_power * (10.0 ** (snr_db / 10.0)))))
    s_neg = scale0
    s_pos = scale0
    for _ in range(max_ulp):
        s_neg = np.nextafter(s_neg, np.float32(-np.inf))
        s_pos = np.nextafter(s_pos, np.float32(np.inf))
        for candidate_scale in (s_neg, s_pos):
            candidate_wave = np.clip(audio + noise * candidate_scale, -1.0, 1.0)
            wav_writer(target_path, candidate_wave, 16000, subtype="PCM_16", format="WAV")
            if sha256_file(target_path) == expected_sha256:
                return True
    return False


def ensure_test_audio_verified(
    test_manifest: Path,
    sha_index_path: Path,
    data_config_path: Path,
    *,
    pilot_sample_tar: Path | None = None,
    expected_test_sha256: str | None = EXPECTED_TEST_MANIFEST_SHA256,
) -> dict[str, Any]:
    """Verify test.jsonl and all referenced WAVs against sha256.json, rendering degraded WAVs if missing."""
    import tarfile
    import soundfile as sf

    if not test_manifest.is_file():
        raise ValueError(f"test manifest is missing: {test_manifest}")
    actual_manifest_sha = sha256_file(test_manifest)
    if expected_test_sha256 is not None and actual_manifest_sha != expected_test_sha256:
        raise ValueError(
            f"test.jsonl sha256 mismatch: {actual_manifest_sha} != {expected_test_sha256}"
        )
    rows = read_jsonl(test_manifest)
    if not rows:
        raise ValueError("test.jsonl is empty")
    validate_rows(rows, "test")
    for row in rows:
        if row.get("split") != "test":
            raise ValueError(f"row {row.get('sample_id')} has non-test split: {row.get('split')}")

    sha_map = load_sha_index(sha_index_path)
    if "test.jsonl" in sha_map and sha_map["test.jsonl"] != actual_manifest_sha:
        raise ValueError("test.jsonl hash does not match sha256.json")

    data_cfg = load_data_config(data_config_path)
    manifest_root = test_manifest.parent
    by_source: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        audio_rel = str(row["audio"])
        if audio_rel not in sha_map:
            raise ValueError(f"audio path {audio_rel} is missing from {sha_index_path}")
        src_id = str(row["source_utterance_id"])
        sc = str(row["scenario"])
        by_source.setdefault(src_id, {})[sc] = row

    extracted_clean = 0
    missing_clean_utts: dict[str, Path] = {}
    for src_id, sc_rows in sorted(by_source.items()):
        if set(sc_rows) != set(SCENARIOS):
            raise ValueError(f"source {src_id} does not have all 5 scenarios: {sorted(sc_rows)}")
        clean_path = manifest_root / str(sc_rows["clean"]["audio"])
        if not clean_path.is_file():
            utt_id = src_id.split(":", 1)[-1]
            missing_clean_utts[f"{utt_id}.wav"] = clean_path

    if missing_clean_utts and pilot_sample_tar is not None and pilot_sample_tar.is_file():
        with tarfile.open(pilot_sample_tar, "r:*") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                norm_name = member.name.lstrip("./")
                base_name = Path(norm_name).name
                if ("/test/" in f"/{norm_name}") and base_name in missing_clean_utts:
                    target = missing_clean_utts[base_name]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    src_f = tf.extractfile(member)
                    if src_f is not None:
                        target.write_bytes(src_f.read())
                        extracted_clean += 1

    rendered_files = 0
    for src_id, sc_rows in sorted(by_source.items()):
        missing_scenarios = [
            sc for sc in SCENARIOS if not (manifest_root / str(sc_rows[sc]["audio"])).is_file()
        ]
        if not missing_scenarios:
            continue
        clean_path = manifest_root / str(sc_rows["clean"]["audio"])
        if not clean_path.is_file():
            raise ValueError(f"cannot render missing scenarios for {src_id}: clean audio {clean_path} is missing")
        clean_wave = load_mono_16k(str(clean_path), sf.read)
        waveforms = render_scenarios(clean_wave, int(data_cfg["seed"]), src_id, data_cfg["augmentation"])
        for sc in missing_scenarios:
            audio_rel = str(sc_rows[sc]["audio"])
            target = manifest_root / audio_rel
            target.parent.mkdir(parents=True, exist_ok=True)
            sf.write(target, waveforms[sc], 16000, subtype="PCM_16", format="WAV")
            if sc == "noise" and sha256_file(target) != sha_map[audio_rel]:
                _reconcile_noise_ulp(
                    clean_wave,
                    int(data_cfg["seed"]),
                    src_id,
                    float(data_cfg["augmentation"]["noise"]["snr_db"]),
                    target,
                    sha_map[audio_rel],
                    sf.write,
                )
            rendered_files += 1

    verified_files = 0
    for row in rows:
        audio_rel = str(row["audio"])
        audio_path = manifest_root / audio_rel
        if not audio_path.is_file():
            raise ValueError(f"test audio file is missing: {audio_path}")
        digest = sha256_file(audio_path)
        if digest != sha_map[audio_rel]:
            raise ValueError(f"test audio sha256 mismatch for {audio_rel}: {digest} != {sha_map[audio_rel]}")
        verified_files += 1

    return {
        "test_manifest_sha256": actual_manifest_sha,
        "test_rows": len(rows),
        "sources": len(by_source),
        "extracted_clean_files": extracted_clean,
        "rendered_files": rendered_files,
        "verified_files": verified_files,
    }


def validate_published_releases(
    sft_dir: Path,
    dpo_dir: Path,
    rl_dir: Path,
    config: dict[str, Any],
    *,
    enforce_pinned_ids: bool = True,
) -> dict[str, dict[str, Any]]:
    """Load and verify the SFT, DPO, and RL published releases and their lineage."""
    sft = load_release(sft_dir)
    dpo = load_dpo_release(dpo_dir)
    rl = load_rl_release(rl_dir)
    releases = {"sft": sft, "dpo": dpo, "rl": rl}

    for stage_name, rel in releases.items():
        if rel.get("status") != "published" or rel.get("publishable") is not True:
            raise ValueError(f"{stage_name} release is not published: status={rel.get('status')}")
        if rel.get("stage") != stage_name:
            raise ValueError(f"{stage_name} release has wrong stage field: {rel.get('stage')}")
        if rel.get("base_revision") != config["model"]["revision"]:
            raise ValueError(f"{stage_name} release base_revision mismatch")
        if rel.get("processor_sha256") != config["model"]["processor_sha256"]:
            raise ValueError(f"{stage_name} release processor_sha256 mismatch")
        if rel.get("eval_decode_sha256") != config["decode_eval_sha256"]:
            raise ValueError(f"{stage_name} release eval_decode_sha256 mismatch")
        if enforce_pinned_ids:
            expected_id, expected_sha = EXPECTED_RELEASES[stage_name]
            if rel.get("release_id") != expected_id:
                raise ValueError(f"{stage_name} release_id {rel.get('release_id')} != {expected_id}")
            if rel.get("adapter_sha256") != expected_sha:
                raise ValueError(f"{stage_name} adapter_sha256 {rel.get('adapter_sha256')} != {expected_sha}")

    if dpo.get("parent_release") != sft.get("release_id") or dpo.get("parent_release_sha256") != sft.get("adapter_sha256"):
        raise ValueError("DPO release parent lineage does not match the SFT release")
    if rl.get("parent_release") != dpo.get("release_id") or rl.get("parent_release_sha256") != dpo.get("adapter_sha256"):
        raise ValueError("RL release parent lineage does not match the DPO release")

    return releases


def zero_lora_parameters(model: Any) -> int:
    """Zero all lora_B weights so the LoRA injection acts as an exact identity (Base model)."""
    count = 0
    for name, param in model.named_parameters():
        if "lora_B" in name:
            param.data.zero_()
            if float(param.detach().abs().sum().item()) != 0.0:
                raise ValueError(f"failed to zero lora_B parameter: {name}")
            count += 1
    if count == 0:
        raise ValueError("no lora_B parameters found on model")
    return count


def mean_greedy_reward_from_scored(scored_rows: list[dict[str, Any]], scoring_cfg: dict[str, Any]) -> float:
    if not scored_rows:
        return 0.0
    clip_raw = scoring_cfg.get("reward_clip", [-1.0, 1.0])
    return mean_greedy_reward(
        scored_rows,
        empty_penalty=float(scoring_cfg.get("empty_penalty", 0.5)),
        repeated_penalty=float(scoring_cfg.get("repeated_penalty", 0.5)),
        reward_clip=(float(clip_raw[0]), float(clip_raw[1])),
    )


def stage_summary_entry(
    stage: str,
    release_meta: dict[str, Any] | None,
    metrics: dict[str, Any],
    scored_rows: list[dict[str, Any]],
    scoring_cfg: dict[str, Any],
    predictions_sha256: str,
    metrics_sha256: str,
    scored_sha256: str,
) -> dict[str, Any]:
    by_sc = metrics["by_scenario"]
    degraded_cer = [float(by_sc[name]["micro_cer"]) for name in DEGRADED_SCENARIOS]
    degraded_macro_cer = sum(degraded_cer) / len(DEGRADED_SCENARIOS)
    degraded_edits = sum(int(by_sc[name]["edit_count"]) for name in DEGRADED_SCENARIOS)
    overall = metrics.get("overall", metrics)
    return {
        "stage": stage,
        "release_id": "base_qwen3_asr_06b" if release_meta is None else str(release_meta["release_id"]),
        "source_run_id": None if release_meta is None else release_meta.get("source_run_id"),
        "source_step": 0 if release_meta is None else int(release_meta["source_step"]),
        "adapter_sha256": "base_zero_lora" if release_meta is None else str(release_meta["adapter_sha256"]),
        "samples": int(overall["samples"]),
        "edit_count": int(overall["edit_count"]),
        "reference_char_count": int(overall["reference_char_count"]),
        "micro_cer": float(overall["micro_cer"]),
        "macro_cer": float(overall["macro_cer"]),
        "degraded_edit_count": degraded_edits,
        "degraded_macro_cer": degraded_macro_cer,
        "mean_greedy_reward": mean_greedy_reward_from_scored(scored_rows, scoring_cfg),
        "empty_rate": float(overall["empty_rate"]),
        "repeated_rate": float(overall["repeated_rate"]),
        "long_rate": float(overall["long_rate"]),
        "truncation_rate": float(overall["truncation_rate"]),
        "inference_error_rate": float(overall["inference_error_rate"]),
        "valid_output_rate": float(overall["valid_output_rate"]),
        "by_scenario": {
            sc: {
                "samples": int(by_sc[sc]["samples"]),
                "edit_count": int(by_sc[sc]["edit_count"]),
                "reference_char_count": int(by_sc[sc]["reference_char_count"]),
                "micro_cer": float(by_sc[sc]["micro_cer"]),
                "macro_cer": float(by_sc[sc]["macro_cer"]),
            }
            for sc in SCENARIOS
        },
        "predictions_sha256": predictions_sha256,
        "metrics_sha256": metrics_sha256,
        "scored_sha256": scored_sha256,
    }


def build_benchmark_summary(
    *,
    run_id: str,
    config: dict[str, Any],
    audio_verification: dict[str, Any],
    stage_entries: dict[str, dict[str, Any]],
    val_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the 4-stage comparison summary with deltas vs Base and vs Previous stage."""
    base_entry = stage_entries["base"]
    comparisons: dict[str, dict[str, Any]] = {}
    prev_stage = "base"
    for stage in STAGES:
        cur = stage_entries[stage]
        prev = stage_entries[prev_stage]
        sc_deltas_vs_base = {
            sc: {
                "edit_delta": int(cur["by_scenario"][sc]["edit_count"]) - int(base_entry["by_scenario"][sc]["edit_count"]),
                "micro_cer_delta": float(cur["by_scenario"][sc]["micro_cer"]) - float(base_entry["by_scenario"][sc]["micro_cer"]),
            }
            for sc in SCENARIOS
        }
        sc_deltas_vs_prev = {
            sc: {
                "edit_delta": int(cur["by_scenario"][sc]["edit_count"]) - int(prev["by_scenario"][sc]["edit_count"]),
                "micro_cer_delta": float(cur["by_scenario"][sc]["micro_cer"]) - float(prev["by_scenario"][sc]["micro_cer"]),
            }
            for sc in SCENARIOS
        }
        comparisons[stage] = {
            "previous_stage": prev_stage,
            "total_edit_delta_vs_base": int(cur["edit_count"]) - int(base_entry["edit_count"]),
            "total_edit_delta_vs_prev": int(cur["edit_count"]) - int(prev["edit_count"]),
            "micro_cer_delta_vs_base": float(cur["micro_cer"]) - float(base_entry["micro_cer"]),
            "micro_cer_delta_vs_prev": float(cur["micro_cer"]) - float(prev["micro_cer"]),
            "degraded_edit_delta_vs_base": int(cur["degraded_edit_count"]) - int(base_entry["degraded_edit_count"]),
            "degraded_edit_delta_vs_prev": int(cur["degraded_edit_count"]) - int(prev["degraded_edit_count"]),
            "degraded_macro_cer_delta_vs_base": float(cur["degraded_macro_cer"]) - float(base_entry["degraded_macro_cer"]),
            "degraded_macro_cer_delta_vs_prev": float(cur["degraded_macro_cer"]) - float(prev["degraded_macro_cer"]),
            "reward_delta_vs_base": float(cur["mean_greedy_reward"]) - float(base_entry["mean_greedy_reward"]),
            "reward_delta_vs_prev": float(cur["mean_greedy_reward"]) - float(prev["mean_greedy_reward"]),
            "by_scenario_vs_base": sc_deltas_vs_base,
            "by_scenario_vs_prev": sc_deltas_vs_prev,
        }
        prev_stage = stage

    return {
        "schema_version": PHASE5_BENCHMARK_SCHEMA,
        "run_id": run_id,
        "split": "test",
        "seed": int(config["seed"]),
        "base_model": str(config["model"]["name"]),
        "base_revision": str(config["model"]["revision"]),
        "processor_sha256": str(config["model"]["processor_sha256"]),
        "eval_decode_sha256": str(config["decode_eval_sha256"]),
        "normalization_version": str(config["scoring"].get("cer_normalization", "zh-cer-v1")),
        "audit_disposition": "unlistened",
        "test_manifest_verification": audio_verification,
        "stages": {stage: stage_entries[stage] for stage in STAGES},
        "comparisons": comparisons,
        "val_reference": val_reference,
        "generated_at": utc_now(),
    }


def format_phase5_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Phase 5 Sealed `test.jsonl` Four-Stage Benchmark Report",
        "",
        f"- **run_id**: `{summary['run_id']}`",
        f"- **split**: `{summary['split']}` (`{summary['test_manifest_verification']['test_rows']}` rows, `{summary['test_manifest_verification']['sources']}` sources × 5 scenarios)",
        f"- **test_manifest_sha256**: `{summary['test_manifest_verification']['test_manifest_sha256']}`",
        f"- **base_model**: `{summary['base_model']}` (`revision={summary['base_revision']}`)",
        f"- **eval_decode_sha256**: `{summary['eval_decode_sha256']}`",
        f"- **audit_disposition**: `{summary['audit_disposition']}`",
        "",
        "## 1. Overall `test.jsonl` Trajectory (2,500 Rows)",
        "",
        "| Stage | Release ID |Total Edits | Δ vs Base | Δ vs Prev | Overall Micro CER | Degraded Edits (4 Scen) | Degraded Macro CER | Mean Greedy Reward |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stage in STAGES:
        st = summary["stages"][stage]
        cmp_row = summary["comparisons"][stage]
        lines.append(
            f"| `{stage.upper()}` | `{st['release_id']}` | `{st['edit_count']}` | "
            f"`{cmp_row['total_edit_delta_vs_base']:+d}` | `{cmp_row['total_edit_delta_vs_prev']:+d}` | "
            f"`{st['micro_cer']:.8f}` | `{st['degraded_edit_count']}` | "
            f"`{st['degraded_macro_cer']:.8f}` | `{st['mean_greedy_reward']:.8f}` |"
        )
    lines.extend(
        [
            "",
            "## 2. Per-Scenario `test.jsonl` Edits & Micro CER (500 Rows / Scenario)",
            "",
            "| Scenario | Base Edits (CER) | SFT Step 30 Edits (CER) | DPO Step 40 Edits (CER) | RL Step 5 Edits (CER) | RL Δ vs Base | RL Δ vs DPO |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for sc in SCENARIOS:
        b_sc = summary["stages"]["base"]["by_scenario"][sc]
        s_sc = summary["stages"]["sft"]["by_scenario"][sc]
        d_sc = summary["stages"]["dpo"]["by_scenario"][sc]
        r_sc = summary["stages"]["rl"]["by_scenario"][sc]
        d_base = int(r_sc["edit_count"]) - int(b_sc["edit_count"])
        d_dpo = int(r_sc["edit_count"]) - int(d_sc["edit_count"])
        lines.append(
            f"| `{sc}` | `{b_sc['edit_count']}` (`{b_sc['micro_cer']:.6f}`) | "
            f"`{s_sc['edit_count']}` (`{s_sc['micro_cer']:.6f}`) | "
            f"`{d_sc['edit_count']}` (`{d_sc['micro_cer']:.6f}`) | "
            f"`{r_sc['edit_count']}` (`{r_sc['micro_cer']:.6f}`) | "
            f"`{d_base:+d}` | `{d_dpo:+d}` |"
        )
    lines.append("")
    return "\n".join(lines)


def sync_phase5_files(output_dir: Path, drive_run_dir: Path | None, relatives: list[str]) -> str:
    if drive_run_dir is None:
        return "not_requested"
    for rel in relatives:
        src = output_dir / rel
        if not src.is_file():
            continue
        dst = drive_run_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        if sha256_file(src) != sha256_file(dst):
            raise ValueError(f"Drive readback sha256 mismatch for {rel}")
    return "synced"


def load_val_reference_summary(repo_root: Path) -> dict[str, Any] | None:
    paths = {
        "base": repo_root / "configs" / "train" / "aishell_val_base_20261015.json",
        "sft": repo_root / "configs" / "train" / "aishell_val_sft_step30_20261016.json",
        "dpo": repo_root / "configs" / "train" / "aishell_val_dpo_step40_20261022.json",
        "rl": repo_root / "configs" / "train" / "aishell_val_rl_step5_20261023.json",
    }
    if not all(p.is_file() for p in paths.values()):
        return None
    return {stage: json.loads(path.read_text(encoding="utf-8")) for stage, path in paths.items()}


def run_phase5_benchmark(
    *,
    config_path: Path,
    test_manifest: Path,
    manifest_sha_index: Path,
    data_config_path: Path,
    sft_release_dir: Path,
    dpo_release_dir: Path,
    rl_release_dir: Path,
    output_dir: Path,
    run_id: str,
    seed: int,
    drive_root: Path | None = None,
    pilot_sample_tar: Path | None = None,
    enforce_pinned_hashes: bool = True,
    decode_fn: Callable[..., list[dict[str, Any]]] | None = None,
    model_loader: Callable[[dict[str, Any]], tuple[Any, Any, dict[str, Any]]] | None = None,
    adapter_loader: Callable[[Any, Path], None] | None = None,
) -> dict[str, Any]:
    config = load_rl_config(config_path)
    if int(seed) != int(config["seed"]):
        raise ValueError(f"CLI seed {seed} does not match config seed {config['seed']}")
    if enforce_pinned_hashes and config["decode_eval_sha256"] != EXPECTED_EVAL_DECODE_SHA256:
        raise ValueError(
            f"decode_eval_sha256 {config['decode_eval_sha256']} != {EXPECTED_EVAL_DECODE_SHA256}"
        )

    if pilot_sample_tar is None and drive_root is not None:
        candidate = drive_root / "data" / "aishell1" / "20261015" / "pilot_sample_audio.tar"
        if candidate.is_file():
            pilot_sample_tar = candidate

    audio_info = ensure_test_audio_verified(
        test_manifest,
        manifest_sha_index,
        data_config_path,
        pilot_sample_tar=pilot_sample_tar,
        expected_test_sha256=EXPECTED_TEST_MANIFEST_SHA256 if enforce_pinned_hashes else None,
    )
    releases = validate_published_releases(
        sft_release_dir,
        dpo_release_dir,
        rl_release_dir,
        config,
        enforce_pinned_ids=enforce_pinned_hashes,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    drive_run_dir = (drive_root / "runs" / run_id) if (drive_root is not None and drive_root.is_dir()) else None
    atomic_write_json(
        output_dir / "status.json",
        {"status": "running", "reason": "phase5_test_eval", "time": utc_now(), "run_id": run_id},
    )

    rows = read_jsonl(test_manifest)
    manifest_sha = sha256_file(test_manifest)

    if model_loader is None or adapter_loader is None or decode_fn is None:
        from train.train_sft import decode_eval_rows, load_adapter, load_sft_model

        if model_loader is None:
            model_loader = load_sft_model
        if adapter_loader is None:
            adapter_loader = load_adapter
        if decode_fn is None:
            decode_fn = decode_eval_rows

    _match_training_runtime(config)
    model, processor, _trainable = model_loader(config)

    stage_entries: dict[str, dict[str, Any]] = {}
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
        print(f"PHASE5_STAGE_START stage={stage} resume={done}/{len(rows)}", flush=True)

        if done < len(rows):
            _match_training_runtime(config)
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
                    print(f"PHASE5_PROGRESS stage={stage} done={absolute}/{len(rows)}", flush=True)
                    if absolute % 500 == 0 or absolute == len(rows):
                        sync_phase5_files(output_dir, drive_run_dir, [pred_rel])

                decode_fn(
                    model,
                    processor,
                    rows[done:],
                    test_manifest,
                    config,
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
            decode_eval_sha256=config["decode_eval_sha256"],
            wer_tokenizer=config["scoring"]["wer_tokenizer"],
            wer_tokenizer_version=config["scoring"].get("wer_tokenizer_version"),
            wer_dictionary_sha256=config["scoring"].get("wer_dictionary_sha256"),
            wer_hmm=bool(config["scoring"].get("wer_hmm", False)),
        )
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

        stage_entries[stage] = stage_summary_entry(
            stage,
            rel_meta,
            metrics,
            scored,
            config["scoring"],
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
            f"PHASE5_STAGE_DONE stage={stage} edits={stage_entries[stage]['edit_count']} "
            f"micro_cer={stage_entries[stage]['micro_cer']:.8f} "
            f"degraded_macro_cer={stage_entries[stage]['degraded_macro_cer']:.8f}",
            flush=True,
        )

    summary = build_benchmark_summary(
        run_id=run_id,
        config=config,
        audio_verification=audio_info,
        stage_entries=stage_entries,
        val_reference=load_val_reference_summary(ROOT),
    )
    atomic_write_json(output_dir / "benchmark_summary.json", summary)
    atomic_write_text(output_dir / "run_report.md", format_phase5_report(summary))
    atomic_write_json(
        output_dir / "status.json",
        {
            "status": "completed",
            "reason": "all_stages_evaluated",
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
        f"PHASE5_BENCHMARK_DONE run_id={run_id} sync={sync_status} "
        f"base_edits={stage_entries['base']['edit_count']} "
        f"sft_edits={stage_entries['sft']['edit_count']} "
        f"dpo_edits={stage_entries['dpo']['edit_count']} "
        f"rl_edits={stage_entries['rl']['edit_count']}",
        flush=True,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 5 封存 test.jsonl 四阶段终局评测")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument("--manifest-sha-index", required=True, type=Path)
    parser.add_argument("--data-config", default=Path("configs/data/manifest_mvp.yaml"), type=Path)
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
    parser.add_argument("--pilot-sample-tar", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run_phase5_benchmark(
            config_path=args.config,
            test_manifest=args.test_manifest,
            manifest_sha_index=args.manifest_sha_index,
            data_config_path=args.data_config,
            sft_release_dir=args.sft_release_dir,
            dpo_release_dir=args.dpo_release_dir,
            rl_release_dir=args.rl_release_dir,
            output_dir=args.output_dir,
            run_id=args.run_id,
            seed=args.seed,
            drive_root=args.drive_root,
            pilot_sample_tar=args.pilot_sample_tar,
        )
    except (OSError, ValueError, json.JSONDecodeError, RunStop) as exc:
        print(f"PHASE5_BENCHMARK_FAIL {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
