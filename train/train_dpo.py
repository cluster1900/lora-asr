#!/usr/bin/env python3
"""DPO pilot for Qwen3-ASR-0.6B on the published SFT adapter.

The loss is the mean over pairs of -logsigmoid(beta * (policy margin - reference
margin)). Reference logprobs are the cached sequence sums. The policy keeps the
SFT training hook. This run does not mark unlistened clips as heard.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.dpo_contract import (  # noqa: E402
    assert_audit_still_unlistened,
    build_clean_rejected_map,
    check_pair_metadata,
    dropout_anchor_step_slice,
    judge_dpo_checkpoint,
    listen_waiver_record,
    locate_dropout_edits,
    mismatched_token_mask,
    select_dropout_anchor_rows,
    select_training_pairs,
    validate_pair_rows,
)
from train.dpo_schema import load_dpo_config, semantic_dpo_config_sha256  # noqa: E402
from train.reference_logprob import (  # noqa: E402
    completed_logprob_rows,
    load_published_release,
    logprob_errors,
    ordered_pairs,
    select_recheck,
)
from train.sft_checkpoint import atomic_write_json, atomic_write_text, step_name, utc_now  # noqa: E402
from train.sft_contract import (  # noqa: E402
    iterate_updates,
    pending_checkpoint_eval,
    runtime_budget_disabled,
    runtime_budget_exceeded,
    select_sft_checkpoint,
    should_interrupt,
    update_ends_schedule,
)
from train.sft_schema import SCENARIOS, sha256_file  # noqa: E402
from train.train_sft import (  # noqa: E402
    EvalRuntimeStop,
    HorizonScheduler,
    RunStop,
    append_jsonl,
    collate_example,
    current_lr,
    environment_report,
    evaluate_val,
    load_adapter,
    load_audio,
    load_resume_state,
    load_rng,
    load_sft_model,
    log_event,
    move_batch,
    partial_decode_stop_record,
    prefix_text,
    prepare_checkpoint,
    prune_old_checkpoints,
    read_jsonl,
    resolve_from_repo,
    resource_report,
    set_seeds,
    supervised_nll_sum,
    sync_drive,
    sync_drive_files,
    thinker_inputs,
    validate_rows,
    write_latest,
    write_status,
)

RUN_ID_PREFIX = "_dpo_06b_zh_seed"
STEP0_ABS_LIMIT = 1.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练 Qwen3-ASR-0.6B 的 DPO LoRA")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--pair-train", type=Path)
    parser.add_argument("--pair-val", type=Path)
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--reference-report", type=Path)
    parser.add_argument("--pair-metadata", type=Path)
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--audio-root", required=True, type=Path)
    parser.add_argument("--null-pairs-dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def _require_sha(path: Path, expected: str, label: str) -> None:
    digest = sha256_file(path)
    if digest != expected:
        raise RunStop("data_failed", f"{label} sha256 {digest} does not match the config", 4)


def _check_null_pairs(directory: Path, config: dict[str, Any]) -> None:
    dpo = config["dpo"]
    for name, key in (
        ("train_pairs.jsonl", "null_train_pairs_sha256"),
        ("val_pairs.jsonl", "null_val_pairs_sha256"),
    ):
        path = directory / name
        if not path.is_file():
            raise RunStop("data_failed", f"missing null pair file {path}", 4)
        _require_sha(path, dpo[key], name)
        first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        if first.get("ref_chosen_logp") is not None or first.get("ref_rejected_logp") is not None:
            raise RunStop("data_failed", f"{name} no longer has null reference logprobs", 4)
    audit_path = directory / "pair_audit.json"
    if audit_path.is_file():
        try:
            assert_audit_still_unlistened(json.loads(audit_path.read_text(encoding="utf-8")))
        except ValueError as exc:
            raise RunStop("data_failed", str(exc), 4) from exc


def _check_metadata(path: Path, config: dict[str, Any], release: dict[str, Any]) -> str:
    pinned = config["dpo"].get("pair_metadata_sha256")
    if pinned:
        _require_sha(path, pinned, "pair metadata")
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        check_pair_metadata(payload, config, release)
    except ValueError as exc:
        raise RunStop("data_failed", str(exc), 4) from exc
    return sha256_file(path)


def _check_cache(path: Path, train_pairs: list[dict[str, Any]], val_pairs: list[dict[str, Any]]) -> None:
    cached = completed_logprob_rows(path, ordered_pairs(train_pairs, val_pairs))
    expected = ordered_pairs(train_pairs, val_pairs)
    if len(cached) != len(expected):
        raise RunStop("data_failed", "reference cache does not cover every pair", 4)
    for pair, saved in zip(expected, cached):
        for key in ("ref_chosen_logp", "ref_rejected_logp"):
            if float(pair[key]) != float(saved[key]):
                raise RunStop("data_failed", f"{pair['pair_id']} {key} does not match the cache", 4)


def _example(
    processor: Any,
    audio: Any,
    pair_id: str,
    text: str,
    *,
    prefix: str,
    eos_token: str,
    eos_id: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    try:
        return collate_example(
            processor,
            {"sample_id": pair_id, "text": text},
            audio,
            prefix=prefix,
            eos_token=eos_token,
            eos_id=eos_id,
            max_input_tokens=int(config["data"]["max_input_tokens"]),
            max_transcript_tokens=int(config["data"]["max_transcript_tokens"]),
        )
    except ValueError as exc:
        raise RunStop("data_failed", str(exc), 4) from exc


def _supervised_label_list(labels: Any) -> list[int]:
    shift_labels = labels[:, 1:].reshape(-1)
    return [int(value) for value in shift_labels.tolist() if int(value) != -100]


def _supervised_nll_with_mask(logits: Any, labels: Any, token_mask: Any | None = None) -> tuple[Any, int]:
    import torch

    if token_mask is None:
        return supervised_nll_sum(logits, labels)
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    nll = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    )
    sup_mask = shift_labels.reshape(-1) != -100
    count = int(sup_mask.sum().item())
    if count == 0:
        raise ValueError("microbatch has no supervised tokens")
    sup_nll = nll[sup_mask]
    if int(token_mask.numel()) != count:
        raise ValueError("token_mask length does not match supervised token count")
    full_nll = sup_nll.sum()
    diff_nll = sup_nll[token_mask].sum()
    return diff_nll + (full_nll - diff_nll).detach(), count


def _sequence_logp(model: Any, batch: dict[str, Any], token_mask: Any | None = None) -> tuple[Any, int]:
    outputs = model.thinker(**thinker_inputs(batch), use_cache=False)
    nll_sum, count = _supervised_nll_with_mask(outputs.logits, batch["labels"], token_mask=token_mask)
    return -nll_sum, count


def _score_text(
    model: Any,
    processor: Any,
    row: dict[str, Any],
    text: str,
    audio: Any,
    *,
    prefix: str,
    eos_token: str,
    eos_id: int,
    config: dict[str, Any],
    device: Any,
    dtype: Any,
    token_mask: Any | None = None,
) -> tuple[Any, float, int]:
    example = _example(
        processor,
        audio,
        str(row["pair_id"]),
        text,
        prefix=prefix,
        eos_token=eos_token,
        eos_id=eos_id,
        config=config,
    )
    batch = move_batch(dict(example), device, dtype)
    logp, count = _sequence_logp(model, batch, token_mask=token_mask)
    value = float(logp.detach().float().item())
    return logp, value, count


def _load_audio_for_pair(row: dict[str, Any], audio_root: Path, config: dict[str, Any]) -> Any:
    from inference.run_inference import resolve_audio_path

    audio = load_audio(resolve_audio_path(audio_root / "placeholder.jsonl", str(row["audio"])))
    if audio.shape[0] / 16000.0 > float(config["data"]["max_audio_seconds"]):
        raise RunStop("data_failed", f"{row['pair_id']} is longer than max_audio_seconds", 4)
    return audio


def write_run_report(run_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# DPO run",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- status: `{summary['status']}`",
        f"- reason: `{summary['reason']}`",
        f"- global_step: `{summary['global_step']}`",
        f"- publishable: `{summary['publishable']}`",
        f"- listen_waiver: `{summary.get('listen_waiver')}`",
        f"- config_sha256: `{summary.get('config_sha256')}`",
        f"- git_sha: `{summary.get('git_sha')}`",
        f"- dirty: `{summary.get('dirty')}`",
        f"- sync: `{summary.get('sync')}`",
        "",
        str(summary.get("note") or "publishable 保持 false。听音处置仍是 unlistened。"),
        "",
    ]
    atomic_write_text(run_dir / "run_report.md", "\n".join(lines))


def write_gate(
    run_dir: Path,
    config: dict[str, Any],
    val_sha: str,
    decisions: list[dict[str, Any]] | None = None,
    selected: dict[str, Any] | None = None,
    gate_status: str = "incomparable",
) -> None:
    payload = {
        "schema_version": "qwen3-asr-gate-v1",
        "status": gate_status,
        "publishable": False,
        "profile": config["profile"],
        "reason": (
            "dropout probe does not select a parent. "
            "listen dispositions stay unlistened and this pilot has no passed DPO smoke"
            if config["profile"] == "dropout_probe"
            else "listen dispositions stay unlistened and this pilot has no passed DPO smoke"
        ),
        "listen_waiver": config["dpo"]["listen_waiver"],
        "audit_disposition": "unlistened",
        "dpo_smoke_passed": False,
        "val_manifest_sha256": val_sha,
        "baseline_sha256": config["baseline"]["sha256"],
        "parent_metrics_sha256": config["dpo"]["parent"]["sha256"],
        "selected_step": None if selected is None else int(selected["step"]),
        "decisions": decisions or [],
    }
    atomic_write_json(run_dir / "gate.json", payload)


def run_training(args: argparse.Namespace, config: dict[str, Any], resume_state: dict[str, Any] | None) -> int:
    import torch

    run_dir = resolve_from_repo(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0 if resume_state is None else int(resume_state["global_step"])
    try:
        if config["runtime"]["deterministic_algorithms"]:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        set_seeds(int(config["seed"]))
        write_status(run_dir, "preflight", "starting", global_step)
        log_event(run_dir, "preflight_started", run_id=args.run_id, resume=bool(args.resume))
        atomic_write_json(
            run_dir / "listen_waiver.json",
            listen_waiver_record(config["dpo"].get("launch_utterance")),
        )
        info = environment_report(config)
        if info.get("dirty") and not config["runtime"]["allow_dirty"]:
            raise RunStop("failed", "worktree is dirty and allow_dirty is false", 2)
        atomic_write_json(run_dir / "environment.json", info)
        report = resource_report(config, run_dir)
        atomic_write_json(run_dir / "resource_report.json", report)

        train_manifest = resolve_from_repo(Path(config["data"]["train_manifest"]))
        val_manifest = resolve_from_repo(Path(config["data"]["val_manifest"]))
        metadata_path = resolve_from_repo(Path(config["data"]["manifest_metadata"]))
        train_rows = read_jsonl(train_manifest)
        val_rows = read_jsonl(val_manifest)
        validate_rows(train_rows, "train")
        validate_rows(val_rows, "val")
        pair_train = read_jsonl(resolve_from_repo(Path(config["dpo"]["pair_train_manifest"])))
        pair_val = read_jsonl(resolve_from_repo(Path(config["dpo"]["pair_val_manifest"])))
        min_per_scenario = int(config["dpo"]["min_pairs_per_scenario"])
        train_counts = validate_pair_rows(pair_train, split="train", min_per_scenario=min_per_scenario)
        val_counts = validate_pair_rows(pair_val, split="val", min_per_scenario=min_per_scenario)
        _require_sha(
            resolve_from_repo(Path(config["dpo"]["pair_train_manifest"])),
            config["dpo"]["filled_train_pairs_sha256"],
            "filled train pairs",
        )
        _require_sha(
            resolve_from_repo(Path(config["dpo"]["pair_val_manifest"])),
            config["dpo"]["filled_val_pairs_sha256"],
            "filled val pairs",
        )
        cache_path = resolve_from_repo(Path(config["dpo"]["reference_cache"]))
        report_path = resolve_from_repo(Path(config["dpo"]["reference_report"]))
        _require_sha(cache_path, config["dpo"]["reference_cache_sha256"], "reference cache")
        _require_sha(report_path, config["dpo"]["reference_report_sha256"], "reference report")
        ref_report = json.loads(report_path.read_text(encoding="utf-8"))
        if ref_report.get("status") != "reference_cached":
            raise RunStop("data_failed", "reference report is not reference_cached", 4)
        _check_cache(cache_path, pair_train, pair_val)
        # The cache check above sees the full pair files. Filtering stays in memory.
        scenarios = config["dpo"]["train_scenarios"]
        val_scenarios = config["dpo"].get("val_scenarios")
        if val_scenarios is None:
            val_scenarios = scenarios
        exclude_clean_identical = bool(config["dpo"].get("exclude_clean_identical_rejected", False))
        clean_rejected_map = build_clean_rejected_map(pair_train) if exclude_clean_identical else None
        max_edit_count = config["dpo"].get("max_edit_count")
        pair_filter_kwargs = {
            "aligned_substitution_only": bool(config["dpo"].get("aligned_substitution_only", False)),
            "ref_margin_min": config["dpo"].get("ref_margin_min"),
            "ref_margin_max": config["dpo"].get("ref_margin_max"),
        }
        train_filter_kwargs = {
            **pair_filter_kwargs,
            "max_edit_count": max_edit_count,
            "exclude_clean_identical_rejected": exclude_clean_identical,
            "clean_rejected_by_utterance": clean_rejected_map,
        }
        train_pair_rows = select_training_pairs(pair_train, scenarios, **train_filter_kwargs)
        pref_rows = select_training_pairs(pair_val, val_scenarios, **pair_filter_kwargs)
        if (
            scenarios is not None
            or val_scenarios is not None
            or exclude_clean_identical
            or max_edit_count is not None
            or any(value is not None and value is not False for value in pair_filter_kwargs.values())
        ):
            required_train_scenarios = tuple(scenarios) if scenarios is not None else SCENARIOS
            required_val_scenarios = tuple(val_scenarios) if val_scenarios is not None else SCENARIOS
            train_counts = validate_pair_rows(
                train_pair_rows,
                split="train",
                min_per_scenario=min_per_scenario,
                required_scenarios=required_train_scenarios,
            )
            val_counts = validate_pair_rows(
                pref_rows,
                split="val",
                min_per_scenario=min_per_scenario,
                required_scenarios=required_val_scenarios,
            )
        if len(train_pair_rows) < int(config["dpo"]["min_train_pairs"]) or len(pref_rows) < int(config["gate"]["min_val_pairs"]):
            raise RunStop("data_failed", "pair count is below the DPO floor", 4)
        scenario_label = "all" if scenarios is None else ",".join(scenarios)
        val_scenario_label = "all" if val_scenarios is None else ",".join(val_scenarios)
        print(
            f"DPO_TRAIN_SCENARIOS {scenario_label} val_scenarios={val_scenario_label} "
            f"train_pairs={len(train_pair_rows)} val_pairs={len(pref_rows)}",
            flush=True,
        )
        anchor_weight = float(config["dpo"].get("dropout_anchor_weight", 0.0))
        anchor_samples_per_step = int(config["dpo"].get("dropout_anchor_samples_per_step", 1))
        anchor_rows = select_dropout_anchor_rows(pair_train) if anchor_weight > 0.0 else []
        if anchor_weight > 0.0 and not anchor_rows:
            raise RunStop("data_failed", "dropout anchor has no parent hypotheses", 4)
        if anchor_rows:
            samples_note = "" if anchor_samples_per_step == 1 else f" samples_per_step={anchor_samples_per_step}"
            print(
                f"DPO_ANCHOR rows={len(anchor_rows)} weight={anchor_weight:.8g} target=rejected_text{samples_note}",
                flush=True,
            )
        release = load_published_release(resolve_from_repo(args.release_dir))
        if release.get("release_id") != config["model"]["parent_release"]:
            raise RunStop("failed", "release_id does not match the config", 2)
        if release.get("adapter_sha256") != config["model"]["parent_release_sha256"]:
            raise RunStop("failed", "parent_release_sha256 does not match the published adapter", 2)
        if int(release.get("source_step", -1)) != 30:
            raise RunStop("failed", "DPO parent release is not step 30", 2)
        metadata_sha = _check_metadata(resolve_from_repo(Path(config["dpo"]["pair_metadata"])), config, release)
        if args.null_pairs_dir is not None:
            _check_null_pairs(resolve_from_repo(args.null_pairs_dir), config)
        manifest_sha = {
            "train": sha256_file(train_manifest),
            "val": sha256_file(val_manifest),
            "metadata": sha256_file(metadata_path),
            "pair_metadata": metadata_sha,
            "filled_train_pairs": config["dpo"]["filled_train_pairs_sha256"],
            "filled_val_pairs": config["dpo"]["filled_val_pairs_sha256"],
            "reference_cache": config["dpo"]["reference_cache_sha256"],
        }
        config_sha = semantic_dpo_config_sha256(config, manifest_sha)
        if config["profile"] in {"pilot", "dropout_probe"}:
            drive_root = Path(os.environ.get("QWEN3_ASR_DRIVE_ROOT", "/content/drive/MyDrive/qwen3-asr"))
            if not drive_root.is_dir():
                raise RunStop("failed", f"{config['profile']} requires a mounted Drive root", 2)
            baseline = config["baseline"]
            if manifest_sha["val"] != baseline["manifest_sha256"]:
                raise RunStop("failed", "val manifest does not match the pinned base", 2)
            if config["decode_eval_sha256"] != baseline["decode_eval_sha256"]:
                raise RunStop("failed", "decode eval hash does not match the pinned base", 2)
        if resume_state is not None and resume_state["config_sha256"] != config_sha:
            raise RunStop("resume_rejected", "semantic config hash does not match the checkpoint", 2)
        if resume_state is not None and resume_state["manifest_sha256"] != manifest_sha:
            raise RunStop("resume_rejected", "manifest hash does not match the checkpoint", 2)

        model, processor, lora_report = load_sft_model(config)
        load_adapter(model, resolve_from_repo(args.release_dir) / "adapter" / "adapter.safetensors")
        atomic_write_json(run_dir / "trainable_parameters.json", lora_report)
        print(f"DPO_LORA trainable={lora_report['trainable_numel']} total={lora_report['total_numel']}", flush=True)
        eos_token = processor.tokenizer.eos_token
        eos_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if not isinstance(eos_token, str) or not isinstance(eos_id, int):
            raise RunStop("failed", "tokenizer did not provide an EOS token", 2)
        prefix = prefix_text(processor, config["decode"]["eval"]["context"], config["official_language"])
        names = [name for name, param in model.named_parameters() if param.requires_grad]
        if resume_state is not None and names != resume_state["trainable_names"]:
            raise RunStop("resume_rejected", "trainable parameter names changed", 2)
        device = next(model.parameters()).device
        dtype = torch.bfloat16
        audio_root = resolve_from_repo(args.audio_root)
        beta = float(config["dpo"]["beta"])

        loss_token_scope = str(config["dpo"].get("loss_token_scope", "sequence"))

        def score_pair_side(row: dict[str, Any], text: str, audio: Any, *, keep_graph: bool) -> tuple[Any | None, float, int]:
            logp, value, count = _score_text(
                model,
                processor,
                row,
                text,
                audio,
                prefix=prefix,
                eos_token=eos_token,
                eos_id=int(eos_id),
                config=config,
                device=device,
                dtype=dtype,
            )
            if keep_graph:
                return logp, value, count
            del logp
            model.zero_grad(set_to_none=True)
            return None, value, count

        def score_pair_for_training(row: dict[str, Any], audio: Any) -> tuple[Any, Any, int, int]:
            if loss_token_scope != "mismatched_only":
                chosen_logp, _, chosen_count = score_pair_side(row, str(row["chosen_text"]), audio, keep_graph=True)
                rejected_logp, _, rejected_count = score_pair_side(
                    row, str(row["rejected_text"]), audio, keep_graph=True
                )
                return chosen_logp, rejected_logp, chosen_count, rejected_count
            chosen_example = _example(
                processor,
                audio,
                str(row["pair_id"]),
                str(row["chosen_text"]),
                prefix=prefix,
                eos_token=eos_token,
                eos_id=int(eos_id),
                config=config,
            )
            rejected_example = _example(
                processor,
                audio,
                str(row["pair_id"]),
                str(row["rejected_text"]),
                prefix=prefix,
                eos_token=eos_token,
                eos_id=int(eos_id),
                config=config,
            )
            try:
                mask_list = mismatched_token_mask(
                    _supervised_label_list(chosen_example["labels"]),
                    _supervised_label_list(rejected_example["labels"]),
                )
            except ValueError as exc:
                raise RunStop("data_failed", f"{row['pair_id']} {exc}", 4) from exc
            token_mask = torch.tensor(mask_list, dtype=torch.bool, device=device)
            chosen_batch = move_batch(dict(chosen_example), device, dtype)
            chosen_logp, chosen_count = _sequence_logp(model, chosen_batch, token_mask=token_mask)
            rejected_batch = move_batch(dict(rejected_example), device, dtype)
            rejected_logp, rejected_count = _sequence_logp(model, rejected_batch, token_mask=token_mask)
            return chosen_logp, rejected_logp, chosen_count, rejected_count

        if resume_state is None:
            probe = train_pair_rows[0]
            probe_audio = _load_audio_for_pair(probe, audio_root, config)
            model.train()
            chosen_logp, rejected_logp, chosen_count, rejected_count = score_pair_for_training(probe, probe_audio)
            if chosen_count != int(probe["chosen_token_count"]) or rejected_count != int(probe["rejected_token_count"]):
                raise RunStop("data_failed", "probe token count does not match the pair", 4)
            delta = beta * (
                (chosen_logp - rejected_logp)
                - (float(probe["ref_chosen_logp"]) - float(probe["ref_rejected_logp"]))
            )
            probe_loss = torch.nn.functional.softplus(-delta)
            if not torch.isfinite(probe_loss):
                raise RunStop("failed", "probe DPO loss is not finite", 1)
            probe_loss.backward()
            saw_grad = any(
                param.grad is not None and torch.isfinite(param.grad).all()
                for param in model.parameters()
                if param.requires_grad
            )
            if not saw_grad:
                raise RunStop("failed", "probe produced no finite LoRA gradient", 2)
            model.zero_grad(set_to_none=True)
            set_seeds(int(config["seed"]))
            checked = select_recheck(pair_val, int(config["seed"]), 20)
            max_abs = 0.0
            model.train()
            for row in checked:
                audio = _load_audio_for_pair(row, audio_root, config)
                for text_key, ref_key, count_key in (
                    ("chosen_text", "ref_chosen_logp", "chosen_token_count"),
                    ("rejected_text", "ref_rejected_logp", "rejected_token_count"),
                ):
                    _, value, count = score_pair_side(row, str(row[text_key]), audio, keep_graph=False)
                    if count != int(row[count_key]):
                        raise RunStop("data_failed", f"{row['pair_id']} token count drifted", 4)
                    absolute, _relative = logprob_errors(float(row[ref_key]), value)
                    max_abs = max(max_abs, absolute)
            print(f"DPO_STEP0 checked={len(checked)} max_abs={max_abs:.8g}", flush=True)
            log_event(run_dir, "step0_identity", checked=len(checked), max_abs=max_abs)
            if max_abs > STEP0_ABS_LIMIT:
                raise RunStop("failed", f"step-0 policy logprob differs from the cache by {max_abs}", 1)
            print("DPO_PREFLIGHT_OK", flush=True)
            log_event(run_dir, "preflight_ok", train_counts=train_counts, val_counts=val_counts)

        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=float(config["optim"]["learning_rate"]),
            betas=tuple(config["optim"]["betas"]),
            eps=float(config["optim"]["eps"]),
            weight_decay=float(config["optim"]["weight_decay"]),
        )
        scheduler = HorizonScheduler(optimizer, config)
        if resume_state is not None:
            checkpoint = run_dir / "checkpoints" / step_name(global_step)
            load_adapter(model, checkpoint / "adapter" / "adapter.safetensors")
            optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location=device, weights_only=False))
            scheduler.load_state_dict(torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=False))
            load_rng(checkpoint / "rng_state.pt")
            print(f"DPO_RESUME global_step={global_step}", flush=True)
            log_event(run_dir, "resume_ok", global_step=global_step)
        else:
            load_adapter(model, resolve_from_repo(args.release_dir) / "adapter" / "adapter.safetensors")
        interrupt_consumed = bool(resume_state and resume_state.get("interrupt_consumed"))
        samples_seen = 0 if resume_state is None else int(resume_state["samples_seen"])
        resolved_yaml = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
        atomic_write_text(run_dir / "resolved_config.yaml", resolved_yaml)
        atomic_write_json(run_dir / "manifest_sha256.json", manifest_sha)
        write_status(run_dir, "running", "optimizer_ready", global_step)
        scheduled_updates = list(
            iterate_updates(
                train_pair_rows,
                seed=int(config["seed"]),
                accumulation=int(config["batch"]["gradient_accumulation_steps"]),
                max_steps=int(config["train"]["max_steps"]),
                max_epochs=int(config["train"]["max_epochs"]),
            )
        )
        if not scheduled_updates:
            raise RunStop("failed", "sampler produced no updates", 1)
        schedule_end_step = int(scheduled_updates[-1]["global_step"])
        if resume_state is not None:
            matched = False
            for update in scheduled_updates:
                if int(update["global_step"]) == global_step:
                    if int(update["epoch"]) != int(resume_state["epoch"]) or int(update["cursor_after"]) != int(
                        resume_state["cursor_after"]
                    ):
                        raise RunStop("resume_rejected", "sampler cursor does not match the checkpoint", 2)
                    matched = True
                    break
            if not matched:
                raise RunStop("resume_rejected", "checkpoint step is outside this sampler schedule", 2)
        started = time.monotonic()
        last_saved = global_step if resume_state is not None else 0
        sync_result = "not_saved"
        stop_reason = "max_steps"
        decisions: list[dict[str, Any]] = []
        decision_path = run_dir / "eval_decisions.jsonl"
        if resume_state is not None and decision_path.is_file():
            decisions = [json.loads(line) for line in decision_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        budget_disabled = runtime_budget_disabled(os.environ.get("QWEN3_ASR_DISABLE_RUNTIME_BUDGET"))
        if budget_disabled:
            print("DPO_RUNTIME_BUDGET disabled", flush=True)
            log_event(run_dir, "runtime_budget_disabled")

        def budget_exceeded() -> bool:
            if budget_disabled:
                return False
            return runtime_budget_exceeded(
                time.monotonic() - started,
                float(config["runtime"]["max_runtime_minutes"]),
                float(config["runtime"]["shutdown_margin_minutes"]),
            )

        def pause_for_runtime(record: dict[str, Any] | None) -> int:
            step_to_keep = last_saved
            if step_to_keep <= 0:
                raise RunStop("failed", "runtime budget ended before the first checkpoint", 1)
            files: list[tuple[Path, str]] = []
            if record is not None:
                relative = f"eval/{step_name(step_to_keep)}/runtime_stop.json"
                destination = run_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(destination, record)
                files.append((destination, relative))
            write_status(run_dir, "paused", "max_runtime", step_to_keep)
            files.append((run_dir / "status.json", "status.json"))
            if decision_path.is_file():
                files.append((decision_path, "eval_decisions.jsonl"))
            pause_sync = sync_drive_files(args.run_id, files)
            write_run_report(
                run_dir,
                {
                    "run_id": args.run_id,
                    "status": "paused",
                    "reason": "max_runtime",
                    "global_step": step_to_keep,
                    "publishable": False,
                    "listen_waiver": config["dpo"]["listen_waiver"],
                    "config_sha256": config_sha,
                    "git_sha": info.get("git_sha"),
                    "dirty": info.get("dirty"),
                    "sync": pause_sync,
                    "note": "运行时间达到上限。停在最后一个已保存的 checkpoint。",
                },
            )
            sync_drive_files(args.run_id, [(run_dir / "run_report.md", "run_report.md")])
            print(f"DPO_RUNTIME_STOP step={step_to_keep}", flush=True)
            print("DPO_EXIT 0", flush=True)
            return 0

        def preference_eval(step: int) -> dict[str, Any]:
            from train.dpo_contract import length_buckets, pair_dpo_loss, preference_accuracy, raw_policy_preference

            chosen_values: list[float] = []
            rejected_values: list[float] = []
            ref_chosen: list[float] = []
            ref_rejected: list[float] = []
            deltas_length: list[int] = []
            model.train()
            for index, row in enumerate(pref_rows, start=1):
                audio = _load_audio_for_pair(row, audio_root, config)
                _, chosen_value, chosen_count = score_pair_side(row, str(row["chosen_text"]), audio, keep_graph=False)
                _, rejected_value, rejected_count = score_pair_side(
                    row, str(row["rejected_text"]), audio, keep_graph=False
                )
                if chosen_count != int(row["chosen_token_count"]) or rejected_count != int(row["rejected_token_count"]):
                    raise RunStop("data_failed", f"{row['pair_id']} token count drifted at step {step}", 4)
                chosen_values.append(chosen_value)
                rejected_values.append(rejected_value)
                ref_chosen.append(float(row["ref_chosen_logp"]))
                ref_rejected.append(float(row["ref_rejected_logp"]))
                deltas_length.append(int(row["char_length_delta"]))
                if index == 1 or index % 250 == 0 or index == len(pref_rows):
                    print(f"DPO_PREF_PROGRESS step={step} {index}/{len(pref_rows)}", flush=True)
            _loss, deltas = pair_dpo_loss(chosen_values, rejected_values, ref_chosen, ref_rejected, beta)
            summary = preference_accuracy(deltas)
            summary["raw_policy_accuracy"] = raw_policy_preference(chosen_values, rejected_values)
            summary["length_buckets"] = length_buckets(deltas_length, chosen_values, rejected_values)
            summary["mean_loss"] = _loss
            destination = run_dir / "eval" / step_name(step) / "preference.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(destination, summary)
            return summary

        def run_eval(step: int) -> str:
            if budget_exceeded():
                pause_for_runtime(
                    partial_decode_stop_record(step, len(val_rows), [], int(config["decode"]["eval"]["max_new_tokens"]))
                )
                return "paused"
            preference = preference_eval(step)
            adapter_sha = sha256_file(run_dir / "checkpoints" / step_name(step) / "adapter" / "adapter.safetensors")
            try:
                metrics = evaluate_val(
                    model,
                    processor,
                    val_rows,
                    val_manifest,
                    config,
                    run_dir,
                    step,
                    adapter_sha,
                    deadline=budget_exceeded,
                )
            except EvalRuntimeStop as exc:
                pause_for_runtime(exc.record)
                return "paused"
            scored_path = run_dir / "eval" / step_name(step) / "scored.jsonl"
            locate = locate_dropout_edits(read_jsonl(scored_path), metrics, config["dpo"]["parent"])
            decision = judge_dpo_checkpoint(
                metrics,
                config["dpo"]["parent"],
                config["baseline"],
                config["gate"],
                preference,
            )
            decision["step"] = step
            decision.update(locate)
            decisions.append(decision)
            append_jsonl(decision_path, decision)
            sync_drive_files(
                args.run_id,
                [
                    (decision_path, "eval_decisions.jsonl"),
                    (scored_path, f"eval/{step_name(step)}/scored.jsonl"),
                ],
            )
            print(
                f"DPO_LOCATE signature={locate['locate_signature']} "
                f"gap_insertions={locate['locate_gap_insertions']} "
                f"dropout_length_ratio={locate['locate_dropout_length_ratio']:.6f} "
                f"dropout_delta={locate['locate_dropout_delta']:.6f} "
                f"clean_delta={decision['clean_delta_vs_parent']:.6f}",
                flush=True,
            )
            print(
                f"DPO_GATE step={step} passed={decision['passed']} risk_stop={decision['risk_stop']} "
                f"preference={decision['preference_accuracy']:.6f}",
                flush=True,
            )
            log_event(run_dir, "eval_written", global_step=step, preference=decision["preference_accuracy"])
            if decision["risk_stop"]:
                return "gate_risk"
            return "ok"

        decided_steps = [int(item["step"]) for item in decisions if "step" in item]
        if resume_state is not None and pending_checkpoint_eval(
            global_step,
            int(config["train"]["eval_steps"]),
            decided_steps,
            schedule_end_step=schedule_end_step,
        ):
            pending_result = run_eval(global_step)
            if pending_result == "paused":
                return 0
            if pending_result == "gate_risk":
                stop_reason = "gate_risk"
        training_updates = (
            []
            if stop_reason == "gate_risk"
            else [update for update in scheduled_updates if int(update["global_step"]) > global_step]
        )
        for update in training_updates:
            if budget_exceeded():
                return pause_for_runtime(None)
            step = int(update["global_step"])
            optimizer.zero_grad(set_to_none=True)
            model.train()
            step_started = time.monotonic()
            pair_losses: list[float] = []
            pair_count = len(update["rows"])
            for row in update["rows"]:
                audio = _load_audio_for_pair(row, audio_root, config)
                chosen_logp, rejected_logp, chosen_count, rejected_count = score_pair_for_training(row, audio)
                if chosen_count != int(row["chosen_token_count"]) or rejected_count != int(row["rejected_token_count"]):
                    raise RunStop("data_failed", f"{row['pair_id']} token count drifted", 4)
                delta = beta * (
                    (chosen_logp - rejected_logp)
                    - (float(row["ref_chosen_logp"]) - float(row["ref_rejected_logp"]))
                )
                loss = torch.nn.functional.softplus(-delta)
                if not torch.isfinite(loss):
                    raise RunStop("failed", f"non-finite DPO loss at global_step {step}", 1)
                (loss / pair_count).backward()
                pair_losses.append(float(loss.detach().float().item()))
            anchor_value = 0.0
            anchor_tokens = 0
            if anchor_rows:
                step_anchor_rows = dropout_anchor_step_slice(
                    anchor_rows,
                    step,
                    anchor_samples_per_step,
                )
                anchor_scale = (
                    anchor_weight
                    if len(step_anchor_rows) == 1
                    else (anchor_weight / len(step_anchor_rows))
                )
                for anchor_row in step_anchor_rows:
                    anchor_audio = _load_audio_for_pair(anchor_row, audio_root, config)
                    anchor_logp, _, row_anchor_tokens = score_pair_side(
                        anchor_row,
                        str(anchor_row["rejected_text"]),
                        anchor_audio,
                        keep_graph=True,
                    )
                    if row_anchor_tokens != int(anchor_row["rejected_token_count"]):
                        raise RunStop(
                            "data_failed",
                            f"{anchor_row['pair_id']} anchor token count drifted",
                            4,
                        )
                    anchor_loss = anchor_scale * (-anchor_logp) / row_anchor_tokens
                    if not torch.isfinite(anchor_loss):
                        raise RunStop("failed", f"non-finite dropout anchor loss at global_step {step}", 1)
                    anchor_loss.backward()
                    anchor_value += float(anchor_loss.detach().float().item())
                    anchor_tokens += int(row_anchor_tokens)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad],
                float(config["optim"]["max_grad_norm"]),
            )
            if not torch.isfinite(grad_norm):
                raise RunStop("failed", f"non-finite grad norm at global_step {step}", 1)
            # The scheduler is born at multiplier 0. Advance it before Adam so this
            # update and the logged lr both use lr_multiplier(step).
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            global_step = step
            samples_seen += pair_count
            loss_value = sum(pair_losses) / pair_count
            append_jsonl(
                run_dir / "metrics.jsonl",
                {
                    "global_step": step,
                    "micro_step": samples_seen,
                    "samples_seen": samples_seen,
                    "group_visits": 0,
                    "loss": loss_value,
                    "grad_norm": float(grad_norm),
                    "lr": current_lr(optimizer),
                    "pairs": pair_count,
                    "anchor_loss": anchor_value,
                    "anchor_tokens": anchor_tokens,
                    "elapsed_s": time.monotonic() - step_started,
                    "gpu_free_bytes": int(torch.cuda.mem_get_info()[0]),
                },
            )
            anchor_note = ""
            if anchor_rows:
                anchor_note = f" anchor_loss={anchor_value:.6f} anchor_tokens={anchor_tokens}"
            print(
                f"DPO_STEP global_step={step}/{config['train']['max_steps']} loss={loss_value:.6f} "
                f"grad_norm={float(grad_norm):.6f} lr={current_lr(optimizer):.8g} "
                f"pairs={pair_count}{anchor_note}",
                flush=True,
            )
            interrupt = should_interrupt(step, config["train"]["interrupt_after_step"], interrupt_consumed)
            if interrupt:
                interrupt_consumed = True
            ends_schedule = update_ends_schedule(
                update,
                row_count=len(train_pair_rows),
                max_steps=int(config["train"]["max_steps"]),
                max_epochs=int(config["train"]["max_epochs"]),
            )
            need_save = (
                step % int(config["train"]["save_steps"]) == 0
                or step == int(config["train"]["max_steps"])
                or interrupt
                or ends_schedule
            )
            if need_save:
                state = {
                    "schema_version": "qwen3-asr-training-state-v1",
                    "stage": "dpo",
                    "global_step": step,
                    "micro_step": samples_seen,
                    "samples_seen": samples_seen,
                    "group_visits": 0,
                    "accumulation_boundary": 0,
                    "epoch": int(update["epoch"]),
                    "cursor_after": int(update["cursor_after"]),
                    "config_sha256": config_sha,
                    "manifest_sha256": manifest_sha,
                    "eval_decode_sha256": config["decode_eval_sha256"],
                    "parent_release_sha256": config["model"]["parent_release_sha256"],
                    "publishable": False,
                    "interrupt_consumed": interrupt_consumed,
                    "trainable_names": names,
                    "profile": config["profile"],
                    "listen_waiver": config["dpo"]["listen_waiver"],
                }
                sampler_state = {
                    "schema_version": "qwen3-asr-sampler-v1",
                    "splits": {
                        "train": {
                            "epoch": int(update["epoch"]),
                            "cursor": int(update["cursor_after"]),
                            "shuffle_seed": int(config["seed"]),
                            "visit": samples_seen,
                        }
                    },
                }
                digest = prepare_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    run_dir=run_dir,
                    config=config,
                    state=state,
                    sampler_state=sampler_state,
                    resolved_yaml=resolved_yaml,
                )
                try:
                    sync_result = sync_drive(run_dir, args.run_id, step, digest)
                except RunStop:
                    write_status(run_dir, "sync_failed", "drive copy failed before latest.json", step)
                    raise
                write_latest(run_dir / "latest.json", step=step, checkpoint_sha256=digest)
                log_event(run_dir, "checkpoint_committed", global_step=step, sync=sync_result, sha256=digest)
                prune_old_checkpoints(run_dir, keep_last=int(config["train"]["keep_last"]), current_step=step)
                last_saved = step
                atomic_write_json(run_dir / "training_state.json", state)
            if need_save and (step % int(config["train"]["eval_steps"]) == 0 or ends_schedule):
                eval_result = run_eval(step)
                if eval_result == "paused":
                    return 0
                if eval_result == "gate_risk":
                    stop_reason = "gate_risk"
                    break
            if interrupt:
                print(f"DPO_INTERRUPT step={step}", flush=True)
                write_status(run_dir, "paused", "interrupt_after_step", step)
                write_gate(run_dir, config, manifest_sha["val"], decisions)
                print("DPO_EXIT 0", flush=True)
                return 0
        if global_step == 0 or last_saved != global_step:
            raise RunStop("failed", "run ended without a checkpoint at the last completed step", 1)
        chosen = select_sft_checkpoint(decisions)
        final_status = "completed" if chosen is not None else "gate_failed"
        gate_status = "passed" if chosen is not None else "failed"
        if stop_reason == "gate_risk" and chosen is not None:
            final_status = "completed"
            gate_status = "passed"
        elif stop_reason == "gate_risk":
            final_status = "gate_failed"
            gate_status = "failed"
        selected = None if config["profile"] == "dropout_probe" else chosen
        write_status(run_dir, final_status, stop_reason, global_step)
        write_gate(run_dir, config, manifest_sha["val"], decisions, selected=selected, gate_status=gate_status)
        note = (
            f"dropout probe locate_signature={decisions[-1].get('locate_signature')}。"
            "这一步不作为父版本。publishable 保持 false。听音处置仍是 unlistened。"
            if config["profile"] == "dropout_probe" and decisions
            else "publishable 保持 false。听音处置仍是 unlistened。DPO smoke 还没有单独通过。"
        )
        if config["dpo"].get("launch_utterance") == "开始吧":
            note += "2026-10-18 的启动语是「开始吧」。2026-10-17 的听音豁免不覆盖这批 gap-strip 抽查。"
        write_run_report(
            run_dir,
            {
                "run_id": args.run_id,
                "status": final_status,
                "reason": stop_reason,
                "global_step": global_step,
                "publishable": False,
                "listen_waiver": config["dpo"]["listen_waiver"],
                "config_sha256": config_sha,
                "git_sha": info.get("git_sha"),
                "dirty": info.get("dirty"),
                "sync": sync_result,
                "note": note,
            },
        )
        sync_drive_files(
            args.run_id,
            [
                (run_dir / "status.json", "status.json"),
                (run_dir / "gate.json", "gate.json"),
                (run_dir / "run_report.md", "run_report.md"),
                (run_dir / "listen_waiver.json", "listen_waiver.json"),
            ],
        )
        print(f"DPO_GATE_STATUS {gate_status}", flush=True)
        print(f"DPO_DONE global_step={global_step}", flush=True)
        print("DPO_EXIT 0", flush=True)
        return 0
    except RunStop as exc:
        write_status(run_dir, exc.status, str(exc), global_step)
        log_event(run_dir, exc.status, message=str(exc))
        print(f"DPO_EXIT {exc.code} {exc.status}: {exc}", file=sys.stderr, flush=True)
        return exc.code
    except torch.cuda.OutOfMemoryError as exc:
        write_status(run_dir, "oom", str(exc), global_step)
        log_event(run_dir, "oom", message=str(exc))
        print(f"DPO_EXIT 1 oom: {exc}", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:
        write_status(run_dir, "failed", f"{type(exc).__name__}: {exc}", global_step)
        log_event(run_dir, "failed", message=str(exc), traceback=traceback.format_exc())
        print(f"DPO_EXIT 1 failed: {exc}", file=sys.stderr, flush=True)
        return 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_dpo_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"config rejected: {exc}", file=sys.stderr)
        return 2
    if args.seed != int(config["seed"]):
        print(f"seed mismatch: cli {args.seed} config {config['seed']}", file=sys.stderr)
        return 2
    if config["runtime"]["deterministic_algorithms"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if re.fullmatch(rf"\d{{8}}T\d{{6}}{re.escape(RUN_ID_PREFIX)}{args.seed}", args.run_id) is None:
        print("run_id must look like YYYYMMDDTHHMMSS_dpo_06b_zh_seed<seed>", file=sys.stderr)
        return 2
    if args.manifest is not None:
        config["data"]["train_manifest"] = str(args.manifest)
    if args.val_manifest is not None:
        config["data"]["val_manifest"] = str(args.val_manifest)
    if args.pair_train is not None:
        config["dpo"]["pair_train_manifest"] = str(args.pair_train)
    if args.pair_val is not None:
        config["dpo"]["pair_val_manifest"] = str(args.pair_val)
    if args.reference_cache is not None:
        config["dpo"]["reference_cache"] = str(args.reference_cache)
    if args.reference_report is not None:
        config["dpo"]["reference_report"] = str(args.reference_report)
    if args.pair_metadata is not None:
        config["dpo"]["pair_metadata"] = str(args.pair_metadata)
    lock_sha = sha256_file(ROOT / "requirements.lock")
    if lock_sha != config["runtime"]["environment_lock_sha256"]:
        print("environment lock hash does not match requirements.lock", file=sys.stderr)
        return 2
    run_dir = resolve_from_repo(args.output_dir)
    resume_state = None
    if args.resume:
        try:
            resume_state = load_resume_state(run_dir)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            if run_dir.exists():
                write_status(run_dir, "resume_rejected", str(exc), 0)
            print(f"resume rejected: {exc}", file=sys.stderr)
            return 2
    elif run_dir.exists() and any(run_dir.iterdir()):
        print(f"output dir already exists: {run_dir}", file=sys.stderr)
        return 2
    for key in ("train_manifest", "val_manifest", "manifest_metadata"):
        path = resolve_from_repo(Path(config["data"][key]))
        if not path.is_file():
            print(f"missing {key}: {path}", file=sys.stderr)
            return 2
    for key in ("pair_train_manifest", "pair_val_manifest", "reference_cache", "reference_report", "pair_metadata"):
        path = resolve_from_repo(Path(config["dpo"][key]))
        if not path.is_file():
            print(f"missing {key}: {path}", file=sys.stderr)
            return 2
    if not resolve_from_repo(args.release_dir).is_dir():
        print(f"missing release dir: {args.release_dir}", file=sys.stderr)
        return 2
    return run_training(args, config, resume_state)


if __name__ == "__main__":
    raise SystemExit(main())
