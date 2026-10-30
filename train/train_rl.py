#!/usr/bin/env python3
"""Phase 4 RL runner (group_pg_mixed_v1) for Qwen3-ASR-0.6B on the published DPO adapter.

Implements 1 greedy + 3 sampled rollout groups (G=4), clipped CER-based rewards,
group-mean-centered advantages, full-group zeroing (including zero KL) when all
rewards in a group are equal, zero-signal window optimizer/scheduler skipping,
frozen reference LoRA checksum verification, and held-out exact conditional
vocabulary KL (raw_kl) on fixed step-0 greedy prefixes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Iterator

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.run_inference import (  # noqa: E402
    parse_generation,
    resolve_audio_path,
    stop_reason_from_ids,
)
from train.dpo_release import load_published_dpo_release  # noqa: E402
from train.rl_contract import (  # noqa: E402
    accumulate_window_groups,
    aligned_group_mismatch_mask,
    candidate_seed_int,
    evaluate_rollout_group,
    exact_step0_conditional_kl_tensor,
    group_loss_from_token_logprobs,
    judge_rl_checkpoint,
    mean_greedy_reward,
    prefix_records_sha256,
    reference_tensors_sha256,
)
from train.rl_schema import load_rl_config, semantic_rl_config_sha256  # noqa: E402
from train.sft_checkpoint import atomic_write_json, atomic_write_text, step_name, utc_now  # noqa: E402
from train.sft_contract import (  # noqa: E402
    epoch_order,
    pending_checkpoint_eval,
    runtime_budget_disabled,
    runtime_budget_exceeded,
    select_sft_checkpoint,
    should_interrupt,
)
from train.sft_schema import sha256_file  # noqa: E402
from train.train_sft import (  # noqa: E402
    EvalRuntimeStop,
    HorizonScheduler,
    RunStop,
    append_jsonl,
    collate_example,
    current_lr,
    decode_eval_rows,
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
    sync_drive,
    sync_drive_files,
    thinker_inputs,
    validate_rows,
    write_latest,
    write_status,
)

RUN_ID_PREFIX = "_rl_06b_zh_seed"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练 Qwen3-ASR-0.6B 的 RL LoRA (group_pg_mixed_v1)")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--release-dir", required=True, type=Path)
    parser.add_argument("--audio-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def snapshot_lora_tensors(model: Any) -> dict[str, Any]:
    """Clone all trainable LoRA tensors to CPU as an independent frozen reference."""
    snapshot: dict[str, Any] = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            snapshot[name] = param.detach().cpu().clone().contiguous()
    if not snapshot:
        raise ValueError("model has no trainable LoRA parameters to snapshot")
    return snapshot


@contextlib.contextmanager
def use_reference_adapter(model: Any, reference_lora: dict[str, Any]) -> Iterator[None]:
    """Temporarily swap frozen reference LoRA weights into ``model`` under no_grad/eval."""
    import torch

    was_training = bool(getattr(model, "training", False))
    policy_backup: dict[str, Any] = {}
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in reference_lora:
                policy_backup[name] = param.detach().clone()
                param.copy_(reference_lora[name].to(device=param.device, dtype=param.dtype))
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in policy_backup:
                    param.copy_(policy_backup[name])
        if was_training:
            model.train()


def _per_token_supervised_logprobs(logits: Any, labels: Any) -> tuple[Any, list[float]]:
    """Return 1D tensor of supervised token log-probabilities (FP32) and Python float list."""
    import torch

    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    nll = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    )
    mask = shift_labels.reshape(-1) != -100
    if int(mask.sum().item()) == 0:
        raise ValueError("candidate has no supervised tokens")
    token_logp = -nll[mask]
    values = [float(x) for x in token_logp.detach().cpu().tolist()]
    return token_logp, values


def generate_group_candidates(
    model: Any,
    processor: Any,
    row: dict[str, Any],
    audio: Any,
    *,
    config: dict[str, Any],
    epoch: int,
    visit_index: int,
    global_step: int,
    prefix: str,
    eos_id: int,
) -> list[dict[str, Any]]:
    """Generate G=4 candidates (1 greedy + 3 sampled) from the current policy."""
    import torch

    rollout_cfg = config["decode"]["rollout"]
    group_size = int(rollout_cfg["group_size"])
    greedy_count = int(rollout_cfg["greedy_candidates"])
    max_new_tokens = int(rollout_cfg["max_new_tokens"])
    max_transcript_tokens = int(config["data"]["max_transcript_tokens"])
    sample_id = str(row["sample_id"])
    device = next(model.parameters()).device
    dtype = torch.bfloat16
    raw_inputs = processor(text=[prefix], audio=[audio], return_tensors="pt", padding=True)
    inputs = thinker_inputs(move_batch(dict(raw_inputs), device, dtype))
    prompt_len = int(inputs["input_ids"].shape[1])
    was_training = bool(getattr(model, "training", False))
    model.eval()
    candidates: list[dict[str, Any]] = []
    try:
        for cand_idx in range(group_size):
            is_greedy = cand_idx < greedy_count
            cand_seed = candidate_seed_int(
                int(config["seed"]),
                epoch,
                visit_index,
                sample_id,
                cand_idx,
            )
            torch.manual_seed(cand_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cand_seed)
            gen_kwargs: dict[str, Any] = {
                **inputs,
                "do_sample": not is_greedy,
                "num_beams": 1,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": float(rollout_cfg.get("repetition_penalty", 1.0)),
                "eos_token_id": eos_id,
            }
            if not is_greedy:
                gen_kwargs["temperature"] = float(rollout_cfg["temperature"])
                gen_kwargs["top_p"] = float(rollout_cfg["top_p"])
                gen_kwargs["top_k"] = int(rollout_cfg["top_k"])
            try:
                with torch.no_grad():
                    generated = model.generate(**gen_kwargs)
                sequences = generated.sequences if hasattr(generated, "sequences") else generated
                token_ids = [int(t) for t in sequences[0, prompt_len:].detach().cpu().tolist()]
                stop_reason = stop_reason_from_ids(token_ids, eos_id, max_new_tokens)
                visible = processor.batch_decode(
                    torch.tensor([token_ids], dtype=torch.long),
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                prediction_raw = parse_generation(visible)
                reencoded = processor.tokenizer.encode(prediction_raw, add_special_tokens=False)
                if len(reencoded) > max_transcript_tokens:
                    error = "ValueError: candidate transcript exceeds max_transcript_tokens"
                else:
                    error = None
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as exc:
                token_ids = []
                stop_reason = "error"
                prediction_raw = ""
                error = f"{type(exc).__name__}: {exc}"
            candidates.append(
                {
                    "candidate_index": cand_idx,
                    "candidate_seed": cand_seed,
                    "is_greedy": is_greedy,
                    "token_ids": token_ids,
                    "stop_reason": stop_reason,
                    "prediction_raw": prediction_raw,
                    "error": error,
                    "decode_rollout_sha256": config["decode_rollout_sha256"],
                    "policy_snapshot_step": global_step,
                }
            )
    finally:
        if was_training:
            model.train()
    return candidates


def write_rl_run_report(run_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# RL run",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- status: `{summary['status']}`",
        f"- reason: `{summary['reason']}`",
        f"- global_step: `{summary['global_step']}`",
        f"- group_visits: `{summary.get('group_visits', 0)}`",
        f"- publishable: `{summary['publishable']}`",
        f"- config_sha256: `{summary.get('config_sha256')}`",
        f"- reference_sha256: `{summary.get('reference_sha256')}`",
        f"- step0_reward: `{summary.get('step0_reward')}`",
        f"- git_sha: `{summary.get('git_sha')}`",
        f"- dirty: `{summary.get('dirty')}`",
        f"- sync: `{summary.get('sync')}`",
        "",
        str(summary.get("note") or "publishable 保持 false。"),
        "",
    ]
    atomic_write_text(run_dir / "run_report.md", "\n".join(lines))


def write_rl_gate(
    run_dir: Path,
    config: dict[str, Any],
    val_sha: str,
    decisions: list[dict[str, Any]] | None = None,
    selected: dict[str, Any] | None = None,
    gate_status: str = "incomparable",
    reference_sha256: str | None = None,
    step0_prefix_sha256: str | None = None,
) -> None:
    payload = {
        "schema_version": "qwen3-asr-gate-v1",
        "stage": "rl",
        "status": gate_status,
        "publishable": False,
        "profile": config["profile"],
        "val_manifest_sha256": val_sha,
        "baseline_sha256": None if config.get("baseline") is None else config["baseline"]["sha256"],
        "parent_metrics_sha256": config["rl"]["parent"]["sha256"],
        "reference_sha256": reference_sha256,
        "step0_prefix_sha256": step0_prefix_sha256,
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
        info = environment_report(config)
        if info.get("dirty") and not config["runtime"]["allow_dirty"]:
            raise RunStop("failed", "worktree is dirty and allow_dirty is false", 2)
        atomic_write_json(run_dir / "environment.json", info)
        report = resource_report(config, run_dir)
        atomic_write_json(run_dir / "resource_report.json", report)

        train_manifest = resolve_from_repo(Path(config["data"]["train_manifest"]))
        val_manifest = resolve_from_repo(Path(config["data"]["val_manifest"]))
        metadata_path = resolve_from_repo(Path(config["data"]["manifest_metadata"]))
        all_train_rows = read_jsonl(train_manifest)
        val_rows = read_jsonl(val_manifest)
        validate_rows(all_train_rows, "train")
        validate_rows(val_rows, "val")
        dropout_train_rows = [r for r in all_train_rows if r.get("scenario") == "dropout"]
        train_scenarios = config["rl"].get("train_scenarios")
        if train_scenarios is not None:
            allowed_sc = set(train_scenarios)
            train_rows = [r for r in all_train_rows if r.get("scenario") in allowed_sc]
            if not train_rows:
                raise RunStop("data_failed", "no train rows remain after rl.train_scenarios filter", 3)
            print(
                f"RL_TRAIN_SCENARIOS {','.join(train_scenarios)} train_rows={len(train_rows)} "
                f"dropout_anchor_rows={len(dropout_train_rows)}",
                flush=True,
            )
        else:
            train_rows = all_train_rows
        release_dir = resolve_from_repo(args.release_dir)
        release = load_published_dpo_release(release_dir)
        if release.get("release_id") != config["model"]["parent_release"]:
            raise RunStop("failed", "DPO release_id does not match the RL config", 2)
        if release.get("adapter_sha256") != config["model"]["parent_release_sha256"]:
            raise RunStop("failed", "parent_release_sha256 does not match the published DPO adapter", 2)
        if int(release.get("source_step", -1)) != 40:
            raise RunStop("failed", "RL parent DPO release is not step 40", 2)
        manifest_sha = {
            "train": sha256_file(train_manifest),
            "val": sha256_file(val_manifest),
            "metadata": sha256_file(metadata_path),
        }
        config_sha = semantic_rl_config_sha256(config, manifest_sha)
        if config["profile"] == "pilot":
            drive_root = Path(os.environ.get("QWEN3_ASR_DRIVE_ROOT", "/content/drive/MyDrive/qwen3-asr"))
            if not drive_root.is_dir():
                raise RunStop("failed", "RL pilot requires a mounted Drive root", 2)
            baseline = config["baseline"]
            if manifest_sha["val"] != baseline["manifest_sha256"]:
                raise RunStop("failed", "val manifest does not match the pinned base", 2)
            if config["decode_eval_sha256"] != baseline["decode_eval_sha256"]:
                raise RunStop("failed", "decode eval hash does not match the pinned base", 2)
        if resume_state is not None and resume_state["config_sha256"] != config_sha:
            raise RunStop("resume_rejected", "semantic RL config hash does not match the checkpoint", 2)
        if resume_state is not None and resume_state["manifest_sha256"] != manifest_sha:
            raise RunStop("resume_rejected", "manifest hash does not match the checkpoint", 2)

        model, processor, lora_report = load_sft_model(config)
        load_adapter(model, release_dir / "adapter" / "adapter.safetensors")
        reference_lora = snapshot_lora_tensors(model)
        reference_sha = reference_tensors_sha256(reference_lora)
        if resume_state is not None and resume_state.get("reference_sha256") != reference_sha:
            raise RunStop("resume_rejected", "reference LoRA checksum does not match the checkpoint", 2)
        atomic_write_json(run_dir / "trainable_parameters.json", lora_report)
        print(
            f"RL_LORA trainable={lora_report['trainable_numel']} total={lora_report['total_numel']} "
            f"ref_sha256={reference_sha}",
            flush=True,
        )
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
        scoring_cfg = config["scoring"]
        reward_clip = (float(scoring_cfg["reward_clip"][0]), float(scoring_cfg["reward_clip"][1]))
        empty_pen = float(scoring_cfg["empty_penalty"])
        repeated_pen = float(scoring_cfg["repeated_penalty"])
        kl_beta = float(config["rl"]["kl_beta"])

        def load_row_audio(row: dict[str, Any]) -> Any:
            audio = load_audio(resolve_audio_path(audio_root / "placeholder.jsonl", str(row["audio"])))
            if audio.shape[0] / 16000.0 > float(config["data"]["max_audio_seconds"]):
                raise RunStop("data_failed", f"{row['sample_id']} is longer than max_audio_seconds", 4)
            return audio

        def score_candidate_tokens(
            row: dict[str, Any],
            text: str,
            audio: Any,
            *,
            require_grad: bool = True,
        ) -> tuple[Any, list[float], list[float], Any]:
            example = collate_example(
                processor,
                {"sample_id": str(row["sample_id"]), "text": text},
                audio,
                prefix=prefix,
                eos_token=eos_token,
                eos_id=int(eos_id),
                max_input_tokens=int(config["data"]["max_input_tokens"]),
                max_transcript_tokens=int(config["data"]["max_transcript_tokens"]),
            )
            batch = move_batch(dict(example), device, dtype)
            with use_reference_adapter(model, reference_lora):
                ref_out = model.thinker(**thinker_inputs(batch), use_cache=False)
                _ref_tensor, ref_list = _per_token_supervised_logprobs(ref_out.logits, batch["labels"])
            with torch.set_grad_enabled(require_grad):
                pol_out = model.thinker(**thinker_inputs(batch), use_cache=False)
                pol_tensor, pol_list = _per_token_supervised_logprobs(pol_out.logits, batch["labels"])
            return pol_tensor, pol_list, ref_list, batch

        if resume_state is None:
            probe_row = train_rows[0]
            probe_audio = load_row_audio(probe_row)
            pol_tensor, pol_list, ref_list, _batch = score_candidate_tokens(
                probe_row,
                str(probe_row["text"]),
                probe_audio,
            )
            if len(pol_list) != len(ref_list):
                raise RunStop("failed", "probe policy/reference token count mismatch", 1)
            probe_loss = -pol_tensor.mean()
            if not torch.isfinite(probe_loss):
                raise RunStop("failed", "probe RL loss is not finite", 1)
            probe_loss.backward()
            saw_grad = any(
                param.grad is not None and torch.isfinite(param.grad).all()
                for param in model.parameters()
                if param.requires_grad
            )
            if not saw_grad:
                raise RunStop("failed", "probe produced no finite LoRA gradient", 2)
            model.zero_grad(set_to_none=True)
            if reference_tensors_sha256(reference_lora) != reference_sha:
                raise RunStop("failed", "reference LoRA checksum changed during preflight", 1)
            print("RL_PREFLIGHT_OK", flush=True)
            log_event(run_dir, "preflight_ok", reference_sha256=reference_sha)

        # Step-0 baseline greedy CER, reward, and fixed prefix token IDs
        step0_dir = run_dir / "eval" / step_name(0)
        step0_summary_path = step0_dir / "step0_summary.json"
        if step0_summary_path.is_file():
            step0_summary = json.loads(step0_summary_path.read_text(encoding="utf-8"))
            if step0_summary.get("reference_sha256") != reference_sha:
                raise RunStop("failed", "pre-seeded step0_summary reference_sha256 mismatch", 2)
            step0_reward = float(step0_summary["step0_mean_reward"])
            step0_prefix_sha = str(step0_summary["step0_prefix_sha256"])
            sync_drive_files(
                args.run_id,
                [
                    (step0_summary_path, "eval/step_000000/step0_summary.json"),
                    (step0_dir / "scored.jsonl", "eval/step_000000/scored.jsonl"),
                    (step0_dir / "metrics.json", "eval/step_000000/metrics.json"),
                    (step0_dir / "by_scenario.csv", "eval/step_000000/by_scenario.csv"),
                ],
            )
            print(
                f"RL_STEP0 micro_cer={float(step0_summary['step0_micro_cer']):.8f} "
                f"mean_reward={step0_reward:.8f} prefix_sha256={step0_prefix_sha} raw_kl=0.0",
                flush=True,
            )
        else:
            step0_metrics = evaluate_val(
                model,
                processor,
                val_rows,
                val_manifest,
                config,
                run_dir,
                0,
                release["adapter_sha256"],
            )
            step0_scored = read_jsonl(step0_dir / "scored.jsonl")
            step0_reward = mean_greedy_reward(
                step0_scored,
                empty_penalty=empty_pen,
                repeated_penalty=repeated_pen,
                reward_clip=reward_clip,
            )
            prefix_records = []
            for s_row in step0_scored:
                pred_text = str(s_row.get("prediction_raw") or "")
                tok_ids = processor.tokenizer.encode(pred_text, add_special_tokens=False) + [int(eos_id)]
                prefix_records.append(
                    {
                        "sample_id": str(s_row["sample_id"]),
                        "prediction_raw": pred_text,
                        "token_ids": [int(t) for t in tok_ids],
                    }
                )
            step0_prefix_sha = prefix_records_sha256(prefix_records)
            step0_summary = {
                "schema_version": "qwen3-asr-rl-step0-v1",
                "step0_micro_cer": step0_metrics["micro_cer"],
                "step0_mean_reward": step0_reward,
                "step0_prefix_sha256": step0_prefix_sha,
                "step0_raw_kl": 0.0,
                "reference_sha256": reference_sha,
            }
            atomic_write_json(step0_summary_path, step0_summary)
            sync_drive_files(
                args.run_id,
                [
                    (step0_summary_path, "eval/step_000000/step0_summary.json"),
                    (step0_dir / "scored.jsonl", "eval/step_000000/scored.jsonl"),
                    (step0_dir / "metrics.json", "eval/step_000000/metrics.json"),
                    (step0_dir / "by_scenario.csv", "eval/step_000000/by_scenario.csv"),
                ],
            )
            print(
                f"RL_STEP0 micro_cer={step0_metrics['micro_cer']:.8f} "
                f"mean_reward={step0_reward:.8f} prefix_sha256={step0_prefix_sha} raw_kl=0.0",
                flush=True,
            )
            log_event(run_dir, "step0_evaluated", **step0_summary)

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
            print(f"RL_RESUME global_step={global_step}", flush=True)
            log_event(run_dir, "resume_ok", global_step=global_step)

        interrupt_consumed = bool(resume_state and resume_state.get("interrupt_consumed"))
        samples_seen = 0 if resume_state is None else int(resume_state["samples_seen"])
        group_visits = 0 if resume_state is None else int(resume_state["group_visits"])
        invalid_groups = 0 if resume_state is None else int(resume_state.get("invalid_groups", 0))
        consecutive_zero_signal = 0 if resume_state is None else int(resume_state.get("consecutive_zero_signal", 0))
        start_epoch = 0 if resume_state is None else int(resume_state["epoch"])
        start_cursor = 0 if resume_state is None else int(resume_state["cursor_after"])

        resolved_yaml = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
        atomic_write_text(run_dir / "resolved_config.yaml", resolved_yaml)
        atomic_write_json(run_dir / "manifest_sha256.json", manifest_sha)
        write_status(run_dir, "running", "optimizer_ready", global_step)

        budget_disabled = runtime_budget_disabled(os.environ.get("QWEN3_ASR_DISABLE_RUNTIME_BUDGET"))
        if budget_disabled:
            print("RL_RUNTIME_BUDGET disabled", flush=True)
            log_event(run_dir, "runtime_budget_disabled")
        started = time.monotonic()
        last_saved = global_step if resume_state is not None else 0
        sync_result = "not_saved"
        stop_reason = "max_steps"
        decisions: list[dict[str, Any]] = []
        decision_path = run_dir / "eval_decisions.jsonl"
        if resume_state is not None and decision_path.is_file():
            decisions = [json.loads(line) for line in decision_path.read_text(encoding="utf-8").splitlines() if line.strip()]

        def budget_exceeded() -> bool:
            if budget_disabled:
                return False
            return runtime_budget_exceeded(
                time.monotonic() - started,
                float(config["runtime"]["max_runtime_minutes"]),
                float(config["runtime"]["shutdown_margin_minutes"]),
            )

        def compute_val_raw_kl(step: int) -> float:
            step0_scored = read_jsonl(step0_dir / "scored.jsonl")
            by_id = {str(r["sample_id"]): r for r in val_rows}
            total_kl = 0.0
            total_tokens = 0
            was_training = bool(getattr(model, "training", False))
            model.eval()
            try:
                with torch.no_grad():
                    for idx, s_row in enumerate(step0_scored, start=1):
                        sid = str(s_row["sample_id"])
                        m_row = by_id[sid]
                        audio = load_row_audio(m_row)
                        pred_text = str(s_row.get("prediction_raw") or "")
                        example = collate_example(
                            processor,
                            {"sample_id": sid, "text": pred_text},
                            audio,
                            prefix=prefix,
                            eos_token=eos_token,
                            eos_id=int(eos_id),
                            max_input_tokens=int(config["data"]["max_input_tokens"]),
                            max_transcript_tokens=int(config["data"]["max_transcript_tokens"]),
                        )
                        batch = move_batch(dict(example), device, dtype)
                        pol_out = model.thinker(**thinker_inputs(batch), use_cache=False)
                        with use_reference_adapter(model, reference_lora):
                            ref_out = model.thinker(**thinker_inputs(batch), use_cache=False)
                        mean_kl, tok_count = exact_step0_conditional_kl_tensor(
                            pol_out.logits,
                            ref_out.logits,
                            batch["labels"],
                        )
                        total_kl += mean_kl * tok_count
                        total_tokens += tok_count
                        if idx == 1 or idx % 500 == 0 or idx == len(step0_scored):
                            print(f"RL_KL_PROGRESS step={step} {idx}/{len(step0_scored)}", flush=True)
            finally:
                if was_training:
                    model.train()
            return (total_kl / total_tokens) if total_tokens > 0 else 0.0

        def run_eval(step: int) -> str:
            ref_ok = reference_tensors_sha256(reference_lora) == reference_sha
            adapter_sha = sha256_file(run_dir / "checkpoints" / step_name(step) / "adapter" / "adapter.safetensors")
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
            scored_path = run_dir / "eval" / step_name(step) / "scored.jsonl"
            scored_rows = read_jsonl(scored_path)
            cur_reward = mean_greedy_reward(
                scored_rows,
                empty_penalty=empty_pen,
                repeated_penalty=repeated_pen,
                reward_clip=reward_clip,
            )
            raw_kl = compute_val_raw_kl(step)
            if config["profile"] == "smoke" or config.get("baseline") is None:
                decision = {
                    "step": step,
                    "passed": False,
                    "risk_stop": False,
                    "incomparable": True,
                    "mean_greedy_reward": cur_reward,
                    "step0_reward": step0_reward,
                    "reward_delta_vs_step0": cur_reward - step0_reward,
                    "raw_kl": raw_kl,
                    "reference_matched": ref_ok,
                }
            else:
                decision = judge_rl_checkpoint(
                    metrics,
                    config["rl"]["parent"],
                    config["baseline"],
                    config["gate"],
                    step0_reward=step0_reward,
                    current_reward=cur_reward,
                    raw_kl=raw_kl,
                    reference_matched=ref_ok,
                )
                decision["step"] = step
            decisions.append(decision)
            append_jsonl(decision_path, decision)
            sync_drive_files(
                args.run_id,
                [
                    (decision_path, "eval_decisions.jsonl"),
                    (scored_path, f"eval/{step_name(step)}/scored.jsonl"),
                    (run_dir / "eval" / step_name(step) / "metrics.json", f"eval/{step_name(step)}/metrics.json"),
                    (run_dir / "eval" / step_name(step) / "by_scenario.csv", f"eval/{step_name(step)}/by_scenario.csv"),
                ],
            )
            print(
                f"RL_GATE step={step} passed={decision['passed']} risk_stop={decision['risk_stop']} "
                f"reward={cur_reward:.6f} delta={cur_reward - step0_reward:+.6f} raw_kl={raw_kl:.8f}",
                flush=True,
            )
            if decision.get("risk_stop"):
                return "gate_risk"
            return "ok"

        max_steps = int(config["train"]["max_steps"])
        max_epochs = int(config["train"]["max_epochs"])
        accum_target = int(config["batch"]["gradient_accumulation_steps"])
        max_visits = int(config["rl"]["max_group_visits"])
        no_signal_limit = int(config["rl"]["no_signal_stop_groups"])
        invalid_rate_max = float(config["rl"]["invalid_group_rate_max"])
        loss_token_scope = str(config["rl"].get("loss_token_scope", "sequence"))
        max_edit_count = int(config["rl"].get("max_edit_count") or 2)
        accum_signal_only = bool(config["rl"].get("accumulate_signal_groups_only", False))
        dropout_anchor_weight = float(config["rl"].get("dropout_anchor_weight", 0.0))
        dropout_parent_cache: dict[str, str] = {}
        rollout_dir = run_dir / "rollouts"
        rollout_dir.mkdir(parents=True, exist_ok=True)

        def supervised_label_ids_for_text(sample_id: str, text: str, audio: Any) -> list[int]:
            example = collate_example(
                processor,
                {"sample_id": sample_id, "text": text},
                audio,
                prefix=prefix,
                eos_token=eos_token,
                eos_id=int(eos_id),
                max_input_tokens=int(config["data"]["max_input_tokens"]),
                max_transcript_tokens=int(config["data"]["max_transcript_tokens"]),
            )
            labels = example["labels"]
            return [int(v) for v in labels[labels != -100].tolist()]

        def compute_dropout_anchor_backward(step_idx: int) -> tuple[float, int]:
            if dropout_anchor_weight <= 0.0 or not dropout_train_rows:
                return 0.0, 0
            a_row = dropout_train_rows[(step_idx - 1) % len(dropout_train_rows)]
            a_sid = str(a_row["sample_id"])
            a_audio = load_row_audio(a_row)
            if a_sid not in dropout_parent_cache:
                with use_reference_adapter(model, reference_lora):
                    ref_cands = generate_group_candidates(
                        model,
                        processor,
                        a_row,
                        a_audio,
                        config={
                            **config,
                            "decode": {
                                **config["decode"],
                                "rollout": {
                                    **config["decode"]["rollout"],
                                    "group_size": 1,
                                    "greedy_candidates": 1,
                                },
                            },
                        },
                        epoch=0,
                        visit_index=step_idx,
                        global_step=0,
                        prefix=prefix,
                        eos_id=int(eos_id),
                    )
                dropout_parent_cache[a_sid] = str(ref_cands[0].get("prediction_raw") or "")
            parent_text = dropout_parent_cache[a_sid]
            if not parent_text.strip():
                return 0.0, 0
            example = collate_example(
                processor,
                {"sample_id": a_sid, "text": parent_text},
                a_audio,
                prefix=prefix,
                eos_token=eos_token,
                eos_id=int(eos_id),
                max_input_tokens=int(config["data"]["max_input_tokens"]),
                max_transcript_tokens=int(config["data"]["max_transcript_tokens"]),
            )
            batch = move_batch(dict(example), device, dtype)
            pol_out = model.thinker(**thinker_inputs(batch), use_cache=False)
            pol_t, pol_l = _per_token_supervised_logprobs(pol_out.logits, batch["labels"])
            anchor_loss_t = dropout_anchor_weight * (-pol_t.mean())
            if not torch.isfinite(anchor_loss_t):
                raise RunStop("failed", f"non-finite RL dropout anchor loss at step {step_idx}", 1)
            anchor_loss_t.backward()
            return float(anchor_loss_t.detach().item()), len(pol_l)

        optimizer.zero_grad(set_to_none=True)
        window_groups: list[dict[str, Any]] = []

        for epoch in range(start_epoch, max_epochs):
            shuffled = epoch_order(train_rows, int(config["seed"]), epoch)
            cursor_start = start_cursor if epoch == start_epoch else 0
            for cursor in range(cursor_start, len(shuffled)):
                if global_step >= max_steps:
                    break
                if group_visits >= max_visits or consecutive_zero_signal >= no_signal_limit:
                    stop_reason = "no_learning_signal"
                    break
                row = shuffled[cursor]
                group_visits += 1
                group_id = f"ep{epoch:03d}_v{group_visits:06d}_{row['sample_id']}"
                audio = load_row_audio(row)
                candidates = generate_group_candidates(
                    model,
                    processor,
                    row,
                    audio,
                    config=config,
                    epoch=epoch,
                    visit_index=group_visits,
                    global_step=global_step,
                    prefix=prefix,
                    eos_id=int(eos_id),
                )
                group_eval = evaluate_rollout_group(
                    candidates,
                    str(row["text"]),
                    group_size=int(config["decode"]["rollout"]["group_size"]),
                    empty_penalty=empty_pen,
                    repeated_penalty=repeated_pen,
                    reward_clip=reward_clip,
                )
                rollout_shard = rollout_dir / f"step_{global_step + 1:06d}.jsonl"
                if not group_eval["group_valid"]:
                    invalid_groups += 1
                    for cand in group_eval["candidates"]:
                        append_jsonl(
                            rollout_shard,
                            {
                                "run_id": args.run_id,
                                "global_step": global_step,
                                "group_id": group_id,
                                "visit_index": group_visits,
                                "sample_id": row["sample_id"],
                                "source_utterance_id": row.get("source_utterance_id"),
                                "scenario": row.get("scenario"),
                                **cand,
                                "policy_logp": None,
                                "reference_logp": None,
                                "k3_proxy_train": None,
                            },
                        )
                    if group_visits == 100 and (invalid_groups / group_visits) > invalid_rate_max:
                        raise RunStop("gate_failed", f"invalid_group_rate {invalid_groups / group_visits:.4f} > {invalid_rate_max}", 1)
                    continue

                aligned_info: dict[str, Any] | None = None
                effective_zero_adv = bool(group_eval["zero_advantage"])
                if not effective_zero_adv and loss_token_scope == "mismatched_only":
                    cand_label_ids = [
                        supervised_label_ids_for_text(str(row["sample_id"]), str(c["prediction_raw"]), audio)
                        for c in group_eval["candidates"]
                    ]
                    aligned_info = aligned_group_mismatch_mask(
                        group_eval["candidates"],
                        cand_label_ids,
                        max_edit_count=max_edit_count,
                    )
                    if not aligned_info["aligned_valid"]:
                        effective_zero_adv = True

                if effective_zero_adv:
                    consecutive_zero_signal += 1
                    for cand in group_eval["candidates"]:
                        append_jsonl(
                            rollout_shard,
                            {
                                "run_id": args.run_id,
                                "global_step": global_step,
                                "group_id": group_id,
                                "visit_index": group_visits,
                                "sample_id": row["sample_id"],
                                "source_utterance_id": row.get("source_utterance_id"),
                                "scenario": row.get("scenario"),
                                **cand,
                                "policy_logp": None,
                                "reference_logp": None,
                                "k3_proxy_train": 0.0,
                            },
                        )
                    if not accum_signal_only:
                        window_groups.append({"group_valid": True, "zero_advantage": True, "group_loss": 0.0, "k3_proxy_mean": 0.0})
                else:
                    consecutive_zero_signal = 0
                    pol_lists: list[list[float]] = []
                    ref_lists: list[list[float]] = []
                    if aligned_info is not None:
                        advantages = [float(a) for a in aligned_info["aligned_advantages"]]
                        active_count = len(aligned_info["aligned_indices"])
                        token_mask_list = list(aligned_info["token_mask"])
                    else:
                        advantages = [float(c["advantage"]) for c in group_eval["candidates"]]
                        active_count = len(group_eval["candidates"])
                        token_mask_list = None
                    group_denom = float(active_count * accum_target)
                    # Differentiable surrogate per candidate: -A_g * mean_t(logp_policy) + kl_beta * mean_t(expm1(d) - d)
                    for idx_c, (cand, adv_val) in enumerate(zip(group_eval["candidates"], advantages)):
                        is_active = aligned_info is None or idx_c in aligned_info["aligned_indices"]
                        pol_t, pol_l, ref_l, _ = score_candidate_tokens(
                            row,
                            str(cand["prediction_raw"]),
                            audio,
                            require_grad=is_active,
                        )
                        pol_lists.append(pol_l)
                        ref_lists.append(ref_l)
                        if not is_active:
                            pol_t = None
                            continue
                        ref_t = torch.tensor(ref_l, dtype=torch.float32, device=pol_t.device)
                        diff = ref_t - pol_t
                        k3_t = torch.expm1(diff) - diff
                        if token_mask_list is not None:
                            mask_t = torch.tensor(token_mask_list, dtype=torch.bool, device=pol_t.device)
                            pg_t = pol_t[mask_t].mean()
                        else:
                            pg_t = pol_t.mean()
                        cand_loss_tensor = -adv_val * pg_t + kl_beta * k3_t.mean()
                        if not torch.isfinite(cand_loss_tensor):
                            raise RunStop("failed", f"non-finite RL group loss at visit {group_visits}", 1)
                        (cand_loss_tensor / group_denom).backward()
                        pol_t = None
                        cand_loss_tensor = None
                    loss_info = group_loss_from_token_logprobs(
                        advantages,
                        pol_lists,
                        ref_lists,
                        kl_beta=kl_beta,
                        zero_advantage=False,
                    )
                    for cand, p_l, r_l, k3_m in zip(
                        group_eval["candidates"],
                        pol_lists,
                        ref_lists,
                        loss_info["candidate_k3_means"],
                    ):
                        append_jsonl(
                            rollout_shard,
                            {
                                "run_id": args.run_id,
                                "global_step": global_step + 1,
                                "group_id": group_id,
                                "visit_index": group_visits,
                                "sample_id": row["sample_id"],
                                "source_utterance_id": row.get("source_utterance_id"),
                                "scenario": row.get("scenario"),
                                **cand,
                                "policy_logp": {"sum": sum(p_l), "tokens": len(p_l)},
                                "reference_logp": {"sum": sum(r_l), "tokens": len(r_l)},
                                "k3_proxy_train": k3_m,
                            },
                        )
                    window_groups.append(
                        {
                            "group_valid": True,
                            "zero_advantage": False,
                            "group_loss": float(loss_info["group_loss"]),
                            "k3_proxy_mean": float(loss_info["k3_proxy_mean"]),
                        }
                    )

                if len(window_groups) == accum_target:
                    win = accumulate_window_groups(window_groups)
                    window_groups = []
                    if not win["should_step_optimizer"]:
                        optimizer.zero_grad(set_to_none=True)
                        print(
                            f"RL_ZERO_WINDOW visits={group_visits} consecutive_zero={consecutive_zero_signal}",
                            flush=True,
                        )
                        continue
                    step = global_step + 1
                    anchor_loss_val, anchor_tok_count = compute_dropout_anchor_backward(step)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [param for param in model.parameters() if param.requires_grad],
                        float(config["optim"]["max_grad_norm"]),
                    )
                    if not torch.isfinite(grad_norm):
                        raise RunStop("failed", f"non-finite RL grad norm at global_step {step}", 1)
                    scheduler.step()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step = step
                    samples_seen += accum_target
                    append_jsonl(
                        run_dir / "metrics.jsonl",
                        {
                            "global_step": step,
                            "samples_seen": samples_seen,
                            "group_visits": group_visits,
                            "invalid_groups": invalid_groups,
                            "loss": win["window_loss"],
                            "grad_norm": float(grad_norm),
                            "lr": current_lr(optimizer),
                            "anchor_loss": anchor_loss_val,
                            "anchor_tokens": anchor_tok_count,
                        },
                    )
                    print(
                        f"RL_STEP global_step={step}/{max_steps} loss={win['window_loss']:.6f} "
                        f"grad_norm={float(grad_norm):.6f} lr={current_lr(optimizer):.8g} visits={group_visits} "
                        f"anchor_loss={anchor_loss_val:.6f} anchor_tokens={anchor_tok_count}",
                        flush=True,
                    )
                    interrupt = should_interrupt(step, config["train"]["interrupt_after_step"], interrupt_consumed)
                    if interrupt:
                        interrupt_consumed = True
                    need_save = (
                        step % int(config["train"]["save_steps"]) == 0
                        or step == max_steps
                        or interrupt
                    )
                    if need_save:
                        if reference_tensors_sha256(reference_lora) != reference_sha:
                            raise RunStop("failed", "reference LoRA checksum drifted before save", 1)
                        state = {
                            "schema_version": "qwen3-asr-training-state-v1",
                            "stage": "rl",
                            "global_step": step,
                            "micro_step": samples_seen,
                            "samples_seen": samples_seen,
                            "group_visits": group_visits,
                            "invalid_groups": invalid_groups,
                            "consecutive_zero_signal": consecutive_zero_signal,
                            "accumulation_boundary": 0,
                            "epoch": epoch,
                            "cursor_after": cursor + 1,
                            "config_sha256": config_sha,
                            "manifest_sha256": manifest_sha,
                            "eval_decode_sha256": config["decode_eval_sha256"],
                            "parent_release_sha256": config["model"]["parent_release_sha256"],
                            "reference_sha256": reference_sha,
                            "publishable": False,
                            "interrupt_consumed": interrupt_consumed,
                            "trainable_names": names,
                            "profile": config["profile"],
                        }
                        sampler_state = {
                            "schema_version": "qwen3-asr-sampler-v1",
                            "splits": {
                                "train": {
                                    "epoch": epoch,
                                    "cursor": cursor + 1,
                                    "shuffle_seed": int(config["seed"]),
                                    "visit": group_visits,
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
                        sync_result = sync_drive(run_dir, args.run_id, step, digest)
                        write_latest(run_dir / "latest.json", step=step, checkpoint_sha256=digest)
                        last_saved = step
                        atomic_write_json(run_dir / "training_state.json", state)
                        sync_drive_files(
                            args.run_id,
                            [
                                (run_dir / "metrics.jsonl", "metrics.jsonl"),
                                (run_dir / "training_state.json", "training_state.json"),
                            ],
                        )
                    if need_save and step % int(config["train"]["eval_steps"]) == 0:
                        eval_res = run_eval(step)
                        if eval_res == "gate_risk":
                            stop_reason = "gate_risk"
                            break
                    if interrupt:
                        print(f"RL_INTERRUPT step={step}", flush=True)
                        write_status(run_dir, "paused", "interrupt_after_step", step)
                        write_rl_gate(
                            run_dir,
                            config,
                            manifest_sha["val"],
                            decisions,
                            reference_sha256=reference_sha,
                            step0_prefix_sha256=step0_prefix_sha,
                        )
                        print("RL_EXIT 0", flush=True)
                        return 0
            if global_step >= max_steps or stop_reason in {"gate_risk", "no_learning_signal"}:
                break

        if reference_tensors_sha256(reference_lora) != reference_sha:
            raise RunStop("failed", "reference LoRA checksum drifted at exit", 1)
        chosen = select_sft_checkpoint(decisions) if config["profile"] == "pilot" else None
        final_status = "completed" if (chosen is not None or config["profile"] == "smoke") else "gate_failed"
        gate_status = "incomparable" if config["profile"] == "smoke" else ("passed" if chosen is not None else "failed")
        write_status(run_dir, final_status, stop_reason, global_step)
        write_rl_gate(
            run_dir,
            config,
            manifest_sha["val"],
            decisions,
            selected=chosen,
            gate_status=gate_status,
            reference_sha256=reference_sha,
            step0_prefix_sha256=step0_prefix_sha,
        )
        write_rl_run_report(
            run_dir,
            {
                "run_id": args.run_id,
                "status": final_status,
                "reason": stop_reason,
                "global_step": global_step,
                "group_visits": group_visits,
                "publishable": False,
                "config_sha256": config_sha,
                "reference_sha256": reference_sha,
                "step0_reward": step0_reward,
                "git_sha": info.get("git_sha"),
                "dirty": info.get("dirty"),
                "sync": sync_result,
            },
        )
        sync_drive_files(
            args.run_id,
            [
                (run_dir / "status.json", "status.json"),
                (run_dir / "gate.json", "gate.json"),
                (run_dir / "run_report.md", "run_report.md"),
            ],
        )
        print(f"RL_GATE_STATUS {gate_status}", flush=True)
        print(f"RL_DONE global_step={global_step} visits={group_visits}", flush=True)
        print("RL_EXIT 0", flush=True)
        return 0
    except RunStop as exc:
        write_status(run_dir, exc.status, str(exc), global_step)
        log_event(run_dir, exc.status, message=str(exc))
        print(f"RL_EXIT {exc.code} {exc.status}: {exc}", file=sys.stderr, flush=True)
        return exc.code
    except torch.cuda.OutOfMemoryError as exc:
        write_status(run_dir, "oom", str(exc), global_step)
        log_event(run_dir, "oom", message=str(exc))
        print(f"RL_EXIT 1 oom: {exc}", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:
        write_status(run_dir, "failed", f"{type(exc).__name__}: {exc}", global_step)
        log_event(run_dir, "failed", message=str(exc), traceback=traceback.format_exc())
        print(f"RL_EXIT 1 failed: {exc}", file=sys.stderr, flush=True)
        return 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_rl_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"config rejected: {exc}", file=sys.stderr)
        return 2
    if args.seed != int(config["seed"]):
        print(f"seed mismatch: cli {args.seed} config {config['seed']}", file=sys.stderr)
        return 2
    if config["runtime"]["deterministic_algorithms"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if re.fullmatch(rf"\d{{8}}T\d{{6}}{re.escape(RUN_ID_PREFIX)}{args.seed}", args.run_id) is None:
        print("run_id must look like YYYYMMDDTHHMMSS_rl_06b_zh_seed<seed>", file=sys.stderr)
        return 2
    if args.manifest is not None:
        config["data"]["train_manifest"] = str(args.manifest)
    if args.val_manifest is not None:
        config["data"]["val_manifest"] = str(args.val_manifest)
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
        children = [p.name for p in run_dir.iterdir()]
        eval_children = [p.name for p in (run_dir / "eval").iterdir()] if (run_dir / "eval").is_dir() else []
        if children != ["eval"] or eval_children != ["step_000000"]:
            print(f"output dir already exists: {run_dir}", file=sys.stderr)
            return 2
    for key in ("train_manifest", "val_manifest", "manifest_metadata"):
        path = resolve_from_repo(Path(config["data"][key]))
        if not path.is_file():
            print(f"missing {key}: {path}", file=sys.stderr)
            return 2
    if not resolve_from_repo(args.release_dir).is_dir():
        print(f"missing release dir: {args.release_dir}", file=sys.stderr)
        return 2
    return run_training(args, config, resume_state)


if __name__ == "__main__":
    raise SystemExit(main())
