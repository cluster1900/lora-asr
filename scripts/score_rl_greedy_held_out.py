#!/usr/bin/env python3
"""Append one greedy Full Held-out row for a saved RL checkpoint.

The row uses the trainer's ``evaluate_rl_validation(decode_mode="greedy")``.
This process does not construct an optimizer, resume training, or export a
merged model. A provenance-valid row for the same step is left in place.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

FULL_HELD_OUT_SCOPE = "Full Held-out"
FROZEN_SHA256 = "b0701db2ec3735222179e21fb4bf3c49c256d8da5d3a85cb7e61bfa6b7d9cd99"
FROZEN_ROWS = 1698


def greedy_row_ok(record: Mapping[str, Any]) -> bool:
    """Match the server gate: one greedy full-pool row with integer counts."""
    if record.get("val_decode") != "greedy":
        return False
    if record.get("val_eval_scope") != FULL_HELD_OUT_SCOPE:
        return False
    if record.get("val_manifest_sha256") != FROZEN_SHA256:
        return False
    assigned = record.get("val_assigned_rows")
    manifest_rows = record.get("val_manifest_rows")
    rollouts = record.get("val_rollouts")
    if type(assigned) is not int or type(manifest_rows) is not int or type(rollouts) is not int:
        return False
    if record.get("val_mean_reward") is None:
        return False
    return assigned == manifest_rows == rollouts == FROZEN_ROWS


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def _step_matches(record: Mapping[str, Any], step: int) -> bool:
    value = record.get("global_step", record.get("step"))
    return type(value) is int and value == step


def append_decision(records: Sequence[Mapping[str, Any]], step: int) -> str:
    """Return skip, refuse, or append for one global step.

    A training row without ``val_mean_reward`` does not count. A greedy
    Full Held-out row that fails provenance blocks a second append.
    """
    held = [
        record
        for record in records
        if _step_matches(record, step)
        and record.get("val_mean_reward") is not None
        and record.get("val_eval_scope") == FULL_HELD_OUT_SCOPE
        and record.get("val_decode") == "greedy"
    ]
    if any(greedy_row_ok(record) for record in held):
        return "skip"
    if held:
        return "refuse"
    return "append"


def build_greedy_row(
    step: int,
    mean_reward: float,
    error_rate: float,
    manifest_rows: int,
    manifest_sha256: str,
    assigned_rows: int,
    rollouts: int,
    timestamp: str,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    return {
        "global_step": int(step),
        "val_mean_reward": round(float(mean_reward), 4),
        "val_error_rate": round(float(error_rate), 4),
        "val_eval_scope": FULL_HELD_OUT_SCOPE,
        "val_decode": "greedy",
        "decoding": {
            "max_new_tokens": int(max_new_tokens),
            "do_sample": False,
            "num_beams": 1,
            "num_return_sequences": 1,
        },
        "val_manifest_rows": int(manifest_rows),
        "val_manifest_sha256": manifest_sha256,
        "val_assigned_rows": int(assigned_rows),
        "val_rollouts": int(rollouts),
        "scored_by": "score_rl_greedy_held_out",
        "timestamp": timestamp,
    }


def checkpoint_complete(run_dir: Path, step: int) -> bool:
    root = run_dir / "checkpoints" / f"step_{step}"
    return (root / "optimizer.pt").is_file() and (
        root / "adapter" / "adapter_model.safetensors"
    ).is_file()


def resolve_score_step(state: Mapping[str, Any], run_dir: Path) -> int:
    """Score the checkpoint named by ``pipeline_state.global_step``."""
    step = state.get("global_step")
    if type(step) is not int or step < 0:
        raise ValueError("pipeline_state.global_step must be a non-negative integer")
    if not checkpoint_complete(run_dir, step):
        raise FileNotFoundError(
            f"checkpoint step_{step} needs adapter_model.safetensors and optimizer.pt"
        )
    return step


def _load_reward_config(config: Mapping[str, Any], repo_root: Path) -> Dict[str, Any]:
    reward_cfg = dict(config.get("reward") or {})
    if "components" in reward_cfg:
        return reward_cfg
    cfg_path_str = reward_cfg.get("config_path")
    candidate = (
        (repo_root / cfg_path_str)
        if cfg_path_str
        else (repo_root / "configs" / "train" / "reward_config.yaml")
    )
    if not candidate.is_file():
        raise FileNotFoundError(f"reward config missing: {candidate}")
    import yaml

    loaded = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"reward config is not a mapping: {candidate}")
    return loaded


def _init_distributed() -> tuple:
    import torch
    import torch.distributed as dist

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, f"cuda:{local_rank}", True
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return 0, device, False


def score_checkpoint(args: argparse.Namespace) -> int:
    import inspect

    import torch
    import torch.distributed as dist
    import yaml

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from train.train_rl import (  # noqa: WPS433
        RLAudioDataset,
        _append_jsonl,
        compute_file_sha256,
        evaluate_rl_validation,
        load_asr_for_rl,
        training_token_budget,
    )

    if "decode_mode" not in inspect.signature(evaluate_rl_validation).parameters:
        raise RuntimeError(
            "evaluate_rl_validation has no decode_mode; refusing the sample eval path"
        )

    rank, device, is_distributed = _init_distributed()
    exit_code = 0
    try:
        loss_log = Path(args.loss_log)
        decision = append_decision(read_jsonl(loss_log), int(args.step))
        if decision == "skip":
            if rank == 0:
                print(f"SKIP existing greedy held-out row for step {args.step}")
            return 0
        if decision == "refuse":
            if rank == 0:
                print(
                    f"REFUSE step {args.step}: greedy held-out row exists and fails provenance",
                    file=sys.stderr,
                )
            return 1

        checkpoint = Path(args.checkpoint).resolve()
        training_state_path = checkpoint / "training_state.json"
        training_state = json.loads(training_state_path.read_text(encoding="utf-8"))
        if int(training_state["global_step"]) != int(args.step):
            raise ValueError(
                f"checkpoint global_step {training_state.get('global_step')} != {args.step}"
            )
        config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
        reward_cfg = _load_reward_config(config, repo_root)
        val_manifest = Path(args.val_manifest).resolve()
        manifest_sha = compute_file_sha256(val_manifest)
        dataset = RLAudioDataset(val_manifest)
        if len(dataset) != FROZEN_ROWS or manifest_sha != FROZEN_SHA256:
            raise ValueError(
                f"val manifest rows={len(dataset)} sha={manifest_sha} "
                f"does not match the frozen pool"
            )

        asr_model, lora_model, _meta, _model_id, _revision = load_asr_for_rl(
            config,
            device,
            resume_from_checkpoint=checkpoint,
            rank=rank,
        )
        mean_reward, error_rate, rollouts, assigned = evaluate_rl_validation(
            lora_model,
            asr_model,
            dataset,
            device,
            max_eval_samples=None,
            reward_config=reward_cfg,
            eval_seed=int(args.eval_seed),
            decode_mode="greedy",
        )
        if is_distributed:
            dist.barrier()
        if rank == 0:
            row = build_greedy_row(
                int(args.step),
                float(mean_reward),
                float(error_rate),
                len(dataset),
                manifest_sha,
                int(assigned),
                int(rollouts),
                datetime.datetime.now(datetime.timezone.utc).isoformat(),
                training_token_budget(config),
            )
            fresh = append_decision(read_jsonl(loss_log), int(args.step))
            if fresh == "skip":
                print(f"SKIP existing greedy held-out row for step {args.step}")
            elif fresh == "refuse" or not greedy_row_ok(row):
                print(
                    f"REFUSE step {args.step}: assigned={assigned} rollouts={rollouts} "
                    f"reward={row['val_mean_reward']}",
                    file=sys.stderr,
                )
                exit_code = 1
            else:
                _append_jsonl(loss_log, row)
                written = [
                    item
                    for item in read_jsonl(loss_log)
                    if _step_matches(item, int(args.step)) and greedy_row_ok(item)
                ]
                if not written:
                    print(f"REFUSE step {args.step}: row was not readable after append", file=sys.stderr)
                    exit_code = 1
                else:
                    print(
                        f"greedy_held_out step={args.step} "
                        f"reward={row['val_mean_reward']} "
                        f"error={row['val_error_rate']} "
                        f"assigned={row['val_assigned_rows']} rollouts={row['val_rollouts']}"
                    )
        if is_distributed:
            code_tensor = torch.tensor([exit_code], device=device, dtype=torch.int64)
            dist.broadcast(code_tensor, src=0)
            exit_code = int(code_tensor.item())
            dist.barrier()
    finally:
        if is_distributed and dist.is_initialized():
            dist.destroy_process_group()
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=int)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--loss-log", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--decision-only", action="store_true")
    parser.add_argument("--resolve-step", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.resolve_step:
        if args.run_dir is None:
            raise SystemExit("--resolve-step requires --run-dir")
        state_path = Path(args.run_dir) / "pipeline_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        print(resolve_score_step(state, Path(args.run_dir)))
        return 0
    if args.step is None:
        raise SystemExit("--step is required")
    if args.decision_only:
        if args.loss_log is None:
            raise SystemExit("--decision-only requires --loss-log")
        print(append_decision(read_jsonl(Path(args.loss_log)), int(args.step)))
        return 0
    required = {
        "--loss-log": args.loss_log,
        "--config": args.config,
        "--checkpoint": args.checkpoint,
        "--val-manifest": args.val_manifest,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit(f"missing {' '.join(missing)}")
    return score_checkpoint(args)


if __name__ == "__main__":
    sys.exit(main())
