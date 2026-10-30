"""Parse an RL training YAML (stage: rl). SFT and DPO loaders still reject non-null rl blocks."""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any

import yaml

from train.dpo_schema import PARENT_METRICS_SCHEMA, _resolve_existing
from train.sft_schema import (
    SCENARIOS,
    _as_float,
    _as_int,
    _as_str,
    _hex,
    _mapping,
    _reject_unknown,
    canonical_json,
    load_train_config,
    sha256_file,
    sha256_text,
)

DPO_RELEASE_ID = "dpo_06b_zh_20261022T133007_step000040"
RL_FIELDS = {
    "algorithm",
    "advantage",
    "kl_beta",
    "updates_per_rollout",
    "max_group_visits",
    "no_signal_stop_groups",
    "invalid_group_rate_max",
    "parent_metrics",
    "train_scenarios",
    "loss_token_scope",
    "max_edit_count",
    "accumulate_signal_groups_only",
    "dropout_anchor_weight",
}


def _load_dpo_parent_metrics(value: str, config_path: Path) -> dict[str, Any]:
    path = _resolve_existing(value, config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.suffix in {".yaml", ".yml"} else None
    if payload is None:
        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("RL parent metrics must be an object")
    _reject_unknown(
        payload,
        {
            "schema_version",
            "split",
            "parent",
            "release_id",
            "source_step",
            "samples",
            "micro_cer",
            "by_scenario",
        },
        "RL parent metrics",
    )
    if payload.get("schema_version") != PARENT_METRICS_SCHEMA:
        raise ValueError(f"RL parent metrics schema_version must be {PARENT_METRICS_SCHEMA}")
    if payload.get("split") != "val" or payload.get("parent") != "dpo":
        raise ValueError("RL parent metrics must be the DPO val parent")
    by_scenario = _mapping(payload.get("by_scenario"), "RL parent metrics by_scenario")
    _reject_unknown(by_scenario, set(SCENARIOS), "RL parent metrics by_scenario")
    parsed: dict[str, dict[str, Any]] = {}
    for name in SCENARIOS:
        row = _mapping(by_scenario.get(name), f"parent.by_scenario.{name}")
        _reject_unknown(row, {"samples", "micro_cer"}, f"parent.by_scenario.{name}")
        parsed[name] = {
            "samples": _as_int(row.get("samples"), f"parent.by_scenario.{name}.samples", minimum=1),
            "micro_cer": _as_float(row.get("micro_cer"), f"parent.by_scenario.{name}.micro_cer", minimum=0.0),
        }
    return {
        "metrics": value,
        "sha256": sha256_file(path),
        "release_id": _as_str(payload.get("release_id"), "parent.release_id"),
        "source_step": _as_int(payload.get("source_step"), "parent.source_step", minimum=1),
        "samples": _as_int(payload.get("samples"), "parent.samples", minimum=1),
        "micro_cer": _as_float(payload.get("micro_cer"), "parent.micro_cer", minimum=0.0),
        "by_scenario": parsed,
    }


def _parse_rl_block(raw: dict[str, Any], config_path: Path) -> dict[str, Any]:
    _reject_unknown(raw, RL_FIELDS, "rl")
    if raw.get("algorithm") != "group_pg_mixed_v1":
        raise ValueError("rl.algorithm must be group_pg_mixed_v1")
    if raw.get("advantage") != "group_mean_centered":
        raise ValueError("rl.advantage must be group_mean_centered")
    kl_beta = _as_float(raw.get("kl_beta"), "rl.kl_beta", minimum=0.0, maximum=1.0)
    if kl_beta <= 0.0:
        raise ValueError("rl.kl_beta must be > 0")
    updates_per_rollout = _as_int(raw.get("updates_per_rollout"), "rl.updates_per_rollout", minimum=1, maximum=1)
    max_group_visits = _as_int(raw.get("max_group_visits"), "rl.max_group_visits", minimum=6, maximum=10000)
    no_signal_stop_groups = _as_int(raw.get("no_signal_stop_groups"), "rl.no_signal_stop_groups", minimum=1, maximum=100)
    invalid_group_rate_max = _as_float(
        raw.get("invalid_group_rate_max"),
        "rl.invalid_group_rate_max",
        minimum=0.0,
        maximum=0.1,
    )
    parent = _load_dpo_parent_metrics(_as_str(raw.get("parent_metrics"), "rl.parent_metrics"), config_path)
    raw_train_scenarios = raw.get("train_scenarios")
    train_scenarios: list[str] | None = None
    if raw_train_scenarios is not None:
        if not isinstance(raw_train_scenarios, list) or not raw_train_scenarios:
            raise ValueError("rl.train_scenarios must be a non-empty list of scenarios")
        seen: list[str] = []
        for item in raw_train_scenarios:
            name = _as_str(item, "rl.train_scenarios[]")
            if name not in SCENARIOS or name in seen:
                raise ValueError(f"invalid or duplicate rl.train_scenarios entry: {name}")
            seen.append(name)
        train_scenarios = seen
    loss_token_scope = raw.get("loss_token_scope", "sequence")
    if loss_token_scope not in {"sequence", "mismatched_only"}:
        raise ValueError("rl.loss_token_scope must be sequence or mismatched_only")
    raw_max_edit = raw.get("max_edit_count")
    max_edit_count: int | None = None
    if raw_max_edit is not None:
        max_edit_count = _as_int(raw_max_edit, "rl.max_edit_count", minimum=1, maximum=5)
    raw_accum_signal = raw.get("accumulate_signal_groups_only", False)
    if not isinstance(raw_accum_signal, bool):
        raise ValueError("rl.accumulate_signal_groups_only must be a boolean")
    dropout_anchor_weight = _as_float(
        raw.get("dropout_anchor_weight", 0.0),
        "rl.dropout_anchor_weight",
        minimum=0.0,
        maximum=0.1,
    )
    return {
        "algorithm": "group_pg_mixed_v1",
        "advantage": "group_mean_centered",
        "kl_beta": kl_beta,
        "updates_per_rollout": updates_per_rollout,
        "max_group_visits": max_group_visits,
        "no_signal_stop_groups": no_signal_stop_groups,
        "invalid_group_rate_max": invalid_group_rate_max,
        "train_scenarios": train_scenarios,
        "loss_token_scope": loss_token_scope,
        "max_edit_count": max_edit_count,
        "accumulate_signal_groups_only": raw_accum_signal,
        "dropout_anchor_weight": dropout_anchor_weight,
        "parent_metrics": parent["metrics"],
        "parent": parent,
    }


def _check_dpo_step40_parent(resolved: dict[str, Any]) -> None:
    if resolved["model"]["parent_release"] != DPO_RELEASE_ID:
        raise ValueError(f"RL parent_release must be the published step-40 DPO release ({DPO_RELEASE_ID})")
    if resolved["rl"]["parent"]["release_id"] != DPO_RELEASE_ID:
        raise ValueError("RL parent metrics must name the step-40 DPO release")
    if int(resolved["rl"]["parent"]["source_step"]) != 40:
        raise ValueError("RL parent metrics must be step 40")


def _is_aligned_rl_pilot(resolved: dict[str, Any]) -> bool:
    rl = resolved["rl"]
    return (
        resolved["profile"] == "pilot"
        and rl.get("train_scenarios") == ["noise", "reverb", "far_field"]
        and rl.get("loss_token_scope") == "mismatched_only"
        and rl.get("max_edit_count") == 2
        and rl.get("accumulate_signal_groups_only") is True
        and float(rl.get("dropout_anchor_weight", 0.0)) == 0.05
    )


def _check_rl_budget(resolved: dict[str, Any]) -> None:
    profile = resolved["profile"]
    if profile not in {"smoke", "pilot"}:
        raise ValueError("RL profile must be smoke or pilot")
    if _is_aligned_rl_pilot(resolved):
        expected = {
            "learning_rate": 1e-6,
            "dropout": 0.0,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "max_steps": 20,
            "max_epochs": 1,
            "save_steps": 5,
            "eval_steps": 5,
            "max_runtime_minutes": 120,
            "kl_beta": 0.10,
            "updates_per_rollout": 1,
            "max_group_visits": 1000,
            "no_signal_stop_groups": 100,
            "invalid_group_rate_max": 0.01,
        }
        if resolved["optim"].get("scheduler_max_steps") != 50:
            raise ValueError("aligned RL pilot requires optim.scheduler_max_steps == 50")
    else:
        if (
            resolved["rl"].get("train_scenarios") is not None
            or resolved["rl"].get("loss_token_scope") != "sequence"
            or resolved["rl"].get("max_edit_count") is not None
            or resolved["rl"].get("accumulate_signal_groups_only") is not False
            or float(resolved["rl"].get("dropout_anchor_weight", 0.0)) != 0.0
        ):
            raise ValueError("partial or unapproved aligned RL flags are not allowed outside rl_pilot_aligned")
        expected_common = {
            "learning_rate": 1e-5,
            "dropout": 0.0,
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "save_steps": 5,
            "eval_steps": 5,
            "kl_beta": 0.04,
            "updates_per_rollout": 1,
            "no_signal_stop_groups": 20,
            "invalid_group_rate_max": 0.01,
        }
        if profile == "pilot":
            expected = {
                **expected_common,
                "max_steps": 50,
                "max_epochs": 1,
                "max_runtime_minutes": 120,
                "max_group_visits": 500,
            }
        else:
            expected = {
                **expected_common,
                "max_steps": 6,
                "max_epochs": 100,
                "max_runtime_minutes": 30,
                "max_group_visits": 500,
            }
        if resolved["optim"].get("scheduler_max_steps") is not None:
            raise ValueError("RL config must not set optim.scheduler_max_steps")
    actual = {
        "learning_rate": float(resolved["optim"]["learning_rate"]),
        "dropout": float(resolved["lora"]["dropout"]),
        "micro_batch_size": int(resolved["batch"]["micro_batch_size"]),
        "gradient_accumulation_steps": int(resolved["batch"]["gradient_accumulation_steps"]),
        "max_steps": int(resolved["train"]["max_steps"]),
        "max_epochs": int(resolved["train"]["max_epochs"]),
        "save_steps": int(resolved["train"]["save_steps"]),
        "eval_steps": int(resolved["train"]["eval_steps"]),
        "max_runtime_minutes": int(resolved["runtime"]["max_runtime_minutes"]),
        "kl_beta": float(resolved["rl"]["kl_beta"]),
        "updates_per_rollout": int(resolved["rl"]["updates_per_rollout"]),
        "max_group_visits": int(resolved["rl"]["max_group_visits"]),
        "no_signal_stop_groups": int(resolved["rl"]["no_signal_stop_groups"]),
        "invalid_group_rate_max": float(resolved["rl"]["invalid_group_rate_max"]),
    }
    mismatches = [name for name, value in expected.items() if actual[name] != value]
    if mismatches:
        raise ValueError(f"RL {profile} budget does not match the frozen table: {mismatches}")
    _check_dpo_step40_parent(resolved)


def load_rl_config(path: Path) -> dict[str, Any]:
    """Parse stage rl. Shared fields go through the SFT loader after rl is set back to null."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    if raw.get("stage") != "rl":
        raise ValueError("this loader only accepts stage rl")
    if raw.get("dpo") is not None:
        raise ValueError("RL config requires dpo: null")
    rl_raw = raw.get("rl")
    if not isinstance(rl_raw, dict):
        raise ValueError("RL config requires an rl mapping")
    model = raw.get("model")
    if not isinstance(model, dict):
        raise ValueError("model must be a mapping")
    parent_release = _as_str(model.get("parent_release"), "model.parent_release")
    parent_sha = _hex(model.get("parent_release_sha256"), "model.parent_release_sha256", 64)
    parsed_rl = _parse_rl_block(rl_raw, path)
    cloned = copy.deepcopy(raw)
    cloned["stage"] = "sft"
    cloned["rl"] = None
    cloned["model"]["parent_release"] = None
    cloned["model"]["parent_release_sha256"] = None
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(cloned, handle, allow_unicode=True, sort_keys=False)
        temporary = Path(handle.name)
    try:
        resolved = load_train_config(temporary)
    finally:
        temporary.unlink(missing_ok=True)
    if float(resolved["lora"]["dropout"]) != 0.0:
        raise ValueError("RL LoRA dropout must be 0")
    resolved["stage"] = "rl"
    resolved["model"]["parent_release"] = parent_release
    resolved["model"]["parent_release_sha256"] = parent_sha
    resolved["rl"] = parsed_rl
    resolved["decode_rollout_sha256"] = sha256_text(canonical_json(resolved["decode"]["rollout"]))
    resolved["yaml_sha256"] = sha256_file(path)
    _check_rl_budget(resolved)
    return resolved


def semantic_rl_config_sha256(resolved: dict[str, Any], manifest_sha256: dict[str, str]) -> str:
    """Hash the RL contract. Local paths and --resume stay outside."""
    rl = resolved["rl"]
    parent = rl["parent"]
    baseline_block = None
    if resolved.get("baseline") is not None:
        baseline_block = {
            "sha256": resolved["baseline"]["sha256"],
            "manifest_sha256": resolved["baseline"]["manifest_sha256"],
            "decode_eval_sha256": resolved["baseline"]["decode_eval_sha256"],
            "predictions_sha256": resolved["baseline"]["predictions_sha256"],
        }
    payload = {
        "schema_version": resolved["schema_version"],
        "stage": "rl",
        "profile": resolved["profile"],
        "seed": resolved["seed"],
        "model": resolved["model"],
        "runtime": resolved["runtime"],
        "lora": resolved["lora"],
        "data": {
            "max_audio_seconds": resolved["data"]["max_audio_seconds"],
            "max_transcript_tokens": resolved["data"]["max_transcript_tokens"],
            "max_input_tokens": resolved["data"]["max_input_tokens"],
            "overlength": resolved["data"]["overlength"],
            "sampler": resolved["data"]["sampler"],
            "train_manifest_sha256": manifest_sha256["train"],
            "val_manifest_sha256": manifest_sha256["val"],
            "manifest_metadata_sha256": manifest_sha256["metadata"],
        },
        "batch": resolved["batch"],
        "optim": resolved["optim"],
        "train": resolved["train"],
        "scoring": resolved["scoring"],
        "decode": resolved["decode"],
        "gate": resolved["gate"],
        "baseline": baseline_block,
        "dpo": None,
        "rl": {
            "algorithm": rl["algorithm"],
            "advantage": rl["advantage"],
            "kl_beta": rl["kl_beta"],
            "updates_per_rollout": rl["updates_per_rollout"],
            "max_group_visits": rl["max_group_visits"],
            "no_signal_stop_groups": rl["no_signal_stop_groups"],
            "invalid_group_rate_max": rl["invalid_group_rate_max"],
            "parent_metrics_sha256": parent["sha256"],
            "parent_release_id": parent["release_id"],
            "parent_source_step": parent["source_step"],
            "parent_by_scenario": parent["by_scenario"],
            **(
                {
                    "train_scenarios": rl["train_scenarios"],
                    "loss_token_scope": rl["loss_token_scope"],
                    "max_edit_count": rl["max_edit_count"],
                    "accumulate_signal_groups_only": rl["accumulate_signal_groups_only"],
                    "dropout_anchor_weight": rl["dropout_anchor_weight"],
                }
                if _is_aligned_rl_pilot(resolved)
                else {}
            ),
        },
        "decode_eval_sha256": resolved["decode_eval_sha256"],
        "decode_rollout_sha256": resolved["decode_rollout_sha256"],
    }
    return sha256_text(canonical_json(payload))
