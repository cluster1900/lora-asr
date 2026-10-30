"""Parse a DPO training YAML. The SFT loader still rejects a non-null dpo block."""

from __future__ import annotations

import copy
import tempfile
from pathlib import Path
from typing import Any

import yaml

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

PARENT_METRICS_SCHEMA = "qwen3-asr-parent-metrics-v1"
DPO_FIELDS = {
    "beta",
    "pair_train_manifest",
    "pair_val_manifest",
    "reference_cache",
    "reference_report",
    "pair_metadata",
    "parent_metrics",
    "listen_waiver",
    "audit_disposition",
    "min_train_pairs",
    "min_pairs_per_scenario",
    "logp_reduction",
    "train_scenarios",
    "null_train_pairs_sha256",
    "null_val_pairs_sha256",
    "filled_train_pairs_sha256",
    "filled_val_pairs_sha256",
    "reference_cache_sha256",
    "reference_report_sha256",
    "launch_utterance",
    "pair_metadata_sha256",
    "source_null_train_pairs_sha256",
    "source_null_val_pairs_sha256",
    "aligned_substitution_only",
    "exclude_clean_identical_rejected",
    "max_edit_count",
    "loss_token_scope",
    "val_scenarios",
    "ref_margin_min",
    "ref_margin_max",
    "dropout_anchor_weight",
    "dropout_anchor_target",
    "dropout_anchor_samples_per_step",
}
PILOT_RELEASE_ID = "sft_06b_zh_20261015T234939_step000030"
NON_DROPOUT_SCENARIOS = ["clean", "noise", "reverb", "far_field"]
DEGRADED_NONDROPOUT_SCENARIOS = ["noise", "reverb", "far_field"]


def _resolve_existing(value: str, config_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute() and path.is_file():
        return path
    repo_path = Path(__file__).resolve().parents[1] / value
    if repo_path.is_file():
        return repo_path
    beside = config_path.parent / value
    if beside.is_file():
        return beside
    raise ValueError(f"parent metrics file is missing: {value}")


def _load_parent_metrics(value: str, config_path: Path) -> dict[str, Any]:
    path = _resolve_existing(value, config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) if path.suffix in {".yaml", ".yml"} else None
    if payload is None:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("parent metrics must be an object")
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
        "parent metrics",
    )
    if payload.get("schema_version") != PARENT_METRICS_SCHEMA:
        raise ValueError(f"parent metrics schema_version must be {PARENT_METRICS_SCHEMA}")
    if payload.get("split") != "val" or payload.get("parent") != "sft":
        raise ValueError("parent metrics must be the SFT val parent")
    by_scenario = _mapping(payload.get("by_scenario"), "parent metrics by_scenario")
    _reject_unknown(by_scenario, set(SCENARIOS), "parent metrics by_scenario")
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


def _optional_sha(raw: dict[str, Any], key: str) -> str | None:
    if key not in raw or raw.get(key) is None:
        return None
    return _hex(raw.get(key), f"dpo.{key}", 64)


def _parse_launch_fields(raw: dict[str, Any]) -> dict[str, Any]:
    """The 2026-10-18 utterance is optional. Old configs omit it."""
    if "launch_utterance" not in raw or raw.get("launch_utterance") is None:
        launch = None
    else:
        launch = raw.get("launch_utterance")
        if launch != "开始吧":
            raise ValueError("dpo.launch_utterance must be 开始吧")
    source_train = _optional_sha(raw, "source_null_train_pairs_sha256")
    source_val = _optional_sha(raw, "source_null_val_pairs_sha256")
    metadata_sha = _optional_sha(raw, "pair_metadata_sha256")
    extras = (source_train, source_val, metadata_sha)
    if launch is None and any(item is not None for item in extras):
        raise ValueError("source pair hashes require dpo.launch_utterance")
    if launch is not None and any(item is None for item in extras):
        raise ValueError("gap-strip launch requires source pair hashes and pair_metadata_sha256")
    return {
        "launch_utterance": launch,
        "source_null_train_pairs_sha256": source_train,
        "source_null_val_pairs_sha256": source_val,
        "pair_metadata_sha256": metadata_sha,
    }


def _parse_pair_filter_fields(raw: dict[str, Any]) -> dict[str, Any]:
    aligned_raw = raw.get("aligned_substitution_only", False)
    if not isinstance(aligned_raw, bool):
        raise ValueError("dpo.aligned_substitution_only must be a boolean")
    exclude_clean_raw = raw.get("exclude_clean_identical_rejected", False)
    if not isinstance(exclude_clean_raw, bool):
        raise ValueError("dpo.exclude_clean_identical_rejected must be a boolean")
    max_edit_raw = raw.get("max_edit_count")
    max_edit_count = None if max_edit_raw is None else _as_int(max_edit_raw, "dpo.max_edit_count", minimum=1)
    loss_token_scope_raw = raw.get("loss_token_scope", "sequence")
    if not isinstance(loss_token_scope_raw, str) or loss_token_scope_raw not in {"sequence", "mismatched_only"}:
        raise ValueError("dpo.loss_token_scope must be sequence or mismatched_only")
    if loss_token_scope_raw == "mismatched_only" and not aligned_raw:
        raise ValueError("dpo.loss_token_scope=mismatched_only requires dpo.aligned_substitution_only=true")
    min_raw = raw.get("ref_margin_min")
    max_raw = raw.get("ref_margin_max")
    ref_margin_min = None if min_raw is None else _as_float(min_raw, "dpo.ref_margin_min")
    ref_margin_max = None if max_raw is None else _as_float(max_raw, "dpo.ref_margin_max")
    if ref_margin_min is not None and ref_margin_max is not None and ref_margin_min >= ref_margin_max:
        raise ValueError("dpo.ref_margin_min must be strictly less than dpo.ref_margin_max")
    return {
        "aligned_substitution_only": aligned_raw,
        "exclude_clean_identical_rejected": exclude_clean_raw,
        "max_edit_count": max_edit_count,
        "loss_token_scope": loss_token_scope_raw,
        "ref_margin_min": ref_margin_min,
        "ref_margin_max": ref_margin_max,
    }


def _parse_dpo_block(raw: dict[str, Any], config_path: Path) -> dict[str, Any]:
    _reject_unknown(raw, DPO_FIELDS, "dpo")
    if raw.get("logp_reduction") != "sequence_sum":
        raise ValueError("dpo.logp_reduction must be sequence_sum")
    if raw.get("listen_waiver") != "user_20261017_skip_listen":
        raise ValueError("dpo.listen_waiver must record the 2026-10-17 skip-listen decision")
    if raw.get("audit_disposition") != "unlistened":
        raise ValueError("dpo.audit_disposition must stay unlistened")
    beta = _as_float(raw.get("beta"), "dpo.beta", minimum=0.0)
    if beta <= 0.0:
        raise ValueError("dpo.beta must be > 0")
    parent = _load_parent_metrics(_as_str(raw.get("parent_metrics"), "dpo.parent_metrics"), config_path)
    train_scenarios = _parse_scenario_list(raw, "train_scenarios")
    val_scenarios = _parse_scenario_list(raw, "val_scenarios")
    launch = _parse_launch_fields(raw)
    pair_filters = _parse_pair_filter_fields(raw)
    anchor = _parse_dropout_anchor(raw, train_scenarios, str(pair_filters["loss_token_scope"]))
    return {
        "beta": beta,
        "pair_train_manifest": _as_str(raw.get("pair_train_manifest"), "dpo.pair_train_manifest"),
        "pair_val_manifest": _as_str(raw.get("pair_val_manifest"), "dpo.pair_val_manifest"),
        "reference_cache": _as_str(raw.get("reference_cache"), "dpo.reference_cache"),
        "reference_report": _as_str(raw.get("reference_report"), "dpo.reference_report"),
        "pair_metadata": _as_str(raw.get("pair_metadata"), "dpo.pair_metadata"),
        "parent_metrics": parent["metrics"],
        "parent": parent,
        "listen_waiver": "user_20261017_skip_listen",
        "audit_disposition": "unlistened",
        "min_train_pairs": _as_int(raw.get("min_train_pairs"), "dpo.min_train_pairs", minimum=1),
        "min_pairs_per_scenario": _as_int(raw.get("min_pairs_per_scenario"), "dpo.min_pairs_per_scenario", minimum=1),
        "logp_reduction": "sequence_sum",
        "train_scenarios": train_scenarios,
        "val_scenarios": val_scenarios,
        "null_train_pairs_sha256": _hex(raw.get("null_train_pairs_sha256"), "dpo.null_train_pairs_sha256", 64),
        "null_val_pairs_sha256": _hex(raw.get("null_val_pairs_sha256"), "dpo.null_val_pairs_sha256", 64),
        "filled_train_pairs_sha256": _hex(raw.get("filled_train_pairs_sha256"), "dpo.filled_train_pairs_sha256", 64),
        "filled_val_pairs_sha256": _hex(raw.get("filled_val_pairs_sha256"), "dpo.filled_val_pairs_sha256", 64),
        "reference_cache_sha256": _hex(raw.get("reference_cache_sha256"), "dpo.reference_cache_sha256", 64),
        "reference_report_sha256": _hex(raw.get("reference_report_sha256"), "dpo.reference_report_sha256", 64),
        **launch,
        **pair_filters,
        **anchor,
    }


def _parse_dropout_anchor(
    raw: dict[str, Any],
    train_scenarios: list[str] | None,
    loss_token_scope: str,
) -> dict[str, Any]:
    """Small parent-hypothesis NLL on dropout. Weight 0 leaves old configs unchanged."""
    weight = _as_float(raw.get("dropout_anchor_weight", 0.0), "dpo.dropout_anchor_weight", minimum=0.0, maximum=0.1)
    target = raw.get("dropout_anchor_target", "rejected_text")
    if target != "rejected_text":
        raise ValueError("dpo.dropout_anchor_target must be rejected_text")
    samples_per_step = _as_int(
        raw.get("dropout_anchor_samples_per_step", 1),
        "dpo.dropout_anchor_samples_per_step",
        minimum=1,
        maximum=4,
    )
    if weight > 0.0:
        if loss_token_scope != "mismatched_only":
            raise ValueError("dropout anchor requires dpo.loss_token_scope=mismatched_only")
        if train_scenarios is None or "dropout" in train_scenarios:
            raise ValueError("dropout anchor requires train_scenarios without dropout")
    elif samples_per_step != 1:
        raise ValueError("dpo.dropout_anchor_samples_per_step requires dpo.dropout_anchor_weight > 0")
    return {
        "dropout_anchor_weight": weight,
        "dropout_anchor_target": target,
        "dropout_anchor_samples_per_step": samples_per_step,
    }


def _parse_scenario_list(raw: dict[str, Any], key: str) -> list[str] | None:
    value = raw.get(key)
    if value is None:
        return None
    if not isinstance(value, list) or not value:
        raise ValueError(f"dpo.{key} must be a non-empty list")
    names: list[str] = []
    for item in value:
        if not isinstance(item, str) or item not in SCENARIOS:
            raise ValueError(f"dpo.{key} has an unknown scenario")
        if item in names:
            raise ValueError(f"dpo.{key} repeats a scenario")
        names.append(item)
    return names


def _check_step30_parent(resolved: dict[str, Any]) -> None:
    if resolved["model"]["parent_release"] != PILOT_RELEASE_ID:
        raise ValueError("DPO parent_release must be the published step-30 SFT release")
    if resolved["dpo"]["parent"]["release_id"] != PILOT_RELEASE_ID:
        raise ValueError("DPO parent metrics must name the step-30 release")
    if int(resolved["dpo"]["parent"]["source_step"]) != 30:
        raise ValueError("DPO parent metrics must be step 30")


def _allows_anchor_stop40(resolved: dict[str, Any]) -> bool:
    """True only for the 0.05 rejected-text anchor stopped at step 40.

    164 pairs with accumulation 4 finish one epoch at step 41. That tail is
    not a multiple of eval_steps. Run 20261020T041218 passed at step 40 and
    the next update flipped one dropout utterance. Other pilot configs stay
    at max_steps 100.
    """
    dpo = resolved["dpo"]
    weight = float(dpo.get("dropout_anchor_weight", 0.0))
    return (
        int(resolved["train"]["max_steps"]) == 40
        and abs(weight - 0.05) <= 1e-12
        and dpo.get("dropout_anchor_target") == "rejected_text"
        and dpo.get("loss_token_scope") == "mismatched_only"
        and int(resolved["optim"]["warmup_steps"]) == 10
    )


def _check_pilot_budget(resolved: dict[str, Any]) -> None:
    if resolved["profile"] != "pilot":
        return
    expected = {
        "learning_rate": 5e-6,
        "dropout": 0.0,
        "gradient_accumulation_steps": 4,
        "max_steps": 40 if _allows_anchor_stop40(resolved) else 100,
        "max_epochs": 1,
        "save_steps": 10,
        "eval_steps": 10,
        "beta": 0.1,
        "max_runtime_minutes": 120,
    }
    actual = {
        "learning_rate": float(resolved["optim"]["learning_rate"]),
        "dropout": float(resolved["lora"]["dropout"]),
        "gradient_accumulation_steps": int(resolved["batch"]["gradient_accumulation_steps"]),
        "max_steps": int(resolved["train"]["max_steps"]),
        "max_epochs": int(resolved["train"]["max_epochs"]),
        "save_steps": int(resolved["train"]["save_steps"]),
        "eval_steps": int(resolved["train"]["eval_steps"]),
        "beta": float(resolved["dpo"]["beta"]),
        "max_runtime_minutes": int(resolved["runtime"]["max_runtime_minutes"]),
    }
    mismatches = [name for name, value in expected.items() if actual[name] != value]
    if mismatches:
        raise ValueError(f"DPO pilot budget does not match the frozen table: {mismatches}")
    scheduler_max_steps = resolved["optim"].get("scheduler_max_steps")
    if scheduler_max_steps is not None:
        if not _allows_anchor_stop40(resolved) or int(scheduler_max_steps) != 100:
            raise ValueError("DPO pilot scheduler_max_steps is only allowed as 100 on anchor stop40 configs")
    if resolved["dpo"]["train_scenarios"] not in (
        None,
        NON_DROPOUT_SCENARIOS,
        DEGRADED_NONDROPOUT_SCENARIOS,
    ):
        raise ValueError(
            "DPO pilot train_scenarios must be omitted, [clean, noise, reverb, far_field], or [noise, reverb, far_field]"
        )
    if resolved["dpo"].get("val_scenarios") not in (None, NON_DROPOUT_SCENARIOS):
        raise ValueError("DPO pilot val_scenarios must be omitted or [clean, noise, reverb, far_field]")
    if resolved["dpo"].get("exclude_clean_identical_rejected"):
        train_scenarios = resolved["dpo"]["train_scenarios"]
        if train_scenarios is None or "clean" in train_scenarios:
            raise ValueError("dpo.exclude_clean_identical_rejected requires train_scenarios without clean")
    if resolved["dpo"].get("launch_utterance"):
        raise ValueError("DPO pilot must not record the gap-strip launch utterance")
    _check_step30_parent(resolved)


def _check_probe_budget(resolved: dict[str, Any]) -> None:
    if resolved["profile"] != "dropout_probe":
        return
    expected = {
        "learning_rate": 5e-5,
        "dropout": 0.0,
        "gradient_accumulation_steps": 4,
        "micro_batch_size": 1,
        "max_steps": 10,
        "max_epochs": 1,
        "save_steps": 10,
        "eval_steps": 10,
        "keep_last": 1,
        "warmup_steps": 0,
        "beta": 0.1,
        "max_runtime_minutes": 120,
    }
    actual = {
        "learning_rate": float(resolved["optim"]["learning_rate"]),
        "dropout": float(resolved["lora"]["dropout"]),
        "gradient_accumulation_steps": int(resolved["batch"]["gradient_accumulation_steps"]),
        "micro_batch_size": int(resolved["batch"]["micro_batch_size"]),
        "max_steps": int(resolved["train"]["max_steps"]),
        "max_epochs": int(resolved["train"]["max_epochs"]),
        "save_steps": int(resolved["train"]["save_steps"]),
        "eval_steps": int(resolved["train"]["eval_steps"]),
        "keep_last": int(resolved["train"]["keep_last"]),
        "warmup_steps": int(resolved["optim"]["warmup_steps"]),
        "beta": float(resolved["dpo"]["beta"]),
        "max_runtime_minutes": int(resolved["runtime"]["max_runtime_minutes"]),
    }
    mismatches = [name for name, value in expected.items() if actual[name] != value]
    if mismatches:
        raise ValueError(f"DPO dropout probe budget does not match the frozen table: {mismatches}")
    if resolved["optim"].get("scheduler_max_steps") is not None:
        raise ValueError("dropout probe must not set optim.scheduler_max_steps")
    if resolved["dpo"]["train_scenarios"] != ["dropout"]:
        raise ValueError("dropout probe must train only dropout pairs")
    launch = resolved["dpo"].get("launch_utterance")
    if launch is not None:
        dpo = resolved["dpo"]
        if dpo["null_train_pairs_sha256"] == dpo["source_null_train_pairs_sha256"]:
            raise ValueError("gap-strip null train pairs must differ from the frozen source")
        if dpo["null_val_pairs_sha256"] == dpo["source_null_val_pairs_sha256"]:
            raise ValueError("gap-strip null val pairs must differ from the frozen source")
        if dpo["filled_train_pairs_sha256"] == dpo["null_train_pairs_sha256"]:
            raise ValueError("filled train pairs must differ from the null file")
        if dpo["filled_val_pairs_sha256"] == dpo["null_val_pairs_sha256"]:
            raise ValueError("filled val pairs must differ from the null file")
    _check_step30_parent(resolved)


def load_dpo_config(path: Path) -> dict[str, Any]:
    """Parse stage dpo. Shared fields go through the SFT loader after dpo is set back to null."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    if raw.get("stage") != "dpo":
        raise ValueError("this loader only accepts stage dpo")
    if raw.get("rl") is not None:
        raise ValueError("DPO config requires rl: null")
    dpo_raw = raw.get("dpo")
    if not isinstance(dpo_raw, dict):
        raise ValueError("DPO config requires a dpo mapping")
    model = raw.get("model")
    if not isinstance(model, dict):
        raise ValueError("model must be a mapping")
    parent_release = _as_str(model.get("parent_release"), "model.parent_release")
    parent_sha = _hex(model.get("parent_release_sha256"), "model.parent_release_sha256", 64)
    parsed_dpo = _parse_dpo_block(dpo_raw, path)
    cloned = copy.deepcopy(raw)
    cloned["stage"] = "sft"
    cloned["dpo"] = None
    cloned["model"]["parent_release"] = None
    cloned["model"]["parent_release_sha256"] = None
    # The SFT loader only accepts smoke, pilot, or scale. The probe borrows those
    # pilot checks, then the budget below freezes the dropout_probe numbers.
    if raw.get("profile") == "dropout_probe":
        cloned["profile"] = "pilot"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(cloned, handle, allow_unicode=True, sort_keys=False)
        temporary = Path(handle.name)
    try:
        resolved = load_train_config(temporary)
    finally:
        temporary.unlink(missing_ok=True)
    if float(resolved["lora"]["dropout"]) != 0.0:
        raise ValueError("DPO LoRA dropout must be 0")
    resolved["stage"] = "dpo"
    if raw.get("profile") == "dropout_probe":
        resolved["profile"] = "dropout_probe"
    resolved["model"]["parent_release"] = parent_release
    resolved["model"]["parent_release_sha256"] = parent_sha
    resolved["dpo"] = parsed_dpo
    resolved["yaml_sha256"] = sha256_file(path)
    _check_pilot_budget(resolved)
    _check_probe_budget(resolved)
    return resolved


def semantic_dpo_config_sha256(resolved: dict[str, Any], manifest_sha256: dict[str, str]) -> str:
    """Hash the DPO contract. Local paths and --resume stay outside."""
    dpo = resolved["dpo"]
    parent = dpo["parent"]
    payload = {
        "schema_version": resolved["schema_version"],
        "stage": "dpo",
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
        "baseline": {
            "sha256": resolved["baseline"]["sha256"],
            "manifest_sha256": resolved["baseline"]["manifest_sha256"],
            "decode_eval_sha256": resolved["baseline"]["decode_eval_sha256"],
            "predictions_sha256": resolved["baseline"]["predictions_sha256"],
        },
        "dpo": {
            "beta": dpo["beta"],
            "logp_reduction": dpo["logp_reduction"],
            "listen_waiver": dpo["listen_waiver"],
            "audit_disposition": dpo["audit_disposition"],
            "min_train_pairs": dpo["min_train_pairs"],
            "min_pairs_per_scenario": dpo["min_pairs_per_scenario"],
            "train_scenarios": dpo["train_scenarios"],
            "null_train_pairs_sha256": dpo["null_train_pairs_sha256"],
            "null_val_pairs_sha256": dpo["null_val_pairs_sha256"],
            "filled_train_pairs_sha256": dpo["filled_train_pairs_sha256"],
            "filled_val_pairs_sha256": dpo["filled_val_pairs_sha256"],
            "reference_cache_sha256": dpo["reference_cache_sha256"],
            "reference_report_sha256": dpo["reference_report_sha256"],
            "launch_utterance": dpo.get("launch_utterance"),
            "pair_metadata_sha256": dpo.get("pair_metadata_sha256"),
            "source_null_train_pairs_sha256": dpo.get("source_null_train_pairs_sha256"),
            "source_null_val_pairs_sha256": dpo.get("source_null_val_pairs_sha256"),
            "parent_metrics_sha256": parent["sha256"],
            "parent_release_id": parent["release_id"],
            "parent_source_step": parent["source_step"],
            "parent_by_scenario": parent["by_scenario"],
            **(
                {
                    "aligned_substitution_only": bool(dpo.get("aligned_substitution_only", False)),
                    "exclude_clean_identical_rejected": bool(dpo.get("exclude_clean_identical_rejected", False)),
                    "max_edit_count": dpo.get("max_edit_count"),
                    "loss_token_scope": dpo.get("loss_token_scope", "sequence"),
                    "val_scenarios": dpo.get("val_scenarios"),
                    "ref_margin_min": dpo.get("ref_margin_min"),
                    "ref_margin_max": dpo.get("ref_margin_max"),
                    **(
                        {
                            "dropout_anchor_weight": float(dpo.get("dropout_anchor_weight", 0.0)),
                            "dropout_anchor_target": dpo.get("dropout_anchor_target", "rejected_text"),
                            **(
                                {"dropout_anchor_samples_per_step": int(dpo["dropout_anchor_samples_per_step"])}
                                if int(dpo.get("dropout_anchor_samples_per_step", 1)) != 1
                                else {}
                            ),
                        }
                        if float(dpo.get("dropout_anchor_weight", 0.0)) > 0.0
                        else {}
                    ),
                }
                if (
                    bool(dpo.get("aligned_substitution_only", False))
                    or bool(dpo.get("exclude_clean_identical_rejected", False))
                    or dpo.get("max_edit_count") is not None
                    or dpo.get("loss_token_scope", "sequence") != "sequence"
                    or dpo.get("val_scenarios") is not None
                    or dpo.get("ref_margin_min") is not None
                    or dpo.get("ref_margin_max") is not None
                )
                else {}
            ),
        },
        "rl": None,
        "decode_eval_sha256": resolved["decode_eval_sha256"],
    }
    return sha256_text(canonical_json(payload))
