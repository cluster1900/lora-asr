"""Parse qwen3-asr-train-v1. Unknown fields fail before an optimizer exists."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

TRAIN_SCHEMA = "qwen3-asr-train-v1"
PROJECTIONS = ["q_proj", "k_proj", "v_proj", "o_proj"]
LANGUAGE_TO_OFFICIAL = {"zh": "Chinese"}
SCENARIOS = ("clean", "dropout", "far_field", "noise", "reverb")


def canonical_json(payload: Any) -> str:
    """Stable JSON for hashes. NaN and Infinity raise."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_unknown(section: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {unknown}")


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping")
    return value


def _as_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _as_int(value: Any, field: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field} must be <= {maximum}")
    return value


def _as_float(value: Any, field: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and number < minimum:
        raise ValueError(f"{field} must be >= {minimum}")
    if maximum is not None and number > maximum:
        raise ValueError(f"{field} must be <= {maximum}")
    return number


def _as_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _hex(value: Any, field: str, length: int) -> str:
    text = _as_str(value, field)
    if len(text) != length or any(char not in "0123456789abcdef" for char in text):
        raise ValueError(f"{field} must be {length} lowercase hex characters")
    return text


def _optional_str(value: Any, field: str) -> str | None:
    if value is None:
        return None
    return _as_str(value, field)


def _decode_block(raw: dict[str, Any], name: str, *, sampled: bool) -> dict[str, Any]:
    if sampled:
        allowed = {
            "group_size",
            "greedy_candidates",
            "temperature",
            "top_p",
            "top_k",
            "max_new_tokens",
            "seed_strategy",
            "context",
            "language",
            "eos_policy",
        }
    else:
        allowed = {
            "do_sample",
            "num_beams",
            "max_new_tokens",
            "repetition_penalty",
            "no_repeat_ngram_size",
            "context",
            "language",
            "eos_policy",
        }
    _reject_unknown(raw, allowed, name)
    context = raw.get("context")
    if not isinstance(context, str):
        raise ValueError(f"{name}.context must be a string")
    language = raw.get("language")
    if language not in LANGUAGE_TO_OFFICIAL:
        raise ValueError(f"{name}.language must be zh in this MVP")
    if raw.get("eos_policy") != "tokenizer_eos_only":
        raise ValueError(f"{name}.eos_policy must be tokenizer_eos_only")
    max_new_tokens = _as_int(raw.get("max_new_tokens"), f"{name}.max_new_tokens", minimum=1)
    parsed: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "context": context,
        "language": language,
        "eos_policy": "tokenizer_eos_only",
    }
    if sampled:
        if raw.get("seed_strategy") != "sha256_seed_epoch_visit_sample_candidate":
            raise ValueError(f"{name}.seed_strategy is not the contracted strategy")
        parsed.update(
            {
                "group_size": _as_int(raw.get("group_size"), f"{name}.group_size", minimum=1),
                "greedy_candidates": _as_int(raw.get("greedy_candidates"), f"{name}.greedy_candidates", minimum=0),
                "temperature": _as_float(raw.get("temperature"), f"{name}.temperature", minimum=0.0),
                "top_p": _as_float(raw.get("top_p"), f"{name}.top_p", minimum=0.0, maximum=1.0),
                "top_k": _as_int(raw.get("top_k"), f"{name}.top_k", minimum=0),
                "seed_strategy": "sha256_seed_epoch_visit_sample_candidate",
            }
        )
        return parsed
    if _as_bool(raw.get("do_sample"), f"{name}.do_sample"):
        raise ValueError(f"{name}.do_sample must be false")
    num_beams = _as_int(raw.get("num_beams"), f"{name}.num_beams", minimum=1)
    if num_beams != 1:
        raise ValueError(f"{name}.num_beams must be 1")
    parsed.update(
        {
            "do_sample": False,
            "num_beams": num_beams,
            "repetition_penalty": _as_float(raw.get("repetition_penalty"), f"{name}.repetition_penalty", minimum=0.0),
            "no_repeat_ngram_size": _as_int(raw.get("no_repeat_ngram_size"), f"{name}.no_repeat_ngram_size", minimum=0),
        }
    )
    return parsed


def load_train_config(path: Path) -> dict[str, Any]:
    """Parse a training YAML into a normalized dict."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    _reject_unknown(
        raw,
        {
            "schema_version",
            "stage",
            "profile",
            "seed",
            "model",
            "runtime",
            "lora",
            "data",
            "batch",
            "optim",
            "train",
            "scoring",
            "decode",
            "gate",
            "baseline",
            "dpo",
            "rl",
        },
        "config",
    )
    if raw.get("schema_version") != TRAIN_SCHEMA:
        raise ValueError(f"schema_version must be {TRAIN_SCHEMA}")
    if raw.get("stage") != "sft":
        raise ValueError("this runner only accepts stage sft")
    profile = raw.get("profile")
    if profile not in {"smoke", "pilot", "scale"}:
        raise ValueError("profile must be smoke, pilot, or scale")
    seed = _as_int(raw.get("seed"), "seed", minimum=0)

    model = _mapping(raw.get("model"), "model")
    _reject_unknown(model, {"name", "revision", "processor_sha256", "parent_release", "parent_release_sha256"}, "model")
    parent_release = model.get("parent_release")
    parent_release_sha = model.get("parent_release_sha256")
    if parent_release is not None or parent_release_sha is not None:
        raise ValueError("SFT parent_release and parent_release_sha256 must be null")

    runtime = _mapping(raw.get("runtime"), "runtime")
    _reject_unknown(
        runtime,
        {
            "environment_lock_sha256",
            "dtype",
            "world_size",
            "attention",
            "gradient_checkpointing",
            "num_workers",
            "allow_dirty",
            "tf32",
            "deterministic_algorithms",
            "max_runtime_minutes",
            "shutdown_margin_minutes",
            "min_free_gpu_fraction",
            "min_free_disk_gb",
            "min_free_ram_gb",
        },
        "runtime",
    )
    if runtime.get("dtype") != "bfloat16":
        raise ValueError("runtime.dtype must be bfloat16")
    if _as_int(runtime.get("world_size"), "runtime.world_size") != 1:
        raise ValueError("runtime.world_size must be 1")
    if runtime.get("attention") != "sdpa":
        raise ValueError("runtime.attention must be sdpa")
    if _as_int(runtime.get("num_workers"), "runtime.num_workers", minimum=0) != 0:
        raise ValueError("runtime.num_workers must be 0")
    if _as_bool(runtime.get("tf32"), "runtime.tf32"):
        raise ValueError("runtime.tf32 must be false")

    lora = _mapping(raw.get("lora"), "lora")
    _reject_unknown(lora, {"rank", "alpha", "dropout", "scope", "projections"}, "lora")
    if lora.get("scope") != "decoder_attention_only":
        raise ValueError("lora.scope must be decoder_attention_only")
    if lora.get("projections") != PROJECTIONS:
        raise ValueError("lora.projections must be [q_proj, k_proj, v_proj, o_proj]")

    data = _mapping(raw.get("data"), "data")
    _reject_unknown(
        data,
        {"train_manifest", "val_manifest", "manifest_metadata", "max_audio_seconds", "max_transcript_tokens", "max_input_tokens", "overlength", "sampler"},
        "data",
    )
    if data.get("overlength") != "reject":
        raise ValueError("data.overlength must be reject")
    if data.get("sampler") != "seeded_shuffle":
        raise ValueError("data.sampler must be seeded_shuffle")

    batch = _mapping(raw.get("batch"), "batch")
    _reject_unknown(batch, {"micro_batch_size", "gradient_accumulation_steps", "eval_batch_size"}, "batch")
    if _as_int(batch.get("micro_batch_size"), "batch.micro_batch_size") != 1:
        raise ValueError("this SFT runner requires micro_batch_size 1")
    if _as_int(batch.get("eval_batch_size"), "batch.eval_batch_size") != 1:
        raise ValueError("batch.eval_batch_size must be 1")

    optim = _mapping(raw.get("optim"), "optim")
    _reject_unknown(
        optim,
        {
            "name",
            "learning_rate",
            "betas",
            "eps",
            "weight_decay",
            "max_grad_norm",
            "scheduler",
            "warmup_steps",
            "min_lr_ratio",
            "scheduler_max_steps",
        },
        "optim",
    )
    if optim.get("name") != "adamw":
        raise ValueError("optim.name must be adamw")
    if optim.get("scheduler") != "cosine":
        raise ValueError("optim.scheduler must be cosine")
    betas = optim.get("betas")
    if not isinstance(betas, list) or len(betas) != 2:
        raise ValueError("optim.betas must have two numbers")

    train = _mapping(raw.get("train"), "train")
    _reject_unknown(
        train,
        {"max_steps", "max_epochs", "log_steps", "save_steps", "eval_steps", "keep_last", "interrupt_after_step"},
        "train",
    )
    max_steps = _as_int(train.get("max_steps"), "train.max_steps", minimum=1)
    warmup_steps = _as_int(optim.get("warmup_steps"), "optim.warmup_steps", minimum=0)
    if warmup_steps >= max_steps:
        raise ValueError("optim.warmup_steps must be < train.max_steps")
    scheduler_max_steps_raw = optim.get("scheduler_max_steps")
    if scheduler_max_steps_raw is None:
        scheduler_max_steps = None
    else:
        scheduler_max_steps = _as_int(scheduler_max_steps_raw, "optim.scheduler_max_steps", minimum=1)
        if scheduler_max_steps < max_steps:
            raise ValueError("optim.scheduler_max_steps must be >= train.max_steps")
        if warmup_steps >= scheduler_max_steps:
            raise ValueError("optim.warmup_steps must be < optim.scheduler_max_steps")
    interrupt = train.get("interrupt_after_step")
    if interrupt is None:
        interrupt_after_step = None
    else:
        interrupt_after_step = _as_int(interrupt, "train.interrupt_after_step", minimum=1)
        if interrupt_after_step >= max_steps:
            raise ValueError("train.interrupt_after_step must be < train.max_steps")
        if profile != "smoke":
            raise ValueError("interrupt_after_step is only valid for profile smoke")

    scoring = _mapping(raw.get("scoring"), "scoring")
    _reject_unknown(
        scoring,
        {
            "normalization",
            "wer_tokenizer",
            "wer_tokenizer_version",
            "wer_dictionary_sha256",
            "wer_hmm",
            "empty_penalty",
            "repeated_penalty",
            "reward_clip",
        },
        "scoring",
    )
    if scoring.get("normalization") != "zh-cer-v1":
        raise ValueError("scoring.normalization must be zh-cer-v1")
    wer_tokenizer = scoring.get("wer_tokenizer")
    wer_version = scoring.get("wer_tokenizer_version")
    wer_dictionary = scoring.get("wer_dictionary_sha256")
    if wer_tokenizer is None:
        if wer_version is not None or wer_dictionary is not None:
            raise ValueError("null wer_tokenizer requires null version and dictionary hash")
    else:
        _as_str(wer_tokenizer, "scoring.wer_tokenizer")
        _as_str(wer_version, "scoring.wer_tokenizer_version")
        _hex(wer_dictionary, "scoring.wer_dictionary_sha256", 64)
    reward_clip = scoring.get("reward_clip")
    if not isinstance(reward_clip, list) or len(reward_clip) != 2:
        raise ValueError("scoring.reward_clip must have two numbers")
    clip_low = _as_float(reward_clip[0], "scoring.reward_clip[0]")
    clip_high = _as_float(reward_clip[1], "scoring.reward_clip[1]")
    if clip_low >= clip_high:
        raise ValueError("scoring.reward_clip must be increasing")

    decode = _mapping(raw.get("decode"), "decode")
    _reject_unknown(decode, {"eval", "rollout"}, "decode")
    decode_eval = _decode_block(_mapping(decode.get("eval"), "decode.eval"), "decode.eval", sampled=False)
    decode_rollout = _decode_block(_mapping(decode.get("rollout"), "decode.rollout"), "decode.rollout", sampled=True)

    gate = _mapping(raw.get("gate"), "gate")
    gate_fields = {
        "min_rows_per_scenario",
        "clean_cer_increase_max",
        "clean_cer_increase_vs_base_max",
        "degraded_macro_increase_max",
        "scenario_cer_increase_max",
        "min_improved_degraded_scenarios",
        "empty_rate_max",
        "repeated_rate_max",
        "long_rate_max",
        "truncation_rate_max",
        "valid_output_rate_min",
        "min_preference_accuracy",
        "min_val_pairs",
        "min_reward_improvement",
        "raw_kl_max",
        "reward_drop_stop",
    }
    _reject_unknown(gate, gate_fields, "gate")
    parsed_gate = {
        "min_rows_per_scenario": _as_int(gate.get("min_rows_per_scenario"), "gate.min_rows_per_scenario", minimum=1),
        "clean_cer_increase_max": _as_float(gate.get("clean_cer_increase_max"), "gate.clean_cer_increase_max", minimum=0.0),
        "clean_cer_increase_vs_base_max": _as_float(
            gate.get("clean_cer_increase_vs_base_max"), "gate.clean_cer_increase_vs_base_max", minimum=0.0
        ),
        "degraded_macro_increase_max": _as_float(
            gate.get("degraded_macro_increase_max"), "gate.degraded_macro_increase_max"
        ),
        "scenario_cer_increase_max": _as_float(gate.get("scenario_cer_increase_max"), "gate.scenario_cer_increase_max", minimum=0.0),
        "min_improved_degraded_scenarios": _as_int(
            gate.get("min_improved_degraded_scenarios"), "gate.min_improved_degraded_scenarios", minimum=0
        ),
        "empty_rate_max": _as_float(gate.get("empty_rate_max"), "gate.empty_rate_max", minimum=0.0, maximum=1.0),
        "repeated_rate_max": _as_float(gate.get("repeated_rate_max"), "gate.repeated_rate_max", minimum=0.0, maximum=1.0),
        "long_rate_max": _as_float(gate.get("long_rate_max"), "gate.long_rate_max", minimum=0.0, maximum=1.0),
        "truncation_rate_max": _as_float(gate.get("truncation_rate_max"), "gate.truncation_rate_max", minimum=0.0, maximum=1.0),
        "valid_output_rate_min": _as_float(gate.get("valid_output_rate_min"), "gate.valid_output_rate_min", minimum=0.0, maximum=1.0),
        "min_preference_accuracy": _as_float(
            gate.get("min_preference_accuracy"), "gate.min_preference_accuracy", minimum=0.0, maximum=1.0
        ),
        "min_val_pairs": _as_int(gate.get("min_val_pairs"), "gate.min_val_pairs", minimum=1),
        "min_reward_improvement": _as_float(gate.get("min_reward_improvement"), "gate.min_reward_improvement"),
        "raw_kl_max": _as_float(gate.get("raw_kl_max"), "gate.raw_kl_max", minimum=0.0),
        "reward_drop_stop": _as_float(gate.get("reward_drop_stop"), "gate.reward_drop_stop", minimum=0.0),
    }
    if raw.get("dpo") is not None or raw.get("rl") is not None:
        raise ValueError("SFT config requires dpo: null and rl: null")
    baseline = _load_baseline(raw.get("baseline"), profile, path)

    resolved = {
        "schema_version": TRAIN_SCHEMA,
        "stage": "sft",
        "profile": profile,
        "seed": seed,
        "model": {
            "name": _as_str(model.get("name"), "model.name"),
            "revision": _hex(model.get("revision"), "model.revision", 40),
            "processor_sha256": _hex(model.get("processor_sha256"), "model.processor_sha256", 64),
            "parent_release": None,
            "parent_release_sha256": None,
        },
        "runtime": {
            "environment_lock_sha256": _hex(runtime.get("environment_lock_sha256"), "runtime.environment_lock_sha256", 64),
            "dtype": "bfloat16",
            "world_size": 1,
            "attention": "sdpa",
            "gradient_checkpointing": _as_bool(runtime.get("gradient_checkpointing"), "runtime.gradient_checkpointing"),
            "num_workers": 0,
            "allow_dirty": _as_bool(runtime.get("allow_dirty"), "runtime.allow_dirty"),
            "tf32": False,
            "deterministic_algorithms": _as_bool(runtime.get("deterministic_algorithms"), "runtime.deterministic_algorithms"),
            "max_runtime_minutes": _as_int(runtime.get("max_runtime_minutes"), "runtime.max_runtime_minutes", minimum=1),
            "shutdown_margin_minutes": _as_int(
                runtime.get("shutdown_margin_minutes"), "runtime.shutdown_margin_minutes", minimum=0
            ),
            "min_free_gpu_fraction": _as_float(
                runtime.get("min_free_gpu_fraction"), "runtime.min_free_gpu_fraction", minimum=0.0, maximum=1.0
            ),
            "min_free_disk_gb": _as_float(runtime.get("min_free_disk_gb"), "runtime.min_free_disk_gb", minimum=0.0),
            "min_free_ram_gb": _as_float(runtime.get("min_free_ram_gb"), "runtime.min_free_ram_gb", minimum=0.0),
        },
        "lora": {
            "rank": _as_int(lora.get("rank"), "lora.rank", minimum=1),
            "alpha": _as_int(lora.get("alpha"), "lora.alpha", minimum=1),
            "dropout": _as_float(lora.get("dropout"), "lora.dropout", minimum=0.0, maximum=1.0),
            "scope": "decoder_attention_only",
            "projections": list(PROJECTIONS),
        },
        "data": {
            "train_manifest": _as_str(data.get("train_manifest"), "data.train_manifest"),
            "val_manifest": _as_str(data.get("val_manifest"), "data.val_manifest"),
            "manifest_metadata": _as_str(data.get("manifest_metadata"), "data.manifest_metadata"),
            "max_audio_seconds": _as_float(data.get("max_audio_seconds"), "data.max_audio_seconds", minimum=0.1),
            "max_transcript_tokens": _as_int(data.get("max_transcript_tokens"), "data.max_transcript_tokens", minimum=1),
            "max_input_tokens": _as_int(data.get("max_input_tokens"), "data.max_input_tokens", minimum=1),
            "overlength": "reject",
            "sampler": "seeded_shuffle",
        },
        "batch": {
            "micro_batch_size": 1,
            "gradient_accumulation_steps": _as_int(
                batch.get("gradient_accumulation_steps"), "batch.gradient_accumulation_steps", minimum=1
            ),
            "eval_batch_size": 1,
        },
        "optim": {
            "name": "adamw",
            "learning_rate": _as_float(optim.get("learning_rate"), "optim.learning_rate", minimum=0.0),
            "betas": [
                _as_float(betas[0], "optim.betas[0]", minimum=0.0, maximum=1.0),
                _as_float(betas[1], "optim.betas[1]", minimum=0.0, maximum=1.0),
            ],
            "eps": _as_float(optim.get("eps"), "optim.eps", minimum=0.0),
            "weight_decay": _as_float(optim.get("weight_decay"), "optim.weight_decay", minimum=0.0),
            "max_grad_norm": _as_float(optim.get("max_grad_norm"), "optim.max_grad_norm", minimum=0.0),
            "scheduler": "cosine",
            "warmup_steps": warmup_steps,
            "min_lr_ratio": _as_float(optim.get("min_lr_ratio"), "optim.min_lr_ratio", minimum=0.0, maximum=1.0),
            **({"scheduler_max_steps": scheduler_max_steps} if scheduler_max_steps is not None else {}),
        },
        "train": {
            "max_steps": max_steps,
            "max_epochs": _as_int(train.get("max_epochs"), "train.max_epochs", minimum=1),
            "log_steps": _as_int(train.get("log_steps"), "train.log_steps", minimum=1),
            "save_steps": _as_int(train.get("save_steps"), "train.save_steps", minimum=1),
            "eval_steps": _as_int(train.get("eval_steps"), "train.eval_steps", minimum=1),
            "keep_last": _as_int(train.get("keep_last"), "train.keep_last", minimum=1),
            "interrupt_after_step": interrupt_after_step,
        },
        "scoring": {
            "normalization": "zh-cer-v1",
            "wer_tokenizer": None if wer_tokenizer is None else str(wer_tokenizer),
            "wer_tokenizer_version": None if wer_version is None else str(wer_version),
            "wer_dictionary_sha256": None if wer_dictionary is None else str(wer_dictionary),
            "wer_hmm": _as_bool(scoring.get("wer_hmm"), "scoring.wer_hmm"),
            "empty_penalty": _as_float(scoring.get("empty_penalty"), "scoring.empty_penalty", minimum=0.0),
            "repeated_penalty": _as_float(scoring.get("repeated_penalty"), "scoring.repeated_penalty", minimum=0.0),
            "reward_clip": [clip_low, clip_high],
        },
        "decode": {"eval": decode_eval, "rollout": decode_rollout},
        "gate": parsed_gate,
        "baseline": baseline,
        "dpo": None,
        "rl": None,
    }
    if resolved["optim"]["learning_rate"] <= 0 or resolved["optim"]["eps"] <= 0 or resolved["optim"]["max_grad_norm"] <= 0:
        raise ValueError("learning_rate, eps, and max_grad_norm must be > 0")
    if resolved["runtime"]["shutdown_margin_minutes"] >= resolved["runtime"]["max_runtime_minutes"]:
        raise ValueError("shutdown_margin_minutes must be < max_runtime_minutes")
    save_points = resolved["train"]["max_steps"] // resolved["train"]["save_steps"]
    if profile == "pilot" and resolved["train"]["keep_last"] < save_points:
        raise ValueError("pilot keep_last must retain every save point")
    if not resolved["runtime"]["gradient_checkpointing"]:
        raise ValueError("runtime.gradient_checkpointing must be true for this runner")
    resolved["decode_eval_sha256"] = sha256_text(canonical_json(decode_eval))
    resolved["official_language"] = LANGUAGE_TO_OFFICIAL["zh"]
    resolved["yaml_sha256"] = sha256_file(path)
    return resolved


def _baseline_hash_fields(baseline: dict[str, Any] | None) -> dict[str, str] | None:
    if baseline is None:
        return None
    return {
        "sha256": baseline["sha256"],
        "manifest_sha256": baseline["manifest_sha256"],
        "decode_eval_sha256": baseline["decode_eval_sha256"],
        "predictions_sha256": baseline["predictions_sha256"],
    }


def _load_baseline(raw: Any, profile: str, config_path: Path) -> dict[str, Any] | None:
    """Load the pinned val baseline. Smoke may omit it. Pilot may not."""
    if raw is None:
        if profile == "pilot":
            raise ValueError("pilot requires baseline.metrics")
        return None
    block = _mapping(raw, "baseline")
    _reject_unknown(block, {"metrics"}, "baseline")
    metrics_path = Path(_as_str(block.get("metrics"), "baseline.metrics"))
    if not metrics_path.is_absolute():
        metrics_path = Path(__file__).resolve().parents[1] / metrics_path
    if not metrics_path.is_file():
        raise ValueError(f"baseline metrics file is missing: {metrics_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("baseline metrics must be an object")
    _reject_unknown(
        payload,
        {
            "schema_version",
            "split",
            "parent",
            "manifest_sha256",
            "predictions_sha256",
            "decode_eval_sha256",
            "samples",
            "by_scenario",
        },
        "baseline",
    )
    if payload.get("schema_version") != "qwen3-asr-baseline-v1":
        raise ValueError("baseline schema_version must be qwen3-asr-baseline-v1")
    if payload.get("split") != "val" or payload.get("parent") != "base":
        raise ValueError("baseline must be the val base")
    by_scenario = _mapping(payload.get("by_scenario"), "baseline.by_scenario")
    _reject_unknown(by_scenario, set(SCENARIOS), "baseline.by_scenario")
    parsed_scenarios = {}
    for name in SCENARIOS:
        row = _mapping(by_scenario.get(name), f"baseline.by_scenario.{name}")
        _reject_unknown(row, {"samples", "micro_cer"}, f"baseline.by_scenario.{name}")
        parsed_scenarios[name] = {
            "samples": _as_int(row.get("samples"), f"baseline.by_scenario.{name}.samples", minimum=1),
            "micro_cer": _as_float(row.get("micro_cer"), f"baseline.by_scenario.{name}.micro_cer", minimum=0.0),
        }
    return {
        "metrics": str(block["metrics"]),
        "sha256": sha256_file(metrics_path),
        "manifest_sha256": _hex(payload.get("manifest_sha256"), "baseline.manifest_sha256", 64),
        "predictions_sha256": _hex(payload.get("predictions_sha256"), "baseline.predictions_sha256", 64),
        "decode_eval_sha256": _hex(payload.get("decode_eval_sha256"), "baseline.decode_eval_sha256", 64),
        "samples": _as_int(payload.get("samples"), "baseline.samples", minimum=1),
        "by_scenario": parsed_scenarios,
    }


def semantic_config_sha256(resolved: dict[str, Any], manifest_sha256: dict[str, str]) -> str:
    """Hash the training contract. Local paths and the resume flag stay outside."""
    payload = {
        "schema_version": resolved["schema_version"],
        "stage": resolved["stage"],
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
        "baseline": _baseline_hash_fields(resolved["baseline"]),
        "dpo": None,
        "rl": None,
        "decode_eval_sha256": resolved["decode_eval_sha256"],
    }
    return sha256_text(canonical_json(payload))
