"""Load and hash the Phase 0 inference config.

The semantic hash covers decode.eval and scoring. Local paths are not part of
the hash. Unknown fields fail before a model is loaded.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

INFER_SCHEMA = "qwen3-asr-infer-v1"
LANGUAGE_TO_OFFICIAL = {"zh": "Chinese"}


def decode_eval_sha256(decode_eval: dict[str, Any]) -> str:
    """Hash the stable JSON of decode.eval."""
    payload = json.dumps(decode_eval, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reject_unknown(section: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(section) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} fields: {unknown}")


def _as_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _as_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def load_infer_config(path: Path) -> dict[str, Any]:
    """Parse configs/inference YAML and return a normalized dict."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a mapping")
    _reject_unknown(
        raw,
        {"schema_version", "seed", "model", "scoring", "decode"},
        "config",
    )
    if raw.get("schema_version") != INFER_SCHEMA:
        raise ValueError(f"schema_version must be {INFER_SCHEMA}")
    seed = _as_int(raw.get("seed"), "seed")
    if seed < 0:
        raise ValueError("seed must be >= 0")

    model = raw.get("model")
    if not isinstance(model, dict):
        raise ValueError("model must be a mapping")
    _reject_unknown(model, {"name", "revision"}, "model")
    name = model.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("model.name must be a non-empty string")
    revision = model.get("revision")
    if revision is not None and not isinstance(revision, str):
        raise ValueError("model.revision must be a string or null")

    scoring = raw.get("scoring")
    if not isinstance(scoring, dict):
        raise ValueError("scoring must be a mapping")
    _reject_unknown(scoring, {"normalization", "wer_tokenizer"}, "scoring")
    if scoring.get("normalization") != "zh-cer-v1":
        raise ValueError("scoring.normalization must be zh-cer-v1")
    wer_tokenizer = scoring.get("wer_tokenizer")
    if wer_tokenizer is not None and not isinstance(wer_tokenizer, str):
        raise ValueError("scoring.wer_tokenizer must be a string or null")

    decode = raw.get("decode")
    if not isinstance(decode, dict):
        raise ValueError("decode must be a mapping")
    _reject_unknown(decode, {"eval"}, "decode")
    decode_eval = decode.get("eval")
    if not isinstance(decode_eval, dict):
        raise ValueError("decode.eval must be a mapping")
    _reject_unknown(
        decode_eval,
        {
            "do_sample",
            "num_beams",
            "max_new_tokens",
            "repetition_penalty",
            "no_repeat_ngram_size",
            "context",
            "language",
            "eos_policy",
        },
        "decode.eval",
    )
    if _as_bool(decode_eval.get("do_sample"), "decode.eval.do_sample"):
        raise ValueError("decode.eval.do_sample must be false")
    num_beams = _as_int(decode_eval.get("num_beams"), "decode.eval.num_beams")
    max_new_tokens = _as_int(decode_eval.get("max_new_tokens"), "decode.eval.max_new_tokens")
    no_repeat = _as_int(decode_eval.get("no_repeat_ngram_size"), "decode.eval.no_repeat_ngram_size")
    repetition_penalty = decode_eval.get("repetition_penalty")
    if isinstance(repetition_penalty, bool) or not isinstance(repetition_penalty, (int, float)):
        raise ValueError("decode.eval.repetition_penalty must be a number")
    if num_beams != 1 or max_new_tokens < 1 or no_repeat < 0:
        raise ValueError("decode.eval beams, max_new_tokens, or no_repeat_ngram_size is invalid")
    context = decode_eval.get("context")
    if not isinstance(context, str):
        raise ValueError("decode.eval.context must be a string")
    language = decode_eval.get("language")
    if language not in LANGUAGE_TO_OFFICIAL:
        raise ValueError("decode.eval.language must be zh in this MVP")
    if decode_eval.get("eos_policy") != "tokenizer_eos_only":
        raise ValueError("decode.eval.eos_policy must be tokenizer_eos_only")

    normalized_eval = {
        "do_sample": False,
        "num_beams": num_beams,
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": float(repetition_penalty),
        "no_repeat_ngram_size": no_repeat,
        "context": context,
        "language": language,
        "eos_policy": "tokenizer_eos_only",
    }
    return {
        "schema_version": INFER_SCHEMA,
        "seed": seed,
        "model": {"name": name, "revision": revision},
        "scoring": {"normalization": "zh-cer-v1", "wer_tokenizer": wer_tokenizer},
        "decode_eval": normalized_eval,
        "decode_eval_sha256": decode_eval_sha256(normalized_eval),
        "official_language": LANGUAGE_TO_OFFICIAL[language],
    }
