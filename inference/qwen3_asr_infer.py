#!/usr/bin/env python3
"""Unified BF16 inference for Qwen3-ASR base or one PEFT adapter.

Heavy ML dependencies are imported only while loading the model, so ``--help``
and the JSONL/resume utilities also work in a plain Python environment.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, TextIO


DEFAULT_MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
DEFAULT_MODEL_REVISION = "7278e1e70fe206f11671096ffdd38061171dd6e5"
LANGUAGE_ALIASES = {
    "en": "English",
    "english": "English",
    "zh": "Chinese",
    "cn": "Chinese",
    "chinese": "Chinese",
}
ROW_ID_FIELDS = ("sample_id", "id", "utterance_id")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def row_key(item: dict[str, Any], manifest_index: int) -> str:
    """Return a stable explicit ID, falling back to the zero-based row index."""
    for field in ROW_ID_FIELDS:
        value = item.get(field)
        if value is not None and str(value).strip():
            return f"{field}:{value}"
    return f"index:{manifest_index}"


def indexed_rows(rows: Iterable[dict[str, Any]]) -> list[tuple[int, str, dict[str, Any]]]:
    result: list[tuple[int, str, dict[str, Any]]] = []
    seen: set[str] = set()
    for index, item in enumerate(rows):
        key = row_key(item, index)
        if key in seen:
            raise ValueError(f"Duplicate manifest row identity: {key}")
        seen.add(key)
        result.append((index, key, item))
    return result


def completed_keys(path: Path) -> set[str]:
    """Read keys from a durable output file used by ``--resume``."""
    if not path.exists():
        return set()
    keys: set[str] = set()
    for row in read_jsonl(path):
        key = str(row.get("inference_key") or "").strip()
        if not key:
            raise ValueError(
                f"Resume output contains a row without inference_key: {path}. "
                "Use a new output path for legacy prediction files."
            )
        keys.add(key)
    return keys


def ensure_append_boundary(path: Path) -> None:
    """Ensure a valid resume file ends with a newline before appending."""
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb+") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) == b"\n":
            return
        handle.seek(0, os.SEEK_END)
        handle.write(b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def resolve_audio_path(audio: str, manifest_path: Path, audio_root: str | None) -> Path:
    path = Path(audio).expanduser()
    if path.is_absolute():
        return path
    candidates: list[Path] = []
    if audio_root:
        candidates.append(Path(audio_root).expanduser() / path)
    candidates.extend((manifest_path.parent / path, Path.cwd() / path))
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def normalize_language(language: Any) -> str | None:
    if language is None:
        return None
    value = str(language).strip()
    if value.lower() in {"", "auto", "none", "null"}:
        return None
    return LANGUAGE_ALIASES.get(value.lower(), value)


def pick_language(cli_language: str, item: dict[str, Any]) -> str | None:
    if cli_language.lower() == "manifest":
        return normalize_language(item.get("language"))
    return normalize_language(cli_language)


def resolve_torch_dtype(dtype: str) -> Any:
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype == "auto":
        return None
    return mapping[dtype]


def model_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    dtype = resolve_torch_dtype(args.dtype)
    kwargs: dict[str, Any] = {
        "revision": args.model_revision,
        "device_map": args.device_map,
        "max_inference_batch_size": args.max_inference_batch_size,
        "max_new_tokens": args.max_new_tokens,
    }
    if dtype is not None:
        kwargs["dtype"] = dtype
    return kwargs


def attach_adapter(wrapper: Any, adapter_dir: Path, merge_adapter: bool) -> Any:
    """Attach the new project's standard PEFT adapter to the outer ASR model."""
    from peft import PeftModel

    outer_model = getattr(wrapper, "model", None)
    if outer_model is None:
        raise AttributeError("Qwen3-ASR wrapper.model not found; cannot attach adapter")
    adapted = PeftModel.from_pretrained(outer_model, str(adapter_dir))
    if merge_adapter:
        adapted = adapted.merge_and_unload()
    if hasattr(adapted, "eval"):
        adapted.eval()
    wrapper.model = adapted
    return wrapper


def load_model(args: argparse.Namespace) -> Any:
    """Load the official qwen-asr wrapper and optionally one PEFT adapter."""
    from qwen_asr import Qwen3ASRModel

    kwargs = model_kwargs(args)
    try:
        wrapper = Qwen3ASRModel.from_pretrained(args.model_id, **kwargs)
    except TypeError:
        # Older qwen-asr/Transformers combinations used torch_dtype.
        retry_kwargs = dict(kwargs)
        if "dtype" in retry_kwargs:
            retry_kwargs["torch_dtype"] = retry_kwargs.pop("dtype")
        wrapper = Qwen3ASRModel.from_pretrained(args.model_id, **retry_kwargs)
    if args.adapter_dir:
        wrapper = attach_adapter(wrapper, Path(args.adapter_dir).expanduser(), args.merge_adapter)
    if hasattr(wrapper, "eval"):
        wrapper.eval()
    return wrapper


def first_result(result: Any) -> Any:
    if isinstance(result, (list, tuple)):
        return result[0] if result else None
    return result


def result_text(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        for field in ("text", "transcription", "prediction", "content"):
            if result.get(field) is not None:
                return str(result[field]).strip()
        return json.dumps(result, ensure_ascii=False)
    return str(getattr(result, "text", result) or "").strip()


def result_language(result: Any) -> str:
    if result is None:
        return ""
    if isinstance(result, dict):
        return str(result.get("language") or "")
    return str(getattr(result, "language", "") or "")


def transcribe_one(model: Any, audio_path: Path, language: str | None) -> tuple[str, str]:
    result = first_result(model.transcribe(audio=str(audio_path), language=language))
    return result_text(result), result_language(result)


def append_durable(handle: TextIO, row: dict[str, Any]) -> None:
    """Append one complete JSON line and make it durable before continuing."""
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def prediction_metadata(args: argparse.Namespace) -> dict[str, Any]:
    adapter = str(Path(args.adapter_dir).expanduser().resolve()) if args.adapter_dir else None
    return {
        "mode": "adapter" if adapter else "base",
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "adapter_dir": adapter,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "decoding": {
            "max_new_tokens": args.max_new_tokens,
            "max_inference_batch_size": args.max_inference_batch_size,
        },
    }


def infer_rows(
    model: Any,
    pending: Iterable[tuple[int, str, dict[str, Any]]],
    total: int,
    manifest_path: Path,
    output_handle: TextIO,
    args: argparse.Namespace,
) -> int:
    written = 0
    metadata = prediction_metadata(args)
    for manifest_index, key, item in pending:
        started = time.perf_counter()
        out = dict(item)
        out.update(metadata)
        out["manifest_index"] = manifest_index
        out["inference_key"] = key
        language = pick_language(args.language, item)
        out["language_request"] = language or "auto"
        out["prediction"] = ""
        out["predicted_language"] = ""
        out["error"] = ""

        try:
            audio = item.get("audio") or item.get("audio_path")
            if not audio:
                raise ValueError("missing audio/audio_path")
            audio_path = resolve_audio_path(str(audio), manifest_path, args.audio_root)
            out["resolved_audio"] = str(audio_path)
            if not audio_path.exists():
                raise FileNotFoundError(str(audio_path))
            out["prediction"], out["predicted_language"] = transcribe_one(
                model, audio_path, language
            )
        except Exception as exc:  # Keep a single bad sample from aborting the run.
            out["error"] = f"{type(exc).__name__}: {exc}"

        out["infer_seconds"] = round(time.perf_counter() - started, 4)
        append_durable(output_handle, out)
        written += 1
        print(
            f"[{manifest_index + 1}/{total}] key={key} "
            f"scenario={out.get('scenario', '')} error={bool(out['error'])}"
        )
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Input JSONL with audio/answer fields.")
    parser.add_argument("--output-jsonl", required=True, help="Incremental prediction JSONL.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--adapter-dir", default=None, help="Optional standard PEFT adapter directory.")
    parser.add_argument("--audio-root", default=None, help="Optional root for relative audio paths.")
    parser.add_argument("--dtype", default="bfloat16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-inference-batch-size", type=int, default=1)
    parser.add_argument(
        "--language",
        default="manifest",
        help="manifest for per-row en/zh, a language name, or auto.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N rows.")
    parser.add_argument("--resume", action="store_true", help="Skip rows already durably written.")
    parser.add_argument("--merge-adapter", action="store_true", help="Merge adapter after loading.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    manifest_path = Path(args.manifest).expanduser()
    output_path = Path(args.output_jsonl).expanduser()
    rows = read_jsonl(manifest_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    prepared = indexed_rows(rows)
    done = completed_keys(output_path) if args.resume else set()
    pending = [entry for entry in prepared if entry[1] not in done]

    if args.adapter_dir and not Path(args.adapter_dir).expanduser().is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {args.adapter_dir}")
    if not pending:
        print(f"[done] no pending rows; {len(done)} rows already present")
        return

    print(
        f"[load] model={args.model_id}@{args.model_revision} "
        f"adapter={args.adapter_dir or 'none'} dtype={args.dtype}"
    )
    model = load_model(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    if args.resume:
        ensure_append_boundary(output_path)
    with output_path.open(mode, encoding="utf-8") as handle:
        written = infer_rows(model, pending, len(prepared), manifest_path, handle, args)
    print(f"[done] wrote={written} skipped={len(prepared) - written} output={output_path}")


if __name__ == "__main__":
    main()
