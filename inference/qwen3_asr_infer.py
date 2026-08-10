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
DTYPE = "bfloat16"
DEVICE_MAP = "cuda:0"
MAX_NEW_TOKENS = 256
MAX_INFERENCE_BATCH_SIZE = 1
LANGUAGE_NAMES = {
    "en": "English",
    "zh": "Chinese",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL manifest and reject malformed or non-object rows."""
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


def row_key(item: dict[str, Any]) -> str:
    """Return the stable resume key required for every manifest row."""
    value = str(item.get("sample_id") or "").strip()
    if not value:
        raise ValueError("Manifest row is missing sample_id")
    return f"sample_id:{value}"


def indexed_rows(rows: Iterable[dict[str, Any]]) -> list[tuple[int, str, dict[str, Any]]]:
    """Attach source indexes and fail early when sample IDs are duplicated."""
    result: list[tuple[int, str, dict[str, Any]]] = []
    seen: set[str] = set()
    for index, item in enumerate(rows):
        key = row_key(item)
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
                "Use a new output path."
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
    """Resolve relative audio against the explicit root, manifest, then CWD."""
    path = Path(audio).expanduser()
    if path.is_absolute():
        return path
    candidates: list[Path] = []
    if audio_root:
        candidates.append(Path(audio_root).expanduser() / path)
    candidates.extend((manifest_path.parent / path, Path.cwd() / path))
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def normalize_language(language: Any) -> str:
    """Map manifest language codes to names accepted by qwen-asr."""
    value = str(language or "").strip().lower()
    if value not in LANGUAGE_NAMES:
        raise ValueError(f"Invalid manifest language: {language!r}")
    return LANGUAGE_NAMES[value]


def attach_adapter(wrapper: Any, adapter_dir: Path) -> Any:
    """Attach the project's standard PEFT adapter to the outer ASR model."""
    from peft import PeftModel

    outer_model = getattr(wrapper, "model", None)
    if outer_model is None:
        raise AttributeError("Qwen3-ASR wrapper.model not found; cannot attach adapter")
    adapted = PeftModel.from_pretrained(outer_model, str(adapter_dir))
    if hasattr(adapted, "eval"):
        adapted.eval()
    wrapper.model = adapted
    return wrapper


def load_model(adapter_dir: str | None) -> Any:
    """Load the official qwen-asr wrapper and optionally one PEFT adapter."""
    import torch
    from qwen_asr import Qwen3ASRModel

    wrapper = Qwen3ASRModel.from_pretrained(
        DEFAULT_MODEL_ID,
        revision=DEFAULT_MODEL_REVISION,
        device_map=DEVICE_MAP,
        dtype=torch.bfloat16,
        max_inference_batch_size=MAX_INFERENCE_BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    if adapter_dir:
        wrapper = attach_adapter(wrapper, Path(adapter_dir).expanduser())
    if hasattr(wrapper, "eval"):
        wrapper.eval()
    return wrapper


def first_result(result: Any) -> Any:
    """Unwrap the first qwen-asr result while tolerating API shape variants."""
    if isinstance(result, (list, tuple)):
        return result[0] if result else None
    return result


def result_text(result: Any) -> str:
    """Extract normalized transcript text from string, mapping, or object output."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        return str(result.get("text") or "").strip()
    return str(getattr(result, "text", result) or "").strip()


def result_language(result: Any) -> str:
    """Extract the model-reported language without overriding the request."""
    if result is None:
        return ""
    if isinstance(result, dict):
        return str(result.get("language") or "")
    return str(getattr(result, "language", "") or "")


def transcribe_one(model: Any, audio_path: Path, language: str | None) -> tuple[str, str]:
    """Transcribe one audio file and return text plus detected language."""
    result = first_result(model.transcribe(audio=str(audio_path), language=language))
    return result_text(result), result_language(result)


def append_durable(handle: TextIO, row: dict[str, Any]) -> None:
    """Append one complete JSON line and make it durable before continuing."""
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def prediction_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Record the immutable model and decoding contract on every output row."""
    adapter = str(Path(args.adapter_dir).expanduser().resolve()) if args.adapter_dir else None
    return {
        "mode": "adapter" if adapter else "base",
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": DEFAULT_MODEL_REVISION,
        "adapter_dir": adapter,
        "dtype": DTYPE,
        "device_map": DEVICE_MAP,
        "decoding": {
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_inference_batch_size": MAX_INFERENCE_BATCH_SIZE,
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
    """Infer pending rows, persisting success or error after every sample.

    A sample-level exception is serialized into the row instead of terminating
    the batch. This makes long Colab runs resumable and keeps failures visible
    to the evaluator, where they are scored as full deletions.
    """
    written = 0
    metadata = prediction_metadata(args)
    for manifest_index, key, item in pending:
        started = time.perf_counter()
        out = dict(item)
        out.update(metadata)
        out["manifest_index"] = manifest_index
        out["inference_key"] = key
        out["language_request"] = ""
        out["prediction"] = ""
        out["predicted_language"] = ""
        out["error"] = ""

        try:
            language = normalize_language(item.get("language"))
            out["language_request"] = language
            audio = item.get("audio")
            if not audio:
                raise ValueError("missing audio")
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
    """Build the fixed base-or-adapter inference command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Input JSONL with audio/answer fields.")
    parser.add_argument("--output-jsonl", required=True, help="Incremental prediction JSONL.")
    parser.add_argument("--adapter-dir", default=None, help="Optional standard PEFT adapter directory.")
    parser.add_argument("--audio-root", default=None, help="Optional root for relative audio paths.")
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N rows.")
    parser.add_argument("--resume", action="store_true", help="Skip rows already durably written.")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments, accepting an explicit list for tests."""
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run resumable inference without loading the model when no rows remain."""
    args = parse_args(argv)
    manifest_path = Path(args.manifest).expanduser()
    output_path = Path(args.output_jsonl).expanduser()
    rows = read_jsonl(manifest_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    prepared = indexed_rows(rows)
    # Determine pending work before model loading; a completed resumed job exits
    # quickly and does not allocate GPU memory.
    done = completed_keys(output_path) if args.resume else set()
    pending = [entry for entry in prepared if entry[1] not in done]

    if args.adapter_dir and not Path(args.adapter_dir).expanduser().is_dir():
        raise FileNotFoundError(f"Adapter directory not found: {args.adapter_dir}")
    if not pending:
        print(f"[done] no pending rows; {len(done)} rows already present")
        return

    print(
        f"[load] model={DEFAULT_MODEL_ID}@{DEFAULT_MODEL_REVISION} "
        f"adapter={args.adapter_dir or 'none'} dtype={DTYPE}"
    )
    model = load_model(args.adapter_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume else "w"
    if args.resume:
        ensure_append_boundary(output_path)
    with output_path.open(mode, encoding="utf-8") as handle:
        written = infer_rows(model, pending, len(prepared), manifest_path, handle, args)
    print(f"[done] wrote={written} skipped={len(prepared) - written} output={output_path}")


if __name__ == "__main__":
    main()
