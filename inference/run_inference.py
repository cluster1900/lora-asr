#!/usr/bin/env python3
"""Greedy Qwen3-ASR inference with per-row errors.

The project path calls the official Transformers model.generate with
decode.eval. transcribe() is recorded beside it so Phase 0 can see whether
the two texts match. One bad audio file is written and the batch continues.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.jsonl_io import read_jsonl  # noqa: E402
from evaluation.zh_cer import normalize_zh_cer_v1  # noqa: E402
from inference.config import load_infer_config  # noqa: E402

PREDICTION_SCHEMA = "qwen3-asr-prediction-v1"
ASR_TEXT_TAG = "<asr_text>"


@dataclass
class GreedyResult:
    """One successful decode, including the official transcribe() text."""

    prediction_raw: str
    generation_raw: str
    token_ids: list[int]
    stop_reason: str
    transcribe_text: str
    model_revision: str | None
    processor_sha256: str | None


class InferenceBackend(Protocol):
    """The small surface tests replace. The Qwen backend implements it."""

    model_name: str
    model_revision: str | None
    processor_sha256: str | None

    def transcribe_file(self, audio_path: Path) -> GreedyResult:
        """Decode one existing audio file."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_model_revision(expected: str | None, actual: str | None) -> str | None:
    """Fail when a pinned config revision does not match the loaded checkpoint."""
    if not expected:
        return actual
    if actual != expected:
        raise ValueError(f"loaded model revision {actual} does not match config revision {expected}")
    return actual


def parse_generation(visible: str) -> str:
    """Take the transcript after <asr_text> without collapsing repetitions."""
    text = visible.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    if ASR_TEXT_TAG in text:
        text = text.split(ASR_TEXT_TAG, 1)[1]
    return text.strip()


def stop_reason_from_ids(token_ids: list[int], eos_id: int | None, max_new_tokens: int) -> str:
    """EOS ends the sequence. Filling max_new_tokens without EOS is truncation."""
    if eos_id is not None and token_ids and token_ids[-1] == eos_id:
        return "eos"
    if len(token_ids) >= max_new_tokens:
        return "length"
    return "other"


def resolve_audio_path(manifest_path: Path, audio: str) -> Path:
    """Resolve a manifest-relative POSIX path and reject traversal."""
    if not isinstance(audio, str) or not audio or audio.startswith("/") or "\\" in audio:
        raise ValueError(f"audio path must be relative and POSIX: {audio!r}")
    parts = Path(audio).parts
    if ".." in parts:
        raise ValueError(f"audio path must not contain '..': {audio}")
    return (manifest_path.parent / audio).resolve()


class Qwen3AsrBackend:
    """Official qwen-asr Transformers backend. Imported only when constructed."""

    def __init__(self, config: dict[str, Any]) -> None:
        import numpy as np
        import torch
        from qwen_asr import Qwen3ASRModel

        self._np = np
        self._torch = torch
        self._config = config
        decode = config["decode_eval"]
        seed = int(config["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        use_cuda = torch.cuda.is_available()
        dtype = torch.bfloat16 if use_cuda and torch.cuda.is_bf16_supported() else torch.float32
        self.model_name = str(config["model"]["name"])
        self._asr = Qwen3ASRModel.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map="cuda:0" if use_cuda else "cpu",
            max_new_tokens=int(decode["max_new_tokens"]),
            max_inference_batch_size=1,
        )
        revision = getattr(getattr(self._asr.model, "config", None), "_commit_hash", None)
        loaded = revision if isinstance(revision, str) else None
        self.model_revision = require_model_revision(config["model"]["revision"], loaded)
        self.processor_sha256 = None
        tokenizer = self._asr.processor.tokenizer
        eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._eos_id = eos_id if isinstance(eos_id, int) and eos_id >= 0 else tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        self._pad_id = pad_id if isinstance(pad_id, int) else None

    def transcribe_file(self, audio_path: Path) -> GreedyResult:
        from qwen_asr.inference.utils import normalize_audios

        torch = self._torch
        decode = self._config["decode_eval"]
        wavs = normalize_audios(str(audio_path))
        waveform = wavs[0]
        context = decode["context"]
        language = self._config["official_language"]
        processor = self._asr.processor
        messages = [
            {"role": "system", "content": context or ""},
            {"role": "user", "content": [{"type": "audio", "audio": ""}]},
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt = prompt + f"language {language}{ASR_TEXT_TAG}"
        inputs = processor(text=[prompt], audio=[waveform], return_tensors="pt", padding=True)
        inputs = inputs.to(self._asr.model.device).to(self._asr.model.dtype)
        generate_kwargs: dict[str, Any] = {
            "do_sample": False,
            "num_beams": int(decode["num_beams"]),
            "max_new_tokens": int(decode["max_new_tokens"]),
            "repetition_penalty": float(decode["repetition_penalty"]),
        }
        if int(decode["no_repeat_ngram_size"]) > 0:
            generate_kwargs["no_repeat_ngram_size"] = int(decode["no_repeat_ngram_size"])
        with torch.no_grad():
            generated = self._asr.model.generate(**inputs, **generate_kwargs)
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        prompt_width = int(inputs["input_ids"].shape[1])
        new_tokens = sequences[0, prompt_width:].detach().cpu().tolist()
        token_ids: list[int] = []
        for token in new_tokens:
            token_int = int(token)
            if self._pad_id is not None and token_int == self._pad_id:
                break
            token_ids.append(token_int)
            if self._eos_id is not None and token_int == int(self._eos_id):
                break
        reason = stop_reason_from_ids(token_ids, None if self._eos_id is None else int(self._eos_id), int(decode["max_new_tokens"]))
        token_tensor = torch.tensor([token_ids], dtype=torch.long)
        generation_raw = processor.batch_decode(
            token_tensor,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        visible = processor.batch_decode(
            token_tensor,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        official = self._asr.transcribe(
            audio=str(audio_path),
            context=context,
            language=language,
            return_time_stamps=False,
        )
        return GreedyResult(
            prediction_raw=parse_generation(visible),
            generation_raw=generation_raw,
            token_ids=token_ids,
            stop_reason=reason,
            transcribe_text=str(official[0].text),
            model_revision=self.model_revision,
            processor_sha256=self.processor_sha256,
        )


def prediction_row(
    manifest_row: dict[str, Any],
    *,
    config: dict[str, Any],
    audio_sha256: str | None,
    result: GreedyResult | None,
    error: str | None,
    error_type: str | None,
) -> dict[str, Any]:
    """Build one JSONL row. result and error are mutually exclusive."""
    raw = result.prediction_raw if result is not None else ""
    return {
        "schema_version": PREDICTION_SCHEMA,
        "sample_id": manifest_row.get("sample_id"),
        "audio": manifest_row.get("audio"),
        "audio_sha256": audio_sha256,
        "language": manifest_row.get("language"),
        "scenario": manifest_row.get("scenario"),
        "source_utterance_id": manifest_row.get("source_utterance_id"),
        "text": manifest_row.get("text"),
        "prediction_raw": raw,
        "prediction_normalized": normalize_zh_cer_v1(raw),
        "generation_raw": None if result is None else result.generation_raw,
        "token_ids": None if result is None else result.token_ids,
        "stop_reason": "error" if result is None else result.stop_reason,
        "error": error,
        "error_type": error_type,
        "transcribe_text": None if result is None else result.transcribe_text,
        "transcribe_matches_greedy": None if result is None else result.transcribe_text == result.prediction_raw,
        "model": config["model"]["name"],
        "model_revision": None if result is None else result.model_revision,
        "adapter": None,
        "processor_sha256": None if result is None else result.processor_sha256,
        "decode_eval_sha256": config["decode_eval_sha256"],
        "seed": config["seed"],
    }


def run_rows(
    manifest_path: Path,
    manifest_rows: list[dict[str, Any]],
    backend: InferenceBackend,
    config: dict[str, Any],
    output_path: Path,
) -> int:
    """Write every row, then return 1 when any row failed and 0 otherwise."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            try:
                audio_path = resolve_audio_path(manifest_path, str(row.get("audio")))
                if not audio_path.is_file():
                    raise FileNotFoundError(f"audio file does not exist: {row.get('audio')}")
                result = backend.transcribe_file(audio_path)
                payload = prediction_row(
                    row,
                    config=config,
                    audio_sha256=sha256_file(audio_path),
                    result=result,
                    error=None,
                    error_type=None,
                )
            except Exception as exc:  # one bad sample must not abort the batch
                failures += 1
                payload = prediction_row(
                    row,
                    config=config,
                    audio_sha256=None,
                    result=None,
                    error=f"{type(exc).__name__}: {exc}",
                    error_type=type(exc).__name__,
                )
                payload["error_trace"] = traceback.format_exc(limit=8)[-2000:]
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            handle.flush()
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用 decode.eval 的 greedy 设置转写 JSONL manifest")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", default="eval", choices=["eval"])
    parser.add_argument("--seed", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_infer_config(args.config)
        if args.seed != config["seed"]:
            raise ValueError(f"CLI seed {args.seed} does not match config seed {config['seed']}")
        if args.mode != "eval":
            raise ValueError("Phase 0 only implements --mode eval")
        rows = read_jsonl(args.manifest)
        backend: InferenceBackend = Qwen3AsrBackend(config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"run_inference failed before decoding: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"run_inference could not load the model: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return run_rows(args.manifest, rows, backend, config, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
