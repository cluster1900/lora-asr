#!/usr/bin/env python3
"""Run single-GPU batch-1 ASR inference on JSONL manifests with Qwen3-ASR."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

# Ensure HF mirror is used if not explicitly overridden
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LANGUAGE_MAP: Dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "english": "English",
    "chinese": "Chinese",
    "cantonese": "Cantonese",
}

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    torch = None  # type: ignore
    HAVE_TORCH = False

try:
    from qwen_asr import Qwen3ASRModel
    HAVE_QWEN_ASR = True
except ImportError:
    Qwen3ASRModel = None  # type: ignore
    HAVE_QWEN_ASR = False

try:
    from peft import PeftModel
    HAVE_PEFT = True
except ImportError:
    PeftModel = None  # type: ignore
    HAVE_PEFT = False


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to input JSONL manifest (must contain sample_id, audio, text/answer, language).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSONL predictions.",
    )
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen3-ASR-1.7B",
        help="Model identifier or local checkpoint path (default: Qwen/Qwen3-ASR-1.7B).",
    )
    parser.add_argument(
        "--revision",
        default="7278e1e70fe206f11671096ffdd38061171dd6e5",
        help="Model commit SHA/revision.",
    )
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Path to LoRA adapter directory (optional).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if HAVE_TORCH and torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: cuda:0 or cpu).",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Model compute dtype (default: float16 for V100).",
    )
    parser.add_argument(
        "--attention",
        default="eager",
        help="Attention implementation (default: eager for V100).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size (default: 1, strict contract).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional maximum number of samples to evaluate.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not skip existing samples in output file.",
    )
    parser.add_argument(
        "--method",
        default=None,
        help="Inference method tag for prediction records (e.g. base, lora, sft_merged, dpo_merged, rl_merged). "
             "If not specified, automatically inferred from model/adapter.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Automatically run evaluation/eval_wer.py on output predictions.",
    )
    return parser.parse_args(argv)


def get_torch_dtype(dtype_str: str) -> Any:
    if not HAVE_TORCH:
        return None
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str, torch.float16)


def load_model(
    model_id: str,
    revision: Optional[str] = None,
    adapter_dir: Optional[str] = None,
    device: str = "cuda:0",
    dtype: str = "float16",
    attention: str = "eager",
) -> Any:
    """Load Qwen3-ASR model and optionally attach PEFT adapter."""
    if not HAVE_QWEN_ASR:
        raise RuntimeError("qwen_asr package is not installed. Please pip install qwen-asr.")
    if not HAVE_TORCH:
        raise RuntimeError("PyTorch is not installed.")

    torch_dtype = get_torch_dtype(dtype)

    # Qwen3ASRModel.from_pretrained kwargs
    load_kwargs: Dict[str, Any] = {
        "dtype": torch_dtype,
        "device_map": device,
        "attn_implementation": attention,
    }
    if revision:
        load_kwargs["revision"] = revision

    print(f"Loading Qwen3-ASR base model from {model_id} (revision={revision}, dtype={dtype}, attention={attention}, device={device})...")
    model = Qwen3ASRModel.from_pretrained(model_id, **load_kwargs)

    if adapter_dir:
        print(f"Loading PEFT adapter from {adapter_dir}...")
        if hasattr(model, "load_adapter"):
            model.load_adapter(adapter_dir)
        elif hasattr(model, "model") and hasattr(model.model, "thinker") and HAVE_PEFT:
            model.model.thinker = PeftModel.from_pretrained(model.model.thinker, adapter_dir)
        elif hasattr(model, "model") and HAVE_PEFT:
            model.model = PeftModel.from_pretrained(model.model, adapter_dir)
        else:
            raise RuntimeError(f"Cannot attach adapter to model type {type(model).__name__}")

    return model


def extract_prediction_text(result: Any) -> str:
    """Extract clean transcript string from Qwen3ASRModel transcribe return value."""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, list):
        if not result:
            return ""
        first = result[0]
        if isinstance(first, str):
            return first.strip()
        if hasattr(first, "text"):
            return str(first.text).strip()
        if isinstance(first, dict) and "text" in first:
            return str(first["text"]).strip()
        return str(first).strip()
    if hasattr(result, "text"):
        return str(result.text).strip()
    if isinstance(result, dict) and "text" in result:
        return str(result["text"]).strip()
    return str(result).strip()


def run_single_inference(
    model: Any,
    audio_path: str,
    language: Optional[str] = None,
) -> str:
    """Execute single audio transcription with Qwen3ASRModel."""
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Pass audio path and canonical language name to model
    kwargs: Dict[str, Any] = {}
    if language:
        full_lang = LANGUAGE_MAP.get(str(language).strip().lower())
        if full_lang:
            kwargs["language"] = full_lang

    try:
        raw_result = model.transcribe(audio=audio_path, **kwargs)
    except TypeError:
        # Fallback if language is not an accepted kwarg
        raw_result = model.transcribe(audio=audio_path)

    return extract_prediction_text(raw_result)


def load_existing_sample_ids(output_path: Path) -> Set[str]:
    """Read existing output file, sanitize any truncated or invalid lines, and return valid sample_ids."""
    sample_ids: Set[str] = set()
    if not output_path.is_file():
        return sample_ids

    valid_lines: List[str] = []
    had_corrupt = False
    with output_path.open("r", encoding="utf-8") as f:
        content = f.read()

    missing_trailing_newline = bool(content and not content.endswith("\n"))

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
            sid = data.get("sample_id")
            if sid is not None:
                sample_ids.add(str(sid))
                valid_lines.append(stripped)
            else:
                had_corrupt = True
        except Exception:
            had_corrupt = True

    # If truncated/corrupt lines or missing trailing newline was detected, cleanly rewrite
    if had_corrupt or missing_trailing_newline:
        with output_path.open("w", encoding="utf-8") as f:
            for vline in valid_lines:
                f.write(vline + "\n")
            f.flush()
            os.fsync(f.fileno())

    return sample_ids


def run_inference_on_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    model: Any,
    model_id: str,
    revision: Optional[str] = None,
    adapter_dir: Optional[str] = None,
    dtype: str = "float16",
    attention: str = "eager",
    max_samples: Optional[int] = None,
    resume: bool = True,
    transcribe_fn: Optional[Callable[[Any, str, Optional[str]], str]] = None,
    method: Optional[str] = None,
) -> tuple[int, int, int]:
    """Process manifest line by line, writing results and syncing immediately.

    Returns:
        (total_processed, successful_samples, failed_samples)
    """
    manifest_p = Path(manifest_path).resolve()
    output_p = Path(output_path).resolve()
    output_p.parent.mkdir(parents=True, exist_ok=True)

    if resume:
        existing_ids: Set[str] = load_existing_sample_ids(output_p)
        if existing_ids:
            print(f"Resuming: found {len(existing_ids)} already processed samples in {output_p}")
        out_mode = "a"
    else:
        existing_ids = set()
        out_mode = "w"

    if method is None:
        if adapter_dir:
            method = "lora"
        elif "sft" in str(model_id).lower():
            method = "sft_merged"
        elif "dpo" in str(model_id).lower():
            method = "dpo_merged"
        elif "rl" in str(model_id).lower():
            method = "rl_merged"
        else:
            method = "base"

    total_samples = 0
    success_count = 0
    failure_count = 0

    if transcribe_fn is None:
        transcribe_fn = run_single_inference

    with manifest_p.open("r", encoding="utf-8") as in_f, output_p.open(out_mode, encoding="utf-8") as out_f:
        for line_idx, line in enumerate(in_f, start=1):
            if max_samples is not None and total_samples >= max_samples:
                break

            line = line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
            except Exception as e:
                print(f"Skipping malformed JSON line {line_idx} in {manifest_p}: {e}", file=sys.stderr)
                continue

            sample_id = str(row.get("sample_id", f"idx_{line_idx}"))
            if sample_id in existing_ids:
                continue

            audio_path = row.get("audio", "")
            reference_text = str(row.get("text") if row.get("text") is not None else row.get("answer", ""))
            language = row.get("language", "en")
            scenario = row.get("scenario", "clean")
            condition_group = row.get("condition_group", "clean" if scenario == "clean" else "degraded")
            audio_origin = row.get("audio_origin", "clean")

            t0 = time.time()
            pred_text = ""
            err_msg = ""

            try:
                pred_text = transcribe_fn(model, audio_path, language)
                success_count += 1
            except Exception as exc:
                err_msg = f"{type(exc).__name__}: {str(exc)}"
                failure_count += 1
                print(f"Error inferring sample {sample_id} ({audio_path}): {err_msg}", file=sys.stderr)

            duration_sec = round(time.time() - t0, 4)

            record: Dict[str, Any] = {
                "sample_id": sample_id,
                "text": reference_text,
                "prediction": pred_text,
                "language": language,
                "scenario": scenario,
                "condition_group": condition_group,
                "audio_origin": audio_origin,
                "model_id": model_id,
                "model_revision": revision or "",
                "dtype": dtype,
                "attention": attention,
                "method": method,
                "adapter_dir": adapter_dir or "",
                "infer_seconds": duration_sec,
                "error": err_msg,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())

            existing_ids.add(sample_id)
            total_samples += 1

            if total_samples % 10 == 0:
                print(f"Progress: processed {total_samples} samples (success={success_count}, failed={failure_count})")

    print(f"Inference complete: {total_samples} samples processed ({success_count} success, {failure_count} failed).")
    return total_samples, success_count, failure_count


def run_evaluation(predictions_jsonl: Path) -> None:
    """Run evaluation/eval_wer.py on generated predictions and print summary."""
    eval_script = Path(__file__).resolve().parents[1] / "evaluation" / "eval_wer.py"
    if not eval_script.is_file():
        print(f"Warning: evaluation script not found at {eval_script}", file=sys.stderr)
        return

    output_dir = predictions_jsonl.parent / f"{predictions_jsonl.stem}_eval"
    print(f"\nRunning evaluation on {predictions_jsonl} -> {output_dir}...")

    # Import and run eval_wer directly
    try:
        from evaluation import eval_wer
        eval_wer.main([
            "--predictions-jsonl", str(predictions_jsonl),
            "--output-dir", str(output_dir),
        ])
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    model = load_model(
        model_id=args.model_id,
        revision=args.revision,
        adapter_dir=args.adapter_dir,
        device=args.device,
        dtype=args.dtype,
        attention=args.attention,
    )

    total, success, failed = run_inference_on_manifest(
        manifest_path=manifest_path,
        output_path=Path(args.output),
        model=model,
        model_id=args.model_id,
        revision=args.revision,
        adapter_dir=args.adapter_dir,
        dtype=args.dtype,
        attention=args.attention,
        max_samples=args.max_samples,
        resume=not args.no_resume,
        method=args.method,
    )

    if args.eval:
        run_evaluation(Path(args.output))


if __name__ == "__main__":
    main()
