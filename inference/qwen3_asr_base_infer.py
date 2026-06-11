#!/usr/bin/env python3
"""Qwen3-ASR baseline 推理入口。

读取评测 JSONL manifest，对每条音频调用 Qwen3-ASR-1.7B 生成转写，并把
预测结果写成 JSONL。这个脚本只负责 baseline 推理，不加载 LoRA，也不做
router。后续 LoRA/router 推理应复用相同的输入输出约定。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


QWEN_LANGUAGE_ALIASES = {
    "en": "English",
    "english": "English",
    "zh": "Chinese",
    "cn": "Chinese",
    "chinese": "Chinese",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL manifest，跳过空行。"""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出 JSONL，并自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_audio_path(audio: str, manifest_path: Path, audio_root: str | None) -> Path:
    """解析音频路径。

    优先级：
    1. manifest 中的绝对路径
    2. `--audio-root` + manifest 中的相对路径
    3. manifest 所在目录 + 相对路径
    4. 当前工作目录 + 相对路径
    """
    audio_path = Path(audio).expanduser()
    if audio_path.is_absolute():
        return audio_path

    candidates: list[Path] = []
    if audio_root:
        candidates.append(Path(audio_root).expanduser() / audio_path)
    candidates.append(manifest_path.parent / audio_path)
    candidates.append(Path.cwd() / audio_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_torch_dtype(dtype: str) -> Any:
    """把命令行 dtype 转成 qwen-asr 可接受的 torch dtype。

    `auto` 表示不显式传 dtype，交给 qwen-asr 使用默认策略。Colab Free
    常见 T4 GPU 支持 float16，不支持 bfloat16，因此 baseline 默认使用
    float16。
    """
    import torch

    if dtype == "auto":
        return None
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float16":
        return torch.float16
    if dtype == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def load_model(
    model_id: str,
    dtype: str,
    device_map: str,
    max_inference_batch_size: int,
    max_new_tokens: int,
) -> Any:
    """加载 Qwen3-ASR 模型。

    官方 qwen-asr 包提供 `Qwen3ASRModel.from_pretrained`，其 `transcribe`
    接口可以直接接收本地音频路径。这里不再走通用聊天模板，避免把 ASR
    baseline 做成 prompt 工程。
    """
    from qwen_asr import Qwen3ASRModel

    kwargs: dict[str, Any] = {
        "device_map": device_map,
        "max_inference_batch_size": max_inference_batch_size,
        "max_new_tokens": max_new_tokens,
    }
    torch_dtype = resolve_torch_dtype(dtype)
    if torch_dtype is not None:
        kwargs["dtype"] = torch_dtype

    model = Qwen3ASRModel.from_pretrained(model_id, **kwargs)
    if hasattr(model, "eval"):
        model.eval()
    return model


def normalize_language(language: str | None) -> str | None:
    """规范化语言参数。

    Qwen3-ASR 官方示例使用 `None` 做自动识别，使用 `"English"` 这类名称
    强制指定语言。本项目的 MVP 音频是英文，因此 notebook 默认传
    `"English"`；如果用户传 `auto`，这里会转成 `None`。
    """
    if language is None:
        return None

    normalized = language.strip()
    if normalized.lower() in {"", "auto", "none", "null"}:
        return None
    return QWEN_LANGUAGE_ALIASES.get(normalized.lower(), normalized)


def pick_language(cli_language: str, item: dict[str, Any]) -> str | None:
    """为单条样本选择语言。

    `--language manifest` 会读取样本中的 `language` 字段；其他值直接作为
    全局语言设置。这样后续多语言 manifest 不需要改脚本。
    """
    if cli_language == "manifest":
        return normalize_language(item.get("language"))
    return normalize_language(cli_language)


def first_result(result: Any) -> Any:
    """从 qwen-asr 返回值中取出单条结果。"""
    if isinstance(result, (list, tuple)):
        return result[0] if result else None
    return result


def result_text(result: Any) -> str:
    """从 qwen-asr result 对象中提取转写文本。"""
    if result is None:
        return ""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, dict):
        for key in ("text", "transcription", "prediction", "content"):
            if key in result and result[key] is not None:
                return str(result[key]).strip()
        return json.dumps(result, ensure_ascii=False)
    text = getattr(result, "text", None)
    if text is not None:
        return str(text).strip()
    return str(result).strip()


def result_language(result: Any) -> str:
    """从 qwen-asr result 对象中提取识别语言。"""
    if result is None:
        return ""
    if isinstance(result, dict):
        return str(result.get("language") or "")
    return str(getattr(result, "language", "") or "")


def transcribe_one(model: Any, audio_path: Path, language: str | None) -> tuple[str, str]:
    """对单条音频运行 Qwen3-ASR baseline 转写。"""
    result = model.transcribe(
        audio=str(audio_path),
        language=language,
    )
    item = first_result(result)
    return result_text(item), result_language(item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Input JSONL with audio/answer fields.")
    parser.add_argument("--output-jsonl", required=True, help="Prediction JSONL output path.")
    parser.add_argument("--model-id", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--audio-root", default=None, help="Optional root for relative audio paths.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--max-inference-batch-size",
        type=int,
        default=1,
        help="qwen-asr inference batch limit. Keep small on Colab Free to reduce OOM risk.",
    )
    parser.add_argument("--dtype", default="float16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument(
        "--language",
        default="English",
        help="Use English/Chinese/etc, auto for language detection, or manifest for per-row language.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only process first N rows when > 0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser()
    output_path = Path(args.output_jsonl).expanduser()
    rows = read_jsonl(manifest_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    # Hugging Face gated/private models can be accessed through HF_TOKEN/HUGGING_FACE_HUB_TOKEN.
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("[auth] Hugging Face token detected in environment.")

    print(f"[load] model={args.model_id}")
    model = load_model(
        model_id=args.model_id,
        dtype=args.dtype,
        device_map=args.device_map,
        max_inference_batch_size=args.max_inference_batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    outputs: list[dict[str, Any]] = []
    for idx, item in enumerate(rows, start=1):
        started = time.perf_counter()
        out = dict(item)
        audio = item.get("audio") or item.get("audio_path")
        if not audio:
            out["prediction"] = ""
            out["error"] = "missing audio/audio_path"
            outputs.append(out)
            continue

        audio_path = resolve_audio_path(str(audio), manifest_path, args.audio_root)
        out["resolved_audio"] = str(audio_path)
        try:
            if not audio_path.exists():
                raise FileNotFoundError(str(audio_path))
            language = pick_language(args.language, item)
            out["prediction"], out["predicted_language"] = transcribe_one(model, audio_path, language)
            out["language_request"] = language or "auto"
            out["error"] = ""
        except Exception as exc:  # noqa: BLE001 - keep batch inference alive.
            out["prediction"] = ""
            out["predicted_language"] = ""
            out["language_request"] = pick_language(args.language, item) or "auto"
            out["error"] = f"{type(exc).__name__}: {exc}"
        out["infer_seconds"] = round(time.perf_counter() - started, 4)
        outputs.append(out)
        print(f"[{idx}/{len(rows)}] {out.get('scenario', '')} error={bool(out['error'])}")

    write_jsonl(output_path, outputs)
    print(f"[done] saved {len(outputs)} rows to {output_path}")


if __name__ == "__main__":
    main()
