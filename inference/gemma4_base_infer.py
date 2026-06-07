#!/usr/bin/env python3
"""Gemma 4 base ASR 推理入口。

读取评测 JSONL manifest，对每条音频调用 Gemma 4 12B-it 生成转写，并把
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


DEFAULT_PROMPT = (
    "Transcribe the following speech segment in its original language. "
    "Only output the transcription."
)


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


def load_model(model_id: str, dtype: str, device_map: str, trust_remote_code: bool):
    """加载 Gemma 4 processor 和模型。

    官方模型卡推荐 `AutoModelForMultimodalLM`。为了兼容不同 Transformers
    版本，这里在类不存在时回退到 `AutoModelForImageTextToText`。
    """
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForMultimodalLM as AutoModelClass
    except ImportError:
        from transformers import AutoModelForImageTextToText as AutoModelClass

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)

    torch_dtype = dtype
    if dtype == "auto":
        torch_dtype = "auto"
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    elif dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "float32":
        torch_dtype = torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    model = AutoModelClass.from_pretrained(
        model_id,
        dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return processor, model


def move_inputs_to_device(inputs: Any, model: Any) -> Any:
    """把 processor 输出移动到模型所在设备。

    `device_map=auto` 时模型可能被切分，Transformers 通常仍能接受放在
    `model.device` 的输入。若模型没有 `device` 属性，则保持原样。
    """
    device = getattr(model, "device", None)
    if device is None:
        return inputs
    return inputs.to(device)


def parse_generated_text(processor: Any, response: str) -> str:
    """解析模型输出，优先使用 processor 自带解析逻辑。"""
    if hasattr(processor, "parse_response"):
        parsed = processor.parse_response(response)
        if isinstance(parsed, str):
            return parsed.strip()
        if isinstance(parsed, dict):
            for key in ("text", "content", "response"):
                if key in parsed:
                    return str(parsed[key]).strip()
        if isinstance(parsed, list):
            return " ".join(str(item) for item in parsed).strip()
        return str(parsed).strip()
    return response.strip()


def transcribe_one(
    processor: Any,
    model: Any,
    audio_path: Path,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> str:
    """对单条音频运行 Gemma 4 baseline 转写。"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "audio": str(audio_path)},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
    )
    inputs = move_inputs_to_device(inputs, model)
    input_len = inputs["input_ids"].shape[-1]

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        generation_kwargs["temperature"] = temperature

    outputs = model.generate(**inputs, **generation_kwargs)
    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    return parse_generated_text(processor, response)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Input JSONL with audio/answer fields.")
    parser.add_argument("--output-jsonl", required=True, help="Prediction JSONL output path.")
    parser.add_argument("--model-id", default="google/gemma-4-12B-it")
    parser.add_argument("--audio-root", default=None, help="Optional root for relative audio paths.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=0, help="Only process first N rows when > 0.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser()
    output_path = Path(args.output_jsonl).expanduser()
    rows = read_jsonl(manifest_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    # Hugging Face gated models can be accessed through HF_TOKEN/HUGGING_FACE_HUB_TOKEN.
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("[auth] Hugging Face token detected in environment.")

    print(f"[load] model={args.model_id}")
    processor, model = load_model(args.model_id, args.dtype, args.device_map, args.trust_remote_code)

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
            out["prediction"] = transcribe_one(
                processor=processor,
                model=model,
                audio_path=audio_path,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
            )
            out["error"] = ""
        except Exception as exc:  # noqa: BLE001 - keep batch inference alive.
            out["prediction"] = ""
            out["error"] = f"{type(exc).__name__}: {exc}"
        out["infer_seconds"] = round(time.perf_counter() - started, 4)
        outputs.append(out)
        print(f"[{idx}/{len(rows)}] {out.get('scenario', '')} error={bool(out['error'])}")

    write_jsonl(output_path, outputs)
    print(f"[done] saved {len(outputs)} rows to {output_path}")


if __name__ == "__main__":
    main()
