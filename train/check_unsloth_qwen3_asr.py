#!/usr/bin/env python3
"""检查 Unsloth 是否兼容 Qwen3-ASR 训练入口。

本脚本只做兼容性检查，不做训练。它会尝试：

1. 导入 Unsloth。
2. 使用 `FastModel.from_pretrained` 加载 `Qwen/Qwen3-ASR-1.7B`。
3. 遍历加载后模型的 `named_modules()`。
4. 用训练配置里的 include/exclude regex 检查是否能精确定位 audio tower target。
5. 将检查结果写入 JSON，供后续决定使用 Unsloth 还是回退 Transformers + PEFT。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 配置。"""
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """写出 JSON 并自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def project_path(value: str) -> Path:
    """把相对路径解析到项目根目录。"""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def match_targets(module_names: list[str], include_regex: list[str], exclude_regex: list[str]) -> list[str]:
    """使用配置中的 include/exclude regex 匹配目标模块。"""
    includes = [re.compile(pattern) for pattern in include_regex]
    excludes = [re.compile(pattern) for pattern in exclude_regex]
    matched: list[str] = []
    for name in module_names:
        if any(pattern.search(name) for pattern in includes) and not any(
            pattern.search(name) for pattern in excludes
        ):
            matched.append(name)
    return matched


def load_with_unsloth(model_id: str, load_in_4bit: bool, max_seq_length: int) -> tuple[Any, Any]:
    """通过 Unsloth FastModel 加载模型。"""
    from unsloth import FastModel

    return FastModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=load_in_4bit,
        load_in_8bit=False,
        full_finetuning=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/qwen3_asr_lora_mvp.yaml")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--load-in-4bit", action="store_true", default=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="兼容性失败时返回非 0。默认只写 JSON 并正常退出，便于 Colab 继续查看结果。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = project_path(args.config)
    config = read_yaml(config_path)
    model_id = config["model"]["id"]
    probe_dir = project_path(config.get("probe", {}).get("output_dir", "outputs/lora_probe/qwen3_asr_1_7b"))
    output_json = project_path(args.output_json) if args.output_json else probe_dir / "unsloth_compatibility.json"

    lora_config = config.get("lora", {})
    include_regex = lora_config.get("include_regex", [])
    exclude_regex = lora_config.get("exclude_regex", [])
    expected_target_count = int(lora_config.get("expected_target_count", 0))

    result: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "config": str(config_path),
        "backend": config.get("model", {}).get("backend"),
        "compatible": False,
        "checks": {},
        "errors": [],
    }

    try:
        model, tokenizer = load_with_unsloth(
            model_id=model_id,
            load_in_4bit=args.load_in_4bit,
            max_seq_length=args.max_seq_length,
        )
        result["checks"]["unsloth_load"] = True
        result["model_class"] = model.__class__.__name__
        result["tokenizer_class"] = tokenizer.__class__.__name__ if tokenizer is not None else ""

        module_names = [name for name, _module in model.named_modules()]
        matched = match_targets(module_names, include_regex, exclude_regex)
        result["checks"]["named_modules"] = len(module_names)
        result["checks"]["matched_target_count"] = len(matched)
        result["matched_targets_preview"] = matched[:20]
        result["has_audio_tower"] = any(".audio_tower" in name for name in module_names)
        result["has_text_decoder_match"] = any(".thinker.model.layers." in name for name in matched)
        result["has_lm_head_match"] = any("lm_head" in name for name in matched)
        result["expected_target_count"] = expected_target_count

        result["compatible"] = (
            len(module_names) > 0
            and len(matched) == expected_target_count
            and result["has_audio_tower"]
            and not result["has_text_decoder_match"]
            and not result["has_lm_head_match"]
        )
    except Exception as exc:  # noqa: BLE001 - 需要完整记录兼容性失败原因。
        result["checks"]["unsloth_load"] = False
        result["errors"].append(
            {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=20),
            }
        )

    write_json(output_json, result)
    print(f"兼容性结果: {output_json}")
    print(f"compatible = {result['compatible']}")
    if args.strict and not result["compatible"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
