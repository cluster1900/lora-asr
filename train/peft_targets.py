"""PEFT LoRA target 匹配工具。

Qwen3-ASR 的第一版 LoRA 只允许命中 audio tower 的 99 个目标模块。这里把
target 匹配逻辑独立出来，训练脚本、兼容性检查和后续测试都可以复用同一套
正则规则。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MatchedTarget:
    """一条被 LoRA 正则命中的模块记录。"""

    raw_name: str
    prefixed_name: str
    class_name: str
    weight_shape: str
    lora_params: int


def prefixed_module_name(raw_name: str, root_prefix: str = "model") -> str:
    """把底层模型的模块名补成探测输出中的全路径形式。

    探测阶段的根节点是 `model`，训练阶段实际包的是内部 `model.thinker`。
    因此训练脚本可以传入 `root_prefix="model.thinker"`，让相对 thinker 的
    `audio_tower.*` 模块仍能匹配配置中的 `model.thinker.audio_tower.*` 正则。
    """
    return root_prefix if raw_name == "" else f"{root_prefix}.{raw_name}"


def weight_shape(module: Any) -> str:
    """返回模块 weight shape，缺失时返回空字符串。"""
    weight = getattr(module, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape is None:
        return ""
    return "x".join(str(dim) for dim in shape)


def estimate_lora_params(module: Any, rank: int) -> int:
    """按 LoRA A/B 矩阵估算单个线性层的可训练参数量。"""
    weight = getattr(module, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape is None or len(shape) != 2:
        return 0
    out_dim, in_dim = int(shape[0]), int(shape[1])
    return int(rank) * (in_dim + out_dim)


def match_lora_targets(
    model: Any,
    lora_config: dict[str, Any],
    root_prefix: str = "model",
) -> list[MatchedTarget]:
    """使用配置中的 include/exclude regex 匹配底层模型 target。

    配置正则使用探测输出里的全路径；真实 PEFT 模型收到的是相对当前训练
    root 的 raw module name，因此这里同时保留两种名称。
    """
    include_regex = [re.compile(pattern) for pattern in lora_config.get("include_regex", [])]
    exclude_regex = [re.compile(pattern) for pattern in lora_config.get("exclude_regex", [])]
    rank = int(lora_config.get("r", 8))

    matched: list[MatchedTarget] = []
    for raw_name, module in model.named_modules():
        full_name = prefixed_module_name(raw_name, root_prefix=root_prefix)
        if not any(pattern.search(full_name) for pattern in include_regex):
            continue
        if any(pattern.search(full_name) for pattern in exclude_regex):
            continue
        matched.append(
            MatchedTarget(
                raw_name=raw_name,
                prefixed_name=full_name,
                class_name=module.__class__.__name__,
                weight_shape=weight_shape(module),
                lora_params=estimate_lora_params(module, rank),
            )
        )
    return matched


def validate_lora_targets(matched: list[MatchedTarget], lora_config: dict[str, Any]) -> None:
    """校验 target 数量和禁止命中的模块路径。"""
    expected = int(lora_config.get("expected_target_count", 0))
    if expected and len(matched) != expected:
        raise ValueError(f"LoRA target count mismatch: expected={expected}, actual={len(matched)}")

    bad = [
        item.prefixed_name
        for item in matched
        if ".thinker.model.layers." in item.prefixed_name
        or "lm_head" in item.prefixed_name
        or ".audio_tower.conv2d" in item.prefixed_name
    ]
    if bad:
        preview = ", ".join(bad[:5])
        raise ValueError(f"Forbidden LoRA targets matched: {preview}")


def target_summary(matched: list[MatchedTarget]) -> dict[str, Any]:
    """生成可保存的 target 摘要。"""
    return {
        "count": len(matched),
        "estimated_lora_params": sum(item.lora_params for item in matched),
        "targets": [
            {
                "raw_name": item.raw_name,
                "prefixed_name": item.prefixed_name,
                "class_name": item.class_name,
                "weight_shape": item.weight_shape,
                "lora_params": item.lora_params,
            }
            for item in matched
        ],
    }
