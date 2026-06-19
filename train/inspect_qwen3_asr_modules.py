#!/usr/bin/env python3
"""训练前探测 Qwen3-ASR 模块结构并生成 LoRA target 候选。

本脚本用于 LoRA MVP 训练前的“探测阶段”，不是训练入口。它会：

1. 通过官方 `qwen-asr` API 加载 `Qwen/Qwen3-ASR-1.7B`。
2. 从 wrapper 中寻找真实的 `torch.nn.Module` 根节点。
3. 导出 `named_modules()` 快照。
4. 按本项目规则标记 LoRA target 候选分组。
5. 保存 JSON、CSV 和 Markdown 摘要，方便后续人工复核与提交。

推荐在 Colab GPU runtime 中执行。本地如果没有 GPU、没有 qwen-asr 或没有
模型缓存，只建议运行 `--help` 做脚本检查。
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.lora_targets import (  # noqa: E402
    candidate_is_trainable_by_default,
    classify_lora_candidate,
)


def write_json(path: Path, payload: Any) -> None:
    """写出 JSON 文件，并自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出模块摘要 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "module_name",
        "class_name",
        "candidate_group",
        "direct_param_count",
        "has_weight",
        "weight_shape",
        "direct_param_dtypes",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def resolve_torch_dtype(dtype: str) -> Any:
    """把命令行 dtype 转为 torch dtype；`auto` 表示不显式传 dtype。"""
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


def load_qwen3_asr_model(args: argparse.Namespace) -> Any:
    """加载 Qwen3-ASR 模型。

    qwen-asr 的版本可能会随时间变化。这里优先使用当前 baseline 已验证过的
    `dtype` 参数；如果版本改成 `torch_dtype`，则自动重试一次。
    """
    from qwen_asr import Qwen3ASRModel

    base_kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "max_inference_batch_size": args.max_inference_batch_size,
        "max_new_tokens": args.max_new_tokens,
    }
    torch_dtype = resolve_torch_dtype(args.dtype)
    if torch_dtype is not None:
        base_kwargs["dtype"] = torch_dtype

    try:
        model = Qwen3ASRModel.from_pretrained(args.model_id, **base_kwargs)
    except TypeError:
        retry_kwargs = dict(base_kwargs)
        if "dtype" in retry_kwargs:
            retry_kwargs["torch_dtype"] = retry_kwargs.pop("dtype")
        model = Qwen3ASRModel.from_pretrained(args.model_id, **retry_kwargs)

    if hasattr(model, "eval"):
        model.eval()
    return model


def safe_getattr(obj: Any, name: str) -> Any:
    """安全读取属性，避免某些 lazy property 抛错中断探测。"""
    try:
        return getattr(obj, name)
    except Exception:
        return None


def collect_torch_roots(wrapper: Any) -> list[tuple[str, Any]]:
    """从 qwen-asr wrapper 中寻找 torch.nn.Module 根节点。"""
    import torch.nn as nn

    roots: list[tuple[str, Any]] = []
    seen_ids: set[int] = set()

    def add_root(label: str, value: Any) -> None:
        if isinstance(value, nn.Module) and id(value) not in seen_ids:
            roots.append((label, value))
            seen_ids.add(id(value))

    add_root("self", wrapper)

    # 常见 wrapper 属性名。真实命中情况会写入输出，后续据此调整训练入口。
    for attr in [
        "model",
        "module",
        "base_model",
        "asr_model",
        "speech_model",
        "audio_model",
        "language_model",
        "llm",
        "encoder",
        "decoder",
    ]:
        add_root(attr, safe_getattr(wrapper, attr))

    # 兜底扫描 __dict__ 中直接挂载的 torch module。
    for key, value in vars(wrapper).items() if hasattr(wrapper, "__dict__") else []:
        add_root(key, value)

    return roots


def shape_to_string(value: Any) -> str:
    """把 tensor shape 转成稳定字符串。"""
    shape = getattr(value, "shape", None)
    if shape is None:
        return ""
    return "x".join(str(dim) for dim in shape)


def module_record(module_name: str, module: Any) -> dict[str, Any]:
    """把单个 torch module 转成可保存的摘要记录。"""
    class_name = module.__class__.__name__
    direct_params = list(module.parameters(recurse=False)) if hasattr(module, "parameters") else []
    direct_param_count = sum(param.numel() for param in direct_params)
    direct_param_dtypes = sorted({str(param.dtype).replace("torch.", "") for param in direct_params})
    weight = safe_getattr(module, "weight")
    candidate_group = classify_lora_candidate(module_name, class_name)

    return {
        "module_name": module_name,
        "class_name": class_name,
        "candidate_group": candidate_group,
        "direct_param_count": direct_param_count,
        "has_weight": weight is not None,
        "weight_shape": shape_to_string(weight),
        "direct_param_dtypes": ",".join(direct_param_dtypes),
    }


def collect_module_records(roots: list[tuple[str, Any]], limit: int) -> list[dict[str, Any]]:
    """遍历所有 root 的 named_modules，并去重同一个 module 对象。"""
    records: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    for root_name, root in roots:
        for local_name, module in root.named_modules():
            if id(module) in seen_ids:
                continue
            seen_ids.add(id(module))
            full_name = root_name if not local_name else f"{root_name}.{local_name}"
            records.append(module_record(full_name, module))
            if limit > 0 and len(records) >= limit:
                return records

    return records


def build_candidates(records: list[dict[str, Any]]) -> dict[str, Any]:
    """按候选分组汇总模块名。"""
    grouped_full_names: dict[str, list[str]] = defaultdict(list)
    grouped_leaf_names: dict[str, set[str]] = defaultdict(set)

    for record in records:
        group = record.get("candidate_group") or ""
        if not group:
            continue
        name = str(record["module_name"])
        grouped_full_names[group].append(name)
        grouped_leaf_names[group].add(name.rsplit(".", 1)[-1])

    return {
        "groups": {
            group: {
                "count": len(names),
                "full_module_names": names,
                "leaf_module_names": sorted(grouped_leaf_names[group]),
                "trainable_by_default": candidate_is_trainable_by_default(group),
            }
            for group, names in sorted(grouped_full_names.items())
        }
    }


def write_markdown_summary(
    path: Path,
    metadata: dict[str, Any],
    roots: list[tuple[str, Any]],
    records: list[dict[str, Any]],
    candidates: dict[str, Any],
) -> None:
    """写出便于人工阅读的 Markdown 摘要。"""
    class_counts = Counter(str(row["class_name"]) for row in records)
    group_counts = Counter(str(row["candidate_group"] or "not_candidate") for row in records)

    lines = [
        "# Qwen3-ASR 模块探测摘要",
        "",
        "## 元信息",
        "",
        f"- model_id: `{metadata['model_id']}`",
        f"- created_at: `{metadata['created_at']}`",
        f"- dtype: `{metadata['dtype']}`",
        f"- device_map: `{metadata['device_map']}`",
        f"- total_modules: `{len(records)}`",
        f"- elapsed_seconds: `{metadata['elapsed_seconds']:.2f}`",
        "",
        "## Torch Module Root",
        "",
    ]

    for label, root in roots:
        lines.append(f"- `{label}`: `{root.__class__.__name__}`")

    lines.extend(["", "## 候选分组", ""])
    for group, payload in candidates["groups"].items():
        lines.append(
            f"- `{group}`: {payload['count']} 个模块；"
            f"leaf={payload['leaf_module_names']}；"
            f"默认训练={payload['trainable_by_default']}"
        )

    lines.extend(["", "## 模块类 Top 20", ""])
    for class_name, count in class_counts.most_common(20):
        lines.append(f"- `{class_name}`: {count}")

    lines.extend(["", "## 分组计数", ""])
    for group, count in sorted(group_counts.items()):
        lines.append(f"- `{group}`: {count}")

    lines.extend(
        [
            "",
            "## 下一步复核标准",
            "",
            "1. 确认 `speech_projection` 是否真实位于音频/语音路径，而不是误匹配普通文本模块。",
            "2. 第一版优先尝试 `attention_projection`，必要时再加入 `mlp_projection`。",
            "3. `review_only` 和 `speech_conv` 默认不进入第一版 LoRA 训练。",
            "4. 任何最终 target 都必须经过 5-20 step smoke training 验证。",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--output-dir", default="outputs/lora_probe/qwen3_asr_1_7b")
    parser.add_argument("--dtype", default="float16", choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--max-inference-batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--limit-modules",
        type=int,
        default=0,
        help="只导出前 N 个模块。默认 0 表示导出全部模块，主要用于调试脚本。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    started = time.time()
    model = load_qwen3_asr_model(args)
    roots = collect_torch_roots(model)
    if not roots:
        raise RuntimeError("没有在 qwen-asr wrapper 中找到 torch.nn.Module 根节点。")

    records = collect_module_records(roots, args.limit_modules)
    candidates = build_candidates(records)
    elapsed = time.time() - started

    metadata = {
        "model_id": args.model_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dtype": args.dtype,
        "device_map": args.device_map,
        "max_inference_batch_size": args.max_inference_batch_size,
        "max_new_tokens": args.max_new_tokens,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "elapsed_seconds": elapsed,
        "limit_modules": args.limit_modules,
    }

    write_json(
        output_dir / "module_snapshot.json",
        {
            "metadata": metadata,
            "roots": [{"name": label, "class_name": root.__class__.__name__} for label, root in roots],
            "modules": records,
        },
    )
    write_csv(output_dir / "module_summary.csv", records)
    write_json(output_dir / "lora_target_candidates.json", {"metadata": metadata, **candidates})
    write_markdown_summary(output_dir / "lora_target_candidates.md", metadata, roots, records, candidates)

    print(f"输出目录: {output_dir}")
    print(f"模块总数: {len(records)}")
    for group, payload in candidates["groups"].items():
        print(f"{group}: {payload['count']} modules, leaf={payload['leaf_module_names']}")


if __name__ == "__main__":
    main()
