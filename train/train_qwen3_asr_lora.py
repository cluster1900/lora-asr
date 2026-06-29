#!/usr/bin/env python3
"""Qwen3-ASR Transformers + PEFT LoRA training.

这个入口服务 `05C` smoke training 和 `05D` 第一版 LoRA MVP bootstrap 训练：
加载官方 qwen-asr 模型，按配置中的正则精确挂载 audio tower LoRA，运行自定义
训练循环，并保存 adapter 与训练元数据。它仍不是最终大规模训练器，后续还需要
扩展验证集评测、周期性 checkpoint 和断点续训。
"""

from __future__ import annotations

import argparse
from importlib import metadata as importlib_metadata
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.peft_targets import match_lora_targets, target_summary, validate_lora_targets  # noqa: E402


QWEN_LANGUAGE_ALIASES = {
    "en": "English",
    "english": "English",
    "zh": "Chinese",
    "cn": "Chinese",
    "chinese": "Chinese",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL，跳过空行。"""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    """写出 JSON 文件，并自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """追加写入一行 JSONL。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_config(path: Path) -> dict[str, Any]:
    """读取 YAML 训练配置。"""
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_torch_dtype(dtype: str) -> Any:
    """把配置或命令行 dtype 转成 torch dtype。"""
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


def normalize_language(language: str | None) -> str | None:
    """把 manifest 中的语言字段规范成 qwen-asr 使用的名称。"""
    if language is None:
        return None
    value = str(language).strip()
    if value.lower() in {"", "auto", "none", "null"}:
        return None
    return QWEN_LANGUAGE_ALIASES.get(value.lower(), value)


def resolve_audio_path(audio: str, manifest_path: Path, audio_root: str | None) -> Path:
    """解析 manifest 中的音频路径。"""
    audio_path = Path(audio).expanduser()
    if audio_path.is_absolute():
        return audio_path

    candidates: list[Path] = []
    if audio_root:
        candidates.append(Path(audio_root).expanduser() / audio_path)
    candidates.append(manifest_path.parent / audio_path)
    candidates.append(Path.cwd() / audio_path)
    candidates.append(PROJECT_ROOT / audio_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def get_answer(item: dict[str, Any]) -> str:
    """从样本中提取训练目标文本。"""
    for key in ("answer", "text", "transcript", "reference"):
        value = item.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError("missing answer/text/transcript/reference field")


def select_rows(rows: list[dict[str, Any]], include_scenarios: set[str], limit: int) -> list[dict[str, Any]]:
    """按场景和数量筛选训练样本。"""
    selected: list[dict[str, Any]] = []
    for row in rows:
        scenario = str(row.get("scenario", ""))
        if include_scenarios and scenario not in include_scenarios:
            continue
        selected.append(row)
        if limit > 0 and len(selected) >= limit:
            break
    return selected


def validate_audio_paths(
    rows: list[dict[str, Any]],
    manifest_path: Path,
    audio_root: str | None,
    max_examples: int = 10,
) -> int:
    """在加载模型前验证已选样本的音频路径。"""
    missing: list[tuple[int, str, Path]] = []
    for index, row in enumerate(rows, start=1):
        audio = row.get("audio") or row.get("audio_path")
        if not audio:
            raise ValueError(f"row {index} missing audio/audio_path field")
        resolved = resolve_audio_path(str(audio), manifest_path, audio_root)
        if not resolved.exists():
            missing.append((index, str(audio), resolved))

    if missing:
        examples = "\n".join(
            f"- row={index} audio={audio} resolved={resolved}"
            for index, audio, resolved in missing[:max_examples]
        )
        raise FileNotFoundError(
            "Missing audio files for selected training rows: "
            f"{len(missing)}/{len(rows)}.\n"
            f"Examples:\n{examples}\n"
            "Hint: sync or upload data/mvp_eval/audio/ into the Colab project directory, "
            "or point --manifest/--audio-root to a manifest whose audio files exist."
        )
    return len(rows)


def seed_everything(seed: int) -> None:
    """固定 Python、NumPy 和 torch 随机种子。"""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def parse_version_prefix(version: str) -> tuple[int, int, int]:
    """解析版本号前三段数字，避免为环境预检额外引入 packaging 依赖。"""
    parts: list[int] = []
    for token in version.replace("-", ".").split("."):
        digits = ""
        for char in token:
            if not char.isdigit():
                break
            digits += char
        if not digits:
            break
        parts.append(int(digits))
        if len(parts) == 3:
            break
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def validate_peft_environment() -> None:
    """提前识别会导致 PEFT LoRA 注入失败的 Colab 依赖组合。"""
    try:
        torchao_version = importlib_metadata.version("torchao")
    except importlib_metadata.PackageNotFoundError:
        return

    if parse_version_prefix(torchao_version) >= (0, 16, 0):
        return

    raise ImportError(
        "Detected torchao=={version}, but current PEFT requires torchao>0.16.0 when "
        "the package is installed. This Qwen3-ASR smoke training does not require "
        "torchao. In Colab, run `%pip uninstall -y torchao`, then rerun the training "
        "cell. Alternatively install a compatible torchao version if your runtime "
        "explicitly needs torchao.".format(version=torchao_version)
    )


def build_model_kwargs(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """从配置和命令行生成 qwen-asr `from_pretrained` 参数。"""
    torch_dtype = resolve_torch_dtype(args.dtype)
    kwargs: dict[str, Any] = {
        "device_map": args.device_map,
        "max_inference_batch_size": 1,
        "max_new_tokens": args.max_new_tokens,
    }
    if torch_dtype is not None:
        kwargs["dtype"] = torch_dtype

    quantization = str(args.quantization or config.get("model", {}).get("quantization", "none")).lower()
    if quantization in {"4bit", "nf4"}:
        import torch
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype or torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif quantization in {"8bit", "int8"}:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization not in {"none", "false", "no", "0", ""}:
        raise ValueError(f"Unsupported quantization: {quantization}")

    return kwargs


def load_qwen3_asr_wrapper(args: argparse.Namespace, config: dict[str, Any]) -> Any:
    """通过官方 qwen-asr wrapper 加载模型和 processor。"""
    from qwen_asr import Qwen3ASRModel

    kwargs = build_model_kwargs(args, config)
    try:
        return Qwen3ASRModel.from_pretrained(args.model_id, **kwargs)
    except TypeError:
        retry_kwargs = dict(kwargs)
        if "dtype" in retry_kwargs:
            retry_kwargs["torch_dtype"] = retry_kwargs.pop("dtype")
        return Qwen3ASRModel.from_pretrained(args.model_id, **retry_kwargs)


def first_param_device_dtype(model: Any) -> tuple[Any, Any]:
    """返回模型第一个参数所在 device 和浮点 dtype。"""
    import torch

    fallback_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fallback_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    for param in model.parameters():
        dtype = param.dtype if param.is_floating_point() else fallback_dtype
        return param.device, dtype
    return fallback_device, fallback_dtype


def module_param_device_dtype(module: Any) -> tuple[Any, Any] | None:
    """返回模块直属浮点参数的 device/dtype。"""
    for name in ("bias", "weight"):
        param = getattr(module, name, None)
        if param is not None and getattr(param, "is_floating_point", lambda: False)():
            return param.device, param.dtype
    for param in module.parameters(recurse=False):
        if param.is_floating_point():
            return param.device, param.dtype
    return None


def audio_feature_device_dtype(model: Any) -> tuple[Any, Any]:
    """返回音频特征应使用的 device/dtype。

    Qwen3-ASR 的前置音频卷积在 4bit/LoRA 路径中可能保留 float32 bias。
    `input_features` 必须跟随 `audio_tower.conv2d1`，否则会出现
    `Input type (c10::Half) and bias type (float) should be the same`。
    """
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    for candidate in (model, base_model):
        for module_name, module in candidate.named_modules():
            if module_name.endswith("audio_tower.conv2d1"):
                found = module_param_device_dtype(module)
                if found is not None:
                    return found
    return first_param_device_dtype(model)


def build_text_prompt(processor: Any, context: str, language: str | None) -> str:
    """复用官方推理 prompt：system + audio user turn + assistant generation prompt。"""
    messages = [
        {"role": "system", "content": context or ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if language:
        prompt = prompt + f"language {language}<asr_text>"
    return prompt


def load_audio(audio_path: Path) -> Any:
    """使用 qwen-asr 官方工具加载并规范化音频。"""
    from qwen_asr.inference.utils import normalize_audios

    return normalize_audios(str(audio_path))[0]


def move_batch_to_model(batch: dict[str, Any], model: Any) -> dict[str, Any]:
    """把 processor batch 移到模型 device，音频特征转成模型浮点 dtype。"""
    import torch

    device, _ = first_param_device_dtype(model)
    audio_device, audio_dtype = audio_feature_device_dtype(model)
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            moved[key] = value
            continue
        if key == "input_features":
            moved[key] = value.to(device=audio_device, dtype=audio_dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def build_training_batch(
    processor: Any,
    row: dict[str, Any],
    manifest_path: Path,
    audio_root: str | None,
    default_language: str | None,
    context: str,
    max_audio_seconds: float,
) -> dict[str, Any]:
    """构造单样本训练 batch，并只让答案 token 参与 loss。"""
    import torch

    audio = row.get("audio") or row.get("audio_path")
    if not audio:
        raise ValueError("missing audio/audio_path field")

    audio_path = resolve_audio_path(str(audio), manifest_path, audio_root)
    if not audio_path.exists():
        raise FileNotFoundError(str(audio_path))

    wav = load_audio(audio_path)
    if max_audio_seconds > 0 and len(wav) / 16000.0 > max_audio_seconds:
        raise ValueError(f"audio too long: {len(wav) / 16000.0:.2f}s > {max_audio_seconds}s")

    language = normalize_language(row.get("language")) or default_language
    prompt = build_text_prompt(processor=processor, context=context, language=language)
    answer = get_answer(row)
    eos_token = getattr(processor.tokenizer, "eos_token", "") or ""
    full_text = prompt + answer + eos_token

    prompt_inputs = processor(text=prompt, audio=[wav], return_tensors="pt", padding=True)
    full_inputs = processor(text=full_text, audio=[wav], return_tensors="pt", padding=True)

    labels = full_inputs["input_ids"].clone()
    prompt_len = int(prompt_inputs["input_ids"].shape[-1])
    labels[:, :prompt_len] = -100
    if "attention_mask" in full_inputs:
        labels = labels.masked_fill(full_inputs["attention_mask"] == 0, -100)

    if int((labels != -100).sum().item()) <= 0:
        raise ValueError("no answer tokens left in labels")

    batch = dict(full_inputs)
    batch["labels"] = labels.to(dtype=torch.long)
    return batch


def enable_training_features(model: Any, gradient_checkpointing: bool) -> None:
    """关闭 cache，并按需启用 gradient checkpointing。"""
    thinker = getattr(model, "thinker", None)
    text_model = getattr(model, "model", None)
    thinker_text_model = getattr(thinker, "model", None)
    for obj in [model, thinker, text_model, thinker_text_model]:
        config = getattr(obj, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False

    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if gradient_checkpointing and hasattr(thinker, "gradient_checkpointing_enable"):
        thinker.gradient_checkpointing_enable()


def attach_lora(
    model: Any,
    config: dict[str, Any],
    peft_task_type: str,
    gradient_checkpointing: bool,
    target_root_prefix: str,
) -> tuple[Any, dict[str, Any]]:
    """按配置匹配 target，并通过 PEFT 挂载 LoRA。"""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    lora_config = config["lora"]
    matched = match_lora_targets(model, lora_config, root_prefix=target_root_prefix)
    validate_lora_targets(matched, lora_config)
    summary = target_summary(matched)
    summary["target_root_prefix"] = target_root_prefix

    quantization = str(config.get("model", {}).get("quantization", "none")).lower()
    if quantization in {"4bit", "nf4", "8bit", "int8"}:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=gradient_checkpointing,
        )

    peft_kwargs: dict[str, Any] = {
        "r": int(lora_config["r"]),
        "lora_alpha": int(lora_config["alpha"]),
        "lora_dropout": float(lora_config["dropout"]),
        "bias": "none",
        "target_modules": [item.raw_name for item in matched],
    }

    if peft_task_type != "none":
        from peft import TaskType

        peft_kwargs["task_type"] = getattr(TaskType, peft_task_type.upper())

    peft_model = get_peft_model(model, LoraConfig(**peft_kwargs))
    return peft_model, summary


def trainable_parameter_summary(model: Any) -> dict[str, Any]:
    """统计总参数和可训练参数。"""
    total = 0
    trainable = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_ratio": float(trainable / total) if total else 0.0,
    }


def resolve_training_model(wrapper: Any) -> tuple[Any, str]:
    """返回真正支持训练 forward 的 Qwen3-ASR 子模型。

    qwen-asr wrapper 的 `model` 是 `Qwen3ASRForConditionalGeneration`，它主要提供
    `generate()`；内部 `model.thinker` 才实现 `forward(..., labels=...)` 并能计算
    loss。PEFT 因此必须包 thinker，而不是包最外层模型。
    """
    outer_model = getattr(wrapper, "model", None)
    thinker = getattr(outer_model, "thinker", None)
    if thinker is None:
        raise AttributeError("Qwen3-ASR wrapper.model.thinker not found; cannot run training forward.")
    return thinker, "model.thinker"


def parse_csv_set(value: str) -> set[str]:
    """解析逗号分隔的场景列表。"""
    if not value.strip():
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train/qwen3_asr_lora_mvp.yaml")
    parser.add_argument("--manifest", default=None, help="Override config data.train_manifest.")
    parser.add_argument("--audio-root", default=None, help="Optional root for relative audio paths.")
    parser.add_argument("--output-dir", default=None, help="Override config output.checkpoint_dir.")
    parser.add_argument("--model-id", default=None, help="Override config model.id.")
    parser.add_argument("--dtype", default=None, choices=["auto", "bfloat16", "float16", "float32"])
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--quantization", default=None, choices=["none", "4bit", "8bit", "nf4", "int8"])
    parser.add_argument("--language", default="English")
    parser.add_argument("--context", default="")
    parser.add_argument("--include-scenarios", default="", help="Comma-separated scenario filter.")
    parser.add_argument("--limit", type=int, default=0, help="Only use first N selected rows when > 0.")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-audio-seconds", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--preflight-only", action="store_true", help="Load model and attach LoRA, then stop.")
    parser.add_argument("--no-save-adapter", action="store_true")
    return parser.parse_args()


def apply_arg_defaults(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    """把配置中的默认值回填到 argparse namespace。"""
    model_config = config.get("model", {})
    probe_config = config.get("probe", {})
    data_config = config.get("data", {})
    training_config = config.get("training", {})
    output_config = config.get("output", {})

    args.model_id = args.model_id or model_config.get("id", "Qwen/Qwen3-ASR-1.7B")
    args.dtype = args.dtype or probe_config.get("dtype", "float16")
    args.device_map = args.device_map or probe_config.get("device_map", "cuda:0")
    args.quantization = args.quantization or model_config.get("quantization", "none")
    args.manifest = args.manifest or data_config.get("train_manifest")
    args.output_dir = args.output_dir or output_config.get("checkpoint_dir", "checkpoints/qwen3-asr-1.7b-lora")
    args.max_steps = args.max_steps or int(training_config.get("max_steps", training_config.get("smoke_test_steps", 20)))
    args.batch_size = args.batch_size or int(training_config.get("batch_size", 1))
    args.gradient_accumulation_steps = args.gradient_accumulation_steps or int(
        training_config.get("gradient_accumulation_steps", 1)
    )
    args.learning_rate = args.learning_rate or float(training_config.get("learning_rate", 2e-5))
    args.max_audio_seconds = args.max_audio_seconds or float(data_config.get("max_audio_seconds", 20))
    args.peft_task_type = str(training_config.get("peft_task_type", "none")).lower()
    args.gradient_checkpointing = bool(training_config.get("gradient_checkpointing", True))
    args.seed = int(training_config.get("seed", 42))
    return args


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser()
    config = load_config(config_path)
    args = apply_arg_defaults(args, config)
    config.setdefault("model", {})["quantization"] = args.quantization

    if args.batch_size != 1:
        raise ValueError("05C smoke training currently supports batch_size=1 only.")
    if not args.manifest:
        raise ValueError("No train manifest provided.")

    validate_peft_environment()
    seed_everything(args.seed)
    manifest_path = Path(args.manifest).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(manifest_path)
    selected_rows = select_rows(rows, parse_csv_set(args.include_scenarios), args.limit)
    if not selected_rows:
        raise ValueError("No training rows selected.")

    print(f"[data] manifest={manifest_path} selected_rows={len(selected_rows)}")
    checked_audio = validate_audio_paths(selected_rows, manifest_path, args.audio_root)
    print(f"[data] verified_audio_paths={checked_audio}")
    print(f"[load] model={args.model_id} quantization={args.quantization} dtype={args.dtype}")

    wrapper = load_qwen3_asr_wrapper(args, config)
    base_model, target_root_prefix = resolve_training_model(wrapper)
    processor = wrapper.processor
    enable_training_features(base_model, args.gradient_checkpointing)

    peft_model, lora_summary = attach_lora(
        base_model,
        config,
        args.peft_task_type,
        args.gradient_checkpointing,
        target_root_prefix,
    )
    param_summary = trainable_parameter_summary(peft_model)
    lora_summary["trainable_parameter_summary"] = param_summary

    write_json(output_dir / "target_modules.json", lora_summary)
    run_config = {
        "config_path": str(config_path),
        "model_id": args.model_id,
        "manifest": str(manifest_path),
        "output_dir": str(output_dir),
        "dtype": args.dtype,
        "device_map": args.device_map,
        "quantization": args.quantization,
        "training_root": target_root_prefix,
        "language": args.language,
        "context": args.context,
        "limit": args.limit,
        "include_scenarios": sorted(parse_csv_set(args.include_scenarios)),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_audio_seconds": args.max_audio_seconds,
        "seed": args.seed,
        "config": config,
    }
    write_json(output_dir / "training_config.json", run_config)

    print(
        "[lora] targets={targets} trainable={trainable:,} total={total:,}".format(
            targets=lora_summary["count"],
            trainable=param_summary["trainable_params"],
            total=param_summary["total_params"],
        )
    )

    if args.preflight_only:
        write_json(output_dir / "summary.json", {"status": "preflight_ok", **lora_summary})
        print(f"[done] preflight outputs saved to {output_dir}")
        return

    import torch

    peft_model.train()
    optimizer = torch.optim.AdamW((p for p in peft_model.parameters() if p.requires_grad), lr=args.learning_rate)
    log_path = output_dir / "loss_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    started = time.perf_counter()

    default_language = normalize_language(args.language)
    for step in range(1, args.max_steps + 1):
        row = selected_rows[(step - 1) % len(selected_rows)]
        batch = build_training_batch(
            processor=processor,
            row=row,
            manifest_path=manifest_path,
            audio_root=args.audio_root,
            default_language=default_language,
            context=args.context,
            max_audio_seconds=args.max_audio_seconds,
        )
        batch = move_batch_to_model(batch, peft_model)

        outputs = peft_model(**batch, use_cache=False)
        loss = outputs.loss
        if loss is None:
            raise RuntimeError("Model returned loss=None")
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")

        (loss / args.gradient_accumulation_steps).backward()
        if step % args.gradient_accumulation_steps == 0 or step == args.max_steps:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        loss_value = float(loss.detach().cpu().item())
        losses.append(loss_value)
        record = {
            "step": step,
            "loss": loss_value,
            "scenario": row.get("scenario", ""),
            "utterance_id": row.get("utterance_id", ""),
            "answer_tokens": int((batch["labels"] != -100).sum().detach().cpu().item()),
        }
        append_jsonl(log_path, record)
        print(f"[step {step}/{args.max_steps}] loss={loss_value:.6f} scenario={record['scenario']}")

    adapter_dir = output_dir / "adapter"
    if not args.no_save_adapter:
        peft_model.save_pretrained(adapter_dir)
        processor.save_pretrained(output_dir / "processor")

    summary = {
        "status": "trained",
        "elapsed_seconds": round(time.perf_counter() - started, 4),
        "loss_min": min(losses) if losses else math.nan,
        "loss_max": max(losses) if losses else math.nan,
        "loss_last": losses[-1] if losses else math.nan,
        "steps": len(losses),
        "adapter_dir": str(adapter_dir) if not args.no_save_adapter else "",
        "target_modules": lora_summary,
    }
    write_json(output_dir / "summary.json", summary)
    print(f"[done] saved training outputs to {output_dir}")


if __name__ == "__main__":
    main()
