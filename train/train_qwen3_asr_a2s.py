#!/usr/bin/env python3
"""Single-adapter A2S LoRA training for Qwen3-ASR.

The prompt, label masking, outer-forward patch and Trainer structure follow the
official Qwen3-ASR SFT example (Apache-2.0). Project-specific code is limited to
JSONL/config handling, independently derived PEFT targets, phase switching and
pipeline state. No Mega-ASR runtime code is imported or copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs/train/qwen3_asr_public_200k_a2s.yaml"
PHASE_NAMES = ("phase_1", "phase_2", "phase_3")

EXPECTED_GROUPS = {
    "audio_attention": 96,
    "audio_mlp": 48,
    "projection": 3,
    "decoder_attention": 112,
    "decoder_mlp": 84,
}

_AUDIO_ATTENTION_RE = re.compile(
    r"^thinker\.audio_tower\.layers\.(\d+)\.self_attn\.(?:q_proj|k_proj|v_proj|out_proj)$"
)
_AUDIO_MLP_RE = re.compile(r"^thinker\.audio_tower\.layers\.(\d+)\.(?:fc1|fc2)$")
_DECODER_ATTENTION_RE = re.compile(
    r"^thinker\.model\.layers\.(\d+)\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)$"
)
_DECODER_MLP_RE = re.compile(
    r"^thinker\.model\.layers\.(\d+)\.mlp\.(?:gate_proj|up_proj|down_proj)$"
)
_PROJECTION_NAMES = {
    "thinker.audio_tower.conv_out",
    "thinker.audio_tower.proj1",
    "thinker.audio_tower.proj2",
}
_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


@dataclass(frozen=True)
class TargetSpec:
    """One independently discovered Linear LoRA target."""

    module_name: str
    canonical_name: str
    group: str
    layer: int | None
    class_name: str


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required; install requirements-colab.txt") from exc

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Top-level YAML value must be a mapping: {path}")
    return value


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_module_name(name: str) -> str:
    marker = "thinker."
    index = name.find(marker)
    return name[index:] if index >= 0 else name


def classify_target(name: str, class_name: str) -> tuple[str, int | None] | None:
    """Classify only the 343 runtime Linear modules in the local contract."""
    if class_name != "Linear":
        return None
    canonical = canonical_module_name(name)
    match = _AUDIO_ATTENTION_RE.fullmatch(canonical)
    if match:
        return "audio_attention", int(match.group(1))
    match = _AUDIO_MLP_RE.fullmatch(canonical)
    if match:
        return "audio_mlp", int(match.group(1))
    if canonical in _PROJECTION_NAMES:
        return "projection", None
    match = _DECODER_ATTENTION_RE.fullmatch(canonical)
    if match:
        return "decoder_attention", int(match.group(1))
    match = _DECODER_MLP_RE.fullmatch(canonical)
    if match:
        return "decoder_mlp", int(match.group(1))
    return None


def target_specs_from_records(records: Iterable[dict[str, Any]]) -> list[TargetSpec]:
    targets: list[TargetSpec] = []
    for record in records:
        module_name = str(record.get("module_name", ""))
        class_name = str(record.get("class_name", ""))
        result = classify_target(module_name, class_name)
        if result is None:
            continue
        group, layer = result
        targets.append(
            TargetSpec(
                module_name=module_name,
                canonical_name=canonical_module_name(module_name),
                group=group,
                layer=layer,
                class_name=class_name,
            )
        )
    return sorted(targets, key=lambda item: item.canonical_name)


def expected_target_specs() -> list[TargetSpec]:
    """Build the pinned Qwen3-ASR-1.7B target contract without generated files."""
    records: list[dict[str, str]] = []
    for layer in range(24):
        prefix = f"thinker.audio_tower.layers.{layer}"
        records.extend(
            {"module_name": f"{prefix}.self_attn.{name}", "class_name": "Linear"}
            for name in ("q_proj", "k_proj", "v_proj", "out_proj")
        )
        records.extend(
            {"module_name": f"{prefix}.{name}", "class_name": "Linear"}
            for name in ("fc1", "fc2")
        )
    records.extend(
        {"module_name": name, "class_name": "Linear"}
        for name in sorted(_PROJECTION_NAMES)
    )
    for layer in range(28):
        prefix = f"thinker.model.layers.{layer}"
        records.extend(
            {"module_name": f"{prefix}.self_attn.{name}", "class_name": "Linear"}
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        )
        records.extend(
            {"module_name": f"{prefix}.mlp.{name}", "class_name": "Linear"}
            for name in ("gate_proj", "up_proj", "down_proj")
        )
    return target_specs_from_records(records)


def discover_runtime_targets(model: Any) -> list[TargetSpec]:
    records = [
        {"module_name": name, "class_name": module.__class__.__name__}
        for name, module in model.named_modules()
    ]
    return target_specs_from_records(records)


def target_group_counts(targets: Sequence[TargetSpec]) -> dict[str, int]:
    counts = {name: 0 for name in EXPECTED_GROUPS}
    for target in targets:
        counts[target.group] = counts.get(target.group, 0) + 1
    return counts


def target_map_hash(targets: Sequence[TargetSpec]) -> str:
    payload = [
        {"name": item.canonical_name, "group": item.group, "layer": item.layer}
        for item in sorted(targets, key=lambda value: value.canonical_name)
    ]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_target_map(
    targets: Sequence[TargetSpec],
    expected_groups: dict[str, int] | None = None,
    expected_total: int = 343,
) -> None:
    expected = expected_groups or EXPECTED_GROUPS
    actual = target_group_counts(targets)
    if actual != expected:
        raise ValueError(f"LoRA target groups mismatch: expected={expected}, actual={actual}")
    if len(targets) != expected_total:
        raise ValueError(f"LoRA target total mismatch: expected={expected_total}, actual={len(targets)}")
    names = [item.canonical_name for item in targets]
    if len(names) != len(set(names)):
        raise ValueError("LoRA target map contains duplicate canonical names")
    forbidden = [
        name
        for name in names
        if "lm_head" in name or "embed" in name or "norm" in name or ".conv2d" in name
    ]
    if forbidden:
        raise ValueError(f"Forbidden LoRA targets: {forbidden[:5]}")


def active_target_names(
    targets: Sequence[TargetSpec],
    active_scope: str,
    upper_audio_layers: int = 4,
) -> set[str]:
    if active_scope == "all":
        return {item.canonical_name for item in targets}
    if active_scope == "decoder":
        return {
            item.canonical_name for item in targets if item.group.startswith("decoder_")
        }
    if active_scope != "upper_audio_projection":
        raise ValueError(f"Unknown active_scope: {active_scope}")

    audio_layers = [
        item.layer
        for item in targets
        if item.group in {"audio_attention", "audio_mlp"} and item.layer is not None
    ]
    if not audio_layers:
        raise ValueError("Target map has no audio layers")
    first_active_layer = max(audio_layers) - upper_audio_layers + 1
    return {
        item.canonical_name
        for item in targets
        if item.group == "projection"
        or (
            item.group in {"audio_attention", "audio_mlp"}
            and item.layer is not None
            and item.layer >= first_active_layer
        )
    }


def target_for_parameter(parameter_name: str, targets: Sequence[TargetSpec]) -> TargetSpec | None:
    canonical_parameter = canonical_module_name(parameter_name)
    for target in targets:
        if f"{target.canonical_name}.lora_" in canonical_parameter:
            return target
    return None


def configure_phase_trainability(
    model: Any,
    targets: Sequence[TargetSpec],
    phase: dict[str, Any],
    upper_audio_layers: int,
) -> dict[str, Any]:
    active = active_target_names(targets, str(phase["active_scope"]), upper_audio_layers)
    seen: set[str] = set()
    trainable_parameters = 0

    for _, parameter in model.named_parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        target = target_for_parameter(name, targets)
        if target is None:
            continue
        enabled = target.canonical_name in active
        parameter.requires_grad = enabled
        if enabled:
            seen.add(target.canonical_name)
            trainable_parameters += parameter.numel()

    missing = active - seen
    if missing:
        raise ValueError(f"Active LoRA modules have no parameters: {sorted(missing)[:5]}")
    return {
        "scope": phase["active_scope"],
        "active_target_count": len(active),
        "active_targets": sorted(active),
        "trainable_parameters": trainable_parameters,
    }


def optimizer_group_plan(
    model: Any,
    targets: Sequence[TargetSpec],
    phase: dict[str, Any],
    weight_decay: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return runtime optimizer groups plus a JSON-serializable summary."""
    buckets: dict[tuple[str, float], list[Any]] = {}
    names: dict[tuple[str, float], list[str]] = {}
    audio_lr = float(phase.get("audio_learning_rate", 0.0))
    decoder_lr = float(phase.get("decoder_learning_rate", 0.0))

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        target = target_for_parameter(name, targets)
        if target is None:
            raise ValueError(f"Unexpected trainable non-LoRA parameter: {name}")
        family = "decoder" if target.group.startswith("decoder_") else "audio_projection"
        learning_rate = decoder_lr if family == "decoder" else audio_lr
        if learning_rate <= 0:
            raise ValueError(f"Trainable target has non-positive LR: {target.canonical_name}")
        key = (family, learning_rate)
        buckets.setdefault(key, []).append(parameter)
        names.setdefault(key, []).append(name)

    if not buckets:
        raise ValueError("No trainable optimizer parameters")
    runtime = [
        {"params": parameters, "lr": lr, "weight_decay": weight_decay}
        for (family, lr), parameters in sorted(buckets.items())
    ]
    summary = [
        {
            "family": family,
            "learning_rate": lr,
            "parameter_tensors": len(names[(family, lr)]),
            "parameters": sum(parameter.numel() for parameter in buckets[(family, lr)]),
        }
        for family, lr in sorted(buckets)
    ]
    return runtime, summary


def materialize_curriculum(
    rows: Sequence[dict[str, Any]],
    logical_epochs: int,
    thresholds: Sequence[float],
    seed: int,
) -> list[dict[str, Any]]:
    """Build equal ordered exposure segments from cumulative error views."""
    if not rows:
        raise ValueError("Curriculum manifest is empty")
    if not thresholds or sorted(thresholds) != list(thresholds):
        raise ValueError("Curriculum thresholds must be non-empty and increasing")
    total_exposure = len(rows) * int(logical_epochs)
    base_segment, remainder = divmod(total_exposure, len(thresholds))
    result: list[dict[str, Any]] = []

    for index, threshold in enumerate(thresholds):
        eligible = [
            dict(row)
            for row in rows
            if float(row.get("base_error_rate", math.inf)) < float(threshold)
        ]
        if not eligible:
            raise ValueError(f"No curriculum rows satisfy base_error_rate < {threshold}")
        eligible.sort(key=lambda row: str(row.get("sample_id") or row.get("source_id") or row.get("audio")))
        random.Random(seed + index).shuffle(eligible)
        segment_size = base_segment + (1 if index < remainder else 0)
        for position in range(segment_size):
            row = dict(eligible[position % len(eligible)])
            row["curriculum_threshold"] = float(threshold)
            row["curriculum_segment"] = index
            result.append(row)
    return result


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.is_dir():
        return None
    candidates: list[tuple[int, Path]] = []
    for child in output_dir.iterdir():
        match = _CHECKPOINT_RE.fullmatch(child.name)
        if match and child.is_dir():
            candidates.append((int(match.group(1)), child))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def validate_config(config: dict[str, Any]) -> None:
    required_sections = {"model", "data", "lora", "training", "phases", "evaluation"}
    missing = sorted(required_sections - config.keys())
    if missing:
        raise ValueError(f"Missing config sections: {missing}")
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("Only schema_version=1 is supported")
    phases = config.get("phases")
    if not isinstance(phases, list) or [item.get("name") for item in phases] != list(PHASE_NAMES):
        raise ValueError(f"phases must be exactly {PHASE_NAMES}")
    effective = int(config["training"]["per_device_batch_size"]) * int(
        config["training"]["gradient_accumulation_steps"]
    )
    expected_effective = int(config["training"]["effective_batch_size"])
    if effective != expected_effective or effective != 128:
        raise ValueError(
            f"Effective batch must remain 128: micro_batch*accumulation={effective}, "
            f"configured={expected_effective}"
        )
    if config["model"].get("dtype") != "bfloat16":
        raise ValueError("Formal A2S comparison requires model.dtype=bfloat16")


def build_plan(config: dict[str, Any]) -> dict[str, Any]:
    phases: list[dict[str, Any]] = []
    for phase in config["phases"]:
        phases.append(
            {
                "name": phase["name"],
                "role": phase["role"],
                "manifest": config["data"][phase["manifest"]],
                "exposure": (
                    f"30k x {phase['logical_epochs']} logical epochs"
                    if phase["name"] == "phase_1"
                    else f"200k x {phase['epochs']} epoch"
                ),
                "scope": phase["active_scope"],
                "audio_learning_rate": phase["audio_learning_rate"],
                "decoder_learning_rate": phase["decoder_learning_rate"],
                "warmup_ratio": phase["warmup_ratio"],
            }
        )
    return {
        "run_name": config["run_name"],
        "model": config["model"],
        "effective_batch_size": config["training"]["effective_batch_size"],
        "sample_exposure": 460000,
        "phases": phases,
    }


def resolve_audio_path(audio: str, manifest: Path, data_root: Path) -> Path:
    path = Path(audio).expanduser()
    if path.is_absolute():
        return path
    candidates = [data_root / path, manifest.parent / path, PROJECT_ROOT / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def prepare_rows(
    manifest: Path,
    data_root: Path,
    phase: dict[str, Any],
    seed: int,
    smoke_limit: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(manifest)
    source_rows = len(rows)
    if smoke_limit:
        rows = rows[:smoke_limit]
    is_curriculum = phase["name"] == "phase_1" and not smoke_limit
    if is_curriculum:
        rows = materialize_curriculum(
            rows,
            logical_epochs=int(phase["logical_epochs"]),
            thresholds=[float(value) for value in phase["curriculum_thresholds"]],
            seed=seed,
        )

    prepared: list[dict[str, Any]] = []
    missing: list[str] = []
    for index, source in enumerate(rows):
        row = dict(source)
        sample_id = str(row.get("sample_id") or "").strip()
        audio_value = row.get("audio")
        answer = str(row.get("answer") or "").strip()
        if not sample_id:
            raise ValueError(f"Row {index + 1} has no sample_id")
        if not audio_value:
            raise ValueError(f"Row {index + 1} has no audio")
        if not answer:
            raise ValueError(f"Row {index + 1} has no answer")
        resolved_audio = resolve_audio_path(str(audio_value), manifest, data_root)
        if not resolved_audio.is_file():
            missing.append(str(resolved_audio))
            if len(missing) >= 10:
                break
        duration = float(row.get("duration_s") or 0.0)
        prepared.append(
            {
                **row,
                "audio": str(resolved_audio),
                "text": answer,
                "prompt": str(row.get("prompt", "")),
                "duration": duration,
            }
        )
    if missing:
        raise FileNotFoundError(f"Missing audio files (first {len(missing)}): {missing}")
    if not prepared:
        raise ValueError(f"No rows prepared from {manifest}")
    return prepared, {
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256_file(manifest),
        "source_rows": source_rows,
        "materialized_rows": len(prepared),
        "curriculum_ordered": is_curriculum,
    }


def patch_outer_forward(model: Any) -> None:
    """Expose the official outer model forward through its thinker."""
    cls = model.__class__
    if getattr(cls, "_mega_asr_a2s_forward_patched", False):
        return
    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError("Qwen3-ASR model has no thinker.forward; revision is incompatible")

    def forward(
        self: Any,
        input_ids: Any = None,
        attention_mask: Any = None,
        input_features: Any = None,
        feature_attention_mask: Any = None,
        labels: Any = None,
        **kwargs: Any,
    ) -> Any:
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._mega_asr_a2s_forward_patched = True


def build_prefix_messages(prompt: str, audio_array: Any) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn(processor: Any) -> Any:
    def preprocess(example: dict[str, Any]) -> dict[str, Any]:
        messages = build_prefix_messages(str(example.get("prompt", "")), None)
        prefix_text = processor.apply_chat_template(
            [messages], add_generation_prompt=True, tokenize=False
        )[0]
        return {"prefix_text": prefix_text, "target": example["text"]}

    return preprocess


@dataclass
class DataCollatorForQwen3ASR:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        import librosa

        audio_paths = [str(item["audio"]) for item in features]
        prefix_texts = [str(item["prefix_text"]) for item in features]
        targets = [str(item["target"]) for item in features]
        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [prefix + target + eos for prefix, target in zip(prefix_texts, targets)]
        audios = [librosa.load(path, sr=self.sampling_rate, mono=True)[0] for path in audio_paths]

        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_lengths = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for index, prefix_length in enumerate(prefix_lengths):
            labels[index, :prefix_length] = -100
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        if any(int((row != -100).sum()) == 0 for row in labels):
            raise ValueError("Golden batch failed: a sample has no answer labels")
        full_inputs["labels"] = labels
        return full_inputs


def make_trainer_class() -> type[Any]:
    import torch
    from transformers import Trainer

    class A2STrainer(Trainer):
        def __init__(self, *args: Any, optimizer_groups: list[dict[str, Any]], sequential: bool, **kwargs: Any):
            self._a2s_optimizer_groups = optimizer_groups
            self._a2s_sequential = sequential
            super().__init__(*args, **kwargs)

        def _prepare_inputs(self, inputs: Any) -> Any:
            inputs = super()._prepare_inputs(inputs)
            model_dtype = getattr(self.model, "dtype", None)
            if model_dtype is not None:
                for key, value in list(inputs.items()):
                    if torch.is_tensor(value) and value.is_floating_point():
                        inputs[key] = value.to(dtype=model_dtype)
            return inputs

        def _get_train_sampler(self, train_dataset: Any = None) -> Any:
            if self._a2s_sequential:
                dataset = train_dataset if train_dataset is not None else self.train_dataset
                return torch.utils.data.SequentialSampler(dataset)
            return super()._get_train_sampler(train_dataset)

        def create_optimizer(self) -> Any:
            if self.optimizer is None:
                self.optimizer = torch.optim.AdamW(
                    self._a2s_optimizer_groups,
                    lr=max(float(group["lr"]) for group in self._a2s_optimizer_groups),
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            return self.optimizer

    return A2STrainer


def load_dataset_for_phase(
    rows: list[dict[str, Any]], processor: Any
) -> Any:
    from datasets import Dataset

    dataset = Dataset.from_list(rows)
    dataset = dataset.map(make_preprocess_fn(processor), num_proc=1, desc="Build Qwen prompts")
    keep = {"audio", "target", "prefix_text", "duration"}
    drop = [column for column in dataset.column_names if column not in keep]
    if drop:
        dataset = dataset.remove_columns(drop)
    return dataset


def load_qwen_runtime(config: dict[str, Any]) -> tuple[Any, Any]:
    import torch
    from qwen_asr import Qwen3ASRModel
    from transformers import GenerationConfig

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Formal A2S training requires a CUDA GPU with BF16 support")
    model_config = config["model"]
    kwargs: dict[str, Any] = {
        "revision": model_config["revision"],
        "dtype": torch.bfloat16,
        "device_map": None,
        "attn_implementation": model_config.get("attn_implementation", "flash_attention_2"),
    }
    wrapper = Qwen3ASRModel.from_pretrained(model_config["id"], **kwargs)
    model = wrapper.model
    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)
    model.config.use_cache = False
    return model, wrapper.processor


def inject_or_load_adapter(
    model: Any,
    targets: Sequence[TargetSpec],
    lora_config: dict[str, Any],
    adapter_path: Path | None,
) -> Any:
    from peft import LoraConfig, PeftModel, get_peft_model

    if adapter_path is not None:
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"Adapter directory does not exist: {adapter_path}")
        return PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
    peft_config = LoraConfig(
        r=int(lora_config["r"]),
        lora_alpha=int(lora_config["alpha"]),
        lora_dropout=float(lora_config["dropout"]),
        target_modules=[item.module_name for item in targets],
        bias=str(lora_config.get("bias", "none")),
        task_type=None,
    )
    return get_peft_model(model, peft_config)


def make_training_arguments(
    config: dict[str, Any],
    phase: dict[str, Any],
    output_dir: Path,
    curriculum_ordered: bool,
    smoke_steps: int,
) -> Any:
    from transformers import TrainingArguments

    training = config["training"]
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(training["per_device_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        learning_rate=max(
            float(phase.get("audio_learning_rate", 0.0)),
            float(phase.get("decoder_learning_rate", 0.0)),
        ),
        num_train_epochs=1.0 if curriculum_ordered else float(phase.get("epochs", 1)),
        max_steps=int(smoke_steps) if smoke_steps else -1,
        logging_steps=int(training["logging_steps"]),
        lr_scheduler_type=str(training["lr_scheduler_type"]),
        warmup_ratio=float(phase["warmup_ratio"]),
        weight_decay=float(training["weight_decay"]),
        max_grad_norm=float(training["max_grad_norm"]),
        dataloader_num_workers=int(training["dataloader_num_workers"]),
        dataloader_pin_memory=True,
        dataloader_persistent_workers=int(training["dataloader_num_workers"]) > 0,
        dataloader_prefetch_factor=2 if int(training["dataloader_num_workers"]) > 0 else None,
        save_strategy="steps",
        save_steps=int(smoke_steps) if smoke_steps else int(training["save_steps"]),
        save_total_limit=int(training["save_total_limit"]),
        save_safetensors=True,
        eval_strategy="no",
        bf16=True,
        fp16=False,
        tf32=True,
        gradient_checkpointing=bool(training["gradient_checkpointing"]),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=bool(training["group_by_duration"]) and not curriculum_ordered,
        length_column_name="duration",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="none",
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
    )


def load_pipeline_state(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "pipeline_state.json"
    if not path.is_file():
        return {"schema_version": 1, "completed_phases": [], "current_phase": None}
    value = json.loads(path.read_text(encoding="utf-8"))
    if int(value.get("schema_version", 0)) != 1:
        raise ValueError(f"Unsupported pipeline state: {path}")
    return value


def save_resolved_contract(
    output_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    targets: Sequence[TargetSpec],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "config": config,
        "target_map_hash": target_map_hash(targets),
        "target_group_counts": target_group_counts(targets),
        "targets": [asdict(item) for item in targets],
    }
    atomic_write_json(output_dir / "resolved_contract.json", payload)


def resolve_resume_checkpoint(resume: str, phase_dir: Path) -> Path | None:
    value = resume.strip()
    if not value:
        return None
    if value == "auto":
        return find_latest_checkpoint(phase_dir)
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    return path


def resume_value_for_phase(resume: str, phase_dir: Path) -> str:
    """Apply an explicit checkpoint only to the phase directory that owns it."""
    value = resume.strip()
    if value in {"", "auto"}:
        return value
    checkpoint = Path(value).expanduser().resolve()
    try:
        checkpoint.relative_to(phase_dir.resolve())
    except ValueError:
        return ""
    return str(checkpoint)


def run_phase_training(
    config: dict[str, Any],
    phase: dict[str, Any],
    model: Any,
    processor: Any,
    targets: Sequence[TargetSpec],
    manifest: Path,
    data_root: Path,
    phase_dir: Path,
    resume: str,
    smoke_steps: int = 0,
) -> tuple[Path, dict[str, Any]]:
    training = config["training"]
    lora = config["lora"]
    scope_summary = configure_phase_trainability(
        model,
        targets,
        phase,
        upper_audio_layers=int(lora["phase_1_upper_audio_layers"]),
    )
    expected_scope_counts = {"phase_1": 27, "phase_2": 196, "phase_3": 343}
    if scope_summary["active_target_count"] != expected_scope_counts[phase["name"]]:
        raise ValueError(
            f"{phase['name']} target count mismatch: {scope_summary['active_target_count']}"
        )

    rows, data_summary = prepare_rows(
        manifest,
        data_root,
        phase,
        seed=int(config["seed"]),
        smoke_limit=128 if smoke_steps else 0,
    )
    dataset = load_dataset_for_phase(rows, processor)
    collator = DataCollatorForQwen3ASR(
        processor=processor, sampling_rate=int(config["data"]["sampling_rate"])
    )
    optimizer_groups, optimizer_summary = optimizer_group_plan(
        model,
        targets,
        phase,
        weight_decay=float(training["weight_decay"]),
    )
    phase_dir.mkdir(parents=True, exist_ok=True)
    training_args = make_training_arguments(
        config,
        phase,
        phase_dir,
        curriculum_ordered=bool(data_summary["curriculum_ordered"]),
        smoke_steps=smoke_steps,
    )
    if bool(training["gradient_checkpointing"]):
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    TrainerClass = make_trainer_class()
    trainer = TrainerClass(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        optimizer_groups=optimizer_groups,
        sequential=bool(data_summary["curriculum_ordered"]),
    )
    resume_checkpoint = resolve_resume_checkpoint(resume, phase_dir)
    if resume == "auto" and resume_checkpoint is None:
        print(f"[resume] no checkpoint found in {phase_dir}; starting this phase fresh")
    trainer.train(resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None)

    final_adapter = phase_dir / "final_adapter"
    trainer.save_model(str(final_adapter))
    trainer.save_state()
    summary = {
        "phase": phase["name"],
        "global_step": trainer.state.global_step,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        "scope": scope_summary,
        "optimizer_groups": optimizer_summary,
        "data": data_summary,
        "final_adapter": str(final_adapter),
    }
    atomic_write_json(phase_dir / "phase_summary.json", summary)
    return final_adapter, summary


def run_canary(
    config: dict[str, Any],
    phase_name: str,
    adapter_dir: Path,
    output_root: Path,
) -> None:
    manifest = Path(config["data"]["canary_manifest"]).expanduser()
    if not manifest.is_file():
        raise FileNotFoundError(f"Canary manifest is required after {phase_name}: {manifest}")
    canary_dir = output_root / "canary" / phase_name
    predictions = canary_dir / "predictions.jsonl"
    metrics = canary_dir / "metrics.json"
    inference_command = [
        sys.executable,
        str(PROJECT_ROOT / "inference/qwen3_asr_infer.py"),
        "--manifest",
        str(manifest),
        "--output-jsonl",
        str(predictions),
        "--adapter-dir",
        str(adapter_dir),
        "--audio-root",
        str(config["data"]["data_root"]),
        "--resume",
    ]
    evaluation_command = [
        sys.executable,
        str(PROJECT_ROOT / "evaluation/eval_wer.py"),
        "--predictions-jsonl",
        str(predictions),
        "--output-dir",
        str(canary_dir),
    ]
    canary_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(inference_command, cwd=PROJECT_ROOT, check=True)
    subprocess.run(evaluation_command, cwd=PROJECT_ROOT, check=True)
    adapter_metrics = json.loads(metrics.read_text(encoding="utf-8"))
    base_metrics_path = Path(config["evaluation"]["base_canary_metrics"]).expanduser()
    if not base_metrics_path.is_file():
        raise FileNotFoundError(f"Pinned BF16 base canary metrics are required: {base_metrics_path}")
    base_metrics = json.loads(base_metrics_path.read_text(encoding="utf-8"))
    gate = evaluate_canary_gate(base_metrics, adapter_metrics, config["evaluation"])
    atomic_write_json(canary_dir / "gate.json", gate)
    if not gate["passed"]:
        raise RuntimeError(f"{phase_name} canary failed: {gate['failures']}")


def evaluate_canary_gate(
    base_metrics: dict[str, Any],
    adapter_metrics: dict[str, Any],
    gate_config: dict[str, Any],
) -> dict[str, Any]:
    base = base_metrics["overall"]
    adapter = adapter_metrics["overall"]
    failures: list[str] = []
    base_macro = base.get("language_macro_error_rate")
    adapter_macro = adapter.get("language_macro_error_rate")
    if base_macro is None or adapter_macro is None or float(base_macro) <= 0:
        failures.append("missing positive language_macro_error_rate")
        relative_regression = None
    else:
        relative_regression = (float(adapter_macro) - float(base_macro)) / float(base_macro)
        if relative_regression > float(gate_config["max_relative_robust_regression"]):
            failures.append(f"robust macro regression {relative_regression:.4f}")

    valid_output_rate = 1.0 - max(
        float(adapter.get("inference_error_rate", 0.0)),
        float(adapter.get("empty_output_rate", 0.0)),
    )
    if valid_output_rate < float(gate_config["min_valid_output_rate"]):
        failures.append(f"valid output rate {valid_output_rate:.4f}")

    failure_fields = (
        "inference_error_rate",
        "empty_output_rate",
        "repeat_like_output_rate",
        "too_long_output_rate",
        "hallucination_like_output_rate",
    )
    failure_increases: dict[str, float] = {}
    for field in failure_fields:
        increase = float(adapter.get(field, 0.0)) - float(base.get(field, 0.0))
        failure_increases[field] = round(increase, 6)
        if increase > float(gate_config["max_failure_rate_increase"]):
            failures.append(f"{field} increase {increase:.4f}")
    return {
        "passed": not failures,
        "failures": failures,
        "relative_robust_regression": (
            round(relative_regression, 6) if relative_regression is not None else None
        ),
        "valid_output_rate": round(valid_output_rate, 6),
        "failure_rate_increases": failure_increases,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the single-adapter Qwen3-ASR A2S fast path")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--resume", default="", help="Empty, 'auto', or a checkpoint directory")
    parser.add_argument("--smoke-steps", type=int, default=0, help="Run only a 128-row Phase-I smoke")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve()
    config = load_yaml(config_path)
    validate_config(config)
    plan = build_plan(config)
    if args.print_plan:
        print(json.dumps(plan, ensure_ascii=False, indent=2))

    expected_targets = expected_target_specs()
    expected_groups = {key: int(value) for key, value in config["lora"]["expected_groups"].items()}
    validate_target_map(expected_targets, expected_groups, int(config["lora"]["expected_total"]))
    for phase in config["phases"]:
        count = len(
            active_target_names(
                expected_targets,
                phase["active_scope"],
                int(config["lora"]["phase_1_upper_audio_layers"]),
            )
        )
        expected = {"phase_1": 27, "phase_2": 196, "phase_3": 343}[phase["name"]]
        if count != expected:
            raise ValueError(f"Target {phase['name']} scope mismatch: expected={expected}, actual={count}")
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "ok",
                    "target_map_hash": target_map_hash(expected_targets),
                    "target_groups": target_group_counts(expected_targets),
                    "plan": plan,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.smoke_steps < 0:
        raise ValueError("--smoke-steps must be non-negative")

    output_root = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else Path(config["training"]["output_dir"]).expanduser()
    )
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root
        else Path(config["data"]["data_root"]).expanduser()
    )
    base_model, processor = load_qwen_runtime(config)
    runtime_targets = discover_runtime_targets(base_model)
    validate_target_map(runtime_targets, expected_groups, int(config["lora"]["expected_total"]))
    if target_map_hash(runtime_targets) != target_map_hash(expected_targets):
        raise ValueError("Runtime target map differs from the pinned Qwen3-ASR-1.7B contract")

    output_root.mkdir(parents=True, exist_ok=True)
    state = load_pipeline_state(output_root)
    if state.get("target_map_hash") not in {None, target_map_hash(runtime_targets)}:
        raise ValueError("Pipeline target-map hash differs from this runtime")
    save_resolved_contract(output_root, config_path, config, runtime_targets)

    adapter_source = Path(state["last_adapter"]) if state.get("last_adapter") else None
    model = inject_or_load_adapter(base_model, runtime_targets, config["lora"], adapter_source)

    if args.smoke_steps:
        smoke_phase = dict(config["phases"][0])
        smoke_manifest = Path(config["data"]["smoke_manifest"]).expanduser()
        if not smoke_manifest.is_absolute():
            smoke_manifest = PROJECT_ROOT / smoke_manifest
        final_adapter, summary = run_phase_training(
            config,
            smoke_phase,
            model,
            processor,
            runtime_targets,
            smoke_manifest,
            data_root,
            output_root / "smoke",
            args.resume,
            smoke_steps=args.smoke_steps,
        )
        atomic_write_json(
            output_root / "smoke" / "smoke_result.json",
            {"requested_steps": args.smoke_steps, "adapter": str(final_adapter), "summary": summary},
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    for phase in config["phases"]:
        phase_name = phase["name"]
        if phase_name in state.get("completed_phases", []):
            print(f"[pipeline] skip completed {phase_name}")
            continue
        manifest = Path(config["data"][phase["manifest"]]).expanduser()
        phase_dir = output_root / phase_name
        phase_resume = resume_value_for_phase(args.resume, phase_dir)
        existing_checkpoint = find_latest_checkpoint(phase_dir)
        if existing_checkpoint and not phase_resume:
            raise RuntimeError(
                f"Incomplete checkpoint exists for {phase_name}: {existing_checkpoint}. "
                "Use --resume auto or choose a new output directory."
            )
        state.update(
            {
                "schema_version": 1,
                "current_phase": phase_name,
                "target_map_hash": target_map_hash(runtime_targets),
                "config_sha256": sha256_file(config_path),
            }
        )
        atomic_write_json(output_root / "pipeline_state.json", state)
        final_adapter, summary = run_phase_training(
            config,
            phase,
            model,
            processor,
            runtime_targets,
            manifest,
            data_root,
            phase_dir,
            phase_resume,
        )
        run_canary(config, phase_name, final_adapter, output_root)
        completed = list(state.get("completed_phases", []))
        completed.append(phase_name)
        state.update(
            {
                "completed_phases": completed,
                "current_phase": None,
                "last_adapter": str(final_adapter),
                "last_phase_summary": summary,
            }
        )
        atomic_write_json(output_root / "pipeline_state.json", state)

    if all(name in state.get("completed_phases", []) for name in PHASE_NAMES):
        release_dir = output_root / str(config["release"]["adapter_dir"])
        model.save_pretrained(str(release_dir), safe_serialization=True)
        processor.save_pretrained(str(output_root / "release" / "processor"))
        release_manifest = {
            "base_model": config["model"],
            "adapter": str(release_dir),
            "target_map_hash": target_map_hash(runtime_targets),
            "config_sha256": sha256_file(config_path),
            "completed_phases": state["completed_phases"],
            "formal_evaluation_status": "pending",
        }
        atomic_write_json(output_root / str(config["release"]["manifest"]), release_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
