#!/usr/bin/env python3
"""SFT Training Runner for Qwen3-ASR on V100 server.

Supports:
- Official qwen-asr / Transformers API
- Operation directly on model.model.thinker (Qwen3ASRThinkerForConditionalGeneration)
- Exact 199 Linear LoRA targets (3 Projection + 196 Decoder attention/mlp)
- FP16, eager attention, gradient checkpointing with input require grads
- Single-GPU and multi-GPU (DDP via torchrun)
- Label masking (prompt tokens masked with -100, only target transcript trained)
- 10+2 step checkpointing and complete state resumption (optimizer, scheduler, RNG, step)
- Weight merging and export via merge_and_unload()
- Execution contract compliance: environment.json, resolved_config.yaml, manifest_sha256.json, pipeline_state.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure HF mirror, large cache directory on /data, and offline fallback
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/data/mega-asr/cache/huggingface")
if not os.environ.get("HF_TOKEN"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


import numpy as np
import yaml

try:
    import soundfile as sf
    HAVE_SOUNDFILE = True
except ImportError:
    sf = None  # type: ignore
    HAVE_SOUNDFILE = False

try:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, Dataset, DistributedSampler
    HAVE_TORCH = True
except ImportError:
    torch = None  # type: ignore
    dist = None  # type: ignore
    DDP = None  # type: ignore
    DataLoader = None  # type: ignore
    Dataset = object  # type: ignore
    DistributedSampler = None  # type: ignore
    HAVE_TORCH = False

try:
    import peft
    from peft import LoraConfig, PeftModel, get_peft_model
    import peft.tuners.lora as lora_tuners
    HAVE_PEFT = True
except ImportError:
    peft = None  # type: ignore
    LoraConfig = None  # type: ignore
    PeftModel = None  # type: ignore
    get_peft_model = None  # type: ignore
    lora_tuners = None  # type: ignore
    HAVE_PEFT = False

try:
    import transformers
    from transformers import (
        get_linear_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
        get_constant_schedule_with_warmup,
    )
    HAVE_TRANSFORMERS = True
except ImportError:
    transformers = None  # type: ignore
    get_linear_schedule_with_warmup = None  # type: ignore
    get_cosine_schedule_with_warmup = None  # type: ignore
    get_constant_schedule_with_warmup = None  # type: ignore
    HAVE_TRANSFORMERS = False

try:
    from qwen_asr import Qwen3ASRModel
    HAVE_QWEN_ASR = True
except ImportError:
    Qwen3ASRModel = None  # type: ignore
    HAVE_QWEN_ASR = False


# Canonical LoRA target regex pattern matching exactly 199 linear layers
# Audio tower projection: conv_out, proj1, proj2 (3)
# Decoder layers 0..27: self_attn (q,k,v,o) (4*28 = 112) + mlp (gate,up,down) (3*28 = 84)
# Total = 3 + 112 + 84 = 199
LORA_TARGET_REGEX = r"(audio_tower\.(conv_out|proj1|proj2)$|model\.layers\.([0-9]|1[0-9]|2[0-7])\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$)"
EXPECTED_LORA_TARGET_COUNT = 199

LANGUAGE_MAP: Dict[str, str] = {
    "en": "English",
    "zh": "Chinese",
    "english": "English",
    "chinese": "Chinese",
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=str, default="", help="Path to input JSONL manifest.")
    parser.add_argument("--config", type=str, default="configs/train/qwen3_asr_v100.yaml", help="Path to training config YAML.")
    parser.add_argument("--output-dir", type=str, default="", help="Directory to save run outputs, checkpoints, and logs.")
    parser.add_argument("--model-id", type=str, default="", help="Model name or local checkpoint path.")
    parser.add_argument("--revision", type=str, default="", help="Model commit SHA revision.")
    parser.add_argument("--max-steps", type=int, default=None, help="Total training steps (default: from config or 10).")
    parser.add_argument("--save-steps", type=int, default=None, help="Checkpoint saving interval in steps (default: from config or 10).")
    parser.add_argument("--learning-rate", type=float, default=None, help="Peak learning rate (default: from config or 2e-5).")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Optimizer weight decay.")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Linear warmup steps (default: from config or 0).")
    parser.add_argument("--lr-scheduler", type=str, choices=["linear", "cosine", "constant"], default=None, help="LR scheduler type (linear|cosine|constant).")
    parser.add_argument("--sample-strategy", type=str, choices=["standard", "clean_2x", "balanced"], default=None, help="Data sampling strategy (standard|clean_2x|balanced).")
    parser.add_argument("--val-manifest", type=str, default="", help="Optional validation manifest for validation loss logging.")
    parser.add_argument("--eval-steps", type=int, default=0, help="Interval in steps to evaluate validation loss.")
    parser.add_argument("--micro-batch-size", type=int, default=None, help="Micro batch size per GPU (default: from config).")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None, help="Gradient accumulation steps (default: from config).")
    parser.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clipping max norm.")
    parser.add_argument("--resume-from-checkpoint", type=str, default="", help="Path to checkpoint directory to resume from.")
    parser.add_argument("--single-gpu", action="store_true", help="Force single-GPU execution even if distributed env is present.")
    parser.add_argument("--seed", type=int, default=20260722, help="Random seed.")
    parser.add_argument("--export-merged", action="store_true", help="Export merged base model from checkpoint.")
    parser.add_argument("--checkpoint-dir", type=str, default="", help="Checkpoint directory for --export-merged.")
    parser.add_argument("--ignore-dataset-gate", action="store_true", help="Bypass DATASET_COMPLETE.json gate failure check.")
    parser.add_argument("--allow-subset", action="store_true", help="Allow training on NON_STRICT_SUBSET dataset gate status (for pilot/smoke runs).")
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if HAVE_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def calculate_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_target_map_hash(target_names: List[str]) -> str:
    sorted_targets = sorted(target_names)
    joined = "\n".join(sorted_targets)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def get_git_commit(repo_dir: Optional[Path] = None) -> str:
    """Retrieve git commit SHA from env, .git_commit file, or git CLI."""
    if "GIT_COMMIT" in os.environ:
        return os.environ["GIT_COMMIT"]
    if repo_dir is None:
        repo_dir = Path(__file__).resolve().parents[1]
    commit_file = repo_dir / ".git_commit"
    if commit_file.is_file():
        commit_str = commit_file.read_text(encoding="utf-8").strip()
        if commit_str:
            return commit_str
    try:
        import subprocess
        res = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_dir, capture_output=True, text=True, check=False)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "N/A"


def get_environment_info(
    device_name: str,
    world_size: int,
    seed: int,
    model_id: str = "Qwen/Qwen3-ASR-1.7B",
    model_revision: str = "7278e1e70fe206f11671096ffdd38061171dd6e5",
    dtype: str = "float16",
    attention: str = "eager",
) -> Dict[str, Any]:
    gpu_names = []
    if HAVE_TORCH and torch.cuda.is_available():
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    # qwen-asr version
    qwen_asr_ver = "N/A"
    try:
        import importlib.metadata
        qwen_asr_ver = importlib.metadata.version("qwen-asr")
    except Exception:
        if HAVE_QWEN_ASR:
            qwen_asr_ver = getattr(Qwen3ASRModel, "__version__", "N/A")

    # datasets version
    datasets_ver = "N/A"
    try:
        import datasets
        datasets_ver = datasets.__version__
    except Exception:
        pass

    info = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__ if HAVE_TORCH else "N/A",
        "transformers_version": transformers.__version__ if HAVE_TRANSFORMERS else "N/A",
        "peft_version": peft.__version__ if HAVE_PEFT else "N/A",
        "qwen_asr_version": qwen_asr_ver,
        "datasets_version": datasets_ver,
        "git_commit": get_git_commit(),
        "cuda_available": torch.cuda.is_available() if HAVE_TORCH else False,
        "cuda_version": torch.version.cuda if HAVE_TORCH and torch.cuda.is_available() else "N/A",
        "gpu_count": torch.cuda.device_count() if HAVE_TORCH and torch.cuda.is_available() else 0,
        "gpu_names": gpu_names,
        "device": device_name,
        "world_size": world_size,
        "seed": seed,
        "model_id": model_id,
        "model_revision": model_revision,
        "dtype": dtype,
        "attention": attention,
    }
    return info


class ASRDataset(Dataset):
    """Dataset for ASR SFT reading from JSONL manifest."""

    def __init__(self, manifest_path: Path):
        self.samples: List[Dict[str, Any]] = []
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if "audio" in row and ("text" in row or "answer" in row):
                        self.samples.append(row)
                except json.JSONDecodeError as err:
                    print(f"Warning: line {line_idx} invalid json: {err}", file=sys.stderr)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def build_epoch_sample_indices(
    dataset: ASRDataset,
    strategy: str = "standard",
    epoch: int = 0,
    seed: int = 20260722,
) -> List[int]:
    """Build deterministic sample index list for one epoch under the specified strategy."""
    total_samples = len(dataset)
    if total_samples == 0:
        return []

    if strategy == "standard":
        rng = random.Random(seed + epoch)
        indices = list(range(total_samples))
        rng.shuffle(indices)
        return indices

    degraded_indices = [
        i for i, s in enumerate(dataset.samples)
        if s.get("condition_group") == "degraded"
    ]
    en_clean_indices = [
        i for i, s in enumerate(dataset.samples)
        if s.get("condition_group") == "clean" and s.get("language") == "en"
    ]
    zh_clean_indices = [
        i for i, s in enumerate(dataset.samples)
        if s.get("condition_group") == "clean" and s.get("language") == "zh"
    ]
    other_indices = [
        i for i in range(total_samples)
        if i not in set(degraded_indices + en_clean_indices + zh_clean_indices)
    ]

    rng = random.Random(seed + epoch)

    if strategy == "clean_2x":
        indices = list(degraded_indices) + en_clean_indices * 2 + zh_clean_indices * 2 + other_indices
        rng.shuffle(indices)
        return indices

    if strategy == "balanced":
        # Target: 50% degraded, 25% English clean, 25% Chinese clean
        n_deg = len(degraded_indices)
        if n_deg == 0 or len(en_clean_indices) == 0 or len(zh_clean_indices) == 0:
            indices = list(range(total_samples))
            rng.shuffle(indices)
            return indices

        n_clean_each = n_deg // 2
        sampled_en = (
            rng.choices(en_clean_indices, k=n_clean_each)
            if len(en_clean_indices) < n_clean_each
            else rng.sample(en_clean_indices, k=n_clean_each)
        )
        sampled_zh = (
            rng.choices(zh_clean_indices, k=n_clean_each)
            if len(zh_clean_indices) < n_clean_each
            else rng.sample(zh_clean_indices, k=n_clean_each)
        )

        indices = list(degraded_indices) + sampled_en + sampled_zh
        rng.shuffle(indices)
        return indices

    indices = list(range(total_samples))
    rng.shuffle(indices)
    return indices


def prepare_sft_sample(
    sample: Dict[str, Any],
    asr_model: Any,
) -> Optional[Dict[str, Any]]:
    """Process single audio and text sample into tensors with masked labels."""
    audio_path = sample["audio"]
    if not os.path.isabs(audio_path) and not os.path.exists(audio_path):
        return None

    try:
        wav, sr = sf.read(audio_path)
    except Exception as err:
        print(f"Warning: failed to read {audio_path}: {err}", file=sys.stderr)
        return None

    # Handle multi-channel audio by taking channel 0
    if len(wav.shape) > 1:
        wav = wav[:, 0]

    lang_raw = str(sample.get("language", "en")).strip().lower()
    lang_full = LANGUAGE_MAP.get(lang_raw, "English")

    prompt = asr_model._build_text_prompt(context="", force_language=lang_full)
    gold_text = (sample.get("text") or sample.get("answer", "")).strip()
    target_str = gold_text + "<|im_end|>"
    full_text = prompt + target_str

    processor = asr_model.processor
    inputs = processor(text=full_text, audio=wav, return_tensors="pt")
    target_ids = processor.tokenizer.encode(target_str, add_special_tokens=False)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    input_features = inputs["input_features"]
    feature_attention_mask = inputs["feature_attention_mask"]

    labels = input_ids.clone()
    # Mask all tokens prior to target transcript with -100
    target_len = len(target_ids)
    if target_len >= input_ids.shape[1]:
        labels[:, 0] = -100
    else:
        labels[:, :-target_len] = -100

    return {
        "sample_id": sample.get("sample_id", ""),
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "input_features": input_features,
        "feature_attention_mask": feature_attention_mask,
        "labels": labels,
    }


def evaluate_validation_loss(
    model: Any,
    val_dataset: ASRDataset,
    asr_model: Any,
    device: str,
    max_eval_samples: int = 32,
) -> float:
    """Compute mean cross-entropy loss on validation subset in eval mode without gradients with stratified sampling."""
    if not HAVE_TORCH or len(val_dataset) == 0:
        return 0.0
    eval_model = model.module if hasattr(model, "module") else model
    eval_model.eval()
    total_loss = 0.0
    count = 0

    # Build stratified indices across degraded, English clean, and Chinese clean
    eval_indices: List[int] = []
    if hasattr(val_dataset, "samples") and val_dataset.samples:
        deg_idx = [
            i for i, s in enumerate(val_dataset.samples)
            if s.get("condition_group") == "degraded" or (s.get("scenario") and s.get("scenario") != "clean")
        ]
        en_clean_idx = [
            i for i, s in enumerate(val_dataset.samples)
            if (s.get("condition_group") == "clean" or s.get("scenario") == "clean")
            and str(s.get("language", "")).lower() in ("en", "english")
        ]
        zh_clean_idx = [
            i for i, s in enumerate(val_dataset.samples)
            if (s.get("condition_group") == "clean" or s.get("scenario") == "clean")
            and str(s.get("language", "")).lower() in ("zh", "chinese")
        ]

        # Allocate budget: 50% degraded, 25% en clean, 25% zh clean
        n_deg = max(1, max_eval_samples // 2) if deg_idx else 0
        n_en = max(1, max_eval_samples // 4) if en_clean_idx else 0
        n_zh = max(1, max_eval_samples // 4) if zh_clean_idx else 0

        eval_indices.extend(deg_idx[:n_deg])
        eval_indices.extend(en_clean_idx[:n_en])
        eval_indices.extend(zh_clean_idx[:n_zh])

    if not eval_indices:
        eval_indices = list(range(min(len(val_dataset), max_eval_samples)))

    with torch.no_grad():
        for i in eval_indices:
            sample = val_dataset[i]
            prepared = prepare_sft_sample(sample, asr_model)
            if prepared is None:
                continue
            input_ids = prepared["input_ids"].to(device)
            attention_mask = prepared["attention_mask"].to(device)
            input_features = prepared["input_features"].to(device, dtype=torch.float16)
            feature_attention_mask = prepared["feature_attention_mask"].to(device)
            labels = prepared["labels"].to(device)

            outputs = eval_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += float(loss.item())
                count += 1
    eval_model.train()
    return round(total_loss / count, 5) if count > 0 else 0.0


def attach_lora_to_thinker(
    thinker: Any,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> Tuple[Any, List[str], str]:
    """Attach LoRA to thinker with exact 199 targets and return peft model and target hash."""
    if not HAVE_PEFT:
        raise RuntimeError("PEFT is not installed.")

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=LORA_TARGET_REGEX,
    )
    peft_thinker = get_peft_model(thinker, lora_config)

    # Collect and verify matched linear targets
    lora_linear_names: List[str] = []
    for name, module in peft_thinker.named_modules():
        if isinstance(module, lora_tuners.Linear):
            lora_linear_names.append(name)

    target_map_hash = compute_target_map_hash(lora_linear_names)
    if len(lora_linear_names) != EXPECTED_LORA_TARGET_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_LORA_TARGET_COUNT} LoRA Linear targets, found {len(lora_linear_names)}."
        )

    return peft_thinker, lora_linear_names, target_map_hash


def save_checkpoint(
    ckpt_dir: Path,
    peft_thinker: Any,
    processor: Any,
    optimizer: Any,
    scheduler: Any,
    global_step: int,
    epoch: int,
    dataset_index: int,
    world_size: int,
    rank: int,
    manifest_sha256: str,
    model_revision: str,
    seed: int,
    target_map_hash: str,
    micro_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
    sample_strategy: str = "standard",
    lr_scheduler: str = "linear",
) -> None:
    """Save model adapter, training metadata, and per-rank RNG states."""
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save adapter (rank 0 only)
        adapter_dir = ckpt_dir / "adapter"
        peft_thinker.save_pretrained(adapter_dir)
        if processor is not None and hasattr(processor, "save_pretrained"):
            processor.save_pretrained(adapter_dir)

        # 2. Save optimizer & scheduler states (rank 0 only)
        if HAVE_TORCH and optimizer is not None:
            torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
            if scheduler is not None:
                torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")

        # 3. Save training state metadata (rank 0 only)
        state_meta = {
            "global_step": global_step,
            "epoch": epoch,
            "dataset_index": dataset_index,
            "world_size": world_size,
            "micro_batch_size": micro_batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_global_batch_size": micro_batch_size * gradient_accumulation_steps * world_size,
            "manifest_sha256": manifest_sha256,
            "model_revision": model_revision,
            "seed": seed,
            "target_map_hash": target_map_hash,
            "sample_strategy": sample_strategy,
            "lr_scheduler": lr_scheduler,
            "scheduler": "scheduler.pt" if scheduler is not None else None,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with open(ckpt_dir / "training_state.json", "w", encoding="utf-8") as f:
            json.dump(state_meta, f, indent=2)

    if dist is not None and dist.is_initialized():
        dist.barrier()

    # 4. Save per-rank RNG state
    cuda_state = None
    if HAVE_TORCH and torch.cuda.is_available():
        cuda_state = torch.cuda.get_rng_state()

    rng_state = {
        "torch": torch.get_rng_state() if HAVE_TORCH else None,
        "cuda": cuda_state,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if HAVE_TORCH:
        torch.save(rng_state, ckpt_dir / f"rng_state_rank_{rank}.pt")
        if rank == 0:
            torch.save(rng_state, ckpt_dir / "rng_state.pt")
            print(f"[Rank 0] Saved checkpoint at step {global_step} -> {ckpt_dir}")

    if dist is not None and dist.is_initialized():
        dist.barrier()


def resume_from_checkpoint(
    ckpt_dir: Path,
    peft_thinker: Any,
    optimizer: Any,
    scheduler: Any,
    rank: int = 0,
    current_manifest_sha256: str = "",
    current_world_size: int = 1,
    current_model_revision: str = "",
    current_seed: Optional[int] = None,
    current_gradient_accumulation_steps: Optional[int] = None,
    current_target_map_hash: str = "",
    validate_provenance: bool = True,
    require_states: bool = False,
) -> Dict[str, Any]:
    """Restore training states from checkpoint with provenance checks."""
    state_file = ckpt_dir / "training_state.json"
    if not state_file.is_file():
        raise FileNotFoundError(f"Checkpoint state file missing: {state_file}")

    with open(state_file, "r", encoding="utf-8") as f:
        state_meta = json.load(f)

    if validate_provenance:
        saved_manifest_sha = state_meta.get("manifest_sha256", "")
        if current_manifest_sha256 and saved_manifest_sha and saved_manifest_sha != current_manifest_sha256:
            raise ValueError(
                f"Manifest SHA256 mismatch during resume: checkpoint was trained with {saved_manifest_sha}, "
                f"current manifest has {current_manifest_sha256}"
            )

        saved_world_size = state_meta.get("world_size", 1)
        if current_world_size and saved_world_size and saved_world_size != current_world_size:
            raise ValueError(
                f"World size mismatch during resume: checkpoint was trained with world_size={saved_world_size}, "
                f"current world_size is {current_world_size}"
            )

        saved_revision = state_meta.get("model_revision", "")
        if current_model_revision and saved_revision and saved_revision != current_model_revision:
            raise ValueError(
                f"Model revision mismatch during resume: checkpoint was trained with revision={saved_revision}, "
                f"current revision is {current_model_revision}"
            )

        saved_seed = state_meta.get("seed")
        if current_seed is not None and saved_seed is not None and saved_seed != current_seed:
            raise ValueError(
                f"Seed mismatch during resume: checkpoint was trained with seed={saved_seed}, "
                f"current seed is {current_seed}"
            )

        saved_accum = state_meta.get("gradient_accumulation_steps")
        if current_gradient_accumulation_steps is not None and saved_accum is not None and saved_accum != current_gradient_accumulation_steps:
            raise ValueError(
                f"Gradient accumulation steps mismatch during resume: checkpoint was trained with {saved_accum}, "
                f"current is {current_gradient_accumulation_steps}"
            )

        saved_target_hash = state_meta.get("target_map_hash", "")
        if current_target_map_hash and saved_target_hash and saved_target_hash != current_target_map_hash:
            raise ValueError(
                f"Target map hash mismatch during resume: checkpoint targeted {saved_target_hash}, "
                f"current target map is {current_target_map_hash}"
            )

    # Restore optimizer
    opt_file = ckpt_dir / "optimizer.pt"
    if opt_file.is_file():
        if optimizer is not None:
            optimizer.load_state_dict(torch.load(opt_file, map_location="cpu"))
            if rank == 0:
                print(f"Restored optimizer state from {opt_file}")
    elif require_states:
        raise FileNotFoundError(f"Checkpoint optimizer state file missing: {opt_file}")

    # Restore scheduler
    sched_file = ckpt_dir / "scheduler.pt"
    if sched_file.is_file() and scheduler is not None:
        scheduler.load_state_dict(torch.load(sched_file, map_location="cpu"))
        if rank == 0:
            print(f"Restored scheduler state from {sched_file}")

    # Restore per-rank RNG state
    rank_rng_file = ckpt_dir / f"rng_state_rank_{rank}.pt"
    fallback_rng_file = ckpt_dir / "rng_state.pt"
    target_rng_file = rank_rng_file if rank_rng_file.is_file() else (fallback_rng_file if fallback_rng_file.is_file() else None)

    if target_rng_file is not None and target_rng_file.is_file():
        rng_state = torch.load(target_rng_file, map_location="cpu")
        if rng_state.get("torch") is not None and HAVE_TORCH:
            torch.set_rng_state(rng_state["torch"])
        if rng_state.get("cuda") is not None and HAVE_TORCH and torch.cuda.is_available():
            torch.cuda.set_rng_state(rng_state["cuda"])
        if rng_state.get("numpy") is not None:
            np.random.set_state(rng_state["numpy"])
        if rng_state.get("python") is not None:
            random.setstate(rng_state["python"])
        if rank == 0:
            print(f"Restored RNG state from {target_rng_file}")
    elif require_states:
        raise FileNotFoundError(
            f"RNG state file missing for rank {rank}: expected {rank_rng_file} or {fallback_rng_file}"
        )

    return state_meta


def export_merged_model(
    checkpoint_dir: Path,
    output_dir: Path,
    model_id: str,
    revision: str,
    attention: str = "eager",
) -> None:
    """Load adapter from checkpoint, merge into base thinker, and export merged model."""
    if not HAVE_QWEN_ASR or not HAVE_PEFT:
        raise RuntimeError("qwen_asr and peft must be installed to export merged model.")

    adapter_path = checkpoint_dir / "adapter"
    if not adapter_path.is_dir():
        adapter_path = checkpoint_dir

    print(f"Loading base model {model_id} (revision={revision}, attention={attention}) for merge...")
    asr_model = Qwen3ASRModel.from_pretrained(
        model_id,
        revision=revision or None,
        dtype=torch.float16,
        attn_implementation=attention,
    )

    print(f"Attaching and loading adapter from {adapter_path}...")
    peft_thinker = PeftModel.from_pretrained(asr_model.model.thinker, str(adapter_path))

    print("Executing merge_and_unload()...")
    merged_thinker = peft_thinker.merge_and_unload()
    asr_model.model.thinker = merged_thinker

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged model to {output_dir}...")

    # Fix generation_config temperature validation in newer transformers when do_sample is False
    for obj in [asr_model.model, asr_model.model.thinker]:
        if hasattr(obj, "generation_config") and obj.generation_config is not None:
            if not getattr(obj.generation_config, "do_sample", False):
                obj.generation_config.temperature = None

    asr_model.model.save_pretrained(output_dir)
    if hasattr(asr_model, "processor") and hasattr(asr_model.processor, "save_pretrained"):
        asr_model.processor.save_pretrained(output_dir)

    print("Merged model export complete!")


def verify_dataset_gate_for_training(
    gate_file: Path,
    manifest_path: Path,
    allow_subset: bool = False,
    ignore_dataset_gate: bool = False,
) -> Tuple[bool, str]:
    """Verify DATASET_COMPLETE.json gate status before launching training."""
    if not gate_file.is_file():
        return True, "NOT_FOUND"

    with open(gate_file, "r", encoding="utf-8") as gf:
        gate_data = json.load(gf)
    gate_status = gate_data.get("status", "UNKNOWN")

    if gate_status == "FAILED" and not ignore_dataset_gate:
        raise RuntimeError(
            f"Dataset gate check FAILED in {gate_file}. "
            f"Training blocked until dataset satisfies all sub-quotas and isolation checks. "
            f"(Use --ignore-dataset-gate to bypass during testing)"
        )
    if gate_status == "NON_STRICT_SUBSET":
        is_pilot_or_smoke = "pilot" in manifest_path.name or "smoke" in manifest_path.name
        if not (allow_subset or is_pilot_or_smoke or ignore_dataset_gate):
            raise RuntimeError(
                f"Dataset gate status is NON_STRICT_SUBSET in {gate_file}. "
                f"Formal full training requires PASSED gate status. "
                f"To run pilot, smoke, or experimental training on a dataset subset, provide --allow-subset."
            )
    elif gate_status != "PASSED" and not ignore_dataset_gate:
        raise RuntimeError(
            f"Dataset gate status is '{gate_status}' in {gate_file}, expected 'PASSED'. "
            f"(Use --ignore-dataset-gate to bypass during testing)"
        )
    return True, str(gate_status)


def train_sft(args: argparse.Namespace) -> None:
    if not HAVE_TORCH or not HAVE_QWEN_ASR or not HAVE_PEFT:
        raise RuntimeError("Required ML packages (torch, qwen_asr, peft) not available.")

    # 1. Distributed Setup
    is_distributed = False
    rank = 0
    local_rank = 0
    world_size = 1

    if not args.single_gpu and "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        is_distributed = True
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    set_seed(args.seed + rank)

    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Config & Provenance
    config_dict: Dict[str, Any] = {}
    if args.config and Path(args.config).is_file():
        with open(args.config, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}

    model_id = args.model_id or config_dict.get("model", {}).get("model_id", "Qwen/Qwen3-ASR-1.7B")
    revision = args.revision or config_dict.get("model", {}).get("model_revision", "7278e1e70fe206f11671096ffdd38061171dd6e5")
    attention = config_dict.get("model", {}).get("attention_implementation", "eager")

    micro_batch_size = (
        args.micro_batch_size
        if args.micro_batch_size is not None
        else config_dict.get("runtime", {}).get("micro_batch_size", 1)
    )
    gradient_accumulation_steps = (
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else config_dict.get("runtime", {}).get("gradient_accumulation_steps", 16)
    )
    effective_global_batch_size = micro_batch_size * gradient_accumulation_steps * world_size

    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    # Dataset Gate Check
    gate_file = manifest_path.parent / "DATASET_COMPLETE.json"
    _, gate_status = verify_dataset_gate_for_training(
        gate_file,
        manifest_path,
        allow_subset=args.allow_subset,
        ignore_dataset_gate=args.ignore_dataset_gate,
    )
    if rank == 0 and gate_status != "NOT_FOUND":
        print(f"Dataset gate verified from {gate_file}: status={gate_status}")

    # Pilot manifest category check if training on pilot
    if "pilot" in manifest_path.name:
        with open(manifest_path, "r", encoding="utf-8") as mf:
            pilot_lines = [json.loads(line) for line in mf if line.strip()]
        has_deg = any(r.get("condition_group") == "degraded" for r in pilot_lines)
        has_en = any(r.get("condition_group") == "clean" and r.get("language") == "en" for r in pilot_lines)
        has_zh = any(r.get("condition_group") == "clean" and r.get("language") == "zh" for r in pilot_lines)
        if not (has_deg and has_en and has_zh) and not args.ignore_dataset_gate:
            raise RuntimeError(
                f"Pilot manifest {manifest_path} is invalid: must contain degraded, english_clean, and chinese_clean samples."
            )

    manifest_sha256 = calculate_file_sha256(manifest_path)

    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else float(config_dict.get("train", {}).get("learning_rate", 2e-5))
    )
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps is not None
        else int(config_dict.get("train", {}).get("warmup_steps", 0))
    )
    lr_scheduler_type = (
        args.lr_scheduler
        if args.lr_scheduler is not None
        else config_dict.get("train", {}).get("lr_scheduler", "linear" if warmup_steps > 0 else "constant")
    )
    sample_strategy = (
        args.sample_strategy
        if args.sample_strategy is not None
        else config_dict.get("train", {}).get("sample_strategy", "standard")
    )
    eval_steps = (
        args.eval_steps
        if args.eval_steps > 0
        else int(config_dict.get("train", {}).get("eval_steps", 0))
    )
    val_manifest = args.val_manifest or config_dict.get("train", {}).get("val_manifest", "")

    max_steps = (
        args.max_steps
        if args.max_steps is not None
        else int(config_dict.get("train", {}).get("max_steps", 10))
    )
    save_steps = (
        args.save_steps
        if args.save_steps is not None
        else int(config_dict.get("train", {}).get("save_steps", 10))
    )

    resolved_config = {
        "model_id": model_id,
        "revision": revision,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "max_steps": max_steps,
        "save_steps": save_steps,
        "learning_rate": learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_steps": warmup_steps,
        "lr_scheduler": lr_scheduler_type,
        "sample_strategy": sample_strategy,
        "val_manifest": val_manifest,
        "eval_steps": eval_steps,
        "micro_batch_size": micro_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_global_batch_size": effective_global_batch_size,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "world_size": world_size,
        "single_gpu": args.single_gpu,
        "device": device,
        "dtype": "float16",
        "attention": attention,
        "expected_lora_targets": EXPECTED_LORA_TARGET_COUNT,
    }

    if rank == 0:
        print(
            f"Resolved training config: micro_batch_size={micro_batch_size}, "
            f"grad_accum={gradient_accumulation_steps}, world_size={world_size}, "
            f"effective_global_batch_size={effective_global_batch_size}, attention={attention}"
        )
        # Write environment.json & resolved_config.yaml & manifest_sha256.json
        with open(output_dir / "environment.json", "w", encoding="utf-8") as f:
            json.dump(
                get_environment_info(
                    device,
                    world_size,
                    args.seed,
                    model_id=model_id,
                    model_revision=revision,
                    dtype="float16",
                    attention=attention,
                ),
                f,
                indent=2,
            )

        with open(output_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(resolved_config, f, default_flow_style=False)

        with open(output_dir / "manifest_sha256.json", "w", encoding="utf-8") as f:
            json.dump({"manifest": str(manifest_path), "sha256": manifest_sha256}, f, indent=2)

    # 3. Model Loading & LoRA Attachment
    if rank == 0:
        print(f"Loading Qwen3-ASR model from {model_id} (revision={revision}, dtype=float16, attention={attention})...")

    load_kwargs: Dict[str, Any] = {
        "dtype": torch.float16,
        "attn_implementation": attention,
    }
    if revision:
        load_kwargs["revision"] = revision

    asr_model = Qwen3ASRModel.from_pretrained(model_id, **load_kwargs)
    thinker = asr_model.model.thinker
    thinker.to(device)

    # Gradient checkpointing
    if hasattr(thinker, "enable_input_require_grads"):
        thinker.enable_input_require_grads()
    if hasattr(thinker, "gradient_checkpointing_enable"):
        thinker.gradient_checkpointing_enable()

    # LoRA attachment or loading from resume checkpoint
    resumed_step = 0
    resumed_epoch = 0
    resumed_sample_idx = 0

    if args.resume_from_checkpoint:
        resume_ckpt_dir = Path(args.resume_from_checkpoint)
        adapter_path = resume_ckpt_dir / "adapter"
        if not adapter_path.is_dir():
            adapter_path = resume_ckpt_dir
        if rank == 0:
            print(f"Resuming LoRA adapter from {adapter_path}...")
        peft_thinker = PeftModel.from_pretrained(thinker, str(adapter_path), is_trainable=True)

        lora_linear_names = [n for n, m in peft_thinker.named_modules() if isinstance(m, lora_tuners.Linear)]
        target_map_hash = compute_target_map_hash(lora_linear_names)
    else:
        if rank == 0:
            print(f"Attaching LoRA with exact {EXPECTED_LORA_TARGET_COUNT} Linear targets...")
        peft_thinker, lora_linear_names, target_map_hash = attach_lora_to_thinker(
            thinker,
            r=config_dict.get("lora", {}).get("r", 16),
            lora_alpha=config_dict.get("lora", {}).get("lora_alpha", 32),
            lora_dropout=config_dict.get("lora", {}).get("lora_dropout", 0.05),
        )

    if rank == 0:
        print(f"Verified LoRA target count: {len(lora_linear_names)} (hash: {target_map_hash[:12]})")

    # 4. Optimizer & Scheduler
    trainable_params = [p for p in peft_thinker.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = None
    if HAVE_TRANSFORMERS and lr_scheduler_type and lr_scheduler_type != "none":
        if lr_scheduler_type == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps,
            )
        elif lr_scheduler_type == "cosine" and get_cosine_schedule_with_warmup is not None:
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps,
            )
        elif lr_scheduler_type == "constant":
            if warmup_steps > 0 and get_constant_schedule_with_warmup is not None:
                scheduler = get_constant_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                )

    # 5. Restore optimizer and states if resuming
    if args.resume_from_checkpoint:
        resume_ckpt_dir = Path(args.resume_from_checkpoint)
        meta = resume_from_checkpoint(
            ckpt_dir=resume_ckpt_dir,
            peft_thinker=peft_thinker,
            optimizer=optimizer,
            scheduler=scheduler,
            rank=rank,
            current_manifest_sha256=manifest_sha256,
            current_world_size=world_size,
            current_model_revision=revision,
            current_seed=args.seed,
            current_gradient_accumulation_steps=gradient_accumulation_steps,
            current_target_map_hash=target_map_hash,
            validate_provenance=True,
            require_states=True,
        )
        resumed_step = meta.get("global_step", 0)
        resumed_epoch = meta.get("epoch", 0)
        resumed_sample_idx = meta.get("dataset_index", 0)
        if rank == 0:
            print(f"Successfully resumed from step {resumed_step} (dataset_index={resumed_sample_idx})")

    # Wrap in DDP if multi-GPU
    train_model = peft_thinker
    if is_distributed:
        train_model = DDP(peft_thinker, device_ids=[local_rank], find_unused_parameters=True)
        if hasattr(train_model, "_set_static_graph"):
            train_model._set_static_graph()

    # 6. Dataset
    dataset = ASRDataset(manifest_path)
    if rank == 0:
        print(f"Loaded {len(dataset)} samples from manifest {manifest_path} (sampling strategy: {sample_strategy})")

    val_dataset = None
    if val_manifest and Path(val_manifest).is_file():
        val_dataset = ASRDataset(Path(val_manifest))
        if rank == 0:
            print(f"Loaded {len(val_dataset)} validation samples from {val_manifest} for loss evaluation")

    # 7. Training Loop
    global_step = resumed_step
    loss_log_path = output_dir / "loss_log.jsonl"

    if rank == 0:
        print(f"Starting training: global_step={global_step}/{max_steps} on {world_size} rank(s)...")

    start_time = time.time()
    optimizer.zero_grad()

    epoch_cache: Dict[int, List[int]] = {}

    def get_epoch_indices(ep: int) -> List[int]:
        if ep not in epoch_cache:
            epoch_cache[ep] = build_epoch_sample_indices(
                dataset,
                strategy=sample_strategy,
                epoch=ep,
                seed=args.seed,
            )
        return epoch_cache[ep]

    epoch_0_indices = get_epoch_indices(0)
    virtual_epoch_len = max(len(epoch_0_indices), 1)

    while global_step < max_steps:
        step_t0 = time.time()
        step_loss_acc = 0.0

        for g in range(gradient_accumulation_steps):
            total_micro_step = global_step * gradient_accumulation_steps + g
            cluster_sample_idx = total_micro_step * world_size + rank
            epoch = cluster_sample_idx // virtual_epoch_len
            in_epoch_idx = cluster_sample_idx % virtual_epoch_len

            idx = get_epoch_indices(epoch)[in_epoch_idx]
            sample = dataset[idx]
            prepared = prepare_sft_sample(sample, asr_model)
            if prepared is None:
                raise RuntimeError(f"Failed to prepare sample {sample.get('sample_id')} at dataset idx {idx}")

            input_ids = prepared["input_ids"].to(device)
            attention_mask = prepared["attention_mask"].to(device)
            input_features = prepared["input_features"].to(device, dtype=torch.float16)
            feature_attention_mask = prepared["feature_attention_mask"].to(device)
            labels = prepared["labels"].to(device)

            outputs = train_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"Non-finite loss detected at step {global_step}, micro {g}: {loss.item()}")

            scaled_loss = loss / gradient_accumulation_steps
            scaled_loss.backward()
            step_loss_acc += loss.item()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        optimizer.zero_grad()
        global_step += 1

        step_duration = time.time() - step_t0
        avg_loss = step_loss_acc / gradient_accumulation_steps
        current_lr = optimizer.param_groups[0]["lr"]

        val_loss = None
        if val_dataset is not None and eval_steps > 0 and (global_step % eval_steps == 0 or global_step >= max_steps):
            val_loss = evaluate_validation_loss(
                model=peft_thinker,
                val_dataset=val_dataset,
                asr_model=asr_model,
                device=device,
                max_eval_samples=64,
            )

        log_entry = {
            "step": global_step,
            "loss": round(float(avg_loss), 5),
            "lr": current_lr,
            "step_seconds": round(step_duration, 4),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        if val_loss is not None:
            log_entry["val_loss"] = val_loss

        if rank == 0:
            val_str = f" - Val Loss: {val_loss:.4f}" if val_loss is not None else ""
            print(f"Step [{global_step}/{max_steps}] - Loss: {log_entry['loss']:.4f}{val_str} - LR: {current_lr:.2e} - {step_duration:.2f}s")
            with open(loss_log_path, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(log_entry) + "\n")

        # Checkpoint saving
        if global_step % save_steps == 0 or global_step >= max_steps:
            ckpt_dir = output_dir / "checkpoints" / f"step_{global_step}"
            save_checkpoint(
                ckpt_dir=ckpt_dir,
                peft_thinker=peft_thinker,
                processor=asr_model.processor,
                optimizer=optimizer,
                scheduler=scheduler,
                global_step=global_step,
                epoch=(global_step * gradient_accumulation_steps * world_size) // virtual_epoch_len,
                dataset_index=global_step * gradient_accumulation_steps * world_size,
                world_size=world_size,
                rank=rank,
                manifest_sha256=manifest_sha256,
                model_revision=revision,
                seed=args.seed,
                target_map_hash=target_map_hash,
                micro_batch_size=micro_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                sample_strategy=sample_strategy,
                lr_scheduler=str(lr_scheduler_type),
            )

            if rank == 0:
                pipeline_state = {
                    "phase": "sft",
                    "global_step": global_step,
                    "max_steps": max_steps,
                    "world_size": world_size,
                    "latest_checkpoint": str(ckpt_dir),
                    "status": "completed" if global_step >= max_steps else "in_progress",
                    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                }
                with open(output_dir / "pipeline_state.json", "w", encoding="utf-8") as psf:
                    json.dump(pipeline_state, psf, indent=2)

            if is_distributed:
                dist.barrier()

    total_time = time.time() - start_time
    if rank == 0:
        print(f"Training completed: reached global_step {global_step} in {total_time:.1f}s")

    if is_distributed:
        dist.destroy_process_group()


def main() -> None:
    args = parse_args()
    if args.export_merged:
        if not args.checkpoint_dir or not args.output_dir:
            print("Error: --export-merged requires --checkpoint-dir and --output-dir", file=sys.stderr)
            sys.exit(1)
        config_dict: Dict[str, Any] = {}
        if args.config and Path(args.config).is_file():
            with open(args.config, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f) or {}
        model_id = args.model_id or config_dict.get("model", {}).get("model_id", "Qwen/Qwen3-ASR-1.7B")
        revision = args.revision or config_dict.get("model", {}).get("model_revision", "7278e1e70fe206f11671096ffdd38061171dd6e5")
        attention = config_dict.get("model", {}).get("attention_implementation", "eager")
        export_merged_model(
            checkpoint_dir=Path(args.checkpoint_dir),
            output_dir=Path(args.output_dir),
            model_id=model_id,
            revision=revision,
            attention=attention,
        )
        return

    if not args.manifest or not args.output_dir:
        print("Error: training requires --manifest and --output-dir", file=sys.stderr)
        sys.exit(1)

    train_sft(args)


if __name__ == "__main__":
    main()
