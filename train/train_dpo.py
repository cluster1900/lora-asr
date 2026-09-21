#!/usr/bin/env python3
"""DPO (Direct Preference Optimization) Training Runner for Qwen3-ASR on V100 server.

Supports:
- Official qwen-asr / Transformers API
- Operation directly on model.model.thinker (Qwen3ASRThinkerForConditionalGeneration)
- Exact 199 Linear LoRA targets (3 Projection + 196 Decoder attention/mlp)
- FP16, eager attention, gradient checkpointing with input require grads
- Single-GPU and multi-GPU (DDP via torchrun)
- Precomputed reference log probabilities (ref_chosen_logp, ref_rejected_logp) to cut GPU VRAM by ~50%
- Real-time tracking of DPO loss, reward margin, and preference accuracy
- Held-out validation evaluation of preference accuracy
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
    import torch.nn.functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, Dataset, DistributedSampler
    HAVE_TORCH = True
except ImportError:
    torch = None  # type: ignore
    dist = None  # type: ignore
    F = None  # type: ignore
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


LORA_TARGET_REGEX = re.compile(
    r"^(audio_tower\.(conv_out|proj1|proj2)|model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj)))$"
)

LANGUAGE_MAP = {
    "en": "English",
    "english": "English",
    "zh": "Chinese",
    "chinese": "Chinese",
}


def compute_file_sha256(path: Path) -> str:
    """Compute sha256 checksum of a file."""
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_environment_info() -> Dict[str, Any]:
    """Capture environment and hardware provenance."""
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
    }
    if HAVE_TORCH:
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["device_count"] = torch.cuda.device_count()
            info["devices"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    if HAVE_TRANSFORMERS:
        info["transformers_version"] = transformers.__version__
    if HAVE_PEFT:
        info["peft_version"] = peft.__version__
    info["qwen_asr_version"] = "0.0.6"
    info["model_id"] = "Qwen/Qwen3-ASR-1.7B"
    info["model_revision"] = "7278e1e70fe206f11671096ffdd38061171dd6e5"
    info["dtype"] = "float16"
    info["attention"] = "eager"
    return info


class DPOPreferenceDataset(Dataset):
    """Dataset for DPO preference pairs."""

    def __init__(self, manifest_path: Path):
        self.manifest_path = manifest_path
        self.samples: List[Dict[str, Any]] = []
        if manifest_path.is_file():
            with open(manifest_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def prepare_dpo_sample(
    sample: Dict[str, Any],
    asr_model: Any,
) -> Optional[Dict[str, Any]]:
    """Tokenize audio and build masked labels for chosen and rejected transcripts."""
    audio_path = sample["audio"]
    if not os.path.isabs(audio_path) and not os.path.exists(audio_path):
        return None

    try:
        wav, sr = sf.read(audio_path)
    except Exception as err:
        print(f"Warning: failed to read {audio_path}: {err}", file=sys.stderr)
        return None

    if len(wav.shape) > 1:
        wav = wav[:, 0]

    lang_raw = str(sample.get("language", "en")).strip().lower()
    lang_full = LANGUAGE_MAP.get(lang_raw, "English")

    processor = asr_model.processor
    prompt = asr_model._build_text_prompt(context="", force_language=lang_full)

    chosen_text = sample.get("chosen", "").strip()
    rejected_text = sample.get("rejected", "").strip()

    if not chosen_text or not rejected_text:
        return None

    # Prepare chosen
    chosen_target_str = chosen_text + "<|im_end|>"
    chosen_full_text = prompt + chosen_target_str
    inputs_chosen = processor(text=chosen_full_text, audio=wav, return_tensors="pt")
    chosen_target_ids = processor.tokenizer.encode(chosen_target_str, add_special_tokens=False)

    chosen_input_ids = inputs_chosen["input_ids"]
    chosen_attention_mask = inputs_chosen["attention_mask"]
    input_features = inputs_chosen["input_features"]
    feature_attention_mask = inputs_chosen["feature_attention_mask"]

    chosen_labels = chosen_input_ids.clone()
    tlen_c = len(chosen_target_ids)
    if tlen_c >= chosen_input_ids.shape[1]:
        chosen_labels[:, 0] = -100
    else:
        chosen_labels[:, :-tlen_c] = -100

    # Prepare rejected
    rejected_target_str = rejected_text + "<|im_end|>"
    rejected_full_text = prompt + rejected_target_str
    inputs_rejected = processor(text=rejected_full_text, audio=wav, return_tensors="pt")
    rejected_target_ids = processor.tokenizer.encode(rejected_target_str, add_special_tokens=False)

    rejected_input_ids = inputs_rejected["input_ids"]
    rejected_attention_mask = inputs_rejected["attention_mask"]

    rejected_labels = rejected_input_ids.clone()
    tlen_r = len(rejected_target_ids)
    if tlen_r >= rejected_input_ids.shape[1]:
        rejected_labels[:, 0] = -100
    else:
        rejected_labels[:, :-tlen_r] = -100

    max_len = max(chosen_input_ids.shape[1], rejected_input_ids.shape[1])
    pad_token_id = processor.tokenizer.pad_token_id if (hasattr(processor, "tokenizer") and processor.tokenizer.pad_token_id is not None) else 0

    c_pad = max_len - chosen_input_ids.shape[1]
    if c_pad > 0:
        c_ids_padded = F.pad(chosen_input_ids, (0, c_pad), value=pad_token_id)
        c_mask_padded = F.pad(chosen_attention_mask, (0, c_pad), value=0)
        c_labels_padded = F.pad(chosen_labels, (0, c_pad), value=-100)
    else:
        c_ids_padded = chosen_input_ids
        c_mask_padded = chosen_attention_mask
        c_labels_padded = chosen_labels

    r_pad = max_len - rejected_input_ids.shape[1]
    if r_pad > 0:
        r_ids_padded = F.pad(rejected_input_ids, (0, r_pad), value=pad_token_id)
        r_mask_padded = F.pad(rejected_attention_mask, (0, r_pad), value=0)
        r_labels_padded = F.pad(rejected_labels, (0, r_pad), value=-100)
    else:
        r_ids_padded = rejected_input_ids
        r_mask_padded = rejected_attention_mask
        r_labels_padded = rejected_labels

    batch_input_ids = torch.cat([c_ids_padded, r_ids_padded], dim=0)
    batch_attention_mask = torch.cat([c_mask_padded, r_mask_padded], dim=0)
    batch_labels = torch.cat([c_labels_padded, r_labels_padded], dim=0)
    batch_input_features = torch.cat([input_features, input_features], dim=0)
    batch_feature_attention_mask = torch.cat([feature_attention_mask, feature_attention_mask], dim=0)

    ref_chosen_logp = float(sample.get("ref_chosen_logp", 0.0))
    ref_rejected_logp = float(sample.get("ref_rejected_logp", 0.0))

    return {
        "sample_id": sample.get("sample_id", ""),
        "batch_input_features": batch_input_features,
        "batch_feature_attention_mask": batch_feature_attention_mask,
        "batch_input_ids": batch_input_ids,
        "batch_attention_mask": batch_attention_mask,
        "batch_labels": batch_labels,
        "ref_chosen_logp": ref_chosen_logp,
        "ref_rejected_logp": ref_rejected_logp,
    }


def compute_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Compute per-sequence autoregressive log-probabilities masked on target tokens."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    mask = (shift_labels != ignore_index)
    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0
    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token = torch.gather(
        log_probs, dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return (per_token * mask).sum(dim=-1)


def evaluate_dpo_validation(
    model: Any,
    val_dataset: DPOPreferenceDataset,
    asr_model: Any,
    device: str,
    beta: float = 0.1,
    max_eval_samples: Optional[int] = None,
) -> Tuple[float, float]:
    """Evaluate DPO loss and preference accuracy on validation set without gradients."""
    if not HAVE_TORCH or len(val_dataset) == 0:
        return 0.0, 0.0
    eval_model = model.module if hasattr(model, "module") else model
    eval_model.eval()

    total_loss = 0.0
    correct_prefs = 0
    count = 0

    if max_eval_samples is not None and max_eval_samples > 0:
        indices = list(range(min(len(val_dataset), max_eval_samples)))
    else:
        indices = list(range(len(val_dataset)))

    with torch.no_grad():
        for i in indices:
            sample = val_dataset[i]
            prepared = prepare_dpo_sample(sample, asr_model)
            if prepared is None:
                continue

            input_features = prepared["batch_input_features"].to(
                device, dtype=torch.float16 if device.startswith("cuda") else torch.float32
            )
            feat_mask = prepared["batch_feature_attention_mask"].to(device)
            input_ids = prepared["batch_input_ids"].to(device)
            attention_mask = prepared["batch_attention_mask"].to(device)
            labels = prepared["batch_labels"].to(device)

            outputs = eval_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_features=input_features,
                feature_attention_mask=feat_mask,
            )
            logps = compute_logps(outputs.logits, labels)
            pi_c_logp = logps[0]
            pi_r_logp = logps[1]

            ref_c_logp = torch.tensor(prepared["ref_chosen_logp"], device=device)
            ref_r_logp = torch.tensor(prepared["ref_rejected_logp"], device=device)

            pi_ratio = pi_c_logp - pi_r_logp
            ref_ratio = ref_c_logp - ref_r_logp
            logits = pi_ratio - ref_ratio

            loss = -F.logsigmoid(beta * logits).item()
            total_loss += loss
            if logits.item() > 0:
                correct_prefs += 1
            count += 1

    mean_loss = total_loss / max(1, count)
    accuracy = correct_prefs / max(1, count)
    return mean_loss, accuracy


def setup_lora_dpo(thinker: Any, config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Inject exact 199 canonical Linear LoRA targets into thinker model."""
    target_modules: List[str] = []
    for name, module in thinker.named_modules():
        if isinstance(module, torch.nn.Linear) and LORA_TARGET_REGEX.match(name):
            target_modules.append(name)

    lora_cfg = config.get("lora", {})
    r = int(lora_cfg.get("r", 16))
    alpha = int(lora_cfg.get("lora_alpha", 32))
    dropout = float(lora_cfg.get("lora_dropout", 0.05))

    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    lora_model = get_peft_model(thinker, peft_config)

    target_map_canonical = sorted(target_modules)
    target_map_hash = hashlib.sha256(
        json.dumps(target_map_canonical).encode()
    ).hexdigest()

    meta = {
        "target_count": len(target_modules),
        "target_map_hash": target_map_hash,
        "target_modules": target_map_canonical,
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
    }
    return lora_model, meta


def export_merged_model(checkpoint_dir: Path, output_dir: Path, config: Dict[str, Any]) -> None:
    """Load base model and adapter, execute merge_and_unload(), and save standalone weights."""
    model_cfg = config.get("model", {})
    model_id = str(model_cfg.get("model_id", "Qwen/Qwen3-ASR-1.7B"))
    revision = model_cfg.get("model_revision", "7278e1e70fe206f11671096ffdd38061171dd6e5")

    print(f"Loading model {model_id} (revision={revision}) for merge...")
    asr_model = Qwen3ASRModel.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float16,
        device_map="cpu",
        attn_implementation="eager",
    )

    adapter_path = checkpoint_dir / "adapter"
    if not adapter_path.exists():
        adapter_path = checkpoint_dir

    print(f"Attaching and loading adapter from {adapter_path}...")
    asr_model.model.thinker = PeftModel.from_pretrained(
        asr_model.model.thinker,
        str(adapter_path),
    )

    print("Executing merge_and_unload()...")
    asr_model.model.thinker = asr_model.model.thinker.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving merged DPO model to {output_dir}...")

    # Fix generation_config temperature validation in newer transformers when do_sample is False
    for obj in [asr_model.model, asr_model.model.thinker]:
        if hasattr(obj, "generation_config") and obj.generation_config is not None:
            if not getattr(obj.generation_config, "do_sample", False):
                obj.generation_config.temperature = None

    asr_model.model.save_pretrained(str(output_dir))
    if hasattr(asr_model, "processor") and hasattr(asr_model.processor, "save_pretrained"):
        asr_model.processor.save_pretrained(str(output_dir))
    print("Merged DPO model export complete!")


def train_dpo(
    manifest_path: Path,
    config_path: Path,
    output_dir: Path,
    val_manifest_path: Optional[Path] = None,
    resume_from_checkpoint: Optional[Path] = None,
    max_steps_override: Optional[int] = None,
    save_steps_override: Optional[int] = None,
    single_gpu: bool = False,
    allow_subset: bool = False,
    val_eval_samples: Optional[int] = None,
) -> None:
    """Main entrypoint for single-GPU or 4-card DDP DPO training."""
    rank = 0
    world_size = 1
    local_rank = 0
    is_distributed = False

    if not single_gpu and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        is_distributed = True
        device = f"cuda:{local_rank}"
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve runtime parameters
    runtime_cfg = config.get("runtime", {})
    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {})

    seed = int(runtime_cfg.get("seed", 20260722))
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)

    max_steps = max_steps_override or int(train_cfg.get("max_steps", 150))
    save_steps = save_steps_override or int(train_cfg.get("save_steps", 25))
    eval_steps = int(train_cfg.get("eval_steps", 25))
    val_eval_samples = val_eval_samples if val_eval_samples is not None else train_cfg.get("val_eval_samples", None)
    lr = float(train_cfg.get("learning_rate", 5.0e-6))
    warmup_steps = int(train_cfg.get("warmup_steps", 20))
    beta = float(train_cfg.get("beta", 0.1))
    grad_accum_steps = int(runtime_cfg.get("gradient_accumulation_steps", 16))

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)

    # Verify dataset gate
    dataset_gate_path = manifest_path.parent / "DATASET_COMPLETE.json"
    if dataset_gate_path.is_file():
        with open(dataset_gate_path, "r", encoding="utf-8") as dgf:
            gate_data = json.load(dgf)
        gate_status = gate_data.get("status", "UNKNOWN")
        if gate_status == "FAILED":
            raise RuntimeError(f"Dataset gate check FAILED in {dataset_gate_path}.")
        if gate_status == "NON_STRICT_SUBSET":
            is_pilot_or_smoke = "pilot" in manifest_path.name or "smoke" in manifest_path.name
            if not (allow_subset or is_pilot_or_smoke):
                raise RuntimeError(
                    f"Dataset gate status is NON_STRICT_SUBSET in {dataset_gate_path}. "
                    f"Formal full training requires PASSED gate status. "
                    f"Provide --allow-subset for pilot or smoke runs."
                )

    # Load dataset
    train_dataset = DPOPreferenceDataset(manifest_path)
    val_dataset = DPOPreferenceDataset(val_manifest_path) if val_manifest_path else None

    # Load base model
    model_id = str(model_cfg.get("model_id", "Qwen/Qwen3-ASR-1.7B"))
    revision = model_cfg.get("model_revision", "7278e1e70fe206f11671096ffdd38061171dd6e5")

    if rank == 0:
        print(f"Loading initial model {model_id} (revision={revision}) on rank {rank}...")

    asr_model = Qwen3ASRModel.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        attn_implementation="eager",
    )
    thinker = asr_model.model.thinker
    thinker.to(device)
    thinker.train()

    # Enable gradient checkpointing and input grads
    if runtime_cfg.get("gradient_checkpointing", True):
        if hasattr(thinker, "enable_input_require_grads"):
            thinker.enable_input_require_grads()
        if hasattr(thinker, "gradient_checkpointing_enable"):
            thinker.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    lora_model, lora_meta = setup_lora_dpo(thinker, config)

    if is_distributed:
        ddp_model = DDP(
            lora_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    else:
        ddp_model = lora_model

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        [p for p in ddp_model.parameters() if p.requires_grad],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )

    if warmup_steps > 0:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_steps,
        )
    else:
        scheduler = None

    start_step = 0
    # Resumption
    if resume_from_checkpoint and resume_from_checkpoint.is_dir():
        state_file = resume_from_checkpoint / "training_state.json"
        if state_file.is_file():
            with open(state_file, "r") as sf_f:
                st_data = json.load(sf_f)
                start_step = st_data.get("global_step", 0)

        opt_file = resume_from_checkpoint / "optimizer.pt"
        if opt_file.is_file():
            optimizer.load_state_dict(torch.load(opt_file, map_location=device))

        sched_file = resume_from_checkpoint / "scheduler.pt"
        if sched_file.is_file() and scheduler is not None:
            scheduler.load_state_dict(torch.load(sched_file, map_location=device))

        rng_file = resume_from_checkpoint / f"rng_state_rank_{rank}.pt"
        if rng_file.is_file():
            rng_state = torch.load(rng_file)
            random.setstate(rng_state["python_rng"])
            np.random.set_state(rng_state["numpy_rng"])
            torch.set_rng_state(rng_state["torch_rng"])
            if torch.cuda.is_available() and "cuda_rng" in rng_state:
                torch.cuda.set_rng_state(rng_state["cuda_rng"])

        if rank == 0:
            print(f"Resumed from checkpoint {resume_from_checkpoint} at global step {start_step}")

    # Write provenance files on rank 0
    if rank == 0:
        with open(output_dir / "resolved_config.yaml", "w") as f:
            yaml.dump(config, f)
        with open(output_dir / "environment.json", "w") as f:
            json.dump(get_environment_info(), f, indent=2)
        with open(output_dir / "manifest_sha256.json", "w") as f:
            json.dump({
                "manifest": str(manifest_path),
                "sha256": compute_file_sha256(manifest_path),
                "samples_count": len(train_dataset),
            }, f, indent=2)

    loss_log_path = output_dir / "loss_log.jsonl"
    global_step = start_step

    sample_cursor = global_step * grad_accum_steps * world_size
    n_train = len(train_dataset)

    optimizer.zero_grad()
    accum_count = 0
    accum_loss = 0.0
    accum_acc = 0.0
    step_start_time = time.time()

    if rank == 0:
        print(f"Starting DPO training: global_step={global_step} -> {max_steps}, accum={grad_accum_steps}, world_size={world_size}")

    while global_step < max_steps:
        sample_idx = (sample_cursor + rank) % max(1, n_train)
        sample = train_dataset[sample_idx]
        sample_cursor += world_size

        prepared = prepare_dpo_sample(sample, asr_model)
        if prepared is None:
            continue

        input_features = prepared["batch_input_features"].to(
            device, dtype=torch.float16 if device.startswith("cuda") else torch.float32
        )
        feat_mask = prepared["batch_feature_attention_mask"].to(device)
        input_ids = prepared["batch_input_ids"].to(device)
        attention_mask = prepared["batch_attention_mask"].to(device)
        labels = prepared["batch_labels"].to(device)

        outputs = ddp_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feat_mask,
        )
        logps = compute_logps(outputs.logits, labels)
        pi_c_logp = logps[0]
        pi_r_logp = logps[1]

        ref_c_logp = torch.tensor(prepared["ref_chosen_logp"], device=device)
        ref_r_logp = torch.tensor(prepared["ref_rejected_logp"], device=device)

        pi_ratio = pi_c_logp - pi_r_logp
        ref_ratio = ref_c_logp - ref_r_logp
        logits = pi_ratio - ref_ratio

        # Step 0 alignment invariant check
        if global_step == 0 and accum_count == 0:
            c_diff = abs(pi_c_logp.item() - ref_c_logp.item())
            r_diff = abs(pi_r_logp.item() - ref_r_logp.item())
            if rank == 0:
                print(f"[Step 0 Sanity Check] Chosen logp diff={c_diff:.6e}, Rejected logp diff={r_diff:.6e}")
            if c_diff > 0.05 or r_diff > 0.05:
                raise ValueError(
                    f"Step 0 logp mismatch: chosen diff={c_diff:.4f}, rejected diff={r_diff:.4f}. "
                    "Precomputed reference logps do not match online policy Step 0 logps. "
                    "Ensure prompt template and tokenizer settings in build_dpo_pairs.py match train_dpo.py."
                )

        loss = -F.logsigmoid(beta * logits).mean()
        scaled_loss = loss / grad_accum_steps
        scaled_loss.backward()

        accum_loss += loss.item()
        accum_acc += (logits.item() > 0)
        accum_count += 1

        if accum_count == grad_accum_steps:
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            step_duration = time.time() - step_start_time
            step_start_time = time.time()

            mean_loss = accum_loss / grad_accum_steps
            mean_acc = accum_acc / grad_accum_steps
            accum_loss = 0.0
            accum_acc = 0.0
            accum_count = 0

            # Gather metrics across ranks if distributed
            if is_distributed:
                loss_t = torch.tensor([mean_loss, mean_acc], device=device)
                dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
                mean_loss = (loss_t[0] / world_size).item()
                mean_acc = (loss_t[1] / world_size).item()

            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else lr

            if rank == 0:
                log_entry = {
                    "global_step": global_step,
                    "loss": round(mean_loss, 5),
                    "preference_accuracy": round(mean_acc, 4),
                    "learning_rate": current_lr,
                    "step_seconds": round(step_duration, 3),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                print(f"[Step {global_step}/{max_steps}] loss={mean_loss:.4f}, pref_acc={mean_acc:.3f}, lr={current_lr:.2e}, time={step_duration:.2f}s")
                with open(loss_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")

            # Periodic validation
            if val_dataset and (global_step % eval_steps == 0 or global_step == max_steps):
                # Always evaluate full validation set at max_steps
                samples_to_eval = None if global_step == max_steps else val_eval_samples
                v_loss, v_acc = evaluate_dpo_validation(
                    ddp_model, val_dataset, asr_model, device, beta=beta, max_eval_samples=samples_to_eval
                )
                if rank == 0:
                    eval_tag = "Full Held-out" if samples_to_eval is None else f"Sub-{samples_to_eval}"
                    print(f"--- [Validation @ Step {global_step} ({eval_tag})] val_loss={v_loss:.4f}, val_pref_acc={v_acc:.4f} ---")
                    val_log_entry = {
                        "global_step": global_step,
                        "val_loss": round(v_loss, 5),
                        "val_preference_accuracy": round(v_acc, 4),
                        "val_eval_scope": eval_tag,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    }
                    with open(loss_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(val_log_entry) + "\n")

            # Periodic Checkpoint
            if global_step % save_steps == 0 or global_step == max_steps:
                ckpt_dir = output_dir / "checkpoints" / f"step_{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                if rank == 0:
                    # Save adapter
                    model_to_save = ddp_model.module if hasattr(ddp_model, "module") else ddp_model
                    model_to_save.save_pretrained(str(ckpt_dir / "adapter"))
                    if hasattr(asr_model, "processor") and asr_model.processor is not None:
                        asr_model.processor.save_pretrained(str(ckpt_dir / "adapter"))
                    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
                    if scheduler is not None:
                        torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
                    with open(ckpt_dir / "training_state.json", "w") as f:
                        json.dump({
                            "global_step": global_step,
                            "manifest_sha256": compute_file_sha256(manifest_path),
                            "world_size": world_size,
                            "target_map_hash": lora_meta["target_map_hash"],
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }, f, indent=2)

                if is_distributed:
                    dist.barrier()

                # Save rank-specific RNG
                rng_state = {
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                }
                if torch.cuda.is_available():
                    rng_state["cuda_rng"] = torch.cuda.get_rng_state()
                torch.save(rng_state, ckpt_dir / f"rng_state_rank_{rank}.pt")

                if is_distributed:
                    dist.barrier()

                if rank == 0:
                    print(f"[Rank 0] Saved checkpoint at step {global_step} -> {ckpt_dir}")

    if rank == 0:
        print(f"DPO Training completed successfully at global step {global_step}!")
        with open(output_dir / "pipeline_state.json", "w") as f:
            json.dump({
                "stage": "dpo_pilot",
                "status": "COMPLETED",
                "final_step": global_step,
                "world_size": world_size,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, f, indent=2)

    if is_distributed:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="DPO Training Runner for Qwen3-ASR")
    parser.add_argument("--manifest", type=Path, default=None, help="Path to input DPO manifest JSONL.")
    parser.add_argument("--val-manifest", type=Path, default=None, help="Path to held-out DPO validation manifest JSONL.")
    parser.add_argument("--config", type=Path, default=Path("configs/train/qwen3_asr_dpo.yaml"), help="Training config YAML.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory to save training artifacts.")
    parser.add_argument("--resume-from-checkpoint", type=Path, default=None, help="Path to checkpoint directory to resume.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override maximum training steps.")
    parser.add_argument("--save-steps", type=int, default=None, help="Override save steps.")
    parser.add_argument("--val-eval-samples", type=int, default=None, help="Number of held-out validation samples to evaluate at intermediate steps (default: None, evaluating full held-out dataset).")
    parser.add_argument("--single-gpu", action="store_true", help="Run in single GPU mode without DDP.")
    parser.add_argument("--allow-subset", action="store_true", help="Allow running on subset manifest.")

    # Export merged arguments
    parser.add_argument("--export-merged", action="store_true", help="Export merged standalone model.")
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Checkpoint to merge.")

    args = parser.parse_args()

    if args.export_merged:
        if not args.checkpoint_dir or not args.output_dir:
            parser.error("--export-merged requires --checkpoint-dir and --output-dir")
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        export_merged_model(args.checkpoint_dir, args.output_dir, config)
        return

    if not args.manifest or not args.output_dir:
        parser.error("Training requires --manifest and --output-dir")

    train_dpo(
        manifest_path=args.manifest,
        config_path=args.config,
        output_dir=args.output_dir,
        val_manifest_path=args.val_manifest,
        resume_from_checkpoint=args.resume_from_checkpoint,
        max_steps_override=args.max_steps,
        save_steps_override=args.save_steps,
        single_gpu=args.single_gpu,
        allow_subset=args.allow_subset,
        val_eval_samples=args.val_eval_samples,
    )


if __name__ == "__main__":
    main()
