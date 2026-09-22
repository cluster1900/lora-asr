#!/usr/bin/env python3
"""RL (GRPO: Group Relative Policy Optimization) Training Runner for Qwen3-ASR on V100 server.

Supports:
- Official qwen-asr / Transformers API
- Operation directly on model.model.thinker (Qwen3ASRThinkerForConditionalGeneration)
- Exact 199 Linear LoRA targets (3 Projection + 196 Decoder attention/mlp)
- FP16, eager attention, gradient checkpointing with input require grads
- Single-GPU and multi-GPU (DDP via torchrun)
- Group sampling with G=4 candidates per audio prompt (temperature=0.7, top_p=0.9)
- Sequence reward calculation per reward_config.yaml (ASR WER/CER + empty/repeat/too_long/hallucination penalties)
- Group advantage normalization with zero-variance protection (std <= epsilon -> advantage = 0)
- Policy gradient with token-level KL divergence regularization against frozen reference model
- Checkpointing (10+2 step contract) and full state resumption (optimizer, scheduler, RNG, training_state)
- Weight merging and export via merge_and_unload()
- Execution contract compliance: environment.json, resolved_config.yaml, manifest_sha256.json, pipeline_state.json, rollouts.jsonl, loss_log.jsonl
"""

from __future__ import annotations

import argparse
import collections
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    from peft import LoraConfig, PeftModel, get_peft_model, set_peft_model_state_dict
    HAVE_PEFT = True
except ImportError:
    peft = None  # type: ignore
    LoraConfig = None  # type: ignore
    PeftModel = None  # type: ignore
    get_peft_model = None  # type: ignore
    set_peft_model_state_dict = None  # type: ignore
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

# Import standard text normalization and metric functions
try:
    from evaluation.eval_wer import (
        edit_distance,
        has_repetition,
        normalize_text,
        tokenize,
    )
except ImportError:
    REPO_ROOT = Path(__file__).resolve().parents[1]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from evaluation.eval_wer import (
        edit_distance,
        has_repetition,
        normalize_text,
        tokenize,
    )


LORA_TARGET_REGEX = re.compile(
    r"^(audio_tower\.(conv_out|proj1|proj2)|model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)|mlp\.(gate_proj|up_proj|down_proj)))$"
)

LANGUAGE_MAP = {
    "en": "English",
    "english": "English",
    "zh": "Chinese",
    "chinese": "Chinese",
    "cantonese": "Cantonese",
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


class RLAudioDataset(Dataset):
    """Dataset for RL audio prompts and reference texts."""

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


def compute_sample_error_rate(reference: str, prediction: str, language: str) -> Tuple[float, int, int]:
    """Compute normalized WER for English or CER for Chinese.
    
    Returns:
        (error_rate, edits, ref_length)
    """
    lang_norm = str(language or "en").strip().lower()
    metric = "cer" if lang_norm in ("zh", "chinese") else "wer"
    norm_ref = normalize_text(reference)
    norm_pred = normalize_text(prediction)
    ref_tokens = tokenize(norm_ref, metric=metric)
    pred_tokens = tokenize(norm_pred, metric=metric)
    if not ref_tokens:
        edits = len(pred_tokens)
        return (0.0 if not pred_tokens else 1.0), edits, 0
    edits = edit_distance(ref_tokens, pred_tokens)
    error_rate = edits / len(ref_tokens)
    return error_rate, edits, len(ref_tokens)


def compute_sequence_reward(
    prediction: str,
    reference: str,
    language: str,
    reward_config: Optional[Dict[str, Any]] = None,
) -> Tuple[float, Dict[str, float], float]:
    """Compute sequence-level reward and diagnostic components.

    Follows configs/train/reward_config.yaml:
      asr reward = 1.0 - min(error_rate, 1.0)
      empty penalty = -0.25 (if empty transcription)
      repeat penalty = -0.25 (if repetition loop)
      too_long penalty = -0.15 (if hypothesis length > 1.5x reference length)
      hallucination penalty = -0.25 (if error_rate >= 0.8 and [too_long or repeated or major edits])
      final reward = clip(sum, -1.0, 1.0)

    Returns:
        (final_reward, reward_components, error_rate)
    """
    error_rate, edits, ref_len = compute_sample_error_rate(reference, prediction, language)
    asr_reward = 1.0 - min(error_rate, 1.0)

    norm_pred = normalize_text(prediction)
    empty_output = not norm_pred

    lang_norm = str(language or "en").strip().lower()
    metric = "cer" if lang_norm in ("zh", "chinese") else "wer"
    pred_tokens = tokenize(norm_pred, metric=metric)
    ref_tokens_count = max(1, ref_len)

    repeated = has_repetition(pred_tokens)
    length_ratio = len(pred_tokens) / ref_tokens_count
    too_long = length_ratio > 1.5

    hallucination_like = (
        not empty_output
        and error_rate >= 0.8
        and (too_long or repeated or edits >= max(3, ref_tokens_count // 2))
    )

    empty_pen = -0.25 if empty_output else 0.0
    repeat_pen = -0.25 if repeated else 0.0
    too_long_pen = -0.15 if too_long else 0.0
    hallucination_pen = -0.25 if hallucination_like else 0.0

    raw_sum = asr_reward + empty_pen + repeat_pen + too_long_pen + hallucination_pen
    final_reward = max(-1.0, min(1.0, raw_sum))

    components = {
        "asr": round(asr_reward, 4),
        "empty": round(empty_pen, 4),
        "repeat": round(repeat_pen, 4),
        "too_long": round(too_long_pen, 4),
        "hallucination": round(hallucination_pen, 4),
    }
    return final_reward, components, round(error_rate, 6)


def compute_group_advantages(
    rewards: Sequence[float],
    epsilon: float = 1e-6,
) -> Tuple[List[float], bool]:
    """Compute normalized GRPO advantages for a group of rollouts.

    A_i = (r_i - mean(r)) / (std(r) + epsilon).
    If std(r) <= epsilon (e.g. homogeneous group where all are correct or all fail),
    all advantages are set to 0.0.

    Returns:
        (advantages, is_zero_variance)
    """
    g = len(rewards)
    if g == 0:
        return [], True
    mean_r = sum(rewards) / g
    var_r = sum((r - mean_r) ** 2 for r in rewards) / g
    std_r = var_r ** 0.5

    if std_r <= epsilon:
        return [0.0] * g, True

    advs = [(r - mean_r) / (std_r + epsilon) for r in rewards]
    return advs, False


def audit_rollouts(
    rollouts_path: Path,
    expected_samples: Optional[int] = None,
    world_size: int = 1,
) -> Dict[str, Any]:
    """Audit rollouts.jsonl for schema conformity, row count, rank representation, and group integrity."""
    if not rollouts_path.is_file():
        raise FileNotFoundError(f"Rollout log not found: {rollouts_path}")

    required_fields = {
        "sample_id",
        "condition_group",
        "rank",
        "group_id",
        "group_size",
        "rollout_rank",
        "policy_checkpoint",
        "rollout_seed",
        "prediction",
        "language",
        "reference_error_rate",
        "reward_components",
        "reward",
        "kl_to_reference",
        "advantage",
    }

    groups: Dict[str, int] = collections.defaultdict(int)
    ranks_found: set[int] = set()
    total_rows = 0

    with open(rollouts_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line_s = line.strip()
            if not line_s:
                continue
            rec = json.loads(line_s)
            missing = required_fields - set(rec.keys())
            if missing:
                raise ValueError(
                    f"Line {line_no} in {rollouts_path} missing required fields: {missing}"
                )
            total_rows += 1
            gid = rec["group_id"]
            groups[gid] += 1
            ranks_found.add(int(rec.get("rank", 0)))

    # Check group sizes: each group must have exactly group_size candidates (4)
    for gid, cnt in groups.items():
        if cnt != 4:
            raise ValueError(f"Group {gid} in {rollouts_path} has {cnt} rollouts, expected 4.")

    if world_size > 1 and len(ranks_found) < world_size:
        print(
            f"Warning: Only {len(ranks_found)} ranks found in rollouts, expected {world_size}",
            file=sys.stderr,
        )

    if expected_samples is not None and total_rows < expected_samples:
        print(
            f"Warning: Total rollout rows {total_rows} is fewer than expected {expected_samples}",
            file=sys.stderr,
        )

    return {
        "total_rows": total_rows,
        "num_groups": len(groups),
        "ranks_represented": sorted(list(ranks_found)),
        "status": "PASSED",
    }


def merge_and_audit_rollouts(output_dir: Path, world_size: int = 1) -> Dict[str, Any]:
    """Merge per-rank rollouts_rank_{r}.jsonl into rollouts.jsonl and audit."""
    rollouts_path = output_dir / "rollouts.jsonl"
    total_lines = 0
    with open(rollouts_path, "w", encoding="utf-8") as out_f:
        for r in range(world_size):
            r_path = output_dir / f"rollouts_rank_{r}.jsonl"
            if r_path.is_file():
                with open(r_path, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        out_f.write(line)
                        total_lines += 1

    return audit_rollouts(rollouts_path, world_size=world_size)



def compute_token_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token autoregressive log-probabilities masked on target tokens.

    Args:
        logits: [batch_size, seq_len, vocab_size]
        labels: [batch_size, seq_len]

    Returns:
        per_token_logps: [batch_size, seq_len - 1] float tensor
        mask: [batch_size, seq_len - 1] bool tensor
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    mask = (shift_labels != ignore_index)
    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0
    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token = torch.gather(
        log_probs, dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return per_token, mask


def setup_lora_rl(thinker: Any, config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
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

    print(f"Loading base model {model_id} (revision={revision}) for merge...")
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
    print(f"Saving merged RL model to {output_dir}...")

    # Fix generation_config temperature validation in newer transformers when do_sample is False
    for obj in [asr_model.model, asr_model.model.thinker]:
        if hasattr(obj, "generation_config") and obj.generation_config is not None:
            if not getattr(obj.generation_config, "do_sample", False):
                obj.generation_config.temperature = None

    asr_model.model.save_pretrained(str(output_dir))
    if hasattr(asr_model, "processor") and hasattr(asr_model.processor, "save_pretrained"):
        asr_model.processor.save_pretrained(str(output_dir))
    print("Merged RL model export complete!")


def evaluate_rl_validation(
    thinker_model: Any,
    asr_model: Any,
    val_dataset: RLAudioDataset,
    device: str,
    group_size: int = 4,
    temperature: float = 0.85,
    top_p: float = 0.92,
    top_k: int = 50,
    max_eval_samples: Optional[int] = None,
) -> Tuple[float, float]:
    """Evaluate mean reward on held-out validation set."""
    if not HAVE_TORCH or len(val_dataset) == 0:
        return 0.0, 0.0

    eval_model = thinker_model.module if hasattr(thinker_model, "module") else thinker_model
    eval_model.eval()

    total_reward = 0.0
    total_rollouts = 0
    total_error_rate = 0.0

    indices = list(range(min(len(val_dataset), max_eval_samples or len(val_dataset))))

    with torch.no_grad():
        for idx in indices:
            sample = val_dataset[idx]
            audio_path = sample.get("audio", "")
            if not os.path.isfile(audio_path):
                continue

            try:
                wav, sr = sf.read(audio_path)
            except Exception:
                continue

            if len(wav.shape) > 1:
                wav = wav[:, 0]

            lang_raw = str(sample.get("language", "en")).strip().lower()
            lang_full = LANGUAGE_MAP.get(lang_raw, "English")
            gold_text = str(sample.get("text") or sample.get("answer") or "")

            prompt = asr_model._build_text_prompt(context="", force_language=lang_full)
            inputs = asr_model.processor(text=prompt, audio=wav, return_tensors="pt")
            inputs = inputs.to(device)
            if device.startswith("cuda"):
                inputs["input_features"] = inputs["input_features"].to(torch.float16)

            prompt_len = inputs["input_ids"].shape[1]

            try:
                # Use model generate to sample G candidates with diversity
                gen_out = asr_model.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=group_size,
                    max_new_tokens=128,
                )
                seqs = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
            except Exception:
                continue

            for g_idx in range(min(group_size, seqs.shape[0])):
                cand_ids = seqs[g_idx, prompt_len:]
                pred_text = asr_model.processor.tokenizer.decode(cand_ids, skip_special_tokens=True).strip()
                rew, _, err = compute_sequence_reward(pred_text, gold_text, lang_raw)
                total_reward += rew
                total_error_rate += err
                total_rollouts += 1

    mean_reward = total_reward / max(1, total_rollouts)
    mean_err = total_error_rate / max(1, total_rollouts)
    return mean_reward, mean_err


def train_rl(
    manifest_path: Path,
    config_path: Path,
    output_dir: Path,
    val_manifest_path: Optional[Path] = None,
    resume_from_checkpoint: Optional[Path] = None,
    max_steps_override: Optional[int] = None,
    save_steps_override: Optional[int] = None,
    eval_steps_override: Optional[int] = None,
    temperature_override: Optional[float] = None,
    top_p_override: Optional[float] = None,
    top_k_override: Optional[int] = None,
    zero_var_thresh_override: Optional[float] = None,
    single_gpu: bool = False,
    allow_subset: bool = False,
    val_eval_samples: Optional[int] = None,
) -> None:
    """Main entrypoint for single-GPU or 4-card DDP GRPO training."""
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

    runtime_cfg = config.get("runtime", {})
    train_cfg = config.get("train", {})
    model_cfg = config.get("model", {})
    grpo_cfg = config.get("grpo", {})

    seed = int(runtime_cfg.get("seed", 20260722))
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)

    max_steps = max_steps_override or int(train_cfg.get("max_steps", 60))
    save_steps = save_steps_override or int(train_cfg.get("save_steps", 10))
    eval_steps = eval_steps_override or int(train_cfg.get("eval_steps", 10))
    val_eval_samples = val_eval_samples if val_eval_samples is not None else train_cfg.get("val_eval_samples", 50)
    lr = float(train_cfg.get("learning_rate", 2.0e-6))
    warmup_steps = int(train_cfg.get("warmup_steps", 10))
    grad_accum_steps = int(runtime_cfg.get("gradient_accumulation_steps", 16))

    group_size = int(grpo_cfg.get("group_size", 4))
    sampling_cfg = grpo_cfg.get("sampling", {})
    temperature = (
        temperature_override
        if temperature_override is not None
        else float(sampling_cfg.get("temperature", 0.85))
    )
    top_p = (
        top_p_override
        if top_p_override is not None
        else float(sampling_cfg.get("top_p", 0.92))
    )
    top_k = (
        top_k_override
        if top_k_override is not None
        else int(sampling_cfg.get("top_k", 50))
    )
    adv_cfg = grpo_cfg.get("advantage", {})
    epsilon = float(adv_cfg.get("epsilon", 1.0e-6))
    zero_var_thresh = (
        zero_var_thresh_override
        if zero_var_thresh_override is not None
        else float(adv_cfg.get("zero_variance_threshold", 0.80))
    )
    kl_cfg = grpo_cfg.get("kl_regularization", {})
    beta = float(kl_cfg.get("beta", 0.04))

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

    # Load datasets
    train_dataset = RLAudioDataset(manifest_path)
    val_dataset = RLAudioDataset(val_manifest_path) if val_manifest_path else None

    # Load base model
    model_id = str(model_cfg.get("model_id", "/data/mega-asr/runs/dpo_pilot_v2/merged_base"))
    revision = model_cfg.get("model_revision", "7278e1e70fe206f11671096ffdd38061171dd6e5")

    if rank == 0:
        print(f"Loading initial policy / reference model from {model_id} on rank {rank}...")

    asr_model = Qwen3ASRModel.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        attn_implementation="eager",
    )
    thinker = asr_model.model.thinker
    thinker.to(device)

    # Enable gradient checkpointing and input require grads on policy thinker
    if runtime_cfg.get("gradient_checkpointing", True):
        if hasattr(thinker, "enable_input_require_grads"):
            thinker.enable_input_require_grads()
        if hasattr(thinker, "gradient_checkpointing_enable"):
            thinker.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    lora_model, lora_meta = setup_lora_rl(thinker, config)

    if resume_from_checkpoint and (resume_from_checkpoint / "adapter").is_dir():
        adapter_path = resume_from_checkpoint / "adapter"
        if rank == 0:
            print(f"Resuming LoRA adapter weights from {adapter_path}...")
        adapter_safetensors = adapter_path / "adapter_model.safetensors"
        adapter_bin = adapter_path / "adapter_model.bin"
        if adapter_safetensors.is_file():
            try:
                from safetensors.torch import load_file
                adapters_weights = load_file(str(adapter_safetensors))
            except ImportError:
                adapters_weights = torch.load(adapter_safetensors, map_location="cpu")
            if set_peft_model_state_dict is not None:
                set_peft_model_state_dict(lora_model, adapters_weights)
            else:
                lora_model.load_state_dict(adapters_weights, strict=False)
        elif adapter_bin.is_file():
            adapters_weights = torch.load(adapter_bin, map_location="cpu")
            if set_peft_model_state_dict is not None:
                set_peft_model_state_dict(lora_model, adapters_weights)
            else:
                lora_model.load_state_dict(adapters_weights, strict=False)

    # Also make sure asr_model.model.thinker references lora_model for generation
    asr_model.model.thinker = lora_model

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
            optimizer.load_state_dict(torch.load(opt_file, map_location=device, weights_only=False))

        sched_file = resume_from_checkpoint / "scheduler.pt"
        if sched_file.is_file() and scheduler is not None:
            scheduler.load_state_dict(torch.load(sched_file, map_location=device, weights_only=False))

        rng_file = resume_from_checkpoint / f"rng_state_rank_{rank}.pt"
        if rng_file.is_file():
            rng_state = torch.load(rng_file, weights_only=False)
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
    rollouts_log_path = output_dir / "rollouts.jsonl"
    global_step = start_step

    sample_cursor = global_step * grad_accum_steps * world_size
    n_train = len(train_dataset)

    optimizer.zero_grad()
    accum_count = 0
    accum_policy_loss = 0.0
    accum_kl_loss = 0.0
    accum_total_loss = 0.0
    accum_rewards: List[float] = []
    accum_zero_vars: int = 0
    consecutive_high_zero_var: int = 0
    step_start_time = time.time()

    if rank == 0:
        print(f"Starting GRPO training: global_step={global_step} -> {max_steps}, accum={grad_accum_steps}, world_size={world_size}, G={group_size}")

    eval_underlying_model = ddp_model.module if hasattr(ddp_model, "module") else ddp_model

    # Evaluate Step 0 Reference baseline on full held-out validation set
    has_step0 = False
    if loss_log_path.is_file():
        try:
            with open(loss_log_path, "r", encoding="utf-8") as f_chk:
                for line in f_chk:
                    rec = json.loads(line)
                    if rec.get("global_step") == 0 and rec.get("val_eval_scope") == "Full Held-out":
                        has_step0 = True
                        if rank == 0:
                            print(f"Step 0 Reference baseline already evaluated in loss_log.jsonl (reward={rec.get('val_mean_reward')}). Skipping re-evaluation.")
                        break
        except Exception:
            has_step0 = False

    if val_dataset and start_step == 0 and not has_step0:
        if rank == 0:
            print(f"Evaluating Step 0 Policy / Reference baseline on full held-out validation set ({len(val_dataset)} samples)...")
        val_rew0, val_err0 = evaluate_rl_validation(
            ddp_model,
            asr_model,
            val_dataset,
            device,
            group_size=group_size,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        if rank == 0:
            print(f"--- [Step 0 Reference Baseline (Full Held-out)] val_mean_reward={val_rew0:.4f}, val_error_rate={val_err0:.4f} ---")
            val_log_entry = {
                "global_step": 0,
                "val_mean_reward": round(val_rew0, 4),
                "val_error_rate": round(val_err0, 4),
                "val_eval_scope": "Full Held-out",
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            with open(loss_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(val_log_entry) + "\n")
        if is_distributed:
            dist.barrier()

    while global_step < max_steps:
        sample_idx = (sample_cursor + rank) % max(1, n_train)
        sample = train_dataset[sample_idx]
        sample_cursor += world_size

        audio_path = sample.get("audio", "")
        if not os.path.isfile(audio_path):
            continue

        try:
            wav, sr = sf.read(audio_path)
        except Exception as err:
            print(f"Warning: Failed to read {audio_path}: {err}", file=sys.stderr)
            continue

        if len(wav.shape) > 1:
            wav = wav[:, 0]

        lang_raw = str(sample.get("language", "en")).strip().lower()
        lang_full = LANGUAGE_MAP.get(lang_raw, "English")
        gold_text = str(sample.get("text") or sample.get("answer") or "")
        sample_id = str(sample.get("sample_id", f"sample_{sample_idx}"))

        prompt = asr_model._build_text_prompt(context="", force_language=lang_full)
        inputs_gen = asr_model.processor(text=prompt, audio=wav, return_tensors="pt")
        inputs_gen = inputs_gen.to(device)
        if device.startswith("cuda"):
            inputs_gen["input_features"] = inputs_gen["input_features"].to(torch.float16)

        prompt_len = inputs_gen["input_ids"].shape[1]

        # 1. Rollout Generation under current Policy (no_grad)
        eval_underlying_model.eval()
        cand_seed = (seed + rank * 100000 + global_step * 1000 + accum_count * 10) % (2**31 - 1)
        torch.manual_seed(cand_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cand_seed)

        with torch.no_grad():
            try:
                gen_out = asr_model.model.generate(
                    **inputs_gen,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=group_size,
                    max_new_tokens=128,
                )
                seqs = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
            except Exception as gen_err:
                print(f"Generation error on {sample_id}: {gen_err}", file=sys.stderr)
                continue

        cand_texts: List[str] = []
        for g_i in range(min(group_size, seqs.shape[0])):
            cand_tokens = seqs[g_i, prompt_len:]
            cand_str = asr_model.processor.tokenizer.decode(cand_tokens, skip_special_tokens=True).strip()
            cand_texts.append(cand_str)

        # Pad candidates if fewer than group_size
        while len(cand_texts) < group_size:
            cand_texts.append(cand_texts[-1] if cand_texts else "")

        # 2. Sequence Rewards and GRPO Advantage
        rewards: List[float] = []
        reward_comps_list: List[Dict[str, float]] = []
        err_rates: List[float] = []
        for cand_t in cand_texts:
            rew, comps, err = compute_sequence_reward(cand_t, gold_text, lang_raw)
            rewards.append(rew)
            reward_comps_list.append(comps)
            err_rates.append(err)

        advantages, is_zero_var = compute_group_advantages(rewards, epsilon=epsilon)
        accum_rewards.extend(rewards)
        if is_zero_var:
            accum_zero_vars += 1

        # 3. Build batched tensors for all G candidates
        # Target tokens include trailing <|im_end|>
        batch_inputs_list = []
        batch_labels_list = []
        pad_token_id = asr_model.processor.tokenizer.pad_token_id or 0

        for cand_t in cand_texts:
            target_str = cand_t + "<|im_end|>"
            full_text = prompt + target_str
            proc_out = asr_model.processor(text=full_text, audio=wav, return_tensors="pt")
            target_ids = asr_model.processor.tokenizer.encode(target_str, add_special_tokens=False)

            in_ids = proc_out["input_ids"]
            attn_m = proc_out["attention_mask"]
            lab = in_ids.clone()
            tlen = len(target_ids)
            if tlen >= in_ids.shape[1]:
                lab[:, 0] = -100
            else:
                lab[:, :-tlen] = -100

            batch_inputs_list.append((in_ids, attn_m, lab))

        max_in_len = max(item[0].shape[1] for item in batch_inputs_list)
        padded_ids = []
        padded_masks = []
        padded_labels = []

        for in_ids, attn_m, lab in batch_inputs_list:
            pad_len = max_in_len - in_ids.shape[1]
            if pad_len > 0:
                p_ids = F.pad(in_ids, (0, pad_len), value=pad_token_id)
                p_mask = F.pad(attn_m, (0, pad_len), value=0)
                p_lab = F.pad(lab, (0, pad_len), value=-100)
            else:
                p_ids = in_ids
                p_mask = attn_m
                p_lab = lab
            padded_ids.append(p_ids)
            padded_masks.append(p_mask)
            padded_labels.append(p_lab)

        batch_input_ids = torch.cat(padded_ids, dim=0).to(device)
        batch_attention_mask = torch.cat(padded_masks, dim=0).to(device)
        batch_labels = torch.cat(padded_labels, dim=0).to(device)

        feat = inputs_gen["input_features"]
        feat_mask = inputs_gen["feature_attention_mask"]
        batch_input_features = feat.repeat(group_size, 1, 1)
        batch_feature_attention_mask = feat_mask.repeat(group_size, 1)

        # 4. Compute Reference Logprobs under Frozen Reference Base (adapter disabled)
        with torch.no_grad():
            eval_underlying_model.eval()
            with eval_underlying_model.disable_adapter():
                ref_outputs = eval_underlying_model(
                    input_ids=batch_input_ids,
                    attention_mask=batch_attention_mask,
                    input_features=batch_input_features,
                    feature_attention_mask=batch_feature_attention_mask,
                )
                ref_token_logps, token_mask = compute_token_logps(ref_outputs.logits, batch_labels)

        # 5. Forward Pass under Policy Model with Gradients
        eval_underlying_model.train()
        policy_outputs = ddp_model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            input_features=batch_input_features,
            feature_attention_mask=batch_feature_attention_mask,
        )
        policy_token_logps, _ = compute_token_logps(policy_outputs.logits, batch_labels)

        # 6. Policy Gradient Loss + Reference KL Regularization
        # token_kl = policy_logp - ref_logp
        token_kl = (policy_token_logps - ref_token_logps) * token_mask
        seq_kls = token_kl.sum(dim=-1) / token_mask.sum(dim=-1).clamp(min=1)

        adv_tensor = torch.tensor(advantages, device=device, dtype=policy_token_logps.dtype)

        # Policy gradient: - A_i * log pi_\theta(t)
        # KL term: + beta * (log pi_\theta(t) - log pi_ref(t))
        token_loss = - (adv_tensor.unsqueeze(-1) * policy_token_logps - beta * token_kl) * token_mask
        seq_losses = token_loss.sum(dim=-1) / token_mask.sum(dim=-1).clamp(min=1)

        policy_term_loss = - (adv_tensor.unsqueeze(-1) * policy_token_logps * token_mask).sum(dim=-1) / token_mask.sum(dim=-1).clamp(min=1)
        kl_term_loss = (beta * token_kl).sum(dim=-1) / token_mask.sum(dim=-1).clamp(min=1)

        group_loss = seq_losses.mean()
        scaled_loss = group_loss / grad_accum_steps
        scaled_loss.backward()

        accum_total_loss += group_loss.item()
        accum_policy_loss += policy_term_loss.mean().item()
        accum_kl_loss += kl_term_loss.mean().item()
        accum_count += 1

        # 7. Record Rollouts to rank-specific rollouts_rank_{rank}.jsonl
        cond_group = sample.get("condition_group", "clean" if "clean" in sample.get("scenario", "") else "degraded")
        rollout_records = []
        for g_i in range(group_size):
            rec = {
                "sample_id": sample_id,
                "condition_group": cond_group,
                "rank": rank,
                "group_id": f"{sample_id}:{global_step}:{rank}:{accum_count}",
                "group_size": group_size,
                "rollout_rank": g_i,
                "policy_checkpoint": f"step_{global_step}",
                "rollout_seed": cand_seed + g_i,
                "prediction": cand_texts[g_i],
                "language": lang_raw,
                "reference_error_rate": err_rates[g_i],
                "reward_components": reward_comps_list[g_i],
                "reward": rewards[g_i],
                "kl_to_reference": round(seq_kls[g_i].item(), 6),
                "advantage": round(advantages[g_i], 6),
                "error": "",
            }
            rollout_records.append(rec)
        rank_rollouts_path = output_dir / f"rollouts_rank_{rank}.jsonl"
        with open(rank_rollouts_path, "a", encoding="utf-8") as rf:
            for rec in rollout_records:
                rf.write(json.dumps(rec, ensure_ascii=False) + "\n")

        # 8. Gradient Step
        if accum_count == grad_accum_steps:
            torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            step_duration = time.time() - step_start_time
            step_start_time = time.time()

            mean_tot_loss = accum_total_loss / grad_accum_steps
            mean_pol_loss = accum_policy_loss / grad_accum_steps
            mean_kl = accum_kl_loss / grad_accum_steps
            mean_rew = sum(accum_rewards) / max(1, len(accum_rewards))
            zero_var_ratio = accum_zero_vars / grad_accum_steps

            accum_total_loss = 0.0
            accum_policy_loss = 0.0
            accum_kl_loss = 0.0
            accum_rewards = []
            accum_zero_vars = 0
            accum_count = 0

            # Gather metrics across ranks if distributed
            if is_distributed:
                metrics_t = torch.tensor([mean_tot_loss, mean_pol_loss, mean_kl, mean_rew, zero_var_ratio], device=device)
                dist.all_reduce(metrics_t, op=dist.ReduceOp.SUM)
                mean_tot_loss = (metrics_t[0] / world_size).item()
                mean_pol_loss = (metrics_t[1] / world_size).item()
                mean_kl = (metrics_t[2] / world_size).item()
                mean_rew = (metrics_t[3] / world_size).item()
                zero_var_ratio = (metrics_t[4] / world_size).item()

            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else lr

            if rank == 0:
                log_entry = {
                    "global_step": global_step,
                    "loss": round(mean_tot_loss, 5),
                    "policy_loss": round(mean_pol_loss, 5),
                    "kl_loss": round(mean_kl, 5),
                    "mean_reward": round(mean_rew, 4),
                    "zero_variance_ratio": round(zero_var_ratio, 4),
                    "learning_rate": current_lr,
                    "step_seconds": round(step_duration, 3),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                print(
                    f"[Step {global_step}/{max_steps}] loss={mean_tot_loss:.4f}, pol_loss={mean_pol_loss:.4f}, "
                    f"kl={mean_kl:.4f}, rew={mean_rew:.3f}, zero_var={zero_var_ratio:.2f}, lr={current_lr:.2e}, time={step_duration:.2f}s"
                )
                with open(loss_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")

                if zero_var_ratio > zero_var_thresh:
                    consecutive_high_zero_var += 1
                    msg = (
                        f"WARNING: batch zero-variance group ratio {zero_var_ratio:.2%} exceeded "
                        f"contract threshold {zero_var_thresh:.2%} ({consecutive_high_zero_var}/2 consecutive steps)."
                    )
                    print(msg, file=sys.stderr)
                    if consecutive_high_zero_var >= 2:
                        fatal_msg = (
                            f"FATAL: zero-variance group ratio {zero_var_ratio:.2%} exceeded "
                            f"contract threshold {zero_var_thresh:.2%} for {consecutive_high_zero_var} consecutive steps at step {global_step}."
                        )
                        print(fatal_msg, file=sys.stderr)
                        pipeline_state = {
                            "stage": "rl_pilot",
                            "global_step": global_step,
                            "world_size": world_size,
                            "status": "FAILED_ZERO_VARIANCE",
                            "zero_variance_ratio": round(zero_var_ratio, 4),
                            "threshold": zero_var_thresh,
                            "consecutive_steps": consecutive_high_zero_var,
                            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        }
                        with open(output_dir / "pipeline_state.json", "w", encoding="utf-8") as pf:
                            json.dump(pipeline_state, pf, indent=2)
                        raise RuntimeError(
                            f"Batch zero-variance group ratio {zero_var_ratio:.2%} exceeded "
                            f"contract threshold {zero_var_thresh:.2%} for 2 consecutive steps. "
                            "Sampling distribution collapsed. Halting RL training."
                        )
                else:
                    consecutive_high_zero_var = 0

            # Periodic validation on full held-out pool
            if val_dataset and (global_step % eval_steps == 0 or global_step == max_steps):
                val_rew, val_err = evaluate_rl_validation(
                    ddp_model,
                    asr_model,
                    val_dataset,
                    device,
                    group_size=group_size,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                if rank == 0:
                    print(f"--- [Validation @ Step {global_step} (Full Held-out)] val_mean_reward={val_rew:.4f}, val_error_rate={val_err:.4f} ---")
                    val_log_entry = {
                        "global_step": global_step,
                        "val_mean_reward": round(val_rew, 4),
                        "val_error_rate": round(val_err, 4),
                        "val_eval_scope": "Full Held-out",
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    }
                    with open(loss_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(val_log_entry) + "\n")
                if is_distributed:
                    dist.barrier()

            # Periodic Checkpoint
            if global_step % save_steps == 0 or global_step == max_steps:
                ckpt_dir = output_dir / "checkpoints" / f"step_{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                if rank == 0:
                    # Save adapter weights
                    adapter_dir = ckpt_dir / "adapter"
                    eval_underlying_model.save_pretrained(str(adapter_dir))
                    # Save optimizer & scheduler
                    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
                    if scheduler is not None:
                        torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
                    # Save state metadata
                    st_meta = {
                        "global_step": global_step,
                        "world_size": world_size,
                        "stage": "rl_pilot",
                        "model_id": model_id,
                        "model_revision": revision,
                        "target_map_hash": lora_meta["target_map_hash"],
                        "manifest": str(manifest_path),
                        "manifest_sha256": compute_file_sha256(manifest_path),
                        "scheduler": "linear" if scheduler is not None else None,
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    }
                    with open(ckpt_dir / "training_state.json", "w", encoding="utf-8") as sf_f:
                        json.dump(st_meta, sf_f, indent=2)

                    # Update pipeline_state.json
                    pipeline_state = {
                        "stage": "rl_pilot",
                        "global_step": global_step,
                        "world_size": world_size,
                        "last_valid_checkpoint": str(ckpt_dir),
                        "status": "COMPLETED" if global_step >= max_steps else "RUNNING",
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    }
                    with open(output_dir / "pipeline_state.json", "w", encoding="utf-8") as pf:
                        json.dump(pipeline_state, pf, indent=2)

                if is_distributed:
                    dist.barrier()

                # Save rank-specific RNG state
                rng_dict = {
                    "python_rng": random.getstate(),
                    "numpy_rng": np.random.get_state(),
                    "torch_rng": torch.get_rng_state(),
                }
                if torch.cuda.is_available():
                    rng_dict["cuda_rng"] = torch.cuda.get_rng_state()
                torch.save(rng_dict, ckpt_dir / f"rng_state_rank_{rank}.pt")

                if is_distributed:
                    dist.barrier()
                if rank == 0:
                    print(f"Checkpoint saved to {ckpt_dir}")
                    try:
                        merge_and_audit_rollouts(output_dir, world_size=world_size)
                    except Exception as merr:
                        print(f"Warning: Rollouts merge on checkpoint: {merr}", file=sys.stderr)

    if is_distributed:
        dist.barrier()

    if rank == 0:
        try:
            audit_res = merge_and_audit_rollouts(output_dir, world_size=world_size)
            print(f"All rollouts merged and audited: {audit_res['total_rows']} rows across ranks {audit_res['ranks_represented']}")
        except Exception as merr:
            print(f"Warning: Final rollouts merge/audit failed: {merr}", file=sys.stderr)

    if is_distributed:
        dist.destroy_process_group()

    if rank == 0:
        print(f"GRPO RL Training successfully finished {global_step} steps!")


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=None, help="Path to input RL manifest (JSONL).")
    parser.add_argument("--val-manifest", default=None, help="Path to validation manifest (JSONL).")
    parser.add_argument("--config", default=None, help="Path to training config YAML.")
    parser.add_argument("--output-dir", default=None, help="Directory to save checkpoints and logs.")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Path to step checkpoint directory to resume.")
    parser.add_argument("--max-steps", type=int, default=None, help="Override maximum training steps.")
    parser.add_argument("--save-steps", type=int, default=None, help="Override checkpoint saving frequency.")
    parser.add_argument("--eval-steps", type=int, default=None, help="Override validation evaluation frequency.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature for rollouts (default: 0.85).")
    parser.add_argument("--top-p", type=float, default=None, help="Sampling top-p for rollouts (default: 0.92).")
    parser.add_argument("--top-k", type=int, default=None, help="Sampling top-k for rollouts (default: 50).")
    parser.add_argument("--zero-variance-thresh", type=float, default=None, help="Batch zero-variance ratio threshold (default: 0.80).")
    parser.add_argument("--single-gpu", action="store_true", help="Run on single GPU without DDP.")
    parser.add_argument("--allow-subset", action="store_true", help="Allow NON_STRICT_SUBSET dataset gate status.")
    parser.add_argument("--val-eval-samples", type=int, default=None, help="Number of validation samples to evaluate.")
    parser.add_argument("--export-merged", action="store_true", help="Export standalone merged model.")
    parser.add_argument("--checkpoint-dir", default=None, help="Checkpoint directory for --export-merged.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.export_merged:
        if not args.checkpoint_dir or not args.output_dir:
            parser.error("--export-merged requires --checkpoint-dir and --output-dir.")
        ckpt_path = Path(args.checkpoint_dir).resolve()
        out_path = Path(args.output_dir).resolve()

        cfg_file = ckpt_path / "training_state.json"
        config: Dict[str, Any] = {}
        if cfg_file.is_file():
            with open(cfg_file, "r") as f:
                st = json.load(f)
                config["model"] = {
                    "model_id": st.get("model_id", "Qwen/Qwen3-ASR-1.7B"),
                    "model_revision": st.get("model_revision", "7278e1e70fe206f11671096ffdd38061171dd6e5"),
                }
        elif args.config:
            with open(args.config, "r") as f:
                config = yaml.safe_load(f)

        export_merged_model(ckpt_path, out_path, config)
        return

    if not args.manifest or not args.config or not args.output_dir:
        parser.error("Training requires --manifest, --config, and --output-dir.")

    train_rl(
        manifest_path=Path(args.manifest).resolve(),
        config_path=Path(args.config).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        val_manifest_path=Path(args.val_manifest).resolve() if args.val_manifest else None,
        resume_from_checkpoint=Path(args.resume_from_checkpoint).resolve() if args.resume_from_checkpoint else None,
        max_steps_override=args.max_steps,
        save_steps_override=args.save_steps,
        eval_steps_override=args.eval_steps,
        temperature_override=args.temperature,
        top_p_override=args.top_p,
        top_k_override=args.top_k,
        zero_var_thresh_override=args.zero_variance_thresh,
        single_gpu=args.single_gpu,
        allow_subset=args.allow_subset,
        val_eval_samples=args.val_eval_samples,
    )


if __name__ == "__main__":
    main()
