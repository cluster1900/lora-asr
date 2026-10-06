#!/usr/bin/env python3
"""RL (GRPO: Group Relative Policy Optimization) Training Runner for Qwen3-ASR on V100 server.

Supports:
- Official qwen-asr / Transformers API
- Operation directly on model.model.thinker (Qwen3ASRThinkerForConditionalGeneration)
- Exact 199 Linear LoRA targets (3 Projection + 196 Decoder attention/mlp)
- FP16, eager attention, gradient checkpointing with input require grads
- Single-GPU and multi-GPU (DDP via torchrun)
- Degraded prompts, greedy anchor, and a length-2 backward batch (1 greedy + 1 winner or a zero-advantage copy)
- Optional second-round samples when the first round does not beat greedy by min_improvement
- Sequence reward calculation per reward_config.yaml (ASR WER/CER + empty/repeat/too_long/hallucination penalties)
- Sequence-level policy gradient (sum over response tokens) with Schulman K3 KL against the frozen reference
- Chunked resume: YAML max_steps is the horizon; CLI --max-steps is only this process's end.
  Resume drops loss and rollout rows past the last complete checkpoint and refuses
  a manifest, world-size, or scheduler mismatch.
- Weight merging and export via merge_and_unload()
- Execution contract compliance: environment.json, resolved_config.yaml, manifest_sha256.json, pipeline_state.json, rollouts.jsonl, loss_log.jsonl
"""

from __future__ import annotations

import argparse
import collections
from collections import Counter, defaultdict
import datetime
import fcntl
import hashlib
import json
import math
import os
import platform
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Ensure HF mirror, large cache directory on /data, and offline fallback
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HOME", "/data/mega-asr/cache/huggingface")
if not os.environ.get("HF_TOKEN"):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import yaml

_TRAIN_DIR = Path(__file__).resolve().parent
if str(_TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_DIR))
from rl_pair_advantage import winner_advantage
from rl_lora_targets import filter_lora_targets
from rl_resume_lr import apply_configured_learning_rate
from rl_local_winner import local_winner_index
from rl_policy_mask import changed_token_mask, policy_keep_ratio, scatter_keep, signed_edit_loss_args
from rl_sample_strategy import (
    degraded_balanced_indices,
    degraded_balanced_summary,
    degraded_skip_regressed_indices,
)
from rl_reference_candidate import append_reference_candidate

SAMPLE_STRATEGIES = frozenset(
    {"balanced", "standard", "degraded", "degraded_skip_regressed", "degraded_balanced"}
)

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


REPO_ROOT = Path(__file__).resolve().parents[1]

from inference.decoding import (
    check_resume_decoding, configure_greedy_model, decoding_contract,
    training_token_budget,
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


def get_git_commit(repo_dir: Optional[Path] = None) -> str:
    """Retrieve git commit SHA from env, .git_commit file, or git CLI."""
    if "GIT_COMMIT" in os.environ and os.environ["GIT_COMMIT"].strip():
        return os.environ["GIT_COMMIT"].strip()
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
    info["git_commit"] = get_git_commit()
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


def resolve_sample_strategy(cli_value: Optional[str], yaml_value: Optional[str]) -> str:
    """Pick a sample strategy: non-empty CLI, else non-empty YAML, else balanced.

    Whitespace-only values are empty. An omitted CLI value must not override YAML.
    """
    for value in (cli_value, yaml_value):
        if value is None:
            continue
        text = str(value).strip().lower()
        if text:
            if text not in SAMPLE_STRATEGIES:
                expected = ", ".join(sorted(SAMPLE_STRATEGIES))
                raise ValueError(f"unsupported sample strategy {text!r}; expected one of: {expected}")
            return text
    return "balanced"


def _clears_min_improvement(gap: float, min_improvement: float) -> bool:
    """True if ``gap`` reaches the threshold, including a decimal 0.02 one ulp under."""
    threshold = float(min_improvement)
    return gap >= threshold or math.isclose(gap, threshold, rel_tol=0.0, abs_tol=1e-9)


def build_epoch_sample_indices(
    dataset: RLAudioDataset,
    strategy: str = "balanced",
    epoch: int = 0,
    seed: int = 20260722,
    strategy_options: Optional[Dict[str, Any]] = None,
) -> List[int]:
    """Build deterministic sample index list for one epoch under the specified strategy."""
    total_samples = len(dataset)
    strategy = str(strategy or "balanced").strip().lower()
    if strategy not in SAMPLE_STRATEGIES:
        expected = ", ".join(sorted(SAMPLE_STRATEGIES))
        raise ValueError(f"unsupported sample strategy {strategy!r}; expected one of: {expected}")
    if total_samples == 0:
        if strategy == "degraded":
            raise ValueError("sample strategy 'degraded' requires at least one degraded row")
        return []

    if strategy == "standard":
        rng = random.Random(seed + epoch)
        indices = list(range(total_samples))
        rng.shuffle(indices)
        return indices

    degraded_indices = [
        i for i, s in enumerate(dataset.samples)
        if s.get("condition_group") == "degraded" or ("clean" not in str(s.get("scenario", "")).lower())
    ]
    en_clean_indices = [
        i for i, s in enumerate(dataset.samples)
        if (s.get("condition_group") == "clean" or "clean" in str(s.get("scenario", "")).lower())
        and str(s.get("language", "en")).strip().lower() in ("en", "english")
    ]
    zh_clean_indices = [
        i for i, s in enumerate(dataset.samples)
        if (s.get("condition_group") == "clean" or "clean" in str(s.get("scenario", "")).lower())
        and str(s.get("language", "en")).strip().lower() in ("zh", "chinese")
    ]

    rng = random.Random(seed + epoch)

    if strategy == "degraded":
        # Only this branch is new. balanced below stays the v9 path.
        if not degraded_indices:
            raise ValueError("sample strategy 'degraded' requires at least one degraded row")
        indices = list(degraded_indices)
        rng.shuffle(indices)
        return indices

    if strategy == "degraded_skip_regressed":
        indices = degraded_skip_regressed_indices(dataset.samples)
        rng.shuffle(indices)
        return indices

    if strategy == "degraded_balanced":
        options = dict(strategy_options or {})
        virtual_epoch_rows = int(options.pop("virtual_epoch_rows", 2000))
        max_repeat = int(options.pop("max_repeat", 4))
        scenario_weights = options.pop("scenario_weights", None)
        if options:
            raise ValueError(
                f"unknown degraded_balanced options: {sorted(options)}"
            )
        return degraded_balanced_indices(
            dataset.samples,
            epoch=epoch,
            seed=seed,
            virtual_epoch_rows=virtual_epoch_rows,
            max_repeat=max_repeat,
            scenario_weights=scenario_weights,
        )

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

    # Fallback to random shuffle
    indices = list(range(total_samples))
    rng.shuffle(indices)
    return indices


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


def max_token_run(tokens: Sequence[str]) -> int:
    """Return the maximum length of consecutive identical tokens."""
    if not tokens:
        return 0
    max_run = 1
    current_run = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 1
    return max_run


def max_token_pair_count(tokens: Sequence[str]) -> int:
    """Return the maximum count of any adjacent 2-token pair in tokens."""
    if len(tokens) < 2:
        return 0
    pairs = list(zip(tokens, tokens[1:]))
    counts = Counter(pairs)
    return max(counts.values()) if counts else 0


def detect_excess_repetition(pred_tokens: Sequence[str], ref_tokens: Sequence[str]) -> bool:
    """Flag repetition loops in hypothesis that exceed natural repetition in reference.

    A hypothesis is flagged as an abnormal repetition loop only if:
    1) It has a 3+ token identical run AND that run exceeds reference run length; OR
    2) It has an adjacent token pair repeating 2+ times AND that count exceeds reference pair count.
    """
    pred_run = max_token_run(pred_tokens)
    ref_run = max_token_run(ref_tokens)
    if pred_run >= 3 and pred_run > ref_run:
        return True

    pred_pairs = max_token_pair_count(pred_tokens)
    ref_pairs = max_token_pair_count(ref_tokens)
    if pred_pairs >= 2 and pred_pairs > ref_pairs:
        return True

    return False


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
      repeat penalty = -0.25 (if excess repetition loop not in reference)
      too_long penalty = -0.15 (if hypothesis length > 1.5x reference length)
      hallucination penalty = -0.25 (if error_rate >= 0.8 and [too_long or repeated or major edits])
      final reward = clip(sum, clip_min, clip_max)

    Returns:
        (final_reward, reward_components, error_rate)
    """
    error_rate, edits, ref_len = compute_sample_error_rate(reference, prediction, language)
    asr_reward = 1.0 - min(error_rate, 1.0)

    cfg = reward_config or {}
    components_cfg = cfg.get("components", {})

    empty_pen_val = float(components_cfg.get("empty", {}).get("penalty", -0.25))
    repeat_pen_val = float(components_cfg.get("repeat", {}).get("penalty", -0.25))

    too_long_cfg = components_cfg.get("too_long", {})
    too_long_pen_val = float(too_long_cfg.get("penalty", -0.15))
    too_long_thresh = float(too_long_cfg.get("length_ratio_threshold", 1.5))

    hallucination_cfg = components_cfg.get("hallucination", {})
    hallucination_pen_val = float(hallucination_cfg.get("penalty", -0.25))
    hallucination_thresh = float(hallucination_cfg.get("error_rate_threshold", 0.8))

    clip_cfg = cfg.get("clip", {})
    clip_min = float(clip_cfg.get("min", -1.0))
    clip_max = float(clip_cfg.get("max", 1.0))

    norm_pred = normalize_text(prediction)
    norm_ref = normalize_text(reference)
    empty_output = not norm_pred

    lang_norm = str(language or "en").strip().lower()
    metric = "cer" if lang_norm in ("zh", "chinese") else "wer"
    pred_tokens = tokenize(norm_pred, metric=metric)
    ref_tokens = tokenize(norm_ref, metric=metric)
    ref_tokens_count = max(1, ref_len)

    repeated = detect_excess_repetition(pred_tokens, ref_tokens)
    length_ratio = len(pred_tokens) / ref_tokens_count
    too_long = length_ratio > too_long_thresh

    hallucination_like = (
        not empty_output
        and error_rate >= hallucination_thresh
        and (too_long or repeated or edits >= max(3, ref_tokens_count // 2))
    )

    empty_pen = empty_pen_val if empty_output else 0.0
    repeat_pen = repeat_pen_val if repeated else 0.0
    too_long_pen = too_long_pen_val if too_long else 0.0
    hallucination_pen = hallucination_pen_val if hallucination_like else 0.0

    raw_sum = asr_reward + empty_pen + repeat_pen + too_long_pen + hallucination_pen
    final_reward = max(clip_min, min(clip_max, raw_sum))

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


def compute_anchored_advantages(
    rewards: Sequence[float],
    anchor_index: int = 0,
    min_improvement: float = 0.02,
) -> Tuple[List[float], str]:
    """Advantage against the greedy hypothesis at ``anchor_index``.

    A sampled candidate gets a positive advantage only when its reward beats
    the greedy decode by at least ``min_improvement``. The best such candidate
    gets 1, and a smaller real improvement gets its gap divided by that best
    gap. The greedy decode and any candidate that does not beat it get 0, so
    a group of merely less-bad samples does not move the policy.

    Returns:
        (advantages, status) where status is ``update``, ``no_improvement``,
        or ``identical``.
    """
    group_size = len(rewards)
    if group_size == 0 or not (0 <= anchor_index < group_size):
        return [0.0] * group_size, "identical"

    anchor_reward = float(rewards[anchor_index])
    gaps = [
        0.0 if index == anchor_index else max(0.0, float(reward) - anchor_reward)
        for index, reward in enumerate(rewards)
    ]
    best_gap = max(gaps) if gaps else 0.0
    if best_gap < min_improvement:
        reward_span = max(rewards) - min(rewards) if rewards else 0.0
        status = "identical" if reward_span <= 1e-6 else "no_improvement"
        return [0.0] * group_size, status

    advantages = [gap / best_gap for gap in gaps]
    return advantages, "update"


def select_anchored_training_pair(
    texts: Sequence[str],
    rewards: Sequence[float],
    anchor_index: int = 0,
    min_improvement: float = 0.02,
    advantage_mode: str = "unit",
    fixed_advantage: float = 0.10,
    local_max_relative: Optional[float] = None,
    language: str = "en",
    advantage_cap: float = 0.05,
    reference_text: Optional[str] = None,
    reference_reward: Optional[float] = None,
) -> Tuple[List[str], List[float], List[float], str]:
    """Always return two sequences so every DDP rank builds the same shape.

    Index 0 is greedy with advantage 0. Index 1 is the best sample that beats
    greedy by at least ``min_improvement``; ties keep the lowest sample index
    and that winner's advantage is 1. A gap equal to ``min_improvement`` trains.
    Otherwise index 1 repeats greedy and both advantages are 0.
    """
    if len(texts) != len(rewards):
        raise ValueError("texts and rewards must have the same length")
    if not rewards or not (0 <= anchor_index < len(rewards)):
        raise ValueError("anchor_index is out of range")

    if (
        reference_text is not None
        and reference_reward is not None
        and local_max_relative is not None
    ):
        texts, rewards = append_reference_candidate(
            texts,
            rewards,
            reference_text,
            float(reference_reward),
        )

    if local_max_relative is not None:
        picked, picked_status = local_winner_index(
            texts,
            rewards,
            language=language,
            min_improvement=min_improvement,
            max_relative=float(local_max_relative),
            anchor_index=anchor_index,
        )
        anchor_text = texts[anchor_index]
        anchor_reward = float(rewards[anchor_index])
        if picked is None:
            return (
                [anchor_text, anchor_text],
                [anchor_reward, anchor_reward],
                [0.0, 0.0],
                picked_status,
            )
        return (
            [anchor_text, texts[picked]],
            [anchor_reward, float(rewards[picked])],
            [0.0, winner_advantage(
                anchor_reward,
                float(rewards[picked]),
                advantage_mode,
                fixed_advantage,
                advantage_cap,
            )],
            "update",
        )

    advantages, status = compute_anchored_advantages(
        rewards,
        anchor_index=anchor_index,
        min_improvement=min_improvement,
    )
    anchor_reward = float(rewards[anchor_index])
    anchor_text = texts[anchor_index]
    winner: Optional[int] = None
    best_advantage = 0.0
    for index, advantage in enumerate(advantages):
        if index == anchor_index:
            continue
        # Equal advantages keep the earlier index.
        if advantage > best_advantage:
            best_advantage = advantage
            winner = index

    # compute_anchored_advantages uses a strict `<`. A decimal gap of 0.02 can
    # land one ulp under that test; still train when the gap reaches it.
    if winner is None:
        best_gap = 0.0
        for index, reward in enumerate(rewards):
            if index == anchor_index:
                continue
            gap = max(0.0, float(reward) - anchor_reward)
            if gap > best_gap and _clears_min_improvement(gap, min_improvement):
                best_gap = gap
                winner = index
                status = "update"

    if winner is None:
        return (
            [anchor_text, anchor_text],
            [anchor_reward, anchor_reward],
            [0.0, 0.0],
            status,
        )
    return (
        [anchor_text, texts[winner]],
        [anchor_reward, float(rewards[winner])],
        [0.0, winner_advantage(
            anchor_reward,
            float(rewards[winner]),
            advantage_mode,
            fixed_advantage,
            advantage_cap,
        )],
        "update",
    )


def search_yield(
    records: Sequence[Mapping[str, Any]],
    min_improvement: float = 0.02,
    train_groups: int = 768,
) -> Dict[str, Any]:
    """Hit rate, winning-gap median, reward mass, and hit-rate SE for K=3 and K=11.

    Each record is one prompt: ``greedy_reward`` plus ``sample_rewards`` (11
    samples; the first 3 are K=3). Mass is ``update_rate * median_winning_gap
    * train_groups``. ``marginal_update_rate`` is the K=11 rate minus the K=3
    rate, so repeated extra samples contribute 0.
    """
    gaps_k3: List[float] = []
    gaps_k11: List[float] = []
    for record in records:
        if "greedy_reward" not in record or record.get("sample_rewards") is None:
            raise ValueError("search_yield record needs greedy_reward and sample_rewards")
        greedy = float(record["greedy_reward"])
        samples = [float(value) for value in record["sample_rewards"]]
        gap_k3 = max((max(0.0, sample - greedy) for sample in samples[:3]), default=0.0)
        gap_k11 = max((max(0.0, sample - greedy) for sample in samples[:11]), default=0.0)
        if _clears_min_improvement(gap_k3, min_improvement):
            gaps_k3.append(gap_k3)
        if _clears_min_improvement(gap_k11, min_improvement):
            gaps_k11.append(gap_k11)

    n = len(records)
    rate_k3 = (len(gaps_k3) / n) if n else 0.0
    rate_k11 = (len(gaps_k11) / n) if n else 0.0
    medians: List[float] = []
    for gaps in (gaps_k3, gaps_k11):
        if not gaps:
            medians.append(0.0)
            continue
        ordered = sorted(gaps)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            medians.append(ordered[mid])
        else:
            medians.append((ordered[mid - 1] + ordered[mid]) / 2.0)
    median_k3, median_k11 = medians
    groups = float(train_groups)
    return {
        "n_prompts": n,
        "update_rate_k3": rate_k3,
        "update_rate_k11": rate_k11,
        "median_winning_gap_k3": median_k3,
        "median_winning_gap_k11": median_k11,
        "marginal_update_rate": rate_k11 - rate_k3,
        "mass_k3": rate_k3 * median_k3 * groups,
        "mass_k11": rate_k11 * median_k11 * groups,
        "se_k3": None if n <= 0 else math.sqrt(rate_k3 * (1.0 - rate_k3) / n),
        "se_k11": None if n <= 0 else math.sqrt(rate_k11 * (1.0 - rate_k11) / n),
        "se_rate_only": True,
    }


# The stop/probe threshold tables. ``decide_rl_stop`` and the probe helpers read
# these values directly; YAML ``train.*`` entries with the same names are only
# contract copies. ``check_config_contract`` rejects a config whose copy differs,
# so editing e.g. ``train.max_raw_kl`` can no longer silently do nothing.
#
# ``pilot`` is the 4-24 step pilot table used by v10-v30 (behaviour unchanged).
# ``scale`` is the v31 full-pool table (approximately one epoch at 160k rows;
# see docs/qwen3-asr/29_rl_v31_scale_design.md):
# no step-4/8/12 search/transfer stops, a larger raw-KL ceiling, and a
# futility stop from ``futility_step`` on when the held-out greedy gain is
# below ``futility_min_gain``.
STOP_PROFILES: Dict[str, Dict[str, float]] = {
    "pilot": {
        "early_held_out_reward_drop": 0.005,
        "max_raw_kl": 5.0e-4,
        "min_step8_greedy_reward_gain": 0.001,
        "step4_lcb_fraction": 0.5,
        "step8_search_mass_floor": 8.50,
        "step8_transfer_mass_floor": 11.33,
        "step12_mass_floor": 17.0,
        "probe_mass_min": 17.0,
    },
    "scale": {
        "early_held_out_reward_drop": 0.005,
        "max_raw_kl": 2.0e-2,
        "futility_step": 640,
        "futility_min_gain": 0.0,
        "futility_patience": 2,
        "probe_mass_min": 17.0,
    },
}
DEFAULT_STOP_PROFILE = "pilot"
# Backwards-compatible name for the pilot table.
STOP_CONTRACT: Dict[str, float] = STOP_PROFILES[DEFAULT_STOP_PROFILE]
# ``length2_loss`` always backpropagates exactly two sequences.
TRAIN_SEQUENCES = 2


def resolve_stop_profile(name: Any) -> str:
    """Return a known stop profile name; ``None``/empty means ``pilot``."""
    text = DEFAULT_STOP_PROFILE if name is None else str(name).strip().lower()
    if not text:
        text = DEFAULT_STOP_PROFILE
    if text not in STOP_PROFILES:
        raise ValueError(
            f"unknown train.stop_profile {name!r}; expected one of {sorted(STOP_PROFILES)}"
        )
    return text


def check_config_contract(train_cfg: Mapping[str, Any], grpo_cfg: Mapping[str, Any]) -> None:
    """Raise ValueError when a YAML contract copy disagrees with the code.

    The active table is ``train.stop_profile`` (default ``pilot``). A threshold
    key that belongs only to another profile is also rejected: it would be
    ignored by the active profile and mislead a reader of the config.
    """
    profile = resolve_stop_profile(train_cfg.get("stop_profile"))
    table = STOP_PROFILES[profile]
    other_keys = set().union(*(set(t) for t in STOP_PROFILES.values())) - set(table)
    mismatches = []
    for key, expected in table.items():
        if key in train_cfg and not math.isclose(
            float(train_cfg[key]), float(expected), rel_tol=0.0, abs_tol=1e-12
        ):
            mismatches.append(
                f"train.{key}={train_cfg[key]!r} (code uses {expected} for stop_profile={profile})"
            )
    for key in sorted(other_keys & set(train_cfg)):
        mismatches.append(f"train.{key} is not used by stop_profile={profile}")
    if "train_sequences" in grpo_cfg and int(grpo_cfg["train_sequences"]) != TRAIN_SEQUENCES:
        mismatches.append(
            f"grpo.train_sequences={grpo_cfg['train_sequences']!r} (code uses {TRAIN_SEQUENCES})"
        )
    if mismatches:
        raise ValueError(
            "config disagrees with hard-coded RL thresholds; change the code and "
            "STOP_PROFILES together, not only the YAML: " + "; ".join(mismatches)
        )


def decide_rl_stop(
    global_step: int,
    cumulative_reward_mass: float,
    greedy_gain: Optional[float] = None,
    raw_kl: Optional[float] = None,
    robust_increase: Optional[float] = None,
    probe_update_rate: Optional[float] = None,
    probe_median_gap: Optional[float] = None,
    identical_ratio: Optional[float] = None,
    mean_reward: Optional[float] = None,
    consecutive_collapsed: int = 0,
    collapse_mean_reward_floor: float = 0.85,
    cumulative_winners: Optional[int] = None,
    consecutive_futile_evals: int = 0,
    profile: str = DEFAULT_STOP_PROFILE,
) -> str:
    """Return the first matching stop status, or ``""`` when training continues.

    This is the only threshold table. ``grad_norm`` is not an argument.
    ``None`` for greedy gain, raw KL, or robust increase means this call has
    no new measurement and that rule does not fire. Search stops (steps 4, 8,
    and 12 low mass) do not fire when greedy gain is at least +0.002.
    ``profile`` selects ``STOP_PROFILES``; ``scale`` has no step-4/8/12
    search/transfer stops and instead stops on a futility streak from
    ``futility_step``.  The caller increments ``consecutive_futile_evals`` only
    when a fresh full held-out evaluation is written.
    """
    step = int(global_step)
    mass = float(cumulative_reward_mass)
    gain = None if greedy_gain is None else float(greedy_gain)
    kl = None if raw_kl is None else float(raw_kl)
    robust = None if robust_increase is None else float(robust_increase)
    profile = resolve_stop_profile(profile)

    if (
        identical_ratio is not None
        and mean_reward is not None
        and float(identical_ratio) > 0.80
        and float(mean_reward) < float(collapse_mean_reward_floor)
        and int(consecutive_collapsed) >= 2
    ):
        return "FAILED_ZERO_VARIANCE"

    c = STOP_PROFILES[profile]
    if gain is not None and gain < -c["early_held_out_reward_drop"]:
        return "STOPPED_REWARD_DROP"

    if kl is not None and kl > c["max_raw_kl"]:
        return "STOPPED_KL"

    if step >= 2 and cumulative_winners is not None and int(cumulative_winners) == 0:
        return "BLOCKED_NO_WINNERS"

    if profile == "scale":
        if (
            gain is not None
            and step >= int(c["futility_step"])
            and gain < c["futility_min_gain"]
            and int(consecutive_futile_evals) >= int(c["futility_patience"])
        ):
            return "BLOCKED_TRANSFER"
        if robust is not None and robust >= 0.0005:
            return "STOPPED_ROBUST"
        return ""

    search_gain = gain is not None and gain < 0.002
    transfer_gain = gain is not None and gain < c["min_step8_greedy_reward_gain"]

    if (
        step >= 4
        and search_gain
        and probe_update_rate is not None
        and probe_median_gap is not None
    ):
        p = min(1.0, max(0.0, float(probe_update_rate)))
        se = math.sqrt(p * (1.0 - p) / 128.0)
        half_floor = c["step4_lcb_fraction"] * (256.0 * max(0.0, p - se) * float(probe_median_gap))
        if mass < half_floor:
            return "BLOCKED_SEARCH"

    if 8 <= step < 12 and search_gain and mass < c["step8_search_mass_floor"]:
        return "BLOCKED_SEARCH"

    if step == 8 and transfer_gain and mass >= c["step8_transfer_mass_floor"]:
        return "BLOCKED_TRANSFER"

    if step >= 12 and transfer_gain and mass >= c["step12_mass_floor"]:
        return "BLOCKED_TRANSFER"

    if step >= 12 and search_gain and mass < c["step12_mass_floor"]:
        return "BLOCKED_SEARCH"

    if robust is not None and robust >= 0.0005:
        return "STOPPED_ROBUST"

    return ""


def load_run_dose(
    source: Any = None,
    records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Sum per-step reward mass and winners. Do not read a chunk's cumulative field.

    ``source`` is JSONL text or a loss-log path. ``records`` is the parsed rows.
    A list passed as ``source`` is treated as records. Each log line is one JSON
    object. Collapse streak is the number of trailing rows with ``collapsed``
    true; rows without that key are skipped, and false stops the count.
    """
    if records is None and isinstance(source, (list, tuple)):
        records = source
        source = None
    if records is None:
        if source is None:
            text = ""
        elif isinstance(source, Path):
            text = source.read_text(encoding="utf-8")
        elif isinstance(source, str):
            stripped = source.lstrip()
            if stripped.startswith("{") or stripped.startswith("["):
                text = source
            else:
                path = Path(source)
                text = path.read_text(encoding="utf-8") if path.is_file() else source
        else:
            raise TypeError("load_run_dose source must be JSONL text, a path, or records")
        parsed = []
        for line_no, line in enumerate(text.splitlines(), 1):
            body = line.strip()
            if not body:
                continue
            try:
                item = json.loads(body)
            except json.JSONDecodeError as exc:
                raise ValueError(f"loss log line {line_no} is not JSON") from exc
            if not isinstance(item, dict):
                raise ValueError(f"loss log line {line_no} is not an object")
            parsed.append(item)
    else:
        parsed = [dict(row) for row in records]

    mass = 0.0
    winners = 0.0
    for row in parsed:
        if "reward_mass_in_step" in row and row["reward_mass_in_step"] is not None:
            mass += float(row["reward_mass_in_step"])
        if "winners_in_step" in row and row["winners_in_step"] is not None:
            winners += float(row["winners_in_step"])

    streak = 0
    for row in reversed(parsed):
        if "collapsed" not in row:
            continue
        if row["collapsed"] is True:
            streak += 1
            continue
        break

    return {
        "cumulative_reward_mass": mass,
        "cumulative_winners": winners,
        "consecutive_collapsed": streak,
    }


def _manifest_fields(rows: Sequence[Mapping[str, Any]], label: str) -> Dict[str, Any]:
    sample_ids: set[str] = set()
    paths: set[str] = set()
    hashes: set[str] = set()
    missing_hash = False
    saw_hash = False
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError(f"{label} manifest row is not an object")
        raw = row.get("audio")
        if raw is None or str(raw).strip() == "":
            raise ValueError("manifest row is missing audio path")
        paths.add(str(raw).strip())
        sample_id = row.get("sample_id")
        if sample_id is not None and str(sample_id).strip():
            sample_ids.add(str(sample_id).strip())
        digest = row.get("audio_sha256")
        if digest is None or str(digest).strip() == "":
            missing_hash = True
        else:
            saw_hash = True
            hashes.add(str(digest).strip().lower())
    return {
        "sample_ids": sample_ids,
        "paths": paths,
        "hashes": hashes,
        "missing_hash": missing_hash,
        "saw_hash": saw_hash,
    }


def _raise_if_overlap(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    left_name: str,
    right_name: str,
) -> None:
    for key, label in (
        ("sample_ids", "sample_id"),
        ("paths", "audio path"),
        ("hashes", "audio_sha256"),
    ):
        shared = left[key] & right[key]
        if shared:
            example = sorted(shared)[0]
            raise ValueError(
                f"BLOCKED_LEAKAGE: {label} overlap between {left_name} and {right_name}: {example}"
            )


def assert_manifests_disjoint(
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    wer_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Raise if train shares a sample id, audio path, or audio hash with val or wer.

    If val or wer has an ``audio_sha256`` and any train row omits it, raise too.
    A missing audio path raises before the leakage checks. Returns
    ``audio_sha256_checked`` false only when hash comparison did not run.
    """
    train = _manifest_fields(train_rows, "train")
    val = _manifest_fields(val_rows, "val")
    wer = _manifest_fields(wer_rows, "wer")
    _raise_if_overlap(train, val, "train", "val")
    _raise_if_overlap(train, wer, "train", "wer")

    if val["saw_hash"] and train["missing_hash"]:
        raise ValueError(
            "BLOCKED_LEAKAGE: val has audio_sha256 but a train row lacks it"
        )
    if wer["saw_hash"] and train["missing_hash"]:
        raise ValueError(
            "BLOCKED_LEAKAGE: wer has audio_sha256 but a train row lacks it"
        )

    other_has_hash = bool(val["saw_hash"] or wer["saw_hash"])
    checked = bool(train["saw_hash"] and not train["missing_hash"] and other_has_hash)
    return {"audio_sha256_checked": checked}


def read_manifest_rows(path: Path) -> List[Dict[str, Any]]:
    """Load a JSONL manifest. A non-object line raises."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path} line {line_no} is not an object")
            rows.append(item)
    return rows


def resolve_val_eval_samples(
    cli_value: Optional[int],
    train_cfg: Mapping[str, Any],
) -> Optional[int]:
    """Return an explicit eval cap, or None for the full held-out manifest.

    The old implicit default of 50 is not a cap. Smoke YAML sets the key;
    the formal YAML omits it.
    """
    if cli_value is not None:
        return int(cli_value)
    if "val_eval_samples" not in train_cfg or train_cfg.get("val_eval_samples") is None:
        return None
    return int(train_cfg["val_eval_samples"])


def projected_train_reward_mass(
    update_rate: float,
    median_gap: float,
    train_groups: int = 768,
) -> float:
    """Reward mass a 12-step run would stack at this hit rate and winning gap."""
    return float(update_rate) * float(median_gap) * float(train_groups)


def decide_search_probe(
    yield_stats: Mapping[str, Any],
    decode_exact_match_rate: float,
    decode_mean_abs_reward_diff: float,
    *,
    mass_min: float = STOP_CONTRACT["probe_mass_min"],
    train_groups: int = 768,
) -> str:
    """Probe decision from decode agreement and recomputed K=11 reward mass.

    Decode mismatch wins over a high mass. Mass is rate x gap x train_groups,
    not the stored ``decision`` string.
    """
    if (
        float(decode_exact_match_rate) < 0.95
        or float(decode_mean_abs_reward_diff) > 0.01
    ):
        return "BLOCKED_DECODE_MISMATCH"
    mass = projected_train_reward_mass(
        yield_stats["update_rate_k11"],
        yield_stats["median_winning_gap_k11"],
        train_groups,
    )
    if mass >= float(mass_min):
        return "GO_GRPO"
    rate = float(yield_stats["update_rate_k11"])
    gap = float(yield_stats["median_winning_gap_k11"])
    if rate >= 0.20 and gap >= 0.05:
        return "MINE_DPO_ONLY"
    return "BLOCKED_NO_HYPOTHESES"


def accept_probe_decision(
    payload: Mapping[str, Any],
    manifest_sha256: str,
    *,
    n_prompts: int = 128,
    train_groups: int = 768,
    mass_min: float = STOP_CONTRACT["probe_mass_min"],
) -> Dict[str, Any]:
    """Reject a probe file that is not a recomputed GO_GRPO for this manifest.

    The stored ``projected_train_reward_mass`` is ignored. Callers must not
    construct an optimizer when this raises.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("probe decision is not an object")
    decision = payload.get("decision")
    if decision != "GO_GRPO":
        raise ValueError(f"probe decision {decision!r} is not GO_GRPO")
    got_n = payload.get("n_prompts")
    if type(got_n) is bool or not isinstance(got_n, (int, float)) or int(got_n) != int(n_prompts):
        raise ValueError(f"probe n_prompts {got_n!r} is not {n_prompts}")
    sha = str(payload.get("manifest_sha256") or "")
    if not sha or sha != str(manifest_sha256):
        raise ValueError("probe manifest_sha256 does not match --manifest")
    try:
        rate = float(payload["update_rate_k11"])
        gap = float(payload["median_winning_gap_k11"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "probe is missing update_rate_k11 or median_winning_gap_k11"
        ) from exc
    mass = projected_train_reward_mass(rate, gap, train_groups)
    if mass < float(mass_min):
        raise ValueError(f"recomputed probe mass {mass:.6f} is below {mass_min}")
    return {
        "decision": "GO_GRPO",
        "n_prompts": int(n_prompts),
        "manifest_sha256": sha,
        "update_rate_k11": rate,
        "median_winning_gap_k11": gap,
        "projected_train_reward_mass": mass,
    }


def build_search_probe_payload(
    records: Sequence[Mapping[str, Any]],
    *,
    manifest_path: str,
    manifest_sha256: str,
    decode_exact_matches: int,
    decode_compared: int,
    decode_abs_reward_sum: float,
    seconds_per_generate_utterance: Optional[float],
    min_improvement: float = 0.02,
    train_groups: int = 768,
    seed: int = 20260722,
    failed_prompts: int = 0,
) -> Dict[str, Any]:
    """Call ``search_yield`` and ``decide_search_probe``. No model and no optimizer."""
    stats = search_yield(
        records,
        min_improvement=min_improvement,
        train_groups=train_groups,
    )
    compared = int(decode_compared)
    if compared <= 0:
        exact_rate = 0.0
        mean_abs = float("inf")
    else:
        exact_rate = int(decode_exact_matches) / compared
        mean_abs = float(decode_abs_reward_sum) / compared
    decision = decide_search_probe(
        stats,
        exact_rate,
        mean_abs,
        mass_min=STOP_CONTRACT["probe_mass_min"],
        train_groups=train_groups,
    )
    mass = projected_train_reward_mass(
        stats["update_rate_k11"],
        stats["median_winning_gap_k11"],
        train_groups,
    )
    return {
        "decision": decision,
        "n_prompts": stats["n_prompts"],
        "failed_prompts": int(failed_prompts),
        "manifest": manifest_path,
        "manifest_sha256": manifest_sha256,
        "update_rate_k3": stats["update_rate_k3"],
        "update_rate_k11": stats["update_rate_k11"],
        "median_winning_gap_k3": stats["median_winning_gap_k3"],
        "median_winning_gap_k11": stats["median_winning_gap_k11"],
        "marginal_update_rate": stats["marginal_update_rate"],
        "mass_k3": stats["mass_k3"],
        "mass_k11": stats["mass_k11"],
        "projected_train_reward_mass": mass,
        "se_k11": stats["se_k11"],
        "se_rate_only": True,
        "decode_exact_match_rate": exact_rate,
        "decode_mean_abs_reward_diff": mean_abs,
        "decode_n": compared,
        "seconds_per_generate_utterance": seconds_per_generate_utterance,
        "seed": int(seed),
        "min_improvement": float(min_improvement),
        "train_groups": int(train_groups),
    }


def is_cuda_oom(exc: BaseException) -> bool:
    """True for CUDA OOM errors without importing them at module load."""
    if exc.__class__.__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(exc).lower()


def generate_second_round_texts(
    generate_n: Any,
    sample_size: int = 8,
    max_generate_batch: int = 4,
    *,
    split_first: bool = True,
) -> List[str]:
    """Generate ``sample_size`` texts in batches of at most ``max_generate_batch``.

    The training path splits first (8 at batch 4 is two calls of 4). A full-size
    attempt that raises CUDA OOM retries with the same split. A split batch that
    still raises propagates to the caller, which must still reach the barrier.
    """
    total = int(sample_size)
    limit = int(max_generate_batch)
    if total <= 0:
        return []
    if limit <= 0:
        raise ValueError("max_generate_batch must be positive")

    def chunk_sizes(count: int) -> List[int]:
        sizes: List[int] = []
        left = count
        while left > 0:
            sizes.append(min(limit, left))
            left -= sizes[-1]
        return sizes

    def run_split() -> List[str]:
        texts: List[str] = []
        for size in chunk_sizes(total):
            batch = list(generate_n(size))
            if len(batch) != size:
                raise RuntimeError(f"expected {size} second-round sequences, got {len(batch)}")
            texts.extend(batch)
        return texts

    if split_first or total <= limit:
        return run_split()
    try:
        batch = list(generate_n(total))
        if len(batch) != total:
            raise RuntimeError(f"expected {total} second-round sequences, got {len(batch)}")
        return batch
    except Exception as exc:
        if not is_cuda_oom(exc):
            raise
        if HAVE_TORCH and torch.cuda.is_available():
            torch.cuda.empty_cache()
        return run_split()


def _median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def finalize_anchored_step_metrics(
    *,
    groups: float,
    winners: float,
    reward_mass: float,
    identical: float,
    no_improvement: float,
    second_round: float,
    first_reward_sum: float,
    first_reward_count: float,
    greedy_reward_sum: float,
    greedy_reward_count: float,
    gaps: Sequence[float],
) -> Dict[str, Any]:
    """Ratios after the global sums. Winner count and reward mass stay sums.

    ``world_size`` is not an argument. Dividing those sums by the world size
    would shrink the dose thresholds by 4.
    """
    groups_f = float(groups)

    def ratio(numerator: float) -> float:
        if groups_f <= 0:
            return 0.0
        return float(numerator) / groups_f

    mean_reward = (
        float(first_reward_sum) / float(first_reward_count)
        if float(first_reward_count) > 0
        else None
    )
    mean_greedy = (
        float(greedy_reward_sum) / float(greedy_reward_count)
        if float(greedy_reward_count) > 0
        else None
    )
    identical_ratio = ratio(identical)
    return {
        "groups": groups_f,
        "winners_in_step": float(winners),
        "reward_mass_in_step": float(reward_mass),
        "update_ratio": ratio(winners),
        "identical_ratio": identical_ratio,
        "zero_variance_ratio": identical_ratio,
        "no_improvement_ratio": ratio(no_improvement),
        "second_round_ratio": ratio(second_round),
        "mean_reward": mean_reward,
        "mean_greedy_reward": mean_greedy,
        "median_gap_in_step": _median(list(gaps)),
    }


def locked_step0_greedy_reward(records: Sequence[Mapping[str, Any]]) -> Optional[float]:
    """Step-0 baseline: greedy Full Held-out only. Sample rows are not a baseline."""
    reward: Optional[float] = None
    found = False
    for record in records:
        if record.get("global_step") != 0:
            continue
        if record.get("val_eval_scope") != "Full Held-out":
            continue
        if record.get("val_decode") != "greedy":
            continue
        found = True
        if record.get("val_mean_reward") is not None:
            reward = float(record["val_mean_reward"])
    return reward if found else None


def greedy_full_held_out_gain(
    records: Sequence[Mapping[str, Any]],
    step: int,
) -> Optional[float]:
    """Greedy full-pool reward at ``step`` minus the greedy step-0 reward."""
    step0: Optional[float] = None
    current: Optional[float] = None
    for record in records:
        if record.get("val_decode") != "greedy":
            continue
        if record.get("val_eval_scope") != "Full Held-out":
            continue
        if record.get("val_mean_reward") is None:
            continue
        if record.get("global_step") == 0:
            step0 = float(record["val_mean_reward"])
        if record.get("global_step") == step:
            current = float(record["val_mean_reward"])
    if step0 is None or current is None:
        return None
    return current - step0


def training_process_status(global_step: int, horizon: int, stop_status: str) -> str:
    """Status this process may write. ``COMPLETED`` is reserved for the driver."""
    del global_step, horizon
    if stop_status:
        return str(stop_status)
    return "CHUNK_DONE"


def allow_in_process_merged_export(
    global_step: int,
    horizon: int,
    stop_status: str,
    enabled: bool,
) -> bool:
    """In-process export only at the horizon when nothing stopped.

    A chunk end below the horizon does not export, even if the flag was left on.
    """
    if not enabled or stop_status:
        return False
    return int(global_step) >= int(horizon) and int(global_step) > 0


def plan_driver_action(
    global_step: int,
    horizon: int,
    *,
    cumulative_reward_mass: float,
    cumulative_winners: float,
    consecutive_collapsed: int,
    greedy_gain: Optional[float],
    robust_increase: Optional[float],
    probe_update_rate: Optional[float],
    probe_median_gap: Optional[float],
    gate_status: Optional[str] = None,
    raw_kl: Optional[float] = None,
    identical_ratio: Optional[float] = None,
    mean_reward: Optional[float] = None,
    collapse_mean_reward_floor: float = 0.75,
    stop_profile: str = DEFAULT_STOP_PROFILE,
    consecutive_futile_evals: int = 0,
) -> Dict[str, Any]:
    """Second ``decide_rl_stop`` after WER. Does not reimplement the thresholds.

    ``PASSED`` with an empty stop is ``CANDIDATE`` and the only export.
    An empty stop below the horizon resumes. ``COMPLETED`` is only the horizon
    with an empty stop and a gate that is not ``PASSED``.
    """
    stop = decide_rl_stop(
        int(global_step),
        float(cumulative_reward_mass),
        greedy_gain=greedy_gain,
        raw_kl=raw_kl,
        robust_increase=robust_increase,
        probe_update_rate=probe_update_rate,
        probe_median_gap=probe_median_gap,
        identical_ratio=identical_ratio,
        mean_reward=mean_reward,
        consecutive_collapsed=int(consecutive_collapsed),
        collapse_mean_reward_floor=float(collapse_mean_reward_floor),
        profile=stop_profile,
        cumulative_winners=int(cumulative_winners),
        consecutive_futile_evals=int(consecutive_futile_evals),
    )
    if gate_status == "PASSED" and stop == "":
        status = "CANDIDATE"
        export_merged = True
        resume = False
    elif stop:
        status = stop
        export_merged = False
        resume = False
    elif int(global_step) >= int(horizon):
        status = "COMPLETED"
        export_merged = False
        resume = False
    else:
        status = "CHUNK_DONE"
        export_merged = False
        resume = True
    return {
        "status": status,
        "export_merged": export_merged,
        "resume": resume,
        "stop": stop,
    }


def last_train_log_row(
    records: Sequence[Mapping[str, Any]],
    step: int,
) -> Optional[Dict[str, Any]]:
    """Last optimizer-step row at ``step`` (has per-step reward mass)."""
    found: Optional[Dict[str, Any]] = None
    for record in records:
        if record.get("global_step") != step:
            continue
        if "reward_mass_in_step" not in record:
            continue
        found = dict(record)
    return found


def abort_preflight(message: str) -> None:
    """Exit non-zero before distributed init, generation, or the optimizer."""
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(message, file=sys.stderr)
    raise SystemExit(1)


def parse_step_token(text: object) -> Optional[int]:
    """Return the integer in ``step_<N>``, or None when the token is absent."""
    match = re.fullmatch(r"step_(\d+)", str(text or ""))
    if not match:
        return None
    return int(match.group(1))


def checkpoint_is_complete(checkpoint: Path, world_size: int = 4) -> bool:
    """True when adapter, optimizer, scheduler, state, and every rank RNG exist."""
    path = Path(checkpoint)
    if world_size < 1:
        return False
    if not (path / "adapter").is_dir():
        return False
    required = ("optimizer.pt", "scheduler.pt", "training_state.json")
    if any(not (path / name).is_file() for name in required):
        return False
    return all((path / f"rng_state_rank_{rank}.pt").is_file() for rank in range(world_size))


def latest_complete_checkpoint_step(output_dir: Path, world_size: int = 4) -> int:
    """Highest committed ``checkpoints/step_<N>``, or 0 when none is complete."""
    root = Path(output_dir) / "checkpoints"
    best = 0
    if not root.is_dir():
        return 0
    for path in root.iterdir():
        step = parse_step_token(path.name)
        if step is None or step <= best or not path.is_dir():
            continue
        if checkpoint_is_complete(path, world_size):
            best = step
    return best


def is_terminal_train_status(status: object) -> bool:
    """True for a designed stop. ``CHUNK_DONE`` is resumable."""
    text = str(status or "")
    if text in {"COMPLETED", "CANDIDATE"}:
        return True
    return text.startswith(("BLOCKED_", "STOPPED_", "FAILED_"))


def select_resume_step(output_dir: Path, world_size: int = 4) -> Dict[str, Any]:
    """Choose the step a chunked launcher may continue from.

    A terminal pipeline status counts only when that step's checkpoint is
    complete. Otherwise the newest complete checkpoint wins, so a crash while
    writing ``pipeline_state.json`` or RNG cannot strand the run or redo a
    committed chunk.
    """
    if world_size < 1:
        raise ValueError("world_size must be positive")
    root = Path(output_dir)
    pipeline_step = 0
    pipeline_status = ""
    pipeline_path = root / "pipeline_state.json"
    if pipeline_path.is_file():
        try:
            payload = json.loads(pipeline_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            try:
                pipeline_step = int(payload.get("global_step") or 0)
            except (TypeError, ValueError):
                pipeline_step = 0
            pipeline_status = str(payload.get("status") or "")
    complete = latest_complete_checkpoint_step(root, world_size)
    pipeline_checkpoint = root / "checkpoints" / f"step_{pipeline_step}"
    pipeline_complete = pipeline_step > 0 and checkpoint_is_complete(pipeline_checkpoint, world_size)
    if is_terminal_train_status(pipeline_status) and pipeline_complete:
        return {
            "step": pipeline_step,
            "status": pipeline_status,
            "terminal": True,
            "source": "pipeline",
        }
    checkpoint_status = ""
    if complete:
        checkpoint_state = root / "checkpoints" / f"step_{complete}" / "training_state.json"
        try:
            state_payload = json.loads(checkpoint_state.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state_payload = {}
        if isinstance(state_payload, dict):
            checkpoint_status = str(state_payload.get("status") or "")
    if complete and is_terminal_train_status(checkpoint_status):
        return {
            "step": complete,
            "status": checkpoint_status,
            "terminal": True,
            "source": "checkpoint",
        }
    status = pipeline_status if pipeline_step == complete and pipeline_complete else ""
    if complete and not status:
        status = "CHUNK_DONE"
    return {
        "step": complete,
        "status": status,
        "terminal": False,
        "source": "checkpoint" if complete != pipeline_step or not pipeline_complete else "pipeline",
    }


def assert_resume_checkpoint(
    checkpoint: Path,
    *,
    manifest_sha256: str,
    world_size: int,
    model_revision: str,
    scheduler_name: str,
    require_scheduler: bool,
    val_manifest_sha256: str = "",
    wer_manifest_sha256: str = "",
    seed: Optional[int] = None,
    gradient_accumulation_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """Refuse a checkpoint whose data identity or schedule cannot be restored.

    Fields added after a historical run are checked only when the checkpoint
    recorded them. Manifest sha, world size, and a required scheduler file are
    always required.
    """
    path = Path(checkpoint)
    state_path = path / "training_state.json"
    if not state_path.is_file():
        raise ValueError(f"resume checkpoint is missing training_state.json: {path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(state, dict):
        raise ValueError("resume training_state.json is not an object")
    saved_manifest = str(state.get("manifest_sha256") or "")
    if not saved_manifest or saved_manifest != str(manifest_sha256):
        raise ValueError(
            "resume manifest sha256 does not match the current manifest: "
            f"checkpoint={saved_manifest or '<missing>'} current={manifest_sha256}"
        )
    saved_world = state.get("world_size")
    if saved_world is None or int(saved_world) != int(world_size):
        raise ValueError(
            f"resume world_size {saved_world!r} does not match the launch world_size {world_size}"
        )
    saved_revision = str(state.get("model_revision") or "")
    if model_revision and saved_revision != str(model_revision):
        raise ValueError(
            f"resume model_revision {saved_revision or '<missing>'} does not match {model_revision}"
        )
    if not (path / "adapter").is_dir():
        raise ValueError(f"resume checkpoint is missing adapter/: {path}")
    if not (path / "optimizer.pt").is_file():
        raise ValueError(f"resume checkpoint is missing optimizer.pt: {path}")
    for rank in range(int(world_size)):
        rng_path = path / f"rng_state_rank_{rank}.pt"
        if not rng_path.is_file():
            raise ValueError(f"resume checkpoint is missing {rng_path.name}")
    saved_scheduler = state.get("scheduler") if "scheduler" in state else None
    if require_scheduler:
        if not (path / "scheduler.pt").is_file():
            raise ValueError(f"resume checkpoint is missing scheduler.pt: {path}")
        if saved_scheduler is not None and str(saved_scheduler) != str(scheduler_name):
            raise ValueError(
                f"resume scheduler {saved_scheduler!r} does not match {scheduler_name!r}"
            )
    _require_saved_text(state, "val_manifest_sha256", val_manifest_sha256, "validation manifest")
    _require_saved_text(state, "wer_manifest_sha256", wer_manifest_sha256, "WER manifest")
    if seed is not None and "seed" in state and int(state["seed"]) != int(seed):
        raise ValueError(f"resume seed {state['seed']!r} does not match {seed}")
    if (
        gradient_accumulation_steps is not None
        and "gradient_accumulation_steps" in state
        and int(state["gradient_accumulation_steps"]) != int(gradient_accumulation_steps)
    ):
        raise ValueError(
            "resume gradient_accumulation_steps "
            f"{state['gradient_accumulation_steps']!r} does not match {gradient_accumulation_steps}"
        )
    return state


def _require_saved_text(state: Mapping[str, Any], key: str, current: str, label: str) -> None:
    if key not in state or not current:
        return
    saved = str(state.get(key) or "")
    if saved != str(current):
        raise ValueError(f"resume {label} sha256 does not match the current file")


def scale_futility_streak(
    records: Sequence[Mapping[str, Any]],
    *,
    profile: str = "scale",
) -> int:
    """Count trailing full held-out evals below the scale futility line.

    Evals before ``futility_step`` do not count. A gain at or above the line
    resets the streak. The last row for a step wins.
    """
    if resolve_stop_profile(profile) != "scale":
        return 0
    table = STOP_PROFILES["scale"]
    minimum_step = int(table["futility_step"])
    minimum_gain = float(table["futility_min_gain"])
    step0 = locked_step0_greedy_reward(records)
    if step0 is None:
        return 0
    measured: Dict[int, float] = {}
    for record in records:
        if record.get("val_decode") != "greedy" or record.get("val_eval_scope") != "Full Held-out":
            continue
        if record.get("val_mean_reward") is None or record.get("global_step") is None:
            continue
        step = int(record["global_step"])
        if step < minimum_step:
            continue
        measured[step] = float(record["val_mean_reward"])
    streak = 0
    for step in sorted(measured):
        if measured[step] - float(step0) < minimum_gain:
            streak += 1
        else:
            streak = 0
    return streak


def _rewrite_jsonl(path: Path, lines: Sequence[str]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    text = ("\n".join(lines) + "\n") if lines else ""
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def truncate_resume_artifacts(output_dir: Path, start_step: int, world_size: int) -> Dict[str, int]:
    """Drop rows written after the last committed optimizer step.

    Logged step ``N`` is committed only when ``checkpoints/step_N`` is the
    resume point. Rollout rows for that step are stored as
    ``policy_checkpoint=step_{N-1}`` because they are appended before
    ``global_step`` increments. Legacy rollout rows without that field stay.
    Concurrent ranks serialize on a lock; repeating the call is a no-op.
    """
    if start_step < 0:
        raise ValueError("start_step must be non-negative")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / ".resume_truncate.lock"
    with open(lock_path, "a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            return _truncate_resume_artifacts_unlocked(root, start_step, world_size)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _jsonl_records(path: Path) -> List[Tuple[str, Dict[str, Any]]]:
    if not path.is_file():
        return []
    rows: List[Tuple[str, Dict[str, Any]]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} line {line_no} is not JSON") from exc
        if not isinstance(record, dict):
            raise ValueError(f"{path} line {line_no} is not an object")
        rows.append((line, record))
    return rows


def _truncate_resume_artifacts_unlocked(
    output_dir: Path,
    start_step: int,
    world_size: int,
) -> Dict[str, int]:
    loss_path = output_dir / "loss_log.jsonl"
    kept_loss: List[str] = []
    dropped_loss = 0
    for line, record in _jsonl_records(loss_path):
        step = record.get("global_step")
        if isinstance(step, bool) or not isinstance(step, int):
            raise ValueError(f"{loss_path} has a loss row without an integer global_step")
        if step > start_step:
            dropped_loss += 1
            continue
        kept_loss.append(line)
    if loss_path.is_file():
        _rewrite_jsonl(loss_path, kept_loss)

    kept_rollouts = 0
    dropped_rollouts = 0
    rollout_paths = [output_dir / "rollouts.jsonl"]
    rollout_paths.extend(output_dir / f"rollouts_rank_{rank}.jsonl" for rank in range(world_size))
    for path in rollout_paths:
        if not path.is_file():
            continue
        kept: List[str] = []
        for line, record in _jsonl_records(path):
            step = parse_step_token(record.get("policy_checkpoint"))
            if step is not None and step >= start_step:
                dropped_rollouts += 1
                continue
            kept.append(line)
            kept_rollouts += 1
        _rewrite_jsonl(path, kept)
    return {
        "kept_loss_rows": len(kept_loss),
        "dropped_loss_rows": dropped_loss,
        "kept_rollout_rows": kept_rollouts,
        "dropped_rollout_rows": dropped_rollouts,
    }


# decide_rl_stop return values. Index 0 is "keep going".
STOP_STATUS_CODES: Tuple[str, ...] = (
    "",
    "FAILED_ZERO_VARIANCE",
    "STOPPED_REWARD_DROP",
    "STOPPED_KL",
    "BLOCKED_NO_WINNERS",
    "BLOCKED_SEARCH",
    "BLOCKED_TRANSFER",
    "STOPPED_ROBUST",
)


def broadcast_stop_status(status: str, *, device: str, is_distributed: bool) -> str:
    """Copy rank 0's stop status to every rank after the log is written.

    Other ranks must not decide from a stale loss-log read. An unknown status
    makes every rank raise after the broadcast.
    """
    if not is_distributed:
        return status
    if dist.get_rank() == 0:
        index = STOP_STATUS_CODES.index(status) if status in STOP_STATUS_CODES else -1
    else:
        index = 0
    tensor = torch.tensor([index], device=device, dtype=torch.int64)
    dist.broadcast(tensor, src=0)
    found = int(tensor.item())
    if found < 0 or found >= len(STOP_STATUS_CODES):
        raise RuntimeError(f"unknown RL stop status index {found}")
    return STOP_STATUS_CODES[found]


def load_asr_for_rl(
    config: Dict[str, Any],
    device: str,
    resume_from_checkpoint: Optional[Path] = None,
    rank: int = 0,
) -> Tuple[Any, Any, Dict[str, Any], str, Any]:
    """Load the DPO Champion policy and attach the 199-target LoRA adapter."""
    if Qwen3ASRModel is None:
        raise RuntimeError("qwen-asr is not installed")
    model_cfg = config.get("model", {})
    runtime_cfg = config.get("runtime", {})
    token_budget = training_token_budget(config)
    if resume_from_checkpoint:
        state = json.loads((resume_from_checkpoint / "training_state.json").read_text())
        check_resume_decoding(state, token_budget)
    model_id = str(model_cfg.get("model_id", "/data/mega-asr/runs/dpo_pilot_v2/merged_base"))
    revision = model_cfg.get("model_revision", "7278e1e70fe206f11671096ffdd38061171dd6e5")
    if rank == 0:
        print(f"Loading initial policy / reference model from {model_id} on rank {rank}...")
    asr_model = Qwen3ASRModel.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float16 if str(device).startswith("cuda") else torch.float32,
        attn_implementation="eager",
        max_new_tokens=token_budget,
    )
    configure_greedy_model(asr_model, token_budget)
    if rank == 0:
        print(f"decoding_contract={json.dumps(decoding_contract(token_budget), sort_keys=True)}")
    thinker = asr_model.model.thinker
    thinker.to(device)
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
        adapters_weights = None
        if adapter_safetensors.is_file():
            try:
                from safetensors.torch import load_file
                adapters_weights = load_file(str(adapter_safetensors))
            except ImportError:
                adapters_weights = torch.load(adapter_safetensors, map_location="cpu")
        elif adapter_bin.is_file():
            adapters_weights = torch.load(adapter_bin, map_location="cpu")
        if adapters_weights is not None:
            if set_peft_model_state_dict is not None:
                set_peft_model_state_dict(lora_model, adapters_weights)
            else:
                lora_model.load_state_dict(adapters_weights, strict=False)
    asr_model.model.thinker = lora_model
    return asr_model, lora_model, lora_meta, model_id, revision


def _json_count(value: float) -> int:
    return int(round(float(value)))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def driver_action_from_records(
    records: Sequence[Mapping[str, Any]],
    *,
    global_step: int,
    horizon: int,
    probe_payload: Mapping[str, Any],
    gate_status: Optional[str],
    robust_increase: Optional[float],
    collapse_mean_reward_floor: float = 0.75,
    stop_profile: str = DEFAULT_STOP_PROFILE,
) -> Dict[str, Any]:
    """Build the post-WER driver action from the loss log and probe rates.

    Dose comes from ``load_run_dose`` (per-step increments, not a chunk cumulative).
    The stop string comes only from ``decide_rl_stop``.
    """
    dose = load_run_dose(records=list(records))
    row = last_train_log_row(records, int(global_step))
    raw_kl = None if row is None or row.get("raw_kl") is None else float(row["raw_kl"])
    identical = (
        None
        if row is None or row.get("zero_variance_ratio") is None
        else float(row["zero_variance_ratio"])
    )
    mean_reward = None if row is None or row.get("mean_reward") is None else float(row["mean_reward"])
    futility_streak = scale_futility_streak(records, profile=stop_profile)
    return plan_driver_action(
        int(global_step),
        int(horizon),
        cumulative_reward_mass=float(dose["cumulative_reward_mass"]),
        cumulative_winners=float(dose["cumulative_winners"]),
        consecutive_collapsed=int(dose["consecutive_collapsed"]),
        greedy_gain=greedy_full_held_out_gain(records, int(global_step)),
        robust_increase=robust_increase,
        probe_update_rate=float(probe_payload["update_rate_k11"]),
        probe_median_gap=float(probe_payload["median_winning_gap_k11"]),
        gate_status=gate_status,
        raw_kl=raw_kl,
        identical_ratio=identical,
        mean_reward=mean_reward,
        collapse_mean_reward_floor=float(collapse_mean_reward_floor),
        stop_profile=stop_profile,
        consecutive_futile_evals=futility_streak,
    )


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
    declared_size: Dict[str, int] = {}
    greedy_rows: Dict[str, int] = collections.defaultdict(int)
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
            size = int(rec["group_size"])
            if gid in declared_size and declared_size[gid] != size:
                raise ValueError(
                    f"Group {gid} declares conflicting group_size "
                    f"{declared_size[gid]} and {size}."
                )
            declared_size[gid] = size
            if rec.get("decode_mode") == "greedy":
                greedy_rows[gid] += 1
            ranks_found.add(int(rec.get("rank", 0)))

    # Each group is complete for its own group_size and has one greedy row.
    for gid, cnt in groups.items():
        expected = declared_size[gid]
        if cnt != expected:
            raise ValueError(
                f"Group {gid} in {rollouts_path} has {cnt} rollouts, expected {expected}."
            )
        if greedy_rows[gid] != 1:
            raise ValueError(
                f"Group {gid} in {rollouts_path} has {greedy_rows[gid]} greedy rows, expected 1."
            )

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


def compute_grpo_group_loss(
    policy_token_logps: torch.Tensor,
    ref_token_logps: torch.Tensor,
    token_mask: torch.Tensor,
    advantages: Sequence[float],
    beta: float,
    zero_variance: bool,
    reduction: str = "sequence_sum",
    policy_keep: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """GRPO policy loss plus Schulman K3 KL for one candidate group.

    The reward is a sequence score. ``sequence_sum`` uses
    ``-A_i * sum_t log pi(y_t) + beta * sum_t K3_t``. ``token_mean`` divides
    that sum by the response length, which shrinks the gradient by about the
    number of response tokens.

    A zero-variance group returns a graph-connected zero so DDP still
    synchronizes. Logged policy/KL contributions are 0 in that case. The
    returned ``seq_kl_mean`` is always the per-token K3 mean, including
    zero-variance groups, and is detached for rollout logging.
    """
    if reduction not in ("sequence_sum", "token_mean"):
        raise ValueError(f"Unsupported GRPO loss reduction: {reduction}")

    log_ratio = (ref_token_logps - policy_token_logps) * token_mask
    log_ratio_clamped = log_ratio.clamp(min=-10.0, max=10.0)
    token_kl = (torch.exp(log_ratio_clamped) - log_ratio_clamped - 1.0) * token_mask
    token_count = token_mask.sum(dim=-1).clamp(min=1)
    seq_kl_mean = token_kl.sum(dim=-1) / token_count

    adv_tensor = torch.tensor(
        list(advantages),
        device=policy_token_logps.device,
        dtype=policy_token_logps.dtype,
    )
    if policy_keep is None:
        policy_region = token_mask
    else:
        if policy_keep.shape != token_mask.shape:
            raise RuntimeError(
                f"policy_keep shape {tuple(policy_keep.shape)} != "
                f"token_mask {tuple(token_mask.shape)}"
            )
        policy_region = token_mask * policy_keep
    policy_token = -(adv_tensor.unsqueeze(-1) * policy_token_logps) * policy_region
    kl_token = (beta * token_kl) * token_mask
    token_loss = policy_token + kl_token

    if reduction == "token_mean":
        policy_seq = policy_token.sum(dim=-1) / token_count
        kl_seq = kl_token.sum(dim=-1) / token_count
        seq_losses = token_loss.sum(dim=-1) / token_count
    else:
        policy_seq = policy_token.sum(dim=-1)
        kl_seq = kl_token.sum(dim=-1)
        seq_losses = token_loss.sum(dim=-1)

    if zero_variance:
        group_loss = policy_token_logps.sum() * 0.0
        zero = torch.zeros((), device=policy_token_logps.device, dtype=policy_token_logps.dtype)
        policy_log: torch.Tensor = zero
        kl_log: torch.Tensor = zero
        raw_kl_log: torch.Tensor = zero
    else:
        group_loss = seq_losses.mean()
        policy_log = policy_seq.mean()
        kl_log = kl_seq.mean()
        raw_kl_log = seq_kl_mean.mean()

    return {
        "group_loss": group_loss,
        "policy_loss": policy_log,
        "kl_loss": kl_log,
        "raw_kl": raw_kl_log,
        "seq_kl_mean": seq_kl_mean.detach(),
    }


def setup_lora_rl(thinker: Any, config: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """Inject canonical Linear LoRA targets into the thinker.

    The default keeps all 199 targets. lora.train_audio_projections false
    drops the three audio-tower projections and leaves 196 decoder targets.
    """
    target_modules: List[str] = []
    for name, module in thinker.named_modules():
        if isinstance(module, torch.nn.Linear) and LORA_TARGET_REGEX.match(name):
            target_modules.append(name)

    lora_cfg = config.get("lora", {})
    train_audio = bool(lora_cfg.get("train_audio_projections", True))
    target_modules = filter_lora_targets(target_modules, train_audio)
    expected = 199 if train_audio else 196
    audio_present = any(name.startswith("audio_tower.") for name in target_modules)
    if len(target_modules) != expected or audio_present != train_audio:
        raise RuntimeError(
            f"LoRA targets {len(target_modules)} audio={audio_present} "
            f"train_audio_projections={train_audio}"
        )
    print(
        f"LoRA targets: {len(target_modules)} train_audio_projections={train_audio}",
        flush=True,
    )
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
    reward_config: Optional[Dict[str, Any]] = None,
    eval_seed: int = 42,
    decode_mode: str = "sample",
) -> Tuple[float, float, int, int]:
    """Evaluate mean reward on the held-out set, sharded across ranks.

    ``decode_mode="greedy"`` uses ``do_sample=False`` and one sequence.
    ``max_eval_samples`` truncates the prefix; the formal path passes None.

    Returns mean reward, mean error, scored rollout count, and the number of
    manifest rows assigned across ranks. Assigned rows cover the evaluated
    prefix exactly once.
    """
    if not HAVE_TORCH or len(val_dataset) == 0:
        return 0.0, 0.0, 0, 0

    eval_model = thinker_model.module if hasattr(thinker_model, "module") else thinker_model
    eval_model.eval()

    is_dist = HAVE_TORCH and dist.is_available() and dist.is_initialized()
    world_size = dist.get_world_size() if is_dist else 1
    rank = dist.get_rank() if is_dist else 0

    mode = str(decode_mode or "sample").strip().lower()
    if mode not in ("greedy", "sample"):
        raise ValueError(f"decode_mode must be greedy or sample, got {decode_mode!r}")
    sequences = 1 if mode == "greedy" else group_size
    prefix = len(val_dataset) if max_eval_samples is None else min(len(val_dataset), int(max_eval_samples))
    all_indices = list(range(prefix))
    indices = [idx for idx in all_indices if idx % world_size == rank]

    total_reward = 0.0
    total_rollouts = 0
    total_error_rate = 0.0

    # Save RNG state to preserve training sequence determinism
    cpu_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        torch.manual_seed(eval_seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(eval_seed + rank)

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
                    if mode == "greedy":
                        gen_out = asr_model.model.generate(
                            **inputs,
                            do_sample=False,
                            num_return_sequences=1,
                            max_new_tokens=asr_model.max_new_tokens,
                        )
                    else:
                        gen_out = asr_model.model.generate(
                            **inputs,
                            do_sample=True,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            num_return_sequences=group_size,
                            max_new_tokens=asr_model.max_new_tokens,
                        )
                    seqs = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
                except Exception:
                    continue

                for g_idx in range(min(sequences, seqs.shape[0])):
                    cand_ids = seqs[g_idx, prompt_len:]
                    pred_text = asr_model.processor.tokenizer.decode(cand_ids, skip_special_tokens=True).strip()
                    rew, _, err = compute_sequence_reward(pred_text, gold_text, lang_raw, reward_config=reward_config)
                    total_reward += rew
                    total_error_rate += err
                    total_rollouts += 1
    finally:
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng_state)

    assigned_rows = float(len(indices))
    if is_dist:
        stat_tensor = torch.tensor(
            [total_reward, total_error_rate, float(total_rollouts), assigned_rows],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(stat_tensor, op=dist.ReduceOp.SUM)
        global_reward, global_error, global_rollouts, global_assigned = stat_tensor.tolist()
    else:
        global_reward = total_reward
        global_error = total_error_rate
        global_rollouts = float(total_rollouts)
        global_assigned = assigned_rows

    mean_reward = global_reward / max(1.0, global_rollouts)
    mean_err = global_error / max(1.0, global_rollouts)
    return mean_reward, mean_err, int(global_rollouts), int(global_assigned)


def _generate_decoded(asr_model: Any, inputs: Mapping[str, Any], prompt_len: int, **gen_kwargs: Any) -> List[str]:
    """Clone inputs and decode one generate() call. Does not step the optimizer."""
    gen_inputs = {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    gen_out = asr_model.model.generate(**gen_inputs, max_new_tokens=asr_model.max_new_tokens, **gen_kwargs)
    seqs = gen_out.sequences if hasattr(gen_out, "sequences") else gen_out
    texts: List[str] = []
    for row_index in range(int(seqs.shape[0])):
        tokens = seqs[row_index, prompt_len:]
        texts.append(
            asr_model.processor.tokenizer.decode(tokens, skip_special_tokens=True).strip()
        )
    return texts


def _prepare_prompt_inputs(asr_model: Any, wav: Any, language: str, device: str) -> Tuple[str, Any, int]:
    lang_full = LANGUAGE_MAP.get(str(language or "en").strip().lower(), "English")
    prompt = asr_model._build_text_prompt(context="", force_language=lang_full)
    inputs = asr_model.processor(text=prompt, audio=wav, return_tensors="pt")
    inputs = inputs.to(device)
    if str(device).startswith("cuda"):
        inputs["input_features"] = inputs["input_features"].to(torch.float16)
    prompt_len = int(inputs["input_ids"].shape[1])
    return prompt, inputs, prompt_len


def execute_search_probe(
    config: Dict[str, Any],
    train_dataset: RLAudioDataset,
    manifest_path: Path,
    output_dir: Path,
    *,
    seed: int,
    probe_prompts: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_improvement: float,
    max_generate_batch: int,
    second_round_size: int,
    group_size: int,
    reward_cfg: Dict[str, Any],
    rank: int,
    world_size: int,
    is_distributed: bool,
    device: str,
) -> None:
    """Score a degraded prefix with 1 greedy + 11 samples. No optimizer and no export."""
    asr_model, _lora_model, _lora_meta, _model_id, _revision = load_asr_for_rl(
        config,
        device,
        resume_from_checkpoint=None,
        rank=rank,
    )
    asr_model.model.eval()
    indices = build_epoch_sample_indices(
        train_dataset,
        strategy="degraded",
        epoch=0,
        seed=seed,
    )[: int(probe_prompts)]
    decode_until = min(32, len(indices))
    records: List[Dict[str, Any]] = []
    decode_exact = 0
    decode_compared = 0
    decode_abs = 0.0
    failed = 0
    generate_seconds = 0.0
    from inference.run_inference import run_single_inference

    mod = 2**31 - 1
    if int(group_size) < 1:
        raise ValueError("group_size must be positive")
    # K=3 is the first three samples. The next eight are the second round, always.
    first_round_samples = 3
    for ordinal, dataset_index in enumerate(indices):
        if ordinal % world_size != rank:
            continue
        sample = train_dataset[int(dataset_index)]
        sample_id = str(sample.get("sample_id", f"sample_{dataset_index}"))
        started = time.time()
        scored = False
        try:
            audio_path = str(sample.get("audio") or "")
            if not audio_path or not os.path.isfile(audio_path):
                raise FileNotFoundError(f"missing audio: {audio_path}")
            wav, _sr = sf.read(audio_path)
            if len(getattr(wav, "shape", [])) > 1:
                wav = wav[:, 0]
            lang_raw = str(sample.get("language", "en")).strip().lower()
            gold_text = str(sample.get("text") or sample.get("answer") or "")
            _prompt, inputs, prompt_len = _prepare_prompt_inputs(asr_model, wav, lang_raw, device)
            cand_seed = (int(seed) + rank * 100000 + ordinal * 1000) % mod
            with torch.no_grad():
                greedy_texts = _generate_decoded(
                    asr_model, inputs, prompt_len, do_sample=False, num_return_sequences=1,
                )
                if not greedy_texts:
                    raise RuntimeError(f"greedy generate returned no text for {sample_id}")
                torch.manual_seed(cand_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(cand_seed)
                first = (
                    _generate_decoded(
                        asr_model,
                        inputs,
                        prompt_len,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        num_return_sequences=first_round_samples,
                    )
                    if first_round_samples
                    else []
                )
                while len(first) < first_round_samples:
                    first.append(first[-1] if first else greedy_texts[0])
                round_seed = (cand_seed + 10007) % mod
                torch.manual_seed(round_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(round_seed)

                def generate_n(count: int) -> List[str]:
                    return _generate_decoded(
                        asr_model,
                        inputs,
                        prompt_len,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        num_return_sequences=count,
                    )

                extra = generate_second_round_texts(
                    generate_n,
                    sample_size=int(second_round_size),
                    max_generate_batch=int(max_generate_batch),
                    split_first=True,
                )
            sample_texts = first[:first_round_samples] + extra
            if len(sample_texts) < 11:
                raise RuntimeError(
                    f"probe expected 11 samples for {sample_id}, got {len(sample_texts)}"
                )
            greedy_reward, _, _ = compute_sequence_reward(
                greedy_texts[0], gold_text, lang_raw, reward_config=reward_cfg,
            )
            sample_rewards = [
                compute_sequence_reward(text, gold_text, lang_raw, reward_config=reward_cfg)[0]
                for text in sample_texts[:11]
            ]
            records.append({
                "greedy_reward": float(greedy_reward),
                "sample_rewards": [float(value) for value in sample_rewards],
            })
            scored = True
            if ordinal < decode_until:
                try:
                    transcribed = run_single_inference(asr_model, audio_path, language=lang_raw)
                    exact = normalize_text(greedy_texts[0]) == normalize_text(transcribed)
                    other_reward, _, _ = compute_sequence_reward(
                        transcribed, gold_text, lang_raw, reward_config=reward_cfg,
                    )
                    abs_gap = abs(float(greedy_reward) - float(other_reward))
                except Exception as exc:
                    print(f"Probe transcribe failed on {sample_id}: {exc}", file=sys.stderr)
                    exact = False
                    abs_gap = 1.0
                decode_compared += 1
                decode_exact += int(exact)
                decode_abs += float(abs_gap)
        except Exception as exc:
            failed += 1
            print(f"Probe generation error on {sample_id}: {exc}", file=sys.stderr)
            if ordinal < decode_until and not scored:
                decode_compared += 1
                decode_abs += 1.0
        finally:
            generate_seconds += time.time() - started

    output_dir.mkdir(parents=True, exist_ok=True)
    partial = {
        "records": records,
        "decode_exact_matches": decode_exact,
        "decode_compared": decode_compared,
        "decode_abs_reward_sum": decode_abs,
        "generate_seconds": generate_seconds,
        "failed_prompts": failed,
    }
    (output_dir / f"probe_rank_{rank}.json").write_text(
        json.dumps(partial, ensure_ascii=False),
        encoding="utf-8",
    )
    if is_distributed:
        dist.barrier()
    if rank == 0:
        try:
            merged_records: List[Dict[str, Any]] = []
            exact_total = 0
            compared_total = 0
            abs_total = 0.0
            seconds_total = 0.0
            failed_total = 0
            attempts = 0
            for local_rank in range(world_size):
                payload = json.loads(
                    (output_dir / f"probe_rank_{local_rank}.json").read_text(encoding="utf-8")
                )
                merged_records.extend(payload.get("records") or [])
                exact_total += int(payload.get("decode_exact_matches") or 0)
                compared_total += int(payload.get("decode_compared") or 0)
                abs_total += float(payload.get("decode_abs_reward_sum") or 0.0)
                seconds_total += float(payload.get("generate_seconds") or 0.0)
                failed_total += int(payload.get("failed_prompts") or 0)
                attempts += len(payload.get("records") or []) + int(payload.get("failed_prompts") or 0)
            seconds_each = (seconds_total / attempts) if attempts else None
            probe_payload = build_search_probe_payload(
                merged_records,
                manifest_path=str(manifest_path),
                manifest_sha256=compute_file_sha256(manifest_path),
                decode_exact_matches=exact_total,
                decode_compared=compared_total,
                decode_abs_reward_sum=abs_total,
                seconds_per_generate_utterance=seconds_each,
                min_improvement=min_improvement,
                train_groups=768,
                seed=seed,
                failed_prompts=failed_total,
            )
            (output_dir / "search_probe.json").write_text(
                json.dumps(probe_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            print(
                f"Search probe decision={probe_payload['decision']} "
                f"n_prompts={probe_payload['n_prompts']} "
                f"mass={probe_payload['projected_train_reward_mass']:.4f}"
            )
        except Exception as exc:
            print(f"Failed to write search_probe.json: {exc}", file=sys.stderr)
    if is_distributed:
        dist.barrier()
    if not (output_dir / "search_probe.json").is_file():
        raise RuntimeError("search_probe.json was not written")

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
    reward_config_path: Optional[Path] = None,
    eval_seed: int = 42,
    sample_strategy: Optional[str] = None,
    export_merged_on_finish: bool = True,
    wer_manifest_path: Optional[Path] = None,
    probe_decision_path: Optional[Path] = None,
    probe_only: bool = False,
    probe_prompts: int = 128,
) -> None:
    """Train anchored GRPO, or run the search probe when ``probe_only`` is set.

    YAML ``train.max_steps`` is the horizon. ``max_steps_override`` is only this
    process's chunk end. Failures still reach the barrier and a length-2 backward.
    """
    with open(config_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    runtime_cfg = config.get("runtime", {})
    train_cfg = config.get("train", {})
    grpo_cfg = config.get("grpo", {})
    token_budget = training_token_budget(config)
    reward_cfg = config.get("reward", {})
    if reward_config_path and Path(reward_config_path).is_file():
        with open(reward_config_path, "r", encoding="utf-8") as reward_handle:
            reward_cfg = yaml.safe_load(reward_handle) or {}
    elif "components" not in reward_cfg:
        cfg_path_str = reward_cfg.get("config_path")
        candidate_path = (
            (REPO_ROOT / cfg_path_str)
            if cfg_path_str
            else (REPO_ROOT / "configs" / "train" / "reward_config.yaml")
        )
        if candidate_path.is_file():
            try:
                with open(candidate_path, "r", encoding="utf-8") as reward_handle:
                    loaded_cfg = yaml.safe_load(reward_handle) or {}
                    if isinstance(loaded_cfg, dict):
                        reward_cfg = loaded_cfg
            except Exception:
                pass

    seed = int(runtime_cfg.get("seed", 20260722))
    sample_strategy = resolve_sample_strategy(sample_strategy, train_cfg.get("sample_strategy"))
    check_config_contract(train_cfg, grpo_cfg)
    stop_profile = resolve_stop_profile(train_cfg.get("stop_profile"))
    sample_strategy_options = train_cfg.get("sample_strategy_options") or {}
    if not isinstance(sample_strategy_options, dict):
        raise ValueError("train.sample_strategy_options must be a mapping")
    horizon = int(train_cfg.get("max_steps", 12))
    chunk_end = horizon if max_steps_override is None else int(max_steps_override)
    save_steps = int(train_cfg.get("save_steps", 4) if save_steps_override is None else save_steps_override)
    eval_steps = int(train_cfg.get("eval_steps", 4) if eval_steps_override is None else eval_steps_override)
    val_eval_limit = resolve_val_eval_samples(val_eval_samples, train_cfg)
    lr = float(train_cfg.get("learning_rate", 1.0e-5))
    warmup_steps = int(train_cfg.get("warmup_steps", 2))
    grad_accum_steps = int(runtime_cfg.get("gradient_accumulation_steps", 16))
    group_size = int(grpo_cfg.get("group_size", 4))
    sampling_cfg = grpo_cfg.get("sampling", {})
    temperature = (
        float(temperature_override)
        if temperature_override is not None
        else float(sampling_cfg.get("temperature", 1.0))
    )
    top_p = float(top_p_override) if top_p_override is not None else float(sampling_cfg.get("top_p", 0.95))
    top_k = int(top_k_override) if top_k_override is not None else int(sampling_cfg.get("top_k", 50))
    adv_cfg = grpo_cfg.get("advantage", {})
    anchor_mode = str(adv_cfg.get("anchor", "none")).strip().lower()
    min_improvement = float(adv_cfg.get("min_improvement", 0.02))
    advantage_mode = str(adv_cfg.get("mode", "unit")).strip().lower()
    fixed_advantage = float(adv_cfg.get("fixed_value", 0.10))
    advantage_cap = float(adv_cfg.get("cap", 0.05))
    local_raw = adv_cfg.get("local_max_relative", None)
    local_max_relative = None if local_raw is None else float(local_raw)
    if local_max_relative is not None and not 0.0 < local_max_relative <= 1.0:
        abort_preflight(
            f"advantage.local_max_relative must be in (0, 1], got {local_max_relative}"
        )
    if advantage_mode not in {"unit", "raw_gap", "fixed", "capped_gap"}:
        abort_preflight(
            f"advantage.mode must be unit, raw_gap, fixed, or capped_gap, got {advantage_mode!r}"
        )
    if advantage_mode == "capped_gap" and not 0.0 < advantage_cap <= 1.0:
        abort_preflight(
            f"advantage.cap must be in (0, 1], got {advantage_cap}"
        )
    zero_var_thresh = (
        float(zero_var_thresh_override)
        if zero_var_thresh_override is not None
        else float(adv_cfg.get("zero_variance_threshold", 0.80))
    )
    collapse_floor = float(adv_cfg.get("collapse_mean_reward_floor", 0.85))
    second_cfg = grpo_cfg.get("second_round") or {}
    second_enabled = bool(second_cfg.get("enabled", True))
    second_sample_size = int(second_cfg.get("sample_size", 8))
    max_generate_batch = int(second_cfg.get("max_generate_batch", 4))
    beta = float((grpo_cfg.get("kl_regularization") or {}).get("beta", 0.04))
    loss_reduction = str(train_cfg.get("loss_reduction", "sequence_sum")).strip().lower()
    if loss_reduction not in ("sequence_sum", "token_mean"):
        abort_preflight(f"train.loss_reduction must be sequence_sum or token_mean, got {loss_reduction!r}")
    policy_token_mask = str(grpo_cfg.get("policy_token_mask", "all")).strip().lower()
    if policy_token_mask not in {"all", "changes_only", "signed_edits"}:
        abort_preflight(
            f"grpo.policy_token_mask must be all, changes_only, or signed_edits, got {policy_token_mask!r}"
        )
    include_reference_raw = grpo_cfg.get("include_reference_candidate", False)
    if not isinstance(include_reference_raw, bool):
        abort_preflight(
            "grpo.include_reference_candidate must be a boolean, "
            f"got {include_reference_raw!r}"
        )
    include_reference_candidate = include_reference_raw
    if anchor_mode != "greedy":
        abort_preflight(f"advantage.anchor must be greedy for v10, got {anchor_mode!r}")

    missing_paths = []
    for label, path in (
        ("manifest", manifest_path),
        ("val-manifest", val_manifest_path),
        ("wer-manifest", wer_manifest_path),
    ):
        if path is None or not Path(path).is_file():
            missing_paths.append(f"--{label} ({path})")
    if missing_paths:
        abort_preflight(
            "missing required manifest, refusing to generate or construct an optimizer: "
            + ", ".join(missing_paths)
        )

    dataset_gate_path = manifest_path.parent / "DATASET_COMPLETE.json"
    if dataset_gate_path.is_file():
        with open(dataset_gate_path, "r", encoding="utf-8") as gate_handle:
            gate_data = json.load(gate_handle)
        gate_status = gate_data.get("status", "UNKNOWN")
        if gate_status == "FAILED":
            abort_preflight(f"Dataset gate check FAILED in {dataset_gate_path}.")
        if gate_status == "NON_STRICT_SUBSET":
            is_pilot_or_smoke = "pilot" in manifest_path.name or "smoke" in manifest_path.name
            if not (allow_subset or is_pilot_or_smoke):
                abort_preflight(
                    f"Dataset gate status is NON_STRICT_SUBSET in {dataset_gate_path}. "
                    "Formal full training requires PASSED. Provide --allow-subset for pilot or smoke runs."
                )

    assert manifest_path is not None and val_manifest_path is not None and wer_manifest_path is not None
    train_rows = read_manifest_rows(manifest_path)
    val_rows = read_manifest_rows(val_manifest_path)
    wer_rows = read_manifest_rows(wer_manifest_path)
    try:
        disjoint = assert_manifests_disjoint(train_rows, val_rows, wer_rows)
    except ValueError as exc:
        abort_preflight(str(exc))

    train_dataset = RLAudioDataset(manifest_path)
    val_dataset = RLAudioDataset(val_manifest_path)
    try:
        epoch_0_indices = build_epoch_sample_indices(
            train_dataset,
            strategy=sample_strategy,
            epoch=0,
            seed=seed,
            strategy_options=sample_strategy_options,
        )
    except ValueError as exc:
        abort_preflight(str(exc))
    if not epoch_0_indices:
        abort_preflight(f"sample strategy {sample_strategy!r} produced an empty epoch")
    virtual_epoch_len = len(epoch_0_indices)
    manifest_sha = compute_file_sha256(manifest_path)
    val_manifest_sha = compute_file_sha256(val_manifest_path)
    wer_manifest_sha = compute_file_sha256(wer_manifest_path)
    scheduler_name = str(train_cfg.get("lr_scheduler", "constant")).strip().lower()
    require_scheduler = scheduler_name == "constant" or (
        scheduler_name == "linear" and warmup_steps > 0
    )
    launch_distributed = not single_gpu and "RANK" in os.environ and "WORLD_SIZE" in os.environ
    launch_world_size = int(os.environ["WORLD_SIZE"]) if launch_distributed else 1
    resume_state: Optional[Dict[str, Any]] = None
    resume_logged_step = 0
    if not probe_only and resume_from_checkpoint:
        try:
            resume_state = assert_resume_checkpoint(
                resume_from_checkpoint,
                manifest_sha256=manifest_sha,
                val_manifest_sha256=val_manifest_sha,
                wer_manifest_sha256=wer_manifest_sha,
                world_size=launch_world_size,
                model_revision=str(config.get("model", {}).get("model_revision", "") or ""),
                scheduler_name=scheduler_name,
                require_scheduler=require_scheduler,
                seed=seed,
                gradient_accumulation_steps=grad_accum_steps,
            )
            resume_logged_step = int(resume_state.get("global_step") or 0)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            abort_preflight(f"resume checkpoint rejected: {exc}")
    if not probe_only and not resume_from_checkpoint:
        existing_training_artifacts = [
            output_dir / "loss_log.jsonl",
            output_dir / "rollouts.jsonl",
            output_dir / "rollouts_rank_0.jsonl",
            output_dir / "checkpoints",
        ]
        if any(path.is_file() and path.stat().st_size > 0 for path in existing_training_artifacts[:3]) or (
            existing_training_artifacts[3].is_dir()
            and any(existing_training_artifacts[3].iterdir())
        ):
            abort_preflight(
                "output directory already contains RL training artifacts; "
                "pass --resume-from-checkpoint to continue instead of starting a fresh run"
            )
    if not probe_only and resume_from_checkpoint:
        try:
            truncate_summary = truncate_resume_artifacts(
                output_dir,
                resume_logged_step,
                launch_world_size,
            )
        except (OSError, ValueError) as exc:
            abort_preflight(f"resume artifact truncate failed: {exc}")
        if int(os.environ.get("RANK", "0")) == 0 and (
            truncate_summary["dropped_loss_rows"] or truncate_summary["dropped_rollout_rows"]
        ):
            print(
                "Truncated uncommitted RL artifacts: "
                + json.dumps(truncate_summary, sort_keys=True),
                flush=True,
            )
    probe_info: Optional[Dict[str, Any]] = None
    if not probe_only:
        if probe_decision_path is None or not Path(probe_decision_path).is_file():
            abort_preflight(
                f"missing --probe-decision ({probe_decision_path}); refusing to construct an optimizer"
            )
        try:
            probe_payload = json.loads(Path(probe_decision_path).read_text(encoding="utf-8"))
            probe_info = accept_probe_decision(probe_payload, manifest_sha)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            abort_preflight(f"probe gate rejected: {exc}")

    rank = 0
    world_size = 1
    local_rank = 0
    is_distributed = False
    if launch_distributed:
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=2))
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        is_distributed = True
        device = f"cuda:{local_rank}"
    else:
        device = "cuda:0" if HAVE_TORCH and torch.cuda.is_available() else "cpu"

    random.seed(seed + rank)
    np.random.seed(seed + rank)
    if HAVE_TORCH:
        torch.manual_seed(seed + rank)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + rank)

    if probe_only:
        if rank == 0:
            output_dir.mkdir(parents=True, exist_ok=True)
        if is_distributed:
            dist.barrier()
        try:
            execute_search_probe(
                config,
                train_dataset,
                manifest_path,
                output_dir,
                seed=seed,
                probe_prompts=int(probe_prompts),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_improvement=min_improvement,
                max_generate_batch=max_generate_batch,
                second_round_size=second_sample_size,
                group_size=group_size,
                reward_cfg=reward_cfg,
                rank=rank,
                world_size=world_size,
                is_distributed=is_distributed,
                device=device,
            )
        finally:
            if is_distributed and dist.is_initialized():
                dist.destroy_process_group()
        return

    assert probe_info is not None
    asr_model, lora_model, lora_meta, model_id, revision = load_asr_for_rl(
        config,
        device,
        resume_from_checkpoint=resume_from_checkpoint,
        rank=rank,
    )
    if resume_state is not None:
        saved_hash = str(resume_state.get("target_map_hash") or "")
        if saved_hash and saved_hash != str(lora_meta["target_map_hash"]):
            abort_preflight(
                "resume LoRA target hash does not match this config: "
                f"{saved_hash} != {lora_meta['target_map_hash']}"
            )
    if is_distributed:
        ddp_model = DDP(
            lora_model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    else:
        ddp_model = lora_model
    eval_underlying_model = ddp_model.module if hasattr(ddp_model, "module") else ddp_model

    optimizer = torch.optim.AdamW(
        [param for param in ddp_model.parameters() if param.requires_grad],
        lr=lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    schedule_name = str(train_cfg.get("lr_scheduler", "constant")).strip().lower()
    if schedule_name == "constant":
        if get_constant_schedule_with_warmup is None:
            raise RuntimeError("Constant LR schedule requires transformers.")
        scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    elif schedule_name == "linear":
        # Horizon, not the chunk end. A 4-step chunk must not finish a linear decay.
        if warmup_steps > 0:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=horizon,
            )
        else:
            scheduler = None
    else:
        raise ValueError(f"Unsupported lr_scheduler: {schedule_name}")

    start_step = 0
    resume_state_dose: Optional[Dict[str, Any]] = None
    resume_futility_streak = 0
    if resume_from_checkpoint and resume_from_checkpoint.is_dir():
        state_file = resume_from_checkpoint / "training_state.json"
        if state_file.is_file():
            with open(state_file, "r", encoding="utf-8") as state_handle:
                state_data = json.load(state_handle)
            start_step = int(state_data.get("global_step", 0))
            if "cumulative_reward_mass" in state_data:
                resume_state_dose = state_data
            resume_futility_streak = int(state_data.get("consecutive_futile_evals", 0))
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

    if bool(train_cfg.get("apply_learning_rate_on_resume", False)):
        apply_configured_learning_rate(optimizer, scheduler, lr)
        if rank == 0:
            print(f"configured_learning_rate={lr:.2e}")

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)
    if is_distributed:
        dist.barrier()

    loss_log_path = output_dir / "loss_log.jsonl"
    logged_dose = load_run_dose(loss_log_path if loss_log_path.is_file() else [])
    if resume_state_dose is not None and rank == 0:
        saved_mass = resume_state_dose.get("cumulative_reward_mass")
        if saved_mass is not None and abs(float(saved_mass) - float(logged_dose["cumulative_reward_mass"])) > 1e-6:
            print(
                "WARNING: training_state cumulative_reward_mass disagrees with loss_log.jsonl; using the log.",
                file=sys.stderr,
            )
    if rank == 0:
        probe_sha = compute_file_sha256(probe_decision_path) if probe_decision_path else ""
        with open(output_dir / "resolved_config.yaml", "w", encoding="utf-8") as handle:
            yaml.dump(config, handle)
        with open(output_dir / "environment.json", "w", encoding="utf-8") as handle:
            json.dump(get_environment_info(), handle, indent=2)
        with open(output_dir / "manifest_sha256.json", "w", encoding="utf-8") as handle:
            json.dump({
                "manifest": str(manifest_path),
                "sha256": manifest_sha,
                "samples_count": len(train_dataset),
                "audio_sha256_checked": bool(disjoint["audio_sha256_checked"]),
                "val_manifest": str(val_manifest_path),
                "wer_manifest": str(wer_manifest_path),
                "probe_decision": "" if probe_decision_path is None else str(probe_decision_path),
                "probe_sha256": probe_sha,
                "probe_decision_value": probe_info["decision"],
                "projected_train_reward_mass": probe_info["projected_train_reward_mass"],
                "update_rate_k11": probe_info["update_rate_k11"],
                "median_winning_gap_k11": probe_info["median_winning_gap_k11"],
            }, handle, indent=2)
        print(
            f"Loaded {len(train_dataset)} training samples "
            f"(sample_strategy='{sample_strategy}', virtual_epoch_len={virtual_epoch_len})"
        )
        if sample_strategy == "degraded_balanced":
            print(
                "degraded_balanced_summary="
                + json.dumps(
                    {
                        **degraded_balanced_summary(
                            train_dataset.samples,
                            epoch_0_indices,
                            virtual_epoch_rows=int(
                                sample_strategy_options.get("virtual_epoch_rows", 2000)
                            ),
                            scenario_weights=sample_strategy_options.get("scenario_weights"),
                        ),
                        "manifest_sha256": manifest_sha,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
        print(
            f"Starting GRPO training: global_step={start_step} -> chunk_end={chunk_end} "
            f"horizon={horizon}, accum={grad_accum_steps}, world_size={world_size}, G={group_size}, "
            f"policy_token_mask={policy_token_mask} "
            f"include_reference_candidate={include_reference_candidate} "
            f"stop_profile={stop_profile} stop_thresholds={json.dumps(STOP_PROFILES[stop_profile], sort_keys=True)}"
        )

    global_step = start_step
    optimizer.zero_grad()
    accum_count = 0
    accum_policy_loss = 0.0
    accum_kl_loss = 0.0
    accum_raw_kl = 0.0
    accum_total_loss = 0.0
    step_groups = 0
    step_winners = 0.0
    step_mass = 0.0
    step_keep_sum = 0.0
    step_keep_count = 0.0
    step_identical = 0.0
    step_no_improvement = 0.0
    step_second = 0.0
    step_first_sum = 0.0
    step_first_count = 0.0
    step_greedy_sum = 0.0
    step_greedy_count = 0.0
    step_gaps: List[float] = []
    step_start_time = time.time()
    existing_logs = _read_jsonl(loss_log_path)
    if stop_profile == "scale":
        consecutive_futile_evals = scale_futility_streak(existing_logs, profile=stop_profile)
        if resume_futility_streak != consecutive_futile_evals and rank == 0:
            print(
                "WARNING: checkpoint futility streak "
                f"{resume_futility_streak} != loss log {consecutive_futile_evals}; using the log.",
                file=sys.stderr,
            )
    else:
        consecutive_futile_evals = 0
    step0_reward = locked_step0_greedy_reward(existing_logs)
    has_step0 = any(
        row.get("global_step") == 0
        and row.get("val_eval_scope") == "Full Held-out"
        and row.get("val_decode") == "greedy"
        for row in existing_logs
    )
    best_val_reward = step0_reward
    best_saved_step: Optional[int] = None
    best_saved_reward: Optional[float] = None
    stop_status = ""
    latest_dose: Dict[str, Any] = logged_dose
    val_manifest_rows = len(val_dataset)
    epoch_cache: Dict[int, List[int]] = {0: epoch_0_indices}
    probe_rate = float(probe_info["update_rate_k11"])
    probe_gap = float(probe_info["median_winning_gap_k11"])
    mod = 2**31 - 1

    def epoch_indices(epoch: int) -> List[int]:
        if epoch not in epoch_cache:
            epoch_cache[epoch] = build_epoch_sample_indices(
                train_dataset,
                strategy=sample_strategy,
                epoch=epoch,
                seed=seed,
                strategy_options=sample_strategy_options,
            )
        return epoch_cache[epoch]

    def length2_loss(wav: Any, prompt: str, texts: Sequence[str], advantages: Sequence[float], apply_update: bool) -> Dict[str, Any]:
        if len(getattr(wav, "shape", [])) > 1:
            wav = wav[:, 0]
        pad_token_id = asr_model.processor.tokenizer.pad_token_id or 0
        rows = []
        for cand_text in texts:
            target_str = cand_text + "<|im_end|>"
            full_text = prompt + target_str
            proc_out = asr_model.processor(text=full_text, audio=wav, return_tensors="pt")
            target_ids = asr_model.processor.tokenizer.encode(target_str, add_special_tokens=False)
            in_ids = proc_out["input_ids"]
            attn_m = proc_out["attention_mask"]
            labels = in_ids.clone()
            target_len = len(target_ids)
            if target_len >= in_ids.shape[1]:
                labels[:, 0] = -100
            else:
                labels[:, :-target_len] = -100
            rows.append((in_ids, attn_m, labels, target_ids))
        feat_src = asr_model.processor(text=prompt, audio=wav, return_tensors="pt")
        feat = feat_src["input_features"]
        feat_mask = feat_src["feature_attention_mask"]
        if str(device).startswith("cuda"):
            feat = feat.to(torch.float16)
        max_in_len = max(item[0].shape[1] for item in rows)
        padded_ids = []
        padded_masks = []
        padded_labels = []
        for in_ids, attn_m, labels, _target_ids in rows:
            pad_len = max_in_len - in_ids.shape[1]
            if pad_len > 0:
                in_ids = F.pad(in_ids, (0, pad_len), value=pad_token_id)
                attn_m = F.pad(attn_m, (0, pad_len), value=0)
                labels = F.pad(labels, (0, pad_len), value=-100)
            padded_ids.append(in_ids)
            padded_masks.append(attn_m)
            padded_labels.append(labels)
        batch_input_ids = torch.cat(padded_ids, dim=0).to(device)
        batch_attention_mask = torch.cat(padded_masks, dim=0).to(device)
        batch_labels = torch.cat(padded_labels, dim=0).to(device)
        batch_input_features = feat.repeat(len(texts), 1, 1).to(device)
        batch_feature_attention_mask = feat_mask.repeat(len(texts), 1).to(device)
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
        ddp_model.train()
        policy_outputs = ddp_model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_mask,
            input_features=batch_input_features,
            feature_attention_mask=batch_feature_attention_mask,
        )
        policy_token_logps, _ = compute_token_logps(policy_outputs.logits, batch_labels)
        policy_keep = None
        keep_ratio = None
        if policy_token_mask == "changes_only":
            if len(rows) != 2:
                raise RuntimeError("changes_only loss expects 2 sequences")
            anchor_ids = list(rows[0][3])
            winner_ids = list(rows[1][3])
            winner_keep = changed_token_mask(anchor_ids, winner_ids)
            if apply_update and (not winner_keep or not any(winner_keep)):
                raise RuntimeError("changes_only update has no changed response tokens")
            if apply_update:
                keep_ratio = policy_keep_ratio(winner_keep)
            per_row = ([1.0] * len(anchor_ids), winner_keep)
            scattered = [
                scatter_keep(labels.view(-1).tolist(), keep)
                for labels, keep in zip(padded_labels, per_row)
            ]
            policy_keep = torch.tensor(
                scattered,
                device=policy_token_logps.device,
                dtype=policy_token_logps.dtype,
            )
        elif policy_token_mask == "signed_edits":
            if len(rows) != 2:
                raise RuntimeError("signed_edits loss expects 2 sequences")
            if apply_update:
                anchor_ids = list(rows[0][3])
                winner_ids = list(rows[1][3])
                signed = signed_edit_loss_args(
                    anchor_ids, winner_ids, float(advantages[1])
                )
                if signed.action != "update":
                    apply_update = False
                    keep_ratio = None
                else:
                    advantages = [
                        signed.sequence_advantage[0],
                        signed.sequence_advantage[1],
                    ]
                    keep_ratio = signed.winner_keep_ratio
                    scattered = [
                        scatter_keep(labels.view(-1).tolist(), keep)
                        for labels, keep in zip(
                            padded_labels,
                            (list(signed.anchor_delete), list(signed.winner_keep)),
                        )
                    ]
                    policy_keep = torch.tensor(
                        scattered,
                        device=policy_token_logps.device,
                        dtype=policy_token_logps.dtype,
                    )
        parts = compute_grpo_group_loss(
            policy_token_logps,
            ref_token_logps,
            token_mask,
            advantages,
            beta=beta,
            zero_variance=not apply_update,
            reduction=loss_reduction,
            policy_keep=policy_keep,
        )
        parts["policy_keep_ratio"] = keep_ratio
        parts["applied_update"] = bool(apply_update)
        return parts

    def write_pipeline(status: str) -> None:
        if rank != 0:
            return
        ckpt = output_dir / "checkpoints" / f"step_{global_step}"
        if best_saved_step is not None:
            best = str(output_dir / "checkpoints" / f"step_{best_saved_step}")
        elif ckpt.is_dir():
            best = str(ckpt)
        else:
            best = ""
        payload = {
            "stage": "rl_pilot",
            "global_step": global_step,
            "world_size": world_size,
            "last_valid_checkpoint": str(ckpt) if ckpt.is_dir() else "",
            "best_checkpoint": best,
            "best_val_reward": None if best_saved_reward is None else round(float(best_saved_reward), 4),
            "step0_val_reward": None if step0_reward is None else round(float(step0_reward), 4),
            "status": status,
            "promoted": False,
            "horizon": horizon,
            "chunk_end": chunk_end,
            "cumulative_reward_mass": latest_dose.get("cumulative_reward_mass"),
            "cumulative_winners": latest_dose.get("cumulative_winners"),
            "consecutive_collapsed": latest_dose.get("consecutive_collapsed"),
            "consecutive_futile_evals": int(consecutive_futile_evals),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        temporary = output_dir / "pipeline_state.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, output_dir / "pipeline_state.json")

    held_out_scope = "Full Held-out" if val_eval_limit is None else "Truncated Held-out"
    if val_dataset is not None and start_step == 0 and not has_step0:
        if rank == 0:
            print(
                f"Evaluating Step 0 greedy baseline "
                f"({held_out_scope}, {val_manifest_rows} manifest rows)..."
            )
        val_rew0, val_err0, val_rollouts0, val_assigned0 = evaluate_rl_validation(
            ddp_model,
            asr_model,
            val_dataset,
            device,
            group_size=group_size,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_eval_samples=val_eval_limit,
            reward_config=reward_cfg,
            eval_seed=eval_seed,
            decode_mode="greedy",
        )
        if held_out_scope == "Full Held-out":
            step0_reward = round(float(val_rew0), 4)
            best_val_reward = step0_reward
        if rank == 0:
            print(
                f"--- [Step 0 Reference Baseline ({held_out_scope})] val_decode=greedy "
                f"val_mean_reward={val_rew0:.4f}, val_error_rate={val_err0:.4f}, "
                f"rows={val_manifest_rows}, sha256={val_manifest_sha}, "
                f"assigned={val_assigned0}, rollouts={val_rollouts0} ---"
            )
            _append_jsonl(loss_log_path, {
                "global_step": 0,
                "val_mean_reward": round(float(val_rew0), 4),
                "val_error_rate": round(float(val_err0), 4),
                "val_eval_scope": held_out_scope,
                "val_decode": "greedy",
                "decoding": decoding_contract(token_budget),
                "val_manifest_rows": int(val_manifest_rows),
                "val_manifest_sha256": val_manifest_sha,
                "val_assigned_rows": int(val_assigned0),
                "val_rollouts": int(val_rollouts0),
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })
        if is_distributed:
            dist.barrier()
    elif start_step > 0 and step0_reward is None and rank == 0:
        print(
            "WARNING: no greedy Full Held-out step-0 baseline in loss_log.jsonl",
            file=sys.stderr,
        )

    while global_step < chunk_end:
        total_micro_step = global_step * grad_accum_steps + accum_count
        cluster_sample_idx = total_micro_step * world_size + rank
        epoch = cluster_sample_idx // virtual_epoch_len
        in_epoch_idx = cluster_sample_idx % virtual_epoch_len
        sample_idx = epoch_indices(epoch)[in_epoch_idx]
        sample = train_dataset[sample_idx]
        sample_id = str(sample.get("sample_id", f"sample_{sample_idx}"))
        audio_path = str(sample.get("audio") or "")
        lang_raw = str(sample.get("language", "en")).strip().lower()
        gold_text = str(sample.get("text") or sample.get("answer") or "")
        reference_text = gold_text if include_reference_candidate and gold_text else None
        reference_reward = None
        if reference_text is not None:
            reference_reward = compute_sequence_reward(
                reference_text,
                reference_text,
                lang_raw,
                reward_config=reward_cfg,
            )[0]
        micro_index = accum_count
        cand_seed = (seed + rank * 100000 + global_step * 1000 + micro_index * 10) % mod
        micro_error = ""
        round1_done = False
        did_second = False
        wav = None
        prompt = ""
        cand_texts: List[str] = []
        try:
            if not audio_path or not os.path.isfile(audio_path):
                raise FileNotFoundError(f"missing audio: {audio_path}")
            wav, _sr = sf.read(audio_path)
            if len(getattr(wav, "shape", [])) > 1:
                wav = wav[:, 0]
            prompt, inputs_gen, prompt_len = _prepare_prompt_inputs(asr_model, wav, lang_raw, device)
            torch.manual_seed(cand_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cand_seed)

            def generate_once(**gen_kwargs: Any) -> List[str]:
                return _generate_decoded(asr_model, inputs_gen, prompt_len, **gen_kwargs)

            with torch.no_grad():
                eval_underlying_model.eval()
                greedy_texts = generate_once(do_sample=False, num_return_sequences=1)
                if not greedy_texts:
                    raise RuntimeError(f"greedy generate returned no text for {sample_id}")
                sample_count = max(group_size - 1, 0)
                sampled_texts = (
                    generate_once(
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        num_return_sequences=sample_count,
                    )
                    if sample_count
                    else []
                )
                while len(sampled_texts) < sample_count:
                    sampled_texts.append(sampled_texts[-1] if sampled_texts else greedy_texts[0])
                cand_texts = greedy_texts[:1] + sampled_texts[:sample_count]
            round1_done = True
            preview_rewards = [
                compute_sequence_reward(text, gold_text, lang_raw, reward_config=reward_cfg)[0]
                for text in cand_texts
            ]
            _preview_texts, _preview_rewards, _preview_advs, preview_status = select_anchored_training_pair(
                cand_texts,
                preview_rewards,
                anchor_index=0,
                min_improvement=min_improvement,
                advantage_mode=advantage_mode,
                fixed_advantage=fixed_advantage,
                local_max_relative=local_max_relative,
                advantage_cap=advantage_cap,
                language=lang_raw,
                reference_text=reference_text,
                reference_reward=reference_reward,
            )
            if preview_status != "update" and second_enabled:
                round_seed = (cand_seed + 10007) % mod
                torch.manual_seed(round_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(round_seed)

                def generate_n(count: int) -> List[str]:
                    return generate_once(
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        num_return_sequences=count,
                    )

                extra = generate_second_round_texts(
                    generate_n,
                    sample_size=second_sample_size,
                    max_generate_batch=max_generate_batch,
                    split_first=True,
                )
                cand_texts.extend(extra)
                did_second = True
        except Exception as exc:
            micro_error = str(exc)
            print(f"Generation error on {sample_id}: {exc}", file=sys.stderr)
            if not round1_done:
                cand_texts = []
                did_second = False
                wav = None

        if is_distributed:
            dist.barrier()

        failure = not cand_texts
        reward_comps_list: List[Dict[str, float]] = []
        err_rates: List[float] = []
        rewards: List[float] = []
        rollout_advs: List[float] = []
        winner_index = None
        if failure:
            wav = np.zeros(16000, dtype=np.float32)
            prompt = asr_model._build_text_prompt(context="", force_language="English")
            train_texts = [".", "."]
            train_advs = [0.0, 0.0]
            apply_update = False
            status = "identical"
        else:
            for cand_text in cand_texts:
                rew, comps, err = compute_sequence_reward(
                    cand_text, gold_text, lang_raw, reward_config=reward_cfg,
                )
                rewards.append(float(rew))
                reward_comps_list.append(comps)
                err_rates.append(float(err))
            rollout_advs, _adv_status = compute_anchored_advantages(
                rewards,
                anchor_index=0,
                min_improvement=min_improvement,
            )
            train_texts, train_rewards, train_advs, status = select_anchored_training_pair(
                cand_texts,
                rewards,
                anchor_index=0,
                min_improvement=min_improvement,
                advantage_mode=advantage_mode,
                fixed_advantage=fixed_advantage,
                local_max_relative=local_max_relative,
                advantage_cap=advantage_cap,
                language=lang_raw,
                reference_text=reference_text,
                reference_reward=reference_reward,
            )
            apply_update = status == "update"
            if (
                include_reference_candidate
                and status == "update"
                and gold_text
                and train_texts[1] == gold_text
                and gold_text not in cand_texts
            ):
                ref_reward, ref_comps, ref_err = compute_sequence_reward(
                    gold_text, gold_text, lang_raw, reward_config=reward_cfg,
                )
                cand_texts.append(gold_text)
                rewards.append(float(ref_reward))
                reward_comps_list.append(ref_comps)
                err_rates.append(float(ref_err))
                rollout_advs.append(float(train_advs[1]))
            if status == "update":
                target_text = train_texts[1]
                target_reward = float(train_rewards[1])
                for index, text in enumerate(cand_texts):
                    same_reward = math.isclose(
                        float(rewards[index]), target_reward, rel_tol=0.0, abs_tol=1e-9,
                    )
                    if index != 0 and winner_index is None and text == target_text and same_reward:
                        winner_index = index
        loss_parts = length2_loss(wav, prompt, train_texts, train_advs, apply_update)
        apply_update = bool(loss_parts.get("applied_update", apply_update))
        scaled_loss = loss_parts["group_loss"] / grad_accum_steps
        scaled_loss.backward()
        seq_kls = loss_parts["seq_kl_mean"]
        accum_total_loss += float(loss_parts["group_loss"].item())
        accum_policy_loss += float(loss_parts["policy_loss"].item())
        accum_kl_loss += float(loss_parts["kl_loss"].item())
        accum_raw_kl += float(loss_parts["raw_kl"].item())
        if loss_parts.get("policy_keep_ratio") is not None:
            step_keep_sum += float(loss_parts["policy_keep_ratio"])
            step_keep_count += 1.0
        accum_count += 1
        step_groups += 1
        if not failure:
            if status == "identical":
                step_identical += 1
            elif status == "no_improvement":
                step_no_improvement += 1
            if status == "update" and apply_update:
                step_winners += 1
                gap = float(train_rewards[1]) - float(train_rewards[0])
                step_mass += gap
                step_gaps.append(gap)
            elif status == "update":
                step_identical += 1
            if did_second:
                step_second += 1
            first_round = rewards[:group_size]
            step_first_sum += sum(first_round)
            step_first_count += len(first_round)
            step_greedy_sum += float(rewards[0])
            step_greedy_count += 1
            kl_greedy = round(float(seq_kls[0].item()), 6)
            kl_winner = round(float(seq_kls[1].item()), 6)
            cond_group = sample.get(
                "condition_group",
                "clean" if "clean" in str(sample.get("scenario", "")) else "degraded",
            )
            rollout_records = []
            for group_index, pred_text in enumerate(cand_texts):
                if group_index == 0:
                    kl_value = kl_greedy
                elif winner_index is not None and group_index == winner_index:
                    kl_value = kl_winner
                else:
                    kl_value = 0.0
                rollout_records.append({
                    "sample_id": sample_id,
                    "condition_group": cond_group,
                    "rank": rank,
                    "group_id": f"{sample_id}:{global_step}:{rank}:{accum_count}",
                    "group_size": len(cand_texts),
                    "rollout_rank": group_index,
                    "round": 1 if group_index < group_size else 2,
                    "decode_mode": (
                        "greedy" if group_index == 0
                        else "reference"
                        if (
                            include_reference_candidate
                            and gold_text
                            and pred_text == gold_text
                            and group_index >= group_size
                        )
                        else "sample"
                    ),
                    "trained": bool(apply_update and winner_index is not None and group_index == winner_index),
                    "advantage_status": status,
                    "policy_checkpoint": f"step_{global_step}",
                    "decoding": decoding_contract(asr_model.max_new_tokens),
                    "rollout_seed": cand_seed + group_index,
                    "prediction": pred_text,
                    "language": lang_raw,
                    "reference_error_rate": err_rates[group_index],
                    "reward_components": reward_comps_list[group_index],
                    "reward": rewards[group_index],
                    "kl_to_reference": kl_value,
                    "advantage": round(float(rollout_advs[group_index]), 6),
                    "error": micro_error,
                })
            rank_rollouts_path = output_dir / f"rollouts_rank_{rank}.jsonl"
            with open(rank_rollouts_path, "a", encoding="utf-8") as rollout_handle:
                for record in rollout_records:
                    rollout_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        if accum_count == grad_accum_steps:
            grad_norm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), max_norm=1.0)
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
            mean_raw_kl = accum_raw_kl / grad_accum_steps
            packed_local = [
                float(step_groups),
                float(step_winners),
                float(step_mass),
                float(step_identical),
                float(step_no_improvement),
                float(step_second),
                float(step_first_sum),
                float(step_first_count),
                float(step_greedy_sum),
                float(step_greedy_count),
                float(step_keep_sum),
                float(step_keep_count),
            ]
            gaps_local = list(step_gaps)
            accum_total_loss = 0.0
            accum_policy_loss = 0.0
            accum_kl_loss = 0.0
            accum_raw_kl = 0.0
            accum_count = 0
            step_groups = 0
            step_winners = 0.0
            step_mass = 0.0
            step_keep_sum = 0.0
            step_keep_count = 0.0
            step_identical = 0.0
            step_no_improvement = 0.0
            step_second = 0.0
            step_first_sum = 0.0
            step_first_count = 0.0
            step_greedy_sum = 0.0
            step_greedy_count = 0.0
            step_gaps = []
            if is_distributed:
                packed = torch.tensor(packed_local, device=device, dtype=torch.float64)
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                packed_local = packed.tolist()
                gap_tensor = torch.full((grad_accum_steps,), -1.0, device=device, dtype=torch.float64)
                if gaps_local:
                    gap_tensor[: len(gaps_local)] = torch.tensor(gaps_local, device=device, dtype=torch.float64)
                gathered_gaps = [torch.empty_like(gap_tensor) for _ in range(world_size)]
                dist.all_gather(gathered_gaps, gap_tensor)
                gaps_local = [
                    value
                    for tensor in gathered_gaps
                    for value in tensor.tolist()
                    if value >= 0.0
                ]
                loss_tensor = torch.tensor(
                    [mean_tot_loss, mean_pol_loss, mean_kl, mean_raw_kl],
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                mean_tot_loss = float(loss_tensor[0].item() / world_size)
                mean_pol_loss = float(loss_tensor[1].item() / world_size)
                mean_kl = float(loss_tensor[2].item() / world_size)
                mean_raw_kl = float(loss_tensor[3].item() / world_size)
            reduced = finalize_anchored_step_metrics(
                groups=packed_local[0],
                winners=packed_local[1],
                reward_mass=packed_local[2],
                identical=packed_local[3],
                no_improvement=packed_local[4],
                second_round=packed_local[5],
                first_reward_sum=packed_local[6],
                first_reward_count=packed_local[7],
                greedy_reward_sum=packed_local[8],
                greedy_reward_count=packed_local[9],
                gaps=gaps_local,
            )
            mean_reward = reduced["mean_reward"]
            collapsed = bool(
                mean_reward is not None
                and float(reduced["identical_ratio"]) > 0.80
                and float(mean_reward) < collapse_floor
            )
            greedy_gain: Optional[float] = None
            full_eval = False
            if val_dataset is not None and (global_step % eval_steps == 0 or global_step == chunk_end):
                val_rew, val_err, val_rollouts, val_assigned = evaluate_rl_validation(
                    ddp_model,
                    asr_model,
                    val_dataset,
                    device,
                    group_size=group_size,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_eval_samples=val_eval_limit,
                    reward_config=reward_cfg,
                    eval_seed=eval_seed,
                    decode_mode="greedy",
                )
                logged_val = round(float(val_rew), 4)
                full_eval = val_eval_limit is None
                if full_eval and step0_reward is not None:
                    greedy_gain = logged_val - float(step0_reward)
                if full_eval and (best_val_reward is None or logged_val >= best_val_reward):
                    best_val_reward = logged_val
                will_save = global_step % save_steps == 0 or global_step == chunk_end
                if full_eval and will_save and (best_saved_reward is None or logged_val >= best_saved_reward):
                    best_saved_reward = logged_val
                    best_saved_step = global_step
                if rank == 0:
                    gain_text = "n/a" if greedy_gain is None else f"{greedy_gain:+.4f}"
                    print(
                        f"--- [Validation @ Step {global_step} ({held_out_scope})] val_decode=greedy "
                        f"val_mean_reward={logged_val:.4f} gain={gain_text} "
                        f"rows={val_manifest_rows} sha256={val_manifest_sha} "
                        f"assigned={val_assigned} rollouts={val_rollouts} ---"
                    )
                    _append_jsonl(loss_log_path, {
                        "global_step": global_step,
                        "val_mean_reward": logged_val,
                        "val_error_rate": round(float(val_err), 4),
                        "val_eval_scope": held_out_scope,
                        "val_decode": "greedy",
                        "decoding": decoding_contract(token_budget),
                        "val_manifest_rows": int(val_manifest_rows),
                        "val_manifest_sha256": val_manifest_sha,
                        "val_assigned_rows": int(val_assigned),
                        "val_rollouts": int(val_rollouts),
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    })
            if rank == 0:
                prior = load_run_dose(loss_log_path if loss_log_path.is_file() else [])
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else lr
                reward_text = "n/a" if mean_reward is None else f"{float(mean_reward):.4f}"
                train_row = {
                    "global_step": global_step,
                    "loss": round(mean_tot_loss, 5),
                    "policy_loss": round(mean_pol_loss, 5),
                    "kl_loss": round(mean_kl, 6),
                    "raw_kl": round(mean_raw_kl, 6),
                    "mean_reward": None if mean_reward is None else round(float(mean_reward), 4),
                    "mean_greedy_reward": (
                        None if reduced["mean_greedy_reward"] is None
                        else round(float(reduced["mean_greedy_reward"]), 4)
                    ),
                    "zero_variance_ratio": round(float(reduced["zero_variance_ratio"]), 4),
                    "no_improvement_ratio": round(float(reduced["no_improvement_ratio"]), 4),
                    "update_ratio": round(float(reduced["update_ratio"]), 4),
                    "second_round_ratio": round(float(reduced["second_round_ratio"]), 4),
                    "winners_in_step": _json_count(reduced["winners_in_step"]),
                    "reward_mass_in_step": round(float(reduced["reward_mass_in_step"]), 6),
                    "cumulative_winners": _json_count(
                        float(prior["cumulative_winners"]) + float(reduced["winners_in_step"])
                    ),
                    "cumulative_reward_mass": round(
                        float(prior["cumulative_reward_mass"]) + float(reduced["reward_mass_in_step"]),
                        6,
                    ),
                    "collapsed": collapsed,
                    "median_gap_in_step": reduced["median_gap_in_step"],
                    "grad_norm": round(float(grad_norm), 6),
                    "loss_reduction": loss_reduction,
                    "policy_keep_ratio": (
                        None
                        if packed_local[11] <= 0
                        else round(float(packed_local[10]) / float(packed_local[11]), 4)
                    ),
                    "learning_rate": current_lr,
                    "step_seconds": round(step_duration, 3),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                _append_jsonl(loss_log_path, train_row)
                print(
                    f"[Step {global_step}/{chunk_end}] loss={mean_tot_loss:.4f}, "
                    f"pol_loss={mean_pol_loss:.4f}, raw_kl={mean_raw_kl:.6f}, "
                    f"kl_loss={mean_kl:.6f}, rew={reward_text}, "
                    f"zero_var={float(reduced['zero_variance_ratio']):.2f}, "
                    f"winners_in_step={train_row['winners_in_step']}, "
                    f"cumulative_reward_mass={train_row['cumulative_reward_mass']}, "
                    f"second_round_ratio={float(reduced['second_round_ratio']):.2f}, "
                    f"policy_keep={train_row['policy_keep_ratio']}, "
                    f"grad_norm={float(grad_norm):.4f}, lr={current_lr:.2e}, time={step_duration:.2f}s"
                )
            if is_distributed:
                dist.barrier()
            if full_eval and greedy_gain is not None:
                stop_table = STOP_PROFILES[stop_profile]
                if (
                    stop_profile == "scale"
                    and global_step >= int(stop_table["futility_step"])
                ):
                    if greedy_gain < float(stop_table["futility_min_gain"]):
                        consecutive_futile_evals += 1
                    else:
                        consecutive_futile_evals = 0
                else:
                    consecutive_futile_evals = 0
            if rank == 0:
                latest_dose = load_run_dose(loss_log_path)
                # Fresh raw_kl every optimizer step. Greedy gain only from a full eval just written.
                stop_status = decide_rl_stop(
                    global_step,
                    float(latest_dose["cumulative_reward_mass"]),
                    greedy_gain=greedy_gain,
                    raw_kl=float(mean_raw_kl),
                    robust_increase=None,
                    probe_update_rate=probe_rate,
                    probe_median_gap=probe_gap,
                    identical_ratio=float(reduced["identical_ratio"]),
                    mean_reward=None if mean_reward is None else float(mean_reward),
                    consecutive_collapsed=int(latest_dose["consecutive_collapsed"]),
                    collapse_mean_reward_floor=collapse_floor,
                    cumulative_winners=_json_count(latest_dose["cumulative_winners"]),
                    consecutive_futile_evals=consecutive_futile_evals,
                    profile=stop_profile,
                )
            else:
                stop_status = ""
            stop_status = broadcast_stop_status(
                stop_status,
                device=device,
                is_distributed=is_distributed,
            )
            should_save = bool(stop_status) or global_step % save_steps == 0 or global_step == chunk_end
            if should_save:
                ckpt_dir = output_dir / "checkpoints" / f"step_{global_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                if rank == 0:
                    adapter_dir = ckpt_dir / "adapter"
                    eval_underlying_model.save_pretrained(str(adapter_dir))
                    torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
                    if scheduler is not None:
                        torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
                    state_meta = {
                        "decoding": decoding_contract(token_budget),
                        "global_step": global_step,
                        "status": training_process_status(global_step, horizon, stop_status),
                        "world_size": world_size,
                        "stage": "rl_pilot",
                        "model_id": model_id,
                        "model_revision": revision,
                        "target_map_hash": lora_meta["target_map_hash"],
                        "manifest": str(manifest_path),
                        "manifest_sha256": manifest_sha,
                        "val_manifest_sha256": val_manifest_sha,
                        "wer_manifest_sha256": wer_manifest_sha,
                        "seed": seed,
                        "gradient_accumulation_steps": grad_accum_steps,
                        "scheduler": schedule_name if scheduler is not None else None,
                        "cumulative_reward_mass": latest_dose.get("cumulative_reward_mass"),
                        "cumulative_winners": latest_dose.get("cumulative_winners"),
                        "consecutive_collapsed": latest_dose.get("consecutive_collapsed"),
                        "consecutive_futile_evals": int(consecutive_futile_evals),
                        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    }
                    state_tmp = ckpt_dir / "training_state.json.tmp"
                    state_tmp.write_text(
                        json.dumps(state_meta, indent=2),
                        encoding="utf-8",
                    )
                    os.replace(state_tmp, ckpt_dir / "training_state.json")
                    with open(ckpt_dir / "resolved_config.yaml", "w", encoding="utf-8") as config_handle:
                        yaml.safe_dump(config, config_handle)
                if is_distributed:
                    dist.barrier()
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
                    write_pipeline(training_process_status(global_step, horizon, stop_status))
                    print(f"Checkpoint saved to {ckpt_dir}")
                    try:
                        merge_and_audit_rollouts(output_dir, world_size=world_size)
                    except Exception as merge_error:
                        print(f"Warning: Rollouts merge on checkpoint: {merge_error}", file=sys.stderr)
            if stop_status:
                if rank == 0:
                    print(f"Stopping at step {global_step}: status={stop_status}")
                break

    status = training_process_status(global_step, horizon, stop_status)
    if rank == 0:
        write_pipeline(status)
        try:
            audit_res = merge_and_audit_rollouts(output_dir, world_size=world_size)
            print(
                f"All rollouts merged and audited: {audit_res['total_rows']} rows "
                f"across ranks {audit_res['ranks_represented']}"
            )
        except Exception as merge_error:
            print(f"Warning: Final rollouts merge/audit failed: {merge_error}", file=sys.stderr)
    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        if allow_in_process_merged_export(global_step, horizon, stop_status, export_merged_on_finish):
            export_ckpt = output_dir / "checkpoints" / f"step_{global_step}"
            merged_dir = output_dir / "merged_base"
            if export_ckpt.is_dir():
                print(f"Exporting merged model from {export_ckpt} to {merged_dir}...")
                try:
                    export_merged_model(export_ckpt, merged_dir, config)
                    print(f"Exported merged model successfully to {merged_dir}")
                except Exception as export_error:
                    print(f"Warning: Failed to auto-export merged model: {export_error}", file=sys.stderr)
            else:
                print(f"Warning: skip in-process export, missing {export_ckpt}", file=sys.stderr)
        print(f"GRPO RL Training finished status={status} global_step={global_step}")


def build_parser() -> argparse.ArgumentParser:
    """Build command line argument parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=None, help="Path to input RL manifest (JSONL).")
    parser.add_argument("--val-manifest", default=None, help="Path to validation manifest (JSONL).")
    parser.add_argument(
        "--wer-manifest",
        default=None,
        help="Path to the frozen 2,867-row validation.jsonl used only for the leakage check.",
    )
    parser.add_argument("--config", default=None, help="Path to training config YAML.")
    parser.add_argument("--output-dir", default=None, help="Directory to save checkpoints and logs.")
    parser.add_argument("--resume-from-checkpoint", default=None, help="Path to step checkpoint directory to resume.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Chunk end for this process. Does not replace YAML train.max_steps (the horizon).",
    )
    parser.add_argument("--save-steps", type=int, default=None, help="Override checkpoint saving frequency.")
    parser.add_argument("--eval-steps", type=int, default=None, help="Override validation evaluation frequency.")
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature for rollouts.")
    parser.add_argument("--top-p", type=float, default=None, help="Sampling top-p for rollouts.")
    parser.add_argument("--top-k", type=int, default=None, help="Sampling top-k for rollouts.")
    parser.add_argument("--zero-variance-thresh", type=float, default=None, help="Logged zero-variance ratio threshold.")
    parser.add_argument("--single-gpu", action="store_true", help="Run on single GPU without DDP.")
    parser.add_argument("--allow-subset", action="store_true", help="Allow NON_STRICT_SUBSET dataset gate status.")
    parser.add_argument(
        "--val-eval-samples",
        type=int,
        default=None,
        help="Truncate held-out eval to this many rows. Omit it for the full pool.",
    )
    parser.add_argument("--reward-config", default=None, help="Path to reward_config.yaml.")
    parser.add_argument("--eval-seed", type=int, default=42, help="Deterministic seed for validation evaluation.")
    parser.add_argument(
        "--sample-strategy",
        default=None,
        choices=["balanced", "standard", "degraded", "degraded_skip_regressed", "degraded_balanced"],
        help="Sampling strategy. Empty lets YAML train.sample_strategy win, else balanced.",
    )
    parser.add_argument(
        "--probe-decision",
        default=None,
        help="Path to search_probe.json. Required before the optimizer unless --probe-only.",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Run the 128-prompt search probe and exit. Does not construct an optimizer.",
    )
    parser.add_argument("--probe-prompts", type=int, default=128, help="Degraded prompts in --probe-only.")
    parser.add_argument(
        "--export-merged-on-finish",
        action="store_true",
        default=True,
        help="Export merged weights only when global_step reaches the YAML horizon and nothing stopped.",
    )
    parser.add_argument(
        "--no-export-merged-on-finish",
        dest="export_merged_on_finish",
        action="store_false",
        help="Do not auto-export a merged model when this process exits.",
    )
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
            with open(cfg_file, "r", encoding="utf-8") as handle:
                state = json.load(handle)
            config["model"] = {
                "model_id": state.get("model_id", "Qwen/Qwen3-ASR-1.7B"),
                "model_revision": state.get("model_revision", "7278e1e70fe206f11671096ffdd38061171dd6e5"),
            }
        elif args.config:
            with open(args.config, "r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
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
        reward_config_path=Path(args.reward_config).resolve() if args.reward_config else None,
        eval_seed=args.eval_seed,
        sample_strategy=args.sample_strategy,
        export_merged_on_finish=args.export_merged_on_finish,
        wer_manifest_path=Path(args.wer_manifest).resolve() if args.wer_manifest else None,
        probe_decision_path=Path(args.probe_decision).resolve() if args.probe_decision else None,
        probe_only=bool(args.probe_only),
        probe_prompts=int(args.probe_prompts),
    )


if __name__ == "__main__":
    main()
