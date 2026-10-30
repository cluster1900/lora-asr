#!/usr/bin/env python3
"""SFT runner for Qwen3-ASR-0.6B decoder-attention LoRA.

The loss is the sum of supervised-token NLL divided by the supervised token
count. ``--resume`` loads only the checkpoint named by ``latest.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
import wave
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.eval_cer import evaluate, sha256_file, write_scenario_csv  # noqa: E402
from evaluation.jsonl_io import read_jsonl  # noqa: E402
from inference.run_inference import (  # noqa: E402
    ASR_TEXT_TAG,
    parse_generation,
    stop_reason_from_ids,
)
from train.sft_checkpoint import (  # noqa: E402
    atomic_write_json,
    atomic_write_text,
    commit_checkpoint,
    prune_old_checkpoints,
    read_latest,
    step_name,
    utc_now,
    verify_checkpoint,
    write_checksums,
    write_latest,
)
from train.sft_contract import (  # noqa: E402
    PROCESSOR_FILES,
    assert_shift_supervises_transcript,
    assert_trainable_names,
    is_trainable_lora_parameter,
    iterate_updates,
    judge_sft_checkpoint,
    mask_labels,
    partial_decode_stop_record,
    pending_checkpoint_eval,
    processor_tree_sha256,
    runtime_budget_disabled,
    runtime_budget_exceeded,
    select_sft_checkpoint,
    should_interrupt,
)
from train.sft_schema import load_train_config, semantic_config_sha256  # noqa: E402

RUN_ID_PREFIX = "_sft_06b_zh_seed"
LORA_TARGET_REGEX = r"^thinker\.model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)$"


class RunStop(Exception):
    """A contract failure that already has a status name and exit code."""

    def __init__(self, status: str, message: str, code: int) -> None:
        super().__init__(message)
        self.status = status
        self.code = code


class EvalRuntimeStop(Exception):
    """Val decode stopped before the last row because the runtime budget was used up."""

    def __init__(self, record: dict[str, Any]):
        super().__init__("eval stopped because the runtime budget was exhausted")
        self.record = record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练 Qwen3-ASR-0.6B 的 SFT LoRA")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--val-manifest", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--resume", action="store_true")
    return parser


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def log_event(run_dir: Path, event: str, **payload: Any) -> None:
    append_jsonl(run_dir / "events.jsonl", {"time": utc_now(), "event": event, **payload})


def write_status(run_dir: Path, status: str, reason: str, global_step: int) -> None:
    atomic_write_json(
        run_dir / "status.json",
        {"status": status, "reason": reason, "time": utc_now(), "global_step": global_step},
    )


def resolve_from_repo(value: Path) -> Path:
    if value.is_absolute():
        return value
    return (ROOT / value).resolve()


def git_info() -> dict[str, Any]:
    def run(args: list[str]) -> str:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()

    try:
        sha = run(["git", "rev-parse", "HEAD"])
        porcelain = run(["git", "status", "--porcelain"])
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        return {"git_sha": None, "dirty": None, "porcelain": [], "error": str(exc)}
    lines = [line for line in porcelain.splitlines() if line]
    return {"git_sha": sha, "dirty": bool(lines), "porcelain": lines}


def load_audio(path: Path) -> np.ndarray:
    """Read one 16 kHz mono PCM16 WAV into float32 samples."""
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        rate = handle.getframerate()
        frames = handle.readframes(handle.getnframes())
    if channels != 1 or width != 2 or rate != 16000:
        raise ValueError(f"audio must be 16 kHz mono PCM16: {path}")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if samples.size == 0:
        raise ValueError(f"audio is empty: {path}")
    return np.ascontiguousarray(samples)


def prefix_text(processor: Any, context: str, official_language: str) -> str:
    """Use the same prefix as greedy inference."""
    messages = [
        {"role": "system", "content": context or ""},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    if isinstance(prompt, list):
        prompt = prompt[0]
    return str(prompt) + f"language {official_language}{ASR_TEXT_TAG}"


def move_batch(batch: dict[str, Any], device: Any, dtype: Any) -> dict[str, Any]:
    import torch

    moved = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            moved[key] = value
            continue
        if value.is_floating_point():
            moved[key] = value.to(device=device, dtype=dtype)
        else:
            moved[key] = value.to(device=device)
    return moved


def collate_example(
    processor: Any,
    row: dict[str, Any],
    audio: np.ndarray,
    *,
    prefix: str,
    eos_token: str,
    eos_id: int,
    max_input_tokens: int,
    max_transcript_tokens: int,
) -> dict[str, Any]:
    """Build one supervised example. Padding is masked without erasing a real EOS."""
    import torch

    target = str(row["text"])
    full_inputs = processor(
        text=[prefix + target + eos_token],
        audio=[audio],
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    prefix_inputs = processor(
        text=[prefix],
        audio=[audio],
        return_tensors="pt",
        padding=True,
        truncation=False,
    )
    prefix_len = int(prefix_inputs["attention_mask"].sum().item())
    input_ids = [int(token) for token in full_inputs["input_ids"][0].tolist()]
    attention_mask = [int(bit) for bit in full_inputs["attention_mask"][0].tolist()]
    prefix_ids = [int(token) for token in prefix_inputs["input_ids"][0][:prefix_len].tolist()]
    if prefix_ids != input_ids[:prefix_len]:
        raise ValueError(f"prefix tokens do not align for {row.get('sample_id')}")
    if len(input_ids) > max_input_tokens:
        raise ValueError(f"input longer than max_input_tokens for {row.get('sample_id')}")
    labels = mask_labels(input_ids, attention_mask, prefix_len)
    assert_shift_supervises_transcript(labels, prefix_len, eos_id)
    supervised = sum(1 for token in labels if token != -100)
    if supervised > max_transcript_tokens + 1:
        raise ValueError(f"transcript longer than max_transcript_tokens for {row.get('sample_id')}")
    full_inputs["labels"] = torch.tensor([labels], dtype=torch.long)
    full_inputs["prefix_len"] = prefix_len
    full_inputs["supervised_tokens"] = supervised
    return full_inputs


def supervised_nll_sum(logits: Any, labels: Any) -> tuple[Any, int]:
    """Return the sum of shifted NLL and the supervised token count."""
    import torch

    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    nll = torch.nn.functional.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    )
    mask = shift_labels.reshape(-1) != -100
    count = int(mask.sum().item())
    if count == 0:
        raise ValueError("microbatch has no supervised tokens")
    return nll[mask].sum(), count


def set_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trainable_report(model: Any) -> dict[str, Any]:
    names = []
    rows = []
    trainable = 0
    total = 0
    for name, param in model.named_parameters():
        count = int(param.numel())
        total += count
        param.requires_grad = is_trainable_lora_parameter(name)
        if param.requires_grad:
            trainable += count
            names.append(name)
            rows.append({"name": name, "shape": list(param.shape), "numel": count})
    accepted = assert_trainable_names(names)
    for name, param in model.named_parameters():
        if "lora_B" in name and param.detach().abs().sum().item() != 0:
            raise RunStop("failed", f"LoRA B is not zero: {name}", 2)
    return {
        "schema_version": "qwen3-asr-trainable-v1",
        "parameters": rows,
        "trainable_names": accepted,
        "trainable_numel": trainable,
        "total_numel": total,
        "trainable_fraction": (trainable / total) if total else 0.0,
    }


def apply_lora(model: Any, config: dict[str, Any]) -> dict[str, Any]:
    from peft import LoraConfig, inject_adapter_in_model

    inject_adapter_in_model(
        LoraConfig(
            r=int(config["lora"]["rank"]),
            lora_alpha=int(config["lora"]["alpha"]),
            lora_dropout=float(config["lora"]["dropout"]),
            target_modules=LORA_TARGET_REGEX,
            bias="none",
        ),
        model,
    )
    try:
        report = trainable_report(model)
    except ValueError as exc:
        samples = [name for name, _module in model.named_modules() if name.endswith("q_proj")][:30]
        raise RunStop("failed", f"{exc}; q_proj modules seen: {samples}", 2) from exc
    if not report["parameters"]:
        samples = [name for name, _module in model.named_modules() if name.endswith("q_proj")][:30]
        raise RunStop("failed", f"no decoder LoRA parameters matched. q_proj modules: {samples}", 2)
    return report


def enable_training_mode(model: Any) -> None:
    import torch

    model.thinker.config.use_cache = False
    if hasattr(model, "config"):
        model.config.use_cache = False
    embeddings = model.thinker.get_input_embeddings()

    def mark_grad(_module: Any, _inputs: Any, output: Any) -> Any:
        if torch.is_tensor(output):
            output.requires_grad_(True)
        return output

    embeddings.register_forward_hook(mark_grad)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


def resource_report(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RunStop("blocked_dtype", "SFT requires a BF16 CUDA GPU", 2)
    free_gpu, total_gpu = torch.cuda.mem_get_info()
    if free_gpu / total_gpu < float(config["runtime"]["min_free_gpu_fraction"]):
        raise RunStop("failed", "free GPU memory is below min_free_gpu_fraction", 2)
    disk = shutil.disk_usage(run_dir)
    free_disk_gb = disk.free / (1024 ** 3)
    if free_disk_gb < float(config["runtime"]["min_free_disk_gb"]):
        raise RunStop("failed", "free disk is below min_free_disk_gb", 2)
    meminfo = Path("/proc/meminfo")
    if not meminfo.is_file():
        raise RunStop("failed", "MemAvailable cannot be read from /proc/meminfo", 2)
    available_kb = None
    for line in meminfo.read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            available_kb = int(line.split()[1])
    if available_kb is None or available_kb / (1024 ** 2) < float(config["runtime"]["min_free_ram_gb"]):
        raise RunStop("failed", "free RAM is below min_free_ram_gb", 2)
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_bytes": int(total_gpu),
        "gpu_free_bytes": int(free_gpu),
        "free_disk_gb": free_disk_gb,
        "free_ram_gb": available_kb / (1024 ** 2),
        "dtype": "bfloat16",
        "world_size": 1,
    }


def environment_report(config: dict[str, Any]) -> dict[str, Any]:
    import torch

    info = git_info()
    info.update(
        {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "environment_lock_sha256": config["runtime"]["environment_lock_sha256"],
        }
    )
    try:
        import peft
        import transformers

        info["transformers"] = transformers.__version__
        info["peft"] = peft.__version__
    except ImportError:
        pass
    try:
        import qwen_asr

        info["qwen_asr"] = getattr(qwen_asr, "__version__", None)
    except ImportError:
        pass
    return info


def save_rng(path: Path) -> None:
    import torch

    torch.save(
        {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        path,
    )


def load_rng(path: Path) -> None:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    if payload.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda"])


def save_adapter(model: Any, path: Path) -> str:
    from safetensors.torch import save_file

    tensors = {
        name: param.detach().cpu().contiguous()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path))
    return sha256_file(path)


def load_adapter(model: Any, path: Path) -> None:
    from safetensors.torch import load_file

    tensors = load_file(str(path))
    named = dict(model.named_parameters())
    expected = {name for name, param in named.items() if param.requires_grad}
    if set(tensors) != expected:
        raise RunStop("resume_rejected", "adapter parameter names do not match the current LoRA map", 2)
    for name, value in tensors.items():
        param = named[name]
        param.data.copy_(value.to(device=param.device, dtype=param.dtype))


class HorizonScheduler:
    """Step-counted warmup plus cosine. Loading state does not restart warmup."""

    def __init__(self, optimizer: Any, config: dict[str, Any]) -> None:
        from train.sft_contract import lr_multiplier

        self._multiplier = lr_multiplier
        self.optimizer = optimizer
        self.warmup = int(config["optim"]["warmup_steps"])
        scheduler_total = config["optim"].get("scheduler_max_steps")
        self.total = int(config["train"]["max_steps"] if scheduler_total is None else scheduler_total)
        self.min_ratio = float(config["optim"]["min_lr_ratio"])
        self.base_lr = float(config["optim"]["learning_rate"])
        self.completed = 0
        self._apply(0)

    def _apply(self, step: int) -> None:
        multiplier = self._multiplier(step, self.warmup, self.total, self.min_ratio)
        for group in self.optimizer.param_groups:
            group["lr"] = self.base_lr * multiplier

    def step(self) -> None:
        self.completed += 1
        self._apply(self.completed)

    def state_dict(self) -> dict[str, Any]:
        return {
            "completed": self.completed,
            "warmup": self.warmup,
            "total": self.total,
            "min_lr_ratio": self.min_ratio,
            "base_lr": self.base_lr,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if (
            int(state["warmup"]) != self.warmup
            or int(state["total"]) != self.total
            or float(state["min_lr_ratio"]) != self.min_ratio
            or float(state["base_lr"]) != self.base_lr
        ):
            raise RunStop("resume_rejected", "scheduler horizon does not match the config", 2)
        self.completed = int(state["completed"])
        self._apply(self.completed)


def current_lr(optimizer: Any) -> float:
    return float(optimizer.param_groups[0]["lr"])


def thinker_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    """Keep the tensors the thinker forward and generate paths both accept."""
    required = ("input_ids", "attention_mask", "input_features", "feature_attention_mask")
    missing = [key for key in required if key not in batch]
    if missing:
        raise RunStop("failed", f"processor batch is missing {missing}", 2)
    return {key: batch[key] for key in required}


def write_run_report(run_dir: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# SFT run",
        "",
        f"- run_id: `{summary['run_id']}`",
        f"- status: `{summary['status']}`",
        f"- reason: `{summary['reason']}`",
        f"- global_step: `{summary['global_step']}`",
        f"- publishable: `{summary['publishable']}`",
        f"- config_sha256: `{summary.get('config_sha256')}`",
        f"- git_sha: `{summary.get('git_sha')}`",
        f"- dirty: `{summary.get('dirty')}`",
        f"- sync: `{summary.get('sync')}`",
        "",
        str(summary.get("note") or "publishable 保持 false。"),
        "",
    ]
    atomic_write_text(run_dir / "run_report.md", "\n".join(lines))


def sync_drive(run_dir: Path, run_id: str, step: int, digest: str) -> str:
    """Copy a committed checkpoint when Drive is already mounted."""
    raw = os.environ.get("QWEN3_ASR_DRIVE_ROOT", "/content/drive/MyDrive/qwen3-asr")
    root = Path(raw)
    if not root.is_dir():
        return "sync_skipped"
    destination = root / "runs" / run_id / "checkpoints" / ".incoming" / step_name(step)
    final = root / "runs" / run_id / "checkpoints" / step_name(step)
    if destination.exists() or final.exists():
        raise RunStop("sync_failed", f"Drive checkpoint already exists for {step_name(step)}", 1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(run_dir / "checkpoints" / step_name(step), destination)
    from train.sft_checkpoint import verify_checkpoint_files

    verify_checkpoint_files(destination, digest)
    os.rename(destination, final)
    verify_checkpoint_files(final, digest)
    write_latest(root / "runs" / run_id / "latest.json", step=step, checkpoint_sha256=digest)
    readback = sha256_file(final / "sha256.json")
    original = sha256_file(run_dir / "checkpoints" / step_name(step) / "sha256.json")
    if readback != original:
        raise RunStop("sync_failed", "Drive readback of sha256.json does not match", 1)
    return "synced"


def sync_drive_files(run_id: str, files: list[tuple[Path, str]]) -> str:
    """Copy existing run files onto Drive. A missing Drive root is sync_skipped."""
    raw = os.environ.get("QWEN3_ASR_DRIVE_ROOT", "/content/drive/MyDrive/qwen3-asr")
    root = Path(raw)
    if not root.is_dir():
        return "sync_skipped"
    for source, relative in files:
        parts = Path(relative).parts
        if Path(relative).is_absolute() or ".." in parts:
            raise RunStop("sync_failed", f"refusing drive path {relative}", 1)
        destination = root / "runs" / run_id / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        if sha256_file(destination) != sha256_file(source):
            raise RunStop("sync_failed", f"Drive readback does not match {relative}", 1)
    return "synced"


def prepare_checkpoint(
    model: Any,
    optimizer: Any,
    scheduler: HorizonScheduler,
    *,
    run_dir: Path,
    config: dict[str, Any],
    state: dict[str, Any],
    sampler_state: dict[str, Any],
    resolved_yaml: str,
) -> str:
    import torch

    staging = run_dir / "checkpoints" / f".tmp-{step_name(state['global_step'])}-{os.getpid()}"
    if staging.exists():
        shutil.rmtree(staging)
    adapter_dir = staging / "adapter"
    adapter_sha = save_adapter(model, adapter_dir / "adapter.safetensors")
    atomic_write_json(
        adapter_dir / "adapter_config.json",
        {
            "rank": config["lora"]["rank"],
            "alpha": config["lora"]["alpha"],
            "dropout": config["lora"]["dropout"],
            "target_modules": LORA_TARGET_REGEX,
            "trainable_names": state["trainable_names"],
        },
    )
    torch.save(optimizer.state_dict(), staging / "optimizer.pt")
    torch.save(scheduler.state_dict(), staging / "scheduler.pt")
    save_rng(staging / "rng_state.pt")
    atomic_write_json(staging / "sampler_state.json", sampler_state)
    state["adapter_sha256"] = adapter_sha
    state["optimizer_step"] = scheduler.completed
    state["scheduler_step"] = scheduler.completed
    atomic_write_json(staging / "training_state.json", state)
    atomic_write_text(staging / "resolved_config.yaml", resolved_yaml)
    atomic_write_json(staging / "resolved_config.json", config)
    atomic_write_json(staging / "manifest_sha256.json", state["manifest_sha256"])
    payload = []
    for path in staging.rglob("*"):
        if path.is_file():
            payload.append(path.relative_to(staging).as_posix())
    digest = write_checksums(staging, payload, utc_now())
    try:
        committed = commit_checkpoint(staging, run_dir, int(state["global_step"]))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    if committed != digest:
        raise RunStop("failed", "committed checkpoint hash changed", 1)
    return digest


def load_sft_model(config: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    """Load the pinned base, check the processor hash, and attach decoder LoRA."""
    import torch
    from huggingface_hub import snapshot_download
    from qwen_asr import Qwen3ASRModel

    snapshot = snapshot_download(
        repo_id=config["model"]["name"],
        revision=config["model"]["revision"],
        allow_patterns=list(PROCESSOR_FILES),
    )
    processor_sha = processor_tree_sha256(Path(snapshot))
    if processor_sha != config["model"]["processor_sha256"]:
        raise RunStop("failed", f"processor_sha256 mismatch: {processor_sha}", 2)
    wrapper = Qwen3ASRModel.from_pretrained(
        config["model"]["name"],
        revision=config["model"]["revision"],
        dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation=config["runtime"]["attention"],
        max_inference_batch_size=1,
        max_new_tokens=int(config["decode"]["eval"]["max_new_tokens"]),
    )
    model = wrapper.model
    loaded_revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    if loaded_revision != config["model"]["revision"]:
        raise RunStop("failed", f"loaded revision {loaded_revision} does not match the config", 2)
    enable_training_mode(model)
    return model, wrapper.processor, apply_lora(model, config)


def decode_eval_rows(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    val_manifest: Path,
    config: dict[str, Any],
    step: int,
    adapter_sha: str,
    deadline: Callable[[], bool] | None = None,
    on_progress: Callable[[int, int], None] | None = None,
    on_row: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Greedy-decode val rows with the same settings as a training eval."""
    import torch

    from inference.run_inference import resolve_audio_path

    model.eval()
    prefix = prefix_text(processor, config["decode"]["eval"]["context"], config["official_language"])
    eos_id = int(processor.tokenizer.convert_tokens_to_ids("<|im_end|>"))
    predictions = []
    token_lengths: list[int] = []
    max_new_tokens = int(config["decode"]["eval"]["max_new_tokens"])
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for row in rows:
        if deadline is not None and deadline():
            raise EvalRuntimeStop(partial_decode_stop_record(step, len(rows), token_lengths, max_new_tokens))
        sample_id = str(row["sample_id"])
        try:
            audio = load_audio(resolve_audio_path(val_manifest, str(row["audio"])))
            inputs = processor(text=[prefix], audio=[audio], return_tensors="pt", padding=True)
            inputs = thinker_inputs(move_batch(dict(inputs), device, dtype))
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=int(config["decode"]["eval"]["num_beams"]),
                    max_new_tokens=int(config["decode"]["eval"]["max_new_tokens"]),
                    repetition_penalty=float(config["decode"]["eval"]["repetition_penalty"]),
                    eos_token_id=eos_id,
                )
            sequences = generated.sequences if hasattr(generated, "sequences") else generated
            prompt_width = int(inputs["input_ids"].shape[1])
            token_ids = [int(token) for token in sequences[0, prompt_width:].detach().cpu().tolist()]
            if token_ids and token_ids[-1] == eos_id:
                kept = token_ids
            else:
                kept = token_ids
            reason = stop_reason_from_ids(kept, eos_id, int(config["decode"]["eval"]["max_new_tokens"]))
            visible = processor.batch_decode(
                torch.tensor([kept], dtype=torch.long),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            generation_raw = processor.batch_decode(
                torch.tensor([kept], dtype=torch.long),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )[0]
            prediction = {
                "sample_id": sample_id,
                "text": row["text"],
                "prediction_raw": parse_generation(visible),
                "generation_raw": generation_raw,
                "token_ids": kept,
                "stop_reason": reason,
                "error": None,
                "adapter": adapter_sha,
            }
        except (OSError, RuntimeError, ValueError) as exc:
            prediction = {
                "sample_id": sample_id,
                "text": row.get("text"),
                "prediction_raw": "",
                "generation_raw": None,
                "token_ids": None,
                "stop_reason": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "adapter": adapter_sha,
            }
        predictions.append(prediction)
        if on_row is not None:
            on_row(prediction)
        token_ids = prediction.get("token_ids")
        token_lengths.append(len(token_ids) if isinstance(token_ids, list) else 0)
        done = len(predictions)
        if on_progress is not None and (done == 1 or done % 250 == 0 or done == len(rows)):
            on_progress(done, len(rows))
    return predictions


def evaluate_val(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    val_manifest: Path,
    config: dict[str, Any],
    run_dir: Path,
    step: int,
    adapter_sha: str,
    deadline: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    predictions = decode_eval_rows(
        model,
        processor,
        rows,
        val_manifest,
        config,
        step,
        adapter_sha,
        deadline,
    )
    step_dir = run_dir / "eval" / step_name(step)
    step_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = run_dir / "predictions" / f"{step_name(step)}.jsonl"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    scored, metrics = evaluate(
        rows,
        predictions,
        manifest_sha256=sha256_file(val_manifest),
        predictions_sha256=sha256_file(prediction_path),
        decode_eval_sha256=config["decode_eval_sha256"],
        wer_tokenizer=config["scoring"]["wer_tokenizer"],
        wer_tokenizer_version=config["scoring"].get("wer_tokenizer_version"),
        wer_dictionary_sha256=config["scoring"].get("wer_dictionary_sha256"),
        wer_hmm=bool(config["scoring"].get("wer_hmm", False)),
    )
    with (step_dir / "scored.jsonl").open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    atomic_write_json(step_dir / "metrics.json", metrics)
    write_scenario_csv(step_dir / "by_scenario.csv", metrics["by_scenario"])
    model.train()
    return metrics


def validate_rows(rows: list[dict[str, Any]], split_name: str) -> None:
    seen = set()
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in seen:
            raise RunStop("data_failed", f"duplicate or missing sample_id in {split_name}", 4)
        seen.add(sample_id)
        if row.get("language") != "zh" or not isinstance(row.get("text"), str) or not row["text"].strip():
            raise RunStop("data_failed", f"{sample_id} has no Chinese transcript", 4)
        if not isinstance(row.get("audio"), str):
            raise RunStop("data_failed", f"{sample_id} is missing audio", 4)


def run_training(args: argparse.Namespace, config: dict[str, Any], resume_state: dict[str, Any] | None) -> int:
    import torch

    run_dir = resolve_from_repo(args.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0 if resume_state is None else int(resume_state["global_step"])
    try:
        if config["runtime"]["deterministic_algorithms"]:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        set_seeds(int(config["seed"]))
        write_status(run_dir, "preflight", "starting", global_step)
        log_event(run_dir, "preflight_started", run_id=args.run_id, resume=bool(args.resume))
        info = environment_report(config)
        if info.get("dirty") and not config["runtime"]["allow_dirty"]:
            raise RunStop("failed", "worktree is dirty and allow_dirty is false", 2)
        atomic_write_json(run_dir / "environment.json", info)
        report = resource_report(config, run_dir)
        atomic_write_json(run_dir / "resource_report.json", report)
        train_manifest = resolve_from_repo(Path(config["data"]["train_manifest"]))
        val_manifest = resolve_from_repo(Path(config["data"]["val_manifest"]))
        metadata_path = resolve_from_repo(Path(config["data"]["manifest_metadata"]))
        train_rows = read_jsonl(train_manifest)
        val_rows = read_jsonl(val_manifest)
        validate_rows(train_rows, "train")
        validate_rows(val_rows, "val")
        manifest_sha = {
            "train": sha256_file(train_manifest),
            "val": sha256_file(val_manifest),
            "metadata": sha256_file(metadata_path),
        }
        config_sha = semantic_config_sha256(config, manifest_sha)
        if config["profile"] == "pilot":
            drive_root = Path(os.environ.get("QWEN3_ASR_DRIVE_ROOT", "/content/drive/MyDrive/qwen3-asr"))
            if not drive_root.is_dir():
                raise RunStop("failed", "pilot requires a mounted Drive root", 2)
            baseline = config.get("baseline")
            if not isinstance(baseline, dict):
                raise RunStop("failed", "pilot requires a baseline", 2)
            if manifest_sha["val"] != baseline["manifest_sha256"]:
                raise RunStop("failed", "val manifest does not match the pinned base", 2)
            if config["decode_eval_sha256"] != baseline["decode_eval_sha256"]:
                raise RunStop("failed", "decode eval hash does not match the pinned base", 2)
        if resume_state is not None and resume_state["config_sha256"] != config_sha:
            raise RunStop("resume_rejected", "semantic config hash does not match the checkpoint", 2)
        if resume_state is not None and resume_state["manifest_sha256"] != manifest_sha:
            raise RunStop("resume_rejected", "manifest hash does not match the checkpoint", 2)
        model, processor, lora_report = load_sft_model(config)
        atomic_write_json(run_dir / "trainable_parameters.json", lora_report)
        print(f"SFT_LORA trainable={lora_report['trainable_numel']} total={lora_report['total_numel']}", flush=True)
        eos_token = processor.tokenizer.eos_token
        eos_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if not isinstance(eos_token, str) or not isinstance(eos_id, int):
            raise RunStop("failed", "tokenizer did not provide an EOS token", 2)
        prefix = prefix_text(processor, config["decode"]["eval"]["context"], config["official_language"])
        from inference.run_inference import resolve_audio_path

        def example_for(manifest: Path, row: dict[str, Any]) -> dict[str, Any]:
            audio_path = resolve_audio_path(manifest, str(row["audio"]))
            audio = load_audio(audio_path)
            duration = audio.shape[0] / 16000.0
            if duration > float(config["data"]["max_audio_seconds"]):
                raise RunStop("data_failed", f"{row['sample_id']} is longer than max_audio_seconds", 4)
            try:
                return collate_example(
                    processor,
                    row,
                    audio,
                    prefix=prefix,
                    eos_token=eos_token,
                    eos_id=int(eos_id),
                    max_input_tokens=int(config["data"]["max_input_tokens"]),
                    max_transcript_tokens=int(config["data"]["max_transcript_tokens"]),
                )
            except ValueError as exc:
                raise RunStop("data_failed", str(exc), 4) from exc

        names = [name for name, param in model.named_parameters() if param.requires_grad]
        if resume_state is not None and names != resume_state["trainable_names"]:
            raise RunStop("resume_rejected", "trainable parameter names changed", 2)
        device = next(model.parameters()).device
        dtype = torch.bfloat16
        if resume_state is None:
            probe_row = next(
                iterate_updates(
                    train_rows,
                    seed=int(config["seed"]),
                    accumulation=1,
                    max_steps=1,
                    max_epochs=1,
                )
            )["rows"][0]
            probe = move_batch(dict(example_for(train_manifest, probe_row)), device, dtype)
            model.train()
            outputs = model.thinker(**thinker_inputs(probe), use_cache=False)
            token_sum, _count = supervised_nll_sum(outputs.logits, probe["labels"])
            if not torch.isfinite(token_sum):
                raise RunStop("failed", "probe loss is not finite", 1)
            token_sum.backward()
            saw_grad = any(param.grad is not None and torch.isfinite(param.grad).all() for param in model.parameters() if param.requires_grad)
            if not saw_grad:
                raise RunStop("failed", "probe produced no finite LoRA gradient", 2)
            model.zero_grad(set_to_none=True)
            set_seeds(int(config["seed"]))
            print("SFT_PREFLIGHT_OK", flush=True)
            log_event(run_dir, "preflight_ok")
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=float(config["optim"]["learning_rate"]),
            betas=tuple(config["optim"]["betas"]),
            eps=float(config["optim"]["eps"]),
            weight_decay=float(config["optim"]["weight_decay"]),
        )
        scheduler = HorizonScheduler(optimizer, config)
        if resume_state is not None:
            checkpoint = run_dir / "checkpoints" / step_name(global_step)
            load_adapter(model, checkpoint / "adapter" / "adapter.safetensors")
            optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location=device, weights_only=False))
            scheduler.load_state_dict(torch.load(checkpoint / "scheduler.pt", map_location="cpu", weights_only=False))
            load_rng(checkpoint / "rng_state.pt")
            print(f"SFT_RESUME global_step={global_step}", flush=True)
            log_event(run_dir, "resume_ok", global_step=global_step)
        interrupt_consumed = bool(resume_state and resume_state.get("interrupt_consumed"))
        samples_seen = 0 if resume_state is None else int(resume_state["samples_seen"])
        resolved_yaml = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
        atomic_write_text(run_dir / "resolved_config.yaml", resolved_yaml)
        atomic_write_json(run_dir / "manifest_sha256.json", manifest_sha)
        write_status(run_dir, "running", "optimizer_ready", global_step)
        updates = iterate_updates(
            train_rows,
            seed=int(config["seed"]),
            accumulation=int(config["batch"]["gradient_accumulation_steps"]),
            max_steps=int(config["train"]["max_steps"]),
            max_epochs=int(config["train"]["max_epochs"]),
        )
        if resume_state is not None:
            matched = False
            for update in updates:
                if int(update["global_step"]) == global_step:
                    if int(update["epoch"]) != int(resume_state["epoch"]) or int(update["cursor_after"]) != int(
                        resume_state["cursor_after"]
                    ):
                        raise RunStop("resume_rejected", "sampler cursor does not match the checkpoint", 2)
                    matched = True
                    break
            if not matched:
                raise RunStop("resume_rejected", "checkpoint step is outside this sampler schedule", 2)
        started = time.monotonic()
        max_runtime_minutes = float(config["runtime"]["max_runtime_minutes"])
        shutdown_margin_minutes = float(config["runtime"]["shutdown_margin_minutes"])
        last_saved = global_step if resume_state is not None else 0
        sync_result = "not_saved"
        stop_reason = "max_steps"
        decisions: list[dict[str, Any]] = []
        decision_path = run_dir / "eval_decisions.jsonl"
        if resume_state is not None and decision_path.is_file():
            decisions = [
                json.loads(line)
                for line in decision_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        budget_disabled = runtime_budget_disabled(os.environ.get("QWEN3_ASR_DISABLE_RUNTIME_BUDGET"))
        if budget_disabled:
            print("SFT_RUNTIME_BUDGET disabled", flush=True)
            log_event(run_dir, "runtime_budget_disabled")

        def budget_exceeded() -> bool:
            if budget_disabled:
                return False
            return runtime_budget_exceeded(
                time.monotonic() - started,
                max_runtime_minutes,
                shutdown_margin_minutes,
            )

        def pause_for_runtime(record: dict[str, Any] | None) -> int:
            step_to_keep = last_saved
            if step_to_keep <= 0:
                raise RunStop("failed", "runtime budget ended before the first checkpoint", 1)
            if record is not None and int(record["step"]) != step_to_keep:
                raise RunStop("failed", "runtime stop step does not match the saved checkpoint", 1)
            files: list[tuple[Path, str]] = []
            if record is not None:
                relative = f"eval/{step_name(step_to_keep)}/runtime_stop.json"
                destination = run_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                atomic_write_json(destination, record)
                files.append((destination, relative))
            write_status(run_dir, "paused", "max_runtime", step_to_keep)
            files.append((run_dir / "status.json", "status.json"))
            if decision_path.is_file():
                files.append((decision_path, "eval_decisions.jsonl"))
            pause_sync = sync_drive_files(args.run_id, files)
            write_run_report(
                run_dir,
                {
                    "run_id": args.run_id,
                    "status": "paused",
                    "reason": "max_runtime",
                    "global_step": step_to_keep,
                    "publishable": False,
                    "config_sha256": config_sha,
                    "git_sha": info.get("git_sha"),
                    "dirty": info.get("dirty"),
                    "sync": pause_sync,
                    "note": (
                        "运行时间达到 max_runtime_minutes 减去关机余量。"
                        "停在最后一个已保存的 checkpoint。未完成的 val 不写 metrics，也不进入 gate。"
                        "恢复时先补这个 step 的评测，再继续训练。"
                    ),
                },
            )
            sync_drive_files(args.run_id, [(run_dir / "run_report.md", "run_report.md")])
            log_event(run_dir, "max_runtime", global_step=step_to_keep, sync=pause_sync, record=record)
            completed = 0 if record is None else int(record["completed_samples"])
            mean_tokens = 0.0 if record is None else float(record["mean_new_tokens"])
            print(
                f"SFT_RUNTIME_STOP step={step_to_keep} completed_samples={completed} "
                f"mean_new_tokens={mean_tokens:.6g}",
                flush=True,
            )
            print("SFT_EXIT 0", flush=True)
            return 0

        def run_eval(step: int) -> str:
            if budget_exceeded():
                pause_for_runtime(
                    partial_decode_stop_record(
                        step,
                        len(val_rows),
                        [],
                        int(config["decode"]["eval"]["max_new_tokens"]),
                    )
                )
                return "paused"
            adapter_sha = sha256_file(
                run_dir / "checkpoints" / step_name(step) / "adapter" / "adapter.safetensors"
            )
            try:
                metrics = evaluate_val(
                    model,
                    processor,
                    val_rows,
                    val_manifest,
                    config,
                    run_dir,
                    step,
                    adapter_sha,
                    deadline=budget_exceeded,
                )
            except EvalRuntimeStop as exc:
                pause_for_runtime(exc.record)
                return "paused"
            log_event(run_dir, "eval_written", global_step=step)
            if config["profile"] == "pilot":
                decision = judge_sft_checkpoint(metrics, config["baseline"], config["gate"])
                decision["step"] = step
                decisions.append(decision)
                append_jsonl(decision_path, decision)
                sync_drive_files(args.run_id, [(decision_path, "eval_decisions.jsonl")])
                print(
                    f"SFT_GATE step={step} passed={decision['passed']} risk_stop={decision['risk_stop']}",
                    flush=True,
                )
                if decision["risk_stop"]:
                    return "gate_risk"
            return "ok"

        decided_steps = [int(item["step"]) for item in decisions if "step" in item]
        if resume_state is not None and pending_checkpoint_eval(
            global_step,
            int(config["train"]["eval_steps"]),
            decided_steps,
        ):
            pending_result = run_eval(global_step)
            if pending_result == "paused":
                return 0
            if pending_result == "gate_risk":
                stop_reason = "gate_risk"
        training_updates = () if stop_reason == "gate_risk" else updates
        for update in training_updates:
            if budget_exceeded():
                return pause_for_runtime(None)
            step = int(update["global_step"])
            window_sum = None
            window_count = 0
            optimizer.zero_grad(set_to_none=True)
            model.train()
            step_started = time.monotonic()
            for row in update["rows"]:
                batch = move_batch(dict(example_for(train_manifest, row)), device, dtype)
                outputs = model.thinker(**thinker_inputs(batch), use_cache=False)
                token_sum, token_count = supervised_nll_sum(outputs.logits, batch["labels"])
                if not torch.isfinite(token_sum):
                    raise RunStop("failed", f"non-finite loss at global_step {step}", 1)
                token_sum.backward()
                window_sum = token_sum.detach() if window_sum is None else window_sum + token_sum.detach()
                window_count += token_count
            scale = 1.0 / window_count
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.mul_(scale)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [param for param in model.parameters() if param.requires_grad],
                float(config["optim"]["max_grad_norm"]),
            )
            if not torch.isfinite(grad_norm):
                raise RunStop("failed", f"non-finite grad norm at global_step {step}", 1)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step = step
            samples_seen += len(update["rows"])
            loss_value = float((window_sum / window_count).item())
            append_jsonl(
                run_dir / "metrics.jsonl",
                {
                    "global_step": step,
                    "micro_step": samples_seen,
                    "samples_seen": samples_seen,
                    "group_visits": 0,
                    "loss": loss_value,
                    "grad_norm": float(grad_norm),
                    "lr": current_lr(optimizer),
                    "supervised_tokens": window_count,
                    "sample_count": len(update["rows"]),
                    "elapsed_s": time.monotonic() - step_started,
                    "gpu_free_bytes": int(torch.cuda.mem_get_info()[0]),
                },
            )
            print(
                f"SFT_STEP global_step={step}/{config['train']['max_steps']} loss={loss_value:.6f} "
                f"grad_norm={float(grad_norm):.6f} lr={current_lr(optimizer):.8g} tokens={window_count} "
                f"samples={len(update['rows'])}",
                flush=True,
            )
            interrupt = should_interrupt(step, config["train"]["interrupt_after_step"], interrupt_consumed)
            if interrupt:
                interrupt_consumed = True
            need_save = step % int(config["train"]["save_steps"]) == 0 or step == int(config["train"]["max_steps"]) or interrupt
            if need_save:
                state = {
                    "schema_version": "qwen3-asr-training-state-v1",
                    "stage": "sft",
                    "global_step": step,
                    "micro_step": samples_seen,
                    "samples_seen": samples_seen,
                    "group_visits": 0,
                    "accumulation_boundary": 0,
                    "epoch": int(update["epoch"]),
                    "cursor_after": int(update["cursor_after"]),
                    "config_sha256": config_sha,
                    "manifest_sha256": manifest_sha,
                    "eval_decode_sha256": config["decode_eval_sha256"],
                    "parent_release_sha256": None,
                    "publishable": False,
                    "interrupt_consumed": interrupt_consumed,
                    "trainable_names": names,
                    "profile": config["profile"],
                }
                sampler_state = {
                    "schema_version": "qwen3-asr-sampler-v1",
                    "splits": {
                        "train": {
                            "epoch": int(update["epoch"]),
                            "cursor": int(update["cursor_after"]),
                            "shuffle_seed": int(config["seed"]),
                            "visit": samples_seen,
                        }
                    },
                }
                digest = prepare_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    run_dir=run_dir,
                    config=config,
                    state=state,
                    sampler_state=sampler_state,
                    resolved_yaml=resolved_yaml,
                )
                try:
                    sync_result = sync_drive(run_dir, args.run_id, step, digest)
                except RunStop:
                    write_status(run_dir, "sync_failed", "drive copy failed before latest.json", step)
                    raise
                write_latest(run_dir / "latest.json", step=step, checkpoint_sha256=digest)
                log_event(run_dir, "checkpoint_committed", global_step=step, sync=sync_result, sha256=digest)
                prune_old_checkpoints(run_dir, keep_last=int(config["train"]["keep_last"]), current_step=step)
                last_saved = step
                atomic_write_json(run_dir / "training_state.json", state)
            if need_save and step % int(config["train"]["eval_steps"]) == 0:
                eval_result = run_eval(step)
                if eval_result == "paused":
                    return 0
                if eval_result == "gate_risk":
                    stop_reason = "gate_risk"
                    break
            if interrupt:
                print(f"SFT_INTERRUPT step={step}", flush=True)
                log_event(run_dir, "interrupt_after_step", global_step=step)
                write_status(run_dir, "paused", "interrupt_after_step", step)
                write_gate(run_dir, config, manifest_sha["val"], decisions)
                write_run_report(
                    run_dir,
                    {
                        "run_id": args.run_id,
                        "status": "paused",
                        "reason": "interrupt_after_step",
                        "global_step": step,
                        "publishable": False,
                        "config_sha256": config_sha,
                        "git_sha": info.get("git_sha"),
                        "dirty": info.get("dirty"),
                        "sync": sync_result,
                        "note": "这份 smoke 的验证集不是 2500 条固定 val。gate 保持 incomparable，不能发布。",
                    },
                )
                print("SFT_EXIT 0", flush=True)
                return 0
        if global_step == 0 or last_saved != global_step:
            raise RunStop("failed", "run ended without a checkpoint at the last completed step", 1)
        if config["profile"] == "pilot":
            selected = select_sft_checkpoint(decisions)
            final_status = "completed" if selected is not None else "gate_failed"
            gate_status = "passed" if selected is not None else "failed"
            note = "pilot 的 publishable 保持 false。通过的 checkpoint 还不是 release。"
        else:
            selected = None
            final_status = "completed"
            gate_status = "incomparable"
            note = "这份 smoke 的验证集不是 2500 条固定 val。gate 保持 incomparable，不能发布。"
        write_status(run_dir, final_status, stop_reason, global_step)
        write_gate(run_dir, config, manifest_sha["val"], decisions, selected=selected, gate_status=gate_status)
        write_run_report(
            run_dir,
            {
                "run_id": args.run_id,
                "status": final_status,
                "reason": stop_reason,
                "global_step": global_step,
                "publishable": False,
                "config_sha256": config_sha,
                "git_sha": info.get("git_sha"),
                "dirty": info.get("dirty"),
                "sync": sync_result,
                "note": note,
            },
        )
        print(f"SFT_GATE_STATUS {gate_status}", flush=True)
        print(f"SFT_DONE global_step={global_step}", flush=True)
        print("SFT_EXIT 0", flush=True)
        return 0
    except RunStop as exc:
        write_status(run_dir, exc.status, str(exc), global_step)
        log_event(run_dir, exc.status, message=str(exc))
        print(f"SFT_EXIT {exc.code} {exc.status}: {exc}", file=sys.stderr, flush=True)
        return exc.code
    except torch.cuda.OutOfMemoryError as exc:
        write_status(run_dir, "oom", str(exc), global_step)
        log_event(run_dir, "oom", message=str(exc))
        print(f"SFT_EXIT 1 oom: {exc}", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:
        write_status(run_dir, "failed", f"{type(exc).__name__}: {exc}", global_step)
        log_event(run_dir, "failed", message=str(exc), traceback=traceback.format_exc())
        print(f"SFT_EXIT 1 failed: {exc}", file=sys.stderr, flush=True)
        return 1


def write_gate(
    run_dir: Path,
    config: dict[str, Any],
    val_sha: str,
    decisions: list[dict[str, Any]] | None = None,
    selected: dict[str, Any] | None = None,
    gate_status: str = "incomparable",
) -> None:
    if config["profile"] != "pilot":
        payload = {
            "schema_version": "qwen3-asr-gate-v1",
            "status": "incomparable",
            "publishable": False,
            "profile": config["profile"],
            "reason": "smoke val is a subset and is not the fixed 2500-row val baseline",
            "val_manifest_sha256": val_sha,
        }
    else:
        payload = {
            "schema_version": "qwen3-asr-gate-v1",
            "status": gate_status,
            "publishable": False,
            "profile": "pilot",
            "reason": "dirty worktree and no release readback",
            "val_manifest_sha256": val_sha,
            "baseline_sha256": config["baseline"]["sha256"],
            "selected_step": None if selected is None else int(selected["step"]),
            "decisions": decisions or [],
        }
    atomic_write_json(run_dir / "gate.json", payload)


def load_resume_state(run_dir: Path) -> dict[str, Any]:
    latest = read_latest(run_dir / "latest.json")
    checkpoint = run_dir / latest["relative_path"]
    verify_checkpoint(checkpoint, str(latest["checkpoint_sha256"]))
    state = json.loads((checkpoint / "training_state.json").read_text(encoding="utf-8"))
    if int(state.get("global_step", -1)) != int(latest["step_name"].split("_")[1]):
        raise ValueError("training_state global_step does not match latest.json")
    return state


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_train_config(args.config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"config rejected: {exc}", file=sys.stderr)
        return 2
    if args.seed != int(config["seed"]):
        print(f"seed mismatch: cli {args.seed} config {config['seed']}", file=sys.stderr)
        return 2
    if config["runtime"]["deterministic_algorithms"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if re.fullmatch(rf"\d{{8}}T\d{{6}}{re.escape(RUN_ID_PREFIX)}{args.seed}", args.run_id) is None:
        print("run_id must look like YYYYMMDDTHHMMSS_sft_06b_zh_seed<seed>", file=sys.stderr)
        return 2
    run_dir = resolve_from_repo(args.output_dir)
    if args.manifest is not None:
        config["data"]["train_manifest"] = str(args.manifest)
    if args.val_manifest is not None:
        config["data"]["val_manifest"] = str(args.val_manifest)
    lock_sha = sha256_file(ROOT / "requirements.lock")
    if lock_sha != config["runtime"]["environment_lock_sha256"]:
        print("environment lock hash does not match requirements.lock", file=sys.stderr)
        return 2
    resume_state = None
    if args.resume:
        try:
            resume_state = load_resume_state(run_dir)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            if run_dir.exists():
                write_status(run_dir, "resume_rejected", str(exc), 0)
            print(f"resume rejected: {exc}", file=sys.stderr)
            return 2
    elif run_dir.exists() and any(run_dir.iterdir()):
        print(f"output dir already exists: {run_dir}", file=sys.stderr)
        return 2
    for key in ("train_manifest", "val_manifest", "manifest_metadata"):
        path = resolve_from_repo(Path(config["data"][key]))
        if not path.is_file():
            print(f"missing {key}: {path}", file=sys.stderr)
            return 2
    atomic_write_json(
        run_dir / "request.json",
        {
            "schema_version": "qwen3-asr-request-v1",
            "run_id": args.run_id,
            "stage": "sft",
            "seed": args.seed,
            "resume": bool(args.resume),
            "allow_dirty": config["runtime"]["allow_dirty"],
            "publishable": False,
            "parent_release": None,
        },
    )
    return run_training(args, config, resume_state)


if __name__ == "__main__":
    raise SystemExit(main())
