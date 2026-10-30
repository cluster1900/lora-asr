"""Pure SFT contracts: LoRA names, labels, token-weighted loss, sampler, smoke subset."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable

SCENARIOS = ("clean", "dropout", "far_field", "noise", "reverb")
PROCESSOR_FILES = (
    "chat_template.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "vocab.json",
)
DECODER_LORA_MODULE = (
    r"thinker\.model\.layers\.\d+\.self_attn\.(?:q_proj|k_proj|v_proj|o_proj)"
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def processor_tree_sha256(root: Path) -> str:
    """Hash the pinned processor file list. Missing files fail the pin."""
    lines = []
    for name in PROCESSOR_FILES:
        path = root / name
        if not path.is_file():
            raise ValueError(f"processor file missing: {name}")
        lines.append(f"{name}\t{sha256_file(path)}")
    return sha256_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def is_decoder_attention_module(name: str) -> bool:
    """True only for thinker text-decoder attention projections.

    A bare ``q_proj`` or ``thinker.audio_tower...q_proj`` stays false.
    """
    import re

    if "audio_tower" in name or "embed_tokens" in name or "lm_head" in name or "projector" in name:
        return False
    return re.search(rf"(?:^|\.){DECODER_LORA_MODULE}$", name) is not None


def is_trainable_lora_parameter(name: str) -> bool:
    """True for the LoRA A/B weights on a decoder attention projection."""
    import re

    match = re.search(
        rf"(?:^|\.)(?P<module>{DECODER_LORA_MODULE})\.lora_[AB](?:\.default)?\.weight$",
        name,
    )
    return match is not None and is_decoder_attention_module(match.group("module"))


def assert_trainable_names(names: Iterable[str]) -> list[str]:
    """Return the accepted names. Any other trainable name fails."""
    accepted = []
    rejected = []
    for name in names:
        if is_trainable_lora_parameter(name):
            accepted.append(name)
        else:
            rejected.append(name)
    if rejected or not accepted:
        raise ValueError(f"trainable parameters outside decoder attention LoRA: {rejected[:20]}")
    return accepted


def mask_labels(input_ids: list[int], attention_mask: list[int], prefix_len: int) -> list[int]:
    """Mask the prefix and padding. Do not drop a real EOS just because pad id equals EOS id."""
    if len(input_ids) != len(attention_mask):
        raise ValueError("input_ids and attention_mask lengths differ")
    if prefix_len <= 0 or prefix_len >= len(input_ids):
        raise ValueError("prefix_len must leave at least one supervised token")
    labels = list(input_ids)
    for index in range(prefix_len):
        labels[index] = -100
    for index, bit in enumerate(attention_mask):
        if bit == 0:
            labels[index] = -100
    return labels


def shift_supervised_ids(labels: list[int]) -> list[int]:
    """Ids that receive loss after the causal shift."""
    return [token for token in labels[1:] if token != -100]


def assert_shift_supervises_transcript(labels: list[int], prefix_len: int, eos_id: int) -> None:
    """The first transcript token and the real EOS both stay in the shifted labels."""
    if labels[prefix_len] == -100:
        raise ValueError("first transcript token is masked")
    supervised = shift_supervised_ids(labels)
    if not supervised or supervised[0] != labels[prefix_len] or supervised[-1] != eos_id:
        raise ValueError("causal shift dropped the first transcript token or the EOS")
    if labels.count(eos_id) != 1:
        raise ValueError("expected exactly one supervised EOS")


def token_weighted_loss(groups: list[list[float]]) -> float:
    """Sum of supervised NLL divided by supervised token count."""
    total = 0.0
    count = 0
    for group in groups:
        if not group:
            raise ValueError("a microbatch has no supervised tokens")
        for value in group:
            if not math.isfinite(value):
                raise ValueError("non-finite token nll")
            total += value
        count += len(group)
    if count == 0:
        raise ValueError("window has no supervised tokens")
    return total / count


def mean_of_mean_losses(groups: list[list[float]]) -> float:
    """The reduction this runner must not use."""
    means = [sum(group) / len(group) for group in groups]
    return sum(means) / len(means)


def lr_multiplier(step: int, warmup: int, total: int, min_ratio: float) -> float:
    """Warmup, then cosine down to min_ratio. ``step`` is the completed optimizer step."""
    if step <= 0:
        return 0.0
    if warmup < 0 or total <= warmup:
        raise ValueError("scheduler horizon is invalid")
    if warmup > 0 and step <= warmup:
        return step / warmup
    if step >= total:
        return min_ratio
    progress = (step - warmup) / (total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine


def sampler_key(seed: int, epoch: int, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{epoch}:{sample_id}:sft-sampler-v1".encode("utf-8")).hexdigest()


def epoch_order(rows: list[dict[str, Any]], seed: int, epoch: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: sampler_key(seed, epoch, str(row["sample_id"])))


def iterate_updates(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    accumulation: int,
    max_steps: int,
    max_epochs: int,
) -> Iterable[dict[str, Any]]:
    """Yield one optimizer update at a time. A short epoch tail keeps its own denominator."""
    if not rows:
        raise ValueError("train manifest is empty")
    epoch = 0
    cursor = 0
    produced = 0
    order = epoch_order(rows, seed, epoch)
    while produced < max_steps and epoch < max_epochs:
        if cursor >= len(order):
            epoch += 1
            cursor = 0
            if epoch >= max_epochs:
                break
            order = epoch_order(rows, seed, epoch)
            continue
        take = min(accumulation, len(order) - cursor)
        produced += 1
        cursor += take
        yield {
            "global_step": produced,
            "epoch": epoch,
            "cursor_after": cursor,
            "rows": order[cursor - take : cursor],
        }


def update_ends_schedule(
    update: dict[str, Any],
    *,
    row_count: int,
    max_steps: int,
    max_epochs: int,
) -> bool:
    """True on the last update ``iterate_updates`` will yield.

    A one-epoch tail such as 164 pairs with accumulation 4 ends at step 41.
    That step is not a multiple of ``save_steps`` and is below ``max_steps``.
    """
    if row_count < 1 or max_steps < 1 or max_epochs < 1:
        raise ValueError("schedule bounds must be positive")
    step = int(update["global_step"])
    if step < 1:
        raise ValueError("global_step must be positive")
    if step >= int(max_steps):
        return True
    return int(update["epoch"]) + 1 >= int(max_epochs) and int(update["cursor_after"]) >= int(row_count)


def should_interrupt(step: int, interrupt_after_step: int | None, consumed: bool) -> bool:
    return interrupt_after_step is not None and not consumed and step == interrupt_after_step


def runtime_budget_disabled(flag: str | None) -> bool:
    """True only for the explicit launch override. The yaml budget stays in the hash."""
    return flag == "1"


def runtime_budget_exceeded(elapsed_s: float, max_runtime_minutes: float, shutdown_margin_minutes: float) -> bool:
    """The budget is exhausted only after the shutdown margin. Equality still fits one more check."""
    if elapsed_s < 0:
        raise ValueError("elapsed_s must be >= 0")
    limit_s = (float(max_runtime_minutes) - float(shutdown_margin_minutes)) * 60.0
    return float(elapsed_s) > limit_s


def pending_checkpoint_eval(
    global_step: int,
    eval_steps: int,
    decided_steps: Iterable[int],
    *,
    schedule_end_step: int | None = None,
) -> bool:
    """A saved eval step still needs a full val before the next optimizer step.

    ``schedule_end_step`` is the last optimizer step of this sampler. It is
    evaluated even when it is not a multiple of ``eval_steps``.
    """
    if global_step <= 0 or eval_steps <= 0:
        return False
    decided = {int(step) for step in decided_steps}
    if int(global_step) in decided:
        return False
    if global_step % eval_steps == 0:
        return True
    return schedule_end_step is not None and int(global_step) == int(schedule_end_step)


def partial_decode_stop_record(
    step: int,
    total_rows: int,
    token_lengths: list[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    """Describe a val decode stopped by the runtime budget. This is not a gate."""
    completed = len(token_lengths)
    if total_rows < 0 or completed > total_rows:
        raise ValueError("completed samples exceed the val set")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    hit = sum(1 for length in token_lengths if length >= max_new_tokens)
    return {
        "schema_version": "qwen3-asr-eval-runtime-stop-v1",
        "step": int(step),
        "total_rows": int(total_rows),
        "completed_samples": completed,
        "remaining_samples": int(total_rows) - completed,
        "mean_new_tokens": (sum(token_lengths) / completed) if completed else 0.0,
        "max_new_tokens_seen": max(token_lengths) if token_lengths else 0,
        "hit_max_new_tokens": hit,
        "max_new_tokens": int(max_new_tokens),
        "gate": False,
    }


DEGRADED_SCENARIOS = ("dropout", "far_field", "noise", "reverb")


def degraded_macro_cer(by_scenario: dict[str, Any]) -> float:
    """Equal-weight mean of the four degraded scenario micro CERs."""
    return sum(float(by_scenario[name]["micro_cer"]) for name in DEGRADED_SCENARIOS) / len(DEGRADED_SCENARIOS)


def judge_sft_checkpoint(
    metrics: dict[str, Any],
    baseline: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Judge one full val against the pinned base. Comparisons use raw micro CER."""
    risks: list[str] = []
    by_scenario = metrics["by_scenario"]
    base_scenarios = baseline["by_scenario"]
    for name in SCENARIOS:
        samples = int(by_scenario[name]["samples"])
        if samples < int(gate["min_rows_per_scenario"]):
            risks.append(f"{name} has {samples} rows")
    scopes = [("overall", metrics), *[(name, by_scenario[name]) for name in SCENARIOS]]
    limits = (
        ("empty_rate", "empty_rate_max"),
        ("repeated_rate", "repeated_rate_max"),
        ("long_rate", "long_rate_max"),
        ("truncation_rate", "truncation_rate_max"),
    )
    for scope, block in scopes:
        for field, limit_name in limits:
            value = float(block[field])
            limit = float(gate[limit_name])
            if value > limit:
                risks.append(f"{scope} {field} {value} exceeds {limit}")
        valid = float(block["valid_output_rate"])
        minimum = float(gate["valid_output_rate_min"])
        if valid < minimum:
            risks.append(f"{scope} valid_output_rate {valid} is below {minimum}")
    clean_delta = float(by_scenario["clean"]["micro_cer"]) - float(base_scenarios["clean"]["micro_cer"])
    if clean_delta > float(gate["clean_cer_increase_max"]):
        risks.append(f"clean micro CER increased by {clean_delta}")
    if clean_delta > float(gate["clean_cer_increase_vs_base_max"]):
        risks.append(f"clean micro CER increased by {clean_delta} against the base cap")
    degraded_deltas = {}
    for name in DEGRADED_SCENARIOS:
        delta = float(by_scenario[name]["micro_cer"]) - float(base_scenarios[name]["micro_cer"])
        degraded_deltas[name] = delta
        if delta > float(gate["scenario_cer_increase_max"]):
            risks.append(f"{name} micro CER increased by {delta}")
    macro_delta = degraded_macro_cer(by_scenario) - degraded_macro_cer(base_scenarios)
    if macro_delta > float(gate["degraded_macro_increase_max"]):
        risks.append(f"degraded macro CER increased by {macro_delta}")
    improved = sum(1 for delta in degraded_deltas.values() if delta < 0.0)
    return {
        "passed": not risks and improved >= int(gate["min_improved_degraded_scenarios"]),
        "risk_stop": bool(risks),
        "improved_degraded_scenarios": improved,
        "clean_cer": float(by_scenario["clean"]["micro_cer"]),
        "degraded_macro_cer": degraded_macro_cer(by_scenario),
        "clean_delta": clean_delta,
        "degraded_deltas": degraded_deltas,
        "macro_delta": macro_delta,
        "risks": risks,
    }


def select_sft_checkpoint(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the passed checkpoint with the lowest degraded macro, then clean, then step."""
    passed = [record for record in records if record.get("passed")]
    if not passed:
        return None
    return min(passed, key=lambda record: (float(record["degraded_macro_cer"]), float(record["clean_cer"]), int(record["step"])))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            rows.append(row)
    return rows


def _groups_with_all_scenarios(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        source_id = row.get("source_utterance_id")
        scenario = row.get("scenario")
        if not isinstance(source_id, str) or scenario not in SCENARIOS:
            raise ValueError("smoke rows need source_utterance_id and a known scenario")
        grouped.setdefault(source_id, []).append(row)
    ready: dict[str, list[dict[str, Any]]] = {}
    for source_id, group in grouped.items():
        found = [row.get("scenario") for row in group]
        if len(found) != len(set(found)):
            raise ValueError(f"duplicate scenario for {source_id}")
        if set(found) == set(SCENARIOS):
            ready[source_id] = sorted(group, key=lambda row: str(row["scenario"]))
    return ready


def select_smoke_sources(rows: list[dict[str, Any]], *, seed: int, count: int, rank_name: str) -> list[str]:
    ready = _groups_with_all_scenarios(rows)
    if len(ready) < count:
        raise ValueError(f"need {count} sources with all five scenarios, found {len(ready)}")
    ranked = sorted(
        ready,
        key=lambda source_id: hashlib.sha256(f"{seed}:{source_id}:{rank_name}".encode("utf-8")).hexdigest(),
    )
    return ranked[:count]


def _resolve_audio(manifest_path: Path, audio: str) -> Path:
    if not isinstance(audio, str) or not audio or audio.startswith("/") or "\\" in audio or ".." in Path(audio).parts:
        raise ValueError(f"audio path must be a relative POSIX path: {audio!r}")
    return manifest_path.parent / audio


def build_smoke_subset(
    train_manifest: Path,
    val_manifest: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    train_sources: int = 10,
    val_sources: int = 2,
    copy_audio: bool = True,
) -> dict[str, Any]:
    """Write a publishable=false smoke manifest. Train and val sources stay disjoint."""
    train_rows = _read_jsonl(train_manifest)
    val_rows = _read_jsonl(val_manifest)
    train_ids = select_smoke_sources(train_rows, seed=seed, count=train_sources, rank_name="sft-smoke-train-v1")
    val_ids = select_smoke_sources(val_rows, seed=seed, count=val_sources, rank_name="sft-smoke-val-v1")
    overlap = sorted(set(train_ids) & set(val_ids))
    if overlap:
        raise ValueError(f"smoke train/val source overlap: {overlap}")
    train_groups = _groups_with_all_scenarios(train_rows)
    val_groups = _groups_with_all_scenarios(val_rows)
    selected_train = [row for source_id in train_ids for row in train_groups[source_id]]
    selected_val = [row for source_id in val_ids for row in val_groups[source_id]]
    output_dir.mkdir(parents=True, exist_ok=True)
    if copy_audio:
        for manifest_path, rows in ((train_manifest, selected_train), (val_manifest, selected_val)):
            for row in rows:
                source = _resolve_audio(manifest_path, str(row["audio"]))
                if not source.is_file():
                    raise ValueError(f"missing audio for smoke subset: {source}")
                destination = output_dir / str(row["audio"])
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    _write_jsonl(output_dir / "train.jsonl", selected_train)
    _write_jsonl(output_dir / "val.jsonl", selected_val)
    metadata = {
        "schema_version": "qwen3-asr-sft-smoke-subset-v1",
        "publishable": False,
        "seed": seed,
        "train_sources": train_ids,
        "val_sources": val_ids,
        "scenarios": list(SCENARIOS),
        "parent_train_sha256": sha256_file(train_manifest),
        "parent_val_sha256": sha256_file(val_manifest),
        "selection": "sha256(seed:source_utterance_id:sft-smoke-train-v1 or sft-smoke-val-v1), first complete five-scenario groups",
    }
    (output_dir / "manifest.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
