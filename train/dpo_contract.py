"""DPO loss, held-out preference, and the parent-plus-base gate.

Log probabilities are sequence sums. The trainer must not add CER or a gold
score as an extra loss. Equal policy and reference log probabilities give
preference accuracy 0.5.
"""

from __future__ import annotations

import math
from typing import Any

from evaluation.zh_cer import (
    GAP_ANCHOR_HIGH,
    GAP_ANCHOR_LOW,
    align_edits,
    edit_distance,
    gap_insertion_count,
    normalize_zh_cer_v1,
)
from train.sft_contract import degraded_macro_cer, judge_sft_checkpoint

GAP_INSERTION_MIN = 30
EARLY_DELETE_CLEAN_DELTA = 0.005
EARLY_DELETE_LENGTH_RATIO = 0.75
QUIET_DROPOUT_ABS_DELTA = 0.002
SCENARIO_NAMES = ("clean", "dropout", "far_field", "noise", "reverb")

LISTEN_WAIVER = "user_20261017_skip_listen"
AUDIT_DISPOSITION = "unlistened"
WAIVER_SCHEMA = "qwen3-asr-dpo-listen-waiver-v1"
GAP_STRIP_METADATA_SCHEMA = "qwen3-asr-dpo-gap-strip-metadata-v1"
GAP_STRIP_LAUNCH = "开始吧"


def stable_neg_log_sigmoid(delta: float) -> float:
    """``-log(sigmoid(delta))`` without overflowing exp."""
    if not math.isfinite(delta):
        raise ValueError("DPO delta is not finite")
    if delta >= 0.0:
        return math.log1p(math.exp(-delta))
    return -delta + math.log1p(math.exp(delta))


def pair_dpo_loss(
    chosen_logp: list[float],
    rejected_logp: list[float],
    ref_chosen_logp: list[float],
    ref_rejected_logp: list[float],
    beta: float,
) -> tuple[float, list[float]]:
    """Mean over pairs of ``-logsigmoid(beta * policy_margin - beta * ref_margin)``."""
    if beta <= 0.0 or not math.isfinite(beta):
        raise ValueError("beta must be a positive finite number")
    rows = (chosen_logp, rejected_logp, ref_chosen_logp, ref_rejected_logp)
    count = len(chosen_logp)
    if count == 0 or any(len(column) != count for column in rows):
        raise ValueError("DPO loss needs one or more aligned pairs")
    losses: list[float] = []
    deltas: list[float] = []
    for chosen, rejected, ref_chosen, ref_rejected in zip(*rows):
        values = (chosen, rejected, ref_chosen, ref_rejected)
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("DPO logprob is not finite")
        delta = float(beta) * (
            (float(chosen) - float(rejected)) - (float(ref_chosen) - float(ref_rejected))
        )
        losses.append(stable_neg_log_sigmoid(delta))
        deltas.append(delta)
    return sum(losses) / count, deltas


def preference_accuracy(deltas: list[float]) -> dict[str, float | int]:
    """Held-out accuracy. Ties count as one half. Comparison uses the raw delta."""
    if not deltas:
        raise ValueError("preference accuracy needs at least one pair")
    positive = 0
    ties = 0
    for delta in deltas:
        if not math.isfinite(delta):
            raise ValueError("preference delta is not finite")
        if delta > 0.0:
            positive += 1
        elif delta == 0.0:
            ties += 1
    total = len(deltas)
    return {
        "pairs": total,
        "positive": positive,
        "ties": ties,
        "negative": total - positive - ties,
        "accuracy": (positive + 0.5 * ties) / total,
    }


def raw_policy_preference(chosen_logp: list[float], rejected_logp: list[float]) -> float:
    """Fraction of pairs whose policy sequence logprob prefers chosen. Ties are not wins."""
    if not chosen_logp or len(chosen_logp) != len(rejected_logp):
        raise ValueError("raw preference needs aligned policy logprobs")
    wins = 0
    for chosen, rejected in zip(chosen_logp, rejected_logp):
        if not math.isfinite(chosen) or not math.isfinite(rejected):
            raise ValueError("policy logprob is not finite")
        if chosen > rejected:
            wins += 1
    return wins / len(chosen_logp)


def length_buckets(
    char_length_deltas: list[int],
    chosen_logp: list[float],
    rejected_logp: list[float],
) -> dict[str, dict[str, float | int]]:
    """Split raw policy accuracy by whether the rejected text is shorter, equal, or longer."""
    if not char_length_deltas or len(char_length_deltas) != len(chosen_logp) or len(chosen_logp) != len(rejected_logp):
        raise ValueError("length buckets need one delta and two logprobs per pair")
    groups: dict[str, list[tuple[float, float]]] = {
        "shorter_rejected": [],
        "equal_length": [],
        "longer_rejected": [],
    }
    for delta, chosen, rejected in zip(char_length_deltas, chosen_logp, rejected_logp):
        if not isinstance(delta, int) or isinstance(delta, bool):
            raise ValueError("char length delta must be an integer")
        if delta < 0:
            key = "shorter_rejected"
        elif delta == 0:
            key = "equal_length"
        else:
            key = "longer_rejected"
        groups[key].append((float(chosen), float(rejected)))
    report: dict[str, dict[str, float | int]] = {}
    for name, rows in groups.items():
        wins = sum(1 for chosen, rejected in rows if chosen > rejected)
        report[name] = {
            "pairs": len(rows),
            "raw_policy_accuracy": (wins / len(rows)) if rows else 0.0,
        }
    return report


def preference_gate_ok(accuracy: float, gate: dict[str, Any]) -> bool:
    """The held-out bar is at least the configured floor and strictly above one half."""
    floor = float(gate["min_preference_accuracy"])
    return float(accuracy) >= floor and float(accuracy) > 0.5


def judge_dpo_checkpoint(
    metrics: dict[str, Any],
    parent_metrics: dict[str, Any],
    base_metrics: dict[str, Any],
    gate: dict[str, Any],
    preference: dict[str, Any],
) -> dict[str, Any]:
    """CER against the SFT parent, clean against the base cap, then preference.

    ``judge_sft_checkpoint`` compares both clean caps to one baseline. The parent
    check uses the 0.01 cap. The 0.02 cap is applied afterwards against the base.
    """
    parent_gate = dict(gate)
    parent_gate["clean_cer_increase_vs_base_max"] = 1.0
    parent_view = {"by_scenario": parent_metrics["by_scenario"]}
    parent_decision = judge_sft_checkpoint(metrics, parent_view, parent_gate)
    risks = list(parent_decision["risks"])
    clean_vs_base = float(metrics["by_scenario"]["clean"]["micro_cer"]) - float(
        base_metrics["by_scenario"]["clean"]["micro_cer"]
    )
    base_cap = float(gate["clean_cer_increase_vs_base_max"])
    if clean_vs_base > base_cap:
        risks.append(f"clean micro CER increased by {clean_vs_base} against the base cap")
    accuracy = float(preference["accuracy"])
    if not preference_gate_ok(accuracy, gate):
        risks.append(f"preference accuracy {accuracy} is below the DPO gate")
    pair_count = int(preference["pairs"])
    if pair_count < int(gate["min_val_pairs"]):
        risks.append(f"val pairs {pair_count} are below min_val_pairs")
    return {
        "passed": not risks and int(parent_decision["improved_degraded_scenarios"]) >= int(gate["min_improved_degraded_scenarios"]),
        "risk_stop": bool(risks),
        "risks": risks,
        "improved_degraded_scenarios": int(parent_decision["improved_degraded_scenarios"]),
        "clean_cer": float(parent_decision["clean_cer"]),
        "degraded_macro_cer": float(parent_decision["degraded_macro_cer"]),
        "clean_delta_vs_parent": float(parent_decision["clean_delta"]),
        "clean_delta_vs_base": clean_vs_base,
        "degraded_deltas": parent_decision["degraded_deltas"],
        "macro_delta": float(parent_decision["macro_delta"]),
        "parent_degraded_macro_cer": degraded_macro_cer(parent_metrics["by_scenario"]),
        "preference_accuracy": accuracy,
        "preference_ties": int(preference["ties"]),
        "raw_policy_accuracy": float(preference["raw_policy_accuracy"]),
        "length_buckets": preference["length_buckets"],
        "publishable": False,
    }


def listen_waiver_record(launch_utterance: str | None = None) -> dict[str, Any]:
    """Record a listen waiver without marking any clip heard.

    The 2026-10-17 utterance stays on every DPO run. ``launch_utterance`` adds
    the 2026-10-18「开始吧」probe and says that older waiver does not cover the
    gap-strip audit.
    """
    record = {
        "schema_version": WAIVER_SCHEMA,
        "decided_on": "2026-10-17",
        "waiver": LISTEN_WAIVER,
        "utterance": "不听，直接开始训练",
        "audit_disposition": AUDIT_DISPOSITION,
        "heard": False,
        "dispositions_unchanged": True,
        "text_review_status": "text_checked_unlistened",
    }
    if launch_utterance is None:
        return record
    if launch_utterance != GAP_STRIP_LAUNCH:
        raise ValueError("gap-strip launch utterance must be 开始吧")
    record["gap_strip_launch"] = {
        "decided_on": "2026-10-18",
        "utterance": GAP_STRIP_LAUNCH,
        "prior_waiver_covers_this_audit": False,
        "heard": False,
        "audit_disposition": AUDIT_DISPOSITION,
        "scope": "one dropout probe on pairs_gap_strip",
    }
    return record


def check_pair_metadata(payload: dict[str, Any], config: dict[str, Any], release: dict[str, Any]) -> None:
    """Check pair metadata against the config. Gap-strip files have no release id."""
    if not isinstance(payload, dict):
        raise ValueError("pair metadata must be an object")
    dpo = config["dpo"]
    if payload.get("schema_version") == GAP_STRIP_METADATA_SCHEMA:
        if dpo.get("launch_utterance") != GAP_STRIP_LAUNCH:
            raise ValueError("gap-strip metadata requires the 2026-10-18 launch utterance")
        if payload.get("status") != "pairs_gap_stripped":
            raise ValueError("gap-strip metadata status is not pairs_gap_stripped")
        if payload.get("trainable") is not False:
            raise ValueError("gap-strip metadata trainable must stay false")
        if payload.get("gap_strip") != "dropout-0.40-0.55":
            raise ValueError("gap-strip metadata gap_strip is not dropout-0.40-0.55")
        if int(payload.get("seed", -1)) != int(config["seed"]):
            raise ValueError("gap-strip metadata seed does not match the config")
        for left, right in (
            ("train_pairs_sha256", "null_train_pairs_sha256"),
            ("val_pairs_sha256", "null_val_pairs_sha256"),
            ("source_train_pairs_sha256", "source_null_train_pairs_sha256"),
            ("source_val_pairs_sha256", "source_null_val_pairs_sha256"),
        ):
            if payload.get(left) != dpo.get(right):
                raise ValueError(f"gap-strip metadata {left} does not match the config")
        release_id = payload.get("release_id")
        if release_id not in (None, config["model"]["parent_release"]):
            raise ValueError("gap-strip metadata names a different release")
        return
    if payload.get("release_id") != config["model"]["parent_release"]:
        raise ValueError("pair metadata release_id does not match the config")
    if payload.get("adapter_sha256") != release.get("adapter_sha256"):
        raise ValueError("pair metadata adapter hash does not match the release")
    if payload.get("train_pairs_sha256") != dpo["null_train_pairs_sha256"]:
        raise ValueError("pair metadata does not pin the null train pairs")
    if payload.get("val_pairs_sha256") != dpo["null_val_pairs_sha256"]:
        raise ValueError("pair metadata does not pin the null val pairs")


def _is_aligned_substitution_pair(row: dict[str, Any], *, max_edit_count: int | None = None) -> bool:
    """True when chosen and rejected are punctuation-free, equal-length, equal-token substitutions."""
    chosen = str(row.get("chosen_text", ""))
    rejected = str(row.get("rejected_text", ""))
    if not chosen or not rejected:
        return False
    if normalize_zh_cer_v1(chosen) != chosen or normalize_zh_cer_v1(rejected) != rejected:
        return False
    if len(chosen) != len(rejected):
        return False
    if "char_length_delta" in row and row["char_length_delta"] is not None and int(row["char_length_delta"]) != 0:
        return False
    if int(row.get("chosen_token_count", -1)) != int(row.get("rejected_token_count", -2)):
        return False
    hamming = sum(1 for left, right in zip(chosen, rejected) if left != right)
    if hamming <= 0:
        return False
    if max_edit_count is not None and hamming > int(max_edit_count):
        return False
    if (
        max_edit_count is not None
        and "edit_count" in row
        and row["edit_count"] is not None
        and int(row["edit_count"]) > int(max_edit_count)
    ):
        return False
    return hamming == edit_distance(list(chosen), list(rejected))


def mismatched_token_mask(chosen_labels: list[int], rejected_labels: list[int]) -> list[bool]:
    """Boolean mask of supervised token positions where chosen and rejected differ."""
    if not chosen_labels or not rejected_labels:
        raise ValueError("mismatched_token_mask requires non-empty label sequences")
    if len(chosen_labels) != len(rejected_labels):
        raise ValueError("mismatched_token_mask requires equal-length label sequences")
    mask = [int(left) != int(right) for left, right in zip(chosen_labels, rejected_labels)]
    if not any(mask):
        return [True] * len(chosen_labels)
    return mask


def build_clean_rejected_map(rows: list[dict[str, Any]]) -> dict[str, str]:
    """Map ``source_utterance_id`` to normalized ``rejected_text`` from ``clean`` rows."""
    mapping: dict[str, str] = {}
    for row in rows:
        if row.get("scenario") != "clean":
            continue
        source_id = row.get("source_utterance_id")
        rejected = row.get("rejected_text")
        if isinstance(source_id, str) and source_id and isinstance(rejected, str) and rejected:
            mapping[source_id] = normalize_zh_cer_v1(rejected)
    return mapping


def select_dropout_anchor_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Train dropout rows whose parent greedy text can be the NLL teacher.

    The teacher is ``rejected_text``. Gold ``chosen_text`` stays out of this list's loss.
    File order is the training cycle. Empty hypotheses are skipped.
    """
    selected: list[dict[str, Any]] = []
    for row in rows:
        if row.get("scenario") != "dropout":
            continue
        if row.get("split") not in (None, "train"):
            continue
        text = row.get("rejected_text")
        if not isinstance(text, str) or not text.strip():
            continue
        selected.append(row)
    return selected


def dropout_anchor_step_slice(
    anchor_rows: list[dict[str, Any]],
    step: int,
    samples_per_step: int = 1,
) -> list[dict[str, Any]]:
    """Select the deterministic slice of dropout anchor rows for 1-based optimizer ``step``."""
    if not anchor_rows:
        raise ValueError("dropout_anchor_step_slice requires non-empty anchor_rows")
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError("step must be a positive integer")
    if isinstance(samples_per_step, bool) or not isinstance(samples_per_step, int) or samples_per_step < 1:
        raise ValueError("samples_per_step must be a positive integer")
    start = (step - 1) * samples_per_step
    return [anchor_rows[(start + offset) % len(anchor_rows)] for offset in range(samples_per_step)]


def select_training_pairs(
    rows: list[dict[str, Any]],
    scenarios: list[str] | None,
    *,
    aligned_substitution_only: bool = False,
    max_edit_count: int | None = None,
    ref_margin_min: float | None = None,
    ref_margin_max: float | None = None,
    exclude_clean_identical_rejected: bool = False,
    clean_rejected_by_utterance: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Keep every pair when filters are omitted. Filtering happens in memory after the cache check."""
    if max_edit_count is not None and (isinstance(max_edit_count, bool) or int(max_edit_count) < 1):
        raise ValueError("max_edit_count must be an integer >= 1")
    allowed = None if scenarios is None else set(scenarios)
    clean_map = (
        (build_clean_rejected_map(rows) if clean_rejected_by_utterance is None else clean_rejected_by_utterance)
        if exclude_clean_identical_rejected
        else None
    )
    selected: list[dict[str, Any]] = []
    for row in rows:
        if allowed is not None and row.get("scenario") not in allowed:
            continue
        if exclude_clean_identical_rejected:
            if row.get("scenario") == "clean":
                continue
            source_id = row.get("source_utterance_id")
            if (
                clean_map is not None
                and isinstance(source_id, str)
                and clean_map.get(source_id) == normalize_zh_cer_v1(str(row.get("rejected_text", "")))
            ):
                continue
        if aligned_substitution_only and not _is_aligned_substitution_pair(row, max_edit_count=max_edit_count):
            continue
        if max_edit_count is not None and not aligned_substitution_only:
            chosen_norm = normalize_zh_cer_v1(str(row.get("chosen_text", "")))
            rejected_norm = normalize_zh_cer_v1(str(row.get("rejected_text", "")))
            edits = (
                int(row["edit_count"])
                if "edit_count" in row and row["edit_count"] is not None
                else edit_distance(list(chosen_norm), list(rejected_norm))
            )
            if edits > int(max_edit_count):
                continue
        if ref_margin_min is not None or ref_margin_max is not None:
            margin = float(row["ref_chosen_logp"]) - float(row["ref_rejected_logp"])
            if not math.isfinite(margin):
                continue
            if ref_margin_min is not None and margin < float(ref_margin_min):
                continue
            if ref_margin_max is not None and margin > float(ref_margin_max):
                continue
        selected.append(row)
    return selected


def _in_dropout_gap(index: int, length: int) -> bool:
    anchor = index / length
    return GAP_ANCHOR_LOW <= anchor <= GAP_ANCHOR_HIGH


def strip_dropout_gap(reference: str, hypothesis: str) -> tuple[str, str]:
    """Remove the silenced 0.40–0.55 span from both dropout strings.

    The caller uses this only for dropout. Matches, substitutions, and
    deletions inside the span drop out of both sides. An insertion drops out
    when its anchor, the reference index still ahead, is inside the span.
    Endpoints 0.40 and 0.55 are inside.
    """
    if not reference:
        raise ValueError("dropout gap strip needs a non-empty reference")
    length = len(reference)
    chosen: list[str] = []
    rejected: list[str] = []
    for item in align_edits(list(reference), list(hypothesis)):
        op = item["op"]
        ref_index = int(item["ref_index"])
        if op == "insert":
            if _in_dropout_gap(ref_index, length):
                continue
            rejected.append(hypothesis[int(item["hyp_index"])])
            continue
        if _in_dropout_gap(ref_index, length):
            continue
        if op in {"match", "substitute"}:
            chosen.append(reference[ref_index])
            rejected.append(hypothesis[int(item["hyp_index"])])
        elif op == "delete":
            chosen.append(reference[ref_index])
    return "".join(chosen), "".join(rejected)


def classify_dropout_signature(
    *,
    gap_insertions: int,
    dropout_length_ratio: float,
    dropout_delta: float,
    clean_delta: float,
) -> str:
    """Name one pre-registered dropout failure. Threshold equality follows the training plan."""
    gap = gap_insertions >= GAP_INSERTION_MIN
    early = clean_delta > EARLY_DELETE_CLEAN_DELTA or dropout_length_ratio < EARLY_DELETE_LENGTH_RATIO
    if gap and early:
        return "MIXED"
    if gap:
        return "GAP_INSERT"
    if early:
        return "EARLY_DELETE"
    if abs(dropout_delta) < QUIET_DROPOUT_ABS_DELTA and gap_insertions < GAP_INSERTION_MIN:
        return "QUIET"
    return "OTHER"


def locate_dropout_edits(
    scored_rows: list[dict[str, Any]],
    metrics: dict[str, Any],
    parent_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Align dropout hypotheses and attach the signature fields stored on a decision."""
    dropout_rows = [row for row in scored_rows if row.get("scenario") == "dropout"]
    if not dropout_rows:
        raise ValueError("dropout localization needs dropout scored rows")
    gap_insertions = 0
    hypothesis_chars = 0
    reference_chars = 0
    for row in dropout_rows:
        reference = row.get("reference_normalized")
        hypothesis = row.get("scored_prediction_normalized")
        if not isinstance(reference, str) or not reference or not isinstance(hypothesis, str):
            raise ValueError("dropout scored row is missing normalized text")
        gap_insertions += gap_insertion_count(reference, hypothesis)
        reference_chars += len(reference)
        hypothesis_chars += len(hypothesis)
    length_ratio = hypothesis_chars / reference_chars
    dropout_delta = float(metrics["by_scenario"]["dropout"]["micro_cer"]) - float(
        parent_metrics["by_scenario"]["dropout"]["micro_cer"]
    )
    clean_delta = float(metrics["by_scenario"]["clean"]["micro_cer"]) - float(
        parent_metrics["by_scenario"]["clean"]["micro_cer"]
    )
    return {
        "locate_signature": classify_dropout_signature(
            gap_insertions=gap_insertions,
            dropout_length_ratio=length_ratio,
            dropout_delta=dropout_delta,
            clean_delta=clean_delta,
        ),
        "locate_gap_insertions": gap_insertions,
        "locate_dropout_length_ratio": length_ratio,
        "locate_dropout_delta": dropout_delta,
    }


def validate_pair_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
    min_per_scenario: int,
    required_scenarios: tuple[str, ...] | None = None,
) -> dict[str, int]:
    """Reject ties, empty logprobs, and a required scenario that is under the floor."""
    required = SCENARIO_NAMES if required_scenarios is None else required_scenarios
    if not required:
        raise ValueError("required_scenarios is empty")
    if not rows:
        raise ValueError(f"{split} pairs are empty")
    seen: set[str] = set()
    counts = {name: 0 for name in SCENARIO_NAMES}
    for index, row in enumerate(rows, start=1):
        if row.get("schema_version") != "qwen3-asr-dpo-pair-v1":
            raise ValueError(f"{split} pair {index} has the wrong schema")
        if row.get("split") != split:
            raise ValueError(f"{split} pair {index} is marked {row.get('split')}")
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or pair_id in seen:
            raise ValueError(f"{split} pair {index} has a duplicate or missing pair_id")
        seen.add(pair_id)
        scenario = row.get("scenario")
        if scenario not in counts:
            raise ValueError(f"{pair_id} has an unknown scenario")
        counts[str(scenario)] += 1
        if row.get("chosen_text") != row.get("reference_text"):
            raise ValueError(f"{pair_id} chosen text is not the reference")
        rejected = row.get("rejected_text")
        if not isinstance(rejected, str) or not rejected.strip():
            raise ValueError(f"{pair_id} rejected text is empty")
        for key in ("ref_chosen_logp", "ref_rejected_logp"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{pair_id} {key} is not a finite logprob")
        for key in ("chosen_token_count", "rejected_token_count"):
            count = row.get(key)
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(f"{pair_id} {key} must include EOS")
        delta = row.get("char_length_delta")
        if isinstance(delta, bool) or not isinstance(delta, int):
            raise ValueError(f"{pair_id} char_length_delta must be an integer")
        if row.get("bucket") not in (None, "pair"):
            raise ValueError(f"{pair_id} is not a trainable pair")
    short = [name for name in required if counts[name] < min_per_scenario]
    if short:
        raise ValueError(f"{split} scenarios below min_pairs_per_scenario: {short}")
    return counts


def assert_audit_still_unlistened(audit: dict[str, Any]) -> None:
    """Refuse a file whose dispositions were rewritten to listened or passed."""
    if audit.get("status") not in {None, "unlistened"}:
        raise ValueError(f"pair audit status is {audit.get('status')}")
    by_scenario = audit.get("by_scenario")
    if not isinstance(by_scenario, dict):
        raise ValueError("pair audit is missing by_scenario")
    for name, block in by_scenario.items():
        if not isinstance(block, dict):
            raise ValueError(f"pair audit block {name} is not an object")
        disposition = block.get("disposition")
        if disposition != AUDIT_DISPOSITION:
            raise ValueError(f"pair audit {name} disposition is {disposition}")
