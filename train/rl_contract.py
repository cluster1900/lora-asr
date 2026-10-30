"""Pure contract helpers for Phase 4 RL (group_pg_mixed_v1).

Implements deterministic candidate seeds, clipped CER-based rewards, G=4
group validity and mean-centered advantages, zero-advantage full-group zeroing
(including zero KL), k3 token proxy KL, exact vocabulary-summed conditional KL
on fixed step-0 greedy prefixes, reference tensor checksums, and RL gate logic.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from evaluation.zh_cer import normalize_zh_cer_v1, score_pair
from train.sft_schema import SCENARIOS, canonical_json, sha256_text


def _has_punctuation(text: str) -> bool:
    return not text or normalize_zh_cer_v1(text) != text


def candidate_seed_int(
    seed: int,
    epoch: int,
    visit_index: int,
    sample_id: str,
    candidate_index: int,
) -> int:
    """Map (seed, epoch, visit_index, sample_id, candidate_index) to a 63-bit int."""
    if not sample_id:
        raise ValueError("sample_id must be non-empty")
    payload = f"{int(seed)}:{int(epoch)}:{int(visit_index)}:{sample_id}:{int(candidate_index)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def compute_candidate_reward(
    reference_text: str,
    prediction_raw: str,
    *,
    stop_reason: str | None = "eos",
    error: str | None = None,
    empty_penalty: float = 0.5,
    repeated_penalty: float = 0.5,
    reward_clip: tuple[float, float] = (-1.0, 1.0),
) -> dict[str, Any]:
    """Compute R = clip(1 - CER - 0.5*empty - 0.5*repeated, -1, 1) and components."""
    inference_error = bool(str(error or "").strip())
    scored = score_pair(
        reference_text,
        prediction_raw,
        inference_error=inference_error,
        stop_reason=stop_reason,
    )
    cer = float(scored["sample_cer"])
    empty_flag = 1.0 if scored["empty_output"] else 0.0
    repeated_flag = 1.0 if scored["repeated_output"] else 0.0
    empty_pen = float(empty_penalty) * empty_flag
    repeated_pen = float(repeated_penalty) * repeated_flag
    unclipped = 1.0 - cer - empty_pen - repeated_pen
    low, high = float(reward_clip[0]), float(reward_clip[1])
    if low >= high:
        raise ValueError("reward_clip requires low < high")
    clipped = max(low, min(high, unclipped))
    return {
        "reward": clipped,
        "unclipped_reward": unclipped,
        "cer": cer,
        "edit_count": int(scored["edit_count"]),
        "reference_char_count": int(scored["reference_char_count"]),
        "empty_output": bool(scored["empty_output"]),
        "repeated_output": bool(scored["repeated_output"]),
        "long_output": bool(scored["long_output"]),
        "truncated": bool(scored["truncated"]),
        "inference_error": bool(scored["inference_error"]),
        "valid_output": bool(scored["valid_output"]),
        "prediction_normalized": str(scored["prediction_normalized"]),
        "empty_penalty": empty_pen,
        "repeated_penalty": repeated_pen,
    }


def mean_greedy_reward(
    rows: list[dict[str, Any]],
    *,
    empty_penalty: float = 0.5,
    repeated_penalty: float = 0.5,
    reward_clip: tuple[float, float] = (-1.0, 1.0),
) -> float:
    """Compute the mean clipped greedy reward over val rows."""
    if not rows:
        raise ValueError("cannot compute mean greedy reward on empty rows")
    total = 0.0
    for row in rows:
        ref = str(row.get("reference_raw") if "reference_raw" in row else row.get("text", ""))
        pred = str(row.get("prediction_raw") or "")
        comp = compute_candidate_reward(
            ref,
            pred,
            stop_reason=row.get("stop_reason"),
            error=row.get("error"),
            empty_penalty=empty_penalty,
            repeated_penalty=repeated_penalty,
            reward_clip=reward_clip,
        )
        total += float(comp["reward"])
    return total / len(rows)


def evaluate_rollout_group(
    candidates: list[dict[str, Any]],
    reference_text: str,
    *,
    group_size: int = 4,
    empty_penalty: float = 0.5,
    repeated_penalty: float = 0.5,
    reward_clip: tuple[float, float] = (-1.0, 1.0),
) -> dict[str, Any]:
    """Score a rollout group of size G.

    Truncation (stop_reason == 'length') or inference error on any candidate
    invalidates the entire group; partial groups are never formed. Empty or
    repeated outputs that ended with 'eos' remain valid candidates.
    """
    if len(candidates) != int(group_size):
        raise ValueError(f"rollout group must have exactly {group_size} candidates, got {len(candidates)}")
    scored_candidates: list[dict[str, Any]] = []
    group_valid = True
    invalid_reasons: list[str] = []
    for index, cand in enumerate(candidates):
        stop_reason = cand.get("stop_reason")
        error = cand.get("error")
        has_error = bool(str(error or "").strip()) or stop_reason == "error"
        truncated = stop_reason == "length"
        if has_error:
            group_valid = False
            invalid_reasons.append(f"candidate_{index}_error")
        elif truncated:
            group_valid = False
            invalid_reasons.append(f"candidate_{index}_truncated")
        elif stop_reason != "eos":
            group_valid = False
            invalid_reasons.append(f"candidate_{index}_stop_{stop_reason}")
        comp = compute_candidate_reward(
            reference_text,
            str(cand.get("prediction_raw") or ""),
            stop_reason=stop_reason if isinstance(stop_reason, str) else None,
            error=str(error) if has_error else None,
            empty_penalty=empty_penalty,
            repeated_penalty=repeated_penalty,
            reward_clip=reward_clip,
        )
        scored_candidates.append(
            {
                **cand,
                "prediction_normalized": comp["prediction_normalized"],
                "reward_components": comp,
                "reward": comp["reward"] if (not has_error and not truncated and stop_reason == "eos") else None,
            }
        )
    if not group_valid:
        for item in scored_candidates:
            item["advantage"] = None
        return {
            "group_valid": False,
            "invalid_reasons": invalid_reasons,
            "zero_advantage": False,
            "mean_reward": None,
            "candidates": scored_candidates,
        }
    rewards = [float(item["reward"]) for item in scored_candidates]
    mean_r = sum(rewards) / len(rewards)
    zero_adv = all(r == rewards[0] for r in rewards)
    for item, reward in zip(scored_candidates, rewards):
        item["advantage"] = 0.0 if zero_adv else (reward - mean_r)
    return {
        "group_valid": True,
        "invalid_reasons": [],
        "zero_advantage": zero_adv,
        "mean_reward": mean_r,
        "candidates": scored_candidates,
    }


def k3_token_kl(policy_token_logp: list[float], ref_token_logp: list[float]) -> tuple[list[float], float]:
    """Compute per-token k3 = expm1(d) - d where d = logp_ref - logp_policy in FP32."""
    if not policy_token_logp or len(policy_token_logp) != len(ref_token_logp):
        raise ValueError("policy and reference token logprobs must be non-empty and equal length")
    values: list[float] = []
    for p_logp, r_logp in zip(policy_token_logp, ref_token_logp):
        p_val = float(p_logp)
        r_val = float(r_logp)
        if not (math.isfinite(p_val) and math.isfinite(r_val)):
            raise ValueError("non-finite token logprob in k3 computation")
        diff = r_val - p_val
        try:
            k3 = math.expm1(diff) - diff
        except OverflowError as exc:
            raise ValueError("overflow in k3_token_kl") from exc
        if not math.isfinite(k3):
            raise ValueError("non-finite k3 token KL")
        values.append(k3)
    return values, sum(values) / len(values)


def group_loss_from_token_logprobs(
    advantages: list[float],
    policy_token_logps: list[list[float]],
    ref_token_logps: list[list[float]],
    *,
    kl_beta: float,
    zero_advantage: bool = False,
) -> dict[str, Any]:
    """Compute L_group = mean_g[-A_g * mean_t(logp_policy) + kl_beta * mean_t(k3)].

    When all rewards in the group are equal (zero_advantage), both policy and
    KL losses for the entire group are zero so pure KL cannot pull the policy
    back toward the reference on zero-signal groups.
    """
    if not advantages or len(advantages) != len(policy_token_logps) or len(advantages) != len(ref_token_logps):
        raise ValueError("advantages, policy_token_logps, and ref_token_logps must have the same non-zero length")
    beta = float(kl_beta)
    if beta < 0.0 or not math.isfinite(beta):
        raise ValueError("kl_beta must be finite and >= 0")
    is_zero = bool(zero_advantage) or all(float(a) == 0.0 for a in advantages)
    if is_zero:
        count = len(advantages)
        return {
            "group_loss": 0.0,
            "policy_loss": 0.0,
            "kl_loss": 0.0,
            "k3_proxy_mean": 0.0,
            "zero_advantage": True,
            "candidate_losses": [0.0] * count,
            "candidate_k3_means": [0.0] * count,
        }
    cand_losses: list[float] = []
    cand_pg: list[float] = []
    cand_kl: list[float] = []
    cand_k3: list[float] = []
    for adv, p_tokens, r_tokens in zip(advantages, policy_token_logps, ref_token_logps):
        adv_val = float(adv)
        if not math.isfinite(adv_val):
            raise ValueError("non-finite advantage")
        if not p_tokens:
            raise ValueError("candidate has no supervised tokens")
        mean_logp = sum(float(x) for x in p_tokens) / len(p_tokens)
        if not math.isfinite(mean_logp):
            raise ValueError("non-finite policy mean_t(logp)")
        _, mean_k3 = k3_token_kl(p_tokens, r_tokens)
        pg_term = -adv_val * mean_logp
        kl_term = beta * mean_k3
        total_term = pg_term + kl_term
        if not math.isfinite(total_term):
            raise ValueError("non-finite candidate RL loss")
        cand_pg.append(pg_term)
        cand_kl.append(kl_term)
        cand_k3.append(mean_k3)
        cand_losses.append(total_term)
    n = len(cand_losses)
    return {
        "group_loss": sum(cand_losses) / n,
        "policy_loss": sum(cand_pg) / n,
        "kl_loss": sum(cand_kl) / n,
        "k3_proxy_mean": sum(cand_k3) / n,
        "zero_advantage": False,
        "candidate_losses": cand_losses,
        "candidate_k3_means": cand_k3,
    }


def aligned_group_mismatch_mask(
    candidates: list[dict[str, Any]],
    candidate_label_ids: list[list[int]],
    *,
    max_edit_count: int = 2,
) -> dict[str, Any]:
    """Isolate equal-token-length, punctuation-free substitution candidates within a rollout group.

    Selects the highest-reward candidate ``i_best`` as anchor. Finds all
    positive copies matching ``y_best`` and all negative candidates with the
    same supervised token count, no punctuation, and ``1 <= mismatches <= max_edit_count``
    sharing the same first mismatch position ``t_0``. Re-centers advantages over
    the aligned subset so ``sum(aligned_advantages) == 0`` and masks policy
    gradient to position ``t_0`` where the prefix context is 100% identical.
    """
    if not candidates or len(candidates) != len(candidate_label_ids):
        raise ValueError("candidates and candidate_label_ids must have the same non-zero length")
    if int(max_edit_count) < 1:
        raise ValueError("max_edit_count must be >= 1")
    best_idx = max(range(len(candidates)), key=lambda idx: float(candidates[idx]["reward"]))
    best_reward = float(candidates[best_idx]["reward"])
    best_pred = str(candidates[best_idx].get("prediction_raw") or "")
    best_ids = list(candidate_label_ids[best_idx])
    if len(best_ids) < 2 or _has_punctuation(best_pred):
        return {
            "aligned_valid": False,
            "best_index": best_idx,
            "mismatch_index": None,
            "token_mask": [],
            "aligned_indices": [],
            "aligned_advantages": [0.0] * len(candidates),
        }
    pos_indices: list[int] = []
    neg_candidates: list[tuple[int, int]] = []
    for idx, (cand, tok_ids) in enumerate(zip(candidates, candidate_label_ids)):
        pred = str(cand.get("prediction_raw") or "")
        if _has_punctuation(pred) or len(tok_ids) != len(best_ids):
            continue
        if list(tok_ids) == best_ids:
            pos_indices.append(idx)
            continue
        if tok_ids[-1] != best_ids[-1]:
            continue
        diff_positions = [pos for pos, (a, b) in enumerate(zip(tok_ids, best_ids)) if a != b]
        if 1 <= len(diff_positions) <= int(max_edit_count) and float(cand["reward"]) < best_reward - 1e-12:
            neg_candidates.append((idx, diff_positions[0]))
    if not pos_indices or not neg_candidates:
        return {
            "aligned_valid": False,
            "best_index": best_idx,
            "mismatch_index": None,
            "token_mask": [False] * len(best_ids),
            "aligned_indices": [],
            "aligned_advantages": [0.0] * len(candidates),
        }
    target_t0 = neg_candidates[0][1]
    neg_indices = [idx for idx, first_pos in neg_candidates if first_pos == target_t0]
    aligned_indices = sorted(pos_indices + neg_indices)
    mean_reward = sum(float(candidates[i]["reward"]) for i in aligned_indices) / len(aligned_indices)
    aligned_advantages = [
        (float(candidates[i]["reward"]) - mean_reward) if i in aligned_indices else 0.0
        for i in range(len(candidates))
    ]
    if all(abs(a) <= 1e-12 for a in aligned_advantages):
        return {
            "aligned_valid": False,
            "best_index": best_idx,
            "mismatch_index": None,
            "token_mask": [False] * len(best_ids),
            "aligned_indices": [],
            "aligned_advantages": [0.0] * len(candidates),
        }
    token_mask = [pos == target_t0 for pos in range(len(best_ids))]
    return {
        "aligned_valid": True,
        "best_index": best_idx,
        "mismatch_index": target_t0,
        "token_mask": token_mask,
        "aligned_indices": aligned_indices,
        "aligned_advantages": aligned_advantages,
    }


def accumulate_window_groups(group_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine groups in an accumulation window.

    All valid groups are weighted equally (including zero-advantage valid
    groups with loss 0). Invalid groups are excluded from the denominator.
    If every valid group in the window has zero advantage, optimizer and
    scheduler steps must be skipped.
    """
    valid_groups = [g for g in group_records if g.get("group_valid") is True]
    invalid_groups = [g for g in group_records if g.get("group_valid") is not True]
    if not valid_groups:
        return {
            "valid_group_count": 0,
            "invalid_group_count": len(invalid_groups),
            "signal_group_count": 0,
            "zero_advantage_group_count": 0,
            "should_step_optimizer": False,
            "window_loss": 0.0,
        }
    signal_groups = [g for g in valid_groups if not bool(g.get("zero_advantage", False))]
    zero_groups = [g for g in valid_groups if bool(g.get("zero_advantage", False))]
    if not signal_groups:
        return {
            "valid_group_count": len(valid_groups),
            "invalid_group_count": len(invalid_groups),
            "signal_group_count": 0,
            "zero_advantage_group_count": len(zero_groups),
            "should_step_optimizer": False,
            "window_loss": 0.0,
        }
    total_loss = sum(0.0 if g.get("zero_advantage") else float(g["group_loss"]) for g in valid_groups)
    return {
        "valid_group_count": len(valid_groups),
        "invalid_group_count": len(invalid_groups),
        "signal_group_count": len(signal_groups),
        "zero_advantage_group_count": len(zero_groups),
        "should_step_optimizer": True,
        "window_loss": total_loss / len(valid_groups),
    }


def exact_conditional_vocab_kl(
    policy_log_probs: list[list[float]],
    ref_log_probs: list[list[float]],
) -> float:
    """Exact token-averaged conditional KL sum_v p_policy(v)*(log p_policy(v) - log p_ref(v))."""
    if not policy_log_probs or len(policy_log_probs) != len(ref_log_probs):
        raise ValueError("policy and reference log_prob tables must be non-empty and equal length")
    total_kl = 0.0
    for p_row, r_row in zip(policy_log_probs, ref_log_probs):
        if not p_row or len(p_row) != len(r_row):
            raise ValueError("vocabulary dimension mismatch in conditional KL")
        token_kl = 0.0
        for lp, lr in zip(p_row, r_row):
            lp_f = float(lp)
            lr_f = float(lr)
            if not (math.isfinite(lp_f) and math.isfinite(lr_f)):
                raise ValueError("non-finite log_prob in conditional KL")
            prob = math.exp(lp_f)
            if prob > 0.0:
                token_kl += prob * (lp_f - lr_f)
        if not math.isfinite(token_kl):
            raise ValueError("non-finite token conditional KL")
        total_kl += max(0.0, token_kl)
    return total_kl / len(policy_log_probs)


def exact_step0_conditional_kl_tensor(
    policy_logits: Any,
    ref_logits: Any,
    labels: Any,
    *,
    chunk_size: int = 128,
) -> tuple[float, int]:
    """Compute exact vocabulary-summed KL on supervised positions in FP32 chunks."""
    import torch

    shift_policy = policy_logits[:, :-1, :]
    shift_ref = ref_logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels.reshape(-1) != -100
    indices = mask.nonzero(as_tuple=False).reshape(-1)
    count = int(indices.numel())
    if count == 0:
        raise ValueError("step-0 prefix has no supervised tokens")
    flat_policy = shift_policy.reshape(-1, shift_policy.size(-1)).index_select(0, indices)
    flat_ref = shift_ref.reshape(-1, shift_ref.size(-1)).index_select(0, indices)
    kl_sum = 0.0
    for start in range(0, count, max(1, int(chunk_size))):
        end = min(count, start + max(1, int(chunk_size)))
        logp_pol = torch.log_softmax(flat_policy[start:end].float(), dim=-1)
        logp_ref = torch.log_softmax(flat_ref[start:end].float(), dim=-1)
        prob_pol = torch.exp(logp_pol)
        token_kl = (prob_pol * (logp_pol - logp_ref)).sum(dim=-1)
        if not torch.isfinite(token_kl).all():
            raise ValueError("non-finite token KL in exact_step0_conditional_kl_tensor")
        kl_sum += float(torch.clamp(token_kl, min=0.0).sum().item())
    return kl_sum / count, count


def prefix_records_sha256(prefix_records: list[dict[str, Any]]) -> str:
    """Hash fixed step-0 greedy prefix token IDs across val."""
    normalized = [
        {
            "sample_id": str(item["sample_id"]),
            "token_ids": [int(t) for t in item["token_ids"]],
            "prediction_raw": str(item.get("prediction_raw", "")),
        }
        for item in prefix_records
    ]
    return sha256_text(canonical_json(normalized))


def reference_tensors_sha256(named_tensors: dict[str, Any]) -> str:
    """Compute a deterministic SHA-256 over frozen reference LoRA tensors."""
    if not named_tensors:
        raise ValueError("reference tensor map is empty")
    digest = hashlib.sha256()
    for name in sorted(named_tensors):
        tensor = named_tensors[name]
        digest.update(name.encode("utf-8"))
        if hasattr(tensor, "detach"):
            cpu_tensor = tensor.detach().cpu().contiguous()
            digest.update(str(tuple(cpu_tensor.shape)).encode("utf-8"))
            digest.update(str(cpu_tensor.dtype).encode("utf-8"))
            raw_bytes = bytes(cpu_tensor.untyped_storage())
            digest.update(raw_bytes)
        elif isinstance(tensor, (bytes, bytearray)):
            digest.update(bytes(tensor))
        else:
            raise TypeError(f"unsupported reference tensor type for {name}: {type(tensor)}")
    return digest.hexdigest()


def judge_rl_checkpoint(
    metrics: dict[str, Any],
    parent_metrics: dict[str, Any],
    base_metrics: dict[str, Any],
    gate_cfg: dict[str, Any],
    *,
    step0_reward: float,
    current_reward: float,
    raw_kl: float,
    reference_matched: bool,
) -> dict[str, Any]:
    """Judge an RL val checkpoint against DPO parent, base clean cap, step-0 reward, and raw_kl."""
    from train.dpo_contract import judge_dpo_checkpoint

    dummy_pref = {
        "accuracy": 1.0,
        "pairs": int(gate_cfg.get("min_val_pairs", 100)),
        "wins": int(gate_cfg.get("min_val_pairs", 100)),
        "ties": 0,
        "raw_policy_accuracy": 1.0,
        "length_buckets": {},
    }
    base_decision = judge_dpo_checkpoint(metrics, parent_metrics, base_metrics, gate_cfg, dummy_pref)
    risks = [r for r in base_decision["risks"] if "preference" not in r and "val pairs" not in r]
    if not reference_matched:
        risks.append("reference LoRA tensor checksum changed")
    if not math.isfinite(raw_kl) or raw_kl > float(gate_cfg["raw_kl_max"]):
        risks.append(f"raw_kl {raw_kl} exceeds raw_kl_max {gate_cfg['raw_kl_max']}")
    reward_delta = float(current_reward) - float(step0_reward)
    if not math.isfinite(reward_delta):
        risks.append("non-finite reward_delta")
    elif reward_delta < -float(gate_cfg["reward_drop_stop"]):
        risks.append(f"reward dropped by {-reward_delta} (> {gate_cfg['reward_drop_stop']})")
    risk_stop = len(risks) > 0
    reward_passed = math.isfinite(reward_delta) and reward_delta >= float(gate_cfg["min_reward_improvement"])
    cer_passed = (
        not risk_stop
        and int(base_decision["improved_degraded_scenarios"]) >= int(gate_cfg["min_improved_degraded_scenarios"])
        and all(
            int(metrics["by_scenario"].get(name, {}).get("samples", 0)) >= int(gate_cfg["min_rows_per_scenario"])
            for name in SCENARIOS
        )
    )
    passed = cer_passed and reward_passed
    return {
        "passed": passed,
        "risk_stop": risk_stop,
        "risks": risks,
        "micro_cer": float(metrics["micro_cer"]),
        "clean_cer": base_decision["clean_cer"],
        "clean_delta_vs_parent": base_decision["clean_delta_vs_parent"],
        "clean_delta_vs_base": base_decision["clean_delta_vs_base"],
        "degraded_macro_cer": base_decision["degraded_macro_cer"],
        "parent_degraded_macro_cer": base_decision["parent_degraded_macro_cer"],
        "degraded_macro_delta": base_decision["macro_delta"],
        "improved_degraded_scenarios": base_decision["improved_degraded_scenarios"],
        "scenario_deltas": base_decision["degraded_deltas"],
        "step0_reward": float(step0_reward),
        "mean_greedy_reward": float(current_reward),
        "reward_delta_vs_step0": reward_delta,
        "reward_passed": reward_passed,
        "raw_kl": float(raw_kl),
        "reference_matched": bool(reference_matched),
    }
