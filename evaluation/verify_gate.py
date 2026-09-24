#!/usr/bin/env python3
"""Evaluate and verify model gates across training stages (SFT Pilot, DPO Pilot, RL Pilot).

Loads base and pilot evaluation metrics, compares degraded scenario improvements,
clean retention, valid output rate, and failure rates against contract thresholds,
and exports a machine-readable gate.json containing provenance hashes.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON from path."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_gate(
    base_metrics: Dict[str, Any],
    pilot_metrics: Dict[str, Any],
    sft_metrics: Optional[Dict[str, Any]] = None,
    dpo_metrics: Optional[Dict[str, Any]] = None,
    stage: str = "sft_pilot",
    min_degraded_improvements: int = 1,
    max_clean_regression: float = 0.02,
    max_cumulative_clean_regression: float = 0.025,
    max_robust_macro_regression: Optional[float] = None,
    min_valid_output_rate: float = 0.95,
    max_empty_output_rate: float = 0.002,
    max_failure_increase: float = 0.05,
    preference_accuracy: Optional[float] = None,
    min_preference_accuracy: float = 0.55,
    reward_improvement: Optional[float] = None,
    min_reward_improvement: float = 0.002,
    zero_variance_ratio: Optional[float] = None,
    max_zero_variance_ratio: float = 0.75,
    manifest_path: Optional[Path] = None,
    base_predictions_path: Optional[Path] = None,
    pilot_predictions_path: Optional[Path] = None,
    sft_predictions_path: Optional[Path] = None,
    dpo_predictions_path: Optional[Path] = None,
    base_metrics_path: Optional[Path] = None,
    pilot_metrics_path: Optional[Path] = None,
    sft_metrics_path: Optional[Path] = None,
    dpo_metrics_path: Optional[Path] = None,
    dpo_val_manifest_path: Optional[Path] = None,
    dpo_loss_log_path: Optional[Path] = None,
    rl_val_manifest_path: Optional[Path] = None,
    rl_loss_log_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compare base, sft, dpo, and pilot metrics and determine gate status."""
    if max_robust_macro_regression is None:
        max_robust_macro_regression = 0.0 if stage in ("dpo_pilot", "rl_pilot") else 0.005
    base_overall = base_metrics.get("overall", {})
    pilot_overall = pilot_metrics.get("overall", {})
    # For DPO, reference is SFT; for RL, reference is DPO (falling back to SFT or Base)
    if stage == "rl_pilot":
        ref_metrics = dpo_metrics if dpo_metrics is not None else (sft_metrics if sft_metrics is not None else base_metrics)
    elif stage == "dpo_pilot":
        ref_metrics = sft_metrics if sft_metrics is not None else base_metrics
    else:
        ref_metrics = base_metrics
    ref_overall = ref_metrics.get("overall", {})

    # 1. Degraded scenario improvements (relative to ref: SFT for DPO, Base for SFT)
    ref_scenarios = {
        item["group"]: item["error_rate"]
        for item in ref_metrics.get("by_scenario", [])
        if item.get("group") and item.get("error_rate") is not None
    }
    pilot_scenarios = {
        item["group"]: item["error_rate"]
        for item in pilot_metrics.get("by_scenario", [])
        if item.get("group") and item.get("error_rate") is not None
    }

    improved_scenarios: List[str] = []
    for group, ref_rate in sorted(ref_scenarios.items()):
        # Filter only degraded scenarios (non-clean)
        scenario = group.split("|")[-1] if "|" in group else group
        if scenario == "clean":
            continue
        pilot_rate = pilot_scenarios.get(group)
        if pilot_rate is not None and pilot_rate < ref_rate:
            improved_scenarios.append(group)

    improvements_count = len(improved_scenarios)
    degraded_passed = improvements_count >= min_degraded_improvements

    # 2. Clean error rate increase (relative to ref stage)
    ref_clean_macro = ref_overall.get("clean_language_macro_error_rate")
    base_clean_macro = base_overall.get("clean_language_macro_error_rate")
    pilot_clean_macro = pilot_overall.get("clean_language_macro_error_rate")

    clean_increase = 0.0
    if ref_clean_macro is not None and pilot_clean_macro is not None:
        clean_increase = round(pilot_clean_macro - ref_clean_macro, 6)
    clean_passed = clean_increase <= max_clean_regression

    # Cumulative clean increase relative to Base
    cumulative_clean_increase = 0.0
    if base_clean_macro is not None and pilot_clean_macro is not None:
        cumulative_clean_increase = round(pilot_clean_macro - base_clean_macro, 6)
    cumulative_clean_passed = (
        cumulative_clean_increase <= max_cumulative_clean_regression
        if stage in ("dpo_pilot", "rl_pilot")
        else True
    )

    # Per-language clean breakdown
    lang_clean_breakdown: Dict[str, Any] = {}
    for group in ("en|clean", "zh|clean"):
        if group in ref_scenarios and group in pilot_scenarios:
            lang = group.split("|")[0]
            r_rate = ref_scenarios[group]
            p_rate = pilot_scenarios[group]
            lang_clean_breakdown[lang] = {
                "ref_clean_error_rate": r_rate,
                "pilot_clean_error_rate": p_rate,
                "increase": round(p_rate - r_rate, 6),
            }

    # 3. Robust macro regression (relative to ref)
    ref_robust_macro = ref_overall.get("robust_language_macro_error_rate")
    pilot_robust_macro = pilot_overall.get("robust_language_macro_error_rate")
    robust_increase = 0.0
    if ref_robust_macro is not None and pilot_robust_macro is not None:
        robust_increase = round(pilot_robust_macro - ref_robust_macro, 6)
        robust_passed = robust_increase <= max_robust_macro_regression
    else:
        robust_passed = True

    # 4. Preference accuracy for DPO (strictly mandatory for dpo_pilot)
    if stage == "dpo_pilot":
        preference_passed = (preference_accuracy is not None) and (preference_accuracy >= min_preference_accuracy)
    else:
        preference_passed = True

    # 5. Held-out reward improvement for RL (strictly mandatory for rl_pilot)
    if stage == "rl_pilot":
        reward_passed = (reward_improvement is not None) and (reward_improvement >= min_reward_improvement)
        zero_variance_passed = (zero_variance_ratio is None) or (zero_variance_ratio <= max_zero_variance_ratio)
    else:
        reward_passed = True
        zero_variance_passed = True

    # 6. Valid output rate, empty output rate & failure rate
    pilot_samples = int(pilot_overall.get("samples", 0))
    pilot_infer_errors = int(pilot_overall.get("inference_errors", 0))
    pilot_empty_outputs = int(pilot_overall.get("empty_outputs", 0))

    if pilot_samples > 0:
        valid_output_rate = round(1.0 - (pilot_infer_errors + pilot_empty_outputs) / pilot_samples, 6)
        pilot_failure_rate = round((pilot_infer_errors + pilot_empty_outputs) / pilot_samples, 6)
        pilot_empty_rate = round(pilot_empty_outputs / pilot_samples, 6)
    else:
        valid_output_rate = 1.0
        pilot_failure_rate = 0.0
        pilot_empty_rate = 0.0

    valid_output_passed = valid_output_rate >= min_valid_output_rate
    empty_output_passed = pilot_empty_rate <= max_empty_output_rate

    ref_samples = int(ref_overall.get("samples", 0))
    ref_infer_errors = int(ref_overall.get("inference_errors", 0))
    ref_empty_outputs = int(ref_overall.get("empty_outputs", 0))
    ref_failure_rate = (
        round((ref_infer_errors + ref_empty_outputs) / ref_samples, 6)
        if ref_samples > 0
        else 0.0
    )
    base_failure_rate = ref_failure_rate

    failure_rate_increase = round(pilot_failure_rate - ref_failure_rate, 6)
    failure_rate_passed = failure_rate_increase <= max_failure_increase

    overall_passed = (
        degraded_passed
        and clean_passed
        and cumulative_clean_passed
        and robust_passed
        and preference_passed
        and reward_passed
        and zero_variance_passed
        and valid_output_passed
        and empty_output_passed
        and failure_rate_passed
    )

    checks = {
        "degraded_improvement": "PASSED" if degraded_passed else "FAILED",
        "clean_retention": "PASSED" if clean_passed else "FAILED",
        "robust_retention": "PASSED" if robust_passed else "FAILED",
        "valid_output_rate": "PASSED" if valid_output_passed else "FAILED",
        "empty_output_rate": "PASSED" if empty_output_passed else "FAILED",
        "failure_rate": "PASSED" if failure_rate_passed else "FAILED",
    }
    if stage in ("dpo_pilot", "rl_pilot"):
        checks["clean_cumulative_retention"] = "PASSED" if cumulative_clean_passed else "FAILED"
    if stage == "dpo_pilot":
        checks["preference_accuracy"] = "PASSED" if preference_passed else "FAILED"
    if stage == "rl_pilot":
        checks["held_out_reward"] = "PASSED" if reward_passed else "FAILED"
        checks["zero_variance"] = "PASSED" if zero_variance_passed else "FAILED"

    provenance: Dict[str, Any] = {
        "manifest": str(manifest_path) if manifest_path else "",
        "manifest_sha256": compute_file_sha256(manifest_path) if manifest_path else "",
        "base_metrics_path": str(base_metrics_path) if base_metrics_path else "",
        "base_metrics_sha256": compute_file_sha256(base_metrics_path) if base_metrics_path else "",
        "base_predictions_path": str(base_predictions_path) if base_predictions_path else "",
        "base_predictions_sha256": compute_file_sha256(base_predictions_path) if base_predictions_path else "",
        "sft_metrics_path": str(sft_metrics_path) if sft_metrics_path else "",
        "sft_metrics_sha256": compute_file_sha256(sft_metrics_path) if sft_metrics_path else "",
        "sft_predictions_path": str(sft_predictions_path) if sft_predictions_path else "",
        "sft_predictions_sha256": compute_file_sha256(sft_predictions_path) if sft_predictions_path else "",
        "pilot_metrics_path": str(pilot_metrics_path) if pilot_metrics_path else "",
        "pilot_metrics_sha256": compute_file_sha256(pilot_metrics_path) if pilot_metrics_path else "",
        "pilot_predictions_path": str(pilot_predictions_path) if pilot_predictions_path else "",
        "pilot_predictions_sha256": compute_file_sha256(pilot_predictions_path) if pilot_predictions_path else "",
    }
    if dpo_metrics_path:
        provenance["dpo_metrics_path"] = str(dpo_metrics_path)
        provenance["dpo_metrics_sha256"] = compute_file_sha256(dpo_metrics_path)
    if dpo_predictions_path:
        provenance["dpo_predictions_path"] = str(dpo_predictions_path)
        provenance["dpo_predictions_sha256"] = compute_file_sha256(dpo_predictions_path)
    if dpo_val_manifest_path:
        provenance["dpo_val_manifest_path"] = str(dpo_val_manifest_path)
        provenance["dpo_val_manifest_sha256"] = compute_file_sha256(dpo_val_manifest_path)
    if dpo_loss_log_path:
        provenance["dpo_loss_log_path"] = str(dpo_loss_log_path)
        provenance["dpo_loss_log_sha256"] = compute_file_sha256(dpo_loss_log_path)
    if rl_val_manifest_path:
        provenance["rl_val_manifest_path"] = str(rl_val_manifest_path)
        provenance["rl_val_manifest_sha256"] = compute_file_sha256(rl_val_manifest_path)
    if rl_loss_log_path:
        provenance["rl_loss_log_path"] = str(rl_loss_log_path)
        provenance["rl_loss_log_sha256"] = compute_file_sha256(rl_loss_log_path)

    thresholds_dict: Dict[str, Any] = {
        "min_degraded_scenario_improvements": min_degraded_improvements,
        "clean_error_rate_increase_max": max_clean_regression,
        "robust_error_rate_increase_max": max_robust_macro_regression,
        "valid_output_rate_min": min_valid_output_rate,
        "empty_output_rate_max": max_empty_output_rate,
        "failure_rate_increase_max": max_failure_increase,
    }
    if stage in ("dpo_pilot", "rl_pilot"):
        thresholds_dict["clean_cumulative_error_rate_increase_max"] = max_cumulative_clean_regression
    if stage == "dpo_pilot":
        thresholds_dict["preference_accuracy_min"] = min_preference_accuracy
    if stage == "rl_pilot":
        thresholds_dict["reward_improvement_min"] = min_reward_improvement
        thresholds_dict["zero_variance_ratio_max"] = max_zero_variance_ratio

    gate_record = {
        "stage": stage,
        "gate_status": "PASSED" if overall_passed else "FAILED",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "thresholds": thresholds_dict,
        "metrics": {
            "degraded_scenario_improvements_count": improvements_count,
            "improved_scenarios": improved_scenarios,
            "clean_error_rate_increase": clean_increase,
            "robust_error_rate_increase": robust_increase,
            "preference_accuracy": preference_accuracy,
            "held_out_reward_improvement": reward_improvement,
            "zero_variance_ratio": zero_variance_ratio,
            "valid_output_rate": valid_output_rate,
            "empty_output_rate": pilot_empty_rate,
            "failure_rate_increase": failure_rate_increase,
            "ref_clean_macro": ref_clean_macro,
            "ref_robust_macro": ref_robust_macro,
            "base_clean_macro": base_clean_macro,
            "base_robust_macro": base_overall.get("robust_language_macro_error_rate"),
            "sft_clean_macro": sft_metrics.get("overall", {}).get("clean_language_macro_error_rate") if sft_metrics else None,
            "sft_robust_macro": sft_metrics.get("overall", {}).get("robust_language_macro_error_rate") if sft_metrics else None,
            "dpo_clean_macro": dpo_metrics.get("overall", {}).get("clean_language_macro_error_rate") if dpo_metrics else None,
            "dpo_robust_macro": dpo_metrics.get("overall", {}).get("robust_language_macro_error_rate") if dpo_metrics else None,
            "pilot_clean_macro": pilot_clean_macro,
            "pilot_robust_macro": pilot_overall.get("robust_language_macro_error_rate"),
            "base_failure_rate": base_failure_rate,
            "pilot_failure_rate": pilot_failure_rate,
            "language_clean_breakdown": lang_clean_breakdown,
        },
        "checks": checks,
        "provenance": provenance,
    }

    return gate_record


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for gate verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-metrics", required=True, help="Path to Base metrics.json")
    parser.add_argument("--pilot-metrics", required=True, help="Path to Pilot metrics.json")
    parser.add_argument("--base-predictions", default=None, help="Path to Base predictions.jsonl (optional)")
    parser.add_argument("--pilot-predictions", default=None, help="Path to Pilot predictions.jsonl (optional)")
    parser.add_argument("--manifest", default=None, help="Path to evaluation manifest (optional)")
    parser.add_argument("--stage", default="sft_pilot", help="Evaluation stage tag (default: sft_pilot)")
    parser.add_argument("--output", required=True, help="Path to write gate.json")
    parser.add_argument(
        "--min-degraded-improvements",
        type=int,
        default=1,
        help="Minimum number of improved degraded scenarios (default: 1)",
    )
    parser.add_argument(
        "--max-clean-regression",
        type=float,
        default=0.02,
        help="Maximum allowed increase in clean macro error rate (default: 0.02)",
    )
    parser.add_argument(
        "--max-robust-regression",
        type=float,
        default=None,
        help="Maximum allowed increase in robust macro error rate (default: 0.0 for DPO/RL, 0.005 for SFT)",
    )
    parser.add_argument(
        "--min-valid-output-rate",
        type=float,
        default=0.95,
        help="Minimum valid output rate (default: 0.95)",
    )
    parser.add_argument(
        "--max-empty-rate",
        type=float,
        default=0.002,
        help="Maximum allowed empty output rate (default: 0.002)",
    )
    parser.add_argument(
        "--max-failure-increase",
        type=float,
        default=0.05,
        help="Maximum allowed increase in failure rate (default: 0.05)",
    )
    parser.add_argument("--sft-metrics", default=None, help="Path to SFT metrics.json (for DPO/RL stage comparison)")
    parser.add_argument("--sft-predictions", default=None, help="Path to SFT predictions.jsonl (optional)")
    parser.add_argument(
        "--preference-accuracy",
        type=float,
        default=None,
        help="Held-out preference validation accuracy (for DPO stage)",
    )
    parser.add_argument(
        "--min-preference-accuracy",
        type=float,
        default=0.55,
        help="Minimum required preference accuracy for DPO (default: 0.55)",
    )
    parser.add_argument(
        "--max-cumulative-clean-regression",
        type=float,
        default=0.025,
        help="Maximum allowed cumulative clean regression relative to Base (default: 0.025)",
    )
    parser.add_argument(
        "--dpo-loss-log",
        type=str,
        default=None,
        help="Path to DPO training loss_log.jsonl to auto-extract held-out preference accuracy",
    )
    parser.add_argument(
        "--dpo-val-manifest",
        type=str,
        default=None,
        help="Path to DPO held-out validation manifest (val_dpo_pairs.jsonl) for provenance recording",
    )
    parser.add_argument(
        "--dpo-step",
        type=int,
        default=None,
        help="Target step number in dpo_loss_log to extract val_preference_accuracy for (default: last step)",
    )
    parser.add_argument("--dpo-metrics", default=None, help="Path to DPO metrics.json (for RL stage comparison)")
    parser.add_argument("--dpo-predictions", default=None, help="Path to DPO predictions.jsonl (optional)")
    parser.add_argument(
        "--reward-improvement",
        type=float,
        default=None,
        help="Held-out reward improvement for RL stage",
    )
    parser.add_argument(
        "--min-reward-improvement",
        type=float,
        default=0.002,
        help="Minimum required held-out reward improvement for RL (default: 0.002)",
    )
    parser.add_argument(
        "--rl-loss-log",
        type=str,
        default=None,
        help="Path to RL training loss_log.jsonl to auto-extract held-out reward improvement",
    )
    parser.add_argument(
        "--rl-val-manifest",
        type=str,
        default=None,
        help="Path to RL held-out validation manifest (rl_val_pool.jsonl) for provenance recording",
    )
    parser.add_argument(
        "--rl-step",
        type=int,
        default=None,
        help="Target step number in rl_loss_log to extract held-out validation reward for (default: last step)",
    )
    parser.add_argument(
        "--max-zero-variance-ratio",
        type=float,
        default=0.75,
        help="Maximum allowed zero-variance group ratio for RL (default: 0.75)",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Main execution function."""
    args = build_parser().parse_args(argv)

    base_metrics_path = Path(args.base_metrics).resolve()
    pilot_metrics_path = Path(args.pilot_metrics).resolve()

    if not base_metrics_path.is_file():
        print(f"Error: Base metrics file not found: {base_metrics_path}", file=sys.stderr)
        return 1
    if not pilot_metrics_path.is_file():
        print(f"Error: Pilot metrics file not found: {pilot_metrics_path}", file=sys.stderr)
        return 1

    base_metrics = load_json(base_metrics_path)
    pilot_metrics = load_json(pilot_metrics_path)

    sft_metrics_path = Path(args.sft_metrics).resolve() if args.sft_metrics else None
    sft_metrics = load_json(sft_metrics_path) if sft_metrics_path and sft_metrics_path.is_file() else None

    dpo_metrics_path = Path(args.dpo_metrics).resolve() if args.dpo_metrics else None
    dpo_metrics = load_json(dpo_metrics_path) if dpo_metrics_path and dpo_metrics_path.is_file() else None

    # Auto-extract preference_accuracy from dpo_loss_log if not explicitly provided
    preference_accuracy = args.preference_accuracy
    if args.stage == "dpo_pilot" and preference_accuracy is None and args.dpo_loss_log:
        loss_log_p = Path(args.dpo_loss_log).resolve()
        if loss_log_p.is_file():
            with open(loss_log_p, "r", encoding="utf-8") as f_ll:
                for line in f_ll:
                    line_s = line.strip()
                    if line_s:
                        try:
                            rec = json.loads(line_s)
                            step_val = rec.get("global_step", rec.get("step"))
                            if args.dpo_step is not None and step_val != args.dpo_step:
                                continue
                            if "val_preference_accuracy" in rec and rec["val_preference_accuracy"] is not None:
                                preference_accuracy = float(rec["val_preference_accuracy"])
                        except Exception:
                            pass

    # Auto-extract reward_improvement and zero_variance_ratio from rl_loss_log
    reward_improvement = args.reward_improvement
    zero_variance_ratio = None
    if args.stage == "rl_pilot" and args.rl_loss_log:
        rl_loss_log_p = Path(args.rl_loss_log).resolve()
        if rl_loss_log_p.is_file():
            val_records: Dict[int, Dict[str, Any]] = {}
            train_zero_vars: List[float] = []
            with open(rl_loss_log_p, "r", encoding="utf-8") as f_rll:
                for line in f_rll:
                    line_s = line.strip()
                    if line_s:
                        try:
                            rec = json.loads(line_s)
                            # Validation record: strictly require val_eval_scope == "Full Held-out"
                            if "val_mean_reward" in rec and rec["val_mean_reward"] is not None:
                                scope = rec.get("val_eval_scope", "")
                                step_v = rec.get("global_step", rec.get("step"))
                                if scope == "Full Held-out" and step_v is not None:
                                    val_records[int(step_v)] = rec
                            # Training record with zero-variance ratio
                            if "zero_variance_ratio" in rec and rec["zero_variance_ratio"] is not None:
                                train_zero_vars.append(float(rec["zero_variance_ratio"]))
                        except Exception:
                            pass

            if train_zero_vars:
                zero_variance_ratio = round(sum(train_zero_vars) / len(train_zero_vars), 4)

            if reward_improvement is None:
                # Strict same-scope validation: require Step 0 baseline on Full Held-out
                if 0 in val_records:
                    step0_reward = float(val_records[0]["val_mean_reward"])
                    target_step = args.rl_step
                    if target_step is not None:
                        if target_step in val_records:
                            reward_improvement = round(float(val_records[target_step]["val_mean_reward"]) - step0_reward, 4)
                    else:
                        non_zero_steps = sorted([s for s in val_records.keys() if s > 0])
                        if non_zero_steps:
                            last_step = non_zero_steps[-1]
                            reward_improvement = round(float(val_records[last_step]["val_mean_reward"]) - step0_reward, 4)

    manifest_p = Path(args.manifest).resolve() if args.manifest else None
    base_pred_p = Path(args.base_predictions).resolve() if args.base_predictions else None
    pilot_pred_p = Path(args.pilot_predictions).resolve() if args.pilot_predictions else None
    sft_pred_p = Path(args.sft_predictions).resolve() if args.sft_predictions else None
    dpo_pred_p = Path(args.dpo_predictions).resolve() if args.dpo_predictions else None
    dpo_val_manifest_p = Path(args.dpo_val_manifest).resolve() if args.dpo_val_manifest else None
    dpo_loss_log_p = Path(args.dpo_loss_log).resolve() if args.dpo_loss_log else None
    rl_val_manifest_p = Path(args.rl_val_manifest).resolve() if args.rl_val_manifest else None
    rl_loss_log_p = Path(args.rl_loss_log).resolve() if args.rl_loss_log else None
    output_p = Path(args.output).resolve()

    gate_record = evaluate_gate(
        base_metrics=base_metrics,
        pilot_metrics=pilot_metrics,
        sft_metrics=sft_metrics,
        dpo_metrics=dpo_metrics,
        stage=args.stage,
        min_degraded_improvements=args.min_degraded_improvements,
        max_clean_regression=args.max_clean_regression,
        max_cumulative_clean_regression=args.max_cumulative_clean_regression,
        max_robust_macro_regression=args.max_robust_regression,
        min_valid_output_rate=args.min_valid_output_rate,
        max_empty_output_rate=args.max_empty_rate,
        max_failure_increase=args.max_failure_increase,
        preference_accuracy=preference_accuracy,
        min_preference_accuracy=args.min_preference_accuracy,
        reward_improvement=reward_improvement,
        min_reward_improvement=args.min_reward_improvement,
        zero_variance_ratio=zero_variance_ratio,
        max_zero_variance_ratio=args.max_zero_variance_ratio,
        manifest_path=manifest_p,
        base_predictions_path=base_pred_p,
        pilot_predictions_path=pilot_pred_p,
        sft_predictions_path=sft_pred_p,
        dpo_predictions_path=dpo_pred_p,
        base_metrics_path=base_metrics_path,
        pilot_metrics_path=pilot_metrics_path,
        sft_metrics_path=sft_metrics_path,
        dpo_metrics_path=dpo_metrics_path,
        dpo_val_manifest_path=dpo_val_manifest_p,
        dpo_loss_log_path=dpo_loss_log_p,
        rl_val_manifest_path=rl_val_manifest_p,
        rl_loss_log_path=rl_loss_log_p,
    )

    output_p.parent.mkdir(parents=True, exist_ok=True)
    with output_p.open("w", encoding="utf-8") as f:
        json.dump(gate_record, f, indent=2, ensure_ascii=False)

    print(f"Gate verification written to: {output_p}")
    print(f"Stage: {gate_record['stage']}")
    print(f"Status: {gate_record['gate_status']}")
    print(f"Checks: {json.dumps(gate_record['checks'], ensure_ascii=False)}")
    print(
        f"Metrics: improved_scenarios={gate_record['metrics']['degraded_scenario_improvements_count']} "
        f"({gate_record['metrics']['improved_scenarios']}), "
        f"clean_reg={gate_record['metrics']['clean_error_rate_increase']}, "
        f"valid_rate={gate_record['metrics']['valid_output_rate']}, "
        f"failure_inc={gate_record['metrics']['failure_rate_increase']}"
    )

    return 0 if gate_record["gate_status"] == "PASSED" else 1


if __name__ == "__main__":
    sys.exit(main())
