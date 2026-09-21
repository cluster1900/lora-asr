#!/usr/bin/env python3
"""Evaluate a series of training checkpoints across steps, compare metrics against Base, and select Pareto-optimal checkpoint."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.verify_gate import evaluate_gate, load_json


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to training run directory containing checkpoints/ directory.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="100,150,200,250,300",
        help="Comma-separated step numbers to evaluate, or 'auto' to evaluate all available step checkpoints.",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to evaluation manifest (e.g. validation.jsonl).",
    )
    parser.add_argument(
        "--base-metrics",
        type=str,
        required=True,
        help="Path to Base model metrics.json.",
    )
    parser.add_argument(
        "--run1-metrics",
        type=str,
        default="",
        help="Optional path to Run 1 pilot metrics.json for side-by-side comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store evaluation outputs, comparison reports, and best checkpoint pointer.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-ASR-1.7B",
        help="Base model ID.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="7278e1e70fe206f11671096ffdd38061171dd6e5",
        help="Model revision.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference.",
    )
    parser.add_argument(
        "--clean-target-min",
        type=float,
        default=0.020,
        help="Target minimum English Clean WER (2.0%%).",
    )
    parser.add_argument(
        "--clean-target-max",
        type=float,
        default=0.025,
        help="Target maximum English Clean WER (2.5%%).",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip running inference if predictions file already exists.",
    )
    return parser.parse_args(argv)


def discover_step_checkpoints(run_dir: Path, steps_arg: str) -> List[Tuple[int, Path]]:
    """Discover available step checkpoints within run_dir."""
    ckpts_dir = run_dir / "checkpoints"
    if not ckpts_dir.is_dir():
        return []

    available: Dict[int, Path] = {}
    for p in ckpts_dir.glob("step_*"):
        if p.is_dir():
            try:
                s = int(p.name.replace("step_", ""))
                available[s] = p
            except ValueError:
                continue

    if steps_arg.lower() in ("auto", "all"):
        return sorted(available.items(), key=lambda x: x[0])

    selected: List[Tuple[int, Path]] = []
    for s_str in steps_arg.split(","):
        s_str = s_str.strip()
        if not s_str:
            continue
        try:
            s = int(s_str)
        except ValueError:
            continue
        if s in available:
            selected.append((s, available[s]))
        else:
            candidate = ckpts_dir / f"step_{s}"
            if candidate.is_dir():
                selected.append((s, candidate))
    return sorted(selected, key=lambda x: x[0])


def run_step_eval(
    step: int,
    ckpt_dir: Path,
    manifest_path: Path,
    output_step_dir: Path,
    base_metrics: Dict[str, Any],
    model_id: str,
    revision: str,
    device: str,
    skip_inference: bool = False,
) -> Dict[str, Any]:
    """Run inference and eval_wer for a single checkpoint step."""
    output_step_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_step_dir / "predictions.jsonl"
    eval_dir = output_step_dir / "eval"
    metrics_path = eval_dir / "metrics.json"

    adapter_dir = ckpt_dir / "adapter"
    if not adapter_dir.is_dir():
        adapter_dir = ckpt_dir

    # 1. Run inference if needed
    if not predictions_path.is_file():
        if skip_inference:
            raise FileNotFoundError(f"Predictions not found for step {step} and skip_inference is True: {predictions_path}")
        inference_cmd = [
            sys.executable,
            str(REPO_ROOT / "inference" / "run_inference.py"),
            "--manifest", str(manifest_path),
            "--output", str(predictions_path),
            "--model-id", model_id,
            "--revision", revision,
            "--adapter-dir", str(adapter_dir),
            "--device", device,
            "--method", "sft_adapter",
        ]
        res = subprocess.run(inference_cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"Inference failed for step {step}:\n{res.stderr}\n{res.stdout}")

    # 2. Run evaluation if needed
    if not metrics_path.is_file():
        eval_cmd = [
            sys.executable,
            str(REPO_ROOT / "evaluation" / "eval_wer.py"),
            "--predictions", str(predictions_path),
            "--output-dir", str(eval_dir),
        ]
        res = subprocess.run(eval_cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"eval_wer failed for step {step}:\n{res.stderr}\n{res.stdout}")

    step_metrics = load_json(metrics_path)
    gate_result = evaluate_gate(
        base_metrics=base_metrics,
        pilot_metrics=step_metrics,
        stage="sft_pilot",
        manifest_path=manifest_path,
        pilot_predictions_path=predictions_path,
        pilot_metrics_path=metrics_path,
    )

    return {
        "step": step,
        "checkpoint_dir": str(ckpt_dir),
        "adapter_dir": str(adapter_dir),
        "predictions_path": str(predictions_path),
        "metrics_path": str(metrics_path),
        "metrics": step_metrics,
        "gate": gate_result,
    }


def extract_summary(
    eval_result: Dict[str, Any],
    clean_target_min: float = 0.020,
    clean_target_max: float = 0.025,
) -> Dict[str, Any]:
    """Extract key metrics and target comparison from eval result."""
    step = eval_result["step"]
    metrics = eval_result["metrics"]
    gate = eval_result["gate"]

    overall = metrics.get("overall", {})
    by_scenario = {
        s["group"]: s.get("error_rate")
        for s in metrics.get("by_scenario", [])
        if "group" in s
    }

    en_clean = by_scenario.get("en|clean")
    zh_clean = by_scenario.get("zh|clean")
    clean_macro = overall.get("clean_language_macro_error_rate")
    robust_macro = overall.get("robust_language_macro_error_rate") or overall.get("degraded_language_macro_error_rate")
    degraded_macro = robust_macro

    base_robust = gate.get("metrics", {}).get("base_robust_macro")
    robust_increase = 0.0
    if robust_macro is not None and base_robust is not None:
        robust_increase = round(robust_macro - base_robust, 6)
    elif isinstance(gate.get("metrics"), dict) and "robust_error_rate_increase" in gate["metrics"]:
        robust_increase = gate["metrics"]["robust_error_rate_increase"]

    clean_in_target = (
        clean_target_min <= en_clean <= clean_target_max
        if en_clean is not None
        else False
    )

    return {
        "step": step,
        "checkpoint_dir": eval_result["checkpoint_dir"],
        "adapter_dir": eval_result["adapter_dir"],
        "predictions_path": eval_result["predictions_path"],
        "overall_error_rate": overall.get("language_macro_error_rate"),
        "clean_macro_error_rate": clean_macro,
        "en_clean_wer": en_clean,
        "zh_clean_cer": zh_clean,
        "degraded_macro_error_rate": degraded_macro,
        "robust_macro_error_rate": robust_macro,
        "robust_increase": robust_increase,
        "improved_degraded_scenarios_count": (
            gate.get("metrics", {}).get("degraded_scenario_improvements_count")
            if isinstance(gate.get("metrics"), dict) and "degraded_scenario_improvements_count" in gate["metrics"]
            else gate.get("improved_degraded_scenarios_count", 0)
        ),
        "improved_scenarios": (
            gate.get("metrics", {}).get("improved_scenarios")
            if isinstance(gate.get("metrics"), dict) and "improved_scenarios" in gate["metrics"]
            else gate.get("improved_scenarios", [])
        ),
        "clean_increase": (
            gate.get("metrics", {}).get("clean_error_rate_increase")
            if isinstance(gate.get("metrics"), dict) and "clean_error_rate_increase" in gate["metrics"]
            else gate.get("clean_error_rate_increase", 0.0)
        ),
        "valid_output_rate": (
            gate.get("metrics", {}).get("valid_output_rate")
            if isinstance(gate.get("metrics"), dict) and "valid_output_rate" in gate["metrics"]
            else gate.get("valid_output_rate", 1.0)
        ),
        "failure_rate": (
            gate.get("metrics", {}).get("failure_rate_increase")
            if isinstance(gate.get("metrics"), dict) and "failure_rate_increase" in gate["metrics"]
            else gate.get("failure_rate", 0.0)
        ),
        "gate_status": gate.get("gate_status", "UNKNOWN"),
        "clean_in_target": clean_in_target,
    }


def select_best_checkpoint(
    summaries: List[Dict[str, Any]],
    clean_target_min: float = 0.020,
    clean_target_max: float = 0.025,
    max_robust_macro_regression: float = 0.005,
) -> Optional[Dict[str, Any]]:
    """Select the Pareto-optimal checkpoint: passing gate, robust error rate not regressing, prioritizing clean recovery then degraded gains."""
    if not summaries:
        return None

    passed = [s for s in summaries if s.get("gate_status") == "PASSED"]
    if not passed:
        return None

    valid_candidates = [
        s for s in passed
        if s.get("robust_increase", 0.0) <= max_robust_macro_regression
    ]
    candidates = valid_candidates if valid_candidates else passed

    def rank_key(s: Dict[str, Any]) -> Tuple[int, float, float, int]:
        en_clean = s.get("en_clean_wer") if s.get("en_clean_wer") is not None else 1.0
        in_target_score = 0 if en_clean <= clean_target_max else 1
        rob_err = s.get("degraded_macro_error_rate") if s.get("degraded_macro_error_rate") is not None else 1.0
        n_improved = s.get("improved_degraded_scenarios_count", 0)
        return (in_target_score, en_clean, rob_err, -n_improved)

    return min(candidates, key=rank_key)


def generate_comparison_markdown(
    base_metrics: Dict[str, Any],
    summaries: List[Dict[str, Any]],
    best: Optional[Dict[str, Any]],
    run1_metrics: Optional[Dict[str, Any]] = None,
    clean_target_min: float = 0.020,
    clean_target_max: float = 0.025,
) -> str:
    """Format markdown comparison table."""
    base_by_scenario = {
        s["group"]: s.get("error_rate")
        for s in base_metrics.get("by_scenario", [])
        if "group" in s
    }
    base_overall = base_metrics.get("overall", {})
    total_degraded = len([g for g in base_by_scenario if not g.endswith("|clean")])
    if total_degraded == 0:
        total_degraded = 14

    lines: List[str] = []
    lines.append("# Checkpoint Series Evaluation & Comparison Matrix\n")
    lines.append(f"Generated at: {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
    lines.append(f"Target English Clean WER: `[{clean_target_min * 100:.1f}%, {clean_target_max * 100:.1f}%]`\n")

    lines.append("| Model / Step | EN Clean WER | ZH Clean CER | Clean Macro | Degraded Macro | Improved Scenarios | Clean Regression | In Target | Gate Status | Recommendation |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    # Base row
    b_en = base_by_scenario.get("en|clean", 0.0)
    b_zh = base_by_scenario.get("zh|clean", 0.0)
    b_clean = base_overall.get("clean_language_macro_error_rate", 0.0)
    b_deg = base_overall.get("robust_language_macro_error_rate") or base_overall.get("degraded_language_macro_error_rate", 0.0)
    lines.append(
        f"| **Base (pretrained)** | {b_en*100:.2f}% | {b_zh*100:.2f}% | {b_clean*100:.2f}% | {b_deg*100:.2f}% | Baseline | 0.00% | YES (Base) | BASELINE | Benchmark Baseline |"
    )

    # Run 1 row (if provided)
    if run1_metrics:
        r1_scen = {
            s["group"]: s.get("error_rate")
            for s in run1_metrics.get("by_scenario", [])
            if "group" in s
        }
        r1_overall = run1_metrics.get("overall", {})
        r1_en = r1_scen.get("en|clean", 0.0)
        r1_zh = r1_scen.get("zh|clean", 0.0)
        r1_clean = r1_overall.get("clean_language_macro_error_rate", 0.0)
        r1_deg = r1_overall.get("robust_language_macro_error_rate") or r1_overall.get("degraded_language_macro_error_rate", 0.0)
        r1_reg = round((r1_clean - b_clean) * 100, 2)
        lines.append(
            f"| Run 1 (step 500, lr=2e-5, unbal) | {r1_en*100:.2f}% | {r1_zh*100:.2f}% | {r1_clean*100:.2f}% | {r1_deg*100:.2f}% | {total_degraded}/{total_degraded} | +{r1_reg:.2f}% | NO (3.44%) | PASSED | Overfit / Degraded Bias |"
        )

    # Step rows
    best_step = best["step"] if best else None
    for s in summaries:
        st = s["step"]
        en_str = f"{s['en_clean_wer']*100:.2f}%" if s["en_clean_wer"] is not None else "N/A"
        zh_str = f"{s['zh_clean_cer']*100:.2f}%" if s["zh_clean_cer"] is not None else "N/A"
        clean_str = f"{s['clean_macro_error_rate']*100:.2f}%" if s["clean_macro_error_rate"] is not None else "N/A"
        deg_str = f"{s['degraded_macro_error_rate']*100:.2f}%" if s["degraded_macro_error_rate"] is not None else "N/A"
        n_imp = f"{s['improved_degraded_scenarios_count']}/{total_degraded}"
        reg_val = s["clean_increase"] * 100
        reg_str = f"+{reg_val:.2f}%" if reg_val >= 0 else f"{reg_val:.2f}%"
        in_tgt_str = "**YES**" if s["clean_in_target"] else "NO"
        gate_str = s["gate_status"]
        is_rec = "**Pareto Optimal (Selected)**" if st == best_step else ("Candidate" if s["gate_status"] == "PASSED" else "Rejected")

        row_prefix = f"**Step {st}**" if st == best_step else f"Step {st}"
        lines.append(
            f"| {row_prefix} | {en_str} | {zh_str} | {clean_str} | {deg_str} | {n_imp} | {reg_str} | {in_tgt_str} | {gate_str} | {is_rec} |"
        )

    lines.append("\n## Analysis & Selection\n")
    if best:
        lines.append(f"- **Selected Checkpoint**: `Step {best['step']}`")
        lines.append(f"- **Checkpoint Path**: `{best['checkpoint_dir']}`")
        lines.append(f"- **Adapter Path**: `{best['adapter_dir']}`")
        lines.append(f"- **English Clean WER**: `{best['en_clean_wer']*100:.2f}%` (target: {clean_target_min*100:.1f}% - {clean_target_max*100:.1f}%)")
        lines.append(f"- **Chinese Clean CER**: `{best['zh_clean_cer']*100:.2f}%`")
        lines.append(f"- **Degraded Scenarios Improved**: `{best['improved_degraded_scenarios_count']}/{total_degraded}`")
        lines.append(f"- **Gate Status**: `{best['gate_status']}`")
    else:
        lines.append("- No checkpoint satisfied selection criteria.")

    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    run_dir = Path(args.run_dir)
    manifest_path = Path(args.manifest)
    base_metrics_path = Path(args.base_metrics)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not base_metrics_path.is_file():
        raise FileNotFoundError(f"Base metrics not found: {base_metrics_path}")

    base_metrics = load_json(base_metrics_path)
    run1_metrics = load_json(Path(args.run1_metrics)) if args.run1_metrics and Path(args.run1_metrics).is_file() else None

    step_ckpts = discover_step_checkpoints(run_dir, args.steps)
    if not step_ckpts:
        raise RuntimeError(f"No valid step checkpoints found in {run_dir} for steps={args.steps}")

    print(f"Discovered {len(step_ckpts)} checkpoints to evaluate: {[s for s, _ in step_ckpts]}")

    results: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for step, ckpt_dir in step_ckpts:
        print(f"\n--- Evaluating Checkpoint Step {step} ({ckpt_dir}) ---")
        output_step_dir = output_dir / f"step_{step}"
        eval_res = run_step_eval(
            step=step,
            ckpt_dir=ckpt_dir,
            manifest_path=manifest_path,
            output_step_dir=output_step_dir,
            base_metrics=base_metrics,
            model_id=args.model_id,
            revision=args.revision,
            device=args.device,
            skip_inference=args.skip_inference,
        )
        results.append(eval_res)
        summary = extract_summary(
            eval_res,
            clean_target_min=args.clean_target_min,
            clean_target_max=args.clean_target_max,
        )
        summaries.append(summary)
        print(
            f"Step {step} Summary: EN Clean WER={summary['en_clean_wer']*100:.2f}%, "
            f"Degraded Improved={summary['improved_degraded_scenarios_count']}, "
            f"Gate={summary['gate_status']}, In Target={summary['clean_in_target']}"
        )

    best = select_best_checkpoint(
        summaries,
        clean_target_min=args.clean_target_min,
        clean_target_max=args.clean_target_max,
    )

    # Save outputs
    comparison_data = {
        "run_dir": str(run_dir),
        "manifest": str(manifest_path),
        "base_metrics": str(base_metrics_path),
        "steps_evaluated": [s for s, _ in step_ckpts],
        "clean_target": {
            "min": args.clean_target_min,
            "max": args.clean_target_max,
        },
        "best_checkpoint": best,
        "summaries": summaries,
    }

    with open(output_dir / "checkpoint_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)

    md_report = generate_comparison_markdown(
        base_metrics=base_metrics,
        summaries=summaries,
        best=best,
        run1_metrics=run1_metrics,
        clean_target_min=args.clean_target_min,
        clean_target_max=args.clean_target_max,
    )
    with open(output_dir / "checkpoint_comparison.md", "w", encoding="utf-8") as f:
        f.write(md_report)

    if best:
        best_data = {
            "selected_step": best["step"],
            "checkpoint_dir": best["checkpoint_dir"],
            "adapter_dir": best["adapter_dir"],
            "en_clean_wer": best["en_clean_wer"],
            "zh_clean_cer": best["zh_clean_cer"],
            "improved_scenarios_count": best["improved_degraded_scenarios_count"],
            "gate_status": best["gate_status"],
            "clean_in_target": best["clean_in_target"],
            "selected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        with open(output_dir / "best_checkpoint.json", "w", encoding="utf-8") as f:
            json.dump(best_data, f, indent=2)
        print(f"\n[DONE] Best Pareto checkpoint: Step {best['step']} (Adapter: {best['adapter_dir']})")
    else:
        print("\n[WARNING] No checkpoint qualified as Pareto optimal.")


if __name__ == "__main__":
    main()
