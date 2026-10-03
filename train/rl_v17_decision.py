"""Follow-up action after a scored rl_pilot_v17 checkpoint.

v16 used the same local-correction update and stopped at step 8 while Robust
was still above DPO, even though that increase had fallen since step 4 and
the greedy reward was still rising. v17 keeps training through that level
check, and through the unfiltered step-12 mass floor, while Robust does not
rise and the logged reward mass stays at or above the step-8 search floor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HARD_STOPS = (
    "STOPPED_KL",
    "STOPPED_REWARD_DROP",
    "STOPPED_ROBUST",
    "FAILED_ZERO_VARIANCE",
    "BLOCKED_NO_WINNERS",
)
CONTINUABLE = ("CHUNK_DONE", "BLOCKED_TRANSFER")
SEARCH_MASS_FLOOR = 8.50
ROBUST_CEILING = 0.0005
ROBUST_RISE = 1.0e-6


def followup_action(
    train_status: str,
    step: int,
    horizon: int,
    improvement: float,
    robust_increase: float,
    gate_status: str,
    previous_robust: float | None,
    reward_mass: float,
) -> tuple[str, str]:
    """Choose passed, continue, or stop for one scored v17 checkpoint."""
    status = str(train_status)
    robust = float(robust_increase)
    if gate_status == "PASSED":
        return "passed", "gate passed"
    if robust >= ROBUST_CEILING:
        return "stop", "robust increased"
    if float(improvement) < 0.0:
        return "stop", "greedy reward fell below step 0"
    if status in HARD_STOPS:
        return "stop", f"trainer stop {status}"
    if status == "BLOCKED_SEARCH" and float(reward_mass) < SEARCH_MASS_FLOOR:
        return "stop", "search mass below the step-8 floor"
    if int(step) >= 8 and previous_robust is None:
        return "stop", "missing prior robust"
    if previous_robust is not None and robust > float(previous_robust) + ROBUST_RISE:
        return "stop", "robust rose"
    if int(step) >= int(horizon):
        return "stop", "horizon reached"
    if status in CONTINUABLE or status == "BLOCKED_SEARCH":
        return "continue", "robust did not rise"
    return "stop", f"unrecognized status {status}"


def _previous_robust(run: Path, step: int) -> float | None:
    if int(step) < 8:
        return None
    prior = run / f"gate_step_{int(step) - 4}.json"
    if not prior.is_file():
        raise SystemExit(f"missing prior gate {prior.name}")
    payload = json.loads(prior.read_text(encoding="utf-8"))
    return float(payload["metrics"]["robust_error_rate_increase"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    args = parser.parse_args()
    run = args.run_dir
    state = json.loads((run / "pipeline_state.json").read_text(encoding="utf-8"))
    if int(state.get("global_step", -1)) != int(args.step):
        raise SystemExit(
            f"pipeline global_step {state.get('global_step')} != {args.step}"
        )
    if "cumulative_reward_mass" not in state:
        raise SystemExit("pipeline_state has no cumulative_reward_mass")
    gate = json.loads((run / f"gate_step_{args.step}.json").read_text(encoding="utf-8"))
    metrics = gate["metrics"]
    action, reason = followup_action(
        train_status=str(state.get("status", "")),
        step=int(args.step),
        horizon=int(args.horizon),
        improvement=float(metrics["held_out_reward_improvement"]),
        robust_increase=float(metrics["robust_error_rate_increase"]),
        gate_status=str(gate.get("gate_status", "")),
        previous_robust=_previous_robust(run, int(args.step)),
        reward_mass=float(state["cumulative_reward_mass"]),
    )
    payload = {
        "action": action,
        "reason": reason,
        "scored_step": int(args.step),
        "train_status": state.get("status"),
        "held_out_reward_improvement": float(metrics["held_out_reward_improvement"]),
        "robust_error_rate_increase": float(metrics["robust_error_rate_increase"]),
        "gate_status": gate.get("gate_status"),
        "previous_robust_error_rate_increase": _previous_robust(run, int(args.step)),
        "cumulative_reward_mass": float(state["cumulative_reward_mass"]),
    }
    (run / "switch_decision.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
