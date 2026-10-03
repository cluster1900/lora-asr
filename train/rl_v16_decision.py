"""Follow-up action after a scored rl_pilot_v16 checkpoint.

The trainer can still return BLOCKED_TRANSFER when held-out gain is below
+0.001. This driver treats that status as continuable while the greedy reward
is non-negative and Robust has not stayed above DPO past step 8.
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
    "BLOCKED_SEARCH",
)
CONTINUABLE = ("CHUNK_DONE", "BLOCKED_TRANSFER")


def followup_action(
    train_status: str,
    step: int,
    horizon: int,
    improvement: float,
    robust_increase: float,
    gate_status: str,
) -> tuple[str, str]:
    """Choose passed, continue, or stop for one scored v16 checkpoint."""
    status = str(train_status)
    if gate_status == "PASSED":
        return "passed", "gate passed"
    if float(robust_increase) >= 0.0005:
        return "stop", "robust increased"
    if float(improvement) < 0.0:
        return "stop", "greedy reward fell below step 0"
    if status in HARD_STOPS:
        return "stop", f"trainer stop {status}"
    if int(step) >= 8 and float(robust_increase) > 0.0:
        return "stop", "robust still above DPO"
    if int(step) >= int(horizon):
        return "stop", "horizon reached"
    if status in CONTINUABLE:
        return "continue", "local-correction run can take another chunk"
    return "stop", f"unrecognized status {status}"


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
    gate = json.loads((run / f"gate_step_{args.step}.json").read_text(encoding="utf-8"))
    metrics = gate["metrics"]
    action, reason = followup_action(
        train_status=str(state.get("status", "")),
        step=int(args.step),
        horizon=int(args.horizon),
        improvement=float(metrics["held_out_reward_improvement"]),
        robust_increase=float(metrics["robust_error_rate_increase"]),
        gate_status=str(gate.get("gate_status", "")),
    )
    payload = {
        "action": action,
        "reason": reason,
        "scored_step": int(args.step),
        "train_status": state.get("status"),
        "held_out_reward_improvement": float(metrics["held_out_reward_improvement"]),
        "robust_error_rate_increase": float(metrics["robust_error_rate_increase"]),
        "gate_status": gate.get("gate_status"),
    }
    (run / "switch_decision.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(action)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
