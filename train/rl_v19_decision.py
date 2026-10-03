"""Follow-up action after a scored rl_pilot_v19 checkpoint.

The action table is the v17 table. The prior Robust value is the newest gate
strictly before the scored step. A KL stop can land off the 4-step grid, and
reading ``step - 4`` then asks for a gate that was never written.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train.rl_v17_decision import followup_action


def latest_prior_robust(run: Path, step: int) -> float | None:
    """Return the Robust increase of the newest gate scored before ``step``."""
    current = int(step)
    if current < 8:
        return None
    found: list[tuple[int, Path]] = []
    for path in run.glob("gate_step_*.json"):
        suffix = path.stem.removeprefix("gate_step_")
        if not suffix.isdigit():
            continue
        number = int(suffix)
        if number < current:
            found.append((number, path))
    if not found:
        raise SystemExit(f"missing prior gate before step {current}")
    _, prior = max(found)
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
    previous = latest_prior_robust(run, int(args.step))
    action, reason = followup_action(
        train_status=str(state.get("status", "")),
        step=int(args.step),
        horizon=int(args.horizon),
        improvement=float(metrics["held_out_reward_improvement"]),
        robust_increase=float(metrics["robust_error_rate_increase"]),
        gate_status=str(gate.get("gate_status", "")),
        previous_robust=previous,
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
        "previous_robust_error_rate_increase": previous,
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
