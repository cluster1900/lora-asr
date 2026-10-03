"""Winner advantage for a length-2 anchored GRPO pair.

``unit`` keeps the historical winner advantage of 1. ``raw_gap`` uses the
reward gap. ``fixed`` uses a configured constant. ``capped_gap`` uses the
reward gap up to ``cap``. The greedy row stays 0.
"""

from __future__ import annotations


def winner_advantage(
    anchor_reward: float,
    winner_reward: float,
    mode: str,
    fixed_value: float = 0.10,
    cap: float = 0.05,
) -> float:
    name = str(mode).strip().lower()
    gap = max(0.0, float(winner_reward) - float(anchor_reward))
    if name == "unit":
        return 1.0
    if name == "raw_gap":
        return gap
    if name == "fixed":
        value = float(fixed_value)
        if not 0.0 < value <= 1.0:
            raise ValueError(f"fixed advantage must be in (0, 1], got {value}")
        return value
    if name == "capped_gap":
        limit = float(cap)
        if not 0.0 < limit <= 1.0:
            raise ValueError(f"capped advantage must be in (0, 1], got {limit}")
        return min(gap, limit)
    raise ValueError(
        f"advantage mode must be unit, raw_gap, fixed, or capped_gap, got {mode!r}"
    )
