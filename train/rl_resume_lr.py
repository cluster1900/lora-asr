"""Write the configured learning rate after a checkpoint resume.

Loading ``scheduler.pt`` restores the learning rate stored in that
checkpoint. A run that resumes and wants a different rate has to overwrite
the optimizer groups and the scheduler base rates after that load.
"""

from __future__ import annotations

import math
from typing import Any


def apply_configured_learning_rate(optimizer: Any, scheduler: Any, learning_rate: float) -> float:
    """Set ``learning_rate`` on the optimizer and, when present, the scheduler.

    ``initial_lr`` is updated only for groups that already have it. Scheduler
    ``last_epoch`` is left alone, so a finished warmup stays finished.
    """
    lr = float(learning_rate)
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError(f"learning rate must be a positive finite number, got {learning_rate!r}")
    groups = getattr(optimizer, "param_groups", None)
    if not groups:
        raise ValueError("optimizer has no param groups")
    for group in groups:
        group["lr"] = lr
        if "initial_lr" in group:
            group["initial_lr"] = lr
    base_lrs = getattr(scheduler, "base_lrs", None) if scheduler is not None else None
    if base_lrs is not None:
        scheduler.base_lrs = [lr for _ in base_lrs]
    return lr
