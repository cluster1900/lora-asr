"""Choose which LoRA module names receive a gradient.

The trainer's regex still finds the 199 canonical Linear targets. A run can
drop the three audio-tower projections without changing that regex.
"""

from __future__ import annotations


def filter_lora_targets(names: list[str], train_audio_projections: bool) -> list[str]:
    """Return the LoRA names this run trains.

    ``train_audio_projections`` false removes every name that starts with
    ``audio_tower.``. True returns a copy of ``names``.
    """
    if train_audio_projections:
        return list(names)
    return [name for name in names if not name.startswith("audio_tower.")]
