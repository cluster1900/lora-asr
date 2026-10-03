"""Choose which manifest rows an RL epoch may update.

``degraded`` already keeps clean speech out of the optimizer. v25's gate
showed two degraded scenarios moving the wrong way: ``noise`` and
``recording``. ``degraded_skip_regressed`` keeps every other degraded row
and drops those two. The manifest hash stays the full pilot file, so the
probe gate still matches.
"""

from __future__ import annotations

from typing import Mapping, Sequence

REGRESSED_SCENARIOS = frozenset({"noise", "recording"})


def row_scenario(row: Mapping[str, object]) -> str:
    """Return the manifest scenario token, lowercased, without language prefixes."""
    return str(row.get("scenario") or "").strip().lower()


def is_degraded_row(row: Mapping[str, object]) -> bool:
    """Match the trainer's degraded rule: explicit group, or a non-clean scenario."""
    group = str(row.get("condition_group") or "").strip().lower()
    if group == "degraded":
        return True
    return "clean" not in row_scenario(row)


def degraded_skip_regressed_indices(rows: Sequence[Mapping[str, object]]) -> list[int]:
    """Indices of degraded rows whose scenario is not noise or recording.

    The result is manifest order. The trainer shuffles it with the epoch seed.
    An empty selection raises, so a typo cannot fall through to the full file.
    """
    chosen = [
        index
        for index, row in enumerate(rows)
        if is_degraded_row(row) and row_scenario(row) not in REGRESSED_SCENARIOS
    ]
    if not chosen:
        raise ValueError("degraded_skip_regressed selected no rows")
    return chosen
