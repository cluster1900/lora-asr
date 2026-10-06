"""Choose which manifest rows an RL epoch may update.

``degraded`` keeps clean speech out of the optimizer. v25's gate showed two
degraded scenarios moving the wrong way: ``noise`` and ``recording``.
``degraded_skip_regressed`` keeps every other degraded row and drops those two.
``degraded_balanced`` keeps only degraded rows and deterministically balances
language by scenario with bounded replacement sampling.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
import math
from typing import Mapping, Sequence

REGRESSED_SCENARIOS = frozenset({"noise", "recording"})
DEGRADED_BALANCED_SCENARIOS = (
    "noise",
    "recording",
    "dropout",
    "obstructed",
    "distortion",
    "echo",
    "far_field",
)
# The formal v31 pool also accepts rows tagged ``mixed``.  ``degraded_balanced``
# remains the seven-cell pilot sampler; the scale launcher uses this wider
# contract while keeping ``sample_strategy=degraded`` so every physical row is
# eligible exactly once per virtual epoch.
FORMAL_RL_SCENARIOS = DEGRADED_BALANCED_SCENARIOS + ("mixed",)
DEFAULT_DEGRADED_BALANCED_WEIGHTS = {
    "noise": 0.20,
    "recording": 0.20,
    "dropout": 0.15,
    "obstructed": 0.15,
    "distortion": 0.10,
    "echo": 0.10,
    "far_field": 0.10,
}


def row_scenario(row: Mapping[str, object]) -> str:
    """Return the manifest scenario token, lowercased, without language prefixes."""
    return str(row.get("scenario") or "").strip().lower()


def is_degraded_row(row: Mapping[str, object]) -> bool:
    """Match the trainer's degraded rule: explicit group, or a non-clean scenario."""
    group = str(row.get("condition_group") or "").strip().lower()
    if group == "degraded":
        return True
    return "clean" not in row_scenario(row)


def degraded_indices(rows: Sequence[Mapping[str, object]]) -> list[int]:
    """Return all degraded row indices in manifest order."""
    return [index for index, row in enumerate(rows) if is_degraded_row(row)]


def validate_degraded_manifest(
    rows: Sequence[Mapping[str, object]],
    *,
    min_rows: int = 160_000,
    min_cell_rows: int = 640,
    max_cell_fraction: float | None = None,
    require_all_cells: bool = True,
) -> dict[str, object]:
    """Validate the formal RL pool size and language×scenario coverage.

    The v31 scale contract must not silently train on a ``NON_STRICT_SUBSET``.
    Clean rows are allowed in the physical manifest for audit compatibility but
    are rejected from the formal optimizer pool by this check.  The returned
    summary is JSON-serializable so launchers can record the exact preflight.
    """
    if min_rows <= 0 or min_cell_rows <= 0:
        raise ValueError("min_rows and min_cell_rows must be positive")
    if max_cell_fraction is not None and not 0.0 < float(max_cell_fraction) <= 1.0:
        raise ValueError("max_cell_fraction must be in (0, 1]")
    counts: Counter[str] = Counter()
    degraded_total = 0
    clean_total = 0
    for index, row in enumerate(rows):
        if not is_degraded_row(row):
            clean_total += 1
            continue
        language = _normalized_language(row)
        scenario = row_scenario(row)
        if language not in {"en", "zh"} or scenario not in FORMAL_RL_SCENARIOS:
            raise ValueError(
                f"formal RL pool row {index} has unsupported language/scenario "
                f"{language!r}|{scenario!r}"
            )
        counts[f"{language}|{scenario}"] += 1
        degraded_total += 1
    missing = []
    if require_all_cells:
        missing = sorted(
            f"{language}|{scenario}"
            for language in ("en", "zh")
            for scenario in FORMAL_RL_SCENARIOS
            if counts[f"{language}|{scenario}"] < min_cell_rows
        )
    if degraded_total < min_rows:
        raise ValueError(
            f"formal RL pool has {degraded_total} degraded rows; "
            f"requires at least {min_rows}"
        )
    if missing:
        raise ValueError(
            f"formal RL pool cells below {min_cell_rows}: {missing}"
        )
    if max_cell_fraction is not None and degraded_total:
        limit = float(max_cell_fraction)
        oversized = sorted(
            (cell, count, count / degraded_total)
            for cell, count in counts.items()
            if count / degraded_total > limit
        )
        if oversized:
            formatted = [f"{cell}={count}/{degraded_total} ({fraction:.3f})" for cell, count, fraction in oversized]
            raise ValueError(
                f"formal RL pool cells exceed max fraction {limit:.3f}: {formatted}"
            )
    return {
        "degraded_rows": degraded_total,
        "clean_rows": clean_total,
        "min_rows": int(min_rows),
        "min_cell_rows": int(min_cell_rows),
        "max_cell_fraction": None if max_cell_fraction is None else float(max_cell_fraction),
        "require_all_cells": bool(require_all_cells),
        "cell_counts": dict(sorted(counts.items())),
    }


def _normalized_language(row: Mapping[str, object]) -> str:
    """Normalize the two languages supported by the robust ASR manifests."""
    value = str(row.get("language") or "").strip().lower()
    if value in {"en", "english"}:
        return "en"
    if value in {"zh", "chinese"}:
        return "zh"
    return value


def _allocate_weighted_counts(total: int, weights: Mapping[str, float]) -> dict[str, int]:
    """Allocate an integer total using largest fractional remainders."""
    if total < 0:
        raise ValueError("total must be non-negative")
    names = list(weights)
    raw = {name: float(total) * float(weights[name]) for name in names}
    counts = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(
        names,
        key=lambda name: (raw[name] - counts[name], -names.index(name)),
        reverse=True,
    )
    for name in order[:remaining]:
        counts[name] += 1
    return counts


def _validate_balanced_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Validate and normalize the seven degraded scenario weights."""
    missing = [name for name in DEGRADED_BALANCED_SCENARIOS if name not in weights]
    extra = [name for name in weights if name not in DEGRADED_BALANCED_SCENARIOS]
    if missing or extra:
        raise ValueError(
            "scenario weights must contain exactly "
            f"{DEGRADED_BALANCED_SCENARIOS}; missing={missing}, extra={extra}"
        )
    try:
        normalized = {
            name: float(weights[name]) for name in DEGRADED_BALANCED_SCENARIOS
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("scenario weights must be finite numbers") from exc
    if any(not math.isfinite(value) or value <= 0.0 for value in normalized.values()):
        raise ValueError("scenario weights must be finite and positive")
    total = sum(normalized.values())
    if total <= 0.0:
        raise ValueError("scenario weights must have a positive sum")
    return {name: value / total for name, value in normalized.items()}


def degraded_balanced_indices(
    rows: Sequence[Mapping[str, object]],
    *,
    epoch: int = 0,
    seed: int = 20260722,
    virtual_epoch_rows: int = 2000,
    max_repeat: int = 4,
    scenario_weights: Mapping[str, float] | None = None,
) -> list[int]:
    """Build deterministic, language-balanced degraded sampling indices.

    Each language receives half of ``virtual_epoch_rows``. Within a language,
    scenario counts follow ``scenario_weights``. A source row may be selected
    at most ``max_repeat`` times; shortfalls from rare cells are deterministically
    backfilled from other degraded cells of the same language.
    """
    if virtual_epoch_rows <= 0:
        raise ValueError("virtual_epoch_rows must be positive")
    if max_repeat < 1:
        raise ValueError("max_repeat must be positive")
    if virtual_epoch_rows % 2:
        raise ValueError("virtual_epoch_rows must be even for equal language quotas")
    weights = _validate_balanced_weights(
        DEFAULT_DEGRADED_BALANCED_WEIGHTS
        if scenario_weights is None
        else scenario_weights
    )

    cell_indices: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if not is_degraded_row(row):
            continue
        language = _normalized_language(row)
        scenario = row_scenario(row)
        if language not in {"en", "zh"}:
            raise ValueError(
                f"degraded_balanced requires language en/zh, got {language!r} at row {index}"
            )
        if scenario not in weights:
            raise ValueError(
                f"degraded_balanced requires known scenario, got {scenario!r} at row {index}"
            )
        cell_indices[(language, scenario)].append(index)

    missing = [
        f"{language}|{scenario}"
        for language in ("en", "zh")
        for scenario in DEGRADED_BALANCED_SCENARIOS
        if not cell_indices[(language, scenario)]
    ]
    if missing:
        raise ValueError(
            "degraded_balanced requires every language|scenario cell; "
            f"missing={missing}"
        )

    per_language = virtual_epoch_rows // 2
    quotas = _allocate_weighted_counts(per_language, weights)
    selected: list[int] = []
    used = Counter()
    shortfalls = {"en": 0, "zh": 0}

    for language_index, language in enumerate(("en", "zh")):
        for scenario_index, scenario in enumerate(DEGRADED_BALANCED_SCENARIOS):
            cell = (language, scenario)
            candidates = list(cell_indices[cell])
            cell_rng = random.Random(
                int(seed)
                + int(epoch) * 1_000_003
                + language_index * 10_007
                + scenario_index
            )
            target = quotas[scenario]
            picked = 0
            for _ in range(max_repeat):
                cell_rng.shuffle(candidates)
                for index in candidates:
                    if picked >= target:
                        break
                    if used[index] < max_repeat:
                        selected.append(index)
                        used[index] += 1
                        picked += 1
                if picked >= target:
                    break
            shortfalls[language] += target - picked

    for language_index, language in enumerate(("en", "zh")):
        remaining = shortfalls[language]
        if remaining == 0:
            continue
        language_candidates = [
            index
            for scenario in DEGRADED_BALANCED_SCENARIOS
            for index in cell_indices[(language, scenario)]
            if used[index] < max_repeat
        ]
        backfill_rng = random.Random(
            int(seed) + int(epoch) * 1_000_003 + 90_001 + language_index
        )
        for _ in range(max_repeat):
            backfill_rng.shuffle(language_candidates)
            for index in language_candidates:
                if remaining == 0:
                    break
                if used[index] >= max_repeat:
                    continue
                selected.append(index)
                used[index] += 1
                remaining -= 1
            if remaining == 0:
                break
        if remaining:
            raise ValueError(
                f"degraded_balanced cannot fill language quota for {language}; "
                f"shortfall={remaining}"
            )

    final_rng = random.Random(int(seed) + int(epoch) * 1_000_003 + 99_991)
    final_rng.shuffle(selected)
    if len(selected) != virtual_epoch_rows:
        raise AssertionError(
            f"degraded_balanced selected {len(selected)} rows, "
            f"expected {virtual_epoch_rows}"
        )
    return selected


def degraded_balanced_summary(
    rows: Sequence[Mapping[str, object]],
    indices: Sequence[int],
    *,
    virtual_epoch_rows: int | None = None,
    scenario_weights: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Summarize selected cells, clean leakage, repetition, and backfill.

    ``virtual_epoch_rows`` enables the expected quota and same-language
    backfill audit fields. It is optional so callers can summarize arbitrary
    index lists without knowing the sampler configuration.
    """
    cell_counts: Counter[str] = Counter()
    source_counts: Counter[int] = Counter()
    clean_rows = 0
    for index in indices:
        if index < 0 or index >= len(rows):
            raise ValueError(f"sampling index out of range: {index}")
        source_counts[int(index)] += 1
        row = rows[index]
        if not is_degraded_row(row):
            clean_rows += 1
            continue
        cell_counts[f"{_normalized_language(row)}|{row_scenario(row)}"] += 1
    summary: dict[str, object] = {
        "selected_rows": len(indices),
        "clean_rows": clean_rows,
        "cell_counts": dict(sorted(cell_counts.items())),
        "unique_source_rows": len(source_counts),
        "max_repeat": max(source_counts.values(), default=0),
    }
    if virtual_epoch_rows is not None:
        weights = _validate_balanced_weights(
            DEFAULT_DEGRADED_BALANCED_WEIGHTS
            if scenario_weights is None
            else scenario_weights
        )
        if virtual_epoch_rows <= 0 or virtual_epoch_rows % 2:
            raise ValueError("virtual_epoch_rows must be a positive even number")
        quotas = _allocate_weighted_counts(virtual_epoch_rows // 2, weights)
        expected = {
            f"{language}|{scenario}": quotas[scenario]
            for language in ("en", "zh")
            for scenario in DEGRADED_BALANCED_SCENARIOS
        }
        backfill_by_language = {
            language: sum(
                max(0, expected[f"{language}|{scenario}"] - cell_counts.get(f"{language}|{scenario}", 0))
                for scenario in DEGRADED_BALANCED_SCENARIOS
            )
            for language in ("en", "zh")
        }
        summary["target_cell_counts"] = expected
        summary["backfill_by_language"] = backfill_by_language
        summary["backfill_rows"] = sum(backfill_by_language.values())
    return summary


def degraded_skip_regressed_indices(rows: Sequence[Mapping[str, object]]) -> list[int]:
    """Indices of degraded rows whose scenario is not noise or recording."""
    chosen = [
        index
        for index, row in enumerate(rows)
        if is_degraded_row(row) and row_scenario(row) not in REGRESSED_SCENARIOS
    ]
    if not chosen:
        raise ValueError("degraded_skip_regressed selected no rows")
    return chosen
