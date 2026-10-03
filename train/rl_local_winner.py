"""Keep anchored GRPO winners that are local corrections of the greedy text.

English uses words and Chinese uses characters. Normalization matches
``evaluation/eval_wer.py`` so the distance is the same quantity as the gate.
A winner farther than ``max(min_absolute, floor(max_relative * longest))``
tokens does not train, even when its reward gap is larger.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Optional, Sequence


def normalize_text(text: str) -> str:
    """Lowercase text, replace Unicode punctuation, and collapse whitespace."""
    value = str(text or "").strip().lower()
    value = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in value
    )
    return re.sub(r"\s+", " ", value).strip()


def tokenize(text: str, language: str) -> list[str]:
    """Tokenize Chinese by character and every other language by word."""
    if str(language).strip().lower() == "zh":
        return list(text.replace(" ", ""))
    return text.split()


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Levenshtein distance with one dynamic-programming row."""
    previous = list(range(len(hypothesis) + 1))
    for row_index, ref_token in enumerate(reference, start=1):
        current = [row_index]
        for column_index, hyp_token in enumerate(hypothesis, start=1):
            substitution = previous[column_index - 1] + int(ref_token != hyp_token)
            current.append(min(
                previous[column_index] + 1,
                current[column_index - 1] + 1,
                substitution,
            ))
        previous = current
    return previous[-1]


def local_edit_limit(
    anchor_length: int,
    winner_length: int,
    max_relative: float = 0.35,
    min_absolute: int = 2,
) -> int:
    """Maximum token edits allowed between the anchor and a winner."""
    relative = float(max_relative)
    if not 0.0 < relative <= 1.0:
        raise ValueError(f"max_relative must be in (0, 1], got {max_relative}")
    if int(min_absolute) < 0:
        raise ValueError(f"min_absolute must be >= 0, got {min_absolute}")
    longest = max(int(anchor_length), int(winner_length), 1)
    # Thousandths keep 0.35 * 20 at 7. A binary float product can land just under.
    relative_limit = (longest * int(round(relative * 1000.0))) // 1000
    return max(int(min_absolute), relative_limit)


def is_local_correction(
    anchor: str,
    winner: str,
    language: str,
    max_relative: float = 0.35,
    min_absolute: int = 2,
) -> bool:
    """True when the winner stays inside the local edit budget."""
    anchor_tokens = tokenize(normalize_text(anchor), language)
    winner_tokens = tokenize(normalize_text(winner), language)
    distance = edit_distance(anchor_tokens, winner_tokens)
    limit = local_edit_limit(
        len(anchor_tokens),
        len(winner_tokens),
        max_relative=max_relative,
        min_absolute=min_absolute,
    )
    return distance <= limit


def _clears_min_improvement(gap: float, min_improvement: float) -> bool:
    """True when ``gap`` reaches the threshold, including one ulp under 0.02."""
    threshold = float(min_improvement)
    return gap >= threshold or math.isclose(gap, threshold, rel_tol=0.0, abs_tol=1e-9)


def local_winner_index(
    texts: Sequence[str],
    rewards: Sequence[float],
    language: str,
    min_improvement: float = 0.02,
    max_relative: float = 0.35,
    anchor_index: int = 0,
    min_absolute: int = 2,
) -> tuple[Optional[int], str]:
    """Pick the closest-index local sample with the largest reward gap.

    The status is ``update`` when a sample both clears ``min_improvement`` and
    stays inside the edit budget. A reward gap that only comes from a distant
    rewrite is ``no_improvement``, so a second sampling round can still run.
    Identical rewards are ``identical``.
    """
    if len(texts) != len(rewards):
        raise ValueError("texts and rewards must have the same length")
    if not rewards or not 0 <= int(anchor_index) < len(rewards):
        raise ValueError("anchor_index is out of range")

    anchor_reward = float(rewards[anchor_index])
    reward_span = max(rewards) - min(rewards)
    best_gap = 0.0
    winner: Optional[int] = None
    for index, reward in enumerate(rewards):
        if index == anchor_index:
            continue
        gap = max(0.0, float(reward) - anchor_reward)
        if not _clears_min_improvement(gap, min_improvement):
            continue
        if not is_local_correction(
            texts[anchor_index],
            texts[index],
            language,
            max_relative=max_relative,
            min_absolute=min_absolute,
        ):
            continue
        if winner is None or gap > best_gap:
            best_gap = gap
            winner = index
    if winner is not None:
        return winner, "update"
    if reward_span <= 1e-6:
        return None, "identical"
    return None, "no_improvement"
