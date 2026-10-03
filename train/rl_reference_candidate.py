"""Add the gold transcript as one more local-correction candidate.

Sampled groups often have no hypothesis inside the edit budget that beats the
greedy anchor. The manifest reference is scored with the same sequence reward
as any other hypothesis. It still has to clear ``min_improvement`` and
``local_max_relative`` inside ``local_winner_index``. A distant rewrite is
left unselected, so a second sampling round can still run.
"""

from __future__ import annotations

from typing import Sequence

try:
    from rl_local_winner import normalize_text
except ImportError:  # local tests import this module as train.rl_reference_candidate
    from train.rl_local_winner import normalize_text


def append_reference_candidate(
    texts: Sequence[str],
    rewards: Sequence[float],
    reference: str,
    reference_reward: float,
) -> tuple[list[str], list[float]]:
    """Return the candidate lists, appending ``reference`` when it is new.

    A candidate that already normalizes to the reference keeps its own reward.
    An empty reference is not appended. The caller passes ``reference_reward``
    from ``compute_sequence_reward(reference, reference, language)``.
    """
    if len(texts) != len(rewards):
        raise ValueError("texts and rewards must have the same length")
    copied_texts = list(texts)
    copied_rewards = [float(reward) for reward in rewards]
    normalized_reference = normalize_text(reference)
    if not normalized_reference:
        return copied_texts, copied_rewards
    for text in copied_texts:
        if normalize_text(text) == normalized_reference:
            return copied_texts, copied_rewards
    copied_texts.append(str(reference))
    copied_rewards.append(float(reference_reward))
    return copied_texts, copied_rewards
