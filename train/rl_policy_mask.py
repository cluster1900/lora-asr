"""Policy-gradient mask for a length-2 anchored GRPO pair.

The reward advantage stays whatever the caller already chose. Matching tokens,
measured by Levenshtein alignment on token ids, get no policy gradient.
Insertions and substitutions on the winner do. Deleted anchor tokens are not
on the winner, so ``signed_edit_update`` puts the negated gap on those anchor
positions. The KL term is not masked here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence


def token_edit_masks(
    anchor_ids: Sequence[int], winner_ids: Sequence[int]
) -> tuple[list[float], list[float]]:
    """Return ``(winner_keep, anchor_delete)`` from one token-id alignment.

    ``winner_keep`` is 1 on winner substitutions and insertions. ``anchor_delete``
    is 1 on anchor tokens that the alignment drops. Equal tokens stay 0 on both.
    An empty winner marks every anchor token deleted. An empty anchor marks
    every winner token. Ties prefer a match.
    """
    anchor = [int(token) for token in anchor_ids]
    winner = [int(token) for token in winner_ids]
    rows = len(anchor)
    cols = len(winner)
    winner_keep = [0.0] * cols
    anchor_delete = [0.0] * rows
    if cols == 0:
        for index in range(rows):
            anchor_delete[index] = 1.0
        return winner_keep, anchor_delete

    distance = [[0] * (cols + 1) for _ in range(rows + 1)]
    for i in range(1, rows + 1):
        distance[i][0] = i
    for j in range(1, cols + 1):
        distance[0][j] = j
    for i in range(1, rows + 1):
        anchor_token = anchor[i - 1]
        for j in range(1, cols + 1):
            replace = distance[i - 1][j - 1] + (0 if anchor_token == winner[j - 1] else 1)
            distance[i][j] = min(
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
                replace,
            )

    i = rows
    j = cols
    while j > 0:
        if (
            i > 0
            and anchor[i - 1] == winner[j - 1]
            and distance[i][j] == distance[i - 1][j - 1]
        ):
            i -= 1
            j -= 1
            continue
        if (
            i > 0
            and anchor[i - 1] != winner[j - 1]
            and distance[i][j] == distance[i - 1][j - 1] + 1
        ):
            winner_keep[j - 1] = 1.0
            i -= 1
            j -= 1
            continue
        if distance[i][j] == distance[i][j - 1] + 1:
            winner_keep[j - 1] = 1.0
            j -= 1
            continue
        if i > 0 and distance[i][j] == distance[i - 1][j] + 1:
            anchor_delete[i - 1] = 1.0
            i -= 1
            continue
        raise RuntimeError("token alignment could not be traced")
    while i > 0:
        anchor_delete[i - 1] = 1.0
        i -= 1
    return winner_keep, anchor_delete


def changed_token_mask(anchor_ids: Sequence[int], winner_ids: Sequence[int]) -> list[float]:
    """Return a 0/1 mask over ``winner_ids``.

    A 1 marks a winner token that is a substitution or an insertion against
    ``anchor_ids``. A 0 marks a token that aligns as equal. An empty winner
    returns an empty mask. An empty anchor marks every winner token.
    """
    winner_keep, _anchor_delete = token_edit_masks(anchor_ids, winner_ids)
    return winner_keep


@dataclass(frozen=True)
class SignedEditUpdate:
    """Per-token advantages for one anchored pair.

    ``action`` is ``skip`` when the token ids have no substitution, insertion,
    or deletion. ``update`` keeps a positive gap on changed winner tokens and
    the negated gap on deleted anchor tokens. A deletion-only pair is an update.
    """

    action: str
    winner_keep: tuple[float, ...]
    anchor_delete: tuple[float, ...]
    winner_token_advantage: tuple[float, ...]
    anchor_token_advantage: tuple[float, ...]
    winner_keep_ratio: Optional[float]


def signed_edit_update(
    anchor_ids: Sequence[int], winner_ids: Sequence[int], raw_gap: float
) -> SignedEditUpdate:
    """Assign ``raw_gap`` to winner edits and ``-raw_gap`` to deleted anchor tokens.

    Identical token ids skip the update and produce all-zero advantages. A
    winner whose only edit is a deletion still returns ``update`` and does not
    raise. ``raw_gap`` must be finite.
    """
    gap = float(raw_gap)
    if gap != gap or gap in (float("inf"), float("-inf")):
        raise ValueError("raw_gap must be finite")
    winner_keep, anchor_delete = token_edit_masks(anchor_ids, winner_ids)
    if not any(winner_keep) and not any(anchor_delete):
        ratio = None if not winner_keep else policy_keep_ratio(winner_keep)
        return SignedEditUpdate(
            action="skip",
            winner_keep=tuple(winner_keep),
            anchor_delete=tuple(anchor_delete),
            winner_token_advantage=tuple(0.0 for _ in winner_keep),
            anchor_token_advantage=tuple(0.0 for _ in anchor_delete),
            winner_keep_ratio=ratio,
        )
    ratio = None if not winner_keep else policy_keep_ratio(winner_keep)
    return SignedEditUpdate(
        action="update",
        winner_keep=tuple(winner_keep),
        anchor_delete=tuple(anchor_delete),
        winner_token_advantage=tuple(gap * value for value in winner_keep),
        anchor_token_advantage=tuple(-gap * value for value in anchor_delete),
        winner_keep_ratio=ratio,
    )


@dataclass(frozen=True)
class SignedEditLoss:
    """Sequence advantages and masks for the length-2 loss.

    ``sequence_advantage`` is ``(-gap, gap)`` on an update. The anchor mask
    keeps only deletions, so those positions receive ``-gap`` and the other
    anchor positions stay 0. The winner mask keeps substitutions and insertions.
    """

    action: str
    sequence_advantage: tuple[float, float]
    anchor_delete: tuple[float, ...]
    winner_keep: tuple[float, ...]
    winner_keep_ratio: Optional[float]


def signed_edit_loss_args(
    anchor_ids: Sequence[int],
    winner_ids: Sequence[int],
    winner_advantage: float,
) -> SignedEditLoss:
    """Build the length-2 loss arguments for one anchored pair.

    ``winner_advantage`` is the gap already chosen for the winner sequence.
    A deletion-only pair still returns ``update``. Identical token ids return
    ``skip`` with zero sequence advantages.
    """
    update = signed_edit_update(anchor_ids, winner_ids, winner_advantage)
    if update.action == "skip":
        return SignedEditLoss(
            action="skip",
            sequence_advantage=(0.0, 0.0),
            anchor_delete=update.anchor_delete,
            winner_keep=update.winner_keep,
            winner_keep_ratio=update.winner_keep_ratio,
        )
    gap = float(winner_advantage)
    return SignedEditLoss(
        action="update",
        sequence_advantage=(-gap, gap),
        anchor_delete=update.anchor_delete,
        winner_keep=update.winner_keep,
        winner_keep_ratio=update.winner_keep_ratio,
    )


def policy_keep_ratio(keep: Sequence[float]) -> float:
    """Fraction of winner tokens that receive a policy gradient."""
    if not keep:
        raise ValueError("keep mask is empty")
    kept = 0.0
    for value in keep:
        number = float(value)
        if number not in (0.0, 1.0):
            raise ValueError(f"keep values must be 0 or 1, got {value!r}")
        kept += number
    return kept / float(len(keep))


def scatter_keep(
    labels: Sequence[int],
    keep: Sequence[float],
    ignore_index: int = -100,
) -> list[float]:
    """Place ``keep`` onto the shifted response positions of one label row.

    Position 0 of ``labels`` is never predicted. Response tokens are the later
    labels that are not ``ignore_index``, in order. The result has length
    ``len(labels) - 1`` and is 0 on prompt and padding positions.
    """
    if not labels:
        raise ValueError("labels are empty")
    shifted = [0.0] * (len(labels) - 1)
    positions = [
        index - 1
        for index in range(1, len(labels))
        if int(labels[index]) != int(ignore_index)
    ]
    if len(positions) != len(keep):
        raise ValueError(
            f"keep length {len(keep)} does not match {len(positions)} response tokens"
        )
    for pos, value in zip(positions, keep):
        number = float(value)
        if number not in (0.0, 1.0):
            raise ValueError(f"keep values must be 0 or 1, got {value!r}")
        shifted[pos] = number
    return shifted
