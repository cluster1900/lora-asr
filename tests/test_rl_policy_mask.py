"""Changed-token policy mask used by the v24 trainer."""

import unittest

from train.rl_policy_mask import (
    changed_token_mask,
    policy_keep_ratio,
    scatter_keep,
    signed_edit_loss_args,
    signed_edit_update,
)


class PolicyMaskTest(unittest.TestCase):
    def test_identical_ids_keep_nothing(self) -> None:
        mask = changed_token_mask([10, 11, 12], [10, 11, 12])
        self.assertEqual(mask, [0.0, 0.0, 0.0])
        self.assertEqual(policy_keep_ratio(mask), 0.0)

    def test_middle_substitution_keeps_one_token(self) -> None:
        mask = changed_token_mask([10, 11, 12, 13], [10, 99, 12, 13])
        self.assertEqual(mask, [0.0, 1.0, 0.0, 0.0])
        self.assertEqual(policy_keep_ratio(mask), 0.25)

    def test_insertion_and_substitution(self) -> None:
        mask = changed_token_mask([1, 2, 3], [1, 9, 8, 3])
        self.assertEqual(mask, [0.0, 1.0, 1.0, 0.0])

    def test_deletion_does_not_mark_the_surviving_tokens(self) -> None:
        mask = changed_token_mask([1, 2, 3], [1, 3])
        self.assertEqual(mask, [0.0, 0.0])

    def test_empty_anchor_marks_every_winner_token(self) -> None:
        mask = changed_token_mask([], [4, 5])
        self.assertEqual(mask, [1.0, 1.0])

    def test_empty_winner_returns_an_empty_mask(self) -> None:
        self.assertEqual(changed_token_mask([1, 2], []), [])

    def test_match_is_preferred_when_costs_tie(self) -> None:
        mask = changed_token_mask([1, 2], [2])
        self.assertEqual(mask, [0.0])

    def test_scatter_aligns_the_response_suffix(self) -> None:
        labels = [-100, -100, 10, 11, 12, 13]
        keep = changed_token_mask([10, 11, 12, 13], [10, 99, 12, 13])
        shifted = scatter_keep(labels, keep)
        self.assertEqual(shifted, [0.0, 0.0, 1.0, 0.0, 0.0])

    def test_scatter_rejects_a_length_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            scatter_keep([-100, 5, 6], [1.0])

    def test_ratio_rejects_an_empty_mask(self) -> None:
        with self.assertRaises(ValueError):
            policy_keep_ratio([])

    def test_identical_pair_skips_the_update(self) -> None:
        update = signed_edit_update([10, 11, 12], [10, 11, 12], 0.25)
        self.assertEqual(update.action, "skip")
        self.assertEqual(update.winner_token_advantage, (0.0, 0.0, 0.0))
        self.assertEqual(update.anchor_token_advantage, (0.0, 0.0, 0.0))
        self.assertEqual(update.winner_keep_ratio, 0.0)

    def test_substitution_keeps_the_positive_gap_on_the_winner(self) -> None:
        update = signed_edit_update([10, 11, 12, 13], [10, 99, 12, 13], 0.25)
        self.assertEqual(update.action, "update")
        self.assertEqual(update.winner_token_advantage, (0.0, 0.25, 0.0, 0.0))
        self.assertEqual(update.anchor_token_advantage, (0.0, 0.0, 0.0, 0.0))

    def test_deletion_only_winner_updates_the_deleted_anchor_token(self) -> None:
        update = signed_edit_update([1, 2, 3], [1, 3], 0.25)
        self.assertEqual(update.action, "update")
        self.assertEqual(update.winner_keep, (0.0, 0.0))
        self.assertEqual(update.winner_token_advantage, (0.0, 0.0))
        self.assertEqual(update.anchor_delete, (0.0, 1.0, 0.0))
        self.assertEqual(update.anchor_token_advantage, (0.0, -0.25, 0.0))
        self.assertEqual(update.winner_keep_ratio, 0.0)

    def test_empty_winner_is_a_deletion_update(self) -> None:
        update = signed_edit_update([1, 2], [], 0.25)
        self.assertEqual(update.action, "update")
        self.assertEqual(update.winner_keep, ())
        self.assertEqual(update.winner_token_advantage, ())
        self.assertEqual(update.anchor_token_advantage, (-0.25, -0.25))
        self.assertIsNone(update.winner_keep_ratio)

    def test_tied_alignment_deletes_the_unmatched_anchor_prefix(self) -> None:
        update = signed_edit_update([1, 2], [2], 0.25)
        self.assertEqual(changed_token_mask([1, 2], [2]), [0.0])
        self.assertEqual(update.action, "update")
        self.assertEqual(update.winner_token_advantage, (0.0,))
        self.assertEqual(update.anchor_token_advantage, (-0.25, 0.0))

    def test_tied_cost_substitutes_before_it_deletes(self) -> None:
        update = signed_edit_update([1, 2, 3], [1, 9], 0.25)
        self.assertEqual(update.action, "update")
        self.assertEqual(update.winner_token_advantage, (0.0, 0.25))
        self.assertEqual(update.anchor_token_advantage, (0.0, -0.25, 0.0))

    def test_empty_pair_skips(self) -> None:
        update = signed_edit_update([], [], 0.25)
        self.assertEqual(update.action, "skip")
        self.assertEqual(update.winner_token_advantage, ())
        self.assertEqual(update.anchor_token_advantage, ())
        self.assertIsNone(update.winner_keep_ratio)

    def test_loss_args_put_the_negated_gap_on_deleted_tokens(self) -> None:
        loss = signed_edit_loss_args([1, 2, 3], [1, 3], 0.25)
        update = signed_edit_update([1, 2, 3], [1, 3], 0.25)
        self.assertEqual(loss.action, "update")
        self.assertEqual(loss.sequence_advantage, (-0.25, 0.25))
        anchor_token = tuple(loss.sequence_advantage[0] * value for value in loss.anchor_delete)
        winner_token = tuple(loss.sequence_advantage[1] * value for value in loss.winner_keep)
        self.assertEqual(anchor_token, update.anchor_token_advantage)
        self.assertEqual(winner_token, update.winner_token_advantage)
        self.assertEqual(anchor_token, (0.0, -0.25, 0.0))
        self.assertEqual(winner_token, (0.0, 0.0))

    def test_loss_args_skip_identical_ids(self) -> None:
        loss = signed_edit_loss_args([10, 11, 12], [10, 11, 12], 0.25)
        self.assertEqual(loss.action, "skip")
        self.assertEqual(loss.sequence_advantage, (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
