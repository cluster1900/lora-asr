"""Local-correction winner selection for rl_pilot_v16."""

import unittest

from train.rl_local_winner import local_edit_limit, local_winner_index


class LocalWinnerTest(unittest.TestCase):
    def test_relative_limit_uses_thousandths(self) -> None:
        self.assertEqual(local_edit_limit(20, 20), 7)
        self.assertEqual(local_edit_limit(8, 8), 2)
        self.assertEqual(local_edit_limit(1, 1), 2)

    def test_one_word_correction_is_selected(self) -> None:
        index, status = local_winner_index(
            [
                "Ayr has won eight of the last nine meetings in this series.",
                "Iowa has won eight of the last nine meetings in this series.",
            ],
            [0.50, 0.58],
            language="en",
        )
        self.assertEqual(index, 1)
        self.assertEqual(status, "update")

    def test_full_sentence_replacement_does_not_train(self) -> None:
        index, status = local_winner_index(
            [
                "Let's move the conversation forward.",
                "That's my number.",
            ],
            [-0.25, -0.20],
            language="en",
        )
        self.assertIsNone(index)
        self.assertEqual(status, "no_improvement")

    def test_near_sample_beats_a_larger_distant_gap(self) -> None:
        index, status = local_winner_index(
            [
                "Ayr has won eight of the last nine meetings in this series.",
                "Iowa has won eight of the last nine meetings in this series.",
                "In the difficult moments we recognize our thirst for fulfillment.",
            ],
            [0.40, 0.46, 0.95],
            language="en",
        )
        self.assertEqual(index, 1)
        self.assertEqual(status, "update")

    def test_equal_local_gaps_keep_the_lower_index(self) -> None:
        anchor = "the old fairground sideshow featured the magician nightly"
        first = "the old fairground sideshow featured a magician nightly"
        second = "the old fairground sideshow featured one magician nightly"
        index, status = local_winner_index(
            [anchor, first, second],
            [0.50, 0.56, 0.56],
            language="en",
        )
        self.assertEqual(index, 1)
        self.assertEqual(status, "update")

    def test_chinese_character_correction_is_selected(self) -> None:
        index, status = local_winner_index(
            ["大伙子当初失踪。", "大皇子当初失踪。"],
            [0.70, 0.84],
            language="zh",
        )
        self.assertEqual(index, 1)
        self.assertEqual(status, "update")

    def test_gap_below_the_minimum_does_not_train(self) -> None:
        index, status = local_winner_index(
            ["alpha beta gamma delta", "alpha beta gamma delta extra"],
            [0.50, 0.519],
            language="en",
        )
        self.assertIsNone(index)
        self.assertEqual(status, "no_improvement")

    def test_gap_at_the_minimum_trains(self) -> None:
        index, status = local_winner_index(
            ["alpha beta gamma delta epsilon", "alpha beta gamma delta extra"],
            [0.50, 0.52],
            language="en",
        )
        self.assertEqual(index, 1)
        self.assertEqual(status, "update")

    def test_identical_rewards_are_identical(self) -> None:
        index, status = local_winner_index(
            ["same words here", "same words here"],
            [0.40, 0.40],
            language="en",
        )
        self.assertIsNone(index)
        self.assertEqual(status, "identical")


if __name__ == "__main__":
    unittest.main()
