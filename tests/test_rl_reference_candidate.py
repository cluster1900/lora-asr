"""The reference candidate is appended, then the shipped local winner decides."""

import unittest

from train.rl_local_winner import local_winner_index
from train.rl_reference_candidate import append_reference_candidate


class ReferenceCandidateTest(unittest.TestCase):
    def test_close_reference_becomes_the_update(self) -> None:
        texts, rewards = append_reference_candidate(
            ["the cat sat", "the cat sit"],
            [0.40, 0.41],
            "the cat sits",
            1.0,
        )
        self.assertEqual(texts, ["the cat sat", "the cat sit", "the cat sits"])
        self.assertEqual(rewards, [0.40, 0.41, 1.0])
        winner, status = local_winner_index(
            texts,
            rewards,
            language="en",
            min_improvement=0.02,
            max_relative=0.35,
        )
        self.assertEqual(status, "update")
        self.assertEqual(winner, 2)
        self.assertEqual(texts[winner], "the cat sits")

    def test_distant_reference_stays_no_improvement(self) -> None:
        texts, rewards = append_reference_candidate(
            ["the cat sat", "the cat sit"],
            [0.40, 0.41],
            "one two three four five six seven eight nine ten",
            1.0,
        )
        winner, status = local_winner_index(
            texts,
            rewards,
            language="en",
            min_improvement=0.02,
            max_relative=0.35,
        )
        self.assertIsNone(winner)
        self.assertEqual(status, "no_improvement")

    def test_existing_normalized_reference_is_not_duplicated(self) -> None:
        texts, rewards = append_reference_candidate(
            ["hello world", "hello"],
            [0.20, 0.10],
            "Hello, world.",
            1.0,
        )
        self.assertEqual(texts, ["hello world", "hello"])
        self.assertEqual(rewards, [0.20, 0.10])

    def test_empty_reference_is_left_out(self) -> None:
        texts, rewards = append_reference_candidate(
            ["hello world"],
            [0.20],
            "   ",
            1.0,
        )
        self.assertEqual(texts, ["hello world"])
        self.assertEqual(rewards, [0.20])

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            append_reference_candidate(["hello"], [0.1, 0.2], "hello there", 1.0)


if __name__ == "__main__":
    unittest.main()
