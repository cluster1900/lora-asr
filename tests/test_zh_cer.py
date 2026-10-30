"""Contract tests for zh-cer-v1 normalization and health flags."""

from __future__ import annotations

import unittest

from evaluation.zh_cer import (
    align_edits,
    edit_distance,
    gap_insertion_count,
    is_long_output,
    is_repeated,
    normalize_zh_cer_v1,
    score_pair,
)


class ZhCerTest(unittest.TestCase):
    def test_punctuation_and_space_do_not_create_edits(self) -> None:
        self.assertEqual(normalize_zh_cer_v1("你好。"), "你好")
        self.assertEqual(normalize_zh_cer_v1("你好 世界"), "你好世界")
        self.assertEqual(normalize_zh_cer_v1("ABC"), "abc")

    def test_character_edit_distance(self) -> None:
        scored = score_pair("你好", "你号", inference_error=False, stop_reason="eos")
        self.assertEqual(scored["edit_count"], 1)
        self.assertEqual(scored["reference_char_count"], 2)
        self.assertEqual(scored["sample_cer"], 0.5)

    def test_empty_reference_fails(self) -> None:
        with self.assertRaises(ValueError):
            score_pair("，。", "你好", inference_error=False, stop_reason="eos")

    def test_inference_error_scores_as_empty_hypothesis(self) -> None:
        scored = score_pair("你好", "你好", inference_error=True, stop_reason="error")
        self.assertEqual(scored["edit_count"], 2)
        self.assertFalse(scored["empty_output"])
        self.assertTrue(scored["inference_error"])
        self.assertFalse(scored["valid_output"])

    def test_repetition_and_length_flags(self) -> None:
        self.assertTrue(is_repeated("啊啊啊啊"))
        self.assertTrue(is_repeated("你好你好你好"))
        self.assertFalse(is_repeated("我们的目的"))
        self.assertFalse(is_repeated("啊啊"))
        self.assertTrue(is_long_output(1, 2))
        self.assertFalse(is_long_output(2, 3))
        self.assertEqual(edit_distance(list("ab"), list("ab")), 0)

    def test_truncation_follows_stop_reason(self) -> None:
        scored = score_pair("你好", "你好", inference_error=False, stop_reason="length")
        self.assertTrue(scored["truncated"])
        self.assertTrue(scored["valid_output"])

    def test_score_pair_keeps_the_edit_total_only(self) -> None:
        scored = score_pair("abcd", "abXcd", inference_error=False, stop_reason="eos")
        self.assertEqual(
            set(scored),
            {
                "normalization_version",
                "reference_normalized",
                "prediction_normalized",
                "scored_prediction_normalized",
                "edit_count",
                "reference_char_count",
                "sample_cer",
                "empty_output",
                "repeated_output",
                "long_output",
                "truncated",
                "inference_error",
                "valid_output",
            },
        )
        self.assertEqual(scored["edit_count"], 1)

    def test_alignment_matches_edit_distance_and_breaks_ties_toward_substitution(self) -> None:
        samples = [("你好世界", "你好"), ("xy", "yx"), ("abcd", "abXcd"), ("abcd", ""), ("a", "a")]
        for reference, hypothesis in samples:
            operations = align_edits(list(reference), list(hypothesis))
            edits = sum(item["op"] != "match" for item in operations)
            self.assertEqual(edits, edit_distance(list(reference), list(hypothesis)))
        tied = align_edits(list("xy"), list("yx"))
        self.assertEqual([item["op"] for item in tied], ["substitute", "substitute"])

    def test_gap_insertions_use_the_reference_anchor(self) -> None:
        reference = "0123456789"
        self.assertEqual(gap_insertion_count(reference, "0123" + ("X" * 30) + "456789"), 30)
        self.assertEqual(gap_insertion_count(reference, reference + ("X" * 30)), 0)
        self.assertEqual(gap_insertion_count(reference, ("X" * 30) + reference), 0)
        with self.assertRaises(ValueError):
            gap_insertion_count("", "X")


if __name__ == "__main__":
    unittest.main()
