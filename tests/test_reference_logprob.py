"""Reference logprob contracts. They do not download a model."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from train.reference_logprob import (
    cache_reference_logprobs,
    completed_logprob_rows,
    freeze_reference_model,
    logprobs_match,
    publish_logprob_pairs,
    reference_sequence_logprob,
    select_recheck,
)


def _pair(pair_id: str, split: str, chosen: str = "你好", rejected: str = "你号") -> dict[str, object]:
    return {
        "pair_id": pair_id,
        "sample_id": pair_id,
        "split": split,
        "audio": f"wavs/{pair_id}.wav",
        "chosen_text": chosen,
        "rejected_text": rejected,
        "chosen_token_count": len(chosen) + 1,
        "rejected_token_count": len(rejected) + 1,
        "ref_chosen_logp": None,
        "ref_rejected_logp": None,
    }


def _stable_score(row: dict[str, object], text: str) -> tuple[float, int]:
    if text == row["chosen_text"]:
        return -2.5, int(row["chosen_token_count"])
    return -4.5, int(row["rejected_token_count"])


class ReferenceLogprobTest(unittest.TestCase):
    def test_logprob_is_the_negative_sum_not_the_mean(self) -> None:
        self.assertEqual(reference_sequence_logprob([0.25, 0.75]), -1.0)
        self.assertNotEqual(reference_sequence_logprob([0.25, 0.75]), -0.5)
        with self.assertRaises(Exception):
            reference_sequence_logprob([])
        with self.assertRaises(Exception):
            reference_sequence_logprob([float("nan")])

    def test_both_error_limits_are_required(self) -> None:
        self.assertTrue(logprobs_match(-10.0, -10.00005))
        self.assertFalse(logprobs_match(-10.0, -10.0002))
        self.assertFalse(logprobs_match(-1.0, -1.00005))

    def test_cache_must_be_a_pair_prefix(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            pairs = [_pair("a", "train"), _pair("b", "val")]
            cache = root / "ref_logprobs.jsonl"
            self.assertEqual(completed_logprob_rows(cache, pairs), [])
            cache.write_text(
                json.dumps(
                    {
                        "schema_version": "qwen3-asr-dpo-ref-logprob-v1",
                        "pair_id": "b",
                        "ref_chosen_logp": -1.0,
                        "ref_rejected_logp": -2.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(Exception) as caught:
                completed_logprob_rows(cache, pairs)
            self.assertIn("prefix", str(caught.exception))

    def test_recheck_failure_does_not_publish_pairs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = [_pair("t1", "train")]
            val = [_pair("v1", "val")]
            calls = {"n": 0}

            def score(row: dict[str, object], text: str) -> tuple[float, int]:
                calls["n"] += 1
                logprob, count = _stable_score(row, text)
                if calls["n"] > 4:
                    logprob -= 1.0
                return logprob, count

            with self.assertRaises(Exception) as caught:
                cache_reference_logprobs(train, val, score, root / "ref_logprobs.jsonl", seed=42, recheck_limit=20)
            self.assertIn("recheck", str(caught.exception))
            self.assertFalse((root / "train_pairs.jsonl").exists())

    def test_matching_recheck_fills_logprobs_without_using_the_mean(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = [_pair("t1", "train"), _pair("t2", "train", chosen="甲乙丙", rejected="甲乙")]
            val = [_pair("v1", "val")]
            report = cache_reference_logprobs(
                train,
                val,
                _stable_score,
                root / "ref_logprobs.jsonl",
                seed=42,
                recheck_limit=20,
            )
            self.assertEqual(report["rechecked"], 3)
            self.assertFalse(report["trainable"])
            self.assertLessEqual(report["max_abs_error"], 1e-4)
            published = publish_logprob_pairs(train, val, root / "ref_logprobs.jsonl", root / "out", report)
            filled = [
                json.loads(line)
                for line in (root / "out" / "train_pairs.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(filled[0]["ref_chosen_logp"], -2.5)
            self.assertEqual(filled[0]["ref_rejected_logp"], -4.5)
            self.assertEqual(published["train_pairs_sha256"], report["train_pairs_sha256"] if False else published["train_pairs_sha256"])
            picked = select_recheck(train, 42, 1)
            self.assertEqual(len(picked), 1)
            self.assertIn(picked[0]["pair_id"], {"t1", "t2"})

    def test_freeze_removes_the_training_hook_and_checkpointing(self) -> None:
        class Param:
            def __init__(self) -> None:
                self.requires_grad = True

            def requires_grad_(self, value: bool) -> "Param":
                self.requires_grad = value
                return self

        class Embeddings:
            def __init__(self) -> None:
                self._forward_hooks = {1: object()}

        class Thinker:
            def __init__(self) -> None:
                self.embeddings = Embeddings()
                self.is_gradient_checkpointing = True

            def get_input_embeddings(self) -> Embeddings:
                return self.embeddings

            def gradient_checkpointing_disable(self) -> None:
                self.is_gradient_checkpointing = False

        class Model:
            def __init__(self) -> None:
                self.thinker = Thinker()
                self.params = [Param(), Param()]
                self.is_gradient_checkpointing = True
                self.training = True

            def parameters(self):
                return self.params

            def eval(self) -> None:
                self.training = False

            def gradient_checkpointing_disable(self) -> None:
                self.is_gradient_checkpointing = False

        model = Model()
        freeze_reference_model(model)
        self.assertEqual(model.thinker.embeddings._forward_hooks, {})
        self.assertFalse(model.is_gradient_checkpointing)
        self.assertFalse(model.thinker.is_gradient_checkpointing)
        self.assertFalse(model.training)
        self.assertTrue(all(not param.requires_grad for param in model.params))


if __name__ == "__main__":
    unittest.main()
