"""CLI contract for joining a manifest with predictions."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.eval_cer import evaluate, main
from inference.config import decode_eval_sha256, load_infer_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/inference/base_eval.yaml"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


class EvalCerTest(unittest.TestCase):
    def test_micro_and_macro_and_null_wer(self) -> None:
        config = load_infer_config(CONFIG)
        manifest = [
            {"sample_id": "a", "text": "你好", "scenario": "clean", "language": "zh"},
            {"sample_id": "b", "text": "你好", "scenario": "noise", "language": "zh"},
        ]
        predictions = [
            {"sample_id": "a", "text": "你好", "prediction_raw": "你好", "error": None, "stop_reason": "eos"},
            {"sample_id": "b", "text": "你好", "prediction_raw": "你号", "error": None, "stop_reason": "eos"},
        ]
        scored, metrics = evaluate(
            manifest,
            predictions,
            manifest_sha256="m",
            predictions_sha256="p",
            decode_eval_sha256=config["decode_eval_sha256"],
            wer_tokenizer=None,
        )
        self.assertEqual([row["edit_count"] for row in scored], [0, 1])
        self.assertEqual(metrics["edit_count"], 1)
        self.assertEqual(metrics["reference_char_count"], 4)
        self.assertEqual(metrics["micro_cer"], 0.25)
        self.assertEqual(metrics["macro_cer"], 0.25)
        self.assertIsNone(metrics["wer"])
        self.assertEqual(metrics["wer_reason"], "wer_tokenizer_not_configured")
        self.assertEqual(metrics["by_scenario"]["noise"]["micro_cer"], 0.5)

    def test_duplicate_prediction_fails_the_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            predictions = root / "predictions.jsonl"
            _write_jsonl(manifest, [{"sample_id": "a", "text": "你好", "scenario": "clean"}])
            _write_jsonl(
                predictions,
                [
                    {"sample_id": "a", "text": "你好", "prediction_raw": "你好"},
                    {"sample_id": "a", "text": "你好", "prediction_raw": "你好"},
                ],
            )
            code = main([
                "--manifest",
                str(manifest),
                "--predictions",
                str(predictions),
                "--output-dir",
                str(root / "eval"),
                "--config",
                str(CONFIG),
            ])
            self.assertEqual(code, 2)
            self.assertFalse((root / "eval" / "metrics.json").exists())

    def test_config_hash_is_stable(self) -> None:
        config = load_infer_config(CONFIG)
        again = decode_eval_sha256(config["decode_eval"])
        self.assertEqual(config["decode_eval_sha256"], again)
        self.assertFalse(config["decode_eval"]["do_sample"])


if __name__ == "__main__":
    unittest.main()
