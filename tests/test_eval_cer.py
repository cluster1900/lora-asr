"""CLI contract for joining a manifest with predictions."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.eval_cer import evaluate, main
from evaluation.zh_cer import LOCKED_JIEBA_DICT_SHA256, LOCKED_JIEBA_VERSION
from inference.config import decode_eval_sha256, load_infer_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/inference/base_eval.yaml"
WER_CONFIG = ROOT / "configs/inference/base_eval_wer.yaml"


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

    def test_locked_jieba_wer_cli_and_decode_hash_parity(self) -> None:
        base_cfg = load_infer_config(CONFIG)
        wer_cfg = load_infer_config(WER_CONFIG)
        self.assertEqual(wer_cfg["decode_eval_sha256"], base_cfg["decode_eval_sha256"])
        self.assertEqual(wer_cfg["scoring"]["wer_tokenizer"], "jieba")
        self.assertEqual(wer_cfg["scoring"]["wer_tokenizer_version"], LOCKED_JIEBA_VERSION)
        self.assertEqual(wer_cfg["scoring"]["wer_dictionary_sha256"], LOCKED_JIEBA_DICT_SHA256)
        self.assertFalse(wer_cfg["scoring"]["wer_hmm"])

        manifest = [
            {"sample_id": "c1", "text": "甚至出现交易几乎停滞的情况", "scenario": "clean", "language": "zh"},
            {"sample_id": "n1", "text": "来自北汽女排的主攻手", "scenario": "noise", "language": "zh"},
            {"sample_id": "d1", "text": "小行星撞地球", "scenario": "dropout", "language": "zh"},
        ]
        predictions = [
            {
                "sample_id": "c1",
                "prediction_raw": "甚至出现交易几乎停滞的情况。",
                "error": None,
                "stop_reason": "eos",
            },
            {
                "sample_id": "n1",
                "prediction_raw": "来自北汽女才的主攻手",
                "error": None,
                "stop_reason": "eos",
            },
            {
                "sample_id": "d1",
                "prediction_raw": "忽略我",
                "error": "OSError: missing wav",
                "stop_reason": None,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.jsonl"
            pred_path = root / "predictions.jsonl"
            out_dir = root / "eval_wer"
            _write_jsonl(manifest_path, manifest)
            _write_jsonl(pred_path, predictions)
            code = main([
                "--manifest",
                str(manifest_path),
                "--predictions",
                str(pred_path),
                "--output-dir",
                str(out_dir),
                "--config",
                str(WER_CONFIG),
            ])
            self.assertEqual(code, 0)
            metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
            scored = [
                json.loads(line)
                for line in (out_dir / "scored.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            with (out_dir / "by_scenario.csv").open("r", encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertIsNone(metrics["wer_reason"])
        self.assertIsNotNone(metrics["wer"])
        self.assertEqual(metrics["wer"], metrics["micro_wer"])
        self.assertEqual(metrics["wer_tokenizer_info"]["name"], "jieba")
        self.assertEqual(metrics["wer_tokenizer_info"]["version"], LOCKED_JIEBA_VERSION)
        self.assertEqual(metrics["wer_tokenizer_info"]["dictionary_sha256"], LOCKED_JIEBA_DICT_SHA256)
        self.assertFalse(metrics["wer_tokenizer_info"]["hmm"])

        self.assertEqual(scored[0]["word_edit_count"], 0)
        self.assertEqual(scored[0]["sample_wer"], 0.0)
        self.assertGreater(scored[1]["word_edit_count"], 0)
        self.assertEqual(scored[2]["scored_prediction_words"], [])
        self.assertEqual(scored[2]["word_edit_count"], scored[2]["reference_word_count"])
        self.assertEqual(scored[2]["sample_wer"], 1.0)
        self.assertEqual(metrics["by_scenario"]["clean"]["micro_wer"], 0.0)
        self.assertEqual(metrics["by_scenario"]["dropout"]["micro_wer"], 1.0)
        self.assertIn("micro_wer", csv_rows[0])
        self.assertIn("macro_wer", csv_rows[0])

    def test_wer_rejects_bad_tokenizer_version_or_dictionary_hash(self) -> None:
        manifest = [{"sample_id": "a", "text": "你好世界", "scenario": "clean", "language": "zh"}]
        predictions = [{"sample_id": "a", "prediction_raw": "你好世界", "error": None, "stop_reason": "eos"}]
        with self.assertRaisesRegex(ValueError, "unsupported wer_tokenizer"):
            evaluate(
                manifest,
                predictions,
                manifest_sha256="m",
                predictions_sha256="p",
                decode_eval_sha256="d",
                wer_tokenizer="whitespace",
                wer_tokenizer_version="1.0",
                wer_dictionary_sha256="a" * 64,
            )
        with self.assertRaisesRegex(ValueError, "jieba version mismatch"):
            evaluate(
                manifest,
                predictions,
                manifest_sha256="m",
                predictions_sha256="p",
                decode_eval_sha256="d",
                wer_tokenizer="jieba",
                wer_tokenizer_version="9.9.9",
                wer_dictionary_sha256=LOCKED_JIEBA_DICT_SHA256,
            )
        with self.assertRaisesRegex(ValueError, "jieba dictionary sha256 mismatch"):
            evaluate(
                manifest,
                predictions,
                manifest_sha256="m",
                predictions_sha256="p",
                decode_eval_sha256="d",
                wer_tokenizer="jieba",
                wer_tokenizer_version=LOCKED_JIEBA_VERSION,
                wer_dictionary_sha256="0" * 64,
            )
        with tempfile.TemporaryDirectory() as tmp:
            bad_cfg = Path(tmp) / "bad.yaml"
            bad_cfg.write_text(
                CONFIG.read_text(encoding="utf-8").replace(
                    "  wer_tokenizer: null",
                    "  wer_tokenizer: null\n  wer_hmm: true",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "null wer_tokenizer requires wer_hmm to be false"):
                load_infer_config(bad_cfg)

    def test_isolated_jieba_tokenizer_ignores_global_user_words(self) -> None:
        import jieba  # type: ignore[import-untyped]

        manifest = [{"sample_id": "a", "text": "北汽女排主攻手", "scenario": "clean", "language": "zh"}]
        predictions = [{"sample_id": "a", "prediction_raw": "北汽女才主攻手", "error": None, "stop_reason": "eos"}]
        before_scored, _ = evaluate(
            manifest,
            predictions,
            manifest_sha256="m",
            predictions_sha256="p",
            decode_eval_sha256="d",
            wer_tokenizer="jieba",
            wer_tokenizer_version=LOCKED_JIEBA_VERSION,
            wer_dictionary_sha256=LOCKED_JIEBA_DICT_SHA256,
        )
        jieba.add_word("北汽女排主攻手", freq=999999)
        try:
            after_scored, _ = evaluate(
                manifest,
                predictions,
                manifest_sha256="m",
                predictions_sha256="p",
                decode_eval_sha256="d",
                wer_tokenizer="jieba",
                wer_tokenizer_version=LOCKED_JIEBA_VERSION,
                wer_dictionary_sha256=LOCKED_JIEBA_DICT_SHA256,
            )
        finally:
            jieba.del_word("北汽女排主攻手")
        self.assertEqual(before_scored[0]["reference_words"], after_scored[0]["reference_words"])
        self.assertGreater(len(after_scored[0]["reference_words"]), 1)


if __name__ == "__main__":
    unittest.main()
