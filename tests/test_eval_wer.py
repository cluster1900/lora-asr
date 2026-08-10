from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation import eval_wer


class EvalWerTest(unittest.TestCase):
    def test_english_wer_and_chinese_cer_are_separate(self) -> None:
        rows = [
            {
                "sample_id": "en-1",
                "answer": "One two three",
                "prediction": "one too three",
                "language": "en",
                "scenario": "noise",
            },
            {
                "sample_id": "zh-1",
                "answer": "你好世界",
                "prediction": "你好世间",
                "language": "zh",
                "scenario": "noise",
            },
        ]

        scored, metrics = eval_wer.evaluate(rows)

        self.assertEqual(scored[0]["metric"], "wer")
        self.assertEqual(scored[0]["ref_len"], 3)
        self.assertAlmostEqual(scored[0]["error_rate"], 1 / 3, places=6)
        self.assertEqual(scored[1]["metric"], "cer")
        self.assertEqual(scored[1]["ref_len"], 4)
        self.assertAlmostEqual(scored[1]["error_rate"], 0.25)
        self.assertEqual(metrics["overall"]["metric"], "mixed")
        self.assertIsNone(metrics["overall"]["error_rate"])
        self.assertEqual({row["metric"] for row in metrics["by_language"]}, {"wer", "cer"})
        self.assertEqual({row["group"] for row in metrics["by_scenario"]}, {"en|noise", "zh|noise"})

    def test_empty_reference_is_a_hard_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "Empty reference.*bad"):
            eval_wer.score_item({"sample_id": "bad", "answer": "...", "prediction": "x"})

    def test_inference_error_is_scored_as_full_deletion(self) -> None:
        row = eval_wer.score_item({
            "sample_id": "failed",
            "answer": "a correct transcript",
            "prediction": "a correct transcript",
            "language": "en",
            "error": "RuntimeError: decode failed",
        })

        self.assertEqual(row["prediction_raw"], "a correct transcript")
        self.assertEqual(row["scored_prediction_normalized"], "")
        self.assertEqual(row["error_rate"], 1.0)
        self.assertTrue(row["inference_error"])
        self.assertIn("inference_error", row["failure_tags"])

    def test_repetition_and_empty_rates_are_aggregated(self) -> None:
        rows = [
            eval_wer.score_item({
                "answer": "go home now",
                "prediction": "go go go go",
                "language": "en",
                "scenario": "dropout",
            }),
            eval_wer.score_item({
                "answer": "stay here",
                "prediction": "",
                "language": "en",
                "scenario": "dropout",
            }),
        ]
        summary = eval_wer.aggregate(rows, "scenario")[0]

        self.assertEqual(summary["repeat_like_outputs"], 1)
        self.assertEqual(summary["empty_outputs"], 1)
        self.assertEqual(summary["repeat_like_output_rate"], 0.5)
        self.assertEqual(summary["empty_output_rate"], 0.5)

    def test_complete_bench_has_32_cells(self) -> None:
        rows = []
        for language in ("en", "zh"):
            for origin in ("real", "synthetic"):
                for scenario in eval_wer.BENCH_SCENARIOS:
                    reference = "ok words" if language == "en" else "正确"
                    rows.append(eval_wer.score_item({
                        "answer": reference,
                        "prediction": reference,
                        "language": language,
                        "audio_origin": origin,
                        "scenario": scenario,
                    }))

        cells, summary = eval_wer.bench_cells(rows)

        self.assertEqual(len(cells), 32)
        self.assertTrue(summary["complete"])
        self.assertEqual(summary["observed_expected_cells"], 32)
        self.assertEqual(summary["macro_error_rate"], 0.0)
        self.assertEqual(summary["missing_cells"], [])
        self.assertEqual(
            [row["macro_error_rate"] for row in summary["by_source_type"]],
            [0.0, 0.0],
        )
        self.assertTrue(all(row["complete"] for row in summary["by_source_type"]))
        self.assertEqual({row["source_type"] for row in cells}, {"real", "synthetic"})

    def test_cli_writes_json_and_both_csv_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = root / "predictions.jsonl"
            predictions.write_text(json.dumps({
                "answer": "hello world",
                "prediction": "hello world",
                "language": "en",
                "scenario": "noise",
                "audio_origin": "real",
            }) + "\n", encoding="utf-8")

            eval_wer.main([
                "--predictions-jsonl", str(predictions),
                "--output-dir", str(root / "report"),
            ])

            report = root / "report"
            metrics = json.loads((report / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["overall"]["error_rate"], 0.0)
            self.assertTrue((report / "scored.jsonl").is_file())
            self.assertTrue((report / "by_scenario.csv").is_file())
            self.assertTrue((report / "by_cell.csv").is_file())
            self.assertTrue((report / "by_language.csv").is_file())


if __name__ == "__main__":
    unittest.main()
