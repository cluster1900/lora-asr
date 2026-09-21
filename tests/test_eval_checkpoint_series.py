#!/usr/bin/env python3
"""Unit tests for evaluation/eval_checkpoint_series.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.eval_checkpoint_series import (
    discover_step_checkpoints,
    extract_summary,
    generate_comparison_markdown,
    parse_args,
    select_best_checkpoint,
)


class EvalCheckpointSeriesTest(unittest.TestCase):
    def test_parse_args(self) -> None:
        args = parse_args([
            "--run-dir", "/data/mega-asr/runs/sft_pilot_controlled",
            "--manifest", "/data/mega-asr/manifests/validation.jsonl",
            "--base-metrics", "/data/mega-asr/runs/base/metrics.json",
            "--output-dir", "/data/mega-asr/runs/eval_series",
            "--steps", "100,150,200",
            "--clean-target-min", "0.020",
            "--clean-target-max", "0.025",
        ])
        self.assertEqual(args.run_dir, "/data/mega-asr/runs/sft_pilot_controlled")
        self.assertEqual(args.steps, "100,150,200")
        self.assertAlmostEqual(args.clean_target_min, 0.020)
        self.assertAlmostEqual(args.clean_target_max, 0.025)

    def test_discover_step_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            ckpts = run_dir / "checkpoints"
            ckpts.mkdir()
            (ckpts / "step_50").mkdir()
            (ckpts / "step_100").mkdir()
            (ckpts / "step_150").mkdir()
            (ckpts / "step_200").mkdir()

            # Specific selection
            found = discover_step_checkpoints(run_dir, "100,200")
            self.assertEqual([s for s, _ in found], [100, 200])

            # Auto discovery
            found_auto = discover_step_checkpoints(run_dir, "auto")
            self.assertEqual([s for s, _ in found_auto], [50, 100, 150, 200])

    def test_extract_summary_and_target_detection(self) -> None:
        mock_eval_res = {
            "step": 150,
            "checkpoint_dir": "/tmp/step_150",
            "adapter_dir": "/tmp/step_150/adapter",
            "predictions_path": "/tmp/step_150/pred.jsonl",
            "metrics": {
                "overall": {
                    "language_macro_error_rate": 0.08,
                    "clean_language_macro_error_rate": 0.023,
                    "degraded_language_macro_error_rate": 0.09,
                },
                "by_scenario": [
                    {"group": "en|clean", "error_rate": 0.022},
                    {"group": "zh|clean", "error_rate": 0.024},
                    {"group": "en|noise", "error_rate": 0.07},
                ],
            },
            "gate": {
                "gate_status": "PASSED",
                "improved_degraded_scenarios_count": 5,
                "improved_scenarios": ["en|noise"],
                "clean_error_rate_increase": 0.003,
                "valid_output_rate": 1.0,
                "failure_rate": 0.0,
            },
        }

        summary = extract_summary(mock_eval_res, clean_target_min=0.020, clean_target_max=0.025)
        self.assertEqual(summary["step"], 150)
        self.assertAlmostEqual(summary["en_clean_wer"], 0.022)
        self.assertTrue(summary["clean_in_target"])
        self.assertEqual(summary["gate_status"], "PASSED")
        self.assertEqual(summary["improved_degraded_scenarios_count"], 5)

        # Out of target (e.g. 0.034)
        mock_eval_res["metrics"]["by_scenario"][0]["error_rate"] = 0.034
        summary2 = extract_summary(mock_eval_res, clean_target_min=0.020, clean_target_max=0.025)
        self.assertFalse(summary2["clean_in_target"])

    def test_select_best_checkpoint_pareto(self) -> None:
        summaries = [
            {
                "step": 100,
                "checkpoint_dir": "/tmp/100",
                "adapter_dir": "/tmp/100/adapter",
                "gate_status": "PASSED",
                "clean_in_target": True,
                "en_clean_wer": 0.021,
                "zh_clean_cer": 0.022,
                "improved_degraded_scenarios_count": 4,
                "overall_error_rate": 0.095,
            },
            {
                "step": 150,
                "checkpoint_dir": "/tmp/150",
                "adapter_dir": "/tmp/150/adapter",
                "gate_status": "PASSED",
                "clean_in_target": True,
                "en_clean_wer": 0.023,
                "zh_clean_cer": 0.022,
                "improved_degraded_scenarios_count": 6,
                "overall_error_rate": 0.085,
            },
            {
                "step": 300,
                "checkpoint_dir": "/tmp/300",
                "adapter_dir": "/tmp/300/adapter",
                "gate_status": "PASSED",
                "clean_in_target": False,  # Clean regressed past target
                "en_clean_wer": 0.032,
                "zh_clean_cer": 0.025,
                "improved_degraded_scenarios_count": 7,
                "overall_error_rate": 0.075,
            },
            {
                "step": 50,
                "checkpoint_dir": "/tmp/50",
                "adapter_dir": "/tmp/50/adapter",
                "gate_status": "FAILED",
                "clean_in_target": True,
                "en_clean_wer": 0.019,
                "zh_clean_cer": 0.020,
                "improved_degraded_scenarios_count": 0,
                "overall_error_rate": 0.110,
            },
        ]

        best = select_best_checkpoint(summaries, clean_target_min=0.020, clean_target_max=0.025)
        self.assertIsNotNone(best)
        # Step 100 has lower en_clean_wer (0.021 vs 0.023) within the target range while passing the gate
        self.assertEqual(best["step"], 100)

    def test_select_best_checkpoint_rejects_robust_regression_and_picks_step50(self) -> None:
        summaries = [
            {
                "step": 50,
                "checkpoint_dir": "/tmp/50",
                "adapter_dir": "/tmp/50/adapter",
                "gate_status": "PASSED",
                "clean_in_target": True,
                "en_clean_wer": 0.0194,
                "zh_clean_cer": 0.0140,
                "clean_macro_error_rate": 0.0167,
                "degraded_macro_error_rate": 0.1040,
                "robust_increase": -0.0012,  # improved
                "improved_degraded_scenarios_count": 6,
                "overall_error_rate": 0.0424,
            },
            {
                "step": 100,
                "checkpoint_dir": "/tmp/100",
                "adapter_dir": "/tmp/100/adapter",
                "gate_status": "PASSED",
                "clean_in_target": False,
                "en_clean_wer": 0.0306,
                "zh_clean_cer": 0.0144,
                "clean_macro_error_rate": 0.0225,
                "degraded_macro_error_rate": 0.1370,
                "robust_increase": 0.0318,  # regressed (+3.18% > 0.005)
                "improved_degraded_scenarios_count": 6,
                "overall_error_rate": 0.0569,
            },
        ]
        best = select_best_checkpoint(summaries, clean_target_min=0.020, clean_target_max=0.025, max_robust_macro_regression=0.005)
        self.assertIsNotNone(best)
        self.assertEqual(best["step"], 50)

    def test_generate_comparison_markdown(self) -> None:
        base_metrics = {
            "overall": {
                "clean_language_macro_error_rate": 0.020,
                "degraded_language_macro_error_rate": 0.100,
            },
            "by_scenario": [
                {"group": "en|clean", "error_rate": 0.0191},
                {"group": "zh|clean", "error_rate": 0.0209},
            ],
        }
        summaries = [
            {
                "step": 150,
                "checkpoint_dir": "/tmp/150",
                "adapter_dir": "/tmp/150/adapter",
                "gate_status": "PASSED",
                "clean_in_target": True,
                "en_clean_wer": 0.023,
                "zh_clean_cer": 0.022,
                "clean_macro_error_rate": 0.0225,
                "degraded_macro_error_rate": 0.085,
                "improved_degraded_scenarios_count": 6,
                "clean_increase": 0.0025,
                "overall_error_rate": 0.085,
            }
        ]
        best = summaries[0]
        md = generate_comparison_markdown(base_metrics, summaries, best)
        self.assertIn("Checkpoint Series Evaluation", md)
        self.assertIn("Step 150", md)
        self.assertIn("Pareto Optimal (Selected)", md)
        self.assertIn("Base (pretrained)", md)


if __name__ == "__main__":
    unittest.main()
