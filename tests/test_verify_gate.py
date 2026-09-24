from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation import verify_gate


class VerifyGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_base_metrics = {
            "overall": {
                "samples": 100,
                "inference_errors": 0,
                "empty_outputs": 0,
                "clean_language_macro_error_rate": 0.020,
                "robust_language_macro_error_rate": 0.100,
            },
            "by_scenario": [
                {"group": "en|clean", "error_rate": 0.020},
                {"group": "zh|clean", "error_rate": 0.020},
                {"group": "en|noise", "error_rate": 0.120},
                {"group": "zh|echo", "error_rate": 0.080},
            ],
        }

    def test_evaluate_gate_passed(self) -> None:
        pilot_metrics = {
            "overall": {
                "samples": 100,
                "inference_errors": 0,
                "empty_outputs": 0,
                "clean_language_macro_error_rate": 0.025,  # increase +0.005 <= 0.02
                "robust_language_macro_error_rate": 0.090,
            },
            "by_scenario": [
                {"group": "en|clean", "error_rate": 0.025},
                {"group": "zh|clean", "error_rate": 0.025},
                {"group": "en|noise", "error_rate": 0.110},  # improved (0.110 < 0.120)
                {"group": "zh|echo", "error_rate": 0.075},   # improved (0.075 < 0.080)
            ],
        }

        gate = verify_gate.evaluate_gate(
            base_metrics=self.mock_base_metrics,
            pilot_metrics=pilot_metrics,
            stage="sft_pilot",
        )

        self.assertEqual(gate["gate_status"], "PASSED")
        self.assertEqual(gate["checks"]["degraded_improvement"], "PASSED")
        self.assertEqual(gate["checks"]["clean_retention"], "PASSED")
        self.assertEqual(gate["checks"]["valid_output_rate"], "PASSED")
        self.assertEqual(gate["checks"]["failure_rate"], "PASSED")
        self.assertEqual(gate["metrics"]["degraded_scenario_improvements_count"], 2)
        self.assertEqual(gate["metrics"]["improved_scenarios"], ["en|noise", "zh|echo"])
        self.assertAlmostEqual(gate["metrics"]["clean_error_rate_increase"], 0.005)
        self.assertEqual(gate["metrics"]["valid_output_rate"], 1.0)

    def test_evaluate_gate_fails_on_clean_regression(self) -> None:
        pilot_metrics = {
            "overall": {
                "samples": 100,
                "inference_errors": 0,
                "empty_outputs": 0,
                "clean_language_macro_error_rate": 0.045,  # increase +0.025 > 0.02
                "robust_language_macro_error_rate": 0.080,
            },
            "by_scenario": [
                {"group": "en|clean", "error_rate": 0.045},
                {"group": "zh|clean", "error_rate": 0.045},
                {"group": "en|noise", "error_rate": 0.100},  # improved
            ],
        }

        gate = verify_gate.evaluate_gate(
            base_metrics=self.mock_base_metrics,
            pilot_metrics=pilot_metrics,
            stage="sft_pilot",
            max_clean_regression=0.02,
        )

        self.assertEqual(gate["gate_status"], "FAILED")
        self.assertEqual(gate["checks"]["clean_retention"], "FAILED")
        self.assertEqual(gate["checks"]["degraded_improvement"], "PASSED")

    def test_evaluate_gate_fails_without_degraded_improvement(self) -> None:
        pilot_metrics = {
            "overall": {
                "samples": 100,
                "inference_errors": 0,
                "empty_outputs": 0,
                "clean_language_macro_error_rate": 0.020,
                "robust_language_macro_error_rate": 0.120,
            },
            "by_scenario": [
                {"group": "en|clean", "error_rate": 0.020},
                {"group": "zh|clean", "error_rate": 0.020},
                {"group": "en|noise", "error_rate": 0.130},  # worse
                {"group": "zh|echo", "error_rate": 0.085},   # worse
            ],
        }

        gate = verify_gate.evaluate_gate(
            base_metrics=self.mock_base_metrics,
            pilot_metrics=pilot_metrics,
            stage="sft_pilot",
        )

        self.assertEqual(gate["gate_status"], "FAILED")
        self.assertEqual(gate["checks"]["degraded_improvement"], "FAILED")
        self.assertEqual(gate["metrics"]["degraded_scenario_improvements_count"], 0)

    def test_cli_execution_writes_gate_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_p = Path(tmpdir)
            base_p = tmp_p / "base_metrics.json"
            pilot_p = tmp_p / "pilot_metrics.json"
            manifest_p = tmp_p / "manifest.jsonl"
            out_p = tmp_p / "gate.json"

            base_p.write_text(json.dumps(self.mock_base_metrics), encoding="utf-8")
            manifest_p.write_text(json.dumps({"sample_id": "1"}) + "\n", encoding="utf-8")

            pilot_data = {
                "overall": {
                    "samples": 10,
                    "inference_errors": 0,
                    "empty_outputs": 0,
                    "clean_language_macro_error_rate": 0.021,
                    "robust_language_macro_error_rate": 0.080,
                },
                "by_scenario": [
                    {"group": "en|noise", "error_rate": 0.100},
                ],
            }
            pilot_p.write_text(json.dumps(pilot_data), encoding="utf-8")

            exit_code = verify_gate.main([
                "--base-metrics", str(base_p),
                "--pilot-metrics", str(pilot_p),
                "--manifest", str(manifest_p),
                "--output", str(out_p),
            ])

            self.assertEqual(exit_code, 0)
            self.assertTrue(out_p.is_file())

    def test_evaluate_gate_fails_on_robust_regression(self) -> None:
        pilot_metrics = {
            "overall": {
                "samples": 100,
                "inference_errors": 0,
                "empty_outputs": 0,
                "clean_language_macro_error_rate": 0.020,
                "robust_language_macro_error_rate": 0.137,  # increase +0.037 > 0.005
            },
            "by_scenario": [
                {"group": "en|clean", "error_rate": 0.020},
                {"group": "zh|clean", "error_rate": 0.020},
                {"group": "en|noise", "error_rate": 0.110},  # improved
            ],
        }

        gate = verify_gate.evaluate_gate(
            base_metrics=self.mock_base_metrics,
            pilot_metrics=pilot_metrics,
            stage="sft_pilot",
            max_robust_macro_regression=0.005,
        )

        self.assertEqual(gate["gate_status"], "FAILED")
        self.assertEqual(gate["checks"]["robust_retention"], "FAILED")
        self.assertEqual(gate["checks"]["degraded_improvement"], "PASSED")

    def test_evaluate_gate_fails_on_empty_outputs(self) -> None:
        pilot_metrics = {
            "overall": {
                "samples": 1000,
                "inference_errors": 0,
                "empty_outputs": 10,  # 10 / 1000 = 0.010 > 0.002
                "clean_language_macro_error_rate": 0.020,
                "robust_language_macro_error_rate": 0.090,
            },
            "by_scenario": [
                {"group": "en|clean", "error_rate": 0.020},
                {"group": "zh|clean", "error_rate": 0.020},
                {"group": "en|noise", "error_rate": 0.110},
            ],
        }

        gate = verify_gate.evaluate_gate(
            base_metrics=self.mock_base_metrics,
            pilot_metrics=pilot_metrics,
            stage="sft_pilot",
            max_empty_output_rate=0.002,
        )

        self.assertEqual(gate["gate_status"], "FAILED")
        self.assertEqual(gate["checks"]["empty_output_rate"], "FAILED")
        self.assertEqual(gate["checks"]["degraded_improvement"], "PASSED")

    def test_evaluate_gate_dpo_pilot_passed(self) -> None:
        sft_metrics = {
            "overall": {
                "samples": 100,
                "inference_errors": 0,
                "empty_outputs": 0,
                "clean_language_macro_error_rate": 0.022,  # +0.002 from Base
                "robust_language_macro_error_rate": 0.095,
            },
            "by_scenario": [
                {"group": "en|clean", "error_rate": 0.022},
                {"group": "zh|clean", "error_rate": 0.022},
                {"group": "en|noise", "error_rate": 0.115},
                {"group": "zh|echo", "error_rate": 0.075},
            ],
        }
        dpo_metrics = {
            "overall": {
                "samples": 100,
                "inference_errors": 0,
                "empty_outputs": 0,
                "clean_language_macro_error_rate": 0.023,  # +0.001 from SFT <= 0.02, +0.003 from Base <= 0.025
                "robust_language_macro_error_rate": 0.090,  # improved from SFT 0.095
            },
            "by_scenario": [
                {"group": "en|clean", "error_rate": 0.023},
                {"group": "zh|clean", "error_rate": 0.023},
                {"group": "en|noise", "error_rate": 0.110},  # improved relative to SFT 0.115
                {"group": "zh|echo", "error_rate": 0.070},   # improved relative to SFT 0.075
            ],
        }

        gate = verify_gate.evaluate_gate(
            base_metrics=self.mock_base_metrics,
            pilot_metrics=dpo_metrics,
            sft_metrics=sft_metrics,
            stage="dpo_pilot",
            preference_accuracy=0.64,
            min_preference_accuracy=0.55,
        )

        self.assertEqual(gate["gate_status"], "PASSED")
        self.assertEqual(gate["checks"]["preference_accuracy"], "PASSED")
        self.assertEqual(gate["checks"]["clean_cumulative_retention"], "PASSED")
        self.assertEqual(gate["checks"]["degraded_improvement"], "PASSED")
        self.assertEqual(gate["checks"]["clean_retention"], "PASSED")

    def test_evaluate_gate_dpo_fails_on_low_preference_accuracy(self) -> None:
        sft_metrics = {
            "overall": {
                "samples": 100,
                "inference_errors": 0,
                "empty_outputs": 0,
                "clean_language_macro_error_rate": 0.022,
                "robust_language_macro_error_rate": 0.095,
            },
            "by_scenario": [
                {"group": "en|noise", "error_rate": 0.115},
            ],
        }
        dpo_metrics = {
            "overall": {
                "samples": 100,
                "inference_errors": 0,
                "empty_outputs": 0,
                "clean_language_macro_error_rate": 0.022,
                "robust_language_macro_error_rate": 0.090,
            },
            "by_scenario": [
                {"group": "en|noise", "error_rate": 0.110},
            ],
        }

        gate = verify_gate.evaluate_gate(
            base_metrics=self.mock_base_metrics,
            pilot_metrics=dpo_metrics,
            sft_metrics=sft_metrics,
            stage="dpo_pilot",
            preference_accuracy=0.51,  # < 0.55
            min_preference_accuracy=0.55,
        )

        self.assertEqual(gate["gate_status"], "FAILED")
        self.assertEqual(gate["checks"]["preference_accuracy"], "FAILED")

    def test_evaluate_gate_dpo_fails_on_cumulative_clean_regression(self) -> None:
        sft_metrics = {
            "overall": {
                "samples": 100,
                "inference_errors": 0,
                "empty_outputs": 0,
                "clean_language_macro_error_rate": 0.038,
                "robust_language_macro_error_rate": 0.095,
            },
            "by_scenario": [
                {"group": "en|noise", "error_rate": 0.115},
            ],
        }
        dpo_metrics = {
            "overall": {
                "samples": 100,
                "inference_errors": 0,
                "empty_outputs": 0,
                "clean_language_macro_error_rate": 0.048,  # Base is 0.020, cumulative increase +0.028 > 0.025
                "robust_language_macro_error_rate": 0.090,
            },
            "by_scenario": [
                {"group": "en|noise", "error_rate": 0.110},
            ],
        }

        gate = verify_gate.evaluate_gate(
            base_metrics=self.mock_base_metrics,
            pilot_metrics=dpo_metrics,
            sft_metrics=sft_metrics,
            stage="dpo_pilot",
            preference_accuracy=0.60,
            max_cumulative_clean_regression=0.025,
        )

        self.assertEqual(gate["gate_status"], "FAILED")
        self.assertEqual(gate["checks"]["clean_cumulative_retention"], "FAILED")

    def test_evaluate_gate_dpo_fails_on_robust_regression_vs_sft(self) -> None:
        sft_metrics = {
            "overall": {
                "samples": 100,
                "inference_errors": 0,
                "empty_outputs": 0,
                "clean_language_macro_error_rate": 0.020,
                "robust_language_macro_error_rate": 0.095,
            },
            "by_scenario": [
                {"group": "en|noise", "error_rate": 0.115},
            ],
        }
        dpo_metrics = {
            "overall": {
                "samples": 100,
                "inference_errors": 0,
                "empty_outputs": 0,
                "clean_language_macro_error_rate": 0.020,
                "robust_language_macro_error_rate": 0.096,  # SFT is 0.095, regressed by +0.001 > 0.0
            },
            "by_scenario": [
                {"group": "en|noise", "error_rate": 0.110},
            ],
        }

        gate = verify_gate.evaluate_gate(
            base_metrics=self.mock_base_metrics,
            pilot_metrics=dpo_metrics,
            sft_metrics=sft_metrics,
            stage="dpo_pilot",
            preference_accuracy=0.65,
            # max_robust_macro_regression defaults to 0.0 for dpo_pilot
        )

        self.assertEqual(gate["gate_status"], "FAILED")
        self.assertEqual(gate["checks"]["robust_retention"], "FAILED")


    def test_evaluate_gate_dpo_fails_when_preference_accuracy_is_none(self) -> None:
        sft_metrics = {
            "overall": {
                "samples": 100,
                "inference_errors": 0,
                "empty_outputs": 0,
                "clean_language_macro_error_rate": 0.022,
                "robust_language_macro_error_rate": 0.095,
            },
            "by_scenario": [
                {"group": "en|noise", "error_rate": 0.115},
            ],
        }
        dpo_metrics = {
            "overall": {
                "samples": 100,
                "inference_errors": 0,
                "empty_outputs": 0,
                "clean_language_macro_error_rate": 0.022,
                "robust_language_macro_error_rate": 0.090,
            },
            "by_scenario": [
                {"group": "en|noise", "error_rate": 0.110},
            ],
        }

        gate = verify_gate.evaluate_gate(
            base_metrics=self.mock_base_metrics,
            pilot_metrics=dpo_metrics,
            sft_metrics=sft_metrics,
            stage="dpo_pilot",
            preference_accuracy=None,
        )

        self.assertEqual(gate["gate_status"], "FAILED")
        self.assertEqual(gate["checks"]["preference_accuracy"], "FAILED")
        self.assertIsNone(gate["metrics"]["preference_accuracy"])

    def test_cli_dpo_loss_log_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_p = Path(tmpdir)
            base_p = tmp_p / "base_metrics.json"
            pilot_p = tmp_p / "pilot_metrics.json"
            sft_p = tmp_p / "sft_metrics.json"
            loss_log_p = tmp_p / "loss_log.jsonl"
            out_p = tmp_p / "gate.json"

            base_p.write_text(json.dumps(self.mock_base_metrics), encoding="utf-8")
            sft_metrics = {
                "overall": {
                    "samples": 100,
                    "inference_errors": 0,
                    "empty_outputs": 0,
                    "clean_language_macro_error_rate": 0.020,
                    "robust_language_macro_error_rate": 0.095,
                },
                "by_scenario": [
                    {"group": "en|clean", "error_rate": 0.020},
                    {"group": "zh|clean", "error_rate": 0.020},
                    {"group": "en|noise", "error_rate": 0.115},
                ],
            }
            sft_p.write_text(json.dumps(sft_metrics), encoding="utf-8")

            pilot_metrics = {
                "overall": {
                    "samples": 100,
                    "inference_errors": 0,
                    "empty_outputs": 0,
                    "clean_language_macro_error_rate": 0.020,
                    "robust_language_macro_error_rate": 0.090,
                },
                "by_scenario": [
                    {"group": "en|clean", "error_rate": 0.020},
                    {"group": "zh|clean", "error_rate": 0.020},
                    {"group": "en|noise", "error_rate": 0.110},
                ],
            }
            pilot_p.write_text(json.dumps(pilot_metrics), encoding="utf-8")

            # Case 1: loss_log has accuracy 0.50 (< 0.55) -> FAILED
            loss_log_p.write_text(
                json.dumps({"step": 25, "val_preference_accuracy": 0.48}) + "\n" +
                json.dumps({"step": 50, "val_preference_accuracy": 0.50}) + "\n",
                encoding="utf-8"
            )

            exit_code = verify_gate.main([
                "--stage", "dpo_pilot",
                "--base-metrics", str(base_p),
                "--pilot-metrics", str(pilot_p),
                "--sft-metrics", str(sft_p),
                "--dpo-loss-log", str(loss_log_p),
                "--output", str(out_p),
            ])
            self.assertEqual(exit_code, 1)
            gate_data = json.loads(out_p.read_text(encoding="utf-8"))
            self.assertEqual(gate_data["gate_status"], "FAILED")
            self.assertEqual(gate_data["checks"]["preference_accuracy"], "FAILED")
            self.assertEqual(gate_data["metrics"]["preference_accuracy"], 0.50)

            # Case 2: loss_log has accuracy 0.60 (>= 0.55) -> PASSED
            loss_log_p.write_text(
                json.dumps({"step": 50, "val_preference_accuracy": 0.60}) + "\n",
                encoding="utf-8"
            )
            exit_code = verify_gate.main([
                "--stage", "dpo_pilot",
                "--base-metrics", str(base_p),
                "--pilot-metrics", str(pilot_p),
                "--sft-metrics", str(sft_p),
                "--dpo-loss-log", str(loss_log_p),
                "--output", str(out_p),
            ])
            self.assertEqual(exit_code, 0)
            gate_data = json.loads(out_p.read_text(encoding="utf-8"))
            self.assertEqual(gate_data["gate_status"], "PASSED")
            self.assertEqual(gate_data["checks"]["preference_accuracy"], "PASSED")
            self.assertEqual(gate_data["metrics"]["preference_accuracy"], 0.60)

            # Case 3: multi-step loss log with --dpo-step specified
            loss_log_p.write_text(
                json.dumps({"step": 50, "val_preference_accuracy": 0.60}) + "\n" +
                json.dumps({"step": 100, "val_preference_accuracy": 0.45}) + "\n",
                encoding="utf-8"
            )
            exit_code = verify_gate.main([
                "--stage", "dpo_pilot",
                "--base-metrics", str(base_p),
                "--pilot-metrics", str(pilot_p),
                "--sft-metrics", str(sft_p),
                "--dpo-loss-log", str(loss_log_p),
                "--dpo-step", "50",
                "--output", str(out_p),
            ])
            self.assertEqual(exit_code, 0)
            gate_data = json.loads(out_p.read_text(encoding="utf-8"))
            self.assertEqual(gate_data["metrics"]["preference_accuracy"], 0.60)

    def test_evaluate_gate_rl_pilot_passed(self) -> None:
        dpo_metrics = {
            "overall": {
                "samples": 100,
                "inference_errors": 0,
                "empty_outputs": 0,
                "clean_language_macro_error_rate": 0.021,
                "robust_language_macro_error_rate": 0.090,
            },
            "by_scenario": [
                {"group": "en|clean", "error_rate": 0.021},
                {"group": "zh|clean", "error_rate": 0.021},
                {"group": "en|noise", "error_rate": 0.110},
                {"group": "zh|echo", "error_rate": 0.070},
            ],
        }
        rl_metrics = {
            "overall": {
                "samples": 100,
                "inference_errors": 0,
                "empty_outputs": 0,
                "clean_language_macro_error_rate": 0.021,
                "robust_language_macro_error_rate": 0.088,
            },
            "by_scenario": [
                {"group": "en|clean", "error_rate": 0.021},
                {"group": "zh|clean", "error_rate": 0.021},
                {"group": "en|noise", "error_rate": 0.105},
                {"group": "zh|echo", "error_rate": 0.070},
            ],
        }

        gate = verify_gate.evaluate_gate(
            base_metrics=self.mock_base_metrics,
            pilot_metrics=rl_metrics,
            dpo_metrics=dpo_metrics,
            stage="rl_pilot",
            reward_improvement=0.06,
        )

        self.assertEqual(gate["gate_status"], "PASSED")
        self.assertEqual(gate["checks"]["degraded_improvement"], "PASSED")
        self.assertEqual(gate["checks"]["clean_retention"], "PASSED")
        self.assertEqual(gate["checks"]["clean_cumulative_retention"], "PASSED")
        self.assertEqual(gate["checks"]["robust_retention"], "PASSED")
        self.assertEqual(gate["checks"]["held_out_reward"], "PASSED")
        self.assertEqual(gate["metrics"]["degraded_scenario_improvements_count"], 1)
        self.assertEqual(gate["metrics"]["improved_scenarios"], ["en|noise"])

    def test_evaluate_gate_rl_fails_on_low_reward(self) -> None:
        dpo_metrics = {
            "overall": {
                "samples": 100,
                "inference_errors": 0,
                "empty_outputs": 0,
                "clean_language_macro_error_rate": 0.020,
                "robust_language_macro_error_rate": 0.090,
            },
            "by_scenario": [
                {"group": "en|noise", "error_rate": 0.110},
            ],
        }
        rl_metrics = {
            "overall": {
                "samples": 100,
                "inference_errors": 0,
                "empty_outputs": 0,
                "clean_language_macro_error_rate": 0.020,
                "robust_language_macro_error_rate": 0.085,
            },
            "by_scenario": [
                {"group": "en|noise", "error_rate": 0.105},
            ],
        }

        gate = verify_gate.evaluate_gate(
            base_metrics=self.mock_base_metrics,
            pilot_metrics=rl_metrics,
            dpo_metrics=dpo_metrics,
            stage="rl_pilot",
            reward_improvement=0.001,  # < 0.002
        )

        self.assertEqual(gate["gate_status"], "FAILED")
        self.assertEqual(gate["checks"]["held_out_reward"], "FAILED")

    def test_cli_rl_loss_log_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_p = Path(tmpdir)
            base_p = tmp_p / "base_metrics.json"
            pilot_p = tmp_p / "pilot_metrics.json"
            dpo_p = tmp_p / "dpo_metrics.json"
            loss_log_p = tmp_p / "rl_loss_log.jsonl"
            out_p = tmp_p / "gate.json"

            base_p.write_text(json.dumps(self.mock_base_metrics), encoding="utf-8")
            dpo_metrics = {
                "overall": {
                    "samples": 100,
                    "inference_errors": 0,
                    "empty_outputs": 0,
                    "clean_language_macro_error_rate": 0.020,
                    "robust_language_macro_error_rate": 0.090,
                },
                "by_scenario": [
                    {"group": "en|clean", "error_rate": 0.020},
                    {"group": "zh|clean", "error_rate": 0.020},
                    {"group": "en|noise", "error_rate": 0.110},
                ],
            }
            dpo_p.write_text(json.dumps(dpo_metrics), encoding="utf-8")

            pilot_metrics = {
                "overall": {
                    "samples": 100,
                    "inference_errors": 0,
                    "empty_outputs": 0,
                    "clean_language_macro_error_rate": 0.020,
                    "robust_language_macro_error_rate": 0.085,
                },
                "by_scenario": [
                    {"group": "en|clean", "error_rate": 0.020},
                    {"group": "zh|clean", "error_rate": 0.020},
                    {"group": "en|noise", "error_rate": 0.100},
                ],
            }
            pilot_p.write_text(json.dumps(pilot_metrics), encoding="utf-8")

            loss_log_p.write_text(
                json.dumps({"global_step": 0, "val_mean_reward": 0.7802, "val_eval_scope": "Full Held-out"}) + "\n" +
                json.dumps({"global_step": 1, "zero_variance_ratio": 0.15}) + "\n" +
                json.dumps({"global_step": 60, "val_mean_reward": 0.8410, "val_eval_scope": "Full Held-out"}) + "\n",
                encoding="utf-8"
            )

            exit_code = verify_gate.main([
                "--stage", "rl_pilot",
                "--base-metrics", str(base_p),
                "--pilot-metrics", str(pilot_p),
                "--dpo-metrics", str(dpo_p),
                "--rl-loss-log", str(loss_log_p),
                "--output", str(out_p),
            ])
            self.assertEqual(exit_code, 0)
            gate_data = json.loads(out_p.read_text(encoding="utf-8"))
            self.assertEqual(gate_data["gate_status"], "PASSED")
            self.assertEqual(gate_data["checks"]["held_out_reward"], "PASSED")
            self.assertEqual(gate_data["checks"]["zero_variance"], "PASSED")
            self.assertAlmostEqual(gate_data["metrics"]["held_out_reward_improvement"], 0.0608)
            self.assertAlmostEqual(gate_data["metrics"]["zero_variance_ratio"], 0.15)

    def test_cli_rl_loss_log_fails_when_scope_is_sub_or_step_0_missing(self) -> None:
        """Verify RL gate fails when Step 0 is missing or evaluation uses Sub-25 scope."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            base_p = tmp_p / "base_metrics.json"
            pilot_p = tmp_p / "pilot_metrics.json"
            dpo_p = tmp_p / "dpo_metrics.json"
            loss_log_missing_step0 = tmp_p / "loss_log_no_step0.jsonl"
            loss_log_sub25 = tmp_p / "loss_log_sub25.jsonl"
            out_p = tmp_p / "gate.json"

            base_metrics = {
                "overall": {
                    "clean_language_macro_error_rate": 0.020,
                    "robust_language_macro_error_rate": 0.150,
                    "samples": 100,
                    "inference_errors": 0,
                    "empty_outputs": 0,
                },
                "by_scenario": [{"group": "en|clean", "error_rate": 0.020}, {"group": "zh|clean", "error_rate": 0.020}, {"group": "en|noise", "error_rate": 0.150}],
            }
            base_p.write_text(json.dumps(base_metrics), encoding="utf-8")
            dpo_p.write_text(json.dumps(base_metrics), encoding="utf-8")

            pilot_metrics = {
                "overall": {
                    "clean_language_macro_error_rate": 0.020,
                    "robust_language_macro_error_rate": 0.100,
                    "samples": 100,
                    "inference_errors": 0,
                    "empty_outputs": 0,
                },
                "by_scenario": [{"group": "en|clean", "error_rate": 0.020}, {"group": "zh|clean", "error_rate": 0.020}, {"group": "en|noise", "error_rate": 0.100}],
            }
            pilot_p.write_text(json.dumps(pilot_metrics), encoding="utf-8")

            # Missing Step 0
            loss_log_missing_step0.write_text(
                json.dumps({"global_step": 10, "val_mean_reward": 0.7000, "val_eval_scope": "Full Held-out"}) + "\n" +
                json.dumps({"global_step": 60, "val_mean_reward": 0.8500, "val_eval_scope": "Full Held-out"}) + "\n",
                encoding="utf-8"
            )
            exit_code = verify_gate.main([
                "--stage", "rl_pilot",
                "--base-metrics", str(base_p),
                "--pilot-metrics", str(pilot_p),
                "--dpo-metrics", str(dpo_p),
                "--rl-loss-log", str(loss_log_missing_step0),
                "--output", str(out_p),
            ])
            self.assertEqual(exit_code, 1)
            gate_data = json.loads(out_p.read_text(encoding="utf-8"))
            self.assertEqual(gate_data["gate_status"], "FAILED")
            self.assertEqual(gate_data["checks"]["held_out_reward"], "FAILED")

            # Using Sub-25 scope instead of Full Held-out
            loss_log_sub25.write_text(
                json.dumps({"global_step": 0, "val_mean_reward": 0.7000, "val_eval_scope": "Sub-25"}) + "\n" +
                json.dumps({"global_step": 60, "val_mean_reward": 0.8500, "val_eval_scope": "Sub-25"}) + "\n",
                encoding="utf-8"
            )
            exit_code2 = verify_gate.main([
                "--stage", "rl_pilot",
                "--base-metrics", str(base_p),
                "--pilot-metrics", str(pilot_p),
                "--dpo-metrics", str(dpo_p),
                "--rl-loss-log", str(loss_log_sub25),
                "--output", str(out_p),
            ])
            self.assertEqual(exit_code2, 1)
            gate_data2 = json.loads(out_p.read_text(encoding="utf-8"))
            self.assertEqual(gate_data2["gate_status"], "FAILED")
            self.assertEqual(gate_data2["checks"]["held_out_reward"], "FAILED")

    def test_cli_rl_loss_log_fails_when_zero_variance_exceeds_threshold(self) -> None:
        """Verify RL gate fails when zero_variance_ratio exceeds 30% contract threshold."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            base_p = tmp_p / "base_metrics.json"
            pilot_p = tmp_p / "pilot_metrics.json"
            dpo_p = tmp_p / "dpo_metrics.json"
            loss_log_p = tmp_p / "loss_log_high_zero_var.jsonl"
            out_p = tmp_p / "gate.json"

            base_metrics = {
                "overall": {
                    "clean_language_macro_error_rate": 0.020,
                    "robust_language_macro_error_rate": 0.150,
                    "samples": 100,
                    "inference_errors": 0,
                    "empty_outputs": 0,
                },
                "by_scenario": [{"group": "en|clean", "error_rate": 0.020}, {"group": "zh|clean", "error_rate": 0.020}, {"group": "en|noise", "error_rate": 0.150}],
            }
            base_p.write_text(json.dumps(base_metrics), encoding="utf-8")
            dpo_p.write_text(json.dumps(base_metrics), encoding="utf-8")

            pilot_metrics = {
                "overall": {
                    "clean_language_macro_error_rate": 0.020,
                    "robust_language_macro_error_rate": 0.100,
                    "samples": 100,
                    "inference_errors": 0,
                    "empty_outputs": 0,
                },
                "by_scenario": [{"group": "en|clean", "error_rate": 0.020}, {"group": "zh|clean", "error_rate": 0.020}, {"group": "en|noise", "error_rate": 0.100}],
            }
            pilot_p.write_text(json.dumps(pilot_metrics), encoding="utf-8")

            loss_log_p.write_text(
                json.dumps({"global_step": 0, "val_mean_reward": 0.7000, "val_eval_scope": "Full Held-out"}) + "\n" +
                json.dumps({"global_step": 1, "zero_variance_ratio": 0.80}) + "\n" +
                json.dumps({"global_step": 2, "zero_variance_ratio": 0.90}) + "\n" +
                json.dumps({"global_step": 60, "val_mean_reward": 0.8500, "val_eval_scope": "Full Held-out"}) + "\n",
                encoding="utf-8"
            )
            exit_code = verify_gate.main([
                "--stage", "rl_pilot",
                "--base-metrics", str(base_p),
                "--pilot-metrics", str(pilot_p),
                "--dpo-metrics", str(dpo_p),
                "--rl-loss-log", str(loss_log_p),
                "--output", str(out_p),
            ])
            self.assertEqual(exit_code, 1)
            gate_data = json.loads(out_p.read_text(encoding="utf-8"))
            self.assertEqual(gate_data["gate_status"], "FAILED")
            self.assertEqual(gate_data["checks"]["held_out_reward"], "PASSED")
            self.assertEqual(gate_data["checks"]["zero_variance"], "FAILED")
            self.assertAlmostEqual(gate_data["metrics"]["zero_variance_ratio"], 0.85)


if __name__ == "__main__":
    unittest.main()
