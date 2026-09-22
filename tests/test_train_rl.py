#!/usr/bin/env python3
"""Unit tests for train/train_rl.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from train.train_rl import (
    RLAudioDataset,
    audit_rollouts,
    build_parser,
    compute_group_advantages,
    compute_sample_error_rate,
    compute_sequence_reward,
    merge_and_audit_rollouts,
)


class TestRLRewardAndAdvantage(unittest.TestCase):
    """Test reward calculation and advantage normalization."""

    def test_exact_match_reward(self) -> None:
        text = "this is a clean test utterance"
        reward, components, err = compute_sequence_reward(text, text, "en")
        self.assertAlmostEqual(err, 0.0)
        self.assertAlmostEqual(reward, 1.0)
        self.assertEqual(components["asr"], 1.0)
        self.assertEqual(components["empty"], 0.0)
        self.assertEqual(components["repeat"], 0.0)
        self.assertEqual(components["too_long"], 0.0)
        self.assertEqual(components["hallucination"], 0.0)

    def test_empty_prediction_penalty(self) -> None:
        gold = "hello world"
        reward, components, err = compute_sequence_reward("", gold, "en")
        self.assertAlmostEqual(err, 1.0)
        self.assertEqual(components["empty"], -0.25)
        self.assertEqual(components["asr"], 0.0)
        self.assertAlmostEqual(reward, -0.25)

    def test_repetition_penalty(self) -> None:
        gold = "the weather is very nice today"
        pred = "the weather weather weather is nice"
        reward, components, err = compute_sequence_reward(pred, gold, "en")
        self.assertEqual(components["repeat"], -0.25)

    def test_too_long_penalty(self) -> None:
        gold = "short phrase"
        pred = "short phrase with lots of extra unnecessary padded words that make it way too long"
        reward, components, err = compute_sequence_reward(pred, gold, "en")
        self.assertEqual(components["too_long"], -0.15)

    def test_reward_clipping(self) -> None:
        gold = "test phrase"
        pred = "repeat repeat repeat repeat repeat and lots of very long words that hallucinate completely"
        reward, components, err = compute_sequence_reward(pred, gold, "en")
        self.assertGreaterEqual(reward, -1.0)
        self.assertLessEqual(reward, 1.0)

    def test_chinese_cer_reward(self) -> None:
        gold = "今天天气很好"
        pred = "今天天气很好"
        reward, components, err = compute_sequence_reward(pred, gold, "zh")
        self.assertAlmostEqual(err, 0.0)
        self.assertAlmostEqual(reward, 1.0)

        pred_err = "今天天气很差"
        reward2, components2, err2 = compute_sequence_reward(pred_err, gold, "zh")
        self.assertAlmostEqual(err2, 1.0 / 6.0, places=5)
        self.assertAlmostEqual(reward2, 1.0 - 1.0 / 6.0, places=4)

    def test_group_advantages_zero_variance(self) -> None:
        rewards = [0.8, 0.8, 0.8, 0.8]
        advs, is_zero_var = compute_group_advantages(rewards)
        self.assertTrue(is_zero_var)
        self.assertEqual(advs, [0.0, 0.0, 0.0, 0.0])

    def test_group_advantages_normalized(self) -> None:
        rewards = [1.0, 0.5, 0.5, 0.0]
        advs, is_zero_var = compute_group_advantages(rewards)
        self.assertFalse(is_zero_var)
        self.assertAlmostEqual(sum(advs), 0.0, places=5)
        self.assertGreater(advs[0], 0.0)
        self.assertLess(advs[3], 0.0)

class TestRLDatasetAndParser(unittest.TestCase):
    """Test dataset loader and CLI argument parser."""

    def test_rl_dataset_loading(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write(json.dumps({"sample_id": "s1", "audio": "/path/1.wav", "text": "one", "language": "en"}) + "\n")
            f.write(json.dumps({"sample_id": "s2", "audio": "/path/2.wav", "text": "two", "language": "zh"}) + "\n")
            p = Path(f.name)

        try:
            ds = RLAudioDataset(p)
            self.assertEqual(len(ds), 2)
            self.assertEqual(ds[0]["sample_id"], "s1")
            self.assertEqual(ds[1]["sample_id"], "s2")
        finally:
            p.unlink(missing_ok=True)

    def test_cli_parser(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--manifest", "data.jsonl",
            "--config", "conf.yaml",
            "--output-dir", "out_dir",
            "--max-steps", "50",
            "--temperature", "0.85",
            "--top-p", "0.92",
            "--top-k", "50",
            "--zero-variance-thresh", "0.30",
            "--single-gpu",
        ])
        self.assertEqual(args.manifest, "data.jsonl")
        self.assertEqual(args.config, "conf.yaml")
        self.assertEqual(args.output_dir, "out_dir")
        self.assertEqual(args.max_steps, 50)
        self.assertAlmostEqual(args.temperature, 0.85)
        self.assertAlmostEqual(args.top_p, 0.92)
        self.assertEqual(args.top_k, 50)
        self.assertAlmostEqual(args.zero_variance_thresh, 0.30)
        self.assertTrue(args.single_gpu)


class TestRolloutAudit(unittest.TestCase):
    """Test rollout log auditing and multi-rank aggregation."""

    def test_audit_rollouts_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            rollouts_path = tmp_p / "rollouts.jsonl"
            lines = []
            for g_i in range(4):
                lines.append(json.dumps({
                    "sample_id": "s1",
                    "condition_group": "clean",
                    "rank": 0,
                    "group_id": "s1:0:0:0",
                    "group_size": 4,
                    "rollout_rank": g_i,
                    "policy_checkpoint": "step_0",
                    "rollout_seed": 100 + g_i,
                    "prediction": f"pred_{g_i}",
                    "language": "en",
                    "reference_error_rate": 0.0,
                    "reward_components": {"asr": 1.0},
                    "reward": 1.0,
                    "kl_to_reference": 0.01,
                    "advantage": 0.0,
                }))
            rollouts_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            res = audit_rollouts(rollouts_path, world_size=1)
            self.assertEqual(res["status"], "PASSED")
            self.assertEqual(res["total_rows"], 4)
            self.assertEqual(res["num_groups"], 1)
            self.assertEqual(res["ranks_represented"], [0])

    def test_audit_rollouts_invalid_group_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            rollouts_path = tmp_p / "rollouts.jsonl"
            # Only 3 rollouts in group instead of 4
            lines = [json.dumps({
                "sample_id": "s1",
                "condition_group": "clean",
                "rank": 0,
                "group_id": "s1:0:0:0",
                "group_size": 4,
                "rollout_rank": i,
                "policy_checkpoint": "step_0",
                "rollout_seed": 100 + i,
                "prediction": f"pred_{i}",
                "language": "en",
                "reference_error_rate": 0.0,
                "reward_components": {"asr": 1.0},
                "reward": 1.0,
                "kl_to_reference": 0.01,
                "advantage": 0.0,
            }) for i in range(3)]
            rollouts_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                audit_rollouts(rollouts_path, world_size=1)

    def test_merge_and_audit_rollouts_multi_rank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            for r in range(4):
                r_file = tmp_p / f"rollouts_rank_{r}.jsonl"
                lines = [json.dumps({
                    "sample_id": f"s_r{r}",
                    "condition_group": "degraded",
                    "rank": r,
                    "group_id": f"s_r{r}:0:{r}:0",
                    "group_size": 4,
                    "rollout_rank": g_i,
                    "policy_checkpoint": "step_0",
                    "rollout_seed": 1000 * r + g_i,
                    "prediction": f"pred_r{r}_{g_i}",
                    "language": "en",
                    "reference_error_rate": 0.1,
                    "reward_components": {"asr": 0.9},
                    "reward": 0.9,
                    "kl_to_reference": 0.02,
                    "advantage": 0.0,
                }) for g_i in range(4)]
                r_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

            res = merge_and_audit_rollouts(tmp_p, world_size=4)
            self.assertEqual(res["status"], "PASSED")
            self.assertEqual(res["total_rows"], 16)
            self.assertEqual(res["num_groups"], 4)
            self.assertEqual(res["ranks_represented"], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
