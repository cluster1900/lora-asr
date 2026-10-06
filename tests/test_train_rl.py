#!/usr/bin/env python3
"""Unit tests for train/train_rl.py."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from train.train_rl import (
    RLAudioDataset,
    assert_resume_checkpoint,
    audit_rollouts,
    build_epoch_sample_indices,
    build_parser,
    compute_anchored_advantages,
    compute_group_advantages,
    compute_grpo_group_loss,
    compute_sample_error_rate,
    compute_sequence_reward,
    get_environment_info,
    get_git_commit,
    is_terminal_train_status,
    load_run_dose,
    merge_and_audit_rollouts,
    resolve_sample_strategy,
    scale_futility_streak,
    select_resume_step,
    truncate_resume_artifacts,
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

    def test_anchored_advantages_ignore_samples_that_do_not_beat_greedy(self) -> None:
        # Index 0 is greedy. The samples are worse or only slightly better.
        rewards = [0.90, 0.80, 0.91, 0.70]
        advantages, status = compute_anchored_advantages(rewards, anchor_index=0, min_improvement=0.02)
        self.assertEqual(status, "no_improvement")
        self.assertEqual(advantages, [0.0, 0.0, 0.0, 0.0])

    def test_anchored_advantages_scale_real_improvements(self) -> None:
        rewards = [0.80, 0.70, 0.90, 0.84]
        advantages, status = compute_anchored_advantages(rewards, anchor_index=0, min_improvement=0.02)
        self.assertEqual(status, "update")
        self.assertEqual(advantages[0], 0.0)
        self.assertEqual(advantages[1], 0.0)
        self.assertAlmostEqual(advantages[2], 1.0)
        self.assertAlmostEqual(advantages[3], 0.4)

    def test_anchored_advantages_mark_identical_rewards(self) -> None:
        advantages, status = compute_anchored_advantages([0.9, 0.9, 0.9, 0.9], min_improvement=0.02)
        self.assertEqual(status, "identical")
        self.assertEqual(advantages, [0.0, 0.0, 0.0, 0.0])

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
            "--sample-strategy", "balanced",
            "--export-merged-on-finish",
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
        self.assertEqual(args.sample_strategy, "balanced")
        self.assertTrue(args.export_merged_on_finish)


class TestEpochSampleIndices(unittest.TestCase):
    """Test deterministic balanced and standard sampling strategies."""

    def setUp(self) -> None:
        # Create a mock dataset with 10 degraded, 5 clean EN, 5 clean ZH
        samples = []
        for i in range(10):
            samples.append({
                "sample_id": f"deg_{i}",
                "condition_group": "degraded",
                "scenario": "noise",
                "language": "en" if i % 2 == 0 else "zh",
            })
        for i in range(5):
            samples.append({
                "sample_id": f"clean_en_{i}",
                "condition_group": "clean",
                "scenario": "clean",
                "language": "en",
            })
        for i in range(5):
            samples.append({
                "sample_id": f"clean_zh_{i}",
                "condition_group": "clean",
                "scenario": "clean",
                "language": "zh",
            })
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
            self.manifest_path = Path(f.name)
        self.dataset = RLAudioDataset(self.manifest_path)

    def tearDown(self) -> None:
        self.manifest_path.unlink(missing_ok=True)

    def test_balanced_sampling(self) -> None:
        # Balanced: 10 degraded + 5 EN clean + 5 ZH clean = 20 samples
        idx_ep0 = build_epoch_sample_indices(self.dataset, strategy="balanced", epoch=0, seed=42)
        self.assertEqual(len(idx_ep0), 20)

        # Degraded samples should be 50% of the epoch
        deg_count = sum(1 for idx in idx_ep0 if self.dataset[idx]["condition_group"] == "degraded")
        en_clean_count = sum(
            1 for idx in idx_ep0
            if self.dataset[idx]["condition_group"] == "clean" and self.dataset[idx]["language"] == "en"
        )
        zh_clean_count = sum(
            1 for idx in idx_ep0
            if self.dataset[idx]["condition_group"] == "clean" and self.dataset[idx]["language"] == "zh"
        )
        self.assertEqual(deg_count, 10)
        self.assertEqual(en_clean_count, 5)
        self.assertEqual(zh_clean_count, 5)

        # Deterministic with same seed & epoch
        idx_ep0_repeat = build_epoch_sample_indices(self.dataset, strategy="balanced", epoch=0, seed=42)
        self.assertEqual(idx_ep0, idx_ep0_repeat)

        # Different permutation with epoch=1
        idx_ep1 = build_epoch_sample_indices(self.dataset, strategy="balanced", epoch=1, seed=42)
        self.assertEqual(len(idx_ep1), 20)
        self.assertNotEqual(idx_ep0, idx_ep1)

    def test_standard_sampling(self) -> None:
        idx_ep0 = build_epoch_sample_indices(self.dataset, strategy="standard", epoch=0, seed=42)
        self.assertEqual(len(idx_ep0), len(self.dataset))
        self.assertEqual(sorted(idx_ep0), list(range(len(self.dataset))))

    def test_unknown_sampling_strategy_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported sample strategy"):
            resolve_sample_strategy(None, "degraded_balanecd")
        with self.assertRaisesRegex(ValueError, "unsupported sample strategy"):
            build_epoch_sample_indices(self.dataset, strategy="degraded_balanecd")


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
                    "decode_mode": "greedy" if g_i == 0 else "sample",
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
                    "decode_mode": "greedy" if g_i == 0 else "sample",
                }) for g_i in range(4)]
                r_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

            res = merge_and_audit_rollouts(tmp_p, world_size=4)
            self.assertEqual(res["status"], "PASSED")
            self.assertEqual(res["total_rows"], 16)
            self.assertEqual(res["num_groups"], 4)
            self.assertEqual(res["ranks_represented"], [0, 1, 2, 3])

    def test_audit_rollouts_group_without_greedy_row_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            rollouts_path = Path(tmp_dir) / "rollouts.jsonl"
            lines = [json.dumps({
                "sample_id": "s1",
                "condition_group": "degraded",
                "rank": 0,
                "group_id": "s1:0:0:0",
                "group_size": 4,
                "rollout_rank": g_i,
                "policy_checkpoint": "step_0",
                "rollout_seed": 100 + g_i,
                "prediction": f"pred_{g_i}",
                "language": "en",
                "reference_error_rate": 0.1,
                "reward_components": {"asr": 0.9},
                "reward": 0.9,
                "kl_to_reference": 0.01,
                "advantage": 0.0,
                "decode_mode": "sample",
            }) for g_i in range(4)]
            rollouts_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "greedy rows"):
                audit_rollouts(rollouts_path, world_size=1)


class TestGitCommitAndEnvironment(unittest.TestCase):
    """Test git commit resolution and environment metadata."""

    def test_get_git_commit_env(self) -> None:
        import os
        old = os.environ.get("GIT_COMMIT")
        try:
            os.environ["GIT_COMMIT"] = "test_sha_12345"
            self.assertEqual(get_git_commit(), "test_sha_12345")
        finally:
            if old is None:
                os.environ.pop("GIT_COMMIT", None)
            else:
                os.environ["GIT_COMMIT"] = old

    def test_get_git_commit_file(self) -> None:
        import os
        old = os.environ.pop("GIT_COMMIT", None)
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_p = Path(tmp_dir)
                commit_file = tmp_p / ".git_commit"
                commit_file.write_text("file_sha_abcdef\n", encoding="utf-8")
                self.assertEqual(get_git_commit(repo_dir=tmp_p), "file_sha_abcdef")
        finally:
            if old is not None:
                os.environ["GIT_COMMIT"] = old

    def test_get_environment_info_contains_git_commit(self) -> None:
        info = get_environment_info()
        self.assertIn("git_commit", info)
        self.assertIsInstance(info["git_commit"], str)
        self.assertTrue(len(info["git_commit"]) > 0)


class TestSequenceGrpoLoss(unittest.TestCase):
    """Sequence-level GRPO loss must not divide the policy gradient by length."""

    def test_sequence_sum_scales_with_response_length(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        policy = torch.tensor([[-0.25, -0.25, -0.25, -0.25]], requires_grad=True)
        ref = policy.detach().clone()
        mask = torch.ones_like(policy, dtype=torch.bool)
        summed = compute_grpo_group_loss(
            policy, ref, mask, [1.0], beta=0.0, zero_variance=False, reduction="sequence_sum"
        )
        averaged = compute_grpo_group_loss(
            policy, ref, mask, [1.0], beta=0.0, zero_variance=False, reduction="token_mean"
        )
        # -A * sum(log p) = -1 * (-1.0) = 1.0; the token mean is 1.0 / 4.
        self.assertAlmostEqual(summed["group_loss"].item(), 1.0, places=5)
        self.assertAlmostEqual(averaged["group_loss"].item(), 0.25, places=5)
        self.assertAlmostEqual(summed["raw_kl"].item(), 0.0, places=6)

    def test_positive_advantage_raises_chosen_logprob(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        policy = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]], requires_grad=True)
        ref = policy.detach().clone()
        mask = torch.ones_like(policy, dtype=torch.bool)
        loss = compute_grpo_group_loss(
            policy, ref, mask, [1.0, -1.0], beta=0.0, zero_variance=False, reduction="sequence_sum"
        )
        loss["group_loss"].backward()
        self.assertIsNotNone(policy.grad)
        self.assertTrue(torch.all(policy.grad[0] < 0))
        self.assertTrue(torch.all(policy.grad[1] > 0))

    def test_zero_variance_group_has_zero_gradient(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        policy = torch.tensor([[-0.4, -0.2], [-0.4, -0.2]], requires_grad=True)
        ref = policy.detach().clone()
        mask = torch.ones_like(policy, dtype=torch.bool)
        loss = compute_grpo_group_loss(
            policy, ref, mask, [0.0, 0.0], beta=0.04, zero_variance=True, reduction="sequence_sum"
        )
        self.assertEqual(loss["group_loss"].item(), 0.0)
        self.assertEqual(loss["policy_loss"].item(), 0.0)
        loss["group_loss"].backward()
        self.assertIsNotNone(policy.grad)
        self.assertTrue(torch.all(policy.grad == 0))


class TestZeroVarianceLossComputation(unittest.TestCase):
    """Test that zero-variance groups produce zero gradient without breaking backprop."""

    def test_zero_variance_loss_neutralization(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch not available")

        policy_logps = torch.tensor([[-0.5, -0.2], [-0.5, -0.2]], requires_grad=True)
        is_zero_var = True
        if is_zero_var:
            group_loss = 0.0 * policy_logps.sum()
        else:
            group_loss = policy_logps.mean()

        self.assertEqual(group_loss.item(), 0.0)
        group_loss.backward()
        self.assertIsNotNone(policy_logps.grad)
        self.assertTrue(torch.all(policy_logps.grad == 0.0))


def _write_jsonl(path: Path, rows: list) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _rollout_group(group_id: str, policy_step: int | None) -> list:
    rows = []
    for index, mode in ((0, "greedy"), (1, "sample")):
        row = {
            "sample_id": "utt",
            "condition_group": "degraded",
            "rank": 0,
            "group_id": group_id,
            "group_size": 2,
            "rollout_rank": index,
            "decode_mode": mode,
            "rollout_seed": 7 + index,
            "prediction": "hello" if mode == "greedy" else "hello there",
            "language": "en",
            "reference_error_rate": 0.0,
            "reward_components": {"asr": 1.0},
            "reward": 1.0 if mode == "greedy" else 0.8,
            "kl_to_reference": 0.0,
            "advantage": 0.0,
        }
        if policy_step is not None:
            row["policy_checkpoint"] = f"step_{policy_step}"
        rows.append(row)
    return rows


def _complete_checkpoint(root: Path, step: int, *, world_size: int = 1, skip: str = "") -> Path:
    checkpoint = root / "checkpoints" / f"step_{step}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    if skip != "adapter":
        (checkpoint / "adapter").mkdir(exist_ok=True)
    for name in ("optimizer.pt", "scheduler.pt", "training_state.json"):
        if name == skip:
            continue
        (checkpoint / name).write_text("ok", encoding="utf-8")
    for rank in range(world_size):
        name = f"rng_state_rank_{rank}.pt"
        if name == skip:
            continue
        (checkpoint / name).write_bytes(b"rng")
    return checkpoint


def _write_pipeline(root: Path, step: int, status: str) -> None:
    (root / "pipeline_state.json").write_text(
        json.dumps({"global_step": step, "status": status}),
        encoding="utf-8",
    )


class TestResumeArtifactTruncate(unittest.TestCase):
    """Uncommitted optimizer steps must not survive a chunk resume."""

    def test_truncate_drops_uncommitted_rows_and_dose_is_not_doubled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loss_rows = [
                {
                    "global_step": 0,
                    "val_mean_reward": 0.8773,
                    "val_eval_scope": "Full Held-out",
                    "val_decode": "greedy",
                },
                {"global_step": 1, "reward_mass_in_step": 1.0, "winners_in_step": 1},
                {"global_step": 2, "reward_mass_in_step": 5.0, "winners_in_step": 2},
            ]
            committed = _rollout_group("utt:0:0:1", 0)
            replayed = _rollout_group("utt:1:0:1", 1)
            rank_rows = committed + replayed + replayed
            _write_jsonl(root / "loss_log.jsonl", loss_rows)
            _write_jsonl(root / "rollouts_rank_0.jsonl", rank_rows)
            _write_jsonl(root / "rollouts.jsonl", rank_rows)
            self.assertAlmostEqual(load_run_dose(root / "loss_log.jsonl")["cumulative_reward_mass"], 6.0)
            with self.assertRaisesRegex(ValueError, "expected 2"):
                audit_rollouts(root / "rollouts_rank_0.jsonl")

            summary = truncate_resume_artifacts(root, 1, 1)
            self.assertEqual(summary["dropped_loss_rows"], 1)
            self.assertEqual(summary["kept_loss_rows"], 2)
            self.assertEqual(summary["dropped_rollout_rows"], 8)
            self.assertEqual(summary["kept_rollout_rows"], 4)

            kept_loss = _read_jsonl(root / "loss_log.jsonl")
            self.assertEqual([row["global_step"] for row in kept_loss], [0, 1])
            self.assertAlmostEqual(load_run_dose(root / "loss_log.jsonl")["cumulative_reward_mass"], 1.0)
            for name in ("rollouts_rank_0.jsonl", "rollouts.jsonl"):
                kept = _read_jsonl(root / name)
                self.assertEqual([row["policy_checkpoint"] for row in kept], ["step_0", "step_0"])
                audited = audit_rollouts(root / name)
                self.assertEqual(audited["status"], "PASSED")
                self.assertEqual(audited["num_groups"], 1)

            again = truncate_resume_artifacts(root, 1, 1)
            self.assertEqual(again["dropped_loss_rows"], 0)
            self.assertEqual(again["dropped_rollout_rows"], 0)
            self.assertEqual(_read_jsonl(root / "loss_log.jsonl"), kept_loss)

    def test_start_step_zero_keeps_baseline_and_drops_policy_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_jsonl(
                root / "loss_log.jsonl",
                [
                    {"global_step": 0, "val_mean_reward": 0.8773},
                    {"global_step": 1, "reward_mass_in_step": 1.0},
                ],
            )
            legacy = _rollout_group("legacy", None)
            committed = _rollout_group("utt:0:0:1", 0)
            _write_jsonl(root / "rollouts_rank_0.jsonl", legacy + committed)
            summary = truncate_resume_artifacts(root, 0, 1)
            self.assertEqual(summary["kept_loss_rows"], 1)
            self.assertEqual(summary["dropped_loss_rows"], 1)
            self.assertEqual(summary["dropped_rollout_rows"], 2)
            kept = _read_jsonl(root / "rollouts_rank_0.jsonl")
            self.assertEqual([row["group_id"] for row in kept], ["legacy", "legacy"])
            self.assertTrue(all("policy_checkpoint" not in row for row in kept))

    def test_null_policy_checkpoint_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = _rollout_group("legacy", 0)
            for row in rows:
                row["policy_checkpoint"] = None
            _write_jsonl(root / "rollouts_rank_0.jsonl", rows)
            summary = truncate_resume_artifacts(root, 4, 1)
            self.assertEqual(summary["dropped_rollout_rows"], 0)
            self.assertEqual(summary["kept_rollout_rows"], 2)

    def test_bad_loss_step_and_corrupt_json_are_rejected_without_rewrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            loss = root / "loss_log.jsonl"
            original = '{"global_step": true}\n'
            loss.write_text(original, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integer global_step"):
                truncate_resume_artifacts(root, 1, 1)
            self.assertEqual(loss.read_text(encoding="utf-8"), original)

            original = '{"global_step": 1.5}\n'
            loss.write_text(original, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "integer global_step"):
                truncate_resume_artifacts(root, 1, 1)
            self.assertEqual(loss.read_text(encoding="utf-8"), original)

            original = "{not json}\n"
            loss.write_text(original, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not JSON"):
                truncate_resume_artifacts(root, 1, 1)
            self.assertEqual(loss.read_text(encoding="utf-8"), original)

    def test_rank_files_follow_the_launch_world_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_jsonl(root / "rollouts_rank_0.jsonl", _rollout_group("a", 3))
            _write_jsonl(root / "rollouts_rank_1.jsonl", _rollout_group("b", 3))
            summary = truncate_resume_artifacts(root, 3, 2)
            self.assertEqual(summary["dropped_rollout_rows"], 4)
            self.assertEqual(_read_jsonl(root / "rollouts_rank_0.jsonl"), [])
            self.assertEqual(_read_jsonl(root / "rollouts_rank_1.jsonl"), [])


class TestSelectResumeStep(unittest.TestCase):
    def test_terminal_statuses(self) -> None:
        for status in ("COMPLETED", "CANDIDATE", "BLOCKED_TRANSFER", "STOPPED_KL", "FAILED_ZERO_VARIANCE"):
            with self.subTest(status=status):
                self.assertTrue(is_terminal_train_status(status))
        self.assertFalse(is_terminal_train_status("CHUNK_DONE"))
        self.assertFalse(is_terminal_train_status(""))

    def test_incomplete_pipeline_falls_back_to_the_last_complete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _complete_checkpoint(root, 320)
            _complete_checkpoint(root, 640, skip="rng_state_rank_0.pt")
            _write_pipeline(root, 640, "CHUNK_DONE")
            choice = select_resume_step(root, world_size=1)
            self.assertEqual(choice["step"], 320)
            self.assertFalse(choice["terminal"])
            self.assertEqual(choice["status"], "CHUNK_DONE")

    def test_complete_terminal_checkpoint_is_not_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _complete_checkpoint(root, 320)
            _complete_checkpoint(root, 640)
            _write_pipeline(root, 320, "STOPPED_KL")
            choice = select_resume_step(root, world_size=1)
            self.assertEqual(choice["step"], 320)
            self.assertTrue(choice["terminal"])
            self.assertEqual(choice["status"], "STOPPED_KL")
            self.assertEqual(choice["source"], "pipeline")

    def test_newer_complete_checkpoint_beats_a_stale_chunk_done_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _complete_checkpoint(root, 320)
            _complete_checkpoint(root, 640)
            _write_pipeline(root, 320, "CHUNK_DONE")
            choice = select_resume_step(root, world_size=1)
            self.assertEqual(choice["step"], 640)
            self.assertFalse(choice["terminal"])
            self.assertEqual(choice["status"], "CHUNK_DONE")

    def test_corrupt_pipeline_json_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _complete_checkpoint(root, 320)
            (root / "pipeline_state.json").write_text("{", encoding="utf-8")
            choice = select_resume_step(root, world_size=1)
            self.assertEqual(choice["step"], 320)
            self.assertFalse(choice["terminal"])

    def test_incomplete_stop_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _complete_checkpoint(root, 320)
            _complete_checkpoint(root, 640, skip="rng_state_rank_0.pt")
            _write_pipeline(root, 640, "STOPPED_KL")
            choice = select_resume_step(root, world_size=1)
            self.assertEqual(choice["step"], 320)
            self.assertFalse(choice["terminal"])

    def test_terminal_checkpoint_metadata_is_safe_without_pipeline_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = _complete_checkpoint(root, 320)
            (checkpoint / "training_state.json").write_text(
                json.dumps({"global_step": 320, "status": "STOPPED_KL"}),
                encoding="utf-8",
            )
            choice = select_resume_step(root, world_size=1)
            self.assertEqual(choice["step"], 320)
            self.assertTrue(choice["terminal"])
            self.assertEqual(choice["status"], "STOPPED_KL")
            self.assertEqual(choice["source"], "checkpoint")


class TestAssertResumeCheckpoint(unittest.TestCase):
    def _state(self, **extra: object) -> dict:
        state = {
            "manifest_sha256": "abc",
            "world_size": 1,
            "model_revision": "rev",
            "scheduler": "constant",
            "global_step": 320,
        }
        state.update(extra)
        return state

    def _write(self, root: Path, state: dict, *, skip: str = "") -> Path:
        checkpoint = _complete_checkpoint(root, 320, skip=skip)
        if skip != "training_state.json":
            (checkpoint / "training_state.json").write_text(
                json.dumps(state),
                encoding="utf-8",
            )
        return checkpoint

    def _assert(self, checkpoint: Path, **overrides: object):
        kwargs = dict(
            manifest_sha256="abc",
            world_size=1,
            model_revision="rev",
            scheduler_name="constant",
            require_scheduler=True,
            val_manifest_sha256="val",
            wer_manifest_sha256="wer",
            seed=7,
            gradient_accumulation_steps=4,
        )
        kwargs.update(overrides)
        return assert_resume_checkpoint(checkpoint, **kwargs)

    def test_matching_checkpoint_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._write(
                Path(tmp),
                self._state(
                    seed=7,
                    gradient_accumulation_steps=4,
                    val_manifest_sha256="val",
                    wer_manifest_sha256="wer",
                ),
            )
            state = self._assert(checkpoint)
            self.assertEqual(state["global_step"], 320)

    def test_old_checkpoint_without_new_fields_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._write(Path(tmp), self._state())
            state = self._assert(checkpoint)
            self.assertNotIn("seed", state)

    def test_identity_mismatches_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self._write(root, self._state())
            with self.assertRaisesRegex(ValueError, "manifest sha256"):
                self._assert(checkpoint, manifest_sha256="other")
            with self.assertRaisesRegex(ValueError, "world_size"):
                self._assert(checkpoint, world_size=4)
            with self.assertRaisesRegex(ValueError, "model_revision"):
                self._assert(checkpoint, model_revision="other-rev")

    def test_missing_scheduler_file_is_rejected_when_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._write(Path(tmp), self._state(), skip="scheduler.pt")
            with self.assertRaisesRegex(ValueError, "scheduler.pt"):
                self._assert(checkpoint)

    def test_scheduler_null_still_requires_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            present = self._write(root, self._state(scheduler=None))
            self._assert(present)
            missing = self._write(root / "missing", self._state(scheduler=None), skip="scheduler.pt")
            with self.assertRaisesRegex(ValueError, "scheduler.pt"):
                self._assert(missing)

    def test_scheduler_name_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._write(Path(tmp), self._state(scheduler="linear"))
            with self.assertRaisesRegex(ValueError, "scheduler"):
                self._assert(checkpoint)

    def test_present_optional_fields_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = self._write(root, self._state(seed=9))
            with self.assertRaisesRegex(ValueError, "seed"):
                self._assert(checkpoint)
            (checkpoint / "training_state.json").write_text(
                json.dumps(self._state(seed=7, gradient_accumulation_steps=8)),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "gradient_accumulation_steps"):
                self._assert(checkpoint)
            (checkpoint / "training_state.json").write_text(
                json.dumps(
                    self._state(
                        seed=7,
                        gradient_accumulation_steps=4,
                        val_manifest_sha256="old-val",
                    )
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "validation manifest"):
                self._assert(checkpoint)

    def test_optional_scheduler_file_can_be_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = self._write(Path(tmp), self._state(scheduler=None), skip="scheduler.pt")
            state = self._assert(checkpoint, require_scheduler=False)
            self.assertIsNone(state["scheduler"])


class TestScaleFutilityStreak(unittest.TestCase):
    def _records(self) -> list:
        return [
            {
                "global_step": 0,
                "val_mean_reward": 0.8773,
                "val_eval_scope": "Full Held-out",
                "val_decode": "greedy",
            },
            {
                "global_step": 320,
                "val_mean_reward": 0.8770,
                "val_eval_scope": "Full Held-out",
                "val_decode": "greedy",
            },
        ]

    def test_only_full_evals_at_or_after_640_count_and_a_gain_resets(self) -> None:
        records = self._records()
        self.assertEqual(scale_futility_streak(records, profile="scale"), 0)
        self.assertEqual(scale_futility_streak(records, profile="pilot"), 0)
        records.append(
            {
                "global_step": 640,
                "val_mean_reward": 0.8771,
                "val_eval_scope": "Full Held-out",
                "val_decode": "greedy",
            }
        )
        self.assertEqual(scale_futility_streak(records, profile="scale"), 1)
        records.append(
            {
                "global_step": 960,
                "val_mean_reward": 0.8770,
                "val_eval_scope": "Full Held-out",
                "val_decode": "greedy",
            }
        )
        self.assertEqual(scale_futility_streak(records, profile="scale"), 2)
        records.append(
            {
                "global_step": 1280,
                "val_mean_reward": 0.8780,
                "val_eval_scope": "Full Held-out",
                "val_decode": "greedy",
            }
        )
        self.assertEqual(scale_futility_streak(records, profile="scale"), 0)

    def test_last_row_for_a_step_wins(self) -> None:
        records = self._records() + [
            {
                "global_step": 640,
                "val_mean_reward": 0.8700,
                "val_eval_scope": "Full Held-out",
                "val_decode": "greedy",
            },
            {
                "global_step": 640,
                "val_mean_reward": 0.8780,
                "val_eval_scope": "Full Held-out",
                "val_decode": "greedy",
            },
        ]
        self.assertEqual(scale_futility_streak(records, profile="scale"), 0)


class TestResumeWiring(unittest.TestCase):
    def test_truncate_and_pipeline_write_happen_in_the_safe_order(self) -> None:
        trainer = Path(__file__).resolve().parents[1].joinpath("train", "train_rl.py").read_text(
            encoding="utf-8"
        )
        truncate_call = trainer.index("truncate_summary = truncate_resume_artifacts(")
        dist_init = trainer.index("dist.init_process_group")
        self.assertLess(trainer.index("resume_state = assert_resume_checkpoint("), truncate_call)
        self.assertLess(truncate_call, dist_init)
        self.assertLess(trainer.index("torch.save(rng_dict"), trainer.index("write_pipeline(training_process_status"))
        self.assertIn(
            'if not probe_only and resume_from_checkpoint:\n'
            '        try:\n            truncate_summary = truncate_resume_artifacts(',
            trainer,
        )
        self.assertLess(
            trainer.index("resume LoRA target hash"),
            trainer.index("ddp_model = DDP("),
        )
        self.assertIn('os.replace(state_tmp, ckpt_dir / "training_state.json")', trainer)
        self.assertIn('"status": training_process_status(global_step, horizon, stop_status)', trainer)
        self.assertIn("output directory already contains RL training artifacts", trainer)


if __name__ == "__main__":
    unittest.main()
