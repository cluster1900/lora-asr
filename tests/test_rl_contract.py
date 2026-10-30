"""Phase 4 RL contract and schema unit tests. They do not download Qwen or use a GPU."""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from train.dpo_schema import load_dpo_config
from train import train_rl
from train.rl_contract import (
    accumulate_window_groups,
    candidate_seed_int,
    compute_candidate_reward,
    evaluate_rollout_group,
    exact_conditional_vocab_kl,
    group_loss_from_token_logprobs,
    judge_rl_checkpoint,
    k3_token_kl,
    prefix_records_sha256,
    reference_tensors_sha256,
)
from train.rl_schema import DPO_RELEASE_ID, load_rl_config, semantic_rl_config_sha256
from train.sft_schema import SCENARIOS, load_train_config

ROOT = Path(__file__).resolve().parents[1]
RL_SMOKE_YAML = ROOT / "configs" / "train" / "rl_smoke.yaml"
RL_PILOT_YAML = ROOT / "configs" / "train" / "rl_pilot.yaml"


class RlSchemaTest(unittest.TestCase):
    def test_rl_smoke_and_pilot_load_and_reject_sft_dpo_loaders(self) -> None:
        smoke = load_rl_config(RL_SMOKE_YAML)
        pilot = load_rl_config(RL_PILOT_YAML)
        self.assertEqual(smoke["stage"], "rl")
        self.assertEqual(smoke["profile"], "smoke")
        self.assertEqual(smoke["train"]["max_steps"], 6)
        self.assertEqual(smoke["train"]["interrupt_after_step"], 5)
        self.assertEqual(pilot["stage"], "rl")
        self.assertEqual(pilot["profile"], "pilot")
        self.assertEqual(pilot["train"]["max_steps"], 50)
        self.assertEqual(pilot["model"]["parent_release"], DPO_RELEASE_ID)
        self.assertEqual(
            pilot["model"]["parent_release_sha256"],
            "1d28d3119385c178c4b1b9a1e52a68d134f33a9bfd3de36659290a184ba5c51a",
        )
        self.assertEqual(pilot["rl"]["parent"]["source_step"], 40)
        manifest_sha = {"train": "a" * 64, "val": "b" * 64, "metadata": "c" * 64}
        self.assertNotEqual(
            semantic_rl_config_sha256(smoke, manifest_sha),
            semantic_rl_config_sha256(pilot, manifest_sha),
        )
        with self.assertRaises(ValueError):
            load_train_config(RL_PILOT_YAML)
        with self.assertRaises(ValueError):
            load_dpo_config(RL_PILOT_YAML)

    def test_rl_budget_and_parent_release_guards(self) -> None:
        raw = yaml.safe_load(RL_PILOT_YAML.read_text(encoding="utf-8"))
        with TemporaryDirectory() as tmp:
            bad_lr = Path(tmp) / "bad_lr.yaml"
            bad_raw = dict(raw)
            bad_raw["optim"] = {**raw["optim"], "learning_rate": 5e-5}
            bad_lr.write_text(yaml.safe_dump(bad_raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_rl_config(bad_lr)

            bad_parent = Path(tmp) / "bad_parent.yaml"
            bad_parent_raw = dict(raw)
            bad_parent_raw["model"] = {**raw["model"], "parent_release": "sft_06b_zh_20261015T234939_step000030"}
            bad_parent.write_text(yaml.safe_dump(bad_parent_raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_rl_config(bad_parent)


class RlContractTest(unittest.TestCase):
    def test_candidate_seed_is_deterministic_and_distinct(self) -> None:
        s0 = candidate_seed_int(42, 0, 1, "u1__noise", 0)
        s0_again = candidate_seed_int(42, 0, 1, "u1__noise", 0)
        s1 = candidate_seed_int(42, 0, 1, "u1__noise", 1)
        self.assertEqual(s0, s0_again)
        self.assertNotEqual(s0, s1)
        self.assertGreater(s0, 0)

    def test_reward_clips_and_penalizes_empty_and_repeated(self) -> None:
        exact = compute_candidate_reward("今天天气很好", "今天天气很好")
        self.assertAlmostEqual(exact["reward"], 1.0)
        self.assertAlmostEqual(exact["cer"], 0.0)

        empty = compute_candidate_reward("今天天气很好", "")
        # CER=1.0, empty=1 -> 1 - 1.0 - 0.5 = -0.5
        self.assertTrue(empty["empty_output"])
        self.assertAlmostEqual(empty["reward"], -0.5)

        repeated = compute_candidate_reward("今天", "啊啊啊啊啊啊啊啊")
        # CER=4.0, repeated=1 -> 1 - 4.0 - 0.5 = -3.5 -> clipped to -1.0
        self.assertTrue(repeated["repeated_output"])
        self.assertAlmostEqual(repeated["unclipped_reward"], -3.5)
        self.assertAlmostEqual(repeated["reward"], -1.0)

    def test_truncation_or_error_invalidates_entire_group(self) -> None:
        cands = [
            {"prediction_raw": "你好", "stop_reason": "eos", "error": None},
            {"prediction_raw": "你好", "stop_reason": "length", "error": None},
            {"prediction_raw": "你号", "stop_reason": "eos", "error": None},
            {"prediction_raw": "你好", "stop_reason": "eos", "error": None},
        ]
        res = evaluate_rollout_group(cands, "你好")
        self.assertFalse(res["group_valid"])
        self.assertEqual(len(res["candidates"]), 4)
        self.assertTrue(all(item["advantage"] is None for item in res["candidates"]))

    def test_equal_rewards_zero_entire_group_including_kl(self) -> None:
        cands = [
            {"prediction_raw": "你好", "stop_reason": "eos", "error": None}
            for _ in range(4)
        ]
        group = evaluate_rollout_group(cands, "你好")
        self.assertTrue(group["group_valid"])
        self.assertTrue(group["zero_advantage"])
        self.assertEqual([c["advantage"] for c in group["candidates"]], [0.0, 0.0, 0.0, 0.0])

        # Even if policy != reference (non-zero k3), zero_advantage zeroes both policy and KL loss
        loss_info = group_loss_from_token_logprobs(
            [0.0, 0.0, 0.0, 0.0],
            [[-0.5, -0.5]] * 4,
            [[-1.2, -1.2]] * 4,
            kl_beta=0.04,
            zero_advantage=True,
        )
        self.assertEqual(loss_info["group_loss"], 0.0)
        self.assertEqual(loss_info["policy_loss"], 0.0)
        self.assertEqual(loss_info["kl_loss"], 0.0)

    def test_non_zero_advantage_sign_and_window_skip(self) -> None:
        cands = [
            {"prediction_raw": "你好世界", "stop_reason": "eos", "error": None},
            {"prediction_raw": "你好世界", "stop_reason": "eos", "error": None},
            {"prediction_raw": "你号世借", "stop_reason": "eos", "error": None},
            {"prediction_raw": "你号世借", "stop_reason": "eos", "error": None},
        ]
        group = evaluate_rollout_group(cands, "你好世界")
        self.assertTrue(group["group_valid"])
        self.assertFalse(group["zero_advantage"])
        advs = [float(c["advantage"]) for c in group["candidates"]]
        self.assertAlmostEqual(sum(advs), 0.0, places=12)
        self.assertGreater(advs[0], 0.0)
        self.assertLess(advs[2], 0.0)

        # All-zero window skips optimizer/scheduler
        zero_win = accumulate_window_groups(
            [
                {"group_valid": True, "zero_advantage": True, "group_loss": 0.0},
                {"group_valid": True, "zero_advantage": True, "group_loss": 0.0},
            ]
        )
        self.assertFalse(zero_win["should_step_optimizer"])

        # Mixed valid window includes zero-advantage group in denominator
        mixed_win = accumulate_window_groups(
            [
                {"group_valid": True, "zero_advantage": False, "group_loss": 0.8},
                {"group_valid": True, "zero_advantage": True, "group_loss": 0.0},
                {"group_valid": False, "zero_advantage": False, "group_loss": 99.0},
            ]
        )
        self.assertTrue(mixed_win["should_step_optimizer"])
        self.assertAlmostEqual(mixed_win["window_loss"], 0.4)

    def test_k3_differs_from_exact_conditional_vocab_kl_and_rejects_nan(self) -> None:
        _, k3_mean = k3_token_kl([-0.4, -0.6], [-0.4, -0.6])
        self.assertAlmostEqual(k3_mean, 0.0)
        with self.assertRaises(ValueError):
            k3_token_kl([float("nan")], [-0.5])

        # Exact conditional KL over a 2-token vocabulary: log([0.5, 0.5]) vs log([0.5, 0.5]) is 0
        ln_half = math.log(0.5)
        self.assertAlmostEqual(
            exact_conditional_vocab_kl([[ln_half, ln_half]], [[ln_half, ln_half]]),
            0.0,
        )
        # Shifted distribution gives positive exact KL
        p = [math.log(0.8), math.log(0.2)]
        q = [math.log(0.5), math.log(0.5)]
        kl = exact_conditional_vocab_kl([p], [q])
        self.assertGreater(kl, 0.0)

    def test_reference_checksum_and_rl_gate(self) -> None:
        h1 = reference_tensors_sha256({"lora_A": b"\x01\x02", "lora_B": b"\x03\x04"})
        h2 = reference_tensors_sha256({"lora_A": b"\x01\x02", "lora_B": b"\x03\x05"})
        self.assertNotEqual(h1, h2)
        p_sha = prefix_records_sha256([{"sample_id": "s1", "token_ids": [1, 2], "prediction_raw": "甲"}])
        self.assertEqual(len(p_sha), 64)

        pilot = load_rl_config(RL_PILOT_YAML)
        parent = pilot["rl"]["parent"]
        base = pilot["baseline"]
        by_scenario = {}
        for name in SCENARIOS:
            cer = float(parent["by_scenario"][name]["micro_cer"])
            if name == "noise":
                cer -= 0.001
            by_scenario[name] = {
                "samples": 500,
                "micro_cer": cer,
                "empty_rate": 0.0,
                "repeated_rate": 0.0,
                "long_rate": 0.0,
                "truncation_rate": 0.0,
                "valid_output_rate": 1.0,
            }
        metrics = {
            "micro_cer": 0.0627,
            "empty_rate": 0.0,
            "repeated_rate": 0.0,
            "long_rate": 0.0,
            "truncation_rate": 0.0,
            "valid_output_rate": 1.0,
            "by_scenario": by_scenario,
        }
        passed_dec = judge_rl_checkpoint(
            metrics,
            parent,
            base,
            pilot["gate"],
            step0_reward=0.930,
            current_reward=0.933,
            raw_kl=0.0004,
            reference_matched=True,
        )
        self.assertTrue(passed_dec["passed"])
        self.assertFalse(passed_dec["risk_stop"])

        kl_fail = judge_rl_checkpoint(
            metrics,
            parent,
            base,
            pilot["gate"],
            step0_reward=0.930,
            current_reward=0.933,
            raw_kl=0.002,
            reference_matched=True,
        )
        self.assertFalse(kl_fail["passed"])
        self.assertTrue(kl_fail["risk_stop"])

    def test_use_reference_adapter_before_policy_forward_prevents_inplace_version_error(self) -> None:
        import ast
        import contextlib
        import sys
        import types
        from unittest import mock
        from train.train_rl import snapshot_lora_tensors, use_reference_adapter

        fake_torch = types.ModuleType("torch")
        fake_torch.no_grad = contextlib.nullcontext  # type: ignore[attr-defined]

        class _FakeParam:
            def __init__(self, value: float, requires_grad: bool = True) -> None:
                self.value = value
                self.requires_grad = requires_grad
                self.device = "cuda:0"
                self.dtype = "bf16"
                self.version = 0

            def detach(self) -> "_FakeParam":
                return _FakeParam(self.value, requires_grad=False)

            def cpu(self) -> "_FakeParam":
                return self

            def clone(self) -> "_FakeParam":
                return _FakeParam(self.value, requires_grad=False)

            def contiguous(self) -> "_FakeParam":
                return self

            def to(self, **_kwargs: object) -> "_FakeParam":
                return self

            def copy_(self, other: "_FakeParam") -> "_FakeParam":
                self.value = other.value
                self.version += 1
                return self

        class _FakeModel:
            def __init__(self) -> None:
                self.training = True
                self.param = _FakeParam(1.0, requires_grad=True)

            def named_parameters(self):
                return [("thinker.model.layers.0.self_attn.q_proj.lora_A.weight", self.param)]

            def eval(self) -> None:
                self.training = False

            def train(self) -> None:
                self.training = True

            def forward_policy(self) -> tuple[float, int]:
                if not self.training:
                    raise AssertionError("policy forward must run in train mode")
                return self.param.value, self.param.version

            def backward_check(self, saved_version: int) -> None:
                if self.param.version != saved_version:
                    raise RuntimeError(
                        f"one of the variables needed for gradient computation has been modified by an inplace operation: "
                        f"is at version {self.param.version}; expected version {saved_version} instead"
                    )

        model = _FakeModel()
        ref_lora = snapshot_lora_tensors(model)
        model.param.value = 2.0

        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            # Wrong order (policy forward -> reference swap -> backward) must raise version mismatch
            _, bad_saved_v = model.forward_policy()
            with use_reference_adapter(model, ref_lora):
                self.assertEqual(model.param.value, 1.0)
            with self.assertRaises(RuntimeError):
                model.backward_check(bad_saved_v)

            # Correct order per candidate (reference swap -> policy forward -> immediate backward) succeeds across all 4 candidates
            for _ in range(4):
                with use_reference_adapter(model, ref_lora):
                    self.assertFalse(model.training)
                    self.assertEqual(model.param.value, 1.0)
                self.assertTrue(model.training)
                self.assertEqual(model.param.value, 2.0)
                _, saved_v = model.forward_policy()
                model.backward_check(saved_v)

        # Verify AST of train/train_rl.py keeps use_reference_adapter ahead of pol_out in score_candidate_tokens
        source = (ROOT / "train" / "train_rl.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        score_fn = next(
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "score_candidate_tokens"
        )
        with_line = next(node.lineno for node in ast.walk(score_fn) if isinstance(node, ast.With))
        pol_assign_line = next(
            node.lineno
            for node in ast.walk(score_fn)
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pol_out" for t in node.targets)
        )
        self.assertLess(with_line, pol_assign_line)

    def test_aligned_rl_pilot_config_and_group_mismatch_mask(self) -> None:
        from train.rl_contract import aligned_group_mismatch_mask

        aligned_cfg = load_rl_config(ROOT / "configs" / "train" / "rl_pilot_aligned.yaml")
        pilot_cfg = load_rl_config(RL_PILOT_YAML)
        self.assertEqual(aligned_cfg["rl"]["train_scenarios"], ["noise", "reverb", "far_field"])
        self.assertEqual(aligned_cfg["rl"]["loss_token_scope"], "mismatched_only")
        self.assertEqual(aligned_cfg["rl"]["max_edit_count"], 2)
        self.assertTrue(aligned_cfg["rl"]["accumulate_signal_groups_only"])
        self.assertAlmostEqual(aligned_cfg["rl"]["dropout_anchor_weight"], 0.05)
        self.assertAlmostEqual(aligned_cfg["optim"]["learning_rate"], 1e-6)
        self.assertEqual(aligned_cfg["optim"]["warmup_steps"], 5)
        self.assertEqual(aligned_cfg["optim"]["scheduler_max_steps"], 50)
        self.assertEqual(aligned_cfg["batch"]["gradient_accumulation_steps"], 4)
        self.assertEqual(aligned_cfg["train"]["max_steps"], 20)
        manifests = {"train": "1" * 64, "val": "2" * 64, "metadata": "3" * 64}
        self.assertNotEqual(
            semantic_rl_config_sha256(aligned_cfg, manifests),
            semantic_rl_config_sha256(pilot_cfg, manifests),
        )

        # Group with 1 aligned negative, 2 positive copies, and 1 comma-inserted candidate (ignored)
        cands = [
            {"prediction_raw": "去上前途未古的训练班", "reward": 0.90},
            {"prediction_raw": "去上，前途未古的训练班", "reward": 0.90},
            {"prediction_raw": "去上前途未卜的训练班", "reward": 1.00},
            {"prediction_raw": "去上前途未卜的训练班", "reward": 1.00},
        ]
        tok_ids = [
            [10, 11, 12, 13, 14, 99, 16, 17, 18, 19, 151645],
            [10, 11, 88, 12, 13, 14, 99, 16, 17, 18, 19, 151645],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 151645],
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 151645],
        ]
        res = aligned_group_mismatch_mask(cands, tok_ids, max_edit_count=2)
        self.assertTrue(res["aligned_valid"])
        self.assertEqual(res["best_index"], 2)
        self.assertEqual(res["mismatch_index"], 5)
        self.assertEqual(res["aligned_indices"], [0, 2, 3])
        self.assertEqual(sum(1 for flag in res["token_mask"] if flag), 1)
        self.assertTrue(res["token_mask"][5])
        self.assertAlmostEqual(sum(res["aligned_advantages"]), 0.0)
        self.assertEqual(res["aligned_advantages"][1], 0.0)
        self.assertLess(res["aligned_advantages"][0], 0.0)
        self.assertGreater(res["aligned_advantages"][2], 0.0)

    def test_rl_release_prepare_and_publish_preserve_unlistened_audit(self) -> None:
        import json
        from train.rl_release import (
            compare_rl_greedy_texts,
            load_published_rl_release,
            load_rl_release,
            mark_rl_published,
            prepare_rl_release,
        )
        from train.sft_checkpoint import utc_now, write_checksums
        from train.sft_release import greedy_fields

        aligned_path = ROOT / "configs" / "train" / "rl_pilot_aligned.yaml"
        config = load_rl_config(aligned_path)
        manifest_sha = {"train": "a" * 64, "val": "b" * 64, "metadata": "c" * 64}
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "20261023T042034_rl_06b_zh_seed42"
            checkpoint = run / "checkpoints" / "step_000005"
            (checkpoint / "adapter").mkdir(parents=True)
            (checkpoint / "adapter" / "adapter.safetensors").write_bytes(b"selected-rl-adapter")
            (checkpoint / "adapter" / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            state = {
                "schema_version": "qwen3-asr-training-state-v1",
                "stage": "rl",
                "global_step": 5,
                "eval_decode_sha256": config["decode_eval_sha256"],
                "parent_release_sha256": config["model"]["parent_release_sha256"],
                "reference_sha256": "e" * 64,
                "config_sha256": semantic_rl_config_sha256(config, manifest_sha),
                "manifest_sha256": manifest_sha,
            }
            (checkpoint / "training_state.json").write_text(json.dumps(state), encoding="utf-8")
            (checkpoint / "resolved_config.yaml").write_text("profile: pilot\n", encoding="utf-8")
            write_checksums(
                checkpoint,
                [
                    "adapter/adapter.safetensors",
                    "adapter/adapter_config.json",
                    "training_state.json",
                    "resolved_config.yaml",
                ],
                utc_now(),
            )
            decision = {
                "step": 5,
                "passed": True,
                "risk_stop": False,
                "mean_greedy_reward": 0.935913,
                "reward_delta_vs_step0": 0.000178,
                "raw_kl": 0.00033839,
                "reference_matched": True,
            }
            gate = {
                "schema_version": "qwen3-asr-gate-v1",
                "status": "passed",
                "publishable": False,
                "selected_step": 5,
                "decisions": [decision],
            }
            (run / "gate.json").write_text(json.dumps(gate), encoding="utf-8")
            (run / "eval_decisions.jsonl").write_text(json.dumps(decision) + "\n", encoding="utf-8")
            rows = [
                {"sample_id": "a__clean", "prediction_raw": "甲", "stop_reason": "eos", "error": None},
                {"sample_id": "a__dropout", "prediction_raw": "乙", "stop_reason": "eos", "error": None},
            ]
            eval_dir = run / "eval" / "step_000005"
            eval_dir.mkdir(parents=True)
            (eval_dir / "scored.jsonl").write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                encoding="utf-8",
            )
            rel_dir = root / "releases" / "rl_06b_zh_20261023T042034_step000005"
            rel = prepare_rl_release(
                config_path=aligned_path,
                run_dir=run,
                output_dir=rel_dir,
                release_id="rl_06b_zh_20261023T042034_step000005",
            )
            self.assertEqual(rel["status"], "staged")
            self.assertFalse(rel["publishable"])
            self.assertEqual(rel["audit_disposition"], "unlistened")
            self.assertEqual(load_rl_release(rel_dir)["source_step"], 5)
            with self.assertRaises(ValueError):
                load_published_rl_release(rel_dir)

            saved_map = greedy_fields(rows, "saved")
            report = compare_rl_greedy_texts(saved_map, saved_map)
            published = mark_rl_published(rel_dir, report)
            self.assertEqual(published["status"], "published")
            self.assertTrue(published["publishable"])
            self.assertEqual(published["audit_disposition"], "unlistened")
            self.assertEqual(load_published_rl_release(rel_dir)["payload_sha256"], rel["payload_sha256"])


if __name__ == "__main__":
    unittest.main()
