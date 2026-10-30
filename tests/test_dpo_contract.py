"""DPO loss, preference gate, and config contract. No model download."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import yaml

from train.dpo_contract import (
    EARLY_DELETE_CLEAN_DELTA,
    EARLY_DELETE_LENGTH_RATIO,
    GAP_INSERTION_MIN,
    QUIET_DROPOUT_ABS_DELTA,
    assert_audit_still_unlistened,
    check_pair_metadata,
    classify_dropout_signature,
    dropout_anchor_step_slice,
    judge_dpo_checkpoint,
    length_buckets,
    listen_waiver_record,
    locate_dropout_edits,
    pair_dpo_loss,
    preference_accuracy,
    preference_gate_ok,
    select_dropout_anchor_rows,
    select_training_pairs,
    strip_dropout_gap,
    validate_pair_rows,
)
from train.dpo_schema import load_dpo_config, semantic_dpo_config_sha256
from train.sft_schema import load_train_config, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DPO_CONFIG = ROOT / "configs" / "train" / "dpo_pilot.yaml"
NONDROPOUT_CONFIG = ROOT / "configs" / "train" / "dpo_pilot_nondropout.yaml"
ALIGNED_CONFIG = ROOT / "configs" / "train" / "dpo_pilot_aligned.yaml"
EDIT1_SEQ_CONFIG = ROOT / "configs" / "train" / "dpo_pilot_edit1_seq.yaml"
EDIT1_ANCHOR_CONFIG = ROOT / "configs" / "train" / "dpo_pilot_edit1_anchor.yaml"
EDIT1_ANCHOR_STOP40_CONFIG = ROOT / "configs" / "train" / "dpo_pilot_edit1_anchor_stop40.yaml"
EDIT1_ANCHOR_STEP40_H100_CONFIG = ROOT / "configs" / "train" / "dpo_pilot_edit1_anchor_step40_h100.yaml"
EDIT1_ANCHOR_STEP40_H100_A4_CONFIG = ROOT / "configs" / "train" / "dpo_pilot_edit1_anchor_step40_h100_a4.yaml"
PROBE_CONFIG = ROOT / "configs" / "train" / "dpo_dropout_probe.yaml"
GAP_CONFIG = ROOT / "configs" / "train" / "dpo_gap_strip_probe.yaml"
SFT_CONFIG = ROOT / "configs" / "train" / "sft_pilot_lr1e5.yaml"


def _health(micro_cer: float, samples: int = 500) -> dict[str, float | int]:
    return {
        "samples": samples,
        "micro_cer": micro_cer,
        "empty_rate": 0.0,
        "repeated_rate": 0.0,
        "long_rate": 0.0,
        "truncation_rate": 0.0,
        "valid_output_rate": 1.0,
    }


def _metrics(values: dict[str, float]) -> dict[str, object]:
    by_scenario = {name: _health(micro_cer) for name, micro_cer in values.items()}
    overall = _health(sum(values.values()) / len(values), samples=2500)
    return {"by_scenario": by_scenario, **overall}


class DpoLossTest(unittest.TestCase):
    def test_equal_policy_and_reference_is_a_tie(self) -> None:
        loss, deltas = pair_dpo_loss([-2.0, -3.0], [-4.0, -1.0], [-2.0, -3.0], [-4.0, -1.0], 0.1)
        self.assertEqual(deltas, [0.0, 0.0])
        self.assertAlmostEqual(loss, math.log(2.0))
        summary = preference_accuracy(deltas)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["ties"], 2)

    def test_mean_is_over_pairs(self) -> None:
        loss, deltas = pair_dpo_loss([-1.0], [-3.0], [-2.0], [-2.0], 0.1)
        delta = 0.1 * ((-1.0 - -3.0) - (-2.0 - -2.0))
        self.assertAlmostEqual(deltas[0], delta)
        self.assertAlmostEqual(loss, math.log1p(math.exp(-delta)) if delta >= 0 else -delta + math.log1p(math.exp(delta)))

    def test_non_finite_logprob_fails(self) -> None:
        with self.assertRaises(ValueError):
            pair_dpo_loss([float("nan")], [-1.0], [-1.0], [-1.0], 0.1)

    def test_length_buckets_use_rejected_minus_chosen(self) -> None:
        buckets = length_buckets([-2, 0, 3], [-1.0, -1.0, -5.0], [-2.0, -1.0, -4.0])
        self.assertEqual(buckets["shorter_rejected"]["pairs"], 1)
        self.assertEqual(buckets["shorter_rejected"]["raw_policy_accuracy"], 1.0)
        self.assertEqual(buckets["equal_length"]["raw_policy_accuracy"], 0.0)
        self.assertEqual(buckets["longer_rejected"]["pairs"], 1)

    def test_half_is_not_enough_even_when_the_floor_is_lower(self) -> None:
        self.assertFalse(preference_gate_ok(0.5, {"min_preference_accuracy": 0.5}))
        self.assertTrue(preference_gate_ok(0.55, {"min_preference_accuracy": 0.55}))
        self.assertFalse(preference_gate_ok(0.549, {"min_preference_accuracy": 0.55}))


class DpoGateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_dpo_config(DPO_CONFIG)
        self.parent = _metrics(
            {
                "clean": 0.02184598580010923,
                "dropout": 0.18801201529219005,
                "far_field": 0.023211359912616056,
                "noise": 0.057755324959038776,
                "reverb": 0.025122883670125613,
            }
        )
        self.base = {"by_scenario": self.config["baseline"]["by_scenario"]}
        self.preference = {
            "pairs": 1022,
            "ties": 0,
            "accuracy": 0.56,
            "raw_policy_accuracy": 0.4,
            "length_buckets": {},
        }

    def test_parent_improvement_and_preference_pass(self) -> None:
        candidate = _metrics(
            {
                "clean": 0.02184598580010923,
                "dropout": 0.187,
                "far_field": 0.023211359912616056,
                "noise": 0.057755324959038776,
                "reverb": 0.025122883670125613,
            }
        )
        decision = judge_dpo_checkpoint(candidate, self.parent, self.base, self.config["gate"], self.preference)
        self.assertTrue(decision["passed"])
        self.assertFalse(decision["risk_stop"])
        self.assertFalse(decision["publishable"])

    def test_preference_below_the_floor_stops(self) -> None:
        candidate = _metrics(
            {
                "clean": 0.02184598580010923,
                "dropout": 0.187,
                "far_field": 0.023211359912616056,
                "noise": 0.057755324959038776,
                "reverb": 0.025122883670125613,
            }
        )
        preference = dict(self.preference, accuracy=0.5)
        decision = judge_dpo_checkpoint(candidate, self.parent, self.base, self.config["gate"], preference)
        self.assertFalse(decision["passed"])
        self.assertTrue(decision["risk_stop"])

    def test_clean_can_pass_the_parent_cap_and_fail_the_base_cap(self) -> None:
        candidate = _metrics(
            {
                "clean": 0.02184598580010923 + 0.009,
                "dropout": 0.187,
                "far_field": 0.023211359912616056,
                "noise": 0.057755324959038776,
                "reverb": 0.025122883670125613,
            }
        )
        base = {
            "by_scenario": {
                name: {"samples": 500, "micro_cer": 0.0 if name == "clean" else 1.0}
                for name in ("clean", "dropout", "far_field", "noise", "reverb")
            }
        }
        decision = judge_dpo_checkpoint(candidate, self.parent, base, self.config["gate"], self.preference)
        self.assertTrue(any("base cap" in risk for risk in decision["risks"]))
        self.assertFalse(decision["passed"])


class DpoConfigTest(unittest.TestCase):
    def test_sft_loader_rejects_the_dpo_file(self) -> None:
        with self.assertRaises(ValueError):
            load_train_config(DPO_CONFIG)

    def test_sft_yaml_still_has_null_dpo(self) -> None:
        config = load_train_config(SFT_CONFIG)
        self.assertEqual(config["stage"], "sft")
        self.assertIsNone(config["dpo"])
        self.assertAlmostEqual(config["optim"]["learning_rate"], 1e-5)
        self.assertAlmostEqual(config["lora"]["dropout"], 0.05)

    def test_pilot_budget_is_frozen(self) -> None:
        config = load_dpo_config(DPO_CONFIG)
        self.assertEqual(config["stage"], "dpo")
        self.assertEqual(config["model"]["parent_release"], "sft_06b_zh_20261015T234939_step000030")
        self.assertAlmostEqual(config["optim"]["learning_rate"], 5e-6)
        self.assertEqual(config["lora"]["dropout"], 0.0)
        self.assertEqual(config["batch"]["gradient_accumulation_steps"], 4)
        self.assertEqual(config["dpo"]["beta"], 0.1)
        self.assertEqual(config["dpo"]["logp_reduction"], "sequence_sum")
        self.assertEqual(config["dpo"]["listen_waiver"], "user_20261017_skip_listen")
        self.assertEqual(config["dpo"]["audit_disposition"], "unlistened")
        self.assertEqual(config["dpo"]["filled_train_pairs_sha256"], "43473f10fbb9c0349adf2eb76eaf647506978c96eabb5b09847994456519c1b1")
        self.assertEqual(config["train"]["max_steps"], 100)
        self.assertEqual(config["runtime"]["max_runtime_minutes"], 120)
        manifests = {"train": "a" * 64, "val": "b" * 64, "metadata": "c" * 64}
        self.assertEqual(
            semantic_dpo_config_sha256(config, manifests),
            semantic_dpo_config_sha256(config, manifests),
        )

    def test_listened_disposition_is_rejected(self) -> None:
        raw = yaml.safe_load(DPO_CONFIG.read_text(encoding="utf-8"))
        raw["dpo"]["audit_disposition"] = "listened"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dpo_config(path)

    def test_waiver_does_not_mark_clips_heard(self) -> None:
        record = listen_waiver_record()
        self.assertFalse(record["heard"])
        self.assertEqual(record["audit_disposition"], "unlistened")
        self.assertEqual(record["utterance"], "不听，直接开始训练")
        self.assertNotIn("gap_strip_launch", record)
        with self.assertRaises(ValueError):
            assert_audit_still_unlistened({"status": "unlistened", "by_scenario": {"train:clean": {"disposition": "passed"}}})

    def test_gap_strip_launch_does_not_mark_clips_heard(self) -> None:
        record = listen_waiver_record("开始吧")
        self.assertFalse(record["heard"])
        self.assertEqual(record["audit_disposition"], "unlistened")
        self.assertEqual(record["utterance"], "不听，直接开始训练")
        launch = record["gap_strip_launch"]
        self.assertEqual(launch["utterance"], "开始吧")
        self.assertEqual(launch["decided_on"], "2026-10-18")
        self.assertFalse(launch["prior_waiver_covers_this_audit"])
        self.assertFalse(launch["heard"])
        self.assertEqual(launch["audit_disposition"], "unlistened")
        with self.assertRaises(ValueError):
            listen_waiver_record("不听，直接开始训练")

    def test_pair_rows_reject_a_null_logprob(self) -> None:
        row = {
            "schema_version": "qwen3-asr-dpo-pair-v1",
            "pair_id": "u1",
            "split": "val",
            "scenario": "clean",
            "reference_text": "你好",
            "chosen_text": "你好",
            "rejected_text": "你号",
            "ref_chosen_logp": None,
            "ref_rejected_logp": -1.0,
            "chosen_token_count": 2,
            "rejected_token_count": 2,
            "char_length_delta": 0,
        }
        with self.assertRaises(ValueError):
            validate_pair_rows([row], split="val", min_per_scenario=1)

    def test_dropout_probe_budget_is_frozen(self) -> None:
        probe = load_dpo_config(PROBE_CONFIG)
        pilot = load_dpo_config(DPO_CONFIG)
        self.assertEqual(probe["profile"], "dropout_probe")
        self.assertEqual(probe["dpo"]["train_scenarios"], ["dropout"])
        self.assertIsNone(pilot["dpo"]["train_scenarios"])
        self.assertAlmostEqual(probe["optim"]["learning_rate"], 5e-5)
        self.assertEqual(probe["optim"]["warmup_steps"], 0)
        self.assertEqual(probe["train"]["max_steps"], 10)
        self.assertEqual(probe["train"]["keep_last"], 1)
        self.assertEqual(probe["dpo"]["beta"], 0.1)
        self.assertEqual(probe["model"]["parent_release"], pilot["model"]["parent_release"])
        self.assertEqual(probe["dpo"]["null_train_pairs_sha256"], pilot["dpo"]["null_train_pairs_sha256"])
        self.assertEqual(probe["dpo"]["filled_train_pairs_sha256"], pilot["dpo"]["filled_train_pairs_sha256"])
        manifests = {"train": "a" * 64, "val": "b" * 64, "metadata": "c" * 64}
        self.assertNotEqual(
            semantic_dpo_config_sha256(probe, manifests),
            semantic_dpo_config_sha256(pilot, manifests),
        )

    def test_gap_strip_probe_keeps_the_failed_probe_budget(self) -> None:
        gap = load_dpo_config(GAP_CONFIG)
        probe = load_dpo_config(PROBE_CONFIG)
        pilot = load_dpo_config(DPO_CONFIG)
        self.assertEqual(gap["profile"], "dropout_probe")
        self.assertEqual(gap["dpo"]["train_scenarios"], ["dropout"])
        self.assertAlmostEqual(gap["optim"]["learning_rate"], 5e-5)
        self.assertEqual(gap["optim"]["warmup_steps"], 0)
        self.assertEqual(gap["train"]["max_steps"], 10)
        self.assertEqual(gap["train"]["save_steps"], 10)
        self.assertEqual(gap["train"]["eval_steps"], 10)
        self.assertEqual(gap["train"]["keep_last"], 1)
        self.assertEqual(gap["dpo"]["beta"], 0.1)
        self.assertEqual(gap["lora"]["dropout"], 0.0)
        self.assertEqual(gap["batch"]["micro_batch_size"], 1)
        self.assertEqual(gap["batch"]["gradient_accumulation_steps"], 4)
        self.assertEqual(gap["runtime"]["max_runtime_minutes"], 120)
        self.assertEqual(gap["model"]["parent_release"], probe["model"]["parent_release"])
        self.assertEqual(gap["model"]["parent_release_sha256"], probe["model"]["parent_release_sha256"])
        self.assertEqual(gap["dpo"]["listen_waiver"], "user_20261017_skip_listen")
        self.assertEqual(gap["dpo"]["audit_disposition"], "unlistened")
        self.assertEqual(gap["dpo"]["launch_utterance"], "开始吧")
        self.assertIsNone(probe["dpo"]["launch_utterance"])
        self.assertIsNone(pilot["dpo"]["launch_utterance"])
        self.assertEqual(gap["dpo"]["null_train_pairs_sha256"], "4ad364c87d367395d67603fdfb7acf503bd2c0174cfb35518b7b30b0c096505d")
        self.assertEqual(gap["dpo"]["null_val_pairs_sha256"], "caffc52d2674562143578a2d18a73bfed85233c74f6d573495ca77477d695a0d")
        self.assertEqual(gap["dpo"]["filled_train_pairs_sha256"], "7c01e5fc54596784f7e9ef22d3926cf830e3d034b58107aa455ac105bf0c392b")
        self.assertEqual(gap["dpo"]["filled_val_pairs_sha256"], "008fe7b044496d16645b022c2ef7f0677aca9c6d3a5de58f5c0e98ff5b65208b")
        self.assertEqual(gap["dpo"]["reference_cache_sha256"], "dd9db5012841d61dafce571c833c508bac7e7f9e64286c0928dba9e139a2da00")
        self.assertEqual(gap["dpo"]["reference_report_sha256"], "35b9361a7cce77ab2dea805ac468e4a87a88bb066aa2b43d19339fee4c5eb114")
        self.assertEqual(gap["dpo"]["pair_metadata_sha256"], "0b00f8e6d2e043593fcbed6b7eb5e0a47ea164d2b8f43cb20080d6c1860931a7")
        self.assertEqual(gap["dpo"]["source_null_train_pairs_sha256"], probe["dpo"]["null_train_pairs_sha256"])
        self.assertEqual(gap["dpo"]["source_null_val_pairs_sha256"], probe["dpo"]["null_val_pairs_sha256"])
        self.assertNotEqual(gap["dpo"]["null_train_pairs_sha256"], probe["dpo"]["null_train_pairs_sha256"])
        manifests = {"train": "a" * 64, "val": "b" * 64, "metadata": "c" * 64}
        self.assertNotEqual(
            semantic_dpo_config_sha256(gap, manifests),
            semantic_dpo_config_sha256(probe, manifests),
        )
        pinned = {
            ROOT / "results/phase3_dpo_gap_strip_20261018/pairs_gap_strip/train_pairs.jsonl": gap["dpo"]["null_train_pairs_sha256"],
            ROOT / "results/phase3_dpo_gap_strip_20261018/pairs_gap_strip/val_pairs.jsonl": gap["dpo"]["null_val_pairs_sha256"],
            ROOT / "results/phase3_dpo_gap_strip_20261018/pairs_gap_strip/pair_metadata.json": gap["dpo"]["pair_metadata_sha256"],
            ROOT / gap["dpo"]["pair_train_manifest"]: gap["dpo"]["filled_train_pairs_sha256"],
            ROOT / gap["dpo"]["pair_val_manifest"]: gap["dpo"]["filled_val_pairs_sha256"],
            ROOT / gap["dpo"]["reference_cache"]: gap["dpo"]["reference_cache_sha256"],
            ROOT / gap["dpo"]["reference_report"]: gap["dpo"]["reference_report_sha256"],
        }
        for path, digest in pinned.items():
            if path.is_file():
                self.assertEqual(sha256_file(path), digest, path)
        metadata_path = ROOT / "results/phase3_dpo_gap_strip_20261018/pairs_gap_strip/pair_metadata.json"
        if metadata_path.is_file():
            check_pair_metadata(json.loads(metadata_path.read_text(encoding="utf-8")), gap, {})
        source = (ROOT / "train" / "train_dpo.py").read_text(encoding="utf-8")
        self.assertNotIn("strip_dropout_gap", source)

    def test_pilot_rejects_the_gap_strip_launch(self) -> None:
        raw = yaml.safe_load(DPO_CONFIG.read_text(encoding="utf-8"))
        raw["dpo"]["launch_utterance"] = "开始吧"
        raw["dpo"]["pair_metadata_sha256"] = "a" * 64
        raw["dpo"]["source_null_train_pairs_sha256"] = "b" * 64
        raw["dpo"]["source_null_val_pairs_sha256"] = "c" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pilot.yaml"
            path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dpo_config(path)

    def test_gap_strip_metadata_stays_untrainable(self) -> None:
        payload = {
            "schema_version": "qwen3-asr-dpo-gap-strip-metadata-v1",
            "status": "pairs_gap_stripped",
            "trainable": False,
            "gap_strip": "dropout-0.40-0.55",
            "seed": 42,
            "train_pairs_sha256": "a" * 64,
            "val_pairs_sha256": "b" * 64,
            "source_train_pairs_sha256": "c" * 64,
            "source_val_pairs_sha256": "d" * 64,
        }
        config = {
            "seed": 42,
            "model": {"parent_release": "sft_06b_zh_20261015T234939_step000030"},
            "dpo": {
                "launch_utterance": "开始吧",
                "null_train_pairs_sha256": "a" * 64,
                "null_val_pairs_sha256": "b" * 64,
                "source_null_train_pairs_sha256": "c" * 64,
                "source_null_val_pairs_sha256": "d" * 64,
            },
        }
        check_pair_metadata(payload, config, {})
        payload["trainable"] = True
        with self.assertRaises(ValueError):
            check_pair_metadata(payload, config, {})
        payload["trainable"] = False
        config["dpo"]["launch_utterance"] = None
        with self.assertRaises(ValueError):
            check_pair_metadata(payload, config, {})

    def test_pilot_rejects_a_scenario_filter(self) -> None:
        for bad in (["dropout"], ["clean", "dropout"], ["clean", "noise"]):
            raw = yaml.safe_load(DPO_CONFIG.read_text(encoding="utf-8"))
            raw["dpo"]["train_scenarios"] = bad
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "pilot.yaml"
                path.write_text(yaml.safe_dump(raw), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_dpo_config(path)

    def test_nondropout_pilot_keeps_frozen_pilot_budget_and_hashes(self) -> None:
        nondrop = load_dpo_config(NONDROPOUT_CONFIG)
        pilot = load_dpo_config(DPO_CONFIG)
        probe = load_dpo_config(PROBE_CONFIG)
        gap = load_dpo_config(GAP_CONFIG)
        self.assertEqual(nondrop["profile"], "pilot")
        self.assertEqual(nondrop["dpo"]["train_scenarios"], ["clean", "noise", "reverb", "far_field"])
        self.assertAlmostEqual(nondrop["optim"]["learning_rate"], 5e-6)
        self.assertEqual(nondrop["optim"]["warmup_steps"], 2)
        self.assertEqual(nondrop["train"]["max_steps"], 100)
        self.assertEqual(nondrop["train"]["save_steps"], 10)
        self.assertEqual(nondrop["train"]["eval_steps"], 10)
        self.assertEqual(nondrop["train"]["keep_last"], 10)
        self.assertEqual(nondrop["dpo"]["beta"], 0.1)
        self.assertEqual(nondrop["lora"]["dropout"], 0.0)
        self.assertIsNone(nondrop["dpo"]["launch_utterance"])
        for key in (
            "null_train_pairs_sha256",
            "null_val_pairs_sha256",
            "filled_train_pairs_sha256",
            "filled_val_pairs_sha256",
            "reference_cache_sha256",
            "reference_report_sha256",
        ):
            self.assertEqual(nondrop["dpo"][key], pilot["dpo"][key])
        manifests = {"train": "a" * 64, "val": "b" * 64, "metadata": "c" * 64}
        nondrop_sha = semantic_dpo_config_sha256(nondrop, manifests)
        self.assertNotEqual(nondrop_sha, semantic_dpo_config_sha256(pilot, manifests))
        self.assertNotEqual(nondrop_sha, semantic_dpo_config_sha256(probe, manifests))
        self.assertNotEqual(nondrop_sha, semantic_dpo_config_sha256(gap, manifests))

    def test_probe_rejects_the_pilot_learning_rate(self) -> None:
        raw = yaml.safe_load(PROBE_CONFIG.read_text(encoding="utf-8"))
        raw["optim"]["learning_rate"] = 0.000005
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "probe.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_dpo_config(path)

    def test_required_scenarios_ignore_absent_counts(self) -> None:
        row = {
            "schema_version": "qwen3-asr-dpo-pair-v1",
            "pair_id": "u1",
            "split": "val",
            "scenario": "dropout",
            "reference_text": "你好",
            "chosen_text": "你好",
            "rejected_text": "你号",
            "ref_chosen_logp": -1.0,
            "ref_rejected_logp": -2.0,
            "chosen_token_count": 2,
            "rejected_token_count": 2,
            "char_length_delta": 0,
        }
        counts = validate_pair_rows([row], split="val", min_per_scenario=1, required_scenarios=("dropout",))
        self.assertEqual(counts["dropout"], 1)
        self.assertEqual(counts["clean"], 0)
        with self.assertRaises(ValueError):
            validate_pair_rows([row], split="val", min_per_scenario=1)
        kept = select_training_pairs([row, dict(row, pair_id="u2", scenario="clean")], ["dropout"])
        self.assertEqual([item["pair_id"] for item in kept], ["u1"])

    def test_dropout_anchor_uses_parent_hypothesis_and_keeps_the_hard_gate(self) -> None:
        aligned = load_dpo_config(ALIGNED_CONFIG)
        seq = load_dpo_config(EDIT1_SEQ_CONFIG)
        anchor = load_dpo_config(EDIT1_ANCHOR_CONFIG)
        self.assertEqual(anchor["profile"], "pilot")
        self.assertEqual(anchor["gate"], aligned["gate"])
        self.assertEqual(anchor["gate"]["degraded_macro_increase_max"], 0.0)
        self.assertAlmostEqual(anchor["gate"]["min_preference_accuracy"], 0.55)
        self.assertEqual(anchor["dpo"]["loss_token_scope"], "mismatched_only")
        self.assertEqual(anchor["dpo"]["max_edit_count"], 1)
        self.assertEqual(anchor["optim"]["warmup_steps"], 10)
        self.assertAlmostEqual(anchor["dpo"]["dropout_anchor_weight"], 0.05)
        self.assertEqual(anchor["dpo"]["dropout_anchor_target"], "rejected_text")
        self.assertAlmostEqual(aligned["dpo"]["dropout_anchor_weight"], 0.0)
        self.assertAlmostEqual(seq["dpo"]["dropout_anchor_weight"], 0.0)
        manifests = {"train": "a" * 64, "val": "b" * 64, "metadata": "c" * 64}
        anchor_sha = semantic_dpo_config_sha256(anchor, manifests)
        self.assertNotEqual(anchor_sha, semantic_dpo_config_sha256(aligned, manifests))
        self.assertNotEqual(anchor_sha, semantic_dpo_config_sha256(seq, manifests))
        raw = yaml.safe_load(ALIGNED_CONFIG.read_text(encoding="utf-8"))
        raw["dpo"]["dropout_anchor_weight"] = 0.0
        raw["dpo"]["dropout_anchor_target"] = "rejected_text"
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "zero.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            zero = load_dpo_config(path)
        self.assertEqual(
            semantic_dpo_config_sha256(zero, manifests),
            semantic_dpo_config_sha256(aligned, manifests),
        )
        for bad_patch in (
            {"dropout_anchor_weight": 0.2},
            {"dropout_anchor_weight": 0.05, "dropout_anchor_target": "chosen_text"},
            {"dropout_anchor_weight": 0.05, "loss_token_scope": "sequence"},
            {"dropout_anchor_weight": 0.05, "train_scenarios": ["dropout"]},
        ):
            broken = yaml.safe_load(ALIGNED_CONFIG.read_text(encoding="utf-8"))
            broken["dpo"].update(bad_patch)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "bad.yaml"
                path.write_text(yaml.safe_dump(broken), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_dpo_config(path)
        rows = [
            {"scenario": "noise", "split": "train", "rejected_text": "噪声", "chosen_text": "金标"},
            {"scenario": "dropout", "split": "train", "rejected_text": "父版本", "chosen_text": "金标"},
            {"scenario": "dropout", "split": "train", "rejected_text": "  ", "chosen_text": "金标"},
            {"scenario": "dropout", "split": "val", "rejected_text": "验证", "chosen_text": "金标"},
            {"scenario": "dropout", "split": "train", "rejected_text": "下一条", "chosen_text": "另一条金标"},
        ]
        kept = select_dropout_anchor_rows(rows)
        self.assertEqual([item["rejected_text"] for item in kept], ["父版本", "下一条"])
        self.assertTrue(all(item["chosen_text"] != item["rejected_text"] for item in kept))
        source = (ROOT / "train" / "train_dpo.py").read_text(encoding="utf-8")
        loop = source[source.index("for update in training_updates") :]
        self.assertLess(loop.index("anchor_loss.backward()"), loop.index("clip_grad_norm_("))
        self.assertIn('str(anchor_row["rejected_text"])', source)
        self.assertNotIn('anchor_row["chosen_text"]', source)

    def test_anchor_stop40_keeps_the_recipe_and_drops_the_epoch_tail(self) -> None:
        from train.sft_contract import iterate_updates, update_ends_schedule

        anchor = load_dpo_config(EDIT1_ANCHOR_CONFIG)
        stopped = load_dpo_config(EDIT1_ANCHOR_STOP40_CONFIG)
        self.assertEqual(anchor["train"]["max_steps"], 100)
        self.assertEqual(stopped["train"]["max_steps"], 40)
        self.assertEqual(stopped["profile"], "pilot")
        self.assertEqual(stopped["gate"], anchor["gate"])
        self.assertEqual(stopped["dpo"]["loss_token_scope"], "mismatched_only")
        self.assertEqual(stopped["optim"]["warmup_steps"], 10)
        self.assertAlmostEqual(stopped["dpo"]["dropout_anchor_weight"], 0.05)
        self.assertEqual(stopped["dpo"]["dropout_anchor_target"], "rejected_text")
        self.assertAlmostEqual(stopped["optim"]["learning_rate"], 5e-6)
        self.assertAlmostEqual(stopped["optim"]["eps"], 1e-6)
        self.assertAlmostEqual(stopped["optim"]["weight_decay"], 0.0)
        for key in (
            "null_train_pairs_sha256",
            "null_val_pairs_sha256",
            "filled_train_pairs_sha256",
            "filled_val_pairs_sha256",
            "reference_cache_sha256",
            "reference_report_sha256",
        ):
            self.assertEqual(stopped["dpo"][key], anchor["dpo"][key])
        manifests = {"train": "a" * 64, "val": "b" * 64, "metadata": "c" * 64}
        self.assertNotEqual(
            semantic_dpo_config_sha256(stopped, manifests),
            semantic_dpo_config_sha256(anchor, manifests),
        )
        updates = list(
            iterate_updates(
                [{"sample_id": f"s{index}"} for index in range(164)],
                seed=42,
                accumulation=4,
                max_steps=40,
                max_epochs=1,
            )
        )
        self.assertEqual(len(updates), 40)
        self.assertEqual(updates[-1]["global_step"], 40)
        self.assertEqual(len(updates[-1]["rows"]), 4)
        self.assertTrue(update_ends_schedule(updates[-1], row_count=164, max_steps=40, max_epochs=1))
        self.assertFalse(update_ends_schedule(updates[-2], row_count=164, max_steps=40, max_epochs=1))
        for source, patch in (
            (ALIGNED_CONFIG, {"train": {"max_steps": 40}}),
            (EDIT1_ANCHOR_CONFIG, {"train": {"max_steps": 50}}),
            (EDIT1_ANCHOR_STOP40_CONFIG, {"optim": {"warmup_steps": 2}}),
        ):
            broken = yaml.safe_load(source.read_text(encoding="utf-8"))
            for section, values in patch.items():
                broken[section].update(values)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "bad.yaml"
                path.write_text(yaml.safe_dump(broken), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_dpo_config(path)

    def test_step40_horizon100_and_multi_anchor_configs_decouple_cosine_and_average_anchors(self) -> None:
        from train.train_sft import HorizonScheduler

        class _DummyOptimizer:
            def __init__(self) -> None:
                self.param_groups = [{"lr": 0.0}]

        anchor = load_dpo_config(EDIT1_ANCHOR_CONFIG)
        stop40 = load_dpo_config(EDIT1_ANCHOR_STOP40_CONFIG)
        h100 = load_dpo_config(EDIT1_ANCHOR_STEP40_H100_CONFIG)
        h100_a4 = load_dpo_config(EDIT1_ANCHOR_STEP40_H100_A4_CONFIG)

        self.assertNotIn("scheduler_max_steps", anchor["optim"])
        self.assertNotIn("scheduler_max_steps", stop40["optim"])
        self.assertEqual(h100["train"]["max_steps"], 40)
        self.assertEqual(h100["optim"]["scheduler_max_steps"], 100)
        self.assertEqual(h100["dpo"]["dropout_anchor_samples_per_step"], 1)
        self.assertEqual(h100_a4["train"]["max_steps"], 40)
        self.assertEqual(h100_a4["optim"]["scheduler_max_steps"], 100)
        self.assertEqual(h100_a4["dpo"]["dropout_anchor_samples_per_step"], 4)
        self.assertEqual(h100["gate"], anchor["gate"])
        self.assertEqual(h100_a4["gate"], anchor["gate"])

        sched_anchor = HorizonScheduler(_DummyOptimizer(), anchor)
        sched_stop40 = HorizonScheduler(_DummyOptimizer(), stop40)
        sched_h100 = HorizonScheduler(_DummyOptimizer(), h100)
        for step in range(1, 41):
            sched_anchor.step()
            sched_stop40.step()
            sched_h100.step()
            self.assertEqual(
                sched_h100.optimizer.param_groups[0]["lr"],
                sched_anchor.optimizer.param_groups[0]["lr"],
                f"step {step} lr differed between max_steps=100 and step40_h100",
            )
            if step == 20:
                self.assertAlmostEqual(sched_h100.optimizer.param_groups[0]["lr"], 4.864308396768294e-6)
                self.assertAlmostEqual(sched_stop40.optimizer.param_groups[0]["lr"], 3.875e-6)
            if step == 30:
                self.assertAlmostEqual(sched_h100.optimizer.param_groups[0]["lr"], 4.473599997017701e-6)
                self.assertAlmostEqual(sched_stop40.optimizer.param_groups[0]["lr"], 1.625e-6)
            if step == 40:
                self.assertAlmostEqual(sched_h100.optimizer.param_groups[0]["lr"], 3.875e-6)
                self.assertAlmostEqual(sched_stop40.optimizer.param_groups[0]["lr"], 5e-7)

        manifests = {"train": "a" * 64, "val": "b" * 64, "metadata": "c" * 64}
        hashes = [
            semantic_dpo_config_sha256(cfg, manifests)
            for cfg in (anchor, stop40, h100, h100_a4)
        ]
        self.assertEqual(len(set(hashes)), 4)

        raw_default_samples = yaml.safe_load(EDIT1_ANCHOR_STEP40_H100_CONFIG.read_text(encoding="utf-8"))
        raw_default_samples["dpo"]["dropout_anchor_samples_per_step"] = 1
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "default_samples.yaml"
            path.write_text(yaml.safe_dump(raw_default_samples), encoding="utf-8")
            parsed_default_samples = load_dpo_config(path)
        self.assertEqual(
            semantic_dpo_config_sha256(parsed_default_samples, manifests),
            semantic_dpo_config_sha256(h100, manifests),
        )

        pool = [{"pair_id": f"d{idx}"} for idx in range(6)]
        self.assertEqual([r["pair_id"] for r in dropout_anchor_step_slice(pool, 1, 1)], ["d0"])
        self.assertEqual([r["pair_id"] for r in dropout_anchor_step_slice(pool, 6, 1)], ["d5"])
        self.assertEqual([r["pair_id"] for r in dropout_anchor_step_slice(pool, 7, 1)], ["d0"])
        self.assertEqual([r["pair_id"] for r in dropout_anchor_step_slice(pool, 1, 4)], ["d0", "d1", "d2", "d3"])
        self.assertEqual([r["pair_id"] for r in dropout_anchor_step_slice(pool, 2, 4)], ["d4", "d5", "d0", "d1"])
        for bad_args in (([], 1, 1), (pool, 0, 1), (pool, True, 1), (pool, 1, 0), (pool, 1, False)):
            with self.assertRaises(ValueError):
                dropout_anchor_step_slice(*bad_args)

        for source, patch in (
            (EDIT1_ANCHOR_CONFIG, {"optim": {"scheduler_max_steps": 100}}),
            (EDIT1_ANCHOR_STOP40_CONFIG, {"optim": {"scheduler_max_steps": 80}}),
            (EDIT1_ANCHOR_STOP40_CONFIG, {"optim": {"scheduler_max_steps": 30}}),
            (PROBE_CONFIG, {"optim": {"scheduler_max_steps": 100}}),
            (ALIGNED_CONFIG, {"dpo": {"dropout_anchor_samples_per_step": 4}}),
            (EDIT1_ANCHOR_STEP40_H100_CONFIG, {"dpo": {"dropout_anchor_samples_per_step": 0}}),
            (EDIT1_ANCHOR_STEP40_H100_CONFIG, {"dpo": {"dropout_anchor_samples_per_step": 5}}),
            (EDIT1_ANCHOR_STEP40_H100_CONFIG, {"dpo": {"dropout_anchor_samples_per_step": True}}),
        ):
            broken = yaml.safe_load(source.read_text(encoding="utf-8"))
            for section, values in patch.items():
                broken[section].update(values)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "bad.yaml"
                path.write_text(yaml.safe_dump(broken), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_dpo_config(path)

    def test_edit1_sequence_config_keeps_the_hard_gate(self) -> None:
        aligned = load_dpo_config(ALIGNED_CONFIG)
        seq = load_dpo_config(EDIT1_SEQ_CONFIG)
        self.assertEqual(seq["profile"], "pilot")
        self.assertEqual(seq["gate"], aligned["gate"])
        self.assertEqual(seq["gate"]["degraded_macro_increase_max"], 0.0)
        self.assertAlmostEqual(seq["gate"]["min_preference_accuracy"], 0.55)
        self.assertEqual(seq["dpo"]["max_edit_count"], 1)
        self.assertEqual(seq["dpo"]["loss_token_scope"], "sequence")
        self.assertEqual(seq["dpo"]["train_scenarios"], ["noise", "reverb", "far_field"])
        self.assertEqual(seq["dpo"]["val_scenarios"], ["clean", "noise", "reverb", "far_field"])
        self.assertEqual(seq["optim"]["warmup_steps"], 2)
        self.assertAlmostEqual(seq["optim"]["eps"], 1e-6)
        self.assertAlmostEqual(seq["optim"]["weight_decay"], 0.0)
        self.assertAlmostEqual(seq["optim"]["learning_rate"], 5e-6)
        for key in (
            "null_train_pairs_sha256",
            "null_val_pairs_sha256",
            "filled_train_pairs_sha256",
            "filled_val_pairs_sha256",
            "reference_cache_sha256",
            "reference_report_sha256",
        ):
            self.assertEqual(seq["dpo"][key], aligned["dpo"][key])
        manifests = {"train": "a" * 64, "val": "b" * 64, "metadata": "c" * 64}
        self.assertNotEqual(
            semantic_dpo_config_sha256(seq, manifests),
            semantic_dpo_config_sha256(aligned, manifests),
        )
        self.assertIn("loss_token_scope: mismatched_only", ALIGNED_CONFIG.read_text(encoding="utf-8"))
        self.assertNotIn("mismatched_only", EDIT1_SEQ_CONFIG.read_text(encoding="utf-8"))

    def test_aligned_pilot_filters_punctuation_and_unequal_tokens_and_damps_optim(self) -> None:
        from train.dpo_contract import build_clean_rejected_map

        aligned = load_dpo_config(ALIGNED_CONFIG)
        nondrop = load_dpo_config(NONDROPOUT_CONFIG)
        pilot = load_dpo_config(DPO_CONFIG)
        self.assertEqual(aligned["profile"], "pilot")
        self.assertEqual(aligned["dpo"]["train_scenarios"], ["noise", "reverb", "far_field"])
        self.assertEqual(aligned["dpo"]["val_scenarios"], ["clean", "noise", "reverb", "far_field"])
        self.assertTrue(aligned["dpo"]["aligned_substitution_only"])
        self.assertTrue(aligned["dpo"]["exclude_clean_identical_rejected"])
        self.assertEqual(aligned["dpo"]["max_edit_count"], 1)
        self.assertEqual(aligned["dpo"]["loss_token_scope"], "mismatched_only")
        self.assertAlmostEqual(aligned["dpo"]["ref_margin_min"], -4.0)
        self.assertAlmostEqual(aligned["dpo"]["ref_margin_max"], 0.0)
        self.assertFalse(nondrop["dpo"]["aligned_substitution_only"])
        self.assertFalse(nondrop["dpo"]["exclude_clean_identical_rejected"])
        self.assertIsNone(nondrop["dpo"]["max_edit_count"])
        self.assertEqual(nondrop["dpo"]["loss_token_scope"], "sequence")
        self.assertIsNone(nondrop["dpo"]["val_scenarios"])
        self.assertIsNone(nondrop["dpo"]["ref_margin_min"])
        self.assertIsNone(nondrop["dpo"]["ref_margin_max"])
        self.assertAlmostEqual(aligned["optim"]["learning_rate"], 5e-6)
        self.assertEqual(aligned["optim"]["warmup_steps"], 10)
        self.assertAlmostEqual(aligned["optim"]["eps"], 1e-6)
        self.assertAlmostEqual(aligned["optim"]["weight_decay"], 0.0)
        for key in (
            "null_train_pairs_sha256",
            "null_val_pairs_sha256",
            "filled_train_pairs_sha256",
            "filled_val_pairs_sha256",
            "reference_cache_sha256",
            "reference_report_sha256",
        ):
            self.assertEqual(aligned["dpo"][key], pilot["dpo"][key])
        manifests = {"train": "a" * 64, "val": "b" * 64, "metadata": "c" * 64}
        self.assertNotEqual(
            semantic_dpo_config_sha256(aligned, manifests),
            semantic_dpo_config_sha256(nondrop, manifests),
        )
        for bad_patch in (
            {"aligned_substitution_only": "yes"},
            {"exclude_clean_identical_rejected": "yes"},
            {"max_edit_count": 0},
            {"max_edit_count": True},
            {"loss_token_scope": "invalid_scope"},
            {"aligned_substitution_only": False, "loss_token_scope": "mismatched_only"},
            {"train_scenarios": ["clean", "noise", "reverb", "far_field"], "exclude_clean_identical_rejected": True},
            {"val_scenarios": ["noise", "reverb", "far_field"]},
            {"ref_margin_min": 0.0, "ref_margin_max": -1.0},
            {"ref_margin_min": -2.0, "ref_margin_max": -2.0},
        ):
            raw = yaml.safe_load(ALIGNED_CONFIG.read_text(encoding="utf-8"))
            raw["dpo"].update(bad_patch)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "aligned.yaml"
                path.write_text(yaml.safe_dump(raw), encoding="utf-8")
                with self.assertRaises(ValueError):
                    load_dpo_config(path)
        base_row = {
            "schema_version": "qwen3-asr-dpo-pair-v1",
            "pair_id": "ok",
            "source_utterance_id": "utt_ok",
            "split": "train",
            "scenario": "noise",
            "reference_text": "国家级贫困县",
            "chosen_text": "国家级贫困县",
            "rejected_text": "加家级贫困县",
            "edit_count": 1,
            "ref_chosen_logp": -5.5,
            "ref_rejected_logp": -3.5,
            "chosen_token_count": 7,
            "rejected_token_count": 7,
            "char_length_delta": 0,
        }
        clean_row = dict(
            base_row,
            pair_id="clean_dup",
            source_utterance_id="utt_dup",
            scenario="clean",
            reference_text="松江新兵镇",
            chosen_text="松江新兵镇",
            rejected_text="松江新浜镇",
        )
        degraded_dup = dict(
            clean_row,
            pair_id="noise_dup",
            scenario="noise",
        )
        candidates = [
            base_row,
            clean_row,
            degraded_dup,
            dict(base_row, pair_id="two_edits", rejected_text="加家局贫困县", edit_count=2),
            dict(base_row, pair_id="punct", rejected_text="，家级贫困县"),
            dict(base_row, pair_id="tok_mismatch", rejected_token_count=8),
            dict(base_row, pair_id="indel_shift", chosen_text="国家级贫困", rejected_text="家级贫困县"),
            dict(base_row, pair_id="too_low_margin", ref_chosen_logp=-10.0, ref_rejected_logp=-3.5),
            dict(base_row, pair_id="positive_margin", ref_chosen_logp=-2.0, ref_rejected_logp=-3.5),
            dict(base_row, pair_id="dropout_row", scenario="dropout"),
        ]
        filtered = select_training_pairs(
            candidates,
            aligned["dpo"]["train_scenarios"],
            aligned_substitution_only=True,
            max_edit_count=aligned["dpo"]["max_edit_count"],
            exclude_clean_identical_rejected=True,
            clean_rejected_by_utterance=build_clean_rejected_map(candidates),
            ref_margin_min=-4.0,
            ref_margin_max=0.0,
        )
        self.assertEqual([item["pair_id"] for item in filtered], ["ok"])
        local_train = ROOT / "results/phase3_dpo_gap_strip_logprob_20261018/train_pairs.jsonl"
        local_val = ROOT / "results/phase3_dpo_gap_strip_logprob_20261018/val_pairs.jsonl"
        if local_train.is_file() and local_val.is_file():
            train_rows = [json.loads(line) for line in local_train.read_text(encoding="utf-8").splitlines() if line.strip()]
            val_rows = [json.loads(line) for line in local_val.read_text(encoding="utf-8").splitlines() if line.strip()]
            kept_train = select_training_pairs(
                train_rows,
                aligned["dpo"]["train_scenarios"],
                aligned_substitution_only=aligned["dpo"]["aligned_substitution_only"],
                max_edit_count=aligned["dpo"]["max_edit_count"],
                exclude_clean_identical_rejected=aligned["dpo"]["exclude_clean_identical_rejected"],
                clean_rejected_by_utterance=build_clean_rejected_map(train_rows),
                ref_margin_min=aligned["dpo"]["ref_margin_min"],
                ref_margin_max=aligned["dpo"]["ref_margin_max"],
            )
            kept_val = select_training_pairs(
                val_rows,
                aligned["dpo"]["val_scenarios"],
                aligned_substitution_only=aligned["dpo"]["aligned_substitution_only"],
                ref_margin_min=aligned["dpo"]["ref_margin_min"],
                ref_margin_max=aligned["dpo"]["ref_margin_max"],
            )
            self.assertEqual(len(kept_train), 164)
            self.assertEqual(len(kept_val), 118)
            train_counts = validate_pair_rows(
                kept_train,
                split="train",
                min_per_scenario=aligned["dpo"]["min_pairs_per_scenario"],
                required_scenarios=tuple(aligned["dpo"]["train_scenarios"]),
            )
            val_counts = validate_pair_rows(
                kept_val,
                split="val",
                min_per_scenario=aligned["dpo"]["min_pairs_per_scenario"],
                required_scenarios=tuple(aligned["dpo"]["val_scenarios"]),
            )
            self.assertEqual(
                {k: train_counts[k] for k in aligned["dpo"]["train_scenarios"]},
                {"noise": 118, "reverb": 21, "far_field": 25},
            )
            self.assertEqual(
                {k: val_counts[k] for k in aligned["dpo"]["val_scenarios"]},
                {"clean": 26, "noise": 29, "reverb": 33, "far_field": 30},
            )

    def test_mismatched_token_mask_and_straight_through_gradient_isolate_substitution_token(self) -> None:
        from train.dpo_contract import mismatched_token_mask
        from train.train_dpo import _supervised_label_list

        self.assertEqual(
            mismatched_token_mask([10, 20, 30, 99], [10, 25, 30, 99]),
            [False, True, False, False],
        )
        self.assertEqual(
            mismatched_token_mask([10, 20], [10, 20]),
            [True, True],
        )
        with self.assertRaises(ValueError):
            mismatched_token_mask([], [])
        with self.assertRaises(ValueError):
            mismatched_token_mask([1, 2], [1])

        class _FakeSlice:
            def __init__(self, values: list[int]) -> None:
                self._values = values

            def reshape(self, _shape: int) -> "_FakeSlice":
                return self

            def tolist(self) -> list[int]:
                return list(self._values)

        class _FakeLabelTensor:
            def __init__(self, row: list[int]) -> None:
                self._row = row

            def __getitem__(self, idx):
                _, slc = idx
                return _FakeSlice(self._row[slc])

        chosen_sup = _supervised_label_list(_FakeLabelTensor([-100, -100, 1, 2, 3, 4]))
        rejected_sup = _supervised_label_list(_FakeLabelTensor([-100, -100, 1, 5, 3, 4]))
        self.assertEqual(chosen_sup, [1, 2, 3, 4])
        self.assertEqual(rejected_sup, [1, 5, 3, 4])
        self.assertEqual(mismatched_token_mask(chosen_sup, rejected_sup), [False, True, False, False])

        source = (ROOT / "train" / "train_dpo.py").read_text(encoding="utf-8")
        self.assertIn("diff_nll = sup_nll[token_mask].sum()", source)
        self.assertIn("return diff_nll + (full_nll - diff_nll).detach(), count", source)
        self.assertIn('loss_token_scope = str(config["dpo"].get("loss_token_scope", "sequence"))', source)
        self.assertIn('"max_edit_count": max_edit_count', source)

    def test_cache_check_stays_ahead_of_the_scenario_filter(self) -> None:
        source = (ROOT / "train" / "train_dpo.py").read_text(encoding="utf-8")
        self.assertLess(
            source.index("_check_cache(cache_path, pair_train, pair_val)"),
            source.index("select_training_pairs(pair_train, scenarios"),
        )
        self.assertIn("iterate_updates(\n                train_pair_rows,", source)
        loop = source[source.index("for update in training_updates") :]
        self.assertLess(loop.index("scheduler.step()"), loop.index("optimizer.step()"))
        self.assertIn("or ends_schedule", source)
        self.assertIn("schedule_end_step=schedule_end_step", source)
        self.assertIn("enumerate(pref_rows, start=1)", source)
        self.assertIn("select_recheck(pair_val, int(config[\"seed\"]), 20)", source)
        self.assertIn('(run_dir / "gate.json", "gate.json")', source)
        self.assertIn('(run_dir / "listen_waiver.json", "listen_waiver.json")', source)


class DropoutLocateTest(unittest.TestCase):
    def test_signature_thresholds_are_strict_where_the_contract_says_so(self) -> None:
        self.assertEqual(
            classify_dropout_signature(
                gap_insertions=GAP_INSERTION_MIN,
                dropout_length_ratio=EARLY_DELETE_LENGTH_RATIO,
                dropout_delta=0.01,
                clean_delta=EARLY_DELETE_CLEAN_DELTA,
            ),
            "GAP_INSERT",
        )
        self.assertEqual(
            classify_dropout_signature(
                gap_insertions=GAP_INSERTION_MIN - 1,
                dropout_length_ratio=EARLY_DELETE_LENGTH_RATIO,
                dropout_delta=0.0,
                clean_delta=0.0,
            ),
            "QUIET",
        )
        self.assertEqual(
            classify_dropout_signature(
                gap_insertions=0,
                dropout_length_ratio=EARLY_DELETE_LENGTH_RATIO - 0.01,
                dropout_delta=0.0,
                clean_delta=0.0,
            ),
            "EARLY_DELETE",
        )
        self.assertEqual(
            classify_dropout_signature(
                gap_insertions=GAP_INSERTION_MIN,
                dropout_length_ratio=1.0,
                dropout_delta=0.0,
                clean_delta=EARLY_DELETE_CLEAN_DELTA + 0.001,
            ),
            "MIXED",
        )
        self.assertEqual(
            classify_dropout_signature(
                gap_insertions=0,
                dropout_length_ratio=0.9,
                dropout_delta=QUIET_DROPOUT_ABS_DELTA,
                clean_delta=0.0,
            ),
            "OTHER",
        )

    def test_scored_rows_separate_gap_fill_from_early_delete(self) -> None:
        reference = "0123456789"
        parent = _metrics({name: 0.0 for name in ("clean", "dropout", "far_field", "noise", "reverb")})
        gap_rows = [{"scenario": "dropout", "reference_normalized": reference, "scored_prediction_normalized": "0123" + ("X" * 30) + "456789"}]
        gap = locate_dropout_edits(gap_rows, _metrics({"clean": 0.0, "dropout": 0.2, "far_field": 0.0, "noise": 0.0, "reverb": 0.0}), parent)
        self.assertEqual(gap["locate_signature"], "GAP_INSERT")
        self.assertEqual(gap["locate_gap_insertions"], 30)
        short_rows = [{"scenario": "dropout", "reference_normalized": reference, "scored_prediction_normalized": "012"}]
        early = locate_dropout_edits(short_rows, _metrics({"clean": 0.0, "dropout": 0.7, "far_field": 0.0, "noise": 0.0, "reverb": 0.0}), parent)
        self.assertEqual(early["locate_signature"], "EARLY_DELETE")
        self.assertLess(early["locate_dropout_length_ratio"], 0.75)
        quiet_rows = [{"scenario": "dropout", "reference_normalized": reference, "scored_prediction_normalized": reference}]
        quiet = locate_dropout_edits(quiet_rows, parent, parent)
        self.assertEqual(quiet["locate_signature"], "QUIET")
        mixed = locate_dropout_edits(
            gap_rows,
            _metrics({"clean": 0.01, "dropout": 0.2, "far_field": 0.0, "noise": 0.0, "reverb": 0.0}),
            parent,
        )
        self.assertEqual(mixed["locate_signature"], "MIXED")
        with self.assertRaises(ValueError):
            locate_dropout_edits([], parent, parent)


class DropoutGapStripTest(unittest.TestCase):
    def test_gap_deletion_becomes_equal_and_outside_deletion_stays(self) -> None:
        reference = "0123456789ABCDEFGHIJ"
        gap_only = reference[:8] + reference[12:]
        self.assertEqual(strip_dropout_gap(reference, gap_only), (gap_only, gap_only))
        low_edge = reference[:8] + reference[9:]
        self.assertEqual(strip_dropout_gap(reference, low_edge)[0], strip_dropout_gap(reference, low_edge)[1])
        high_edge = reference[:11] + reference[12:]
        self.assertEqual(strip_dropout_gap(reference, high_edge)[0], strip_dropout_gap(reference, high_edge)[1])
        after = reference[:12] + reference[13:]
        left, right = strip_dropout_gap(reference, after)
        self.assertNotEqual(left, right)
        self.assertIn(reference[12], left)
        self.assertNotIn(reference[12], right)

    def test_gap_insertion_drops_and_tail_insertion_stays(self) -> None:
        reference = "0123456789"
        inside = reference[:4] + "X" + reference[4:]
        self.assertEqual(strip_dropout_gap(reference, inside)[0], strip_dropout_gap(reference, inside)[1])
        tail = reference + "Z"
        left, right = strip_dropout_gap(reference, tail)
        self.assertEqual(right, left + "Z")

    def test_outside_substitution_stays(self) -> None:
        reference = "0123456789"
        inside = "0123X56789"
        self.assertEqual(strip_dropout_gap(reference, inside)[0], strip_dropout_gap(reference, inside)[1])
        outside = "012345X789"
        left, right = strip_dropout_gap(reference, outside)
        self.assertNotEqual(left, right)
        self.assertIn("X", right)

    def test_empty_reference_fails_and_the_trainer_does_not_call_it(self) -> None:
        with self.assertRaises(ValueError):
            strip_dropout_gap("", "a")
        source = (ROOT / "train" / "train_dpo.py").read_text(encoding="utf-8")
        self.assertNotIn("strip_dropout_gap", source)


if __name__ == "__main__":
    unittest.main()
