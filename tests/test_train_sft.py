"""Local SFT contract tests. They do not download Qwen or start an optimizer."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from train.sft_checkpoint import (
    commit_checkpoint,
    read_latest,
    verify_checkpoint,
    write_checksums,
    write_latest,
)
from train.sft_contract import (
    SCENARIOS,
    assert_shift_supervises_transcript,
    assert_trainable_names,
    build_smoke_subset,
    is_decoder_attention_module,
    is_trainable_lora_parameter,
    iterate_updates,
    judge_sft_checkpoint,
    lr_multiplier,
    update_ends_schedule,
    mask_labels,
    mean_of_mean_losses,
    partial_decode_stop_record,
    pending_checkpoint_eval,
    processor_tree_sha256,
    runtime_budget_disabled,
    runtime_budget_exceeded,
    select_sft_checkpoint,
    should_interrupt,
    token_weighted_loss,
)
from train.sft_schema import load_train_config, semantic_config_sha256, sha256_file


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "train" / "sft_smoke.yaml"
PILOT_CONFIG = ROOT / "configs" / "train" / "sft_pilot.yaml"
LR_CONFIG = ROOT / "configs" / "train" / "sft_pilot_lr1e5.yaml"
PROCESSOR_SHA = "1b94f6ef846ce0faa88927fd8104c27831c457a69f103e6ec62fa392d315856c"


def _metrics_like(baseline: dict, delta: float, improved: tuple[str, ...], clean_delta: float = 0.0) -> dict:
    by_scenario = {}
    for name, row in baseline["by_scenario"].items():
        change = clean_delta if name == "clean" else delta
        if name in improved:
            change = -0.001
        by_scenario[name] = {
            "samples": row["samples"],
            "micro_cer": row["micro_cer"] + change,
            "empty_rate": 0.0,
            "repeated_rate": 0.0,
            "long_rate": 0.0,
            "truncation_rate": 0.0,
            "valid_output_rate": 1.0,
        }
    return {
        "samples": 2500,
        "empty_rate": 0.0,
        "repeated_rate": 0.0,
        "long_rate": 0.0,
        "truncation_rate": 0.0,
        "valid_output_rate": 1.0,
        "by_scenario": by_scenario,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def source_rows(source_id: str, split: str, scenarios: tuple[str, ...] = SCENARIOS) -> list[dict]:
    return [
        {
            "sample_id": f"{source_id}__{scenario}",
            "source_utterance_id": source_id,
            "audio": f"wavs/{source_id}__{scenario}.wav",
            "text": "今天天气不错",
            "language": "zh",
            "scenario": scenario,
            "split": split,
        }
        for scenario in scenarios
    ]


class SftContractTest(unittest.TestCase):
    def test_smoke_config_loads_and_hashes_are_stable(self) -> None:
        config = load_train_config(SMOKE_CONFIG)
        self.assertEqual(config["profile"], "smoke")
        self.assertEqual(config["train"]["max_steps"], 6)
        self.assertEqual(config["train"]["interrupt_after_step"], 5)
        self.assertEqual(config["batch"]["gradient_accumulation_steps"], 8)
        self.assertIsNone(config["scoring"]["wer_tokenizer"])
        self.assertEqual(config["model"]["processor_sha256"], PROCESSOR_SHA)
        self.assertEqual(config["runtime"]["environment_lock_sha256"], sha256_file(ROOT / "requirements.lock"))
        manifests = {"train": "a" * 64, "val": "b" * 64, "metadata": "c" * 64}
        self.assertEqual(semantic_config_sha256(config, manifests), semantic_config_sha256(config, manifests))
        self.assertIsNone(config["baseline"])

    def test_pilot_config_pins_the_2500_row_base(self) -> None:
        config = load_train_config(PILOT_CONFIG)
        self.assertEqual(config["profile"], "pilot")
        self.assertEqual(config["train"]["max_steps"], 100)
        self.assertEqual(config["train"]["max_epochs"], 1)
        self.assertEqual(config["train"]["save_steps"], 10)
        self.assertEqual(config["train"]["eval_steps"], 10)
        self.assertEqual(config["train"]["keep_last"], 10)
        self.assertIsNone(config["train"]["interrupt_after_step"])
        self.assertEqual(config["baseline"]["manifest_sha256"], "2503945d8b1adbc3f0070a24fe33e5cb8b98e0248c68c550d59955bff33c7de8")
        self.assertEqual(config["decode_eval_sha256"], config["baseline"]["decode_eval_sha256"])
        self.assertEqual(config["baseline"]["by_scenario"]["clean"]["samples"], 500)
        self.assertEqual(config["optim"]["learning_rate"], 0.0001)

    def test_lower_lr_followup_keeps_the_pilot_decode(self) -> None:
        pilot = load_train_config(PILOT_CONFIG)
        follow = load_train_config(LR_CONFIG)
        manifests = {"train": "a" * 64, "val": "b" * 64, "metadata": "c" * 64}
        self.assertEqual(follow["profile"], "pilot")
        self.assertEqual(follow["optim"]["learning_rate"], 1e-5)
        self.assertEqual(follow["train"]["max_steps"], 100)
        self.assertEqual(follow["train"]["keep_last"], 10)
        self.assertIsNone(follow["train"]["interrupt_after_step"])
        self.assertEqual(follow["decode"]["eval"]["max_new_tokens"], 513)
        self.assertEqual(follow["decode"]["eval"]["eos_policy"], "tokenizer_eos_only")
        self.assertEqual(follow["decode_eval_sha256"], pilot["decode_eval_sha256"])
        self.assertEqual(follow["decode_eval_sha256"], follow["baseline"]["decode_eval_sha256"])
        self.assertNotEqual(
            semantic_config_sha256(follow, manifests),
            semantic_config_sha256(pilot, manifests),
        )

    def test_sft_gate_stops_on_regression_and_selects_the_earlier_pass(self) -> None:
        config = load_train_config(PILOT_CONFIG)
        baseline = config["baseline"]
        gate = config["gate"]
        healthy = _metrics_like(baseline, delta=0.0, improved=("dropout",))
        decision = judge_sft_checkpoint(healthy, baseline, gate)
        self.assertTrue(decision["passed"])
        self.assertFalse(decision["risk_stop"])
        flat = _metrics_like(baseline, delta=0.0, improved=())
        flat_decision = judge_sft_checkpoint(flat, baseline, gate)
        self.assertFalse(flat_decision["passed"])
        self.assertFalse(flat_decision["risk_stop"])
        regressed = _metrics_like(baseline, delta=0.0, improved=(), clean_delta=0.0100000001)
        regressed_decision = judge_sft_checkpoint(regressed, baseline, gate)
        self.assertTrue(regressed_decision["risk_stop"])
        exact = _metrics_like(baseline, delta=0.0, improved=("noise",), clean_delta=0.01)
        self.assertFalse(judge_sft_checkpoint(exact, baseline, gate)["risk_stop"])
        first = dict(decision, step=10, degraded_macro_cer=0.2, clean_cer=0.03)
        second = dict(decision, step=20, degraded_macro_cer=0.2, clean_cer=0.02)
        third = dict(decision, step=30, degraded_macro_cer=0.1, clean_cer=0.05, passed=False)
        self.assertEqual(select_sft_checkpoint([first, second, third])["step"], 20)

    def test_runtime_budget_stops_during_eval_without_a_gate(self) -> None:
        self.assertFalse(runtime_budget_exceeded(110 * 60, 120, 10))
        self.assertTrue(runtime_budget_exceeded(110 * 60 + 0.001, 120, 10))
        self.assertFalse(runtime_budget_exceeded(0, 120, 10))
        self.assertTrue(runtime_budget_disabled("1"))
        self.assertFalse(runtime_budget_disabled(None))
        self.assertFalse(runtime_budget_disabled("0"))
        with self.assertRaises(ValueError):
            runtime_budget_exceeded(-1, 120, 10)
        self.assertTrue(pending_checkpoint_eval(30, 10, [10, 20]))
        self.assertFalse(pending_checkpoint_eval(30, 10, [10, 20, 30]))
        self.assertFalse(pending_checkpoint_eval(8, 10, []))
        self.assertFalse(pending_checkpoint_eval(0, 10, []))
        self.assertTrue(pending_checkpoint_eval(41, 10, [10, 20, 30, 40], schedule_end_step=41))
        self.assertFalse(pending_checkpoint_eval(41, 10, [10, 20, 30, 40, 41], schedule_end_step=41))
        self.assertFalse(pending_checkpoint_eval(41, 10, [], schedule_end_step=40))
        record = partial_decode_stop_record(30, 2500, [8, 513, 12], 513)
        self.assertEqual(record["schema_version"], "qwen3-asr-eval-runtime-stop-v1")
        self.assertFalse(record["gate"])
        self.assertNotIn("passed", record)
        self.assertEqual(record["completed_samples"], 3)
        self.assertEqual(record["remaining_samples"], 2497)
        self.assertEqual(record["mean_new_tokens"], (8 + 513 + 12) / 3)
        self.assertEqual(record["max_new_tokens_seen"], 513)
        self.assertEqual(record["hit_max_new_tokens"], 1)
        self.assertIsNone(select_sft_checkpoint([record]))
        with self.assertRaises(ValueError):
            partial_decode_stop_record(30, 2, [1, 1, 1], 513)

    def test_unknown_field_is_rejected_before_a_run_directory(self) -> None:
        with TemporaryDirectory() as temporary:
            config = Path(temporary) / "bad.yaml"
            config.write_text("schema_version: qwen3-asr-train-v1\nextra_field: 1\n", encoding="utf-8")
            output = Path(temporary) / "run"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "train" / "train_sft.py"),
                    "--config",
                    str(config),
                    "--output-dir",
                    str(output),
                    "--run-id",
                    "20261015T000000_sft_06b_zh_seed42",
                    "--seed",
                    "42",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("unknown", result.stderr)
            self.assertFalse(output.exists())

    def test_seed_mismatch_exits_before_creating_a_run(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "run"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "train" / "train_sft.py"),
                    "--config",
                    str(SMOKE_CONFIG),
                    "--output-dir",
                    str(output),
                    "--run-id",
                    "20261015T000000_sft_06b_zh_seed7",
                    "--seed",
                    "7",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("seed mismatch", result.stderr)
            self.assertFalse(output.exists())

    def test_pilot_interrupt_is_rejected(self) -> None:
        raw = yaml.safe_load(SMOKE_CONFIG.read_text(encoding="utf-8"))
        raw["profile"] = "pilot"
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "pilot.yaml"
            path.write_text(yaml.safe_dump(raw), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_train_config(path)

    def test_decoder_lora_names_reject_the_audio_tower(self) -> None:
        good = "thinker.model.layers.0.self_attn.q_proj"
        prefixed = "model.thinker.model.layers.3.self_attn.o_proj"
        audio = "thinker.audio_tower.layers.0.self_attn.q_proj"
        self.assertTrue(is_decoder_attention_module(good))
        self.assertTrue(is_decoder_attention_module(prefixed))
        self.assertFalse(is_decoder_attention_module(audio))
        self.assertFalse(is_decoder_attention_module("q_proj"))
        self.assertFalse(is_decoder_attention_module("thinker.model.embed_tokens"))
        self.assertFalse(is_decoder_attention_module("thinker.lm_head"))
        lora_name = good + ".lora_A.default.weight"
        self.assertTrue(is_trainable_lora_parameter(lora_name))
        self.assertFalse(is_trainable_lora_parameter(audio + ".lora_B.weight"))
        self.assertEqual(assert_trainable_names([lora_name, good + ".lora_B.default.weight"]), [lora_name, good + ".lora_B.default.weight"])
        with self.assertRaises(ValueError):
            assert_trainable_names([lora_name, "thinker.lm_head.weight"])

    def test_eos_stays_supervised_when_pad_id_matches_eos(self) -> None:
        eos_id = 2
        input_ids = [10, 11, 12, 7, 8, eos_id, eos_id, eos_id]
        attention = [1, 1, 1, 1, 1, 1, 0, 0]
        labels = mask_labels(input_ids, attention, prefix_len=3)
        self.assertEqual(labels, [-100, -100, -100, 7, 8, eos_id, -100, -100])
        assert_shift_supervises_transcript(labels, 3, eos_id)

    def test_token_weighted_loss_is_not_the_mean_of_means(self) -> None:
        groups = [[0.0, 2.0], [3.0]]
        self.assertAlmostEqual(token_weighted_loss(groups), 5.0 / 3.0)
        self.assertAlmostEqual(mean_of_mean_losses(groups), 2.0)
        self.assertNotAlmostEqual(token_weighted_loss(groups), mean_of_mean_losses(groups))

    def test_scheduler_resume_does_not_restart_warmup(self) -> None:
        self.assertAlmostEqual(lr_multiplier(1, 2, 6, 0.1), 0.5)
        self.assertAlmostEqual(lr_multiplier(2, 2, 6, 0.1), 1.0)
        self.assertAlmostEqual(lr_multiplier(4, 2, 6, 0.1), 0.55)
        self.assertAlmostEqual(lr_multiplier(6, 2, 6, 0.1), 0.1)
        self.assertGreater(lr_multiplier(5, 2, 6, 0.1), 0.1)

    def test_epoch_tail_uses_its_own_window(self) -> None:
        rows = [{"sample_id": f"s{index}"} for index in range(10)]
        full = [len(update["rows"]) for update in iterate_updates(rows, seed=42, accumulation=8, max_steps=6, max_epochs=100)]
        self.assertEqual(full, [8, 2, 8, 2, 8, 2])
        one_epoch = [len(update["rows"]) for update in iterate_updates(rows, seed=42, accumulation=8, max_steps=6, max_epochs=1)]
        self.assertEqual(one_epoch, [8, 2])
        aligned = list(
            iterate_updates(
                [{"sample_id": f"s{index}"} for index in range(164)],
                seed=42,
                accumulation=4,
                max_steps=100,
                max_epochs=1,
            )
        )
        self.assertEqual(len(aligned), 41)
        self.assertEqual(aligned[-1]["global_step"], 41)
        self.assertEqual(len(aligned[-1]["rows"]), 4)
        self.assertTrue(
            update_ends_schedule(aligned[-1], row_count=164, max_steps=100, max_epochs=1)
        )
        self.assertFalse(
            update_ends_schedule(aligned[-2], row_count=164, max_steps=100, max_epochs=1)
        )
        self.assertTrue(
            update_ends_schedule(
                {"global_step": 2, "epoch": 0, "cursor_after": 10},
                row_count=10,
                max_steps=6,
                max_epochs=1,
            )
        )
        self.assertTrue(
            update_ends_schedule(
                {"global_step": 6, "epoch": 0, "cursor_after": 48},
                row_count=50,
                max_steps=6,
                max_epochs=100,
            )
        )
        self.assertAlmostEqual(lr_multiplier(1, 10, 100, 0.1), 0.1)
        self.assertAlmostEqual(lr_multiplier(10, 10, 100, 0.1), 1.0)
        smoke = [len(update["rows"]) for update in iterate_updates(
            [{"sample_id": f"s{index}"} for index in range(50)],
            seed=42,
            accumulation=8,
            max_steps=6,
            max_epochs=100,
        )]
        self.assertEqual(smoke, [8, 8, 8, 8, 8, 8])

    def test_interrupt_fires_once(self) -> None:
        self.assertTrue(should_interrupt(5, 5, False))
        self.assertFalse(should_interrupt(5, 5, True))
        self.assertFalse(should_interrupt(6, 5, False))

    def test_latest_rejects_a_tampered_or_incoming_checkpoint(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / "staging"
            staging.mkdir()
            (staging / "note.txt").write_text("ok", encoding="utf-8")
            digest = write_checksums(staging, ["note.txt"], "2026-10-15T00:00:00Z")
            run = root / "run"
            run.mkdir()
            committed = commit_checkpoint(staging, run, 5)
            self.assertEqual(committed, digest)
            write_latest(run / "latest.json", step=5, checkpoint_sha256=digest)
            latest = read_latest(run / "latest.json")
            verify_checkpoint(run / latest["relative_path"], digest)
            note = run / "checkpoints" / "step_000005" / "note.txt"
            note.write_text("changed", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_checkpoint(run / latest["relative_path"], digest)
            incoming_latest = run / "incoming-latest.json"
            incoming_latest.write_text(json.dumps({
                "schema_version": "qwen3-asr-latest-v1",
                "step_name": "step_000005",
                "relative_path": "checkpoints/.incoming/step_000005",
                "checkpoint_sha256": digest,
                "updated_at": "2026-10-15T00:00:00Z",
            }), encoding="utf-8")
            with self.assertRaises(ValueError):
                read_latest(incoming_latest)

    def test_smoke_subset_keeps_five_scenarios_and_is_not_publishable(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = [row for index in range(3) for row in source_rows(f"u{index}", "train")]
            train.extend(source_rows("short", "train", ("clean", "noise")))
            val = [row for index in range(2) for row in source_rows(f"v{index}", "val")]
            train_path = root / "train.jsonl"
            val_path = root / "val.jsonl"
            write_jsonl(train_path, train)
            write_jsonl(val_path, val)
            output = root / "smoke"
            metadata = build_smoke_subset(train_path, val_path, output, seed=42, train_sources=2, val_sources=1, copy_audio=False)
            self.assertFalse(metadata["publishable"])
            self.assertEqual(len(metadata["train_sources"]), 2)
            self.assertNotIn("short", metadata["train_sources"])
            train_out = [json.loads(line) for line in (output / "train.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(train_out), 10)
            self.assertEqual(sorted({row["scenario"] for row in train_out}), sorted(SCENARIOS))
            again = build_smoke_subset(train_path, val_path, root / "smoke2", seed=42, train_sources=2, val_sources=1, copy_audio=False)
            self.assertEqual(again["train_sources"], metadata["train_sources"])

    def test_processor_file_list_hash_is_stable(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("chat_template.json", "merges.txt", "preprocessor_config.json", "tokenizer_config.json", "vocab.json"):
                (root / name).write_text(name, encoding="utf-8")
            self.assertEqual(processor_tree_sha256(root), processor_tree_sha256(root))


if __name__ == "__main__":
    unittest.main()
