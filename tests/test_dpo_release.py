"""DPO release contract tests. They do not download Qwen or use a GPU."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from train.dpo_release import (
    compare_dpo_greedy_texts,
    load_dpo_release,
    load_published_dpo_release,
    mark_dpo_published,
    prepare_dpo_release,
)
from train.dpo_schema import load_dpo_config, semantic_dpo_config_sha256
from train.sft_checkpoint import utc_now, write_checksums, write_latest
from train.sft_release import greedy_fields

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "train" / "dpo_pilot_edit1_anchor_step40_h100.yaml"
RUN_NAME = "20261022T133007_dpo_06b_zh_seed42"
RELEASE_ID = "dpo_06b_zh_20261022T133007_step000040"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def scored_rows() -> list[dict]:
    return [
        {"sample_id": "a__clean", "prediction_raw": "甲", "stop_reason": "eos", "error": None},
        {"sample_id": "a__dropout", "prediction_raw": "乙", "stop_reason": "eos", "error": None},
    ]


def build_dpo_run(
    root: Path,
    *,
    passed: bool = True,
    pref_acc: float = 0.6440677966101694,
    latest_step: int = 40,
) -> Path:
    config = load_dpo_config(CONFIG)
    manifest_sha = {
        "train": "a" * 64,
        "val": "b" * 64,
        "metadata": "c" * 64,
        "pair_metadata": "d" * 64,
        "filled_train_pairs": config["dpo"]["filled_train_pairs_sha256"],
        "filled_val_pairs": config["dpo"]["filled_val_pairs_sha256"],
        "reference_cache": config["dpo"]["reference_cache_sha256"],
    }
    run = root / RUN_NAME
    checkpoint = run / "checkpoints" / "step_000040"
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "adapter" / "adapter.safetensors").write_bytes(b"selected-dpo-adapter")
    (checkpoint / "adapter" / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    state = {
        "schema_version": "qwen3-asr-training-state-v1",
        "stage": "dpo",
        "global_step": 40,
        "eval_decode_sha256": config["decode_eval_sha256"],
        "parent_release_sha256": config["model"]["parent_release_sha256"],
        "config_sha256": semantic_dpo_config_sha256(config, manifest_sha),
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
        "step": 40,
        "passed": passed,
        "risk_stop": not passed,
        "preference_accuracy": pref_acc,
        "preference_ties": 0,
        "raw_policy_accuracy": 0.025423728813559324,
        "length_buckets": {"equal_length": {"pairs": 118, "raw_policy_accuracy": 0.025423728813559324}},
    }
    gate = {
        "schema_version": "qwen3-asr-gate-v1",
        "status": "passed" if passed else "failed",
        "publishable": False,
        "audit_disposition": "unlistened",
        "selected_step": 40 if passed else None,
        "decisions": [decision],
    }
    (run / "gate.json").write_text(json.dumps(gate), encoding="utf-8")
    write_jsonl(run / "eval_decisions.jsonl", [decision])
    write_jsonl(run / "eval" / "step_000040" / "scored.jsonl", scored_rows())
    (run / "eval" / "step_000040" / "preference.json").write_text(
        json.dumps({"accuracy": pref_acc, "pairs": 118, "wins": 76, "ties": 0}),
        encoding="utf-8",
    )
    write_latest(run / "latest.json", step=latest_step, checkpoint_sha256="e" * 64)
    return run


class DpoReleaseTest(unittest.TestCase):
    def test_prepare_copies_selected_dpo_adapter_and_syncs_drive(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = build_dpo_run(root)
            output = root / "releases" / RELEASE_ID
            drive = root / "drive_releases" / RELEASE_ID
            release = prepare_dpo_release(
                config_path=CONFIG,
                run_dir=run,
                output_dir=output,
                release_id=RELEASE_ID,
                sync_dir=drive,
            )
            self.assertEqual(release["status"], "staged")
            self.assertFalse(release["publishable"])
            self.assertEqual(release["stage"], "dpo")
            self.assertEqual(release["source_step"], 40)
            self.assertEqual(release["latest_step"], 40)
            self.assertTrue(release["latest_is_release"])
            self.assertEqual(release["parent_release"], "sft_06b_zh_20261015T234939_step000030")
            self.assertEqual(release["audit_disposition"], "unlistened")
            self.assertEqual((output / "adapter" / "adapter.safetensors").read_bytes(), b"selected-dpo-adapter")
            self.assertFalse((output / "optimizer.pt").exists())
            self.assertEqual(load_dpo_release(output), load_dpo_release(drive))
            with self.assertRaises(ValueError):
                load_published_dpo_release(output)
            with self.assertRaises(ValueError):
                prepare_dpo_release(
                    config_path=CONFIG,
                    run_dir=run,
                    output_dir=output,
                    release_id=RELEASE_ID,
                )
            (run / "eval" / "step_000040" / "preference.json").unlink()
            output_fallback = root / "releases_fallback" / RELEASE_ID
            fallback_rel = prepare_dpo_release(
                config_path=CONFIG,
                run_dir=run,
                output_dir=output_fallback,
                release_id=RELEASE_ID,
            )
            self.assertAlmostEqual(fallback_rel["preference_accuracy"], 0.6440677966101694)
            self.assertTrue((output_fallback / "source" / "preference.json").is_file())

    def test_risk_stop_or_low_preference_is_not_releasable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_failed = build_dpo_run(root / "failed", passed=False)
            with self.assertRaises(ValueError):
                prepare_dpo_release(
                    config_path=CONFIG,
                    run_dir=run_failed,
                    output_dir=root / "out1",
                    release_id=RELEASE_ID,
                )
            run_low_pref = build_dpo_run(root / "low_pref", passed=True, pref_acc=0.5)
            with self.assertRaises(ValueError):
                prepare_dpo_release(
                    config_path=CONFIG,
                    run_dir=run_low_pref,
                    output_dir=root / "out2",
                    release_id=RELEASE_ID,
                )

    def test_text_mismatch_stays_unpublished_and_matching_publishes(self) -> None:
        saved = greedy_fields(scored_rows(), "saved")
        bad_fresh = greedy_fields(
            [
                {"sample_id": "a__clean", "prediction_raw": "甲", "stop_reason": "eos", "error": None},
                {"sample_id": "a__dropout", "prediction_raw": "丙", "stop_reason": "eos", "error": None},
            ],
            "fresh",
        )
        bad_report = compare_dpo_greedy_texts(saved, bad_fresh)
        self.assertFalse(bad_report["matched"])
        good_report = compare_dpo_greedy_texts(saved, saved)
        good_report["release_id"] = RELEASE_ID
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "releases" / RELEASE_ID
            prepare_dpo_release(
                config_path=CONFIG,
                run_dir=build_dpo_run(root),
                output_dir=output,
                release_id=RELEASE_ID,
            )
            with self.assertRaises(ValueError):
                mark_dpo_published(output, bad_report)
            before = (output / "adapter" / "adapter.safetensors").read_bytes()
            published = mark_dpo_published(output, good_report)
            self.assertEqual(published["status"], "published")
            self.assertTrue(published["publishable"])
            self.assertEqual(published["audit_disposition"], "unlistened")
            self.assertEqual((output / "adapter" / "adapter.safetensors").read_bytes(), before)
            loaded = load_published_dpo_release(output)
            self.assertEqual(loaded["release_id"], RELEASE_ID)


if __name__ == "__main__":
    unittest.main()
