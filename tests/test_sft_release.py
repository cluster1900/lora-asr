"""SFT release contract tests. They do not download Qwen or use a GPU."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from train.sft_checkpoint import utc_now, write_checksums, write_latest
from train.sft_release import (  # noqa: E402
    completed_hypothesis_rows,
    compare_greedy_texts,
    greedy_fields,
    load_release,
    mark_published,
    prepare_release,
    run_hypotheses,
)
from train.sft_schema import load_train_config, semantic_config_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "train" / "sft_pilot_lr1e5.yaml"
RUN_NAME = "20261015T234939_sft_06b_zh_seed42"
RELEASE_ID = "sft_06b_zh_20261015T234939_step000030"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def scored_rows() -> list[dict]:
    return [
        {"sample_id": "a__clean", "prediction_raw": "甲", "stop_reason": "eos", "error": None},
        {"sample_id": "a__dropout", "prediction_raw": "乙", "stop_reason": "eos", "error": None},
    ]


def build_run(root: Path, *, passed: bool = True) -> Path:
    config = load_train_config(CONFIG)
    manifest_sha = {
        "train": "a" * 64,
        "val": "b" * 64,
        "metadata": "c" * 64,
    }
    run = root / RUN_NAME
    checkpoint = run / "checkpoints" / "step_000030"
    (checkpoint / "adapter").mkdir(parents=True)
    (checkpoint / "adapter" / "adapter.safetensors").write_bytes(b"selected-adapter")
    (checkpoint / "adapter" / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    state = {
        "schema_version": "qwen3-asr-training-state-v1",
        "stage": "sft",
        "global_step": 30,
        "eval_decode_sha256": config["decode_eval_sha256"],
        "config_sha256": semantic_config_sha256(config, manifest_sha),
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
        "step": 30,
        "passed": passed,
        "risk_stop": not passed,
    }
    gate = {
        "schema_version": "qwen3-asr-gate-v1",
        "status": "passed" if passed else "gate_failed",
        "publishable": False,
        "selected_step": 30 if passed else None,
        "decisions": [decision],
    }
    (run / "gate.json").write_text(json.dumps(gate), encoding="utf-8")
    write_jsonl(run / "eval_decisions.jsonl", [decision, {"step": 50, "passed": False, "risk_stop": True}])
    write_jsonl(run / "eval" / "step_000030" / "scored.jsonl", scored_rows())
    write_latest(run / "latest.json", step=50, checkpoint_sha256="d" * 64)
    return run


class SftReleaseTest(unittest.TestCase):
    def test_prepare_copies_selected_adapter_and_records_latest_separately(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = build_run(root)
            output = root / "releases" / RELEASE_ID
            release = prepare_release(
                config_path=CONFIG,
                run_dir=run,
                output_dir=output,
                release_id=RELEASE_ID,
            )
            self.assertEqual(release["status"], "staged")
            self.assertFalse(release["publishable"])
            self.assertEqual(release["source_step"], 30)
            self.assertEqual(release["latest_step"], 50)
            self.assertFalse(release["latest_is_release"])
            self.assertIsNone(release["parent_release_sha256"])
            self.assertEqual((output / "adapter" / "adapter.safetensors").read_bytes(), b"selected-adapter")
            self.assertFalse((output / "optimizer.pt").exists())
            loaded = load_release(output)
            self.assertEqual(loaded["adapter_sha256"], release["adapter_sha256"])
            with self.assertRaises(ValueError):
                prepare_release(
                    config_path=CONFIG,
                    run_dir=run,
                    output_dir=output,
                    release_id=RELEASE_ID,
                )

    def test_risk_stop_is_not_releasable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = build_run(root, passed=False)
            with self.assertRaises(ValueError):
                prepare_release(
                    config_path=CONFIG,
                    run_dir=run,
                    output_dir=root / "releases" / RELEASE_ID,
                    release_id=RELEASE_ID,
                )

    def test_text_mismatch_stays_unpublished(self) -> None:
        saved = greedy_fields(scored_rows(), "saved")
        fresh = greedy_fields(
            [
                {"sample_id": "a__clean", "prediction_raw": "甲", "stop_reason": "eos", "error": None},
                {"sample_id": "a__dropout", "prediction_raw": "丙", "stop_reason": "eos", "error": None},
            ],
            "fresh",
        )
        report = compare_greedy_texts(saved, fresh)
        self.assertFalse(report["matched"])
        self.assertEqual(report["mismatch_count"], 1)
        self.assertEqual(report["mismatches"][0]["sample_id"], "a__dropout")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = build_run(root)
            output = root / "releases" / RELEASE_ID
            prepare_release(
                config_path=CONFIG,
                run_dir=run,
                output_dir=output,
                release_id=RELEASE_ID,
            )
            with self.assertRaises(ValueError):
                mark_published(output, report)

    def test_matching_text_can_publish_without_changing_the_adapter(self) -> None:
        saved = greedy_fields(scored_rows(), "saved")
        report = compare_greedy_texts(saved, saved)
        report["release_id"] = RELEASE_ID
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "releases" / RELEASE_ID
            prepare_release(
                config_path=CONFIG,
                run_dir=build_run(root),
                output_dir=output,
                release_id=RELEASE_ID,
            )
            before = (output / "adapter" / "adapter.safetensors").read_bytes()
            published = mark_published(output, report)
            self.assertEqual(published["status"], "published")
            self.assertTrue(published["publishable"])
            self.assertEqual((output / "adapter" / "adapter.safetensors").read_bytes(), before)
            self.assertTrue((output / "readback.json").is_file())

    def test_staged_release_cannot_export_hypotheses(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "releases" / RELEASE_ID
            prepare_release(
                config_path=CONFIG,
                run_dir=build_run(root),
                output_dir=output,
                release_id=RELEASE_ID,
            )
            with self.assertRaises(ValueError):
                run_hypotheses(CONFIG, output, root / "missing.jsonl", root / "out.jsonl")

    def test_hypothesis_resume_must_match_the_manifest_prefix(self) -> None:
        rows = [{"sample_id": "a__clean"}, {"sample_id": "a__dropout"}]
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "hyp.jsonl"
            path.write_text('{"sample_id":"a__dropout"}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                completed_hypothesis_rows(path, rows)
            path.write_text('{"sample_id":"a__clean"}\n', encoding="utf-8")
            self.assertEqual(completed_hypothesis_rows(path, rows), 1)


if __name__ == "__main__":
    unittest.main()
