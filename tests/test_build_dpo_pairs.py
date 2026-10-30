"""Contract tests for DPO pair building. They do not download a model."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.build_dpo_pairs import build_dpo_pairs, char_token_count, main

WAV = b"RIFF" + b"\x24\x00\x00\x00" + b"WAVE" + b"\x00" * 32


def _release(root: Path, *, published: bool = True) -> Path:
    directory = root / "release"
    directory.mkdir()
    payload = {
        "schema_version": "qwen3-asr-sft-release-v1",
        "release_id": "sft_test_step000030",
        "status": "published" if published else "staged",
        "publishable": published,
        "source_step": 30,
        "base_revision": "rev",
        "processor_sha256": "processor",
        "adapter_sha256": "adapter",
    }
    (directory / "release.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def _row(split: str, scenario: str, suffix: str, text: str = "你好世界") -> dict[str, object]:
    sample_id = f"{suffix}__{scenario}"
    return {
        "sample_id": sample_id,
        "source_utterance_id": suffix,
        "audio": f"wavs/{sample_id}.wav",
        "text": text,
        "scenario": scenario,
        "split": split,
    }


def _prediction(row: dict[str, object], text: str, *, error: str | None = None) -> dict[str, object]:
    return {
        "sample_id": row["sample_id"],
        "text": row["text"],
        "prediction_raw": text,
        "stop_reason": "eos",
        "error": error,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _touch_wavs(manifest: Path, rows: list[dict[str, object]]) -> None:
    for row in rows:
        audio = manifest.parent / str(row["audio"])
        audio.parent.mkdir(parents=True, exist_ok=True)
        audio.write_bytes(WAV)


class BuildDpoPairsTest(unittest.TestCase):
    def _build(
        self,
        root: Path,
        train_rows: list[dict[str, object]],
        train_predictions: list[dict[str, object]],
        val_rows: list[dict[str, object]],
        val_predictions: list[dict[str, object]],
        **kwargs: object,
    ) -> dict[str, object]:
        train_manifest = root / "train" / "train.jsonl"
        val_manifest = root / "val" / "val.jsonl"
        _write_jsonl(train_manifest, train_rows)
        _write_jsonl(val_manifest, val_rows)
        _write_jsonl(root / "train_predictions.jsonl", train_predictions)
        _write_jsonl(root / "val_predictions.jsonl", val_predictions)
        _touch_wavs(train_manifest, train_rows)
        _touch_wavs(val_manifest, val_rows)
        options: dict[str, object] = {
            "max_transcript_tokens": 512,
            "min_val_pairs": 1,
            "min_pairs_per_scenario": 0,
        }
        options.update(kwargs)
        return build_dpo_pairs(
            train_manifest=train_manifest,
            train_predictions=root / "train_predictions.jsonl",
            val_manifest=val_manifest,
            val_predictions=root / "val_predictions.jsonl",
            release_dir=_release(root),
            output_dir=root / "pairs",
            seed=42,
            count_tokens=char_token_count,
            tokenizer_record={"mode": "char-plus-eos", "eos": "synthetic"},
            **options,
        )

    def test_rejected_hypothesis_becomes_a_pair_without_logprobs(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = [_row("train", "noise", "u1")]
            val = [_row("val", "dropout", "v1")]
            report = self._build(
                root,
                train,
                [_prediction(train[0], "你号世界")],
                val,
                [_prediction(val[0], "你好世")],
            )
            self.assertEqual(report["status"], "pairs_built")
            self.assertFalse(report["trainable"])
            pair = json.loads((root / "pairs" / "train_pairs.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(pair["pair_id"], "u1__noise")
            self.assertEqual(pair["chosen_text"], "你好世界")
            self.assertEqual(pair["rejected_text"], "你号世界")
            self.assertEqual(pair["chosen_cer"], 0.0)
            self.assertGreater(pair["rejected_cer"], 0)
            self.assertIsNone(pair["ref_chosen_logp"])
            self.assertIsNone(pair["ref_rejected_logp"])
            self.assertNotIn("sha256", pair)
            self.assertEqual(pair["chosen_token_count"], len("你好世界") + 1)
            metadata = json.loads((root / "pairs" / "pair_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["reference_logprob"], "pending")
            self.assertEqual(metadata["train_pairs_sha256"], report["train_pairs_sha256"])

    def test_tie_and_empty_go_to_rejected_file(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            tie = _row("train", "clean", "tie")
            empty = _row("train", "noise", "empty")
            kept = _row("val", "dropout", "kept")
            self._build(
                root,
                [tie, empty],
                [_prediction(tie, "你好世界"), _prediction(empty, "")],
                [kept],
                [_prediction(kept, "你号世界")],
            )
            rejected = [
                json.loads(line)
                for line in (root / "pairs" / "pair_rejected.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            reasons = {row["sample_id"]: row["reason"] for row in rejected}
            self.assertEqual(reasons, {"tie__clean": "tie", "empty__noise": "empty_rejected"})
            train_pairs = (root / "pairs" / "train_pairs.jsonl").read_text(encoding="utf-8").strip()
            self.assertEqual(train_pairs, "")

    def test_overlength_is_separated_and_short_long_hallucination_stays(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            long_row = _row("train", "noise", "long", text="你好")
            huge = _row("train", "reverb", "huge", text="你好")
            val = _row("val", "dropout", "val")
            hallucination = "你好" * 4
            self._build(
                root,
                [long_row, huge],
                [_prediction(long_row, hallucination), _prediction(huge, "你" * 30)],
                [val],
                [_prediction(val, "你号")],
                max_transcript_tokens=8,
            )
            pairs = [
                json.loads(line)
                for line in (root / "pairs" / "train_pairs.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            over = [
                json.loads(line)
                for line in (root / "pairs" / "overlength_pairs.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["sample_id"] for row in pairs], ["long__noise"])
            self.assertGreater(len(hallucination), int(1.5 * len("你好")))
            self.assertEqual([row["sample_id"] for row in over], ["huge__reverb"])
            self.assertEqual(over[0]["reason"], "overlength")

    def test_duplicate_prediction_writes_no_pair_file(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = _row("train", "noise", "u1")
            val = _row("val", "dropout", "v1")
            train_manifest = root / "train" / "train.jsonl"
            val_manifest = root / "val" / "val.jsonl"
            _write_jsonl(train_manifest, [row])
            _write_jsonl(val_manifest, [val])
            _write_jsonl(root / "train_predictions.jsonl", [_prediction(row, "你号"), _prediction(row, "你号")])
            _write_jsonl(root / "val_predictions.jsonl", [_prediction(val, "你号")])
            _touch_wavs(train_manifest, [row])
            _touch_wavs(val_manifest, [val])
            code = main(
                [
                    "--train-manifest",
                    str(train_manifest),
                    "--train-predictions",
                    str(root / "train_predictions.jsonl"),
                    "--val-manifest",
                    str(val_manifest),
                    "--val-predictions",
                    str(root / "val_predictions.jsonl"),
                    "--release-dir",
                    str(_release(root)),
                    "--output-dir",
                    str(root / "pairs"),
                    "--seed",
                    "42",
                    "--contract-test-char-tokens",
                ]
            )
            self.assertEqual(code, 1)
            self.assertFalse((root / "pairs" / "train_pairs.jsonl").exists())
            report = json.loads((root / "pairs" / "pair_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "failed")

    def test_cross_split_source_and_bad_audio_fail(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = [_row("train", "noise", "same")]
            val = [_row("val", "noise", "same")]
            with self.assertRaises(Exception) as caught:
                self._build(
                    root,
                    train,
                    [_prediction(train[0], "你号世界")],
                    val,
                    [_prediction(val[0], "你号世界")],
                )
            self.assertIn("leaks", str(caught.exception))
            self.assertFalse((root / "pairs" / "train_pairs.jsonl").exists())

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = [_row("train", "noise", "u1")]
            val = [_row("val", "dropout", "v1")]
            train_manifest = root / "train" / "train.jsonl"
            val_manifest = root / "val" / "val.jsonl"
            _write_jsonl(train_manifest, train)
            _write_jsonl(val_manifest, val)
            _write_jsonl(root / "train_predictions.jsonl", [_prediction(train[0], "你号")])
            _write_jsonl(root / "val_predictions.jsonl", [_prediction(val[0], "你号")])
            _touch_wavs(train_manifest, train)
            _touch_wavs(val_manifest, val)
            broken = val_manifest.parent / str(val[0]["audio"])
            broken.write_bytes(b"not-a-wav-file-but-long-enough-to-pass-short-check")
            with self.assertRaises(Exception) as caught:
                build_dpo_pairs(
                    train_manifest=train_manifest,
                    train_predictions=root / "train_predictions.jsonl",
                    val_manifest=val_manifest,
                    val_predictions=root / "val_predictions.jsonl",
                    release_dir=_release(root),
                    output_dir=root / "pairs",
                    seed=42,
                    max_transcript_tokens=512,
                    count_tokens=char_token_count,
                    tokenizer_record={"mode": "char-plus-eos"},
                    min_val_pairs=1,
                    min_pairs_per_scenario=1,
                )
            self.assertIn("header", str(caught.exception))

    def test_staged_release_is_refused(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = [_row("train", "noise", "u1")]
            val = [_row("val", "dropout", "v1")]
            train_manifest = root / "train" / "train.jsonl"
            val_manifest = root / "val" / "val.jsonl"
            _write_jsonl(train_manifest, train)
            _write_jsonl(val_manifest, val)
            _write_jsonl(root / "train_predictions.jsonl", [_prediction(train[0], "你号")])
            _write_jsonl(root / "val_predictions.jsonl", [_prediction(val[0], "你号")])
            _touch_wavs(train_manifest, train)
            _touch_wavs(val_manifest, val)
            with self.assertRaises(Exception) as caught:
                build_dpo_pairs(
                    train_manifest=train_manifest,
                    train_predictions=root / "train_predictions.jsonl",
                    val_manifest=val_manifest,
                    val_predictions=root / "val_predictions.jsonl",
                    release_dir=_release(root, published=False),
                    output_dir=root / "pairs",
                    seed=42,
                    max_transcript_tokens=512,
                    count_tokens=char_token_count,
                    tokenizer_record={"mode": "char-plus-eos"},
                )
            self.assertIn("published", str(caught.exception))

    def test_too_few_val_pairs_exits_3_and_audit_is_stable(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = [_row("train", "noise", "u1")]
            val = [_row("val", "clean", "v1")]
            train_manifest = root / "train" / "train.jsonl"
            val_manifest = root / "val" / "val.jsonl"
            _write_jsonl(train_manifest, train)
            _write_jsonl(val_manifest, val)
            _write_jsonl(root / "train_predictions.jsonl", [_prediction(train[0], "你号世界")])
            _write_jsonl(root / "val_predictions.jsonl", [_prediction(val[0], "你号世界")])
            _touch_wavs(train_manifest, train)
            _touch_wavs(val_manifest, val)
            args = [
                "--train-manifest",
                str(train_manifest),
                "--train-predictions",
                str(root / "train_predictions.jsonl"),
                "--val-manifest",
                str(val_manifest),
                "--val-predictions",
                str(root / "val_predictions.jsonl"),
                "--release-dir",
                str(_release(root)),
                "--output-dir",
                str(root / "pairs"),
                "--seed",
                "42",
                "--contract-test-char-tokens",
            ]
            self.assertEqual(main(args), 3)
            audit = json.loads((root / "pairs" / "pair_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "unlistened")
            self.assertEqual(audit["by_scenario"]["val:clean"]["sample_ids"], ["v1__clean"])
            self.assertEqual(audit["by_scenario"]["val:dropout"]["sampled"], 0)


if __name__ == "__main__":
    unittest.main()
