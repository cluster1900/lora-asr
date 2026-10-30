"""Contract tests for dropout gap stripping. They do not download a model."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.strip_dpo_gap_pairs import main


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _pair(
    split: str,
    scenario: str,
    pair_id: str,
    chosen: str,
    rejected: str,
    *,
    tokens: int = 99,
) -> dict[str, object]:
    return {
        "schema_version": "qwen3-asr-dpo-pair-v1",
        "pair_id": pair_id,
        "sample_id": pair_id,
        "source_utterance_id": pair_id.split("__")[0],
        "audio": f"wavs/{pair_id}.wav",
        "scenario": scenario,
        "split": split,
        "reference_text": chosen,
        "chosen_text": chosen,
        "rejected_text": rejected,
        "chosen_cer": 0.0,
        "rejected_cer": 0.1,
        "chosen_token_count": tokens,
        "rejected_token_count": tokens,
        "ref_chosen_logp": None,
        "ref_rejected_logp": -1.5,
        "normalization_version": "zh-cer-v1",
        "edit_count": 1,
        "reference_char_count": len(chosen),
        "rejected_char_count": len(rejected),
        "char_length_delta": len(rejected) - len(chosen),
        "stop_reason": "eos",
        "bucket": "pair",
    }


class StripDpoGapPairsTest(unittest.TestCase):
    def test_gap_deletion_is_tied_and_outside_text_stays_paired(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "pairs"
            train = [
                _pair("train", "dropout", "u1__dropout", "0123456789", "01236789"),
                _pair("train", "dropout", "u2__dropout", "0123456789", "012345X789"),
                _pair("train", "clean", "u3__clean", "你好世界", "你号世界", tokens=99),
            ]
            val = [
                _pair("val", "dropout", "v1__dropout", "0123456789", "45"),
                _pair("val", "clean", "v2__clean", "你好世界", "你号世界", tokens=99),
                _pair("val", "noise", "v3__noise", "你好世界", "你号世界"),
                _pair("val", "reverb", "v4__reverb", "你好世界", "你号世界"),
                _pair("val", "far_field", "v5__far", "你好世界", "你号世界"),
            ]
            train_path = source / "train_pairs.jsonl"
            val_path = source / "val_pairs.jsonl"
            _write(train_path, train)
            _write(val_path, val)
            before = (_sha(train_path), _sha(val_path))
            output = root / "pairs_gap_strip"
            status = main(
                [
                    "--train-pairs",
                    str(train_path),
                    "--val-pairs",
                    str(val_path),
                    "--expect-train-sha256",
                    before[0],
                    "--expect-val-sha256",
                    before[1],
                    "--output-dir",
                    str(output),
                    "--seed",
                    "42",
                    "--contract-test-char-tokens",
                    "--min-val-pairs",
                    "1",
                    "--min-pairs-per-scenario",
                    "0",
                    "--audit-per-scenario",
                    "1",
                ]
            )
            self.assertEqual(status, 0)
            self.assertEqual((_sha(train_path), _sha(val_path)), before)
            kept = [json.loads(line) for line in (output / "train_pairs.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["pair_id"] for row in kept], ["u2__dropout", "u3__clean"])
            dropout = kept[0]
            self.assertEqual(dropout["chosen_text"], dropout["reference_text"])
            self.assertEqual(dropout["chosen_text"], "01236789")
            self.assertEqual(dropout["rejected_text"], "0123x789")
            self.assertEqual(dropout["source_reference_text"], "0123456789")
            self.assertEqual(dropout["chosen_token_count"], len("01236789") + 1)
            self.assertIsNone(dropout["ref_chosen_logp"])
            self.assertIsNone(dropout["ref_rejected_logp"])
            clean = kept[1]
            self.assertEqual(clean["chosen_text"], "你好世界")
            self.assertEqual(clean["chosen_token_count"], 99)
            self.assertIsNone(clean["ref_rejected_logp"])
            self.assertNotIn("gap_strip", clean)
            ties = [json.loads(line) for line in (output / "gap_ties.jsonl").read_text(encoding="utf-8").splitlines()]
            reasons = {row["pair_id"]: row["reason"] for row in ties}
            self.assertEqual(reasons["u1__dropout"], "tie_after_strip")
            self.assertEqual(reasons["v1__dropout"], "empty_rejected_after_strip")
            audit = json.loads((output / "pair_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["status"], "unlistened")
            self.assertTrue(all(block["disposition"] == "unlistened" for block in audit["by_scenario"].values()))
            report = json.loads((output / "pair_report.json").read_text(encoding="utf-8"))
            self.assertFalse(report["trainable"])
            self.assertEqual(report["train_pairs"], 2)
            self.assertEqual(report["gap_ties"], 2)

    def test_bad_hash_or_frozen_directory_name_writes_nothing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "pairs"
            row = _pair("train", "clean", "u3__clean", "你好世界", "你号世界")
            val_row = _pair("val", "clean", "v2__clean", "你好世界", "你号世界")
            train_path = source / "train_pairs.jsonl"
            val_path = source / "val_pairs.jsonl"
            _write(train_path, [row])
            _write(val_path, [val_row])
            before = train_path.read_bytes()
            output = root / "pairs_gap_strip"
            status = main(
                [
                    "--train-pairs",
                    str(train_path),
                    "--val-pairs",
                    str(val_path),
                    "--expect-train-sha256",
                    "a" * 64,
                    "--expect-val-sha256",
                    _sha(val_path),
                    "--output-dir",
                    str(output),
                    "--seed",
                    "42",
                    "--contract-test-char-tokens",
                    "--min-val-pairs",
                    "1",
                    "--min-pairs-per-scenario",
                    "0",
                ]
            )
            self.assertEqual(status, 1)
            self.assertFalse(output.exists())
            self.assertEqual(train_path.read_bytes(), before)
            frozen = root / "pairs_logprob"
            status = main(
                [
                    "--train-pairs",
                    str(train_path),
                    "--val-pairs",
                    str(val_path),
                    "--expect-train-sha256",
                    _sha(train_path),
                    "--expect-val-sha256",
                    _sha(val_path),
                    "--output-dir",
                    str(frozen),
                    "--seed",
                    "42",
                    "--contract-test-char-tokens",
                ]
            )
            self.assertEqual(status, 1)
            self.assertFalse(frozen.exists())
            self.assertEqual(train_path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
