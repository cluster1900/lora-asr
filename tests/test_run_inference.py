"""Inference keeps going when one audio file is missing."""

from __future__ import annotations

import json
import unittest
import wave
from pathlib import Path
from tempfile import TemporaryDirectory

from inference.config import load_infer_config
from inference.run_inference import (
    GreedyResult,
    parse_generation,
    require_model_revision,
    run_rows,
    stop_reason_from_ids,
)

ROOT = Path(__file__).resolve().parents[1]


class FakeBackend:
    def __init__(self) -> None:
        self.model_name = "Qwen/Qwen3-ASR-0.6B"
        self.model_revision = "test-revision"
        self.processor_sha256 = None
        self.calls: list[Path] = []

    def transcribe_file(self, audio_path: Path) -> GreedyResult:
        self.calls.append(audio_path)
        return GreedyResult(
            prediction_raw="甚至出现交易几乎停滞的情况。",
            generation_raw="language Chinese<asr_text>甚至出现交易几乎停滞的情况。",
            token_ids=[11, 12],
            stop_reason="eos",
            transcribe_text="甚至出现交易几乎停滞的情况。",
            model_revision=self.model_revision,
            processor_sha256=None,
        )


def _wav(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 160)


class RunInferenceTest(unittest.TestCase):
    def test_missing_file_is_recorded_and_later_rows_still_run(self) -> None:
        config = load_infer_config(ROOT / "configs/inference/base_eval.yaml")
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "smoke.jsonl"
            _wav(root / "wavs" / "clean.wav")
            rows = [
                {"sample_id": "bad", "audio": "wavs/missing.wav", "text": "你好", "language": "zh", "scenario": "dropout"},
                {"sample_id": "ok", "audio": "wavs/clean.wav", "text": "你好", "language": "zh", "scenario": "clean"},
            ]
            manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            output = root / "predictions.jsonl"
            backend = FakeBackend()
            code = run_rows(manifest, rows, backend, config, output)
            written = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(code, 1)
            self.assertEqual(written[0]["error_type"], "FileNotFoundError")
            self.assertIsNone(written[0]["token_ids"])
            self.assertEqual(written[1]["prediction_raw"], "甚至出现交易几乎停滞的情况。")
            self.assertTrue(written[1]["transcribe_matches_greedy"])
            self.assertEqual(len(backend.calls), 1)

    def test_pinned_revision_must_match(self) -> None:
        self.assertEqual(require_model_revision(None, "abc"), "abc")
        self.assertEqual(require_model_revision("abc", "abc"), "abc")
        with self.assertRaises(ValueError):
            require_model_revision("abc", None)

    def test_parse_and_stop_reason(self) -> None:
        self.assertEqual(parse_generation("language Chinese<asr_text>你好<|im_end|>"), "你好")
        self.assertEqual(stop_reason_from_ids([1, 2], 2, 8), "eos")
        self.assertEqual(stop_reason_from_ids([1, 2, 3], 9, 3), "length")


if __name__ == "__main__":
    unittest.main()
