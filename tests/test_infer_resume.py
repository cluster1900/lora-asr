from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from inference import qwen3_asr_infer


class FakeModel:
    def __init__(self) -> None:
        self.languages: list[str | None] = []

    def transcribe(self, audio: str, language: str | None):  # type: ignore[no-untyped-def]
        self.languages.append(language)
        if "fail" in audio:
            raise RuntimeError("decode failed")
        return [{"text": Path(audio).stem, "language": language or "auto"}]


def args(**overrides):  # type: ignore[no-untyped-def]
    values = {
        "adapter_dir": None,
        "audio_root": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


class InferResumeTest(unittest.TestCase):
    def test_runtime_contract_is_fixed(self) -> None:
        parsed = qwen3_asr_infer.parse_args(["--manifest", "in.jsonl", "--output-jsonl", "out.jsonl"])
        self.assertEqual(qwen3_asr_infer.DTYPE, "bfloat16")
        self.assertEqual(qwen3_asr_infer.DEVICE_MAP, "cuda:0")
        self.assertFalse(hasattr(parsed, "model_revision"))

    def test_row_identity_requires_unique_sample_id(self) -> None:
        self.assertEqual(qwen3_asr_infer.row_key({"sample_id": "abc"}), "sample_id:abc")
        with self.assertRaisesRegex(ValueError, "missing sample_id"):
            qwen3_asr_infer.row_key({})
        with self.assertRaisesRegex(ValueError, "Duplicate manifest row identity"):
            qwen3_asr_infer.indexed_rows([{"sample_id": "x"}, {"sample_id": "x"}])

    def test_incremental_inference_maps_languages_and_records_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "en.wav").touch()
            (root / "fail.wav").touch()
            manifest = root / "manifest.jsonl"
            rows = [
                {"sample_id": "en", "audio": "en.wav", "language": "en"},
                {"sample_id": "zh", "audio": "fail.wav", "language": "zh"},
                {"sample_id": "missing", "language": "en"},
                {"sample_id": "bad-language", "audio": "en.wav", "language": "english"},
            ]
            prepared = qwen3_asr_infer.indexed_rows(rows)
            output = root / "output.jsonl"
            model = FakeModel()

            with output.open("w", encoding="utf-8") as handle:
                written = qwen3_asr_infer.infer_rows(
                    model, prepared, len(prepared), manifest, handle, args()
                )

            results = qwen3_asr_infer.read_jsonl(output)
            self.assertEqual(written, 4)
            self.assertEqual(len(results), 4)
            self.assertEqual(model.languages, ["English", "Chinese"])
            self.assertEqual(results[0]["prediction"], "en")
            self.assertEqual(results[0]["inference_key"], "sample_id:en")
            self.assertEqual(results[1]["prediction"], "")
            self.assertIn("RuntimeError: decode failed", results[1]["error"])
            self.assertIn("missing audio", results[2]["error"])
            self.assertIn("Invalid manifest language", results[3]["error"])

    def test_resume_with_all_rows_done_does_not_load_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            manifest.write_text(json.dumps({"sample_id": "done", "audio": "unused.wav"}) + "\n")
            output = root / "output.jsonl"
            output.write_text(json.dumps({"inference_key": "sample_id:done"}) + "\n")

            with mock.patch.object(qwen3_asr_infer, "load_model") as loader:
                qwen3_asr_infer.main([
                    "--manifest", str(manifest),
                    "--output-jsonl", str(output),
                    "--resume",
                ])

            loader.assert_not_called()
            self.assertEqual(len(qwen3_asr_infer.read_jsonl(output)), 1)

    def test_resume_adds_missing_line_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output.jsonl"
            output.write_text(json.dumps({"inference_key": "sample_id:first"}))

            qwen3_asr_infer.ensure_append_boundary(output)
            with output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"inference_key": "sample_id:second"}) + "\n")

            self.assertEqual(len(qwen3_asr_infer.read_jsonl(output)), 2)

    def test_help_does_not_import_ml_dependencies(self) -> None:
        script = Path(qwen3_asr_infer.__file__)
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--adapter-dir", result.stdout)
        self.assertIn("--resume", result.stdout)
        self.assertNotIn("--model-id", result.stdout)
        self.assertNotIn("--dtype", result.stdout)
        self.assertNotIn("--language", result.stdout)


if __name__ == "__main__":
    unittest.main()
