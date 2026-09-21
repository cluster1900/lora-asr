from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import run_inference


class RunInferenceTest(unittest.TestCase):
    def test_parse_args_defaults(self) -> None:
        args = run_inference.parse_args([
            "--manifest", "/path/to/manifest.jsonl",
            "--output", "/path/to/output.jsonl",
        ])
        self.assertEqual(args.manifest, "/path/to/manifest.jsonl")
        self.assertEqual(args.output, "/path/to/output.jsonl")
        self.assertEqual(args.model_id, "Qwen/Qwen3-ASR-1.7B")
        self.assertEqual(args.revision, "7278e1e70fe206f11671096ffdd38061171dd6e5")
        self.assertEqual(args.dtype, "float16")
        self.assertEqual(args.attention, "eager")
        self.assertEqual(args.batch_size, 1)
        self.assertFalse(args.no_resume)
        self.assertFalse(args.eval)
        self.assertIsNone(args.method)

    def test_method_deduction_and_specification(self) -> None:
        # Explicit CLI method
        args = run_inference.parse_args([
            "--manifest", "m.jsonl", "--output", "o.jsonl", "--method", "custom_method"
        ])
        self.assertEqual(args.method, "custom_method")

        # Test deduction logic inside run_inference_on_manifest
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest_p = Path(tmpdir) / "m.jsonl"
            manifest_p.write_text(json.dumps({"sample_id": "s1", "text": "hi", "audio": "a.wav"}) + "\n")

            test_cases = [
                ("/runs/sft_pilot/merged_base", None, "sft_merged"),
                ("/runs/dpo_pilot/merged_base", None, "dpo_merged"),
                ("/runs/rl_pilot/merged_base", None, "rl_merged"),
                ("Qwen/Qwen3-ASR-1.7B", "adapter_dir", "lora"),
                ("Qwen/Qwen3-ASR-1.7B", None, "base"),
            ]

            for model_id, adapter_dir, expected_method in test_cases:
                out_p = Path(tmpdir) / f"out_{expected_method}.jsonl"
                run_inference.run_inference_on_manifest(
                    manifest_path=manifest_p,
                    output_path=out_p,
                    model=MagicMock(),
                    model_id=model_id,
                    adapter_dir=adapter_dir,
                    transcribe_fn=lambda m, a, l: "pred",
                )
                record = json.loads(out_p.read_text().strip())
                self.assertEqual(record["method"], expected_method)

    def test_extract_prediction_text_formats(self) -> None:
        # String
        self.assertEqual(run_inference.extract_prediction_text("hello world"), "hello world")

        # Object with text attribute
        mock_obj = MagicMock()
        mock_obj.text = "test transcription"
        self.assertEqual(run_inference.extract_prediction_text([mock_obj]), "test transcription")
        self.assertEqual(run_inference.extract_prediction_text(mock_obj), "test transcription")

        # Dict with text key
        self.assertEqual(run_inference.extract_prediction_text([{"text": "dict text"}]), "dict text")
        self.assertEqual(run_inference.extract_prediction_text({"text": "dict text"}), "dict text")

        # Empty list
        self.assertEqual(run_inference.extract_prediction_text([]), "")

    def test_resumption_skips_existing_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_p = Path(tmpdir)
            manifest_p = tmp_p / "manifest.jsonl"
            output_p = tmp_p / "output.jsonl"

            # Create manifest with 3 items
            items = [
                {"sample_id": "s1", "audio": "a1.wav", "text": "one", "language": "en", "scenario": "clean"},
                {"sample_id": "s2", "audio": "a2.wav", "text": "two", "language": "en", "scenario": "noise"},
                {"sample_id": "s3", "audio": "a3.wav", "text": "three", "language": "en", "scenario": "reverb"},
            ]
            manifest_p.write_text("\n".join(json.dumps(x) for x in items) + "\n", encoding="utf-8")

            # Pre-populate output with s1
            output_p.write_text(
                json.dumps({"sample_id": "s1", "text": "one", "prediction": "one"}) + "\n",
                encoding="utf-8",
            )

            mock_model = MagicMock()
            processed = []

            def mock_transcribe(m, audio, lang):
                processed.append(audio)
                return "transcribed"

            total, succ, fail = run_inference.run_inference_on_manifest(
                manifest_path=manifest_p,
                output_path=output_p,
                model=mock_model,
                model_id="mock_model",
                resume=True,
                transcribe_fn=mock_transcribe,
            )

            # s1 should be skipped, so total processed in this run is 2 (s2 and s3)
            self.assertEqual(total, 2)
            self.assertEqual(succ, 2)
            self.assertEqual(fail, 0)
            self.assertEqual(processed, ["a2.wav", "a3.wav"])

            # Verify output file now contains 3 lines
            out_lines = [json.loads(l) for l in output_p.read_text(encoding="utf-8").strip().split("\n")]
            self.assertEqual(len(out_lines), 3)
            self.assertEqual([l["sample_id"] for l in out_lines], ["s1", "s2", "s3"])

    def test_error_handling_records_error_and_does_not_crash(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_p = Path(tmpdir)
            manifest_p = tmp_p / "manifest.jsonl"
            output_p = tmp_p / "output.jsonl"

            items = [
                {"sample_id": "ok1", "audio": "ok1.wav", "text": "hello", "language": "en", "scenario": "clean"},
                {"sample_id": "fail1", "audio": "bad.wav", "text": "oops", "language": "en", "scenario": "dropout"},
                {"sample_id": "ok2", "audio": "ok2.wav", "text": "world", "language": "en", "scenario": "clean"},
            ]
            manifest_p.write_text("\n".join(json.dumps(x) for x in items) + "\n", encoding="utf-8")

            def mock_transcribe(m, audio, lang):
                if "bad" in audio:
                    raise RuntimeError("Corrupt audio frame")
                return "good transcription"

            total, succ, fail = run_inference.run_inference_on_manifest(
                manifest_path=manifest_p,
                output_path=output_p,
                model=MagicMock(),
                model_id="mock_model",
                transcribe_fn=mock_transcribe,
            )

            self.assertEqual(total, 3)
            self.assertEqual(succ, 2)
            self.assertEqual(fail, 1)

            out_lines = [json.loads(l) for l in output_p.read_text(encoding="utf-8").strip().split("\n")]
            self.assertEqual(len(out_lines), 3)
            self.assertEqual(out_lines[1]["sample_id"], "fail1")
            self.assertEqual(out_lines[1]["prediction"], "")
            self.assertIn("Corrupt audio frame", out_lines[1]["error"])
            self.assertEqual(out_lines[2]["prediction"], "good transcription")

    def test_resumption_sanitizes_missing_trailing_newline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_p = Path(tmpdir) / "output.jsonl"
            # Write a valid line without trailing newline
            output_p.write_text(json.dumps({"sample_id": "s1", "text": "hello"}), encoding="utf-8")

            sample_ids = run_inference.load_existing_sample_ids(output_p)
            self.assertEqual(sample_ids, {"s1"})

            # Verify that output_p now ends with a clean newline
            content = output_p.read_text(encoding="utf-8")
            self.assertTrue(content.endswith("\n"))
            self.assertEqual(content.strip(), json.dumps({"sample_id": "s1", "text": "hello"}))


if __name__ == "__main__":
    unittest.main()
