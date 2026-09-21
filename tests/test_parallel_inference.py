from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference import parallel_inference


class ParallelInferenceTest(unittest.TestCase):
    def test_parse_args_defaults(self) -> None:
        args = parallel_inference.parse_args([
            "--manifest", "/path/to/manifest.jsonl",
            "--output", "/path/to/output.jsonl",
        ])
        self.assertEqual(args.manifest, "/path/to/manifest.jsonl")
        self.assertEqual(args.output, "/path/to/output.jsonl")
        self.assertEqual(args.model_id, "Qwen/Qwen3-ASR-1.7B")
        self.assertEqual(args.revision, "7278e1e70fe206f11671096ffdd38061171dd6e5")
        self.assertEqual(args.gpus, [0, 1, 2, 3])
        self.assertEqual(args.dtype, "float16")
        self.assertEqual(args.attention, "eager")
        self.assertFalse(args.eval)
        self.assertFalse(args.keep_shards)

    def test_split_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "test.jsonl"
            with open(manifest, "w") as f:
                for i in range(10):
                    f.write(json.dumps({"sample_id": f"s{i}", "text": f"text {i}"}) + "\n")

            shards = parallel_inference.split_manifest(manifest, 3)
            self.assertEqual(len(shards), 3)
            self.assertEqual(len(shards[0]), 4)
            self.assertEqual(len(shards[1]), 3)
            self.assertEqual(len(shards[2]), 3)

            # Round robin checks
            self.assertEqual(shards[0][0]["sample_id"], "s0")
            self.assertEqual(shards[1][0]["sample_id"], "s1")
            self.assertEqual(shards[2][0]["sample_id"], "s2")
            self.assertEqual(shards[0][1]["sample_id"], "s3")

    @patch("subprocess.Popen")
    def test_run_parallel_inference_mocked(self, mock_popen: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "test.jsonl"
            samples = [f"s{i}" for i in range(4)]
            with open(manifest, "w") as f:
                for sid in samples:
                    f.write(json.dumps({"sample_id": sid, "text": f"text {sid}"}) + "\n")

            output = Path(tmpdir) / "merged.jsonl"

            # When popen is called, mock worker writing shard outputs
            def side_effect(cmd, **kwargs):
                # find output argument
                out_idx = cmd.index("--output") + 1
                out_file = Path(cmd[out_idx])
                in_idx = cmd.index("--manifest") + 1
                in_file = Path(cmd[in_idx])
                # copy samples from input to output with dummy predictions
                with open(in_file) as inf, open(out_file, "w") as outf:
                    for line in inf:
                        d = json.loads(line)
                        outf.write(json.dumps({
                            "sample_id": d["sample_id"],
                            "prediction": f"pred_{d['sample_id']}",
                            "text": d["text"],
                        }) + "\n")
                mock_proc = MagicMock()
                mock_proc.poll.return_value = 0
                mock_proc.communicate.return_value = ("Done", "")
                return mock_proc

            mock_popen.side_effect = side_effect

            args = argparse.Namespace(
                manifest=str(manifest),
                output=str(output),
                model_id="mock_model",
                revision="mock_rev",
                adapter_dir=None,
                gpus=[0, 1],
                dtype="float16",
                attention="eager",
                batch_size=1,
                method="base",
                eval=False,
                temp_dir=None,
                keep_shards=False,
            )

            res = parallel_inference.run_parallel_inference(args)
            self.assertEqual(res, output.resolve())
            self.assertTrue(output.is_file())

            # Verify contents and order
            lines = [json.loads(l) for l in output.read_text().splitlines() if l.strip()]
            self.assertEqual(len(lines), 4)
            self.assertEqual([item["sample_id"] for item in lines], samples)
            self.assertEqual(lines[0]["prediction"], "pred_s0")


if __name__ == "__main__":
    unittest.main()
