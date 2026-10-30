"""Readback hashes and the seeded Phase 0 noise transform."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from scripts.phase0_readback import readback_files
from scripts.prepare_phase0_audio import add_gaussian_noise, to_mono_16k, write_smoke_manifest
from scripts.run_phase0_baseline import CommandLog, resolve_drive_destination


class Phase0SupportTest(unittest.TestCase):
    def test_readback_detects_a_changed_byte(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            dest = root / "dest"
            source.mkdir()
            (source / "environment.json").write_text("{}", encoding="utf-8")
            report = readback_files(source, dest, ["environment.json"])
            self.assertTrue(report["matched"])
            (dest / "environment.json").write_text("{ }\n", encoding="utf-8")
            # Re-hash the already copied file the way a suspicious read would.
            from scripts.phase0_readback import sha256_file

            self.assertNotEqual(sha256_file(source / "environment.json"), sha256_file(dest / "environment.json"))

    def test_command_log_records_the_child_process(self) -> None:
        with TemporaryDirectory() as tmp:
            log = CommandLog(Path(tmp) / "command_log.md")
            result = log.run("python print", [sys.executable, "-c", "print('logged')"])
            text = (Path(tmp) / "command_log.md").read_text(encoding="utf-8")
            self.assertEqual(result.returncode, 0)
            self.assertIn("exit_code: `0`", text)
            self.assertIn("logged", text)
            self.assertIn("python print", text)

    def test_missing_drive_parent_stays_local(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            destination, status = resolve_drive_destination(
                root / "absent" / "qwen3-asr",
                root / "run",
                "phase0",
            )
            self.assertEqual(status, "missing")
            self.assertEqual(destination, root / "run" / "readback_local")
            parent = root / "mounted"
            parent.mkdir()
            destination, status = resolve_drive_destination(parent / "qwen3-asr", root / "run", "phase0")
            self.assertEqual(status, "mounted")
            self.assertEqual(destination, parent / "qwen3-asr" / "runs" / "phase0")
            self.assertTrue((parent / "qwen3-asr").is_dir())
            self.assertFalse((parent / "qwen3-asr" / ".write_probe").exists())

    def test_noise_is_seeded_and_manifest_has_three_rows(self) -> None:
        clean = np.linspace(-0.2, 0.2, 160, dtype=np.float32)
        first = add_gaussian_noise(clean, 5.0, 42)
        second = add_gaussian_noise(clean, 5.0, 42)
        other = add_gaussian_noise(clean, 5.0, 43)
        self.assertTrue(np.array_equal(first, second))
        self.assertFalse(np.array_equal(first, other))
        mono = to_mono_16k(np.stack([clean, clean], axis=1), 8000)
        self.assertEqual(mono.shape[0], 320)
        with TemporaryDirectory() as tmp:
            manifest = write_smoke_manifest(Path(tmp), clean, first)
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["scenario"] for row in rows], ["clean", "noise", "dropout"])
            self.assertTrue((Path(tmp) / "wavs" / "asr_zh.wav").is_file())
            self.assertFalse((Path(tmp) / "wavs" / "does_not_exist.wav").exists())


if __name__ == "__main__":
    unittest.main()
