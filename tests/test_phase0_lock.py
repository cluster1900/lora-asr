"""Portable Phase 0 lock parsing and the refuse-before-pip rule."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.rebuild_phase0_env import main, parse_portable_lock, pin_rows, torchao_blocks_peft


ROOT = Path(__file__).resolve().parents[1]


class Phase0LockTest(unittest.TestCase):
    def test_repository_lock_separates_cuda_pins_from_install_pins(self) -> None:
        parsed = parse_portable_lock((ROOT / "requirements.lock").read_text(encoding="utf-8"))
        verify = dict(parsed["verify"])
        install = dict(parsed["install"])
        self.assertEqual(verify["torch"], "2.11.0+cu130")
        self.assertEqual(verify["torchaudio"], "2.11.0+cu130")
        self.assertNotIn("torch", install)
        self.assertEqual(install["qwen-asr"], "0.0.6")
        self.assertEqual(install["transformers"], "4.57.6")

    def test_pin_rows_mark_a_missing_package_as_mismatch(self) -> None:
        rows = pin_rows([("qwen-asr", "0.0.6")], {"qwen-asr": None})
        self.assertFalse(rows[0]["match"])
        self.assertIsNone(rows[0]["actual"])

    def test_malformed_lock_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            parse_portable_lock("# schema: qwen3-asr-portable-lock-v1\ntorch>=2.5\n")

    def test_old_torchao_blocks_peft_and_a_missing_copy_does_not(self) -> None:
        self.assertTrue(torchao_blocks_peft("0.10.0"))
        self.assertFalse(torchao_blocks_peft("0.16.0"))
        self.assertFalse(torchao_blocks_peft("0.16.1"))
        self.assertFalse(torchao_blocks_peft(None))

    def test_verify_mismatch_exits_before_pip(self) -> None:
        with TemporaryDirectory() as tmp:
            lock = Path(tmp) / "requirements.lock"
            lock.write_text(
                "\n".join(
                    [
                        "# schema: qwen3-asr-portable-lock-v1",
                        "# verify: torch==2.11.0+cu130",
                        "qwen-asr==0.0.6",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            output = Path(tmp) / "out"
            with patch("scripts.rebuild_phase0_env.installed_version", return_value="2.5.0"), patch(
                "scripts.rebuild_phase0_env.cuda_available", return_value=False
            ), patch("scripts.rebuild_phase0_env.subprocess.run") as run:
                code = main(["--lock", str(lock), "--output-dir", str(output)])
            self.assertEqual(code, 2)
            run.assert_not_called()
            report = (output / "rebuild_report.json").read_text(encoding="utf-8")
            self.assertIn("verify_mismatch", report)
            self.assertIn("qwen3-asr-lock-rebuild-v1", report)


if __name__ == "__main__":
    unittest.main()
