"""scripts/build_rl_verdict_manifest.py on synthetic manifests.

docs/qwen3-asr/29_rl_v31_scale_design.md: the extra verdict set keeps only
degraded rows and must not overlap the RL train/held-out/validation roles.
"""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "build_rl_verdict_manifest", ROOT / "scripts" / "build_rl_verdict_manifest.py"
)
mod = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(mod)


def _row(i: int, group: str = "degraded", language: str = "en", scenario: str = "noise", **extra) -> dict:
    row = {
        "sample_id": f"s{i}",
        "source_utterance_id": f"u{i}",
        "audio_sha256": f"h{i}",
        "audio": f"/audio/s{i}.wav",
        "text": f"text {i}",
        "language": language,
        "scenario": scenario if group == "degraded" else "clean",
        "condition_group": group,
    }
    row.update(extra)
    return row


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


class BuildVerdictManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.source = _write(
            self.tmp / "dpo_val_pool.jsonl",
            [_row(1), _row(2, language="zh", scenario="echo"), _row(3, group="clean")],
        )
        self.other = _write(self.tmp / "rl_train_pool.jsonl", [_row(10), _row(11)])
        self.out = self.tmp / "out" / "rl_verdict_extra_degraded.jsonl"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_keeps_only_degraded_and_writes_complete(self) -> None:
        complete = mod.build(self.source, [self.other], self.out)
        rows = [json.loads(l) for l in self.out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["sample_id"] for r in rows], ["s1", "s2"])
        self.assertEqual(complete["row_count"], 2)
        self.assertEqual(complete["status"], "PASSED")
        self.assertEqual(complete["by_language"], {"en": 1, "zh": 1})
        self.assertEqual(complete["by_language_scenario"], {"en|noise": 1, "zh|echo": 1})
        on_disk = json.loads((self.out.parent / (self.out.name + ".COMPLETE.json")).read_text())
        self.assertEqual(on_disk["manifest_sha256"], mod.sha256_file(self.out))
        self.assertIn(str(self.other), on_disk["excluded"])

    def test_overlap_on_any_key_is_refused(self) -> None:
        for key, value in (("sample_id", "s1"), ("source_utterance_id", "u2"), ("audio_sha256", "h1")):
            with self.subTest(key=key):
                bad = _write(self.tmp / f"bad_{key}.jsonl", [_row(99, **{key: value})])
                with self.assertRaisesRegex(ValueError, key):
                    mod.build(self.source, [self.other, bad], self.tmp / f"o_{key}.jsonl")
                self.assertFalse((self.tmp / f"o_{key}.jsonl").exists())

    def test_rerun_is_idempotent_and_changed_inputs_are_refused(self) -> None:
        first = mod.build(self.source, [self.other], self.out)
        second = mod.build(self.source, [self.other], self.out)
        self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
        _write(self.source, [_row(1), _row(4)])
        with self.assertRaisesRegex(ValueError, "refusing"):
            mod.build(self.source, [self.other], self.out)

    def test_missing_text_or_audio_or_duplicate_is_refused(self) -> None:
        for name, rows in (
            ("notext", [_row(1, text="")]),
            ("noaudio", [_row(1, audio="")]),
            ("dupe", [_row(1), _row(1)]),
            ("nodeg", [_row(1, group="clean")]),
        ):
            with self.subTest(name=name):
                src = _write(self.tmp / f"{name}.jsonl", rows)
                with self.assertRaises(ValueError):
                    mod.build(src, [self.other], self.tmp / f"o_{name}.jsonl")

    def test_identity_keys_must_be_complete_and_unique_in_all_roles(self) -> None:
        missing = _row(20, source_utterance_id="")
        with self.assertRaisesRegex(ValueError, "source_utterance_id"):
            mod.build(self.source, [_write(self.tmp / "missing_identity.jsonl", [missing])], self.tmp / "missing.jsonl")

        duplicate = _write(self.tmp / "duplicate_identity.jsonl", [_row(20), _row(21, audio_sha256="h20")])
        with self.assertRaisesRegex(ValueError, "audio_sha256"):
            mod.build(self.source, [duplicate], self.tmp / "duplicate_identity_out.jsonl")

    def test_check_audio_requires_files(self) -> None:
        with self.assertRaisesRegex(ValueError, "audio files missing"):
            mod.build(self.source, [self.other], self.out, check_audio=True)
        audio_dir = self.tmp / "audio"
        audio_dir.mkdir()
        rows = [_row(i, audio=str(audio_dir / f"s{i}.wav")) for i in (1, 2)]
        for r in rows:
            Path(r["audio"]).write_bytes(b"RIFF")
        src = _write(self.tmp / "with_audio.jsonl", rows)
        self.assertEqual(mod.build(src, [self.other], self.tmp / "ok.jsonl", check_audio=True)["row_count"], 2)

    def test_cli_exit_codes(self) -> None:
        self.assertEqual(
            mod.main(["--source", str(self.source), "--exclude", str(self.other), "--output", str(self.out)]), 0
        )
        bad = _write(self.tmp / "bad.jsonl", [_row(1)])
        self.assertEqual(
            mod.main(["--source", str(self.source), "--exclude", str(bad), "--output", str(self.tmp / "x.jsonl")]), 1
        )


if __name__ == "__main__":
    unittest.main()
