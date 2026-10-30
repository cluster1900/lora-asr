"""Phase 1 manifest builder and validator contract tests.

Synthetic tones prove the gate. They are not an audited speech corpus.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from scripts.build_manifest import main as build_main
from scripts.manifest_contract import assign_split, load_data_config, sha256_file
from scripts.manifest_validate import main as validate_main
from scripts.manifest_validate import validate_manifest_dir


ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = ROOT / "configs" / "data" / "manifest_mvp.yaml"


def _write_tone(path: Path, seconds: float, frequency: float) -> None:
    frames = int(round(seconds * 16000))
    axis = np.arange(frames, dtype=np.float32) / np.float32(16000)
    audio = (np.float32(0.2) * np.sin(np.float32(2.0 * np.pi * frequency) * axis)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, 16000, format="WAV", subtype="PCM_16")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _source_ids(train_count: int, val_count: int) -> tuple[list[str], list[str]]:
    train: list[str] = []
    val: list[str] = []
    index = 0
    while len(train) < train_count or len(val) < val_count:
        source_id = f"s{index:04d}"
        split = assign_split(42, source_id, 0.1, 0.1)
        if split == "train" and len(train) < train_count:
            train.append(source_id)
        elif split == "val" and len(val) < val_count:
            val.append(source_id)
        index += 1
    return train, val


def _fixture_config(path: Path, audited: bool, minimum_val_rows: int) -> None:
    path.write_text(
        "\n".join(
            [
                "schema_version: qwen3-asr-data-v1",
                "seed: 42",
                "source_version: synthetic-fixture",
                f"audited: {'true' if audited else 'false'}",
                "audio_processor_version: phase1-aug-v1",
                "language: zh",
                "sample_rate: 16000",
                "duration_seconds:",
                "  min: 0.5",
                "  max: 30.0",
                "splits:",
                "  val: 0.1",
                "  test: 0.1",
                "quota:",
                f"  min_val_rows_per_scenario: {minimum_val_rows}",
                "scenarios: [clean, noise, reverb, far_field, dropout]",
                "augmentation:",
                "  noise: {snr_db: 5.0}",
                "  reverb: {delay_ms: 50, decay: 0.35, repeats: 5}",
                "  far_field: {gain: 0.3, lowpass_window: 8}",
                "  dropout: {start_ratio: 0.4, width_ratio: 0.15}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _sources(root: Path, specs: list[tuple[str, str, float, float]]) -> Path:
    rows = []
    for source_id, text, seconds, frequency in specs:
        relative = f"audio/{source_id}.wav"
        _write_tone(root / relative, seconds, frequency)
        rows.append({"source_utterance_id": source_id, "audio": relative, "text": text, "language": "zh"})
    path = root / "sources.jsonl"
    _write_jsonl(path, rows)
    return path


def _content_hashes(manifest_dir: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(item for item in manifest_dir.rglob("*") if item.is_file()):
        if path.name == "sha256.json":
            continue
        hashes[path.relative_to(manifest_dir).as_posix()] = sha256_file(path)
    return hashes


def _read_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ManifestContractTest(unittest.TestCase):
    def test_two_builds_match_except_sha_timestamp(self) -> None:
        train_ids, val_ids = _source_ids(1, 1)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "fixture.yaml"
            _fixture_config(config, audited=True, minimum_val_rows=1)
            specs = [
                (train_ids[0], "样本甲", 0.6, 220.0),
                (val_ids[0], "样本乙", 0.6, 330.0),
            ]
            sources = _sources(root / "src", specs)
            first = root / "first"
            second = root / "second"
            self.assertEqual(
                build_main(["--config", str(config), "--sources", str(sources), "--output-dir", str(first), "--seed", "42", "--git-sha", "fixture"]),
                0,
            )
            self.assertEqual(
                build_main(["--config", str(config), "--sources", str(sources), "--output-dir", str(second), "--seed", "42", "--git-sha", "fixture"]),
                0,
            )
            self.assertEqual(_content_hashes(first), _content_hashes(second))
            left = json.loads((first / "sha256.json").read_text(encoding="utf-8"))
            right = json.loads((second / "sha256.json").read_text(encoding="utf-8"))
            self.assertNotIn("generated_at", json.loads((first / "manifest.json").read_text(encoding="utf-8")))
            left.pop("generated_at")
            right.pop("generated_at")
            self.assertEqual(left, right)

            rows = []
            for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
                rows.extend(_read_rows(first / name))
            by_source: dict[str, list[dict[str, object]]] = {}
            for row in rows:
                by_source.setdefault(str(row["source_utterance_id"]), []).append(row)
            one = by_source[train_ids[0]]
            self.assertEqual(len(one), 5)
            self.assertEqual({row["split"] for row in one}, {"train"})
            wav_hashes = {sha256_file(first / str(row["audio"])) for row in one}
            self.assertEqual(len(wav_hashes), 5)
            clean = next(row for row in one if row["scenario"] == "clean")
            self.assertNotIn("augmentation", clean)
            for scenario in ("noise", "reverb", "far_field", "dropout"):
                degraded = next(row for row in one if row["scenario"] == scenario)
                augmentation = degraded["augmentation"]
                self.assertIsInstance(augmentation, dict)
                self.assertEqual(augmentation["processor_version"], "phase1-aug-v1")
            dropout, sample_rate = sf.read(first / f"wavs/{train_ids[0]}__dropout.wav")
            self.assertEqual(sample_rate, 16000)
            start = int(dropout.size * 0.4)
            stop = int(dropout.size * 0.55)
            self.assertTrue(np.all(dropout[start:stop] == 0))
            self.assertGreater(float(np.max(np.abs(dropout[:start]))), 0.0)

    def test_bad_rows_are_rejected_and_the_good_source_remains(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "fixture.yaml"
            _fixture_config(config, audited=True, minimum_val_rows=1)
            _write_tone(root / "short.wav", 0.1, 440.0)
            rows = [
                {"source_utterance_id": "good0001", "audio": "good.wav", "text": "正常样本", "language": "zh"},
                {"source_utterance_id": "miss0001", "audio": "missing.wav", "text": "缺失样本", "language": "zh"},
                {"source_utterance_id": "short001", "audio": "short.wav", "text": "过短样本", "language": "zh"},
                {"source_utterance_id": "latin001", "audio": "good.wav", "text": "hello样本", "language": "zh"},
                {"source_utterance_id": "path0001", "audio": "../outside.wav", "text": "路径样本", "language": "zh"},
            ]
            _write_tone(root / "good.wav", 0.6, 260.0)
            sources = root / "sources.jsonl"
            _write_jsonl(sources, rows)
            output = root / "out"
            code = build_main(["--config", str(config), "--sources", str(sources), "--output-dir", str(output), "--seed", "42", "--git-sha", "fixture"])
            rejected = _read_rows(output / "rejected.jsonl")
            reasons = {row["reason"] for row in rejected}
            self.assertEqual(reasons, {"missing_audio", "duration_out_of_range", "text_has_latin", "unsafe_audio_path"})
            kept = []
            for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
                kept.extend(_read_rows(output / name))
            self.assertEqual({row["source_utterance_id"] for row in kept}, {"good0001"})
            self.assertEqual(len(kept), 5)
            self.assertIn(code, {0, 3})

    def test_formal_config_on_a_tiny_set_is_insufficient(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = _sources(root / "src", [("tiny0001", "样本一", 0.6, 280.0)])
            output = root / "out"
            code = build_main(
                ["--config", str(FORMAL_CONFIG), "--sources", str(sources), "--output-dir", str(output), "--seed", "42", "--git-sha", "fixture"]
            )
            self.assertEqual(code, 3)
            report = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "insufficient_data")
            self.assertEqual(report["source_version"], "aishell1-openslr33-pilot-20261015")
            self.assertTrue(report["audited"])
            self.assertFalse(any(item.get("reason") == "source_not_audited" for item in report["shortages"]))
            val_shortages = [item for item in report["shortages"] if item.get("split") == "val"]
            self.assertEqual(len(val_shortages), 5)
            self.assertTrue(all(item["minimum"] == 50 and item["rows"] < 50 for item in val_shortages))
            unchecked = validate_main(["--manifest", str(output), "--config", str(FORMAL_CONFIG)])
            self.assertEqual(unchecked, 2)

    def test_audited_fixture_can_pass_and_leakage_fails(self) -> None:
        train_ids, val_ids = _source_ids(1, 1)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "fixture.yaml"
            _fixture_config(config, audited=True, minimum_val_rows=1)
            sources = _sources(
                root / "src",
                [
                    (train_ids[0], "样本甲", 0.6, 240.0),
                    (val_ids[0], "样本乙", 0.6, 360.0),
                ],
            )
            output = root / "out"
            code = build_main(["--config", str(config), "--sources", str(sources), "--output-dir", str(output), "--seed", "42", "--git-sha", "fixture"])
            self.assertEqual(code, 0)
            checked = validate_main(["--manifest", str(output), "--config", str(config), "--check-audio"])
            self.assertEqual(checked, 0)
            train_rows = _read_rows(output / "train.jsonl")
            leaked = []
            for row in train_rows:
                copied = dict(row)
                copied["split"] = "val"
                leaked.append(copied)
            with (output / "val.jsonl").open("a", encoding="utf-8") as handle:
                for row in leaked:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            report, leak_code = validate_manifest_dir(output, load_data_config(config), True)
            self.assertEqual(leak_code, 1)
            self.assertTrue(any(str(item).startswith("source_leakage:") for item in report["problems"]))

    def test_official_split_keeps_dev_and_rejects_a_crossing_speaker(self) -> None:
        dev_id = next(
            source_id
            for index in range(1000)
            if assign_split(42, (source_id := f"dev{index:04d}"), 0.1, 0.1) != "val"
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "fixture.yaml"
            _fixture_config(config, audited=True, minimum_val_rows=1)
            specs = (
                ("trn0001", "训练语句", 240.0, "S0002", "train"),
                (dev_id, "开发语句", 360.0, "S0003", "dev"),
                ("trn0002", "同人语句", 300.0, "S0002", "dev"),
                ("bad0001", "错误切分", 280.0, "S0004", "val"),
            )
            rows = []
            for source_id, text, frequency, speaker, official in specs:
                relative = f"audio/{source_id}.wav"
                _write_tone(root / relative, 0.6, frequency)
                rows.append(
                    {
                        "source_utterance_id": source_id,
                        "audio": relative,
                        "text": text,
                        "language": "zh",
                        "speaker_id": speaker,
                        "official_split": official,
                        "source_package": "data_aishell.tgz",
                    }
                )
            sources = root / "sources.jsonl"
            _write_jsonl(sources, rows)
            output = root / "out"
            code = build_main(
                ["--config", str(config), "--sources", str(sources), "--output-dir", str(output), "--seed", "42", "--git-sha", "fixture"]
            )
            self.assertEqual(code, 0)
            val_rows = _read_rows(output / "val.jsonl")
            train_rows = _read_rows(output / "train.jsonl")
            self.assertEqual({row["source_utterance_id"] for row in val_rows}, {dev_id})
            self.assertTrue(all(row["split"] == "val" and row["official_split"] == "dev" and row["speaker_id"] == "S0003" for row in val_rows))
            self.assertEqual({row["source_utterance_id"] for row in train_rows}, {"trn0001"})
            self.assertTrue(all(row["source_package"] == "data_aishell.tgz" for row in train_rows))
            reasons = {row["reason"] for row in _read_rows(output / "rejected.jsonl")}
            self.assertEqual(reasons, {"speaker_cross_split", "bad_official_split"})
            report = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(report["split_audit"]["mode"], "official")
            self.assertEqual(report["split_audit"]["speakers"]["train"], ["S0002"])
            self.assertEqual(report["split_audit"]["speakers"]["val"], ["S0003"])
            self.assertEqual(report["split_audit"]["speaker_intersection"]["train_val"], [])
            self.assertEqual(report["split_audit"]["source_packages"], ["data_aishell.tgz"])
            leaked = []
            for row in val_rows:
                copied = dict(row)
                copied["speaker_id"] = "S0002"
                leaked.append(copied)
            _write_jsonl(output / "val.jsonl", leaked)
            checked, leak_code = validate_manifest_dir(output, load_data_config(config), True)
            self.assertEqual(leak_code, 1)
            self.assertTrue(any(str(item).startswith("speaker_leakage:S0002") for item in checked["problems"]))

    def test_mixed_split_mode_exits_before_writing(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "fixture.yaml"
            _fixture_config(config, audited=False, minimum_val_rows=50)
            _write_tone(root / "audio/hash0001.wav", 0.6, 220.0)
            _write_tone(root / "audio/off0001.wav", 0.6, 330.0)
            sources = root / "sources.jsonl"
            _write_jsonl(
                sources,
                [
                    {"source_utterance_id": "hash0001", "audio": "audio/hash0001.wav", "text": "哈希语句", "language": "zh"},
                    {
                        "source_utterance_id": "off0001",
                        "audio": "audio/off0001.wav",
                        "text": "官方语句",
                        "language": "zh",
                        "speaker_id": "S0002",
                        "official_split": "train",
                    },
                ],
            )
            output = root / "out"
            errors = StringIO()
            with redirect_stderr(errors):
                code = build_main(
                    ["--config", str(config), "--sources", str(sources), "--output-dir", str(output), "--seed", "42", "--git-sha", "fixture"]
                )
            self.assertEqual(code, 2)
            self.assertIn("mixed_split_mode", errors.getvalue())
            self.assertFalse((output / "train.jsonl").exists())

    def test_script_file_entry_rejects_a_seed_mismatch(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "fixture.yaml"
            _fixture_config(config, audited=False, minimum_val_rows=50)
            output = root / "out"
            built = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "build_manifest.py"),
                    "--config",
                    str(config),
                    "--sources",
                    str(root / "missing.jsonl"),
                    "--output-dir",
                    str(output),
                    "--seed",
                    "7",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(built.returncode, 2, built.stderr)
            self.assertIn("does not match", built.stderr)
            self.assertFalse((output / "train.jsonl").exists())
            checked = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "manifest_validate.py"),
                    "--manifest",
                    str(output),
                    "--config",
                    str(config),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(checked.returncode, 2, checked.stderr)
            self.assertIn('"exit_code": 2', checked.stdout)

    def test_seed_mismatch_exits_before_building(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "fixture.yaml"
            _fixture_config(config, audited=False, minimum_val_rows=50)
            sources = root / "sources.jsonl"
            sources.write_text("", encoding="utf-8")
            output = root / "out"
            code = build_main(["--config", str(config), "--sources", str(sources), "--output-dir", str(output), "--seed", "7"])
            self.assertEqual(code, 2)
            self.assertFalse((output / "train.jsonl").exists())

    def test_changing_text_changes_the_jsonl_hash(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "fixture.yaml"
            _fixture_config(config, audited=False, minimum_val_rows=50)
            left_sources = _sources(root / "left", [("tiny0001", "样本一", 0.6, 280.0)])
            right_sources = _sources(root / "right", [("tiny0001", "样本二", 0.6, 280.0)])
            left = root / "left-out"
            right = root / "right-out"
            common = ["--config", str(config), "--seed", "42", "--git-sha", "fixture"]
            self.assertEqual(build_main([*common, "--sources", str(left_sources), "--output-dir", str(left)]), 3)
            self.assertEqual(build_main([*common, "--sources", str(right_sources), "--output-dir", str(right)]), 3)
            left_hash = hashlib.sha256((left / "train.jsonl").read_bytes() + (left / "val.jsonl").read_bytes() + (left / "test.jsonl").read_bytes()).hexdigest()
            right_hash = hashlib.sha256((right / "train.jsonl").read_bytes() + (right / "val.jsonl").read_bytes() + (right / "test.jsonl").read_bytes()).hexdigest()
            self.assertNotEqual(left_hash, right_hash)


if __name__ == "__main__":
    unittest.main()
