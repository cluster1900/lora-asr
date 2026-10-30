"""Pilot sampler contract tests.

The fixture is a few synthetic utterances. It is not an audited corpus.
"""

from __future__ import annotations

import json
import tarfile
import unittest
import wave
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.sample_aishell_pilot import catalog_archive, main


def _wav(seconds: float, sample: int) -> bytes:
    frames = int(round(seconds * 16000))
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(sample.to_bytes(2, "little", signed=True) * frames)
    return buffer.getvalue()


def _speaker_tar(split: str, speaker: str, wavs: list[tuple[str, bytes]]) -> bytes:
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for utterance_id, payload in wavs:
            info = tarfile.TarInfo(f"{split}/{speaker}/{utterance_id}.wav")
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))
    return buffer.getvalue()


class SampleAishellPilotTest(unittest.TestCase):
    def test_catalog_reads_inner_split_and_clipping(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "data_aishell.tgz"
            speaker = _speaker_tar(
                "train",
                "S0002",
                [
                    ("BAC009S0002W0001", _wav(0.5, 1000)),
                    ("BAC009S0002W0002", _wav(0.5, 32767)),
                ],
            )
            with tarfile.open(archive, "w:gz") as outer:
                note = b"not-audio\n"
                info = tarfile.TarInfo("data_aishell/transcript/readme.txt")
                info.size = len(note)
                outer.addfile(info, BytesIO(note))
                payload = speaker
                info = tarfile.TarInfo("data_aishell/wav/S0002.tar.gz")
                info.size = len(payload)
                outer.addfile(info, BytesIO(payload))
            summary = catalog_archive(archive, root / "catalog")
            rows = [
                json.loads(line)
                for line in (root / "catalog" / "durations.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(summary["wavs"], 2)
            self.assertEqual(summary["problems"]["hard_clip"], 1)
            self.assertEqual({row["official_split"] for row in rows}, {"train"})
            self.assertEqual({row["speaker_id"] for row in rows}, {"S0002"})

    def test_select_keeps_official_text_and_rejects_bad_rows(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript = root / "transcript.txt"
            durations = root / "durations.jsonl"
            lines = [
                "BAC009S0001W0001 今天 天气 不错",
                "BAC009S0001W0002 明天 继续 出发",
                "BAC009S0001W0003 这条 被 削波",
                "BAC009S0001W0004 同一 句话",
                "BAC009S0002W0001 市场 正在 回暖",
                "BAC009S0002W0002 出口 保持 增长",
                "BAC009S0002W0003 同一 句话",
                "BAC009S0724W0001 发展 仍需 时间",
                "BAC009S0724W0002 会议 已经 结束",
                "BAC009S0724W0003 全角 ａ级 景区",
                "BAC009S0900W0001 试验 已经 完成",
                "BAC009S0900W0002 结果 将要 公布",
                "BAC009S0900W0003 CEO 宣布 结果",
                "BAC009S0001W0009 没有 音频",
            ]
            transcript.write_text("\n".join(lines) + "\n", encoding="utf-8")
            rows = []
            for utterance_id, split, speaker, clipped in (
                ("BAC009S0001W0001", "train", "S0001", 0),
                ("BAC009S0001W0002", "train", "S0001", 0),
                ("BAC009S0001W0003", "train", "S0001", 4),
                ("BAC009S0001W0004", "train", "S0001", 0),
                ("BAC009S0002W0001", "train", "S0002", 0),
                ("BAC009S0002W0002", "train", "S0002", 0),
                ("BAC009S0002W0003", "train", "S0002", 0),
                ("BAC009S0724W0001", "dev", "S0724", 0),
                ("BAC009S0724W0002", "dev", "S0724", 0),
                ("BAC009S0724W0003", "dev", "S0724", 0),
                ("BAC009S0900W0001", "test", "S0900", 0),
                ("BAC009S0900W0002", "test", "S0900", 0),
                ("BAC009S0900W0003", "test", "S0900", 0),
                ("BAC009S0724W0099", "dev", "S0724", 0),
            ):
                rows.append(
                    {
                        "clipped_samples": clipped,
                        "duration_seconds": 1.5,
                        "format_ok": True,
                        "official_split": split,
                        "speaker_id": speaker,
                        "utterance_id": utterance_id,
                    }
                )
            durations.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            output = root / "sample"
            code = main(
                [
                    "select",
                    "--transcript",
                    str(transcript),
                    "--durations",
                    str(durations),
                    "--output-dir",
                    str(output),
                    "--seed",
                    "42",
                    "--train-seconds",
                    "6",
                    "--min-train-speakers",
                    "2",
                    "--dev-count",
                    "2",
                    "--test-count",
                    "2",
                    "--expect-speakers",
                    "train=2,dev=1,test=1",
                ]
            )
            self.assertEqual(code, 0)
            source = [
                json.loads(line)
                for line in (output / "source.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            ids = {row["source_utterance_id"] for row in source}
            self.assertNotIn("BAC009S0001W0003", ids)
            self.assertNotIn("BAC009S0724W0003", ids)
            self.assertNotIn("BAC009S0900W0003", ids)
            rejected_rows = [
                json.loads(line)
                for line in (output / "rejected.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            duplicate_ids = {row["utterance_id"] for row in rejected_rows if row["reason"] == "duplicate_text"}
            self.assertEqual(len(duplicate_ids), 1)
            self.assertTrue(duplicate_ids <= {"BAC009S0001W0004", "BAC009S0002W0003"})
            self.assertTrue(duplicate_ids.isdisjoint(ids))
            self.assertTrue(all(row["language"] == "zh" for row in source))
            self.assertTrue(all(row["source_package"] == "aishell1-openslr33" for row in source))
            report = json.loads((output / "selection.json").read_text(encoding="utf-8"))
            self.assertGreaterEqual(report["train_seconds"], 6)
            self.assertGreaterEqual(report["train_speakers"], 2)
            self.assertEqual(report["dev_utterances"], 2)
            self.assertEqual(report["test_utterances"], 2)
            selected = [
                json.loads(line)
                for line in (output / "selected.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            by_speaker: dict[str, int] = {}
            for row in selected:
                if row["official_split"] != "train":
                    continue
                by_speaker[row["speaker_id"]] = by_speaker.get(row["speaker_id"], 0) + row["duration_ms"]
            self.assertTrue(all(total <= report["train_cap_ms"] for total in by_speaker.values()))
            reasons = {
                json.loads(line)["reason"]
                for line in (output / "rejected.jsonl").read_text(encoding="utf-8").splitlines()
            }
            self.assertIn("fullwidth_latin", reasons)
            self.assertIn("hard_clip", reasons)
            self.assertIn("text_has_latin", reasons)
            self.assertIn("duplicate_text", reasons)
            self.assertIn("missing_duration", reasons)
            self.assertIn("missing_transcript", reasons)
            again = root / "sample2"
            second = main(
                [
                    "select",
                    "--transcript",
                    str(transcript),
                    "--durations",
                    str(durations),
                    "--output-dir",
                    str(again),
                    "--seed",
                    "42",
                    "--train-seconds",
                    "6",
                    "--min-train-speakers",
                    "2",
                    "--dev-count",
                    "2",
                    "--test-count",
                    "2",
                    "--expect-speakers",
                    "train=2,dev=1,test=1",
                ]
            )
            self.assertEqual(second, 0)
            self.assertEqual(
                (output / "source.jsonl").read_bytes(),
                (again / "source.jsonl").read_bytes(),
            )

    def test_short_train_exits_without_source(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            transcript = root / "transcript.txt"
            durations = root / "durations.jsonl"
            transcript.write_text("BAC009S0001W0001 只有 一句\n", encoding="utf-8")
            durations.write_text(
                json.dumps(
                    {
                        "clipped_samples": 0,
                        "duration_seconds": 1.5,
                        "format_ok": True,
                        "official_split": "train",
                        "speaker_id": "S0001",
                        "utterance_id": "BAC009S0001W0001",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "sample"
            code = main(
                [
                    "select",
                    "--transcript",
                    str(transcript),
                    "--durations",
                    str(durations),
                    "--output-dir",
                    str(output),
                    "--seed",
                    "42",
                    "--train-seconds",
                    "6",
                    "--min-train-speakers",
                    "2",
                    "--dev-count",
                    "1",
                    "--test-count",
                    "1",
                    "--expect-speakers",
                    "train=1,dev=0,test=0",
                ]
            )
            self.assertEqual(code, 3)
            self.assertFalse((output / "source.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
