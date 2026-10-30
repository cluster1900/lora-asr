#!/usr/bin/env python3
"""Select an AISHELL-1 pilot source without marking it audited.

catalog reads the OpenSLR tarball and writes one duration row per wav.
select writes a builder-ready source JSONL. It does not edit the formal
config, does not set audited, and does not start training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
import wave
from array import array
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CATALOG_SCHEMA = "qwen3-asr-aishell-duration-v1"
SELECTION_SCHEMA = "qwen3-asr-aishell-pilot-sample-v1"
SOURCE_PACKAGE = "aishell1-openslr33"
RANK_LABEL = "pilot-v1"
DEDUPE_LABEL = "dedupe"
EXPECTED_SPEAKERS = {"train": 340, "dev": 40, "test": 20}
FULLWIDTH_LATIN = tuple(chr(code) for code in range(0xFF21, 0xFF3B)) + tuple(
    chr(code) for code in range(0xFF41, 0xFF5B)
)


def _rank(seed: int, utterance_id: str, label: str) -> str:
    return hashlib.sha256(f"{seed}:{utterance_id}:{label}".encode("utf-8")).hexdigest()


def _speaker_id(utterance_id: str) -> str | None:
    if not utterance_id.startswith("BAC009S") or "W" not in utterance_id:
        return None
    speaker = utterance_id[6:11]
    if len(speaker) != 5 or speaker[0] != "S" or not speaker[1:].isdigit():
        return None
    return speaker


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip() == "":
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"{path}:{line_number} must be an object")
        rows.append(item)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _pcm16_stats(blob: bytes) -> dict[str, Any]:
    """Read a PCM16 WAV from memory and measure duration and hard clips."""
    try:
        with wave.open(BytesIO(blob), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            frames = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError):
        return {"format_ok": False, "format_problem": "bad_wav"}
    if channels != 1 or width != 2 or rate != 16000:
        return {
            "format_ok": False,
            "format_problem": "bad_format",
            "channels": channels,
            "sample_width": width,
            "sample_rate": rate,
        }
    samples = array("h")
    if len(frames) % 2:
        return {"format_ok": False, "format_problem": "odd_pcm_bytes"}
    samples.frombytes(frames)
    clipped = sum(1 for sample in samples if sample in (32767, -32768))
    return {
        "format_ok": True,
        "duration_seconds": len(samples) / rate,
        "clipped_samples": clipped,
    }


def _member_split(name: str) -> str | None:
    parts = [part for part in name.lstrip("./").split("/") if part]
    if not parts:
        return None
    return parts[0] if parts[0] in EXPECTED_SPEAKERS else None


def _utterance_id(name: str) -> str | None:
    stem = Path(name).name
    if not stem.endswith(".wav"):
        return None
    utterance_id = stem[:-4]
    return utterance_id if _speaker_id(utterance_id) else None


def catalog_archive(archive: Path, output_dir: Path) -> dict[str, Any]:
    """Stream speaker tarballs and write one duration row per wav."""
    output_dir.mkdir(parents=True, exist_ok=True)
    durations_path = output_dir / "durations.jsonl"
    speakers: set[str] = set()
    conflicts: list[str] = []
    with durations_path.open("w", encoding="utf-8") as handle, tarfile.open(archive, "r|gz") as outer:
        for member in outer:
            source = outer.extractfile(member)
            name = member.name
            speaker = Path(name).name[:5]
            usable = (
                source is not None
                and name.endswith(".tar.gz")
                and "/wav/S" in f"/{name}"
                and _speaker_id(f"BAC009{speaker}W0000") == speaker
            )
            if not usable:
                if source is not None:
                    while source.read(1024 * 1024):
                        pass
                continue
            split_seen: set[str] = set()
            with tarfile.open(fileobj=source, mode="r|gz") as inner:
                for wav_member in inner:
                    utterance_id = _utterance_id(wav_member.name)
                    split = _member_split(wav_member.name)
                    if utterance_id is None or split is None:
                        continue
                    blob_file = inner.extractfile(wav_member)
                    blob = b"" if blob_file is None else blob_file.read()
                    stats = _pcm16_stats(blob)
                    row = {
                        "clipped_samples": stats.get("clipped_samples", 0),
                        "duration_seconds": stats.get("duration_seconds"),
                        "format_ok": bool(stats.get("format_ok")),
                        "format_problem": stats.get("format_problem"),
                        "official_split": split,
                        "speaker_id": speaker,
                        "utterance_id": utterance_id,
                    }
                    handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                    split_seen.add(split)
            while source.read(1024 * 1024):
                pass
            if len(split_seen) != 1:
                conflicts.append(speaker)
            speakers.add(speaker)
            if len(speakers) % 20 == 0:
                print(f"CATALOG_SPEAKERS {len(speakers)}", flush=True)
    rows = _read_jsonl(durations_path)
    problems: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        split = row.get("official_split")
        if isinstance(split, str):
            counts[split] += 1
        if row.get("format_ok") and row.get("clipped_samples"):
            problems["hard_clip"] += 1
        if not row.get("format_ok"):
            problems[str(row.get("format_problem") or "bad_wav")] += 1
    summary = {
        "archive": str(archive),
        "conflict_speakers": conflicts,
        "problems": dict(problems),
        "schema": CATALOG_SCHEMA,
        "speakers": len(speakers),
        "split_counts": dict(counts),
        "wavs": len(rows),
    }
    (output_dir / "catalog.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"CATALOG_DONE speakers={len(speakers)} wavs={len(rows)}", flush=True)
    return summary


def load_transcript(path: Path) -> dict[str, str]:
    """Return joined transcript text keyed by utterance id."""
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        utterance_id, _, rest = line.partition(" ")
        if _speaker_id(utterance_id) is None:
            continue
        rows[utterance_id] = "".join(rest.split())
    return rows


def _duration_ms(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return int(round(float(value) * 1000))


def _reject(bucket: list[dict[str, Any]], utterance_id: str, reason: str, speaker: str | None, split: str | None) -> None:
    bucket.append(
        {
            "official_split": split,
            "reason": reason,
            "speaker_id": speaker,
            "utterance_id": utterance_id,
        }
    )


def prepare_rows(
    transcript: dict[str, str],
    duration_rows: list[dict[str, Any]],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Join text and audio measurements, then drop rows that cannot be sampled."""
    from scripts.manifest_contract import text_problem

    rejected: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for row in duration_rows:
        utterance_id = row.get("utterance_id")
        speaker = row.get("speaker_id")
        split = row.get("official_split")
        if not isinstance(utterance_id, str) or _speaker_id(utterance_id) != speaker:
            _reject(rejected, str(utterance_id), "bad_duration_row", speaker if isinstance(speaker, str) else None, None)
            continue
        if utterance_id in by_id or utterance_id in duplicate_ids:
            duplicate_ids.add(utterance_id)
            by_id.pop(utterance_id, None)
            continue
        by_id[utterance_id] = row
    for utterance_id in sorted(duplicate_ids):
        _reject(rejected, utterance_id, "duplicate_duration", _speaker_id(utterance_id), None)

    eligible: list[dict[str, Any]] = []
    for utterance_id, text in transcript.items():
        speaker = _speaker_id(utterance_id)
        measured = by_id.pop(utterance_id, None)
        split = measured.get("official_split") if measured else None
        if measured is None:
            _reject(rejected, utterance_id, "missing_duration", speaker, None)
            continue
        if split not in EXPECTED_SPEAKERS or measured.get("speaker_id") != speaker:
            _reject(rejected, utterance_id, "bad_official_split", speaker, split if isinstance(split, str) else None)
            continue
        problem = text_problem(text)
        if problem:
            _reject(rejected, utterance_id, problem, speaker, split)
            continue
        if any(mark in text for mark in FULLWIDTH_LATIN):
            _reject(rejected, utterance_id, "fullwidth_latin", speaker, split)
            continue
        if measured.get("format_ok") is False:
            _reject(rejected, utterance_id, "bad_format", speaker, split)
            continue
        clipped = measured.get("clipped_samples", 0)
        if isinstance(clipped, int) and clipped > 0:
            _reject(rejected, utterance_id, "hard_clip", speaker, split)
            continue
        duration_ms = _duration_ms(measured.get("duration_seconds"))
        if duration_ms is None or not 500 <= duration_ms <= 30000:
            _reject(rejected, utterance_id, "duration_out_of_range", speaker, split)
            continue
        eligible.append(
            {
                "duration_ms": duration_ms,
                "official_split": split,
                "speaker_id": speaker,
                "text": text,
                "utterance_id": utterance_id,
            }
        )
    for utterance_id, measured in by_id.items():
        _reject(
            rejected,
            utterance_id,
            "missing_transcript",
            measured.get("speaker_id") if isinstance(measured.get("speaker_id"), str) else None,
            measured.get("official_split") if isinstance(measured.get("official_split"), str) else None,
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[row["text"]].append(row)
    kept: list[dict[str, Any]] = []
    for text, rows in grouped.items():
        rows.sort(key=lambda item: _rank(seed, item["utterance_id"], DEDUPE_LABEL))
        kept.append(rows[0])
        for item in rows[1:]:
            _reject(rejected, item["utterance_id"], "duplicate_text", item["speaker_id"], item["official_split"])
    return kept, rejected


def _by_speaker(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["speaker_id"]].append(row)
    for speaker_rows in grouped.values():
        speaker_rows.sort(key=lambda item: _rank(seed, item["utterance_id"], RANK_LABEL))
    return grouped


def select_round_robin(
    rows: list[dict[str, Any]],
    seed: int,
    target_ms: int | None,
    cap_ms: int | None,
    count: int | None,
) -> list[dict[str, Any]] | None:
    """Take utterances speaker by speaker until the duration or count is met."""
    grouped = _by_speaker(rows, seed)
    speakers = sorted(grouped)
    index = {speaker: 0 for speaker in speakers}
    used_ms = {speaker: 0 for speaker in speakers}
    chosen: list[dict[str, Any]] = []
    total_ms = 0
    progressed = True
    while progressed:
        if target_ms is not None and total_ms >= target_ms:
            break
        if count is not None and len(chosen) >= count:
            break
        progressed = False
        for speaker in speakers:
            if target_ms is not None and total_ms >= target_ms:
                break
            if count is not None and len(chosen) >= count:
                break
            cursor = index[speaker]
            items = grouped[speaker]
            if cap_ms is not None:
                while cursor < len(items) and used_ms[speaker] + items[cursor]["duration_ms"] > cap_ms:
                    cursor += 1
            if cursor >= len(items):
                index[speaker] = cursor
                continue
            item = items[cursor]
            index[speaker] = cursor + 1
            chosen.append(item)
            used_ms[speaker] += item["duration_ms"]
            total_ms += item["duration_ms"]
            progressed = True
    if target_ms is not None and total_ms < target_ms:
        return None
    if count is not None and len(chosen) < count:
        return None
    return chosen


def select_pilot(
    transcript: dict[str, str],
    duration_rows: list[dict[str, Any]],
    output_dir: Path,
    seed: int,
    train_seconds: float,
    min_train_speakers: int,
    dev_count: int,
    test_count: int,
    expect_speakers: dict[str, int],
) -> dict[str, Any]:
    """Write the pilot source or an insufficient-data report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "source.jsonl"
    if source_path.exists():
        source_path.unlink()
    eligible, rejected = prepare_rows(transcript, duration_rows, seed)
    observed = {
        split: len({row["speaker_id"] for row in duration_rows if row.get("official_split") == split})
        for split in EXPECTED_SPEAKERS
    }
    if observed != expect_speakers:
        report = {
            "expect_speakers": expect_speakers,
            "observed_speakers": observed,
            "schema": SELECTION_SCHEMA,
            "seed": seed,
            "status": "bad_speaker_counts",
        }
        (output_dir / "selection.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise ValueError(f"speaker counts {observed} != {expect_speakers}")

    target_ms = int(round(train_seconds * 1000))
    cap_ms = target_ms // min_train_speakers
    train_rows = [row for row in eligible if row["official_split"] == "train"]
    dev_rows = [row for row in eligible if row["official_split"] == "dev"]
    test_rows = [row for row in eligible if row["official_split"] == "test"]
    train = select_round_robin(train_rows, seed, target_ms, cap_ms, None)
    dev = select_round_robin(dev_rows, seed, None, None, dev_count)
    test = select_round_robin(test_rows, seed, None, None, test_count)
    train_speakers = 0 if train is None else len({row["speaker_id"] for row in train})
    status = "selected"
    if train is None or dev is None or test is None or train_speakers < min_train_speakers:
        status = "insufficient_data"
        train = [] if train is None else train
        dev = [] if dev is None else dev
        test = [] if test is None else test
    chosen = train + dev + test
    reason_counts: dict[str, int] = defaultdict(int)
    for row in rejected:
        reason_counts[row["reason"]] += 1
    report = {
        "dev_speakers": len({row["speaker_id"] for row in dev}),
        "dev_utterances": len(dev),
        "min_train_speakers": min_train_speakers,
        "rejected_counts": dict(reason_counts),
        "schema": SELECTION_SCHEMA,
        "seed": seed,
        "source_package": SOURCE_PACKAGE,
        "status": status,
        "test_speakers": len({row["speaker_id"] for row in test}),
        "test_utterances": len(test),
        "train_cap_ms": cap_ms,
        "train_seconds": round(sum(row["duration_ms"] for row in train) / 1000, 3),
        "train_speakers": train_speakers,
        "train_target_seconds": train_seconds,
        "train_utterances": len(train),
    }
    _write_jsonl(output_dir / "rejected.jsonl", rejected)
    selected_path = output_dir / "selected.jsonl"
    if status != "selected":
        if selected_path.exists():
            selected_path.unlink()
    else:
        _write_jsonl(
            selected_path,
            [
                {
                    "duration_ms": row["duration_ms"],
                    "official_split": row["official_split"],
                    "speaker_id": row["speaker_id"],
                    "utterance_id": row["utterance_id"],
                }
                for row in chosen
            ],
        )
        source_rows = [
            {
                "audio": f"audio/{row['official_split']}/{row['utterance_id']}.wav",
                "language": "zh",
                "official_split": row["official_split"],
                "source_package": SOURCE_PACKAGE,
                "source_utterance_id": row["utterance_id"],
                "speaker_id": row["speaker_id"],
                "text": row["text"],
            }
            for row in sorted(chosen, key=lambda item: item["utterance_id"])
        ]
        _write_jsonl(source_path, source_rows)
    (output_dir / "selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "train_seconds": report["train_seconds"], "train_speakers": train_speakers}, ensure_ascii=False), flush=True)
    return report


def _parse_expect(value: str) -> dict[str, int]:
    found = {}
    for part in value.split(","):
        split, _, count = part.partition("=")
        if split not in EXPECTED_SPEAKERS or not count.isdigit():
            raise ValueError(f"bad --expect-speakers {value}")
        found[split] = int(count)
    if set(found) != set(EXPECTED_SPEAKERS):
        raise ValueError("--expect-speakers needs train, dev, and test")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    catalog = sub.add_parser("catalog")
    catalog.add_argument("--archive", required=True, type=Path)
    catalog.add_argument("--output-dir", required=True, type=Path)
    select = sub.add_parser("select")
    select.add_argument("--transcript", required=True, type=Path)
    select.add_argument("--durations", required=True, type=Path)
    select.add_argument("--output-dir", required=True, type=Path)
    select.add_argument("--seed", required=True, type=int)
    select.add_argument("--train-seconds", type=float, default=18000)
    select.add_argument("--min-train-speakers", type=int, default=50)
    select.add_argument("--dev-count", type=int, default=500)
    select.add_argument("--test-count", type=int, default=500)
    select.add_argument("--expect-speakers", default="train=340,dev=40,test=20")
    args = parser.parse_args(argv)
    try:
        if args.command == "catalog":
            if not args.archive.is_file():
                print(f"missing archive {args.archive}", file=sys.stderr)
                return 2
            catalog_archive(args.archive, args.output_dir)
            return 0
        if not args.transcript.is_file() or not args.durations.is_file():
            print("missing transcript or durations", file=sys.stderr)
            return 2
        if args.seed < 0 or args.min_train_speakers < 1:
            print("seed and speaker minimum must be non-negative", file=sys.stderr)
            return 2
        report = select_pilot(
            load_transcript(args.transcript),
            _read_jsonl(args.durations),
            args.output_dir,
            args.seed,
            args.train_seconds,
            args.min_train_speakers,
            args.dev_count,
            args.test_count,
            _parse_expect(args.expect_speakers),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0 if report["status"] == "selected" else 3


if __name__ == "__main__":
    raise SystemExit(main())
