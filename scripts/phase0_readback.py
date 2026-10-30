#!/usr/bin/env python3
"""Copy Phase 0 artifacts and read them back by SHA-256.

This is the Phase 0 persistence check. Checkpoint latest.json handling starts
in Phase 2. A byte that changes between the copy and the read fails the check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def readback_files(source_dir: Path, dest_dir: Path, names: list[str]) -> dict[str, object]:
    """Copy named files, then hash the destination bytes again."""
    if not names:
        raise ValueError("readback requires at least one file name")
    dest_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for name in names:
        source = source_dir / name
        if not source.is_file():
            raise FileNotFoundError(f"missing artifact: {source}")
        target = dest_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        source_hash = sha256_file(source)
        dest_hash = sha256_file(target)
        records.append({
            "name": name,
            "bytes": target.stat().st_size,
            "source_sha256": source_hash,
            "readback_sha256": dest_hash,
            "match": source_hash == dest_hash,
        })
    matched = all(bool(record["match"]) for record in records)
    return {"matched": matched, "files": records}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="复制产物并按 SHA-256 读回")
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--dest-dir", required=True, type=Path)
    parser.add_argument("--name", action="append", required=True, dest="names")
    parser.add_argument("--report", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = readback_files(args.source_dir, args.dest_dir, args.names)
    except (OSError, ValueError) as exc:
        print(f"readback failed: {exc}", file=sys.stderr)
        return 2
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["matched"]:
        print("readback hash mismatch", file=sys.stderr)
        return 1
    print(json.dumps({"matched": True, "files": len(report["files"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
