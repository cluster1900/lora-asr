#!/usr/bin/env python3
"""Build the v31 extra verdict manifest (dpo_val_pool degraded rows).

docs/qwen3-asr/29_rl_v31_scale_design.md: the RL v31 paired verdict uses the
full ``validation`` role plus the degraded rows of ``dpo_val_pool``. The second
set was only used for DPO preference validation (never trained on, never seen
by RL). This script copies those rows unchanged, refuses any overlap with the
six training/evaluation roles on ``sample_id``, ``source_utterance_id`` or
``audio_sha256``, and writes ``<output>.COMPLETE.json`` with counts and hashes.

It never edits its inputs. Re-running with identical inputs reproduces the
same bytes; an existing output with a different sha256 is refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

OVERLAP_KEYS = ("sample_id", "source_utterance_id", "audio_sha256")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} is not a JSON object")
            rows.append(row)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_identity_rows(rows: Sequence[Dict[str, Any]], label: str) -> None:
    """Require complete, unique identity keys before doing overlap checks."""
    for key in OVERLAP_KEYS:
        values: List[str] = []
        for index, row in enumerate(rows, start=1):
            value = row.get(key)
            if value is None or not str(value).strip():
                raise ValueError(f"{label} row {index} has no {key}")
            values.append(str(value).strip())
        dupes = [value for value, count in Counter(values).items() if count > 1]
        if dupes:
            raise ValueError(f"duplicate {key} in {label}: {dupes[:3]}")


def select_degraded(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected = [row for row in rows if row.get("condition_group") == "degraded"]
    validate_identity_rows(selected, "degraded source")
    for row in selected:
        if not str(row.get("text", "")).strip():
            raise ValueError(f"{row['sample_id']} has no reference text")
        if not str(row.get("audio", "")).strip():
            raise ValueError(f"{row['sample_id']} has no audio path")
    return selected


def find_overlaps(
    selected: Sequence[Dict[str, Any]],
    excluded: Dict[str, Sequence[Dict[str, Any]]],
) -> Dict[str, Dict[str, int]]:
    """Return {role: {key: count}} for every non-empty intersection."""
    found: Dict[str, Dict[str, int]] = {}
    for role, rows in excluded.items():
        for key in OVERLAP_KEYS:
            ours = {str(row[key]).strip() for row in selected}
            theirs = {str(row[key]).strip() for row in rows}
            hit = len(ours & theirs)
            if hit:
                found.setdefault(role, {})[key] = hit
    return found


def build(
    source: Path,
    excludes: Sequence[Path],
    output: Path,
    *,
    check_audio: bool = False,
) -> Dict[str, Any]:
    selected = select_degraded(read_jsonl(source))
    if not selected:
        raise ValueError(f"{source} has no degraded rows")
    excluded = {str(path): read_jsonl(path) for path in excludes}
    for path, rows in excluded.items():
        validate_identity_rows(rows, f"excluded manifest {path}")
    overlaps = find_overlaps(selected, excluded)
    if overlaps:
        raise ValueError(f"verdict rows overlap excluded roles: {json.dumps(overlaps, sort_keys=True)}")
    if check_audio:
        missing = [row["sample_id"] for row in selected if not Path(row["audio"]).is_file()]
        if missing:
            raise ValueError(f"{len(missing)} audio files missing, e.g. {missing[:3]}")

    body = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected)
    new_sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if output.exists():
        old_sha = sha256_file(output)
        if old_sha != new_sha:
            raise ValueError(f"{output} exists with sha256 {old_sha}, rebuild would be {new_sha}; refusing")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        tmp = output.with_suffix(output.suffix + ".tmp")
        tmp.write_text(body, encoding="utf-8")
        tmp.replace(output)

    complete = {
        "schema_version": 1,
        "role": "rl_verdict_extra_degraded",
        "design": "docs/qwen3-asr/29_rl_v31_scale_design.md",
        "manifest_path": str(output),
        "manifest_sha256": new_sha,
        "row_count": len(selected),
        "source": {"path": str(source), "sha256": sha256_file(source)},
        "excluded": {path: sha256_file(Path(path)) for path in excluded},
        "overlap_keys": list(OVERLAP_KEYS),
        "overlaps": {},
        "audio_checked": bool(check_audio),
        "by_language": dict(sorted(Counter(r.get("language") for r in selected).items())),
        "by_language_scenario": dict(
            sorted(Counter(f"{r.get('language')}|{r.get('scenario')}" for r in selected).items())
        ),
        "status": "PASSED",
    }
    complete_path = output.with_name(output.name + ".COMPLETE.json")
    complete_path.write_text(json.dumps(complete, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return complete


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="dpo_val_pool.jsonl")
    parser.add_argument(
        "--exclude", type=Path, nargs="+", required=True,
        help="Roles that must not overlap; every row needs all three identity keys.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-audio", action="store_true", help="Require every audio path to exist.")
    args = parser.parse_args(argv)
    try:
        complete = build(args.source, args.exclude, args.output, check_audio=args.check_audio)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(
        f"rows={complete['row_count']} sha256={complete['manifest_sha256']} "
        f"by_language={complete['by_language']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
