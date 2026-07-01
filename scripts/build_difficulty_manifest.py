#!/usr/bin/env python3
"""Build a base-difficulty manifest from scored ASR predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_BUCKETS = (
    ("wer_0_10", 0.0, 0.1),
    ("wer_10_30", 0.1, 0.3),
    ("wer_30_50", 0.3, 0.5),
    ("wer_50_70", 0.5, 0.7),
    ("wer_70_plus", 0.7, None),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL and skip blank lines."""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bucket_for_wer(value: float) -> str:
    """Map WER/CER to the fixed v6 difficulty bucket."""
    for name, lower, upper in DEFAULT_BUCKETS:
        if value >= lower and (upper is None or value < upper):
            return name
    return "wer_70_plus"


def token_set(text: str) -> set[str]:
    """Tokenize normalized English-ish text for overlap heuristics."""
    return {token for token in str(text or "").lower().split() if token}


def token_list(text: str) -> list[str]:
    """Tokenize normalized text for repetition heuristics."""
    return [token for token in str(text or "").lower().split() if token]


def has_repeated_bigram(tokens: list[str]) -> bool:
    """Detect repeated bigrams in a prediction."""
    if len(tokens) < 4:
        return False
    bigrams = list(zip(tokens, tokens[1:]))
    counts = Counter(bigrams)
    return any(count >= 2 for count in counts.values())


def has_run_repetition(tokens: list[str], min_run: int = 3) -> bool:
    """Detect a run of the same token."""
    if not tokens:
        return False
    run = 1
    previous = tokens[0]
    for token in tokens[1:]:
        if token == previous:
            run += 1
            if run >= min_run:
                return True
        else:
            previous = token
            run = 1
    return False


def failure_tags(row: dict[str, Any]) -> list[str]:
    """Build failure tags for training-time sampling and review."""
    tags: list[str] = []
    existing = row.get("error_tags")
    if isinstance(existing, list):
        tags.extend(str(tag) for tag in existing)
    elif isinstance(existing, str) and existing.strip():
        tags.extend(tag for tag in existing.split("|") if tag)

    wer = float(row.get("wer", 0.0) or 0.0)
    num_edits = int(row.get("num_edits", 0) or 0)
    ref_len = int(row.get("ref_len", 0) or 0)
    length_ratio = float(row.get("length_ratio", 0.0) or 0.0)
    pred_norm = str(row.get("prediction_normalized") or row.get("prediction") or "")
    ref_tokens = token_set(str(row.get("reference_normalized") or row.get("answer") or ""))
    pred_tokens = token_set(pred_norm)
    pred_token_list = token_list(pred_norm)

    if row.get("error"):
        tags.append("inference_error")
    if bool(row.get("empty_output", False)) or not pred_norm.strip():
        tags.append("empty_output")
    if ref_len > 0 and length_ratio < 0.5:
        tags.append("too_short")
    if ref_len > 0 and length_ratio > 1.5:
        tags.append("too_long")
    if has_repeated_bigram(pred_token_list) or has_run_repetition(pred_token_list):
        tags.append("repeat_like")
    if wer == 0.0:
        tags.append("base_correct")
    if 0.0 < wer < 0.1:
        tags.append("minor_error")
    if wer >= 0.3:
        tags.append("moderate_error")
    if wer >= 0.5:
        tags.append("hard_error")
    if wer >= 0.7:
        tags.append("very_hard_error")
    if wer >= 0.8 and pred_norm.strip():
        tags.append("hallucination_like")
    if wer >= 1.0:
        tags.append("insertion_heavy")
    if ref_len and num_edits >= max(3, ref_len // 2):
        tags.append("many_edits")
    if ref_tokens and pred_tokens and not (ref_tokens & pred_tokens):
        tags.append("low_token_overlap")

    deduped: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        if tag not in seen:
            deduped.append(tag)
            seen.add(tag)
    return deduped


def difficulty_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    """Convert one scored prediction row to a difficulty manifest row."""
    wer = round(float(row.get("wer", 0.0) or 0.0), 6)
    out = dict(row)
    out["split"] = split or row.get("split", "")
    out["base_prediction"] = str(row.get("prediction") or "")
    out["base_prediction_normalized"] = str(row.get("prediction_normalized") or "")
    out["base_wer"] = wer
    out["base_num_edits"] = int(row.get("num_edits", 0) or 0)
    out["base_ref_len"] = int(row.get("ref_len", 0) or 0)
    out["base_metric"] = str(row.get("metric") or "wer")
    out["difficulty_bucket"] = bucket_for_wer(wer)
    out["failure_tags"] = failure_tags(row)
    out["base_empty_output"] = bool(row.get("empty_output", False))
    out["base_length_ratio"] = float(row.get("length_ratio", 0.0) or 0.0)
    out["base_error"] = str(row.get("error") or "")

    # Keep the raw prediction fields for traceability but make base_* the
    # contract consumed by training configs.
    out.pop("prediction", None)
    out.pop("prediction_normalized", None)
    out.pop("num_edits", None)
    out.pop("ref_len", None)
    out.pop("empty_output", None)
    out.pop("length_ratio", None)
    out.pop("error_tags", None)
    return out


def aggregate(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    """Aggregate difficulty rows by one or more keys."""
    buckets: dict[tuple[str, ...], dict[str, Any]] = defaultdict(lambda: {
        "samples": 0,
        "base_num_edits": 0,
        "base_ref_len": 0,
        "base_empty_outputs": 0,
        "tag_counts": Counter(),
    })
    for row in rows:
        key = tuple(str(row.get(field, "UNKNOWN")) for field in group_keys)
        bucket = buckets[key]
        bucket["samples"] += 1
        bucket["base_num_edits"] += int(row.get("base_num_edits", 0) or 0)
        bucket["base_ref_len"] += int(row.get("base_ref_len", 0) or 0)
        bucket["base_empty_outputs"] += int(bool(row.get("base_empty_output", False)))
        bucket["tag_counts"].update(str(tag) for tag in row.get("failure_tags", []))

    result: list[dict[str, Any]] = []
    for key, bucket in sorted(buckets.items()):
        item = {field: key[index] for index, field in enumerate(group_keys)}
        ref_len = int(bucket["base_ref_len"])
        samples = int(bucket["samples"])
        item.update({
            "samples": samples,
            "base_num_edits": int(bucket["base_num_edits"]),
            "base_ref_len": ref_len,
            "base_error_rate": round(bucket["base_num_edits"] / ref_len, 6) if ref_len else 0.0,
            "base_empty_output_rate": round(bucket["base_empty_outputs"] / samples, 6) if samples else 0.0,
            "tag_counts": dict(sorted(bucket["tag_counts"].items())),
        })
        result.append(item)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write aggregate rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "split",
        "scenario",
        "difficulty_bucket",
        "samples",
        "base_num_edits",
        "base_ref_len",
        "base_error_rate",
        "base_empty_output_rate",
        "tag_counts",
    ]
    keys = [key for key in preferred if any(key in row for row in rows)]
    extras = sorted({key for row in rows for key in row if key not in keys})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys + extras, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            if isinstance(out.get("tag_counts"), dict):
                out["tag_counts"] = json.dumps(out["tag_counts"], ensure_ascii=False, sort_keys=True)
            writer.writerow(out)


def build_difficulty(args: argparse.Namespace) -> dict[str, Any]:
    """Build difficulty rows and summary files."""
    scored_path = Path(args.scored_jsonl).expanduser()
    output_jsonl = Path(args.output_jsonl).expanduser()
    summary_json = Path(args.summary_json).expanduser()
    rows = [difficulty_row(row, args.split) for row in read_jsonl(scored_path)]
    if args.expected_rows >= 0 and len(rows) != args.expected_rows:
        raise RuntimeError(f"row count {len(rows)} != expected {args.expected_rows}")

    write_jsonl(output_jsonl, rows)
    by_scenario = aggregate(rows, ["split", "scenario"])
    by_bucket = aggregate(rows, ["split", "difficulty_bucket"])
    by_scenario_bucket = aggregate(rows, ["split", "scenario", "difficulty_bucket"])
    tag_counts = Counter(tag for row in rows for tag in row.get("failure_tags", []))
    bucket_counts = Counter(str(row.get("difficulty_bucket", "")) for row in rows)
    scenario_counts = Counter(str(row.get("scenario", "")) for row in rows)

    summary = {
        "input_scored_jsonl": str(scored_path),
        "output_jsonl": str(output_jsonl),
        "split": args.split,
        "rows": len(rows),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "difficulty_bucket_counts": dict(sorted(bucket_counts.items())),
        "failure_tag_counts": dict(sorted(tag_counts.items())),
        "by_scenario": by_scenario,
        "by_bucket": by_bucket,
        "by_scenario_bucket": by_scenario_bucket,
    }
    write_json(summary_json, summary)
    if args.summary_csv:
        write_csv(Path(args.summary_csv).expanduser(), by_scenario_bucket)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--summary-csv", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--expected-rows", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_difficulty(args)
    print(json.dumps({
        "rows": summary["rows"],
        "difficulty_bucket_counts": summary["difficulty_bucket_counts"],
        "failure_tag_counts": summary["failure_tag_counts"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
