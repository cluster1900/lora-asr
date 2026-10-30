#!/usr/bin/env python3
"""Write a new DPO pair directory with the dropout silence span removed.

The frozen ``pairs/`` and ``pairs_logprob/`` files are only read. Dropout
rows are normalized, then ``strip_dropout_gap`` removes the 0.40–0.55 span
from both sides. Rows that no longer differ go to ``gap_ties.jsonl``.
Reference logprobs stay null. This script does not train.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.jsonl_io import read_jsonl, write_jsonl  # noqa: E402
from evaluation.zh_cer import NORMALIZATION_VERSION, normalize_zh_cer_v1, score_pair  # noqa: E402
from scripts.build_dpo_pairs import (  # noqa: E402
    AUDIT_SCHEMA,
    PAIR_SCHEMA,
    SCENARIOS,
    char_token_count,
    qwen_token_count,
    select_audit,
    sha256_file,
)
from train.dpo_contract import strip_dropout_gap  # noqa: E402

REPORT_SCHEMA = "qwen3-asr-dpo-gap-strip-report-v1"
METADATA_SCHEMA = "qwen3-asr-dpo-gap-strip-metadata-v1"
OUTPUT_DIR_NAME = "pairs_gap_strip"
FROZEN_DIR_NAMES = {"pairs", "pairs_logprob"}
OUTPUT_NAMES = (
    "train_pairs.jsonl",
    "val_pairs.jsonl",
    "gap_ties.jsonl",
    "overlength_pairs.jsonl",
    "pair_metadata.json",
    "pair_audit.json",
    "pair_report.json",
)


class GapStripError(ValueError):
    """The new pair directory must not be published when this is raised."""


def _refuse_output(source_train: Path, source_val: Path, output_dir: Path) -> None:
    output = output_dir.resolve()
    if output.name != OUTPUT_DIR_NAME:
        raise GapStripError(f"output directory must be named {OUTPUT_DIR_NAME}")
    if output.name in FROZEN_DIR_NAMES:
        raise GapStripError("refusing to write a frozen pair directory")
    sources = [source_train.resolve(), source_val.resolve()]
    for source in sources:
        if output == source.parent or output in source.parents:
            raise GapStripError("refusing to write into the source pair directory")
        if source == output or output in source.parents:
            raise GapStripError("refusing to overwrite a source pair file")
    if any((output / name).exists() for name in OUTPUT_NAMES):
        raise GapStripError("output pair files already exist")


def _load_checked(path: Path, expected_sha: str, split: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise GapStripError(f"missing {split} pairs: {path}")
    actual = sha256_file(path)
    if actual != expected_sha:
        raise GapStripError(f"{split} pair sha256 is {actual}")
    rows = read_jsonl(path)
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if row.get("schema_version") != PAIR_SCHEMA:
            raise GapStripError(f"{split} pair {index} has the wrong schema")
        if row.get("split") != split:
            raise GapStripError(f"{split} pair {index} is marked {row.get('split')}")
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or pair_id in seen:
            raise GapStripError(f"{split} pair {index} has a duplicate or missing pair_id")
        seen.add(pair_id)
        if row.get("scenario") not in SCENARIOS:
            raise GapStripError(f"{pair_id} has an unknown scenario")
        if row.get("chosen_text") != row.get("reference_text"):
            raise GapStripError(f"{pair_id} chosen text is not the reference")
        rejected = row.get("rejected_text")
        if not isinstance(rejected, str) or not rejected.strip():
            raise GapStripError(f"{pair_id} rejected text is empty")
    return rows


def _rewrite_row(
    row: dict[str, Any],
    *,
    count_tokens: Callable[[str], int],
    max_transcript_tokens: int,
) -> dict[str, Any]:
    item = dict(row)
    item["ref_chosen_logp"] = None
    item["ref_rejected_logp"] = None
    item.pop("reason", None)
    item["bucket"] = "pair"
    if row["scenario"] != "dropout":
        return item
    reference = normalize_zh_cer_v1(str(row["chosen_text"]))
    hypothesis = normalize_zh_cer_v1(str(row["rejected_text"]))
    if not reference:
        raise GapStripError(f"{row['pair_id']} reference is empty after normalization")
    chosen, rejected = strip_dropout_gap(reference, hypothesis)
    item["source_reference_text"] = row["reference_text"]
    item["source_rejected_text"] = row["rejected_text"]
    item["gap_strip"] = "dropout-0.40-0.55"
    item["reference_text"] = chosen
    item["chosen_text"] = chosen
    item["rejected_text"] = rejected
    if chosen == rejected or rejected == "":
        item["bucket"] = "gap_tie"
        item["reason"] = "tie_after_strip" if chosen == rejected else "empty_rejected_after_strip"
        return item
    stop_reason = row.get("stop_reason") if isinstance(row.get("stop_reason"), str) else None
    scored = score_pair(chosen, rejected, inference_error=False, stop_reason=stop_reason)
    if scored["prediction_normalized"] == scored["reference_normalized"] or scored["edit_count"] == 0:
        item["bucket"] = "gap_tie"
        item["reason"] = "tie_after_strip"
        return item
    chosen_tokens = count_tokens(chosen)
    rejected_tokens = count_tokens(rejected)
    if chosen_tokens < 1 or rejected_tokens < 1:
        raise GapStripError(f"{row['pair_id']} token count must include EOS")
    item["chosen_cer"] = 0.0
    item["rejected_cer"] = scored["sample_cer"]
    item["chosen_token_count"] = chosen_tokens
    item["rejected_token_count"] = rejected_tokens
    item["normalization_version"] = NORMALIZATION_VERSION
    item["edit_count"] = scored["edit_count"]
    item["reference_char_count"] = scored["reference_char_count"]
    item["rejected_char_count"] = len(scored["prediction_normalized"])
    item["char_length_delta"] = len(scored["prediction_normalized"]) - int(scored["reference_char_count"])
    limit = max_transcript_tokens + 1
    if chosen_tokens > limit or rejected_tokens > limit:
        item["bucket"] = "overlength"
        item["reason"] = "overlength"
    return item


def _count_scenarios(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter(str(row["scenario"]) for row in rows)
    return {name: counts[name] for name in SCENARIOS}


def strip_pair_files(
    *,
    train_pairs: Path,
    val_pairs: Path,
    expect_train_sha256: str,
    expect_val_sha256: str,
    output_dir: Path,
    seed: int,
    max_transcript_tokens: int,
    count_tokens: Callable[[str], int],
    tokenizer_record: dict[str, Any],
    min_val_pairs: int,
    min_pairs_per_scenario: int,
    audit_per_scenario: int,
) -> dict[str, Any]:
    """Hash the frozen inputs, then write a sibling directory. Inputs stay intact."""
    if max_transcript_tokens < 1:
        raise GapStripError("max_transcript_tokens must be positive")
    if seed < 0:
        raise GapStripError("seed must be non-negative")
    _refuse_output(train_pairs, val_pairs, output_dir)
    train_rows = _load_checked(train_pairs, expect_train_sha256, "train")
    val_rows = _load_checked(val_pairs, expect_val_sha256, "val")
    source_train_sha = sha256_file(train_pairs)
    source_val_sha = sha256_file(val_pairs)

    buckets: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "gap_tie": [],
        "overlength": [],
    }
    for split, rows in (("train", train_rows), ("val", val_rows)):
        for row in rows:
            rewritten = _rewrite_row(
                row,
                count_tokens=count_tokens,
                max_transcript_tokens=max_transcript_tokens,
            )
            bucket = str(rewritten["bucket"])
            if bucket == "pair":
                buckets[split].append(rewritten)
            elif bucket == "gap_tie":
                buckets["gap_tie"].append(rewritten)
            elif bucket == "overlength":
                buckets["overlength"].append(rewritten)
            else:
                raise GapStripError(f"{row['pair_id']} landed in {bucket}")

    val_counts = _count_scenarios(buckets["val"])
    short = [name for name in SCENARIOS if val_counts[name] < min_pairs_per_scenario]
    status = "pairs_gap_stripped"
    if len(buckets["val"]) < min_val_pairs or short:
        status = "insufficient_data"
    audit = select_audit(buckets["train"] + buckets["val"], seed, audit_per_scenario)
    if audit.get("schema_version") != AUDIT_SCHEMA or audit.get("status") != "unlistened":
        raise GapStripError("gap-strip audit must stay unlistened")
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": status,
        "trainable": False,
        "blocked_by": ["reference_logprob_pending", "pair_audit_unlistened"],
        "seed": seed,
        "source_train_pairs_sha256": source_train_sha,
        "source_val_pairs_sha256": source_val_sha,
        "max_transcript_tokens": max_transcript_tokens,
        "min_val_pairs": min_val_pairs,
        "min_pairs_per_scenario": min_pairs_per_scenario,
        "train_pairs": len(buckets["train"]),
        "val_pairs": len(buckets["val"]),
        "gap_ties": len(buckets["gap_tie"]),
        "overlength_pairs": len(buckets["overlength"]),
        "train_by_scenario": _count_scenarios(buckets["train"]),
        "val_by_scenario": val_counts,
        "short_val_scenarios": short,
        "tokenizer": tokenizer_record,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / ".gap-strip-staging"
    if staging.exists():
        raise GapStripError(f"staging already exists: {staging}")
    staging.mkdir()
    try:
        write_jsonl(staging / "train_pairs.jsonl", buckets["train"])
        write_jsonl(staging / "val_pairs.jsonl", buckets["val"])
        write_jsonl(staging / "gap_ties.jsonl", buckets["gap_tie"])
        write_jsonl(staging / "overlength_pairs.jsonl", buckets["overlength"])
        report["train_pairs_sha256"] = sha256_file(staging / "train_pairs.jsonl")
        report["val_pairs_sha256"] = sha256_file(staging / "val_pairs.jsonl")
        metadata = {
            "schema_version": METADATA_SCHEMA,
            "status": status,
            "trainable": False,
            "reference_logprob": "pending",
            "score_contract": NORMALIZATION_VERSION,
            "gap_strip": "dropout-0.40-0.55",
            "seed": seed,
            "source_train_pairs_sha256": source_train_sha,
            "source_val_pairs_sha256": source_val_sha,
            "train_pairs_sha256": report["train_pairs_sha256"],
            "val_pairs_sha256": report["val_pairs_sha256"],
            "tokenizer": tokenizer_record,
        }
        (staging / "pair_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "pair_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "pair_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if sha256_file(train_pairs) != source_train_sha or sha256_file(val_pairs) != source_val_sha:
            raise GapStripError("source pair file changed while stripping")
        for name in OUTPUT_NAMES:
            (staging / name).replace(output_dir / name)
    finally:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strip the dropout silence span into a new pair directory.")
    parser.add_argument("--train-pairs", type=Path, required=True)
    parser.add_argument("--val-pairs", type=Path, required=True)
    parser.add_argument("--expect-train-sha256", required=True)
    parser.add_argument("--expect-val-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-transcript-tokens", type=int, default=512)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument("--contract-test-char-tokens", action="store_true")
    parser.add_argument("--min-val-pairs", type=int, default=100)
    parser.add_argument("--min-pairs-per-scenario", type=int, default=10)
    parser.add_argument("--audit-per-scenario", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    char_mode = bool(args.contract_test_char_tokens)
    if char_mode == (args.tokenizer_dir is not None):
        print(
            "GAP_STRIP_FAIL pass exactly one of --tokenizer-dir or --contract-test-char-tokens",
            file=sys.stderr,
        )
        return 2
    if len(args.expect_train_sha256) != 64 or len(args.expect_val_sha256) != 64:
        print("GAP_STRIP_FAIL expected sha256 must be 64 hex characters", file=sys.stderr)
        return 2
    try:
        if char_mode:
            counter: Callable[[str], int] = char_token_count
            tokenizer_record = {"mode": "char-plus-eos", "eos": "synthetic", "note": "contract test only"}
        else:
            counter, tokenizer_record = qwen_token_count(args.tokenizer_dir)
        report = strip_pair_files(
            train_pairs=args.train_pairs,
            val_pairs=args.val_pairs,
            expect_train_sha256=args.expect_train_sha256,
            expect_val_sha256=args.expect_val_sha256,
            output_dir=args.output_dir,
            seed=args.seed,
            max_transcript_tokens=args.max_transcript_tokens,
            count_tokens=counter,
            tokenizer_record=tokenizer_record,
            min_val_pairs=args.min_val_pairs,
            min_pairs_per_scenario=args.min_pairs_per_scenario,
            audit_per_scenario=args.audit_per_scenario,
        )
    except GapStripError as exc:
        print(f"GAP_STRIP_FAIL {exc}", file=sys.stderr)
        return 1
    print(
        "GAP_STRIP_STATUS "
        f"{report['status']} train_pairs={report['train_pairs']} "
        f"val_pairs={report['val_pairs']} gap_ties={report['gap_ties']}",
        flush=True,
    )
    if report["status"] == "insufficient_data":
        return 3
    return 0


if __name__ == "__main__":
    main()
