#!/usr/bin/env python3
"""Score ASR JSONL with English WER, Chinese CER, and robust-ASR cells.

The evaluator uses only Python's standard library. English word edits and
Chinese character edits are never presented as one combined WER/CER value.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable, Sequence


BENCH_SCENARIOS = (
    "distortion",
    "dropout",
    "echo",
    "far_field",
    "mixed",
    "noise",
    "obstructed",
    "recording",
)
FAILURE_FIELDS = (
    "inference_errors",
    "empty_outputs",
    "repeat_like_outputs",
    "too_long_outputs",
    "hallucination_like_outputs",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    value = str(text or "").strip().lower()
    value = "".join(
        " " if unicodedata.category(char).startswith("P") else char
        for char in value
    )
    return re.sub(r"\s+", " ", value).strip()


def canonical_language(item: dict[str, Any]) -> str:
    declared = str(item.get("language") or "").strip().lower()
    if declared not in {"en", "zh"}:
        raise ValueError(f"Invalid language for {row_identity(item)}: {declared!r}")
    return declared


def tokenize(text: str, metric: str) -> list[str]:
    if metric == "cer":
        return list(text.replace(" ", ""))
    return text.split()


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row_index, ref_token in enumerate(reference, start=1):
        current = [row_index]
        for column_index, hyp_token in enumerate(hypothesis, start=1):
            substitution = previous[column_index - 1] + int(ref_token != hyp_token)
            current.append(min(
                previous[column_index] + 1,
                current[column_index - 1] + 1,
                substitution,
            ))
        previous = current
    return previous[-1]


def has_repetition(tokens: Sequence[str]) -> bool:
    if not tokens:
        return False
    run = 1
    for index in range(1, len(tokens)):
        run = run + 1 if tokens[index] == tokens[index - 1] else 1
        if run >= 3:
            return True
    if len(tokens) < 4:
        return False
    return any(count >= 2 for count in Counter(zip(tokens, tokens[1:])).values())


def source_origin(item: dict[str, Any]) -> str:
    return str(item.get("audio_origin") or "unknown").strip().lower()


def scenario_name(item: dict[str, Any]) -> str:
    return str(item.get("scenario") or "unknown").strip() or "unknown"


def row_identity(item: dict[str, Any], fallback: int | None = None) -> str:
    for field in ("sample_id", "inference_key"):
        if item.get(field) is not None and str(item[field]).strip():
            return f"{field}:{item[field]}"
    return f"index:{fallback}" if fallback is not None else "unknown"


def score_item(
    item: dict[str, Any],
    item_index: int | None = None,
) -> dict[str, Any]:
    reference_raw = str(item.get("answer") or "")
    prediction_raw = str(item.get("prediction") or "")
    reference_normalized = normalize_text(reference_raw)
    prediction_normalized = normalize_text(prediction_raw)
    if not reference_normalized:
        raise ValueError(f"Empty reference for {row_identity(item, item_index)}")

    language = canonical_language(item)
    metric = "cer" if language == "zh" else "wer"
    reference_tokens = tokenize(reference_normalized, metric)
    inference_error = bool(str(item.get("error") or "").strip())
    # A row marked failed is scored as an empty hypothesis even if a partial
    # string was emitted, so failures can never look like zero-error samples.
    scored_prediction = "" if inference_error else prediction_normalized
    prediction_tokens = tokenize(scored_prediction, metric)
    edits = edit_distance(reference_tokens, prediction_tokens)
    reference_length = len(reference_tokens)
    error_rate = edits / reference_length
    empty_output = not prediction_normalized
    repeated = has_repetition(tokenize(prediction_normalized, metric))
    length_ratio = len(prediction_tokens) / reference_length
    too_long = length_ratio > 1.5
    hallucination_like = (
        not inference_error
        and not empty_output
        and error_rate >= 0.8
        and (too_long or repeated or edits >= max(3, reference_length // 2))
    )
    origin = source_origin(item)

    tags: list[str] = []
    if inference_error:
        tags.append("inference_error")
    if empty_output:
        tags.append("empty_output")
    if repeated:
        tags.append("repeat_like")
    if too_long:
        tags.append("too_long")
    if hallucination_like:
        tags.append("hallucination_like")

    out = dict(item)
    out.update({
        "reference_raw": reference_raw,
        "prediction_raw": prediction_raw,
        "reference_normalized": reference_normalized,
        "prediction_normalized": prediction_normalized,
        "scored_prediction_normalized": scored_prediction,
        "language": language,
        "metric": metric,
        "error_rate": round(error_rate, 6),
        # Legacy consumers read `wer`; use `metric` to distinguish Chinese CER.
        "wer": round(error_rate, 6),
        "num_edits": edits,
        "ref_len": reference_length,
        "length_ratio": round(length_ratio, 6),
        "inference_error": inference_error,
        "empty_output": empty_output,
        "repeat_like": repeated,
        "too_long": too_long,
        "hallucination_like": hallucination_like,
        "failure_tags": tags,
        "audio_origin": origin,
        "source_type": origin,
        "scenario": scenario_name(item),
    })
    return out


def _new_bucket() -> dict[str, Any]:
    return {
        "samples": 0,
        "num_edits": 0,
        "ref_len": 0,
        "metrics": set(),
        "inference_errors": 0,
        "empty_outputs": 0,
        "repeat_like_outputs": 0,
        "too_long_outputs": 0,
        "hallucination_like_outputs": 0,
    }


def _add_to_bucket(bucket: dict[str, Any], row: dict[str, Any]) -> None:
    bucket["samples"] += 1
    bucket["num_edits"] += int(row["num_edits"])
    bucket["ref_len"] += int(row["ref_len"])
    bucket["metrics"].add(str(row["metric"]))
    bucket["inference_errors"] += int(bool(row["inference_error"]))
    bucket["empty_outputs"] += int(bool(row["empty_output"]))
    bucket["repeat_like_outputs"] += int(bool(row["repeat_like"]))
    bucket["too_long_outputs"] += int(bool(row["too_long"]))
    bucket["hallucination_like_outputs"] += int(bool(row["hallucination_like"]))


def _finish_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    samples = int(bucket["samples"])
    metrics = sorted(bucket["metrics"])
    mixed = len(metrics) != 1
    result: dict[str, Any] = {
        "samples": samples,
        "metric": metrics[0] if len(metrics) == 1 else "mixed",
        "num_edits": None if mixed else int(bucket["num_edits"]),
        "ref_len": None if mixed else int(bucket["ref_len"]),
        "error_rate": None,
    }
    if not mixed and bucket["ref_len"]:
        result["error_rate"] = round(bucket["num_edits"] / bucket["ref_len"], 6)
    for field in FAILURE_FIELDS:
        result[field] = int(bucket[field])
        result[field.replace("outputs", "output_rate").replace("errors", "error_rate")] = (
            round(bucket[field] / samples, 6) if samples else 0.0
        )
    return result


def aggregate(
    rows: Sequence[dict[str, Any]],
    group_keys: str | Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    if isinstance(group_keys, str):
        keys = [group_keys]
    else:
        keys = list(group_keys or [])
    buckets: dict[tuple[str, ...], dict[str, Any]] = defaultdict(_new_bucket)
    for row in rows:
        group = tuple(str(row.get(key, "unknown")) for key in keys) if keys else ("ALL",)
        _add_to_bucket(buckets[group], row)

    result: list[dict[str, Any]] = []
    for group, bucket in sorted(buckets.items()):
        item = _finish_bucket(bucket)
        for index, key in enumerate(keys):
            item[key] = group[index]
        item["group"] = "|".join(group)
        result.append(item)
    return result


def overall_summary(rows: Sequence[dict[str, Any]], by_language: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        empty = _finish_bucket(_new_bucket())
        empty["group"] = "ALL"
        empty["language_macro_error_rate"] = None
        return empty
    bucket = _new_bucket()
    for row in rows:
        _add_to_bucket(bucket, row)
    result = _finish_bucket(bucket)
    result["group"] = "ALL"
    rates = [float(row["error_rate"]) for row in by_language if row["error_rate"] is not None]
    result["language_macro_error_rate"] = round(fmean(rates), 6) if rates else None
    return result


def subset_language_macro(
    rows: Sequence[dict[str, Any]], condition_groups: set[str]
) -> float | None:
    selected = [row for row in rows if str(row.get("condition_group")) in condition_groups]
    rates = [
        float(item["error_rate"])
        for item in aggregate(selected, ["language"])
        if item["error_rate"] is not None
    ]
    return round(fmean(rates), 6) if rates else None


def bench_cells(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = [
        row for row in rows
        if row.get("language") in {"en", "zh"}
        and row.get("source_type") in {"real", "synthetic"}
    ]
    cells = aggregate(eligible, ["language", "source_type", "scenario"])
    for cell in cells:
        cell["audio_origin"] = cell["source_type"]
    expected = {
        (language, origin, scenario)
        for language in ("en", "zh")
        for origin in ("real", "synthetic")
        for scenario in BENCH_SCENARIOS
    }
    observed = {
        (str(row["language"]), str(row["source_type"]), str(row["scenario"]))
        for row in cells
    }
    missing = sorted("|".join(cell) for cell in expected - observed)
    unexpected = sorted("|".join(cell) for cell in observed - expected)
    rates = [float(row["error_rate"]) for row in cells if row["error_rate"] is not None]

    def macro_breakdown(field: str, values: Sequence[str], expected_count: int) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for value in values:
            selected = [row for row in cells if row[field] == value]
            selected_rates = [
                float(row["error_rate"])
                for row in selected
                if row["error_rate"] is not None
            ]
            result.append({
                field: value,
                "observed_cells": len(selected),
                "expected_cells": expected_count,
                "complete": len(selected) == expected_count,
                "macro_error_rate": (
                    round(fmean(selected_rates), 6) if selected_rates else None
                ),
            })
        return result

    summary = {
        "expected_cells": 32,
        "observed_expected_cells": len(observed & expected),
        "observed_cells": len(observed),
        "complete": observed == expected,
        "macro_error_rate": round(fmean(rates), 6) if rates else None,
        "missing_cells": missing,
        "unexpected_cells": unexpected,
        "by_source_type": macro_breakdown("source_type", ("real", "synthetic"), 16),
        "by_language": macro_breakdown("language", ("en", "zh"), 16),
    }
    return cells, summary


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "group", "language", "source_type", "audio_origin", "scenario", "condition_group",
        "metric", "samples", "num_edits", "ref_len", "error_rate",
        "inference_error_rate", "empty_output_rate", "repeat_like_output_rate",
        "too_long_output_rate", "hallucination_like_output_rate",
    ]
    fields = {field for row in rows for field in row}
    fieldnames = [field for field in preferred if field in fields]
    fieldnames.extend(sorted(fields - set(fieldnames)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def evaluate(rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scored = [
        score_item(row, item_index=index)
        for index, row in enumerate(rows)
    ]
    by_language = aggregate(scored, ["language"])
    languages = {str(row["language"]) for row in scored}
    # Mixed-language groups include language so word and character edits stay separate.
    scenario_keys = ["scenario"] if len(languages) <= 1 else ["language", "scenario"]
    by_scenario = aggregate(scored, scenario_keys)
    cells, cell_macro = bench_cells(scored)
    overall = overall_summary(scored, by_language)
    overall["robust_language_macro_error_rate"] = subset_language_macro(
        scored, {"atomic", "compound"}
    )
    overall["clean_language_macro_error_rate"] = subset_language_macro(scored, {"clean"})
    metrics = {
        "overall": overall,
        "by_language": by_language,
        "by_scenario": by_scenario,
        "by_audio_origin": aggregate(
            scored,
            ["audio_origin"] if len(languages) <= 1 else ["language", "audio_origin"],
        ),
        "by_condition_group": aggregate(
            scored,
            ["condition_group"] if len(languages) <= 1 else ["language", "condition_group"],
        ),
        "by_cell": cells,
        "cell_macro": cell_macro,
    }
    return scored, metrics


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rows = read_jsonl(Path(args.predictions_jsonl).expanduser())
    scored, metrics = evaluate(rows)

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "scored.jsonl", scored)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "by_scenario.csv", metrics["by_scenario"])
    write_csv(output_dir / "by_cell.csv", metrics["by_cell"])
    write_csv(output_dir / "by_language.csv", metrics["by_language"])
    print(json.dumps({"overall": metrics["overall"], "cell_macro": metrics["cell_macro"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
