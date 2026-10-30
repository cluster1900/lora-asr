#!/usr/bin/env python3
"""Score greedy ASR predictions with zh-cer-v1.

Gold text comes from the manifest. A missing, duplicate, or extra sample fails
the process. Per-row inference errors stay in the score file and are counted
as empty hypotheses so a failure cannot look like a perfect transcript.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.jsonl_io import read_jsonl  # noqa: E402
from evaluation.zh_cer import load_jieba_tokenizer, score_pair, score_wer_pair  # noqa: E402
from inference.config import load_infer_config  # noqa: E402

METRICS_SCHEMA = "qwen3-asr-metrics-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sample_id(row: dict[str, Any], source: str, index: int) -> str:
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ValueError(f"{source} row {index} is missing sample_id")
    return sample_id


def join_rows(
    manifest_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Pair manifest gold with predictions. Any mismatch raises."""
    predictions: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(prediction_rows, start=1):
        sample_id = _require_sample_id(row, "predictions", index)
        if sample_id in predictions:
            raise ValueError(f"duplicate prediction sample_id {sample_id}")
        predictions[sample_id] = row

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: list[str] = []
    for index, row in enumerate(manifest_rows, start=1):
        sample_id = _require_sample_id(row, "manifest", index)
        if sample_id in seen:
            raise ValueError(f"duplicate manifest sample_id {sample_id}")
        seen.append(sample_id)
        if sample_id not in predictions:
            raise ValueError(f"missing prediction for sample_id {sample_id}")
        prediction = predictions[sample_id]
        manifest_text = row.get("text")
        if not isinstance(manifest_text, str):
            raise ValueError(f"manifest sample_id {sample_id} is missing text")
        predicted_text = prediction.get("text")
        if predicted_text is not None and predicted_text != manifest_text:
            raise ValueError(f"gold text mismatch for sample_id {sample_id}")
        pairs.append((row, prediction))

    extra = sorted(set(predictions) - set(seen))
    if extra:
        raise ValueError(f"predictions contain samples absent from the manifest: {extra}")
    return pairs


def _rate(flag_sum: int, count: int) -> float:
    return flag_sum / count if count else 0.0


def _scenario_summary(rows: list[dict[str, Any]], *, include_wer: bool = False) -> dict[str, Any]:
    edits = sum(int(row["edit_count"]) for row in rows)
    refs = sum(int(row["reference_char_count"]) for row in rows)
    count = len(rows)
    summary: dict[str, Any] = {
        "samples": count,
        "edit_count": edits,
        "reference_char_count": refs,
        "micro_cer": (edits / refs) if refs else None,
        "macro_cer": (sum(float(row["sample_cer"]) for row in rows) / count) if count else None,
        "empty_rate": _rate(sum(int(row["empty_output"]) for row in rows), count),
        "repeated_rate": _rate(sum(int(row["repeated_output"]) for row in rows), count),
        "long_rate": _rate(sum(int(row["long_output"]) for row in rows), count),
        "truncation_rate": _rate(sum(int(row["truncated"]) for row in rows), count),
        "inference_error_rate": _rate(sum(int(row["inference_error"]) for row in rows), count),
        "valid_output_rate": _rate(sum(int(row["valid_output"]) for row in rows), count),
    }
    if include_wer:
        word_edits = sum(int(row["word_edit_count"]) for row in rows)
        word_refs = sum(int(row["reference_word_count"]) for row in rows)
        summary.update(
            {
                "word_edit_count": word_edits,
                "reference_word_count": word_refs,
                "micro_wer": (word_edits / word_refs) if word_refs else None,
                "macro_wer": (sum(float(row["sample_wer"]) for row in rows) / count) if count else None,
            }
        )
    return summary


def evaluate(
    manifest_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    *,
    manifest_sha256: str,
    predictions_sha256: str,
    decode_eval_sha256: str,
    wer_tokenizer: str | None,
    wer_tokenizer_version: str | None = None,
    wer_dictionary_sha256: str | None = None,
    wer_hmm: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return scored rows and the metrics object."""
    if not isinstance(wer_hmm, bool):
        raise ValueError("wer_hmm must be a boolean")
    if wer_tokenizer is None:
        if wer_tokenizer_version is not None or wer_dictionary_sha256 is not None:
            raise ValueError("null wer_tokenizer requires null version and dictionary hash")
        if wer_hmm:
            raise ValueError("null wer_tokenizer requires wer_hmm to be false")
        tokenizer = None
    else:
        if wer_tokenizer != "jieba":
            raise ValueError(f"unsupported wer_tokenizer: {wer_tokenizer}")
        if not isinstance(wer_tokenizer_version, str) or not wer_tokenizer_version.strip():
            raise ValueError("wer_tokenizer_version is required when wer_tokenizer is set")
        if not isinstance(wer_dictionary_sha256, str) or len(wer_dictionary_sha256) != 64:
            raise ValueError("wer_dictionary_sha256 must be a 64-char hex string when wer_tokenizer is set")
        tokenizer = load_jieba_tokenizer(
            expected_version=wer_tokenizer_version,
            expected_dictionary_sha256=wer_dictionary_sha256,
        )

    scored: list[dict[str, Any]] = []
    for manifest, prediction in join_rows(manifest_rows, prediction_rows):
        error = prediction.get("error")
        inference_error = bool(str(error or "").strip())
        scored_fields = score_pair(
            str(manifest["text"]),
            str(prediction.get("prediction_raw") or ""),
            inference_error=inference_error,
            stop_reason=prediction.get("stop_reason") if isinstance(prediction.get("stop_reason"), str) else None,
        )
        row = {
            "sample_id": manifest["sample_id"],
            "scenario": manifest.get("scenario"),
            "language": manifest.get("language"),
            "source_utterance_id": manifest.get("source_utterance_id"),
            "reference_raw": manifest["text"],
            "prediction_raw": prediction.get("prediction_raw"),
            "error": error if inference_error else None,
            "stop_reason": prediction.get("stop_reason"),
        }
        row.update(scored_fields)
        if tokenizer is not None:
            row.update(
                score_wer_pair(
                    str(scored_fields["reference_normalized"]),
                    str(scored_fields["scored_prediction_normalized"]),
                    tokenizer,
                    hmm=wer_hmm,
                )
            )
        scored.append(row)

    include_wer = tokenizer is not None
    by_scenario_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_scenario_rows[str(row.get("scenario") or "unknown")].append(row)
    by_scenario = {
        name: _scenario_summary(group, include_wer=include_wer)
        for name, group in sorted(by_scenario_rows.items())
    }
    overall = _scenario_summary(scored, include_wer=include_wer)
    if include_wer:
        wer: float | None = overall["micro_wer"]
        wer_reason: str | None = None
    else:
        wer = None
        wer_reason = "wer_tokenizer_not_configured"
    metrics: dict[str, Any] = {
        "schema_version": METRICS_SCHEMA,
        "normalization_version": "zh-cer-v1",
        "manifest_sha256": manifest_sha256,
        "predictions_sha256": predictions_sha256,
        "decode_eval_sha256": decode_eval_sha256,
        "micro_cer": overall["micro_cer"],
        "macro_cer": overall["macro_cer"],
        "edit_count": overall["edit_count"],
        "reference_char_count": overall["reference_char_count"],
        "samples": overall["samples"],
        "empty_rate": overall["empty_rate"],
        "repeated_rate": overall["repeated_rate"],
        "long_rate": overall["long_rate"],
        "truncation_rate": overall["truncation_rate"],
        "inference_error_rate": overall["inference_error_rate"],
        "valid_output_rate": overall["valid_output_rate"],
        "wer": wer,
        "wer_reason": wer_reason,
    }
    if include_wer:
        assert wer_tokenizer_version is not None and wer_dictionary_sha256 is not None
        metrics.update(
            {
                "micro_wer": overall["micro_wer"],
                "macro_wer": overall["macro_wer"],
                "word_edit_count": overall["word_edit_count"],
                "reference_word_count": overall["reference_word_count"],
                "wer_tokenizer_info": {
                    "name": "jieba",
                    "version": wer_tokenizer_version,
                    "dictionary_sha256": wer_dictionary_sha256.lower(),
                    "hmm": wer_hmm,
                },
            }
        )
    metrics["by_scenario"] = by_scenario
    return scored, metrics


def write_scenario_csv(path: Path, by_scenario: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scenario",
        "samples",
        "edit_count",
        "reference_char_count",
        "micro_cer",
        "macro_cer",
        "empty_rate",
        "repeated_rate",
        "long_rate",
        "truncation_rate",
        "inference_error_rate",
        "valid_output_rate",
    ]
    if any("micro_wer" in summary for summary in by_scenario.values()):
        fields.extend(["word_edit_count", "reference_word_count", "micro_wer", "macro_wer"])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for scenario, summary in by_scenario.items():
            writer.writerow({"scenario": scenario, **summary})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用 zh-cer-v1 给 ASR 预测计算 CER")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_infer_config(args.config)
        manifest_rows = read_jsonl(args.manifest)
        prediction_rows = read_jsonl(args.predictions)
        scored, metrics = evaluate(
            manifest_rows,
            prediction_rows,
            manifest_sha256=sha256_file(args.manifest),
            predictions_sha256=sha256_file(args.predictions),
            decode_eval_sha256=config["decode_eval_sha256"],
            wer_tokenizer=config["scoring"]["wer_tokenizer"],
            wer_tokenizer_version=config["scoring"]["wer_tokenizer_version"],
            wer_dictionary_sha256=config["scoring"]["wer_dictionary_sha256"],
            wer_hmm=config["scoring"]["wer_hmm"],
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"eval_cer failed: {exc}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scored_path = args.output_dir / "scored.jsonl"
    with scored_path.open("w", encoding="utf-8") as handle:
        for row in scored:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_scenario_csv(args.output_dir / "by_scenario.csv", metrics["by_scenario"])
    summary_out: dict[str, Any] = {
        "micro_cer": metrics["micro_cer"],
        "macro_cer": metrics["macro_cer"],
        "samples": metrics["samples"],
        "metrics": str(metrics_path),
    }
    if metrics["wer"] is not None:
        summary_out["wer"] = metrics["wer"]
        summary_out["macro_wer"] = metrics["macro_wer"]
    print(json.dumps(summary_out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
