#!/usr/bin/env python3
"""Build DPO pairs from a published SFT release's greedy hypotheses.

Chosen text is the manifest gold. Rejected text is ``prediction_raw``.
Ties, empty hypotheses, and inference errors go to ``pair_rejected.jsonl``.
Token counts over the SFT transcript cap go to ``overlength_pairs.jsonl``.
Reference logprobs stay null. This script does not train.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.eval_cer import join_rows  # noqa: E402
from evaluation.jsonl_io import read_jsonl, write_jsonl  # noqa: E402
from evaluation.zh_cer import NORMALIZATION_VERSION, normalize_zh_cer_v1, score_pair  # noqa: E402

PAIR_SCHEMA = "qwen3-asr-dpo-pair-v1"
METADATA_SCHEMA = "qwen3-asr-dpo-pair-metadata-v1"
AUDIT_SCHEMA = "qwen3-asr-dpo-pair-audit-v1"
REPORT_SCHEMA = "qwen3-asr-dpo-pair-report-v1"
RELEASE_SCHEMA = "qwen3-asr-sft-release-v1"
EOS_TOKEN = "<|im_end|>"
TOKENIZER_HASH_FILES = (
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "chat_template.json",
)
SCENARIOS = ("clean", "noise", "reverb", "far_field", "dropout")
OUTPUT_NAMES = (
    "train_pairs.jsonl",
    "val_pairs.jsonl",
    "pair_rejected.jsonl",
    "overlength_pairs.jsonl",
    "pair_metadata.json",
    "pair_audit.json",
    "pair_report.json",
)


class PairBuildError(ValueError):
    """The pair files must not be published when this is raised."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def char_token_count(text: str) -> int:
    """Contract-test counter: one token per character, plus a synthetic EOS."""
    return len(text) + 1


def qwen_token_count(tokenizer_dir: Path) -> tuple[Callable[[str], int], dict[str, Any]]:
    """Count transcript tokens plus one EOS. The tokenizer directory must be local."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    eos_id = tokenizer.convert_tokens_to_ids(EOS_TOKEN)
    if not isinstance(eos_id, int) or eos_id < 0:
        raise PairBuildError(f"tokenizer is missing {EOS_TOKEN}")

    def count(text: str) -> int:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if any(not isinstance(token, int) for token in ids):
            raise PairBuildError("tokenizer returned a non-integer token")
        return len(ids) + 1

    missing = [name for name in TOKENIZER_HASH_FILES if not (tokenizer_dir / name).is_file()]
    if missing:
        raise PairBuildError(f"tokenizer dir is missing {', '.join(missing)}")
    files = {name: sha256_file(tokenizer_dir / name) for name in TOKENIZER_HASH_FILES}
    record = {
        "mode": "encode-plus-eos",
        "eos": EOS_TOKEN,
        "eos_id": eos_id,
        "tokenizer_dir": str(tokenizer_dir),
        "files_sha256": files,
    }
    return count, record


def load_published_release(release_dir: Path) -> dict[str, Any]:
    path = release_dir / "release.json"
    if not path.is_file():
        raise PairBuildError(f"missing release.json in {release_dir}")
    release = json.loads(path.read_text(encoding="utf-8"))
    if release.get("schema_version") != RELEASE_SCHEMA:
        raise PairBuildError("release schema mismatch")
    if release.get("status") != "published" or release.get("publishable") is not True:
        raise PairBuildError("DPO pairs require a published release")
    if not isinstance(release.get("release_id"), str) or not release["release_id"]:
        raise PairBuildError("published release is missing release_id")
    return release


def wav_problem(path: Path) -> str | None:
    if not path.is_file():
        return "missing"
    if path.stat().st_size < 44:
        return "short"
    with path.open("rb") as handle:
        header = handle.read(12)
    if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
        return "header"
    return None


def resolve_audio(manifest_path: Path, audio: str) -> Path:
    path = Path(audio)
    if path.is_absolute():
        return path
    return manifest_path.parent / path


def _require_str(row: dict[str, Any], key: str, sample_id: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise PairBuildError(f"{sample_id} is missing {key}")
    return value


def classify_row(
    manifest: dict[str, Any],
    prediction: dict[str, Any],
    *,
    split: str,
    count_tokens: Callable[[str], int],
    max_transcript_tokens: int,
) -> dict[str, Any]:
    """Return one pair record with ``bucket`` set to pair, rejected, or overlength."""
    sample_id = _require_str(manifest, "sample_id", "manifest row")
    if manifest.get("split") != split:
        raise PairBuildError(f"{sample_id} split is {manifest.get('split')}, expected {split}")
    scenario = _require_str(manifest, "scenario", sample_id)
    if scenario not in SCENARIOS:
        raise PairBuildError(f"{sample_id} has unknown scenario {scenario}")
    source_id = _require_str(manifest, "source_utterance_id", sample_id)
    audio = _require_str(manifest, "audio", sample_id)
    reference = manifest.get("text")
    if not isinstance(reference, str):
        raise PairBuildError(f"{sample_id} is missing text")
    saved_reference = prediction.get("reference_raw")
    if saved_reference is not None and saved_reference != reference:
        raise PairBuildError(f"reference_raw mismatch for {sample_id}")
    if not normalize_zh_cer_v1(reference):
        raise PairBuildError(f"{sample_id} reference is empty after normalization")
    rejected = prediction.get("prediction_raw")
    if not isinstance(rejected, str):
        raise PairBuildError(f"{sample_id} is missing prediction_raw")
    scored = score_pair(
        reference,
        rejected,
        inference_error=False,
        stop_reason=prediction.get("stop_reason") if isinstance(prediction.get("stop_reason"), str) else None,
    )
    chosen_tokens = count_tokens(reference)
    rejected_tokens = count_tokens(rejected)
    if chosen_tokens < 1 or rejected_tokens < 1:
        raise PairBuildError(f"{sample_id} token count must include EOS")
    record = {
        "schema_version": PAIR_SCHEMA,
        "pair_id": sample_id,
        "sample_id": sample_id,
        "source_utterance_id": source_id,
        "audio": audio,
        "scenario": scenario,
        "split": split,
        "reference_text": reference,
        "chosen_text": reference,
        "rejected_text": rejected,
        "chosen_cer": 0.0,
        "rejected_cer": scored["sample_cer"],
        "chosen_token_count": chosen_tokens,
        "rejected_token_count": rejected_tokens,
        "ref_chosen_logp": None,
        "ref_rejected_logp": None,
        "normalization_version": NORMALIZATION_VERSION,
        "edit_count": scored["edit_count"],
        "reference_char_count": scored["reference_char_count"],
        "rejected_char_count": len(scored["prediction_normalized"]),
        "char_length_delta": len(scored["prediction_normalized"]) - int(scored["reference_char_count"]),
        "stop_reason": prediction.get("stop_reason"),
    }
    error = prediction.get("error")
    if error not in (None, ""):
        record["bucket"] = "rejected"
        record["reason"] = "inference_error"
        return record
    if scored["prediction_normalized"] == "":
        record["bucket"] = "rejected"
        record["reason"] = "empty_rejected"
        return record
    if scored["prediction_normalized"] == scored["reference_normalized"] or scored["edit_count"] == 0:
        record["bucket"] = "rejected"
        record["reason"] = "tie"
        return record
    limit = max_transcript_tokens + 1
    if chosen_tokens > limit or rejected_tokens > limit:
        record["bucket"] = "overlength"
        record["reason"] = "overlength"
        return record
    record["bucket"] = "pair"
    return record


def _audit_key(seed: int, split: str, scenario: str, sample_id: str) -> str:
    payload = f"{seed}\n{split}\n{scenario}\n{sample_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def select_audit(pairs: list[dict[str, Any]], seed: int, limit: int) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[(row["split"], row["scenario"])].append(row)
    by_scenario: dict[str, Any] = {}
    for split in ("train", "val"):
        for scenario in SCENARIOS:
            rows = grouped.get((split, scenario), [])
            ranked = sorted(rows, key=lambda row: _audit_key(seed, split, scenario, row["sample_id"]))
            chosen = ranked[: min(limit, len(ranked))]
            by_scenario[f"{split}:{scenario}"] = {
                "available": len(rows),
                "sampled": len(chosen),
                "sample_ids": [row["sample_id"] for row in chosen],
                "errors": [],
                "disposition": "unlistened",
            }
    return {
        "schema_version": AUDIT_SCHEMA,
        "seed": seed,
        "sampler": "sha256-rank-v1",
        "status": "unlistened",
        "by_scenario": by_scenario,
    }


def _mean(total: int, count: int) -> float | None:
    if count == 0:
        return None
    return total / count


def _summary(manifest_rows: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, Any]:
    manifest_counts: Counter[str] = Counter(row["scenario"] for row in manifest_rows)
    pair_counts: Counter[str] = Counter()
    sources: dict[str, set[str]] = defaultdict(set)
    char_delta: Counter[str] = Counter()
    token_delta: Counter[str] = Counter()
    for row in pairs:
        scenario = row["scenario"]
        pair_counts[scenario] += 1
        sources[scenario].add(row["source_utterance_id"])
        char_delta[scenario] += int(row["char_length_delta"])
        token_delta[scenario] += int(row["rejected_token_count"]) - int(row["chosen_token_count"])
    scenarios = {}
    for scenario in SCENARIOS:
        source_rows = manifest_counts[scenario]
        kept = pair_counts[scenario]
        scenarios[scenario] = {
            "manifest_rows": source_rows,
            "pairs": kept,
            "sources": len(sources[scenario]),
            "retention": (kept / source_rows) if source_rows else None,
            "mean_char_length_delta": _mean(char_delta[scenario], kept),
            "mean_token_length_delta": _mean(token_delta[scenario], kept),
        }
    return {"rows": len(manifest_rows), "pairs": len(pairs), "by_scenario": scenarios}


def _check_audio_and_leakage(
    splits: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]],
    manifests: dict[str, Path],
) -> None:
    seen_source: dict[str, str] = {}
    seen_audio: dict[str, str] = {}
    for split, joined in splits.items():
        for manifest, _prediction in joined:
            sample_id = manifest["sample_id"]
            source_id = _require_str(manifest, "source_utterance_id", sample_id)
            previous = seen_source.get(source_id)
            if previous is not None and previous != split:
                raise PairBuildError(f"source {source_id} leaks across {previous} and {split}")
            seen_source[source_id] = split
            audio_key = str(resolve_audio(manifests[split], manifest["audio"]).resolve())
            previous_audio = seen_audio.get(audio_key)
            if previous_audio is not None and previous_audio != split:
                raise PairBuildError(f"audio {audio_key} leaks across {previous_audio} and {split}")
            if previous_audio == split:
                raise PairBuildError(f"duplicate audio {audio_key} in {split}")
            seen_audio[audio_key] = split
            problem = wav_problem(Path(audio_key))
            if problem is not None:
                raise PairBuildError(f"audio {problem} for {sample_id}: {audio_key}")


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    published = dict(row)
    published.pop("bucket", None)
    return published


def build_dpo_pairs(
    *,
    train_manifest: Path,
    train_predictions: Path,
    val_manifest: Path,
    val_predictions: Path,
    release_dir: Path,
    output_dir: Path,
    seed: int,
    max_transcript_tokens: int,
    count_tokens: Callable[[str], int],
    tokenizer_record: dict[str, Any],
    min_val_pairs: int = 100,
    min_pairs_per_scenario: int = 10,
    audit_per_scenario: int = 20,
) -> dict[str, Any]:
    """Write pair files. Raises PairBuildError before any pair file is published."""
    if max_transcript_tokens < 1:
        raise PairBuildError("max_transcript_tokens must be positive")
    existing = [name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if existing:
        raise PairBuildError(f"output already exists: {', '.join(existing)}")
    release = load_published_release(release_dir)
    manifest_paths = {"train": train_manifest, "val": val_manifest}
    prediction_paths = {"train": train_predictions, "val": val_predictions}
    manifests = {split: read_jsonl(path) for split, path in manifest_paths.items()}
    predictions = {split: read_jsonl(path) for split, path in prediction_paths.items()}
    try:
        joined = {
            split: join_rows(manifests[split], predictions[split])
            for split in ("train", "val")
        }
    except ValueError as exc:
        raise PairBuildError(str(exc)) from exc
    _check_audio_and_leakage(joined, manifest_paths)

    buckets: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val": [],
        "rejected": [],
        "overlength": [],
    }
    reasons: Counter[str] = Counter()
    for split, rows in joined.items():
        for manifest, prediction in rows:
            record = classify_row(
                manifest,
                prediction,
                split=split,
                count_tokens=count_tokens,
                max_transcript_tokens=max_transcript_tokens,
            )
            bucket = record["bucket"]
            if bucket == "pair":
                buckets[split].append(_public_row(record))
            elif bucket == "rejected":
                reasons[record["reason"]] += 1
                buckets["rejected"].append(_public_row(record))
            else:
                reasons["overlength"] += 1
                buckets["overlength"].append(_public_row(record))

    val_summary = _summary(manifests["val"], buckets["val"])
    short_scenarios = [
        scenario
        for scenario, item in val_summary["by_scenario"].items()
        if item["pairs"] < min_pairs_per_scenario
    ]
    status = "pairs_built"
    if len(buckets["val"]) < min_val_pairs or short_scenarios:
        status = "insufficient_data"
    audit = select_audit(buckets["train"] + buckets["val"], seed, audit_per_scenario)
    report = {
        "schema_version": REPORT_SCHEMA,
        "status": status,
        "trainable": False,
        "blocked_by": ["reference_logprob_pending", "pair_audit_unlistened"],
        "release_id": release["release_id"],
        "seed": seed,
        "max_transcript_tokens": max_transcript_tokens,
        "token_limit": max_transcript_tokens + 1,
        "min_val_pairs": min_val_pairs,
        "min_pairs_per_scenario": min_pairs_per_scenario,
        "short_val_scenarios": short_scenarios,
        "rejection_reasons": dict(reasons),
        "train": _summary(manifests["train"], buckets["train"]),
        "val": val_summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / ".pair-staging"
    if staging.exists():
        raise PairBuildError(f"staging already exists: {staging}")
    staging.mkdir()
    try:
        write_jsonl(staging / "train_pairs.jsonl", buckets["train"])
        write_jsonl(staging / "val_pairs.jsonl", buckets["val"])
        write_jsonl(staging / "pair_rejected.jsonl", buckets["rejected"])
        write_jsonl(staging / "overlength_pairs.jsonl", buckets["overlength"])
        metadata = {
            "schema_version": METADATA_SCHEMA,
            "status": status,
            "trainable": False,
            "release_id": release["release_id"],
            "release_schema": release["schema_version"],
            "source_step": release.get("source_step"),
            "base_revision": release.get("base_revision"),
            "processor_sha256": release.get("processor_sha256"),
            "adapter_sha256": release.get("adapter_sha256"),
            "train_manifest_sha256": sha256_file(train_manifest),
            "val_manifest_sha256": sha256_file(val_manifest),
            "train_predictions_sha256": sha256_file(train_predictions),
            "val_predictions_sha256": sha256_file(val_predictions),
            "train_pairs_sha256": sha256_file(staging / "train_pairs.jsonl"),
            "val_pairs_sha256": sha256_file(staging / "val_pairs.jsonl"),
            "score_contract": NORMALIZATION_VERSION,
            "language_prompt": "not applied; token counts are transcript text plus one EOS",
            "mask_rule": "transcript tokens plus one EOS; prefix and padding are excluded",
            "tokenizer": tokenizer_record,
            "reference_logprob": "pending",
            "seed": seed,
            "resolved": {
                "train_manifest": str(train_manifest),
                "train_predictions": str(train_predictions),
                "val_manifest": str(val_manifest),
                "val_predictions": str(val_predictions),
                "release_dir": str(release_dir),
                "max_transcript_tokens": max_transcript_tokens,
                "min_val_pairs": min_val_pairs,
                "min_pairs_per_scenario": min_pairs_per_scenario,
                "audit_per_scenario": audit_per_scenario,
                "seed": seed,
            },
        }
        (staging / "pair_metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (staging / "pair_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report["train_pairs_sha256"] = metadata["train_pairs_sha256"]
        report["val_pairs_sha256"] = metadata["val_pairs_sha256"]
        (staging / "pair_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for name in OUTPUT_NAMES:
            (staging / name).replace(output_dir / name)
    finally:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
    return report


def _write_failure(output_dir: Path, message: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "pair_report.json"
    if path.exists():
        return
    payload = {
        "schema_version": REPORT_SCHEMA,
        "status": "failed",
        "trainable": False,
        "error": message,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DPO pairs from SFT greedy hypotheses.")
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--train-predictions", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
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
    if args.seed < 0:
        print("DPO_PAIR_FAIL seed must be non-negative", file=sys.stderr)
        return 2
    char_mode = bool(args.contract_test_char_tokens)
    if char_mode == (args.tokenizer_dir is not None):
        print(
            "DPO_PAIR_FAIL pass exactly one of --tokenizer-dir or --contract-test-char-tokens",
            file=sys.stderr,
        )
        return 2
    try:
        if char_mode:
            counter: Callable[[str], int] = char_token_count
            tokenizer_record = {
                "mode": "char-plus-eos",
                "eos": "synthetic",
                "note": "contract test only",
            }
        else:
            counter, tokenizer_record = qwen_token_count(args.tokenizer_dir)
        report = build_dpo_pairs(
            train_manifest=args.train_manifest,
            train_predictions=args.train_predictions,
            val_manifest=args.val_manifest,
            val_predictions=args.val_predictions,
            release_dir=args.release_dir,
            output_dir=args.output_dir,
            seed=args.seed,
            max_transcript_tokens=args.max_transcript_tokens,
            count_tokens=counter,
            tokenizer_record=tokenizer_record,
            min_val_pairs=args.min_val_pairs,
            min_pairs_per_scenario=args.min_pairs_per_scenario,
            audit_per_scenario=args.audit_per_scenario,
        )
    except PairBuildError as exc:
        _write_failure(args.output_dir, str(exc))
        print(f"DPO_PAIR_FAIL {exc}", file=sys.stderr)
        return 1
    print(
        "DPO_PAIR_STATUS "
        f"{report['status']} train_pairs={report['train']['pairs']} "
        f"val_pairs={report['val']['pairs']}",
        flush=True,
    )
    if report["status"] == "insufficient_data":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
