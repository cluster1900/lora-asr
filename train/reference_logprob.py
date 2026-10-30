#!/usr/bin/env python3
"""Cache reference sequence logprobs for an existing DPO pair set.

The reference is the published SFT adapter on the official base. Logprob is
the sum of FP32 token log-probabilities over the transcript plus one EOS.
Prefix and padding are not included. This script does not update weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.jsonl_io import read_jsonl, write_jsonl  # noqa: E402
from train.sft_checkpoint import sha256_file  # noqa: E402

CACHE_SCHEMA = "qwen3-asr-dpo-ref-logprob-v1"
REPORT_SCHEMA = "qwen3-asr-dpo-ref-report-v1"
RELEASE_SCHEMA = "qwen3-asr-sft-release-v1"
ABS_TOL = 1e-4
REL_TOL = 1e-5
Score = Callable[[dict[str, Any], str], tuple[float, int]]


class RefLogprobError(ValueError):
    """The filled pair files must not be published when this is raised."""


def reference_sequence_logprob(token_nll: list[float]) -> float:
    """Negate the sum of supervised token NLL. Do not average the tokens."""
    if not token_nll:
        raise RefLogprobError("reference logprob has no supervised tokens")
    total = 0.0
    for value in token_nll:
        if not math.isfinite(value):
            raise RefLogprobError("non-finite token nll")
        total += float(value)
    result = -total
    if not math.isfinite(result):
        raise RefLogprobError("non-finite reference logprob")
    return result


def logprob_errors(cached: float, fresh: float) -> tuple[float, float]:
    """Return absolute and relative error. Relative uses the larger magnitude."""
    if not math.isfinite(cached) or not math.isfinite(fresh):
        raise RefLogprobError("logprob comparison received a non-finite value")
    absolute = abs(cached - fresh)
    scale = max(abs(cached), abs(fresh))
    relative = 0.0 if scale == 0.0 else absolute / scale
    return absolute, relative


def logprobs_match(cached: float, fresh: float, *, abs_tol: float = ABS_TOL, rel_tol: float = REL_TOL) -> bool:
    absolute, relative = logprob_errors(cached, fresh)
    return absolute <= abs_tol and relative <= rel_tol


def ordered_pairs(train_pairs: list[dict[str, Any]], val_pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(train_pairs) + list(val_pairs)


def _recheck_key(seed: int, row: dict[str, Any]) -> str:
    payload = f"{seed}\n{row['split']}\n{row['pair_id']}".encode()
    return hashlib.sha256(payload).hexdigest()


def select_recheck(rows: list[dict[str, Any]], seed: int, limit: int) -> list[dict[str, Any]]:
    if limit < 0:
        raise RefLogprobError("recheck limit must be non-negative")
    ranked = sorted(rows, key=lambda row: _recheck_key(seed, row))
    return ranked[: min(limit, len(ranked))]


def completed_logprob_rows(path: Path, pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return cached rows when they are a prefix of ``pairs``. An empty file is zero."""
    if not path.is_file() or path.stat().st_size == 0:
        return []
    cached = read_jsonl(path)
    expected = [row["pair_id"] for row in pairs]
    got = []
    for index, row in enumerate(cached, start=1):
        pair_id = row.get("pair_id")
        if not isinstance(pair_id, str) or not pair_id:
            raise RefLogprobError(f"cached row {index} is missing pair_id")
        if row.get("schema_version") != CACHE_SCHEMA:
            raise RefLogprobError(f"cached schema mismatch at {pair_id}")
        got.append(pair_id)
        for key in ("ref_chosen_logp", "ref_rejected_logp"):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RefLogprobError(f"cached {key} is not finite for {pair_id}")
    if got != expected[: len(got)]:
        raise RefLogprobError("existing ref_logprobs.jsonl is not a pair_id prefix")
    return cached


def _score_side(score: Score, row: dict[str, Any], text: str, expected: int, side: str) -> float:
    logprob, count = score(row, text)
    if count != expected:
        raise RefLogprobError(
            f"{row['pair_id']} {side} token count is {count}, pair file says {expected}"
        )
    if not math.isfinite(logprob):
        raise RefLogprobError(f"{row['pair_id']} {side} logprob is not finite")
    return float(logprob)


def _cache_record(row: dict[str, Any], chosen_logp: float, rejected_logp: float) -> dict[str, Any]:
    return {
        "schema_version": CACHE_SCHEMA,
        "pair_id": row["pair_id"],
        "split": row["split"],
        "ref_chosen_logp": chosen_logp,
        "ref_rejected_logp": rejected_logp,
        "chosen_token_count": row["chosen_token_count"],
        "rejected_token_count": row["rejected_token_count"],
    }


def cache_reference_logprobs(
    train_pairs: list[dict[str, Any]],
    val_pairs: list[dict[str, Any]],
    score: Score,
    cache_path: Path,
    *,
    seed: int,
    recheck_limit: int = 20,
    on_progress: Callable[[int, int], None] | None = None,
    sync_fn: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Append missing logprobs and recheck a sample with a second score call."""
    pairs = ordered_pairs(train_pairs, val_pairs)
    if not pairs:
        raise RefLogprobError("no pairs to score")
    cached = completed_logprob_rows(cache_path, pairs)
    total = len(pairs)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    handle = cache_path.open("a", encoding="utf-8")
    try:
        for index in range(len(cached), total):
            row = pairs[index]
            if row.get("ref_chosen_logp") is not None or row.get("ref_rejected_logp") is not None:
                raise RefLogprobError(f"{row['pair_id']} already has a reference logprob")
            chosen = _score_side(score, row, str(row["chosen_text"]), int(row["chosen_token_count"]), "chosen")
            rejected = _score_side(
                score, row, str(row["rejected_text"]), int(row["rejected_token_count"]), "rejected"
            )
            record = _cache_record(row, chosen, rejected)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            done = index + 1
            if on_progress is not None and (done == 1 or done % 250 == 0 or done == total):
                on_progress(done, total)
            if sync_fn is not None and (done % 2500 == 0 or done == total):
                sync_fn(done, total)
    finally:
        handle.close()
    cached = completed_logprob_rows(cache_path, pairs)
    if len(cached) != total:
        raise RefLogprobError(f"cache has {len(cached)} rows after scoring {total}")
    by_id = {row["pair_id"]: row for row in cached}
    checked = 0
    max_abs = 0.0
    max_rel = 0.0
    for split_rows in (train_pairs, val_pairs):
        for row in select_recheck(split_rows, seed, min(recheck_limit, len(split_rows))):
            fresh_chosen = _score_side(
                score, row, str(row["chosen_text"]), int(row["chosen_token_count"]), "chosen"
            )
            fresh_rejected = _score_side(
                score, row, str(row["rejected_text"]), int(row["rejected_token_count"]), "rejected"
            )
            saved = by_id[row["pair_id"]]
            for side, fresh, old in (
                ("chosen", fresh_chosen, float(saved["ref_chosen_logp"])),
                ("rejected", fresh_rejected, float(saved["ref_rejected_logp"])),
            ):
                absolute, relative = logprob_errors(old, fresh)
                max_abs = max(max_abs, absolute)
                max_rel = max(max_rel, relative)
                if not logprobs_match(old, fresh):
                    raise RefLogprobError(
                        f"{row['pair_id']} {side} recheck abs={absolute} rel={relative}"
                    )
            checked += 1
    return {
        "schema_version": REPORT_SCHEMA,
        "status": "reference_cached",
        "trainable": False,
        "blocked_by": ["pair_audit_unlistened"],
        "rows": total,
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "rechecked": checked,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "cache_sha256": sha256_file(cache_path),
        "seed": seed,
    }


def publish_logprob_pairs(
    train_pairs: list[dict[str, Any]],
    val_pairs: list[dict[str, Any]],
    cache_path: Path,
    output_dir: Path,
    report: dict[str, Any],
) -> dict[str, Any]:
    """Write filled pair files only after the cache and recheck have succeeded."""
    pairs = ordered_pairs(train_pairs, val_pairs)
    cached = completed_logprob_rows(cache_path, pairs)
    if len(cached) != len(pairs):
        raise RefLogprobError("refusing to publish a partial reference cache")
    by_id = {row["pair_id"]: row for row in cached}
    if (output_dir / "train_pairs.jsonl").exists() or (output_dir / "val_pairs.jsonl").exists():
        raise RefLogprobError("logprob pair files already exist")

    def fill(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        filled = []
        for row in rows:
            saved = by_id[row["pair_id"]]
            item = dict(row)
            item["ref_chosen_logp"] = saved["ref_chosen_logp"]
            item["ref_rejected_logp"] = saved["ref_rejected_logp"]
            filled.append(item)
        return filled

    output_dir.mkdir(parents=True, exist_ok=True)
    staging = output_dir / ".ref-staging"
    if staging.exists():
        raise RefLogprobError(f"staging already exists: {staging}")
    staging.mkdir()
    try:
        write_jsonl(staging / "train_pairs.jsonl", fill(train_pairs))
        write_jsonl(staging / "val_pairs.jsonl", fill(val_pairs))
        report = dict(report)
        report["train_pairs_sha256"] = sha256_file(staging / "train_pairs.jsonl")
        report["val_pairs_sha256"] = sha256_file(staging / "val_pairs.jsonl")
        (staging / "ref_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for name in ("train_pairs.jsonl", "val_pairs.jsonl", "ref_report.json"):
            (staging / name).replace(output_dir / name)
    finally:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()
    return report


def freeze_reference_model(model: Any) -> None:
    """Drop the SFT training hook so an inference forward does not request gradients."""
    thinker = getattr(model, "thinker", None)
    if thinker is None or not hasattr(thinker, "get_input_embeddings"):
        raise RefLogprobError("reference model has no thinker embeddings")
    embeddings = thinker.get_input_embeddings()
    hooks = getattr(embeddings, "_forward_hooks", None)
    if not isinstance(hooks, dict):
        raise RefLogprobError("reference embedding hooks cannot be cleared")
    hooks.clear()
    for module in (model, thinker):
        disable = getattr(module, "gradient_checkpointing_disable", None)
        if callable(disable):
            disable()
    for module in (model, thinker):
        flag = getattr(module, "is_gradient_checkpointing", False)
        if callable(flag):
            flag = flag()
        if flag:
            raise RefLogprobError("gradient checkpointing is still enabled")
    model.eval()
    seen = False
    for param in model.parameters():
        param.requires_grad_(False)
        seen = True
    if not seen:
        raise RefLogprobError("reference model has no parameters")


def load_published_release(release_dir: Path) -> dict[str, Any]:
    path = release_dir / "release.json"
    if not path.is_file():
        raise RefLogprobError(f"missing release.json in {release_dir}")
    release = json.loads(path.read_text(encoding="utf-8"))
    if release.get("schema_version") != RELEASE_SCHEMA:
        raise RefLogprobError("release schema mismatch")
    if release.get("status") != "published" or release.get("publishable") is not True:
        raise RefLogprobError("reference logprob requires a published release")
    adapter = release_dir / "adapter" / "adapter.safetensors"
    if not adapter.is_file():
        raise RefLogprobError("published release is missing adapter.safetensors")
    if sha256_file(adapter) != release.get("adapter_sha256"):
        raise RefLogprobError("adapter hash does not match release.json")
    return release


def _prepare_runtime(config: dict[str, Any]) -> None:
    if config["runtime"]["deterministic_algorithms"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    from train.train_sft import set_seeds

    if config["runtime"]["deterministic_algorithms"]:
        torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    set_seeds(int(config["seed"]))


def build_reference_scorer(config: dict[str, Any], release_dir: Path, audio_root: Path) -> Score:
    """Load the published adapter and return a scorer. This downloads nothing new when cached."""
    import torch
    from train.train_sft import (
        collate_example,
        load_adapter,
        load_audio,
        load_sft_model,
        move_batch,
        prefix_text,
        supervised_nll_sum,
        thinker_inputs,
    )

    _prepare_runtime(config)
    release = load_published_release(release_dir)
    model, processor, _report = load_sft_model(config)
    load_adapter(model, release_dir / "adapter" / "adapter.safetensors")
    freeze_reference_model(model)
    prefix = prefix_text(processor, config["decode"]["eval"]["context"], config["official_language"])
    eos_token = processor.tokenizer.eos_token
    eos_id = processor.tokenizer.convert_tokens_to_ids("<|im_end|>")
    if not isinstance(eos_token, str) or not isinstance(eos_id, int):
        raise RefLogprobError("tokenizer did not provide an EOS token")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    audio_cache: dict[str, Any] = {}

    def score(row: dict[str, Any], text: str) -> tuple[float, int]:
        audio_value = str(row["audio"])
        audio_path = Path(audio_value)
        if not audio_path.is_absolute():
            audio_path = audio_root / audio_value
        key = str(audio_path)
        if key not in audio_cache:
            audio_cache[key] = load_audio(audio_path)
        example = collate_example(
            processor,
            {"sample_id": row["pair_id"], "text": text},
            audio_cache[key],
            prefix=prefix,
            eos_token=eos_token,
            eos_id=int(eos_id),
            max_input_tokens=int(config["data"]["max_input_tokens"]),
            max_transcript_tokens=int(config["data"]["max_transcript_tokens"]),
        )
        batch = move_batch(dict(example), device, dtype)
        with torch.inference_mode():
            outputs = model.thinker(**thinker_inputs(batch), use_cache=False)
            nll_sum, count = supervised_nll_sum(outputs.logits, batch["labels"])
        return reference_sequence_logprob([float(nll_sum.detach().float().item())]), int(count)

    score.release_id = release["release_id"]  # type: ignore[attr-defined]
    return score


def _copy_cache(cache_path: Path, sync_path: Path) -> None:
    import shutil

    sync_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(cache_path, sync_path)
    if sha256_file(cache_path) != sha256_file(sync_path):
        print(f"REF_LOGP_SYNC_FAIL {sync_path}", flush=True)
        return
    print(f"REF_LOGP_SYNC {sync_path.name}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache DPO reference sequence logprobs.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--pairs-dir", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sync-path", type=Path)
    parser.add_argument("--recheck-limit", type=int, default=20)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from train.sft_schema import load_train_config

        config = load_train_config(args.config)
        if int(config["seed"]) != args.seed:
            raise RefLogprobError(f"seed {args.seed} does not match the config")
        if config["runtime"]["deterministic_algorithms"]:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        train_pairs = read_jsonl(args.pairs_dir / "train_pairs.jsonl")
        val_pairs = read_jsonl(args.pairs_dir / "val_pairs.jsonl")
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        cache_path = output_dir / "ref_logprobs.jsonl"
        if (output_dir / "train_pairs.jsonl").exists():
            raise RefLogprobError("filled train pairs already exist")
        scorer = build_reference_scorer(config, args.release_dir, args.audio_root)

        def on_progress(done: int, total: int) -> None:
            print(f"REF_LOGP_PROGRESS {done}/{total}", flush=True)

        def sync_fn(done: int, total: int) -> None:
            if args.sync_path is None:
                return
            try:
                _copy_cache(cache_path, args.sync_path)
            except OSError as exc:
                print(f"REF_LOGP_SYNC_FAIL {exc}", flush=True)

        print(f"REF_LOGP_RESUME {0 if not cache_path.exists() else 'see-cache'}/{len(train_pairs) + len(val_pairs)}", flush=True)
        report = cache_reference_logprobs(
            train_pairs,
            val_pairs,
            scorer,
            cache_path,
            seed=args.seed,
            recheck_limit=args.recheck_limit,
            on_progress=on_progress,
            sync_fn=sync_fn,
        )
        report["release_id"] = getattr(scorer, "release_id", None)
        report = publish_logprob_pairs(train_pairs, val_pairs, cache_path, output_dir, report)
        if args.sync_path is not None:
            final_dir = args.sync_path.parent / "pairs_logprob"
            final_dir.mkdir(parents=True, exist_ok=True)
            for name in ("train_pairs.jsonl", "val_pairs.jsonl", "ref_logprobs.jsonl", "ref_report.json"):
                source = output_dir / name
                target = final_dir / name
                _copy_cache(source, target)
                if sha256_file(source) != sha256_file(target):
                    raise RefLogprobError(f"final sync hash mismatch for {name}")
        print(
            "REF_LOGP_DONE "
            f"train={report['train_pairs']} val={report['val_pairs']} "
            f"rechecked={report['rechecked']} max_abs={report['max_abs_error']}",
            flush=True,
        )
    except RefLogprobError as exc:
        print(f"REF_LOGP_FAIL {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"REF_LOGP_FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
