"""Paired significance for two scored prediction files.

``eval_wer.py`` writes ``scored.jsonl`` per run. Point-estimate gates on those
runs sit below the noise floor of 4-step RL pilots (see
docs/qwen3-asr/28_rl_v30_trainer_sync_and_stats.md), so this tool compares a
baseline and a candidate sample by sample and reports:

* the mean per-sample ``error_rate`` delta (candidate - baseline; negative is
  better) with a fixed-seed percentile bootstrap 95% interval,
* how many normalized predictions changed, and how many of the changed rows got
  better or worse, with a two-sided exact sign test,
* the same numbers per ``condition_group``, per ``language|condition_group``
  and per ``language|scenario``.

Several scored files per side are merged by ``sample_id`` (v31 judges on
``validation`` plus the ``dpo_val_pool`` degraded rows). Each row must carry
``sample_id``, ``error_rate``, ``prediction_normalized``, ``scenario``,
``language`` and ``condition_group``; paired metadata must match exactly.
``--gate`` adds the
v31 statistical acceptance: degraded CI upper bound < 0 and clean CI upper
bound <= ``--clean-margin`` (docs/qwen3-asr/29_rl_v31_scale_design.md).

It only reads files. It never loads a model and never edits its inputs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_SEED = 20260722
DEFAULT_BOOTSTRAP = 2000
EPS = 1e-12
SCORED_FIELDS = ("sample_id", "error_rate", "prediction_normalized", "scenario", "language", "condition_group")
PAIR_METADATA_FIELDS = ("language", "scenario", "condition_group")


def load_scored(path: Path) -> dict[str, dict[str, Any]]:
    """Read a scored.jsonl keyed by sample_id; duplicate or missing ids are errors."""
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = row.get("sample_id")
            if sample_id is None or not str(sample_id).strip():
                raise ValueError(f"{path}:{line_no} has no sample_id")
            sample_id = str(sample_id)
            if sample_id in rows:
                raise ValueError(f"{path}:{line_no} duplicate sample_id {sample_id!r}")
            for field in SCORED_FIELDS:
                if field not in row:
                    raise ValueError(f"{path}:{line_no} has no {field}")
                if field != "prediction_normalized" and (
                    row[field] is None or not str(row[field]).strip()
                ):
                    raise ValueError(f"{path}:{line_no} has no {field}")
            rows[sample_id] = row
    if not rows:
        raise ValueError(f"{path} has no rows")
    return rows


def pair_rows(
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Return rows paired by sample_id; the two id sets must be identical."""
    only_base = sorted(set(baseline) - set(candidate))
    only_cand = sorted(set(candidate) - set(baseline))
    if only_base or only_cand:
        raise ValueError(
            "sample_id sets differ: "
            f"{len(only_base)} only in baseline (e.g. {only_base[:3]}), "
            f"{len(only_cand)} only in candidate (e.g. {only_cand[:3]})"
        )
    pairs = []
    for key in sorted(baseline):
        base_row, candidate_row = baseline[key], candidate[key]
        mismatches = [
            field for field in PAIR_METADATA_FIELDS
            if base_row.get(field) != candidate_row.get(field)
        ]
        if mismatches:
            raise ValueError(
                f"metadata mismatch for sample_id {key!r}: {', '.join(mismatches)}"
            )
        pairs.append((base_row, candidate_row))
    return pairs


def sign_test_p(better: int, worse: int) -> float:
    """Two-sided exact binomial sign test with p=0.5; ties are excluded."""
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def bootstrap_ci(
    deltas: Sequence[float], *, iterations: int, seed: int
) -> tuple[float, float]:
    """Percentile bootstrap 95% interval for the mean of ``deltas``."""
    n = len(deltas)
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")
    rng = random.Random(seed)
    means = []
    for _ in range(iterations):
        total = 0.0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    lo = means[int(0.025 * (iterations - 1))]
    hi = means[int(math.ceil(0.975 * (iterations - 1)))]
    return lo, hi


def verdict(ci_lo: float, ci_hi: float) -> str:
    """Interpret the interval on the (candidate - baseline) error-rate delta."""
    if ci_hi < 0.0:
        return "significant_improvement"
    if ci_lo > 0.0:
        return "significant_regression"
    return "indistinguishable_from_noise"


def summarize(
    pairs: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    deltas: list[float] = []
    better = worse = tied = changed = 0
    for base, cand in pairs:
        delta = float(cand["error_rate"]) - float(base["error_rate"])
        deltas.append(delta)
        if delta < -EPS:
            better += 1
        elif delta > EPS:
            worse += 1
        else:
            tied += 1
        if base.get("prediction_normalized") != cand.get("prediction_normalized"):
            changed += 1
    n = len(deltas)
    lo, hi = bootstrap_ci(deltas, iterations=iterations, seed=seed)
    return {
        "samples": n,
        "mean_delta": sum(deltas) / n,
        "ci95": [lo, hi],
        "verdict": verdict(lo, hi),
        "changed_predictions": changed,
        "better": better,
        "worse": worse,
        "tied": tied,
        "sign_test_p": sign_test_p(better, worse),
    }


def _group_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('language', 'unknown')}|{row.get('scenario', 'unknown')}"


def _language_condition_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('language', 'unknown')}|{row.get('condition_group', 'unknown')}"


def compare(
    baseline: Mapping[str, Mapping[str, Any]],
    candidate: Mapping[str, Mapping[str, Any]],
    *,
    iterations: int = DEFAULT_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    pairs = pair_rows(baseline, candidate)
    result: dict[str, Any] = {
        "convention": "delta = candidate error_rate - baseline error_rate; negative is better",
        "seed": seed,
        "bootstrap_iterations": iterations,
        "overall": summarize(pairs, iterations=iterations, seed=seed),
        "by_condition_group": {},
        "by_language_condition": {},
        "by_language_scenario": {},
    }
    groupings = (
        ("by_condition_group", lambda row: str(row.get("condition_group", "unknown"))),
        ("by_language_condition", _language_condition_key),
        ("by_language_scenario", _group_key),
    )
    for field, key_fn in groupings:
        buckets: dict[str, list] = {}
        for base, cand in pairs:
            buckets.setdefault(key_fn(base), []).append((base, cand))
        for name in sorted(buckets):
            result[field][name] = summarize(buckets[name], iterations=iterations, seed=seed)
    return result


def load_scored_many(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    """Merge several scored.jsonl files; a sample_id in two files is an error."""
    merged: dict[str, dict[str, Any]] = {}
    origin: dict[str, Path] = {}
    for path in paths:
        for sample_id, row in load_scored(path).items():
            if sample_id in merged:
                raise ValueError(
                    f"sample_id {sample_id!r} appears in both {origin[sample_id]} and {path}"
                )
            merged[sample_id] = row
            origin[sample_id] = path
    return merged


DEFAULT_CLEAN_MARGIN = 0.002


def paired_gate(
    report: Mapping[str, Any],
    *,
    improvement_group: str = "degraded",
    noninferior_group: str = "clean",
    clean_margin: float = DEFAULT_CLEAN_MARGIN,
) -> dict[str, Any]:
    """PASSED iff ``improvement_group`` is significantly better (CI upper < 0)
    and ``noninferior_group``'s CI upper bound is at most ``clean_margin``.

    A missing group fails the gate. Per-language rows are reported by
    ``compare`` but are not gate conditions
    (docs/qwen3-asr/29_rl_v31_scale_design.md).
    """
    groups = report.get("by_condition_group", {})
    failures: list[str] = []
    improve = groups.get(improvement_group)
    if improve is None:
        failures.append(f"no {improvement_group} rows")
    elif not float(improve["ci95"][1]) < 0.0:
        failures.append(
            f"{improvement_group} ci95 upper {float(improve['ci95'][1]):+.6f} is not < 0"
        )
    noninf = groups.get(noninferior_group)
    if noninf is None:
        failures.append(f"no {noninferior_group} rows")
    elif float(noninf["ci95"][1]) > float(clean_margin):
        failures.append(
            f"{noninferior_group} ci95 upper {float(noninf['ci95'][1]):+.6f} > margin {clean_margin:+.6f}"
        )
    return {
        "status": "PASSED" if not failures else "FAILED",
        "failures": failures,
        "improvement_group": improvement_group,
        "noninferior_group": noninferior_group,
        "clean_margin": float(clean_margin),
    }


def tail_gate(
    pairs: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]],
    *,
    required_max_new_tokens: int,
) -> dict[str, Any]:
    """Reject decode-contract drift and a new severe candidate tail failure."""
    expected = {
        "max_new_tokens": int(required_max_new_tokens),
        "do_sample": False,
        "num_beams": 1,
        "num_return_sequences": 1,
    }
    contract_mismatches = 0
    nonfinite_or_negative = 0
    candidate_severe = 0
    severe_regressions = 0
    for base, candidate in pairs:
        for row in (base, candidate):
            if row.get("decoding") != expected:
                contract_mismatches += 1
            value = float(row.get("error_rate", float("nan")))
            if not math.isfinite(value) or value < 0.0:
                nonfinite_or_negative += 1
        cand_rate = float(candidate["error_rate"])
        base_rate = float(base["error_rate"])
        cand_severe = cand_rate >= 2.0 and bool(
            candidate.get("too_long") or candidate.get("hallucination_like")
        )
        base_severe = base_rate >= 2.0 and bool(
            base.get("too_long") or base.get("hallucination_like")
        )
        if cand_severe:
            candidate_severe += 1
        if cand_severe and cand_rate > base_rate and not base_severe:
            severe_regressions += 1
    failures = []
    if contract_mismatches:
        failures.append(f"{contract_mismatches} rows do not match decoding contract")
    if nonfinite_or_negative:
        failures.append(f"{nonfinite_or_negative} rows have invalid error_rate")
    if severe_regressions:
        failures.append(f"{severe_regressions} new severe candidate tail regressions")
    return {
        "status": "PASSED" if not failures else "FAILED",
        "required_max_new_tokens": int(required_max_new_tokens),
        "contract_mismatches": contract_mismatches,
        "invalid_error_rates": nonfinite_or_negative,
        "candidate_severe_outputs": candidate_severe,
        "severe_regressions": severe_regressions,
        "failures": failures,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--baseline", required=True, type=Path, nargs="+",
        help="Baseline scored.jsonl file(s); several files are merged by sample_id.",
    )
    parser.add_argument(
        "--candidate", required=True, type=Path, nargs="+",
        help="Candidate scored.jsonl file(s); must cover the same sample_ids.",
    )
    parser.add_argument("--output", type=Path, help="Write the JSON report here.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP)
    parser.add_argument(
        "--gate", action="store_true",
        help="Add a 'gate' section: degraded CI upper < 0 and clean CI upper <= --clean-margin.",
    )
    parser.add_argument("--clean-margin", type=float, default=DEFAULT_CLEAN_MARGIN)
    parser.add_argument(
        "--tail-gate", action="store_true",
        help="Also require the per-row decoding contract and reject new severe tail regressions.",
    )
    parser.add_argument(
        "--require-max-new-tokens", type=int,
        help="Required max_new_tokens for --tail-gate.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap < 100:
        print("Error: --bootstrap must be at least 100", file=sys.stderr)
        return 2
    try:
        report = compare(
            load_scored_many(args.baseline),
            load_scored_many(args.candidate),
            iterations=args.bootstrap,
            seed=args.seed,
        )
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    report["inputs"] = {
        "baseline": [str(p) for p in args.baseline],
        "candidate": [str(p) for p in args.candidate],
    }
    if args.gate:
        report["gate"] = paired_gate(report, clean_margin=args.clean_margin)
    if args.tail_gate:
        if args.require_max_new_tokens is None or args.require_max_new_tokens <= 0:
            print("Error: --tail-gate requires positive --require-max-new-tokens", file=sys.stderr)
            return 2
        report["tail_gate"] = tail_gate(
            pair_rows(load_scored_many(args.baseline), load_scored_many(args.candidate)),
            required_max_new_tokens=args.require_max_new_tokens,
        )
    text = json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    overall = report["overall"]
    print(
        f"n={overall['samples']} mean_delta={overall['mean_delta']:+.6f} "
        f"ci95=[{overall['ci95'][0]:+.6f}, {overall['ci95'][1]:+.6f}] "
        f"changed={overall['changed_predictions']} better={overall['better']} "
        f"worse={overall['worse']} sign_p={overall['sign_test_p']:.4f} "
        f"verdict={overall['verdict']}"
    )
    if args.gate:
        print(f"gate={report['gate']['status']} failures={report['gate']['failures']}")
    if args.tail_gate:
        print(f"tail_gate={report['tail_gate']['status']} failures={report['tail_gate']['failures']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
