#!/usr/bin/env python3
"""分析 ASR scored JSONL 中的错误样本。

输入通常来自 `evaluation/eval_wer.py` 的 `--scored-jsonl` 输出。该脚本不
重新计算 WER/CER，而是基于已有字段做错误分析：

1. 按 scenario、text_length_bucket 聚合 WER、空输出率和长度异常。
2. 输出 WER 最高的 worst cases，便于人工查看 reference/prediction。
3. 用简单启发式标注 empty、too_short、too_long、repeat_like、
   hallucination_like 等错误类型。

这些标签只用于快速筛选，不等价于人工最终结论。尤其是
hallucination_like 只表示“高错误率且非空输出”，后续仍应抽样听音频。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件并跳过空行。"""
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, data: dict[str, Any]) -> None:
    """写出缩进 JSON，并自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出 JSONL，并自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact_text(value: Any, max_chars: int) -> str:
    """把长文本压成一行，方便 CSV 和终端预览。"""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def words(text: str) -> list[str]:
    """把归一化文本粗略切词，用于重复片段检测。"""
    return str(text or "").lower().split()


def has_repeated_bigram(tokens: list[str]) -> bool:
    """检测连续重复 bigram。

    例如 "the sky is falling the sky is falling" 这类输出通常意味着模型在
    退化音频上进入重复补全模式。
    """
    if len(tokens) < 4:
        return False
    bigrams = list(zip(tokens, tokens[1:]))
    counts = Counter(bigrams)
    return any(count >= 2 for count in counts.values())


def has_run_repetition(tokens: list[str], min_run: int = 3) -> bool:
    """检测同一个 token 连续重复。"""
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


def error_tags(row: dict[str, Any]) -> list[str]:
    """为单条 scored 样本打启发式错误标签。"""
    wer = float(row.get("wer", 0.0) or 0.0)
    ref_len = int(row.get("ref_len", 0) or 0)
    length_ratio = float(row.get("length_ratio", 0.0) or 0.0)
    pred_norm = str(row.get("prediction_normalized") or row.get("prediction") or "")
    pred_tokens = words(pred_norm)
    tags: list[str] = []

    if bool(row.get("empty_output", False)) or not pred_norm.strip():
        tags.append("empty_output")
    if ref_len == 0:
        tags.append("empty_reference")
    if ref_len > 0 and length_ratio < 0.5:
        tags.append("too_short")
    if ref_len > 0 and length_ratio > 1.5:
        tags.append("too_long")
    if wer >= 0.8 and pred_norm.strip():
        tags.append("hallucination_like")
    if wer >= 1.0:
        tags.append("insertion_heavy")
    if has_repeated_bigram(pred_tokens) or has_run_repetition(pred_tokens):
        tags.append("repeat_like")
    if 0.0 < wer < 0.2:
        tags.append("minor_error")
    return tags


def numeric(row: dict[str, Any], key: str) -> float:
    """安全读取数值字段。"""
    return float(row.get(key, 0.0) or 0.0)


def aggregate(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    """按指定字段聚合错误统计。"""
    buckets: dict[tuple[str, ...], dict[str, Any]] = defaultdict(lambda: {
        "samples": 0,
        "num_edits": 0,
        "ref_len": 0,
        "empty_outputs": 0,
        "too_short": 0,
        "too_long": 0,
        "repeat_like": 0,
        "hallucination_like": 0,
    })

    for row in rows:
        group = tuple(str(row.get(key, "UNKNOWN")) for key in keys)
        bucket = buckets[group]
        tags = set(row.get("error_tags", []))
        bucket["samples"] += 1
        bucket["num_edits"] += int(row.get("num_edits", 0) or 0)
        bucket["ref_len"] += int(row.get("ref_len", 0) or 0)
        bucket["empty_outputs"] += int("empty_output" in tags)
        bucket["too_short"] += int("too_short" in tags)
        bucket["too_long"] += int("too_long" in tags)
        bucket["repeat_like"] += int("repeat_like" in tags)
        bucket["hallucination_like"] += int("hallucination_like" in tags)

    result: list[dict[str, Any]] = []
    for group, bucket in sorted(buckets.items()):
        ref_len = bucket["ref_len"]
        samples = bucket["samples"]
        item = {key: group[index] for index, key in enumerate(keys)}
        item.update({
            "samples": samples,
            "num_edits": bucket["num_edits"],
            "ref_len": ref_len,
            "error_rate": round(bucket["num_edits"] / ref_len, 6) if ref_len else 0.0,
            "empty_output_rate": round(bucket["empty_outputs"] / samples, 6) if samples else 0.0,
            "too_short_rate": round(bucket["too_short"] / samples, 6) if samples else 0.0,
            "too_long_rate": round(bucket["too_long"] / samples, 6) if samples else 0.0,
            "repeat_like_rate": round(bucket["repeat_like"] / samples, 6) if samples else 0.0,
            "hallucination_like_rate": round(bucket["hallucination_like"] / samples, 6) if samples else 0.0,
        })
        result.append(item)
    return result


def case_view(row: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """提取人工分析最需要看的字段。"""
    return {
        "scenario": row.get("scenario", ""),
        "text_length_bucket": row.get("text_length_bucket", ""),
        "wer": row.get("wer", 0.0),
        "num_edits": row.get("num_edits", 0),
        "ref_len": row.get("ref_len", 0),
        "length_ratio": row.get("length_ratio", 0.0),
        "error_tags": row.get("error_tags", []),
        "audio": row.get("audio", ""),
        "answer": compact_text(row.get("answer", ""), max_chars),
        "prediction": compact_text(row.get("prediction", ""), max_chars),
        "error": row.get("error", ""),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出 CSV；字段来自所有行的 key 并保持常用字段在前。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "scenario",
        "text_length_bucket",
        "samples",
        "num_edits",
        "ref_len",
        "error_rate",
        "wer",
        "length_ratio",
        "error_tags",
        "audio",
        "answer",
        "prediction",
        "error",
    ]
    present = {key for row in rows for key in row.keys()}
    ordered = [key for key in preferred if key in present]
    extras = [key for key in present if key not in ordered]
    keys = ordered + sorted(extras)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            if isinstance(out.get("error_tags"), list):
                out["error_tags"] = "|".join(out["error_tags"])
            writer.writerow(out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-jsonl", required=True, help="Input scored JSONL from eval_wer.py.")
    parser.add_argument("--output-dir", required=True, help="Directory for analysis outputs.")
    parser.add_argument("--top-k", type=int, default=30, help="Number of worst cases to save.")
    parser.add_argument("--max-text-chars", type=int, default=220, help="Max answer/prediction chars in CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scored_path = Path(args.scored_jsonl).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    rows = read_jsonl(scored_path)

    analyzed: list[dict[str, Any]] = []
    for row in rows:
        out = dict(row)
        out["error_tags"] = error_tags(out)
        analyzed.append(out)

    worst = sorted(
        analyzed,
        key=lambda row: (numeric(row, "wer"), int(row.get("num_edits", 0) or 0)),
        reverse=True,
    )[: args.top_k]
    worst_view = [case_view(row, args.max_text_chars) for row in worst]
    flagged = [row for row in analyzed if row.get("error_tags")]
    flagged_view = [case_view(row, args.max_text_chars) for row in flagged]

    by_scenario = aggregate(analyzed, ["scenario"])
    by_scenario_bucket = aggregate(analyzed, ["scenario", "text_length_bucket"])
    tag_counts = Counter(tag for row in analyzed for tag in row.get("error_tags", []))

    summary = {
        "input": str(scored_path),
        "samples": len(analyzed),
        "tag_counts": dict(sorted(tag_counts.items())),
        "by_scenario": by_scenario,
        "by_scenario_bucket": by_scenario_bucket,
        "top_k": args.top_k,
    }

    write_json(output_dir / "analysis_summary.json", summary)
    write_jsonl(output_dir / "worst_cases.jsonl", worst_view)
    write_csv(output_dir / "worst_cases.csv", worst_view)
    write_jsonl(output_dir / "flagged_cases.jsonl", flagged_view)
    write_csv(output_dir / "by_scenario.csv", by_scenario)
    write_csv(output_dir / "by_scenario_bucket.csv", by_scenario_bucket)

    print(f"[done] analyzed {len(analyzed)} rows")
    print(f"[done] saved summary to {output_dir / 'analysis_summary.json'}")
    print(f"[done] saved worst cases to {output_dir / 'worst_cases.csv'}")


if __name__ == "__main__":
    main()
