#!/usr/bin/env python3
"""计算 ASR prediction JSONL 的 WER/CER 指标。

输入文件通常来自 `inference/qwen3_asr_base_infer.py`，每行至少包含：
`answer` 和 `prediction`。脚本会保留原始字段，并追加 metric、wer、
num_edits、ref_len 等评测字段。

这个脚本承担 baseline、LoRA 和 router 三类推理结果的统一评测入口。
设计上故意不依赖 jiwer、evaluate 等外部库，原因是 Colab Free 环境经常
因为依赖解析升级 numpy、requests、click 等包，进而影响模型推理环境。
因此这里使用标准库实现最小可复现版本：

1. 读取 prediction JSONL。
2. 对 reference/prediction 做轻量归一化。
3. 根据语言自动选择 WER 或 CER。
4. 计算每条样本的编辑距离和错误率。
5. 聚合 overall 与 scenario-level 指标。
6. 保存带明细的 scored JSONL、指标 JSON 和按场景聚合 CSV。

注意：字段名仍然统一使用 `wer`，即使中文样本实际计算的是 CER。
原因是后续训练/评测表格会固定读取这一列；真实指标类型由同一行的
`metric` 字段标识。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL 文件并跳过空行。

    prediction JSONL 在批量推理中可能会被分批追加写入；跳过空行可以让
    人工检查或手动拼接文件后依然能被评测脚本读取。
    """
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """按 JSONL 格式写出结果，并自动创建父目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def has_cjk(text: str) -> bool:
    """判断文本中是否包含中日韩统一表意文字。

    英文 ASR 通常按词计算 WER；中文没有天然空格分词，早期 MVP 先按字符
    计算 CER，避免引入额外分词器带来的环境依赖和口径差异。
    """
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def normalize_text(text: str, lowercase: bool, remove_punctuation: bool) -> str:
    """做轻量文本归一化，避免标点和大小写主导 WER/CER。

    这里不做激进归一化，例如数字读法、繁简转换、英文缩写展开等。
    原因是当前阶段需要先看模型原始 ASR 能力，避免评测脚本把真实错误
    过度“修平”。后续如果引入更复杂归一化，应在测试文档中固定口径。
    """
    text = str(text or "").strip()
    if lowercase:
        # 英文大小写通常不是 ASR 核心错误，默认忽略。
        text = text.lower()
    if remove_punctuation:
        chars = []
        for ch in text:
            # Unicode P* 覆盖中英文标点，例如逗号、句号、问号、引号等。
            # 替换为空格而不是直接删除，是为了避免英文单词被意外粘连。
            if unicodedata.category(ch).startswith("P"):
                chars.append(" ")
            else:
                chars.append(ch)
        text = "".join(chars)
    # 把多个空白压成一个空格，让换行、tab、连续空格不影响 WER。
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str, metric: str) -> list[str]:
    """根据指标类型把文本切成评测 token。

    - WER：按空格切词，适合英文和已分词语言。
    - CER：去掉空格后按字符切分，适合中文 smoke/baseline 阶段。
    """
    if metric == "cer":
        return [ch for ch in text.replace(" ", "")]
    return text.split()


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    """计算 Levenshtein edit distance。

    编辑距离表示把 hypothesis 变成 reference 至少需要多少次插入、删除、
    替换。WER/CER 的核心公式都是：

    error_rate = 编辑次数 / reference token 数

    这里使用动态规划的一维滚动数组实现，避免为长音频转写构建完整矩阵。
    """
    # prev[j] 表示上一行到 hyp 前 j 个 token 的最小编辑次数。
    prev = list(range(len(hyp) + 1))
    for i, ref_item in enumerate(ref, start=1):
        # curr[0] = i，表示 hyp 为空时，需要删除 reference 的前 i 个 token。
        curr = [i]
        for j, hyp_item in enumerate(hyp, start=1):
            cost = 0 if ref_item == hyp_item else 1
            # 三个候选分别对应：删除、插入、替换/匹配。
            curr.append(min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            ))
        prev = curr
    return prev[-1]


def metric_for_item(item: dict[str, Any], ref: str, pred: str) -> str:
    """为单条样本选择 WER 或 CER。

    优先使用 manifest 中的 `language` 字段，因为这是数据侧明确声明；
    如果 language 缺失，再根据 reference/prediction 是否包含 CJK 字符兜底。
    """
    language = str(item.get("language") or "").lower()
    if language in {"zh", "cn", "chinese", "yue", "ja", "jp", "japanese"}:
        return "cer"
    if has_cjk(ref + pred):
        return "cer"
    return "wer"


def score_item(
    item: dict[str, Any],
    lowercase: bool,
    remove_punctuation: bool,
) -> dict[str, Any]:
    """计算单条样本指标，并把评测字段追加到原始样本上。

    输入样本来自推理输出，通常包含 audio、answer、prediction、scenario。
    输出样本会保留所有原始字段，便于后续错误分析时追溯音频、场景、模型
    输出和失败原因。
    """
    # `answer` 是本项目标准字段；`text` 是兼容少量外部数据或临时 manifest。
    ref_raw = str(item.get("answer") or item.get("text") or "")
    pred_raw = str(item.get("prediction") or "")
    metric = metric_for_item(item, ref_raw, pred_raw)

    # 归一化后的文本会写回 scored JSONL，方便人工排查指标为什么这样算。
    ref_norm = normalize_text(ref_raw, lowercase, remove_punctuation)
    pred_norm = normalize_text(pred_raw, lowercase, remove_punctuation)
    ref_tokens = tokenize(ref_norm, metric)
    pred_tokens = tokenize(pred_norm, metric)
    edits = edit_distance(ref_tokens, pred_tokens)
    ref_len = len(ref_tokens)

    # reference 为空通常代表数据有问题。这里给 0.0，避免 smoke test 中断；
    # 后续数据质量检查应单独拦截空 reference。
    score = edits / ref_len if ref_len else 0.0

    out = dict(item)
    out["reference_normalized"] = ref_norm
    out["prediction_normalized"] = pred_norm
    out["metric"] = metric
    out["wer"] = round(float(score), 6)
    out["num_edits"] = int(edits)
    out["ref_len"] = int(ref_len)
    # empty_output 是鲁棒 ASR 的关键风险信号：噪声/远场音频容易触发空转写。
    out["empty_output"] = len(pred_norm) == 0
    # length_ratio 用于快速发现重复输出或过短输出：
    # 远大于 1 可能是重复/幻觉，接近 0 可能是漏识别或空输出。
    out["length_ratio"] = round((len(pred_tokens) / ref_len), 6) if ref_len else 0.0
    return out


def aggregate(rows: list[dict[str, Any]], group_key: str | None = None) -> list[dict[str, Any]]:
    """聚合整体或按字段分组的错误率。

    group_key 为空时输出 overall；传入 `scenario` 时输出 clean/noise/reverb
    等场景级指标。错误率采用“总编辑次数 / 总 reference 长度”，而不是
    每条样本错误率的简单平均，这样长短样本混合时口径更稳定。
    """
    buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "samples": 0,
        "num_edits": 0,
        "ref_len": 0,
        "empty_outputs": 0,
    })
    for row in rows:
        # 缺失 scenario 等分组字段时归入 ALL，避免丢样本。
        key = str(row.get(group_key, "ALL")) if group_key else "ALL"
        bucket = buckets[key]
        bucket["samples"] += 1
        bucket["num_edits"] += int(row.get("num_edits", 0))
        bucket["ref_len"] += int(row.get("ref_len", 0))
        bucket["empty_outputs"] += int(bool(row.get("empty_output", False)))

    result = []
    for key, bucket in sorted(buckets.items()):
        ref_len = bucket["ref_len"]
        error_rate = bucket["num_edits"] / ref_len if ref_len else 0.0
        samples = bucket["samples"]
        result.append({
            "group": key,
            "samples": samples,
            "num_edits": bucket["num_edits"],
            "ref_len": ref_len,
            "error_rate": round(float(error_rate), 6),
            "empty_output_rate": round(bucket["empty_outputs"] / samples, 6) if samples else 0.0,
        })
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """写出按场景聚合的 CSV，方便 Colab 直接预览或复制到表格。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["group", "samples", "num_edits", "ref_len", "error_rate", "empty_output_rate"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。

    输出拆成三个文件：
    - scored JSONL：每条样本的明细，适合错误分析。
    - metrics JSON：overall 与 scenario 汇总，适合实验记录。
    - scenario CSV：表格化场景指标，适合 Colab 展示和人工对比。
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-jsonl", required=True)
    parser.add_argument("--scored-jsonl", required=True)
    parser.add_argument("--metrics-json", required=True)
    parser.add_argument("--metrics-by-scenario-csv", required=True)
    parser.add_argument("--no-lowercase", action="store_true")
    parser.add_argument("--keep-punctuation", action="store_true")
    return parser.parse_args()


def main() -> None:
    """命令行入口：读取预测、逐条评分、聚合指标、写出结果。"""
    args = parse_args()
    rows = read_jsonl(Path(args.predictions_jsonl).expanduser())

    # 每条样本独立评分。即使某条 prediction 为空，也会进入 scored 文件；
    # 推理阶段的错误字段会原样保留，便于定位模型加载、音频读取或生成失败。
    scored = [
        score_item(
            row,
            lowercase=not args.no_lowercase,
            remove_punctuation=not args.keep_punctuation,
        )
        for row in rows
    ]

    # 空输入文件不直接报错，方便流水线先跑通目录和文件写入逻辑。
    overall = aggregate(scored)[0] if scored else {
        "group": "ALL",
        "samples": 0,
        "num_edits": 0,
        "ref_len": 0,
        "error_rate": 0.0,
        "empty_output_rate": 0.0,
    }
    by_scenario = aggregate(scored, "scenario")

    # 三类输出同时保存：明细、机器可读汇总、表格汇总。
    write_jsonl(Path(args.scored_jsonl).expanduser(), scored)
    Path(args.metrics_json).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json).expanduser().write_text(
        json.dumps({"overall": overall, "by_scenario": by_scenario}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(Path(args.metrics_by_scenario_csv).expanduser(), by_scenario)

    print(json.dumps({"overall": overall}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
