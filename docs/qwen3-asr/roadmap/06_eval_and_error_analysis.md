# 06 评测与错误分析

最后更新：2026-06-19

## 背景

ASR 训练不能只看少量样例。必须建立统一评测和错误分析流程，比较 base、LoRA 和后续 router 模式。

## 目标

实现可复现的 WER/CER 评测、scenario 聚合和失败样本分析。

## 范围

本步骤负责评测工具和分析报告，不训练模型。

## 输入

- 评测 JSONL。
- base prediction JSONL。
- LoRA prediction JSONL。
- 可选 router prediction JSONL。

## 输出

- overall metrics。
- scenario-level metrics。
- failure samples。
- clean regression report。
- run summary。
- error analysis summary。
- worst-case JSONL/CSV。
- `hallucination_like`、`repeat_like`、`too_short` 等启发式标签。

## 需要实现的文件

- `evaluation/eval_wer.py`
- `evaluation/normalize_text.py`
- `evaluation/analyze_errors.py`
- `configs/eval/default_eval.yaml`
- `notebooks/04_eval_colab.ipynb`

## 执行步骤

1. 实现英文 WER 和中文 CER。
2. 统一文本归一化。
3. 按 scenario 聚合指标。
4. 计算 empty output rate。
5. 计算 repeated output rate。
6. 计算 length ratio。
7. 输出最差样本、最佳改善样本和 clean regression 样本。
8. 保存 metrics JSON/CSV。
9. 更新测试文档和进度文档。

## 当前 Baseline 错误分析背景

`Qwen/Qwen3-ASR-1.7B` 已在 MVP 150 hard profile 上完成 baseline：

- clean WER：0.010438。
- noise WER：0.336117。
- reverb WER：0.415449。
- dropout WER：0.759916。
- far_field WER：0.897704。
- degraded-only WER：约 0.602296。
- 所有场景 empty output rate：0.0。

这说明下一步错误分析的重点不是空输出，而是：

- `hallucination_like`：预测与 reference 语义可能完全无关。
- `too_short`：退化音频导致只输出短片段。
- `repeat_like`：长音频或强退化下重复同一短语。
- `insertion_heavy`：预测比 reference 长很多，导致 WER 超过 1.0。
- clean regression candidates：clean 场景中少数非零错误样本。

## MVP 错误分析脚本设计

`evaluation/analyze_errors.py` 读取 `eval_wer.py` 产出的 scored JSONL，输出：

- `analysis_summary.json`：overall、scenario、scenario + text_length_bucket 汇总。
- `worst_cases.jsonl`：按 WER 和 num_edits 排序的最差样本。
- `worst_cases.csv`：便于 Colab/pandas/表格查看的最差样本。
- `flagged_cases.jsonl`：按启发式标签筛选出的错误样本。

脚本只做启发式分析，不替代人工判断。尤其是 hallucination-like 标签只能表示“高 WER 且长度不为空的疑似无关输出”，最终仍要抽样听音频和看 reference/prediction。

## 推荐命令

```bash
python3 evaluation/analyze_errors.py \
  --scored-jsonl outputs/baseline_mvp_150/predictions.qwen3_asr_base.mvp_150.scored.jsonl \
  --output-dir outputs/baseline_mvp_150/error_analysis \
  --top-k 30
```

## 已保存的 Baseline 输出

第一批受控 baseline 产物保存在 `outputs/baseline_mvp_150/`，用于后续 LoRA 训练前后的排查对照：

- `predictions.qwen3_asr_base.mvp_150.jsonl`
- `predictions.qwen3_asr_base.mvp_150.scored.jsonl`
- `metrics.qwen3_asr_base.mvp_150.json`
- `metrics_by_scenario.qwen3_asr_base.mvp_150.csv`
- `baseline_mvp_150.colab.jsonl`
- `predictions.oracle.mvp_150.jsonl`
- `predictions.oracle.mvp_150.scored.jsonl`
- `metrics.oracle.mvp_150.json`
- `metrics_by_scenario.oracle.mvp_150.csv`
- `error_analysis/analysis_summary.json`
- `error_analysis/worst_cases.csv`
- `error_analysis/flagged_cases.jsonl`
- `error_analysis/by_scenario.csv`
- `error_analysis/by_scenario_bucket.csv`

这些文件可以进入 git；其他临时输出、缓存、模型权重和 checkpoint 仍不应提交。

## 测试标准

- 10 条 mini eval 可正常计算。
- 空 reference 或空 prediction 有明确处理。
- 同一输入多次评测结果一致。
- 按 scenario 聚合结果与 overall 结果可核对。
- 错误分析脚本能在 oracle scored JSONL 和真实 scored JSONL 上运行。
- `worst_cases.csv` 至少包含 scenario、text_length_bucket、wer、answer、prediction。
- `flagged_cases.jsonl` 每条样本必须包含 `error_tags`。

## 验收标准

- base 与 LoRA 能用同一脚本评测。
- 输出 overall 和 scenario-level 指标。
- 输出至少 5 类失败样本列表。
- 所有指标文件可追溯到 model、adapter、manifest 和配置。
- baseline MVP 150 的 worst cases 已保存。
- LoRA 后能使用同一分析脚本比较 base 与 LoRA。

## 风险

- 文本归一化掩盖真实错误。缓解：保存 raw prediction 和 normalized prediction。
- WER/CER 对多语言混合处理不当。缓解：按 language 字段选择 metric。
- 只看 overall 掩盖场景退化。缓解：强制输出 scenario-level。
