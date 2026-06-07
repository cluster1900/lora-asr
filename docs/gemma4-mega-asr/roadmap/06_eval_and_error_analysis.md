# 06 评测与错误分析

最后更新：2026-06-07

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

## 需要实现的文件

- `evaluation/eval_wer.py`
- `evaluation/normalize_text.py`
- `evaluation/analyze_failures.py`
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

## 测试标准

- 10 条 mini eval 可正常计算。
- 空 reference 或空 prediction 有明确处理。
- 同一输入多次评测结果一致。
- 按 scenario 聚合结果与 overall 结果可核对。

## 验收标准

- base 与 LoRA 能用同一脚本评测。
- 输出 overall 和 scenario-level 指标。
- 输出至少 5 类失败样本列表。
- 所有指标文件可追溯到 model、adapter、manifest 和配置。

## 风险

- 文本归一化掩盖真实错误。缓解：保存 raw prediction 和 normalized prediction。
- WER/CER 对多语言混合处理不当。缓解：按 language 字段选择 metric。
- 只看 overall 掩盖场景退化。缓解：强制输出 scenario-level。

