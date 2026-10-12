# evaluation

## 目录职责

提供中文 `zh-cer-v1` CER 计算、scenario 聚合评测、输出健康度检查（空输出、重复输出、超长输出、截断、失败率）和阶段门禁裁决（clean/degraded gate、held-out preference gate）。WER 仅在配置明确指定并锁定版本的分词器时作为辅助指标计算。manifest SHA-256、归一化版本、模型/processor 或 `decode.eval` hash 不一致时，门禁判定为 `incomparable`。

## 当前文件清单

- `README.md`：本说明文档。评测脚本 `eval_cer.py` 和门禁脚本 `verify_gate.py` 将在 Phase 0 依据 `docs/qwen3-asr/05_testing_plan.md` 实现并登记。

## 主要输入输出

- **输入**：基准评估 JSONL manifest（含 gold `text`、`scenario`、`language`）、推理预测 JSONL（含 `sample_id`、`prediction_raw`、`prediction_normalized`、`error`）以及可选的父版本 metrics（用于 gate 比较）。
- **输出**：逐样本评分文件（编辑数、参考字符数、sample CER、raw/normalized）、聚合 `metrics.json`（micro/macro CER、WER、reward 和健康度）、按场景统计汇总 `by_scenario.csv`、失败样本日志以及带 schema/hash/父版本/step 的门禁判定 `gate.json`。`gate.json.status` 允许 `passed`、`failed`、`incomparable`、`insufficient_data` 或 `insufficient_eval_rows`。

## 使用入口

在 Phase 0 实现后提供：
```bash
# CER 评测
python3 evaluation/eval_cer.py --manifest manifests/val.jsonl --predictions runs/<run_id>/predictions/val.jsonl --output-dir runs/<run_id>/eval

# 门禁裁决
python3 evaluation/verify_gate.py --pilot-metrics runs/<run_id>/metrics.json --base-metrics releases/base/metrics.json --stage sft_pilot --output runs/<run_id>/gate.json
```

`micro_cer = sum(edit_count) / max(sum(reference_char_count), 1)`；`macro_cer` 是 sample CER 平均值。clean/degraded gate 使用 micro CER，scenario 结论要求该 cell 至少 50 行。评测只读 `decode.eval`，不读取 RL rollout 参数；test 仅在发布前评一次。缺预测、重复 sample、行数/manifest hash 不符或 score contract 不符必须失败或 `incomparable`，不能自动丢行后继续。

## 维护要求

指标定义、输出 schema、gate 阈值逻辑发生变更，或新增、重命名、删除文件时，必须在同一提交同步更新对应目录的 `README.md`、测试和评测文档。
