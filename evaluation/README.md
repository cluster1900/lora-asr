# evaluation 目录说明

## 目录职责

本目录负责把推理产生的逐条 JSONL 预测转换成可复现的 ASR 指标与失败样本报告。评测只消费结果，
不加载模型、不执行训练，也不修改输入预测文件。

英文与中文必须分开计量：英文使用 WER，中文使用 CER。包含两种语言的集合不生成伪造的混合
WER/CER，而是通过 `by_language` 和 language macro 汇总。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `eval_wer.py` | 评测入口。执行文本归一化、WER/CER、失败输出检测、scenario 聚合和固定 32-cell Bench 聚合。 |
| `verify_gate.py` | 门禁判定与机器可读 gate.json 生成入口。对比 Base 与当前阶段模型评估指标，校验退化场景改善数、Clean 回退、有效输出率和失败率变化，记录输入 manifest 与预测文件 SHA-256 哈希。 |
| `eval_checkpoint_series.py` | 检查点序列自动化横向评测工具。遍历指定 step 检查点（如 step 100/150/200/250/300），批量运行推理与 WER/CER 评估，输出横向对比矩阵与 Pareto 最优检查点推荐。 |
| `paired_significance.py` | 配对显著性工具。读取两份 `scored.jsonl`（基线、候选），要求每行有 `sample_id`、`error_rate`、`prediction_normalized`、`scenario`、`language`、`condition_group`，并按 `sample_id` 内连接且逐样本元数据一致；输出逐样本 `error_rate` 差（候选减基线，负为改善）的均值与固定 seed 的 bootstrap 95% 区间、预测文本变化数、变好/变差计数和双侧精确符号检验 p 值，并按 `condition_group`、`language\|condition_group` 与 `language\|scenario` 分组。`--baseline`/`--candidate` 可各给多个文件（按 `sample_id` 合并，跨文件重复报错）；`--gate` 输出 v31 统计验收段（degraded 显著改善且 clean 非劣）。只读，不加载模型。 |
| `README.md` | 说明本目录边界、文件职责、输入输出和维护要求。 |

`__pycache__/` 是 Python 自动生成的本地缓存，已被 Git 忽略，不是项目产物，可随时删除。

## 输入

`eval_wer.py` 接收 inference runner 生成的 prediction JSONL；当前 runner 待按 V100 方案重新实现。

每行至少需要：

- `text`：gold transcript，不能为空（优先读取 `text`，向下兼容历史 `answer` 字段）。
- `prediction`：模型预测文本；推理失败时可为空。
- `language`：只接受 `en` 或 `zh`，用于选择 WER/CER。

推荐保留 `sample_id`、`scenario`、`condition_group`（支持 `clean` 与 `degraded`，向下兼容历史 `atomic` 与 `compound`）、`audio_origin`、`error` 等字段，以便生成
分场景指标和失败统计。标记了 `error` 的行按空预测计分，不能因为推理失败而获得虚假的低错误率。

## 输出

指定 `--output-dir` 后生成：

| 产物 | 内容 |
| --- | --- |
| `scored.jsonl` | 原始/归一化文本、metric、edit 数、错误率、长度比和失败标签。 |
| `metrics.json` | overall、by-language、by-scenario、by-origin、by-condition 和 32-cell 汇总。 |
| `by_language.csv` | 英文 WER 与中文 CER。 |
| `by_scenario.csv` | 按语言和场景拆分的指标。 |
| `by_cell.csv` | language x real/synthetic x scenario 的固定 Bench cell。 |

关键失败指标包括 inference error、空输出、重复输出、过长输出和幻觉式输出。训练阶段 canary
直接读取 `metrics.json` 的 robust/clean language macro 与失败率。

## 使用方式

```bash
# 1. 运行 WER/CER 评测
python evaluation/eval_wer.py \
  --predictions-jsonl /path/to/predictions.jsonl \
  --output-dir /path/to/evaluation

# 2. 运行门禁判定并生成机器可读 gate.json（含退化场景改善、Clean 回退、鲁棒宏平均回退上限与空输出率硬拦截）
# 默认 profile=release；pilot_feasibility 只确认小闭环可运行，不具备发布资格。
# SFT Pilot 门禁判定
python evaluation/verify_gate.py \
  --base-metrics /path/to/base/metrics.json \
  --pilot-metrics /path/to/pilot/metrics.json \
  --base-predictions /path/to/base/predictions.jsonl \
  --pilot-predictions /path/to/pilot/predictions.jsonl \
  --manifest /path/to/validation.jsonl \
  --stage sft_pilot \
  --max-robust-regression 0.005 \
  --max-empty-rate 0.002 \
  --output /path/to/gate.json

# DPO Pilot 门禁判定（强校验 held-out preference accuracy >= 0.55，支持 --preference-accuracy 或 --dpo-loss-log 结合可选 --dpo-step 自动提取；未提供时门禁必须判为 FAILED；包含 SFT 预测与评测 Provenance、--dpo-val-manifest 验证集 Provenance、累积 Clean 回退检查，且 Robust Macro 相对 SFT 零恶化 <= 0.0）
python evaluation/verify_gate.py \
  --base-metrics /data/mega-asr/runs/eval_validation_base/predictions_eval/metrics.json \
  --base-predictions /data/mega-asr/runs/eval_validation_base/predictions.jsonl \
  --sft-metrics /data/mega-asr/runs/sft_pilot_controlled/eval_series/step_50/eval/metrics.json \
  --sft-predictions /data/mega-asr/runs/sft_pilot_controlled/eval_series/step_50/predictions.jsonl \
  --pilot-metrics /data/mega-asr/runs/dpo_pilot_v2/predictions_eval/metrics.json \
  --pilot-predictions /data/mega-asr/runs/dpo_pilot_v2/predictions.jsonl \
  --manifest /data/mega-asr/manifests/validation.jsonl \
  --dpo-val-manifest /data/mega-asr/manifests/val_dpo_pairs.jsonl \
  --dpo-loss-log /data/mega-asr/runs/dpo_pilot_v2/loss_log.jsonl \
  --dpo-step 80 \
  --stage dpo_pilot \
  --max-robust-regression 0.0 \
  --max-empty-rate 0.002 \
  --output /data/mega-asr/runs/dpo_pilot_v2/gate.json

# RL Pilot 门禁判定（强校验 held-out reward 提升 >= 0.002，严格基于 Step 0 基线与目标 Step 的统一 val_eval_scope == "Full Held-out" 同口径差值计算；缺失 Step 0 或口径不一致判为 FAILED；强校验 zero-variance group 比例 <= 75%；包含 DPO 预测与评测 Provenance、--rl-val-manifest 验证集 Provenance、Base 累积 Clean 回退 <= 0.025、DPO Clean 回退 <= 0.02，且 Robust Macro 相对 DPO 零恶化 <= 0.0）
python evaluation/verify_gate.py \
  --base-metrics /data/mega-asr/runs/sft_pilot_controlled/merged_base_eval/metrics.json \
  --base-predictions /data/mega-asr/runs/sft_pilot_controlled/merged_base_eval/predictions.jsonl \
  --dpo-metrics /data/mega-asr/runs/dpo_pilot_v2/metrics.json \
  --dpo-predictions /data/mega-asr/runs/dpo_pilot_v2/predictions.jsonl \
  --pilot-metrics /data/mega-asr/runs/rl_pilot/evaluation/metrics.json \
  --pilot-predictions /data/mega-asr/runs/rl_pilot/predictions.jsonl \
  --manifest /data/mega-asr/manifests/validation.jsonl \
  --rl-val-manifest /data/mega-asr/manifests/rl_val_pool.jsonl \
  --rl-loss-log /data/mega-asr/runs/rl_pilot/loss_log.jsonl \
  --stage rl_pilot \
  --max-robust-regression 0.0 \
  --max-empty-rate 0.002 \
  --output /data/mega-asr/runs/rl_pilot/gate.json

# RL Pilot 可行性门禁：允许 reward 不下降、Robust 最多回退 0.001；
# 输出必须使用独立文件，不能覆盖 release gate.json。
python evaluation/verify_gate.py \
  --base-metrics /data/mega-asr/runs/sft_pilot_controlled/merged_base_eval/metrics.json \
  --dpo-metrics /data/mega-asr/runs/dpo_pilot_v2/predictions_eval/metrics.json \
  --pilot-metrics /data/mega-asr/runs/rl_pilot_v27/predictions_step_4_eval/metrics.json \
  --manifest /data/mega-asr/manifests/validation.jsonl \
  --rl-loss-log /data/mega-asr/runs/rl_pilot_v27/loss_log.jsonl \
  --stage rl_pilot \
  --gate-profile pilot_feasibility \
  --output /data/mega-asr/runs/rl_pilot_v27/pilot_gate.json

# 3. 运行多 Checkpoint 序列自动化横向评测与 Pareto 优选
python evaluation/eval_checkpoint_series.py \
  --run-dir /data/mega-asr/runs/sft_pilot_controlled \
  --steps 50,100,150,200,250,300 \
  --manifest /data/mega-asr/manifests/validation.jsonl \
  --base-metrics /data/mega-asr/runs/eval_validation_base/predictions_eval/metrics.json \
  --output-dir /data/mega-asr/runs/sft_pilot_controlled/eval_series
```

```bash
# 配对显著性：比较两份 eval_wer.py 产出的 scored.jsonl（退出码 2 表示输入不匹配或损坏）
python evaluation/paired_significance.py \
  --baseline /data/mega-asr/runs/dpo_pilot_v2/predictions_eval/scored.jsonl \
  --candidate /data/mega-asr/runs/rl_pilot_v30a/predictions_step_4_eval/scored.jsonl \
  --output /data/mega-asr/runs/rl_pilot_v30a/paired_significance_step_4.json
```

输出 `verdict` 为 `significant_improvement`（区间上界 < 0）、`significant_regression`（下界 > 0）或 `indistinguishable_from_noise`。

```bash
# v31 判定（docs/qwen3-asr/29_rl_v31_scale_design.md）：每边可给多个 scored.jsonl，按 sample_id 合并
# （跨文件重复 id 退出 2）；--gate 写入 gate 段：degraded 区间上界 < 0 且 clean 区间上界 <= --clean-margin（默认 0.002）
python evaluation/paired_significance.py \
  --baseline /data/mega-asr/runs/dpo_pilot_v2/predictions_eval/scored.jsonl \
             /data/mega-asr/runs/eval_dpo_champion_verdict_extra/predictions_eval/scored.jsonl \
  --candidate /data/mega-asr/runs/rl_scale_v31/predictions_step_64_eval/scored.jsonl \
              /data/mega-asr/runs/rl_scale_v31/predictions_step_64_extra_eval/scored.jsonl \
  --gate --output /data/mega-asr/runs/rl_scale_v31/paired_verdict_step_64.json
```

报告字段：`overall`、`by_condition_group`、`by_language_condition`（如 `en|degraded`）、`by_language_scenario`、`inputs`，以及 `--gate` 时的 `gate`（`status`、`failures`、`clean_margin`）。分语言结果只报告，不是门禁条件。

对应测试：`tests/test_eval_wer.py`、`tests/test_verify_gate.py`、`tests/test_eval_checkpoint_series.py`、`tests/test_paired_significance.py`。

## 维护要求

新增、重命名或删除本目录文件时，必须同步更新“文件清单”。修改指标定义、输入字段、输出产物或
CLI 参数时，必须在同一提交更新本 README 和对应测试；不得把中英文错误率合并成一个 WER/CER。

## 2026-10-04 当前 v31 修订

`paired_significance.py` 新增 `--tail-gate`（须同时 --gate）与 `--require-max-new-tokens`：要求每行合同一致、错误率有限且非负；按语言×场景报告输出异常计数，任一严重异常（错误率 ≥2 且 too_long/hallucination_like）相对 DPO 恶化即失败。保留原始分数，不移除离群样本。
