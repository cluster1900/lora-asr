# 测试方案

最后更新：2026-10-12

## 背景与范围

训练链路同时涉及中文数据、官方 Qwen3-ASR API、LoRA、DPO 离线参考 logprob、RL rollout 和 Colab + Drive 恢复。单看 loss 或一次成功启动不能证明链路可复现，因此测试必须覆盖数据、推理、评测、训练、持久化和阶段门禁。本文件规定测试夹具、命令、通过标准和失败样本分析方式；数值阈值以 [`04_training_plan.md`](04_training_plan.md) 中启动前冻结的配置为准。

本阶段不下载真实模型或公开数据运行本地单元测试。模型加载、A100 显存、Drive 读回和 5–20 step smoke 属于 Colab GPU/preflight；本地测试使用合成音频路径、合成中文文本、mock logits 和临时目录。

## 测试矩阵

| 层级 | 最小夹具/输入 | 固定命令 | 通过标准 | 产物 |
| --- | --- | --- | --- | --- |
| 目录合同 | 当前受管目录与 README | `python3 -m unittest tests/test_directory_readmes.py -v` | 清单无漏项、章节齐全 | 测试输出 |
| 数据合同 | 含五类 scenario、同源冲突、坏音频的 JSONL | `python3 scripts/manifest_validate.py --manifest <path> --check-audio` | schema、路径、采样率、split、source 无泄漏；坏行有明确错误 | `manifest.json`、`sha256.json`、审计报告 |
| 推理 | 1 条 clean、1 条 degraded、1 条失败样本 | `python3 inference/run_inference.py ...` | 成功行含 raw/normalized；失败行含 `error`，批处理继续 | predictions JSONL |
| 评测 | 固定预测和 gold | `python3 evaluation/eval_cer.py ...` | CER 可重算，按 scenario 聚合，健康度单独统计 | `metrics.json`、逐样本评分 |
| 配置/preflight | 最小 YAML、未知字段、错误 dtype/world size | `python3 train/<stage>.py --config <path> ...` | 非法配置在建 optimizer 前失败；resolved config/hash 已写出 | `preflight` 事件 |
| SFT smoke | 5–20 step，含 clean+degraded | `python3 train/train_sft.py ...` | loss/grad/lr 有限；checkpoint 完整；可恢复并再训 1 step | checkpoint、events、metrics |
| DPO | chosen/rejected、tie、source 交集、离线 logprob | `python3 scripts/build_dpo_pairs.py ...` | pair schema、审计、reference hash 和 held-out preference 正确 | pair manifest、audit、gate |
| RL | 固定 G 候选、空/重复/同 reward 组、reference | `python3 train/train_rl.py ...` | reward 分量、advantage、KL、zero group、reference freeze 合同正确 | rollout、metrics、gate |
| Drive/恢复 | 截断 checkpoint、坏 checksum、runtime 中断 | `python3 scripts/drive_sync.py ...` | `.incoming` 不可恢复；`latest.json` 只指向完整目录；global step 不重复 | sync events、latest |

## 统一指标与门禁

中文主指标是 `zh-cer-v1` CER；WER 只有配置指定分词器时计算，否则为 `null` 并记录原因。每个结果保存 `prediction_raw`、`prediction_normalized`、编辑数、参考字符数、样本 CER、空输出、重复输出、超长输出和推理错误。`micro_cer` 为编辑数总和除以参考字符数总和，`macro_cer` 为样本 CER 平均；clean/degraded 和场景改善门禁使用 `micro_cer`。

比较结果前必须相同：manifest SHA-256、`normalization_version`、`decode.eval` hash、模型 revision、音频前端和评测 schema。任一不同，`gate.json.status` 必须为 `incomparable`，不能用人工解释覆盖。改变 `decode.rollout` 不得改变同一预测在 eval 下的 CER。

阶段通过条件如下，具体数字只能从冻结配置读取：

- SFT：至少一个不少于 50 条的 degraded scenario 相对 base 严格改善；clean 回退、空输出率、重复率和有效输出率满足配置；smoke 保存和恢复通过。
- DPO：held-out preference accuracy 严格高于 0.5 且达到 `min_preference_accuracy`；相对 SFT 的 clean/degraded CER gate 通过；pair 审计和 reference logprob hash 通过。
- RL：相对本 run step 0 的 greedy reward 达到配置最小提升；`raw_kl` 不超过上限；reference checksum 未变；相对 DPO 的 clean/degraded 和健康度 gate 通过。

test 只在发布前以同一 `decode.eval` 评测一次，不参与训练、pair 构造、rollout 或 checkpoint 选择。没有同一 manifest、同一 evaluator 和父版本的 baseline，阶段不得标记完成。

## 训练恢复与异常验收

每个训练阶段至少做一次“保存 → 模拟 runtime 中断 → 重新启动 → `--resume` → 再训 1 step”。测试必须比较 `global_step`、optimizer/scheduler 参数个数、CPU/CUDA RNG 是否可加载、LoRA 目标配置、manifest/config/eval hash 和 reference checksum。恢复不是从任意目录寻找最近文件，而是先读取 `latest.json` 并核对 `checkpoint_manifest.json`。

以下情况必须停止当前 run、写 `events.jsonl` 和 `status.json`，保留最后一个完整 checkpoint，且不生成 release：OOM、非有限 loss/grad、坏音频、schema 错误、Drive checksum 不一致、reference checksum 变化、恢复 hash 不匹配、clean 或健康度停止条件触发。调整配置后新建 `run_id`；不得覆盖失败 run，也不得把失败 run 的指标拼接到新 run 的 step 0。

## 失败样本分析

批量推理和评测失败不应中断整批。分析脚本按 `error_type`、`scenario`、`sample_id` 和 `source_utterance_id` 汇总，保留原始音频路径、模型/adapter、decode hash、错误文本、raw/normalized prediction 和 traceback 摘要。至少抽查每类错误 20 条，区分数据不可读、模型生成空/重复、超长幻觉、归一化问题和 evaluator bug；修复后用同一 sample ID 重放并把前后指标写入 `run_report.md`。

## 本地与 Colab 命令

实现后本地先运行：

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile scripts/*.py train/*.py inference/*.py evaluation/*.py
git diff --check
```

Colab 再运行固定 manifest 的 preflight、clean/degraded 各一条推理和 5–20 step smoke。正式阶段必须额外保存 `environment.json`、`resolved_config.yaml`、`manifest_sha256.json`、原始 predictions、`metrics.json`、`gate.json` 和 `run_report.md`。目录文件清单发生变化时，先更新对应 README，再重跑目录合同测试。

## 验收与影响

完成一个阶段需要同时满足：测试矩阵中该阶段的所有项目通过；smoke checkpoint 能保存、加载、继续；val gate 写出 `passed`；test 结果、baseline 对比、原始预测和归一化预测已归档；结论写入 `00_progress.md`。失败测试只能把阶段保持为 `in_progress` 或 `blocked`，不能以文档结论替代实际产物。

本方案会约束后续代码的 CLI、日志字段、checkpoint 文件名和指标 schema。它不改变已经发布的权重，也不把当前仓库尚未实现的 runner 误标为可执行；Phase 0 完成前，命令中的 runner 仅是待实现接口。
