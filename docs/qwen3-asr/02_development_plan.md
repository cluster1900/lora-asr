# 开发方案

最后更新：2026-06-11

## 目标

创建一个独立的、Colab 优先的鲁棒 ASR 项目，用于训练、评估和使用 Qwen3-ASR-1.7B 的 ASR LoRA adapter，并最终形成类似 Mega-ASR 能力形态的产品雏形。

Mega-ASR 只作为训练思想和评测框架参考。下面的交付物都应由我们自己实现；baseline 可以依赖 Qwen3-ASR 官方 `qwen-asr` API，但不得依赖 Mega-ASR 上游工程的私有 wrapper、训练入口或 LoRA target 规则。

## 开发方式

开发必须按 roadmap 执行：

1. 先完成对应文档。
2. 再实现代码或 notebook。
3. 再运行测试。
4. 再更新进度和实验结果。
5. 最后判断是否达到验收标准。

如果某一步执行中发现设计不合理，先修正文档，再继续实现。

## 步骤总览

| 步骤 | 名称 | 主要交付物 | 开发重点 | 验收重点 |
| --- | --- | --- | --- | --- |
| 00 | 文档治理 | `AGENTS.md`、roadmap、进度文档 | 建立文档先行规则 | 所有步骤有测试和验收标准 |
| 01 | 独立项目骨架 | `configs/`、`scripts/`、`inference/`、`evaluation/`、`train/`、`router/`、`notebooks/` | 与 Mega-ASR 解耦 | 目录职责清楚，示例配置可读 |
| 02 | Baseline 评估 | baseline notebook、base inference、WER/CER 脚本 | 测出 Qwen3-ASR 原始 ASR 能力 | baseline 指标可复现 |
| 03 | 数据 MVP | train/val/test JSONL、数据统计 | 构建可复现数据 manifest | split 无泄漏，scenario 标签齐全 |
| 04 | 音频增强 | 增强脚本、增强配置、metadata | 生成退化音频 | 至少 5 类退化可生成并质检 |
| 05 | LoRA 训练 MVP | QLoRA 训练脚本、collator、adapter checkpoint | 跑通第一版训练 | adapter 可加载，至少一个 degraded 场景提升 |
| 06 | 评测与错误分析 | 统一评测脚本、错误分析报告 | 比较 base/LoRA/router | scenario-level 指标和失败样本齐全 |
| 07 | Router MVP | router 训练、推理、阈值配置 | 控制是否启用 LoRA | router 模式优于 always-base |
| 08 | 规模化与发布 | release candidate、benchmark、model card | 扩数据、长训、发布准备 | 可复现产品候选 |

详细执行标准见 [执行路线图](./roadmap/README.md)。

## 代码目录规划

```text
configs/
  baseline/
  data/
  train/
  eval/
data/
  jsonl/
evaluation/
inference/
notebooks/
router/
scripts/
train/
```

## 功能开发约定

### 配置

- 所有脚本必须能从 YAML/JSON 配置读取关键参数。
- 配置必须记录模型、数据 manifest、输出目录、随机种子。
- 训练配置必须随 checkpoint 保存。

### 数据

- 训练和评测都以 JSONL manifest 为入口。
- 数据路径应支持 Colab Google Drive 根目录变量。
- 每条样本必须保留 `scenario`，用于评测聚合。

### 推理

- 推理入口必须支持单条音频和 JSONL 批量推理。
- 输出必须可进入评测脚本。
- 推理失败要记录错误，不中断整批任务。

### 训练

- 所有训练先跑 smoke test。
- checkpoint 必须可保存、加载和继续训练。
- 训练结果必须与 baseline 对比。

### 评测

- 所有模型结果都使用同一评测脚本。
- 必须输出 overall 和 scenario-level WER/CER。
- 必须保存失败样本，不能只给一个总分。

### Router

- Router 必须报告分类指标和阈值来源。
- Router 输出必须写入推理结果，便于复盘。
- Router 失败时必须有 fallback 策略。

## 阶段推进规则

进入下一步骤前，必须满足：

- 当前步骤文档完整。
- 当前步骤测试通过。
- 当前步骤验收标准满足。
- `00_progress.md` 已更新。
- 如果有新风险或决策，`06_risks_and_decisions.md` 已更新。

## 不做事项

- 不在 `src/MegaASR` 上继续堆 Qwen3-ASR 实现。
- 不把 Mega-ASR 的私有 Qwen3-ASR wrapper 改造成本项目主实现。
- 不绕过 baseline 直接训练。
- 不在没有固定测试集的情况下判断模型好坏。
- 不在没有自有评测结果前宣称达到或超过 Mega-ASR。
