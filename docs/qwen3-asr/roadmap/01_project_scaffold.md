# 01 独立项目骨架

最后更新：2026-06-11

## 背景

当前仓库来源于 Mega-ASR，但我们的最终项目必须独立开发。第一步需要建立与 Mega-ASR 解耦的目录结构，后续所有 Qwen3-ASR 相关代码都放在新结构中。

原 Mega-ASR 工程应被隔离到本地忽略目录 `references/mega-asr-upstream/`，根目录只保留新工程内容。`references/` 不进入 git。

## 目标

创建可维护的独立项目骨架，让数据、训练、推理、评测、router、notebook 和配置各自有清晰位置。

## 范围

本步骤只创建目录、占位 README、基础配置和示例 manifest，不实现真实训练。若根目录仍存在 Mega-ASR 上游源码、脚本、资源或示例音频，本步骤必须先完成参考工程隔离。

## 建议目录

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
references/
  mega-asr-upstream/
```

## 文件规划

- `configs/baseline/qwen3_asr_baseline.yaml`
- `configs/data/mvp_dataset.yaml`
- `configs/train/qwen3_asr_lora_mvp.yaml`
- `configs/eval/default_eval.yaml`
- `data/jsonl/README.md`
- `data/jsonl/baseline_smoke.example.jsonl`
- `notebooks/README.md`
- `scripts/README.md`
- `inference/README.md`
- `evaluation/README.md`
- `train/README.md`
- `router/README.md`

## 执行步骤

1. 将原 Mega-ASR 上游工程隔离到 `references/mega-asr-upstream/`。
2. 确认根目录不再散放 Mega-ASR 的 `src/MegaASR`、`scripts`、`assets`、`examples` 等实现路径。
3. 新建独立目录结构。
4. 给每个目录写简短 README，说明职责和禁止事项。
5. 写 baseline smoke JSONL 示例，不放真实本地私有路径。
6. 写基础 YAML 配置，先记录字段，不要求完整可运行。
7. 更新 `00_progress.md`。

## 输入

- 当前文档规范。
- 架构方案。

## 输出

- 独立项目目录。
- 示例配置。
- 示例 JSONL。

## 测试标准

- `find` 能列出所有规划目录。
- 原 Mega-ASR 上游文件只存在于 `references/mega-asr-upstream/`。
- `git status --ignored references` 显示 `references/` 被忽略。
- `git diff --cached --name-only` 中没有 `references/` 路径。
- 根目录不存在 `src/MegaASR`、原 `scripts/inference.sh`、原 `infer.py` 等上游主入口。
- 示例 JSONL 每行是合法 JSON。
- 配置文件能被 YAML parser 读取。
- 新目录 README 说明清楚该目录职责。

## 验收标准

- 后续 baseline、数据、训练、评测代码都有明确落点。
- 项目骨架中没有依赖 Mega-ASR 上游工程的实现命名；Qwen3-ASR 只作为本项目当前基础模型命名出现。
- 参考工程隔离路径固定为 `references/mega-asr-upstream/`，且 `references/` 被 git 忽略。
- 进度文档已记录完成情况。

## 风险

- 目录过早复杂化。控制方式：只建 MVP 必需目录，不建空的大型框架。
- 与原 Mega-ASR 文件混淆。控制方式：上游工程只放在 `references/mega-asr-upstream/`，新代码不得进入该目录。
