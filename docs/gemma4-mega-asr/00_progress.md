# 开发进度

最后更新：2026-06-07

## 当前状态

当前阶段：文档验收完成，准备进入 `01 独立项目骨架`。

我们已经把 Mega-ASR 作为参考项目进行了初步分析，并决定设计一个独立的 Gemma 4 12B 鲁棒 ASR 训练路径。第一个里程碑是 Colab 友好的 MVP，包含我们自己的数据管线、训练代码、推理封装和评测工具。

## 已完成

- 阅读 Mega-ASR 项目结构。
- 确认 Mega-ASR 已发布的训练代码主要是 A2S-SFT；DG-WGPO RL 代码尚未发布。
- 确认 Mega-ASR 的推理方式：Transformers 后端通过 router 动态启停 LoRA；vLLM 后端会物化 LoRA，不做逐样本路由。
- 确认 Gemma 4 12B 支持原生音频输入。
- 创建 `docs/gemma4-mega-asr` 文档工作区。
- 明确最终项目必须独立开发，Mega-ASR 只作为方法参考。
- 创建执行路线图文档，覆盖文档治理、项目骨架、baseline、数据、增强、训练、评测、router 和规模化发布。
- 创建路线图总览，明确每一步的作用，以及最终目标是完成类似 Mega-ASR 的独立鲁棒 ASR 产品。
- 完成整体文档验收，修复开发方案与 roadmap 脱节问题，并新增产品能力追踪矩阵。
- 已将原 Mega-ASR 上游工程隔离到本地忽略目录 `references/mega-asr-upstream/`，根目录作为新工程主目录。
- 已创建新工程最小骨架：`configs/`、`data/`、`evaluation/`、`inference/`、`notebooks/`、`router/`、`scripts/`、`train/`。
- 已完成 `01 独立项目骨架` 验收：上游工程隔离完成，根目录无上游主入口残留，示例 JSONL/YAML 可解析，Markdown 链接检查通过。

## 进行中

- 准备进入 `02 Baseline 评估`。

## 下一批里程碑

1. 按 [02 Baseline 评估](./roadmap/02_baseline_eval.md) 实现 Gemma 4 ASR baseline。
2. 按 [03 数据 MVP](./roadmap/03_data_mvp.md) 构建小规模 JSONL 数据。
3. 按 [04 音频增强](./roadmap/04_audio_augmentation.md) 实现退化增强。
4. 按 [05 LoRA 训练 MVP](./roadmap/05_lora_training_mvp.md) 实现第一版 QLoRA。
5. 按 [06 评测与错误分析](./roadmap/06_eval_and_error_analysis.md) 跑通 baseline 与微调评测。
6. 按 [07 Router MVP](./roadmap/07_router_mvp.md) 实现 clean/degraded 路由。

## 待确认问题

- 首次训练使用哪种 Colab 资源：免费版、Pro、Pro+，还是外部 GPU runtime？
- MVP 优先语言：英文、中文，还是双语？
- 首个数据源使用 LibriSpeech、Common Voice、AISHELL，还是混合数据？
- MVP 是否只做短音频，还是尽早加入 30-120 秒长音频？

## 日志

### 2026-06-07

创建第一版规划文档。推荐第一版做小而完整的独立闭环：baseline、数据生成、LoRA SFT、评测，以及简单 router 决策。

补充执行路线图文档。后续开发应按 roadmap 顺序执行，每一步先满足文档、测试和验收要求，再进入下一步。

完成整体文档验收。当前文档体系已经可以支撑后续实现，下一步应执行 [01 独立项目骨架](./roadmap/01_project_scaffold.md)。

完成参考工程隔离。原 Mega-ASR 上游源码、脚本、资源和站点文档已移动到本地忽略目录 `references/mega-asr-upstream/`，根目录转为 Gemma 4 Robust ASR 新工程主路径。

创建新工程最小骨架和基础配置。根目录现在以 Gemma 4 Robust ASR 为主，参考工程仅位于 `references/mega-asr-upstream/`，且不会进入 git。

完成 `01 独立项目骨架` 验收：示例 JSONL 解析通过，YAML 配置解析通过，本地 Markdown 链接检查通过。下一步进入 baseline 评估。
