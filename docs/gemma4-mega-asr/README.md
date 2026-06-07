# Gemma 4 鲁棒 ASR 项目方案

最后更新：2026-06-07

本目录用于记录一个基于 Gemma 4 12B 的独立鲁棒语音识别项目方案。

Mega-ASR 只作为参考项目。我们会学习它在鲁棒 ASR、声学退化数据、渐进式微调、质量路由和评测方面的方法，但最终系统必须由我们独立设计和实现，不在 Mega-ASR 的 Qwen3-ASR 专用代码上做魔改。

## 文档索引

- [开发进度](./00_progress.md)
- [项目原则](./00_project_principles.md)
- [架构方案](./01_architecture.md)
- [开发方案](./02_development_plan.md)
- [数据方案](./03_data_plan.md)
- [Colab 训练方案](./04_colab_training_plan.md)
- [测试方案](./05_testing_plan.md)
- [风险与决策](./06_risks_and_decisions.md)
- [文档验收与追踪矩阵](./07_document_acceptance.md)
- [执行路线图](./roadmap/README.md)
- [路线图总览](./roadmap/OVERVIEW.md)

## 当前范围

当前状态：文档设计与整体验收已完成，下一步进入 [01 独立项目骨架](./roadmap/01_project_scaffold.md)。

MVP 会按路线图依次完成：

1. 评估 Gemma 4 12B 原始 ASR 基线。
2. 构建小规模 clean/degraded 语音数据集。
3. 训练第一版 QLoRA ASR adapter。
4. 评估 WER/CER 和典型失败模式。
5. 根据结果决定是否扩大数据、训练时长和 router 复杂度。

最终目标是完成一个类似 Mega-ASR 的鲁棒 ASR 产品：具备鲁棒 ASR LoRA、音频质量 router、统一推理入口、数据增强管线、评测体系和发布文档。但实现方式必须独立于 Mega-ASR，基于 Gemma 4 12B 自行开发。

## 非目标

- 不把 Mega-ASR 作为最终代码库。
- 不依赖 Qwen3-ASR 专用 API。
- 不复制 Mega-ASR 的模块名、LoRA target 规则或推理 wrapper。
- 不在没有自有 benchmark 结果前声称达到 Mega-ASR 同等效果。

## 参考资料

- Mega-ASR 仓库：https://github.com/xzf-thu/Mega-ASR
- Mega-ASR 权重：https://huggingface.co/zhifeixie/Mega-ASR
- Gemma 4 12B 发布说明：https://blog.google/innovation-and-ai/technology/developers-tools/introducing-gemma-4-12b/
- Gemma 4 12B HF 模型：https://huggingface.co/google/gemma-4-12B
- Gemma 4 12B 指令模型：https://huggingface.co/google/gemma-4-12B-it
