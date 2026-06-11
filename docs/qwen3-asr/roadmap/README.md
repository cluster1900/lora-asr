# 执行路线图

最后更新：2026-06-07

本目录把项目后续工作拆成可执行步骤。每一步都必须先完成对应文档，再进入代码或 notebook 实现。

先阅读：[路线图总览](./OVERVIEW.md)

## 路线图总览

| 步骤 | 文档 | 目标 | 依赖 | 完成标志 |
| --- | --- | --- | --- | --- |
| 00 | [文档治理](./00_document_governance.md) | 建立文档先行和功能维度规范 | 无 | 文档结构和更新规则明确 |
| 01 | [独立项目骨架](./01_project_scaffold.md) | 创建与 Mega-ASR 解耦的目录和配置 | 00 | 基础目录、配置、示例 manifest 就绪 |
| 02 | [Baseline 评估](./02_baseline_eval.md) | 测出 Qwen3-ASR 原始 ASR 能力 | 01 | baseline JSONL 和 WER/CER 指标产出 |
| 03 | [数据 MVP](./03_data_mvp.md) | 构建首批 clean/degraded 数据 | 01, 02 | train/val/test manifest 可复现 |
| 04 | [音频增强](./04_audio_augmentation.md) | 实现可控声学退化生成 | 03 | 至少 5 类退化可生成并质检 |
| 05 | [LoRA 训练 MVP](./05_lora_training_mvp.md) | 跑通第一版 Qwen3-ASR QLoRA | 02, 03, 04 | adapter 可保存、加载、评测 |
| 06 | [评测与错误分析](./06_eval_and_error_analysis.md) | 建立统一评测和失败样本分析 | 02, 05 | 指标按 scenario 聚合 |
| 07 | [Router MVP](./07_router_mvp.md) | 判断 clean/degraded 并控制 LoRA | 06 | router 模式优于 always-base |
| 08 | [规模化与发布](./08_scale_up_and_release.md) | 扩大数据、训练和发布流程 | 05, 06, 07 | 可复现 release candidate |

## 执行原则

- 严格按步骤推进，但允许在不破坏依赖的情况下并行准备数据和评测脚本。
- 每一步开始前，必须确认对应文档的测试标准和验收标准已经明确。
- 每一步完成后，必须更新 `../00_progress.md`。
- 如果实际执行中发现文档不适用，先修正文档，再继续实现。
