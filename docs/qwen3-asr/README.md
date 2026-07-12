# Qwen3-ASR 鲁棒 ASR 文档

最后更新：2026-07-12

本项目固定使用 `Qwen/Qwen3-ASR-1.7B`，独立实现数据、训练、推理、评测和发布。
Mega-ASR 只作为方法与外部 baseline。

## 当前状态

历史 baseline、LoRA v1-v5 和评测闭环已经完成，但没有证明相对 BF16 base 的正式净收益。
新的“公开 200k + 官方 Trainer + 343-target BF16 LoRA”主线尚未实现，状态 0/3。

## 唯一主线

1. 160k robust + 20k English clean + 20k Chinese clean；validation 8k+1k+1k。
2. Qwen 官方 Trainer 薄适配，343-target LoRA，`lr=1e-6`，10+2 真 resume。
3. 一次 200k run：step 100 canary、50%/100% validation、固定 test、release adapter。

GPT-5.5 teacher、router、RL、自建增强、全量 difficulty scoring 和 target sweep 不进入第一轮。

## 阅读顺序

- [开发进度](./00_progress.md)：实际完成度和历史结果。
- [架构方案](./01_architecture.md)：当前模块边界和条件触发模块。
- [快速微调执行合同](./02_development_plan.md)：唯一文件、参数、步骤和结果分支。
- [数据方案](./03_data_plan.md)：公开数据 schema、配比、缓存和防泄漏。
- [Colab 方案](./04_colab_training_plan.md)：唯一 Notebook 与恢复方式。
- [测试方案](./05_testing_plan.md)：数据、Trainer、推理、评测和指标门槛。
- [风险与决策](./06_risks_and_decisions.md)：当前决策、风险和回滚。
- [文档与阶段验收](./07_document_acceptance.md)：阶段完成条件。
- [Mega-ASR 差距](./08_mega_asr_gap_analysis.md)：方法证据与目标口径。
- [训练策略](./09_training_finetune_strategy.md)：为什么采用当前配置。

## 历史资料

`roadmap/`、notebook 00-11、旧 v1-v6 配置和历史 checkpoint/output 用于复现与审计，
不再定义下一步。若历史文档与 `02_development_plan.md` 冲突，以后者为准。

## Teacher 接口边界

未来只有未标注真实音频或 transcript 冲突才使用 `TEACHER_API_KEY`、
`TEACHER_BASE_URL`、`TEACHER_MODEL=gpt-5.5`。实际 base URL 必须通过 capability probe；
teacher 不得覆盖 gold transcript，第一轮不实现。

## 官方参考

- Qwen3-ASR：https://github.com/QwenLM/Qwen3-ASR
- Qwen finetuning：https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning
- Voices-in-the-Wild-2M：https://huggingface.co/datasets/zhifeixie/Voices-in-the-Wild-2M
- Voices-in-the-Wild-Bench：https://huggingface.co/datasets/zhifeixie/Voices-in-the-Wild-Bench
- Mega-ASR：https://github.com/xzf-thu/Mega-ASR
- OpenAI Python SDK：https://github.com/openai/openai-python
