# Qwen3-ASR 鲁棒 ASR 项目方案

最后更新：2026-06-29

本目录用于记录一个基于 Qwen3-ASR-1.7B 的独立鲁棒语音识别项目方案。

Mega-ASR 只作为参考项目。我们会学习它在鲁棒 ASR、声学退化数据、渐进式微调、质量路由和评测方面的方法，但最终系统必须由我们独立设计和实现，不在 Mega-ASR 的上游工程代码上做魔改。

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

当前状态：`01 独立项目骨架` 已完成，`02 Baseline 评估` 已完成 MVP 150 hard profile baseline，`06 评测与错误分析` 已产出第一批分析文件，`05A LoRA 训练前探测`、`05B Unsloth 兼容性检查` 和 `05C Transformers + PEFT smoke training` 已完成。正式 LoRA v1 bootstrap 训练和 held-out 评测已完成，但 v1 没有改善 noise/reverb，当前进入 `05E LoRA MVP v2 ablation`。

已完成：

- 原 Mega-ASR 上游工程已隔离到 `references/mega-asr-upstream/`，不作为新工程运行时依赖。
- 新工程目录已建立：`configs/`、`data/`、`evaluation/`、`inference/`、`notebooks/`、`router/`、`scripts/`、`train/`。
- 已实现本地 smoke 音频生成脚本、Qwen3-ASR baseline 推理脚本、WER/CER 评测脚本。
- 已新增 `notebooks/01_baseline_colab.ipynb`，并在 Colab 中完成 clean/noise 各 1 条样本的 smoke baseline。
- 已新增 `scripts/create_mvp_eval_audio.py` 和 `notebooks/02_mvp_150_eval_colab.ipynb`，用于 clean/noise/reverb/far_field/dropout 各 30 条的 baseline MVP 评测。
- 已在 MVP 150 hard profile 上记录 Qwen3-ASR base 指标：clean WER 0.010438，degraded-only WER 约 0.602296。
- 已新增 `evaluation/analyze_errors.py` 并保存 baseline 错误分析输出。
- 已新增 `train/inspect_qwen3_asr_modules.py` 和 `notebooks/03_train_lora_colab.ipynb`，用于 LoRA 训练前模块探测。
- 已保存 `outputs/lora_probe/qwen3_asr_1_7b/` 探测输出，并确定第一版 smoke target：audio tower attention + speech projection。
- 已新增 `train/check_unsloth_qwen3_asr.py`，并确认 Unsloth 当前无法直接加载 `Qwen/Qwen3-ASR-1.7B`。
- 已完成 20 step PEFT smoke training，证明 99 个 audio tower target 可挂载、loss 非 NaN、adapter 可保存。
- 已新增 LoRA MVP bootstrap train/val 数据生成入口，正式训练不直接复用 MVP 150 held-out test。
- 已完成正式 LoRA MVP bootstrap 训练产物同步，`checkpoints/qwen3-asr-1.7b-lora-mvp/summary.json` 记录 `status=trained` 和 `steps=600`。
- 已完成 v1 held-out 评测，结论是不通过：overall 和 degraded-only 均变差，noise/reverb 均变差。

下一步：执行 [05 LoRA 训练 MVP](./roadmap/05_lora_training_mvp.md) 的 `05E`，跑 v2 attention-only、noise/reverb-only、长短句均衡采样短实验，并在固定 MVP 150 held-out test 上比较 base、v1 和 v2。

MVP 会按路线图继续完成：

1. 基于已完成的 Qwen3-ASR-1.7B baseline 和错误分析确定 LoRA 目标场景。
2. 基于探测结果确定第一版 LoRA target。
3. 使用独立 bootstrap train/val 训练第一版 QLoRA ASR adapter。
4. 评估 WER/CER、典型失败模式和 clean regression。
5. 根据结果决定是否进入 router 和扩大训练规模。

最终目标是完成一个类似 Mega-ASR 的鲁棒 ASR 产品：具备鲁棒 ASR LoRA、音频质量 router、统一推理入口、数据增强管线、评测体系和发布文档。但实现方式必须独立于 Mega-ASR，基于 Qwen3-ASR-1.7B 自行开发。

## 非目标

- 不把 Mega-ASR 作为最终代码库。
- baseline 推理依赖 Qwen3-ASR 官方 `qwen-asr` API。
- 不复制 Mega-ASR 的私有模块名、LoRA target 规则或推理 wrapper。
- 不在没有自有 benchmark 结果前声称达到 Mega-ASR 同等效果。

## 参考资料

- Mega-ASR 仓库：https://github.com/xzf-thu/Mega-ASR
- Mega-ASR 权重：https://huggingface.co/zhifeixie/Mega-ASR
- Qwen3-ASR-1.7B HF 模型：https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- Qwen3-ASR 官方工具包：https://github.com/QwenLM/Qwen3-ASR
