# 开发进度

最后更新：2026-07-12

## 当前状态

历史探索闭环已完成，新的快速正式微调主线尚未实现，当前完成度为 **0/3**。

快速主线固定为：

1. 准备公开 200k train、10k validation 和固定 test。
2. 实现 Qwen 官方 Trainer 薄适配、343-target BF16 LoRA 和 10+2 resume smoke。
3. 执行一次 200k 正式训练，经过 100-step canary、50%/100% validation 和固定 test。

唯一执行合同见 `02_development_plan.md`。

## 历史结果

已完成：

- Qwen3-ASR baseline 推理、JSONL prediction、WER/CER 和错误分析。
- MVP 150 合成 hard-profile test。
- Qwen3-ASR 模块探测、Unsloth 兼容性检查和 Transformers + PEFT smoke。
- LoRA v1-v5 训练与 held-out 评测。
- 4bit base recheck 与 v3-v5 checkpoint 对比。
- v6A 合成数据入口和 difficulty 脚本 smoke。

关键指标：

| 模型 | overall | noise | reverb | noise+reverb | clean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4bit base recheck | 0.550313 | 0.450939 | 0.544885 | 0.497912 | 0.008351 |
| v3 target-focus | 0.540292 | 0.413361 | 0.515658 | 0.464509 | 0.008351 |

v3 相对 4bit base 的 noise+reverb 改善 6.71%，overall 改善 1.82%。但历史非量化
baseline overall 为 0.483925，明显优于 v3，因此不能把量化损失误判为 LoRA 收益。

更重要的是：MVP 150 只有 30 条独立 TTS utterance；v3 对 150 行中的 128 行 edit 数完全
不变，仅 12 行改善、10 行变差。历史结果只说明 LoRA 通路可运行，没有证明可泛化的鲁棒
提升。

## 当前代码事实

可复用：

- 数据 JSONL、音频生成和校验基础。
- Qwen3-ASR base/LoRA 历史推理入口。
- WER/CER、scenario 聚合和错误分析。
- 模块快照与 343 个候选 Linear target 证据。

不能直接用于 200k：

- `train/train_qwen3_asr_lora.py` 强制 batch size 1，逐样本重复处理音频。
- YAML 中的 validation、epochs、部分 include_scenarios 没有形成可靠训练合同。
- 没有 scheduler、grad clip、训练期 validation 或 best-checkpoint。
- checkpoint 只保存 adapter/processor，不能恢复 optimizer、scheduler、RNG 和 global step。
- 当前 base/LoRA 推理实现重复，整批结束才写结果，不适合可恢复的 5k 评测。
- 当前 evaluator 不能计算 32-cell 指标，并会把空 reference 静默计为 0 error。
- 仓库没有项目级依赖锁、tests/ 或 CI。

当前 checkout 的 v3-v5 adapter 目录缺少 `adapter_model.safetensors`，不能从仓库直接复载
历史最佳模型；历史 prediction 和 metrics 仍可用于审计。

## 快速主线 Checklist

- [ ] `scripts/prepare_public_robust_manifests.py`
- [ ] `configs/data/public_robust_200k.yaml`
- [ ] `requirements-colab.txt`
- [ ] `train/train_qwen3_asr_lora_official.py`
- [ ] `configs/train/qwen3_asr_public_200k_broad_lora.yaml`
- [ ] `notebooks/12_fast_finetune_colab.ipynb`
- [ ] 统一可恢复推理入口
- [ ] 32-cell evaluator 与最小自动测试
- [ ] 200k/10k/5k manifest 和 validation report
- [ ] BF16 base prediction
- [ ] 10+2 resume smoke
- [ ] 100-step canary
- [ ] 50%/100% validation
- [ ] 固定 test 与 release adapter
- [ ] Mega-ASR 发布模型同 evaluator 外部 baseline

## 当前固定决策

- Train：160k robust + 20k English clean + 20k Chinese clean。
- Validation：8k robust + 1k English clean + 1k Chinese clean。
- Precision：BF16；正式效果不使用 4bit base 口径。
- LoRA：343 target，r=8，alpha=16，dropout=0.05，learning rate 1e-6。
- 训练：1 epoch，effective batch 64，100-step canary，正式候选只保留 50%/100%。
- GPT-5.5 teacher、router、RL、自建增强和全量 difficulty scoring 不进入第一轮。
- 5%-10% 改善时进入压缩 A2S，不直接扩 645,925，不做 target/LR sweep。

## 下一步

只实现步骤 1 的公开数据 CLI 和配置。数据 full validation 通过后再实现 Trainer；在
10+2 smoke 和 golden batch 通过前，不启动 200k 正式训练。

## 未完成风险

- 公开 robust 数据体积约 197.5 GB，错误的 Drive 小文件方案会比训练更慢。
- Voices-in-the-Wild-2M 缺少显式 language/speaker 字段，需要稳定派生与 source-group split。
- Joint broad LoRA 可能造成 clean regression 或 hallucination，必须依赖 20% clean 与 canary。
- 一次直接 SFT 不能保证达到 Mega-ASR 的 A2S+RL 完整效果。

未生成 BF16 正式结果、release adapter 和 Mega-ASR 同 evaluator 外部 baseline 前，阶段不得
标记完成，也不得声称达到 Mega-ASR。
