# 开发进度

最后更新：2026-08-05

## 当前状态

历史探索闭环已完成。新的快速主线已经完成 manifest 选择/curriculum、统一推理、双语评测和
单-adapter A2S runner 的最小代码；正式数据物化、Colab 10+2 GPU smoke 和训练结果尚未执行。

因此当前应区分：**实现 2/3，正式运行 0/3**。未跑 GPU 和固定测试前不能把代码完成误写成
模型效果完成。

快速主线固定为：

1. 完成 metadata probe，准备公开 200k train、10k validation、30k curriculum pool 和
   固定 test。
2. 实现 Qwen 官方 Trainer 薄适配、A2S 三阶段 target 切换和 10+2 resume smoke。
3. 执行一次 A2S 编排训练，经过阶段 512 canary、最终 validation 和固定 test。

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

历史入口不能直接用于 200k：

- `train/train_qwen3_asr_lora.py` 强制 batch size 1，逐样本重复处理音频。
- YAML 中的 validation、epochs、部分 include_scenarios 没有形成可靠训练合同。
- 没有 scheduler、grad clip、训练期 validation 或 best-checkpoint。
- checkpoint 只保存 adapter/processor，不能恢复 optimizer、scheduler、RNG 和 global step。
- 历史 base/LoRA 推理实现重复，整批结束才写结果；新统一入口已替代它们。
- 历史 evaluator 不能计算 32-cell 指标且会错误处理空 reference；新 evaluator 已修正。
- 当前仍没有 CI；本地已增加高风险合同的 stdlib 自动测试。

当前 checkout 的 v3-v5 adapter 目录缺少 `adapter_model.safetensors`，不能从仓库直接复载
历史最佳模型；历史 prediction 和 metrics 仍可用于审计。

## 快速主线 Checklist

- [x] `scripts/prepare_public_robust_manifests.py`
- [x] `configs/data/public_robust_200k.yaml`
- [x] `requirements-colab.txt`
- [x] `train/train_qwen3_asr_a2s.py`
- [x] `configs/train/qwen3_asr_public_200k_a2s.yaml`
- [ ] `notebooks/12_fast_finetune_colab.ipynb`
- [x] 统一可恢复推理入口
- [x] 32-cell evaluator 与最小自动测试
- [ ] metadata-only probe、200k/10k/5k manifest 和 validation report
- [ ] BF16 base validation/test prediction 与 30k curriculum scoring
- [ ] 10+2 resume 和 A2S target-switch smoke
- [ ] Phase I/II/III 512 canary
- [ ] 最终 10k validation
- [ ] 固定 test 与 release adapter
- [ ] Mega-ASR 发布模型同 evaluator 外部 baseline

## 当前固定决策

- Train：160k robust + 20k English clean + 20k Chinese clean。
- Validation：8k robust + 1k English clean + 1k Chinese clean。
- Precision：BF16；正式效果不使用 4bit base 口径。
- LoRA：单 adapter 预注入 343 target；Phase I 只启用 upper-4 audio+projection 27 个，
  Phase II 只启用 decoder 196 个，Phase III 启用全部 343 个。
- 训练：30k x 2 epoch -> 200k x 1 epoch -> 200k x 1 epoch，effective batch 128。
- 学习率：Phase I/II `1e-6`；Phase III audio/projection `5e-7`、decoder `1e-6`。
- Direct SFT 前置实验、teacher、router、RL、自建增强和全量 difficulty scoring 不进入第一轮。

## 下一步

补齐 pinned Hub 数据到四份 candidate JSONL/本地音频的可恢复 staging，并把它接入唯一 Notebook
12。随后依次运行 metadata probe、128-row manifest smoke、Trainer 10+2 GPU resume、golden
batch 和 target-switch smoke；全部通过后才启动正式 A2S。

## 未完成风险

- 公开 robust 数据体积约 197.5 GB，错误的 Drive 小文件方案会比训练更慢。
- 当前数据 CLI 从已物化 candidate JSONL 开始，Hub 下载/落盘 staging 与 Notebook 12 仍未实现；
  在补齐前还不是从干净 Colab 可一键执行的闭环。
- Voices-in-the-Wild-2M 缺少显式 language/speaker 字段，需要先用 metadata probe 验证语言
  规则、`name` source identity 与配额。
- A2S decoder/joint 阶段可能造成 clean regression 或 hallucination，必须依赖 20% clean 与
  阶段 canary。
- 200k A2S 仍小于论文训练规模且不含 RL，目标是快速接近 Mega-ASR-Base，不承诺完整
  Mega-ASR 绝对指标。

未生成 BF16 正式结果、release adapter 和 Mega-ASR 同 evaluator 外部 baseline 前，阶段不得
标记完成，也不得声称达到 Mega-ASR。
