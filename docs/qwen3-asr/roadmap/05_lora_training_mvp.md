# 05 LoRA 训练 MVP

最后更新：2026-06-15

## 背景

Baseline 和数据 MVP 完成后，需要训练第一版 Qwen3-ASR 鲁棒 ASR LoRA，验证参数高效微调是否能改善退化音频识别。

## 目标

在 Colab 上跑通 Qwen3-ASR-1.7B QLoRA/LoRA 训练，产出可加载、可评测的 adapter。

## 范围

本步骤只做第一版监督微调，不做 RL，不做大规模训练。

## 当前 Baseline 对照

MVP 150 hard profile 的 Qwen3-ASR base 结果：

- clean WER：0.010438。
- noise WER：0.336117。
- reverb WER：0.415449。
- dropout WER：0.759916。
- far_field WER：0.897704。
- degraded-only WER：约 0.602296。
- empty output rate：所有场景均为 0.0。

第一版 LoRA MVP 不应直接追求所有 hard degraded 场景都大幅改善。建议目标分层：

- 第一优化目标：noise、reverb。它们错误明显但没有完全崩溃，更适合验证 LoRA 是否能学到鲁棒性。
- 观察目标：dropout、far_field。它们当前 WER 很高，第一版只要求记录是否改善，不作为唯一成败标准。
- 硬门槛：clean regression。clean WER 已经约 1.04%，LoRA 后必须量化 clean 是否退化。

## 输入

- `train.jsonl`
- `val.jsonl`
- 训练配置。
- Qwen3-ASR base model。

## 输出

- LoRA adapter checkpoint。
- `training_config.json`
- trainer state。
- 训练日志。
- eval predictions。

## 需要实现的文件

- `train/train_qwen3_asr_lora.py`
- `train/collator.py`
- `train/lora_targets.py`
- `configs/train/qwen3_asr_lora_mvp.yaml`
- `notebooks/03_train_lora_colab.ipynb`

## 执行步骤

1. 完成 baseline 错误分析，确认第一版训练目标和失败模式。
2. 加载 Qwen3-ASR model。
3. 打印 `model.named_modules()`，保存模块名快照。
4. 定义第一版 LoRA target。
5. 实现音频 + 文本 collator。
6. 跑 5-20 step smoke test。
7. 保存 adapter 并测试加载。
8. 跑 MVP 训练。
9. 在 val/test 上评测。
10. 使用 `evaluation/analyze_errors.py` 对比 base 与 LoRA。
11. 更新训练文档、测试文档、进度文档。

## 初始配置

```yaml
model_id: Qwen/Qwen3-ASR-1.7B
quantization: 4bit
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 2e-5
epochs: 1
max_audio_seconds: 20
seed: 42
```

## 测试标准

- smoke test 能完成 5-20 step。
- loss 非 NaN。
- checkpoint 可保存。
- adapter 可重新加载并推理。
- 推理输出可进入 WER/CER 评测。

## 验收标准

- noise 或 reverb 至少一个场景 WER 相对 base 改善。
- degraded-only WER 有记录，作为综合参考。
- clean regression 有量化记录。
- 空输出率、重复输出率、幻觉式输出率有记录。
- 训练配置、数据 manifest、随机种子随结果保存。
- 进度文档记录训练结论。

## 风险

- Colab 显存不足。缓解：减小 max_audio_seconds、batch size 设为 1、使用 4bit/LoRA，必要时使用 A100。
- LoRA target 不合适。缓解：做 target ablation。
- 模型学会输出模板而非转写。缓解：清理训练目标，保存原始输出，并在评测中统计输出污染。
