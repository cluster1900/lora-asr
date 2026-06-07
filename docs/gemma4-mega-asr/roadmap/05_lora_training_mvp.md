# 05 LoRA 训练 MVP

最后更新：2026-06-07

## 背景

Baseline 和数据 MVP 完成后，需要训练第一版 Gemma 4 鲁棒 ASR LoRA，验证参数高效微调是否能改善退化音频识别。

## 目标

在 Colab 上跑通 Gemma 4 12B QLoRA/LoRA 训练，产出可加载、可评测的 adapter。

## 范围

本步骤只做第一版监督微调，不做 RL，不做大规模训练。

## 输入

- `train.jsonl`
- `val.jsonl`
- 训练配置。
- Gemma 4 base model。

## 输出

- LoRA adapter checkpoint。
- `training_config.json`
- trainer state。
- 训练日志。
- eval predictions。

## 需要实现的文件

- `train/train_gemma_lora.py`
- `train/collator.py`
- `train/lora_targets.py`
- `configs/train/gemma4_lora_mvp.yaml`
- `notebooks/03_train_lora_colab.ipynb`

## 执行步骤

1. 加载 Gemma 4 processor 和 model。
2. 打印 `model.named_modules()`，保存模块名快照。
3. 定义第一版 LoRA target。
4. 实现音频 + 文本 collator。
5. 跑 5-20 step smoke test。
6. 保存 adapter 并测试加载。
7. 跑 MVP 训练。
8. 在 val/test 上评测。
9. 更新训练文档、测试文档、进度文档。

## 初始配置

```yaml
model_id: google/gemma-4-12B-it
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

- 至少一个 degraded 场景 WER 相对 base 改善。
- clean regression 有量化记录。
- 空输出率、重复输出率、幻觉式输出率有记录。
- 训练配置、数据 manifest、随机种子随结果保存。
- 进度文档记录训练结论。

## 风险

- 12B QLoRA 显存不足。缓解：减小 max_audio_seconds、batch size 设为 1、使用 A100。
- LoRA target 不合适。缓解：做 target ablation。
- 模型学会输出模板而非转写。缓解：严格 prompt、清理训练目标、评测输出污染。

