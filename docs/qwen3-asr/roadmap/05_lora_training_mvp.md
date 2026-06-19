# 05 LoRA 训练 MVP

最后更新：2026-06-19

## 背景

Baseline 和数据 MVP 完成后，需要训练第一版 Qwen3-ASR 鲁棒 ASR LoRA，验证参数高效微调是否能改善退化音频识别。

## 目标

在 Colab 上跑通 Qwen3-ASR-1.7B QLoRA/LoRA 训练，产出可加载、可评测的 adapter。

## 范围

本步骤只做第一版监督微调，不做 RL，不做大规模训练。

`05A LoRA 训练前探测` 已完成。本小阶段确认了 Qwen3-ASR 的真实模块结构、
LoRA target 候选和 Colab 资源可行性。`05B Unsloth 兼容性检查` 已完成，结果为
不兼容：Unsloth 当前无法通过 Transformers AutoConfig 加载 `model_type=qwen3_asr`。
下一步进入 `05C Transformers + PEFT LoRA smoke training`：跑 5-20 step，验证
target 可训练、loss 正常下降、adapter 可保存加载。

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

`05A` 训练前探测输出：

- `outputs/lora_probe/qwen3_asr_1_7b/module_snapshot.json`
- `outputs/lora_probe/qwen3_asr_1_7b/module_summary.csv`
- `outputs/lora_probe/qwen3_asr_1_7b/lora_target_candidates.json`
- `outputs/lora_probe/qwen3_asr_1_7b/lora_target_candidates.md`

## 需要实现的文件

- `train/inspect_qwen3_asr_modules.py`
- `train/check_unsloth_qwen3_asr.py`
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

### 05A 训练前探测命令

在 Colab GPU runtime 中执行：

```bash
python train/inspect_qwen3_asr_modules.py \
  --model-id Qwen/Qwen3-ASR-1.7B \
  --output-dir outputs/lora_probe/qwen3_asr_1_7b \
  --dtype float16 \
  --device-map cuda:0 \
  --max-inference-batch-size 1 \
  --max-new-tokens 128
```

也可以直接运行 `notebooks/03_train_lora_colab.ipynb` 的训练前探测部分。该
notebook 默认项目目录为 `/content/drive/MyDrive/qwen3-asr`。

### 05A 通过标准

- 脚本能通过官方 `qwen-asr` API 加载 `Qwen/Qwen3-ASR-1.7B`。
- 能找到至少一个 `torch.nn.Module` 根节点。
- 能导出 `module_snapshot.json` 和 `module_summary.csv`。
- `lora_target_candidates.json` 中包含候选分组和模块名。
- 人工复核后能明确第一版 smoke training 的 target 组。

### 05A 验收标准

- 探测输出已保存到 `outputs/lora_probe/qwen3_asr_1_7b/`。
- 关键输出已提交到仓库，便于后续排查。
- 进度文档记录探测结论、候选 target 和风险。
- 未完成探测前，不开始 `train/train_qwen3_asr_lora.py` 的正式训练实现。

### 05A 探测结果

探测输出来自 Colab GPU runtime，路径为 `outputs/lora_probe/qwen3_asr_1_7b/`。

| 项 | 结果 |
| --- | ---: |
| root module | `model: Qwen3ASRForConditionalGeneration` |
| total modules | 703 |
| audio encoder layers | 24 |
| text decoder layers | 28 |
| attention candidates | 208 |
| MLP candidates | 132 |
| speech projection candidates | 3 |
| `lm_head` | 1 |

候选模块分布：

- `attention_projection`：208 个，其中 audio tower 96 个、text decoder 112 个。
- `mlp_projection`：132 个，其中 audio tower 48 个、text decoder 84 个。
- `speech_projection`：3 个，分别是 `conv_out`、`proj1`、`proj2`。
- `speech_conv`：3 个，默认只观察，不作为第一版 LoRA target。
- `lm_head`：1 个，体量大且会直接改变 token 输出分布，第一版不训练。

### 05B 第一版 smoke target 决策

训练 backend 决策：

- 已尝试 Unsloth。官方 Unsloth 支持普通 Qwen3/Qwen3 MoE 高效微调，但 Qwen3-ASR 是音频 ASR 架构。
- 兼容性检查失败：`transformers==4.57.6` 的 AutoConfig 不识别 `model_type=qwen3_asr`，Unsloth `FastModel.from_pretrained` 无法直接加载 `Qwen/Qwen3-ASR-1.7B`。
- 当前 MVP 回退到 Transformers + PEFT，并保留 qwen-asr 做推理评测入口。

第一版 smoke training 使用 `audio_tower_attention_plus_projection_smoke` 策略：

```text
model.thinker.audio_tower.layers.*.self_attn.q_proj
model.thinker.audio_tower.layers.*.self_attn.k_proj
model.thinker.audio_tower.layers.*.self_attn.v_proj
model.thinker.audio_tower.layers.*.self_attn.out_proj
model.thinker.audio_tower.conv_out
model.thinker.audio_tower.proj1
model.thinker.audio_tower.proj2
```

选择原因：

- 目标直接位于音频路径，优先解决 noise、reverb、far_field、dropout 的声学鲁棒性。
- 不先动 text decoder，降低 clean regression 和 hallucination 风险。
- 不训练 `lm_head`，避免直接改变词表输出倾向。
- `r=8` 预计 LoRA 可训练参数约 1,683,456，适合 Colab Free 做 smoke test。

第一版不训练：

- text decoder attention/MLP。
- audio tower MLP。
- speech conv。
- `lm_head`。

如果第一版 target 能稳定训练但收益不足，再做第二轮 ablation：加入 audio tower MLP，或单独比较 text decoder attention。

### 05B Unsloth 兼容性检查

检查目标：

- `unsloth` 和 `unsloth_zoo` 能在 Colab GPU runtime 中安装。
- `FastModel.from_pretrained` 能否加载 `Qwen/Qwen3-ASR-1.7B`。
- 加载后的模型是否仍暴露 `model.thinker.audio_tower` 模块。
- Unsloth 的 LoRA API 是否支持精确限制到 audio tower target，而不是同时命中文本 decoder。

检查结果：

- 已写出 `outputs/lora_probe/qwen3_asr_1_7b/unsloth_compatibility.json`。
- `compatible=false`。
- 根因：Unsloth 使用标准 Transformers AutoConfig 路径；当前 `transformers==4.57.6` 不认识 `qwen3_asr` 架构。
- 决策：不继续通过依赖 pinning 强推 Unsloth，切换到 `backend: transformers_peft`。

### 05C Transformers + PEFT smoke training

实现目标：

- 用 qwen-asr 或底层 Qwen3-ASR torch model 作为加载入口。
- 用 PEFT 按完整模块正则挂载 LoRA，严格限制到 99 个 audio tower target。
- 先使用 5-20 step smoke test，不追求指标提升。
- 保存 adapter、训练配置、target_modules、loss log 和最小推理输出。

通过标准：

- target 匹配数量等于 `expected_target_count=99`。
- 可训练参数量接近 1,683,456。
- 训练 loss 非 NaN。
- adapter 可保存并重新加载。
- 至少 1 条 clean 与 1 条 degraded 音频能完成 LoRA 推理。

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
