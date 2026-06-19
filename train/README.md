# train

存放 Qwen3-ASR LoRA/QLoRA 训练代码。

目标：

- 加载 Qwen3-ASR。
- 探测 LoRA target。
- 构建音频 + 文本 collator。
- 保存可复现 adapter checkpoint。

当前阶段：

- `inspect_qwen3_asr_modules.py`：训练前探测脚本，在 Colab GPU runtime 中加载官方 `qwen-asr` 模型，导出 `named_modules()` 快照和 LoRA target 候选。
- `check_unsloth_qwen3_asr.py`：Unsloth 兼容性检查脚本。当前结果为不兼容，后续训练回退 Transformers + PEFT。
- `lora_targets.py`：候选 target 分组规则，只做辅助分析，不直接代表最终训练配置。
- `peft_targets.py`：PEFT LoRA target 匹配与校验工具，使用配置中的 include/exclude regex 精确限制训练模块。
- `train_qwen3_asr_lora.py`：Transformers + PEFT smoke training 入口，当前只支持 batch size 1 的 5-20 step 小闭环。

推荐命令：

```bash
python train/inspect_qwen3_asr_modules.py \
  --model-id Qwen/Qwen3-ASR-1.7B \
  --output-dir outputs/lora_probe/qwen3_asr_1_7b \
  --dtype float16 \
  --device-map cuda:0
```

Unsloth 兼容性检查记录：

```bash
python train/check_unsloth_qwen3_asr.py \
  --config configs/train/qwen3_asr_lora_mvp.yaml \
  --output-json outputs/lora_probe/qwen3_asr_1_7b/unsloth_compatibility.json
```

当前 `unsloth_compatibility.json` 中 `compatible=false`，原因是 Unsloth 走标准 Transformers AutoConfig 路径时无法识别 `model_type=qwen3_asr`。下一步实现 PEFT smoke training。

Transformers + PEFT smoke training：

```bash
python train/train_qwen3_asr_lora.py \
  --config configs/train/qwen3_asr_lora_mvp.yaml \
  --manifest data/jsonl/baseline_mvp_150.local.jsonl \
  --audio-root . \
  --output-dir checkpoints/qwen3-asr-1.7b-lora \
  --model-id Qwen/Qwen3-ASR-1.7B \
  --dtype float16 \
  --device-map cuda:0 \
  --quantization 4bit \
  --language English \
  --limit 20 \
  --max-steps 20
```

主要输出：

- `target_modules.json`：本次实际命中的 LoRA target，必须为 99 个 audio tower 模块。
- `training_config.json`：训练配置、manifest、随机种子和命令行覆盖项。
- `loss_log.jsonl`：每一步 loss、场景和样本 id。
- `summary.json`：smoke training 摘要。
- `adapter/`：PEFT adapter。

禁止：

- 不复用 Mega-ASR 的 Qwen3-ASR 训练入口。
