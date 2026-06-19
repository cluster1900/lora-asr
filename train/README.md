# train

存放 Qwen3-ASR LoRA/QLoRA 训练代码。

目标：

- 加载 Qwen3-ASR。
- 探测 LoRA target。
- 构建音频 + 文本 collator。
- 保存可复现 adapter checkpoint。

当前阶段：

- `inspect_qwen3_asr_modules.py`：训练前探测脚本，在 Colab GPU runtime 中加载官方 `qwen-asr` 模型，导出 `named_modules()` 快照和 LoRA target 候选。
- `lora_targets.py`：候选 target 分组规则，只做辅助分析，不直接代表最终训练配置。

推荐命令：

```bash
python train/inspect_qwen3_asr_modules.py \
  --model-id Qwen/Qwen3-ASR-1.7B \
  --output-dir outputs/lora_probe/qwen3_asr_1_7b \
  --dtype float16 \
  --device-map cuda:0
```

禁止：

- 不复用 Mega-ASR 的 Qwen3-ASR 训练入口。
