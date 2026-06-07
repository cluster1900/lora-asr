# inference

存放新工程的推理入口。

目标模式：

- `base`：Gemma 4 原始模型。
- `lora`：Gemma 4 + 鲁棒 ASR LoRA。
- `router`：根据音频质量自动选择 base 或 LoRA。

禁止：

- 不复用 Mega-ASR 的 Qwen3-ASR 推理 wrapper。

## 当前脚本

- `gemma4_base_infer.py`：读取 JSONL manifest，使用 Gemma 4 base 模型生成 ASR prediction JSONL。

## 示例

```bash
python inference/gemma4_base_infer.py \
  --manifest data/jsonl/baseline_smoke.local.jsonl \
  --output-jsonl outputs/baseline/predictions.jsonl \
  --model-id google/gemma-4-12B-it \
  --limit 2
```

该命令需要可访问 Gemma 4 的 Hugging Face 权限和足够显存。
