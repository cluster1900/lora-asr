# inference

存放新工程的推理入口。

目标模式：

- `base`：Qwen3-ASR 原始模型。
- `lora`：Qwen3-ASR + 鲁棒 ASR LoRA。
- `router`：根据音频质量自动选择 base 或 LoRA。

禁止：

- 不复用 Mega-ASR 的 Qwen3-ASR 推理 wrapper。

## 当前脚本

- `qwen3_asr_base_infer.py`：读取 JSONL manifest，使用 Qwen3-ASR base 模型生成 ASR prediction JSONL。
- `qwen3_asr_lora_infer.py`：读取 JSONL manifest，加载 Qwen3-ASR base 和 PEFT adapter，生成 LoRA always-on ASR prediction JSONL。

## 示例

```bash
python inference/qwen3_asr_base_infer.py \
  --manifest data/jsonl/baseline_smoke.local.jsonl \
  --output-jsonl outputs/baseline/predictions.jsonl \
  --model-id Qwen/Qwen3-ASR-1.7B \
  --dtype float16 \
  --device-map cuda:0 \
  --max-inference-batch-size 1 \
  --language English \
  --limit 2
```

该命令需要稳定网络下载模型权重，并建议在 GPU runtime 上运行。

LoRA always-on 推理示例：

```bash
python inference/qwen3_asr_lora_infer.py \
  --manifest data/jsonl/baseline_mvp_150.local.jsonl \
  --output-jsonl outputs/lora_mvp_eval/predictions.qwen3_asr_lora_mvp.mvp_150.jsonl \
  --adapter-dir checkpoints/qwen3-asr-1.7b-lora-mvp/adapter \
  --audio-root . \
  --model-id Qwen/Qwen3-ASR-1.7B \
  --dtype float16 \
  --device-map cuda:0 \
  --quantization 4bit \
  --max-inference-batch-size 1 \
  --max-new-tokens 128 \
  --language English
```

LoRA 推理输出保持 baseline prediction JSONL 兼容格式，并额外写入 `mode=lora`
和 `adapter_dir`。后续统一交给 `evaluation/eval_wer.py` 计算指标。
