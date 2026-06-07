# inference

存放新工程的推理入口。

目标模式：

- `base`：Gemma 4 原始模型。
- `lora`：Gemma 4 + 鲁棒 ASR LoRA。
- `router`：根据音频质量自动选择 base 或 LoRA。

禁止：

- 不复用 Mega-ASR 的 Qwen3-ASR 推理 wrapper。

