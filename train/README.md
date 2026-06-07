# train

存放 Gemma 4 LoRA/QLoRA 训练代码。

目标：

- 加载 Gemma 4。
- 探测 LoRA target。
- 构建音频 + 文本 collator。
- 保存可复现 adapter checkpoint。

禁止：

- 不复用 Mega-ASR 的 Qwen3-ASR 训练入口。

