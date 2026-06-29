# configs

存放新工程的配置文件。

规则：

- 配置必须服务于 Qwen3-ASR Robust ASR。
- 可以使用 Qwen3-ASR 官方 `qwen-asr` 参数，但不要混入 Mega-ASR 上游工程的私有路径、wrapper 或命名。
- 训练、评测、数据构建都应能追溯到对应配置文件。

当前训练配置：

- `configs/train/qwen3_asr_lora_mvp.yaml`：历史 `05C` smoke training 配置，只用于验证 LoRA 通路。
- `configs/train/qwen3_asr_lora_mvp_train.yaml`：正式 LoRA MVP 第一版训练配置，默认读取独立 bootstrap train/val manifest，并保留 MVP 150 作为 held-out test。
