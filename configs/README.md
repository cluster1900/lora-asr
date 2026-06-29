# configs

存放新工程的配置文件。

规则：

- 配置必须服务于 Qwen3-ASR Robust ASR。
- 可以使用 Qwen3-ASR 官方 `qwen-asr` 参数，但不要混入 Mega-ASR 上游工程的私有路径、wrapper 或命名。
- 训练、评测、数据构建都应能追溯到对应配置文件。

当前 baseline 配置：

- `configs/baseline/qwen3_asr_baseline.yaml`：历史 baseline smoke 配置。
- `configs/baseline/qwen3_asr_base_recheck_mvp_150.yaml`：MVP 150 base recheck 配置，默认用 4bit 加载方式复核 fixed held-out test，便于和 LoRA v1/v2 评测对齐。

当前训练配置：

- `configs/train/qwen3_asr_lora_mvp.yaml`：历史 `05C` smoke training 配置，只用于验证 LoRA 通路。
- `configs/train/qwen3_asr_lora_mvp_train.yaml`：正式 LoRA MVP 第一版训练配置，默认读取独立 bootstrap train/val manifest，并保留 MVP 150 作为 held-out test。
- `configs/train/qwen3_asr_lora_mvp_v2_ablation.yaml`：LoRA MVP v2 ablation 配置，默认 attention-only、noise/reverb-only、较低学习率和较短训练步数。
- `configs/train/qwen3_asr_lora_mvp_v3_target_focus.yaml`：LoRA MVP v3 target-focus 配置，回到 v1 的 99 个 target，但只训练 noise/reverb，并以 4bit base recheck 为公平对照。
