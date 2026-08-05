# configs

存放新工程的配置文件。

规则：

- 配置必须服务于 Qwen3-ASR Robust ASR。
- 可以使用 Qwen3-ASR 官方 `qwen-asr` 参数，但不要混入 Mega-ASR 上游工程的私有路径、wrapper 或命名。
- 训练、评测、数据构建都应能追溯到对应配置文件。

当前唯一正式配置（待按执行合同实现）：

- `configs/data/public_robust_200k.yaml`：pinned public data、固定配额、metadata probe、manifest 和 30k curriculum。
- `configs/train/qwen3_asr_public_200k_a2s.yaml`：单 adapter 三阶段 A2S 训练。

以下均为历史 smoke/MVP 配置，只用于复现旧实验，不作为新训练依赖：

历史 baseline 配置：

- `configs/baseline/qwen3_asr_baseline.yaml`：历史 baseline smoke 配置。
- `configs/baseline/qwen3_asr_base_recheck_mvp_150.yaml`：MVP 150 base recheck 配置，默认用 4bit 加载方式复核 fixed held-out test，便于和 LoRA v1/v2 评测对齐。

历史数据配置：

- `configs/data/mvp_dataset.yaml`：早期 MVP 数据配置占位。
- `configs/data/v6a_hard_profile.yaml`：v6A hard-profile 数据构建配置，默认从 `lora_mvp` clean 音频派生 7 类场景 train/val，不使用固定 MVP 150 held-out test。

历史评测配置：

- `configs/eval/default_eval.yaml`：通用评测配置占位。
- `configs/eval/qwen3_asr_v6a_base_difficulty.yaml`：v6A base difficulty scoring 配置，默认对 v6A train/val 跑 4bit base、计算 WER 并输出 difficulty manifest。

历史训练配置：

- `configs/train/qwen3_asr_lora_mvp.yaml`：历史 `05C` smoke training 配置，只用于验证 LoRA 通路。
- `configs/train/qwen3_asr_lora_mvp_train.yaml`：正式 LoRA MVP 第一版训练配置，默认读取独立 bootstrap train/val manifest，并保留 MVP 150 作为 held-out test。
- `configs/train/qwen3_asr_lora_mvp_v2_ablation.yaml`：LoRA MVP v2 ablation 配置，默认 attention-only、noise/reverb-only、较低学习率和较短训练步数。
- `configs/train/qwen3_asr_lora_mvp_v3_target_focus.yaml`：LoRA MVP v3 target-focus 配置，回到 v1 的 99 个 target，但只训练 noise/reverb，并以 4bit base recheck 为公平对照。
- `configs/train/qwen3_asr_lora_mvp_v4_checkpoint_sweep.yaml`：LoRA MVP v4 checkpoint sweep 配置，沿用 v3 方向延长训练并保存中间 checkpoint。
- `configs/train/qwen3_asr_lora_mvp_v5_late_audio_mlp.yaml`：LoRA MVP v5 late audio MLP 配置，用于验证新增 audio tower 后半层 MLP target 是否有效。
