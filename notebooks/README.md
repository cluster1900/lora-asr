# notebooks

> 当前主线尚未实现，唯一计划入口是 `12_fast_finetune_colab.ipynb`。现有 00-11
> notebook 均为历史实验或仓库辅助工具，不再继续扩建 v6A、difficulty、target sweep
> 或 router 路线。执行合同见 `docs/qwen3-asr/02_development_plan.md`。

存放 Colab 优先的 notebook。正式 notebook 放在本目录，`colab/` 不作为主工程目录使用。

规划：

- `00_clone_github_colab.ipynb`：挂载 Google Drive，clone/update GitHub 工程，并打印 commit 与关键修复标记。
- `00_github_commit_push_colab.ipynb`：单独提交并推送 Colab 产生的受控输出；使用 Colab Secret `GITHUB_TOKEN` 授权。
- `01_baseline_colab.ipynb`：Qwen3-ASR-1.7B baseline smoke 推理与 WER/CER 评测。
- `02_mvp_150_eval_colab.ipynb`：读取本地生成并上传到 Drive 的 150 条 MVP 评测集，运行 Qwen3-ASR baseline 推理和场景级 WER/CER。
- `03_train_lora_colab.ipynb`：执行 Qwen3-ASR 模块探测和 20 step Transformers + PEFT smoke training。
- `04_train_lora_mvp_colab.ipynb`：执行正式 LoRA MVP bootstrap 训练，使用独立 clean/noise/reverb train manifest，默认 600 step。
- `05_eval_lora_mvp_colab.ipynb`：加载正式 LoRA MVP adapter，在固定 MVP 150 held-out test 上运行 LoRA always-on 推理、WER/CER 评测和错误分析。
- `06_train_lora_mvp_v2_colab.ipynb`：执行 LoRA MVP v2 ablation，默认 attention-only、noise/reverb-only，并在训练后自动跑 held-out 评测。
- `07_train_lora_mvp_v3_colab.ipynb`：执行 LoRA MVP v3 target-focus，默认使用 v1 的 99 个 target、noise/reverb-only、长短均衡采样，目标是相对 4bit base recheck 接近或超过 10% 改善。
- `08_train_lora_mvp_v4_checkpoint_sweep_colab.ipynb`：执行 LoRA MVP v4 checkpoint sweep，沿用 v3 方向训练到 600 step，并评测 160/320/480/final 600 step checkpoint。
- `09_train_lora_mvp_v5_late_audio_mlp_colab.ipynb`：执行 LoRA MVP v5 late audio MLP ablation，在 99 个音频侧 target 基础上新增 audio tower 12-23 层 `fc1/fc2`，评测 160/320/final 480 step checkpoint。
- `10_make_hard_profile_dataset_colab.ipynb`：生成 v6A hard-profile train/val 数据，默认从 `lora_mvp` clean 音频派生 7 类场景，并输出 stats。
- `11_score_base_difficulty_colab.ipynb`：对 v6A train/val manifest 跑 4bit base inference、WER、错误分析，并生成 difficulty manifest。
- `12_fast_finetune_colab.ipynb`：待实现，公开 200k、官方 Trainer、BF16 broad LoRA 的唯一入口。
- `02_make_dataset_colab.ipynb`
- `16_router_colab.ipynb`：规划中，只有 LoRA 达到门槛后再训练 router。

规则：

- notebook 必须能从 Google Drive 读取输入和写出结果。
- 关键参数应来自 `configs/`。
- 提交到仓库前应清理执行输出、登录 widget 状态和个人 token。
- 每个关键单元应有中文注释，说明输入、输出和失败时应检查什么。
