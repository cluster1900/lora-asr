# Qwen3-ASR Robust ASR

这是一个基于 Qwen3-ASR-1.7B 的独立鲁棒 ASR 项目。

项目目标是完成一个类似 Mega-ASR 能力形态的产品雏形：具备鲁棒 ASR LoRA、音频质量 router、统一推理入口、数据增强管线、评测体系和发布文档。但本项目不以 Mega-ASR 代码作为实现底座，所有新功能都应基于 Qwen3-ASR 的真实 API 和我们自己的工程结构开发。

## 当前状态

当前已完成第一版 baseline smoke 闭环、150 条本地合成 MVP 评测集生成与评测入口，并完成 LoRA 训练前探测：

```text
clean/noise 音频
  -> JSONL manifest
  -> Qwen3-ASR-1.7B baseline 推理
  -> prediction JSONL
  -> WER/CER 评测
  -> overall 与 scenario-level 指标
```

Qwen3-ASR-1.7B 已在 Colab GPU runtime 中完成 MVP 150 hard profile baseline 评测。训练前探测已确认官方 `qwen-asr` wrapper 暴露 `Qwen3ASRForConditionalGeneration` 根节点，第一版 LoRA smoke target 收敛为 audio tower attention + speech projection。Unsloth 兼容性检查已失败，当前训练框架回退为 Transformers + PEFT；20 step smoke training 已跑通。正式 LoRA v1 bootstrap 训练和 held-out 评测已完成，但 noise/reverb 目标场景变差，因此当前进入 v2 快速 ablation，不进入 router。

Colab 侧推荐先运行 `notebooks/00_clone_github_colab.ipynb`，确认 Google Drive 中的工程代码已经更新到 GitHub 最新提交，再继续执行 baseline 或训练 notebook。

## 文档入口

- [项目方案](docs/qwen3-asr/README.md)
- [开发进度](docs/qwen3-asr/00_progress.md)
- [路线图总览](docs/qwen3-asr/roadmap/OVERVIEW.md)
- [执行路线图](docs/qwen3-asr/roadmap/README.md)
- [文档验收与追踪矩阵](docs/qwen3-asr/07_document_acceptance.md)
- [协作规范](AGENTS.md)

## 已落地内容

- `scripts/create_smoke_audio.py`：本地生成 clean/noise smoke 音频和本地 manifest。
- `scripts/create_mvp_eval_audio.py`：本地生成 150 条 MVP 评测音频，覆盖 clean、noise、reverb、far_field、dropout。
- `scripts/create_lora_mvp_dataset.py`：生成正式 LoRA MVP 启动用 bootstrap train/val manifest，默认覆盖 clean、noise、reverb。
- `data/mvp_eval/audio/`：已提交 150 条 MVP smoke 音频，供 Colab 直接拉取后训练前检查和 smoke training 使用。
- `inference/qwen3_asr_base_infer.py`：读取 JSONL manifest，调用 Qwen3-ASR baseline 生成 ASR prediction JSONL。
- `inference/qwen3_asr_lora_infer.py`：加载 Qwen3-ASR base 与 PEFT adapter，生成 LoRA always-on ASR prediction JSONL。
- `evaluation/eval_wer.py`：计算 WER/CER、overall 指标和 scenario-level 指标。
- `evaluation/analyze_errors.py`：分析 scored prediction JSONL，输出 worst cases、场景聚合和错误标签。
- `train/inspect_qwen3_asr_modules.py`：训练前探测 Qwen3-ASR 模块结构，并生成 LoRA target 候选。
- `train/check_unsloth_qwen3_asr.py`：检查 Unsloth 是否能加载 Qwen3-ASR 并精确匹配 audio tower target。
- `train/train_qwen3_asr_lora.py`：使用 Transformers + PEFT 对 Qwen3-ASR 跑 LoRA smoke training。
- `notebooks/00_clone_github_colab.ipynb`：Colab/Google Drive clone 或更新 GitHub 工程，并校验当前 commit 和关键修复。
- `notebooks/00_github_commit_push_colab.ipynb`：Colab/Google Drive 中单独提交并推送受控实验输出，使用 Colab Secret `GITHUB_TOKEN` 授权。
- `notebooks/01_baseline_colab.ipynb`：Colab/Google Drive baseline smoke notebook。
- `notebooks/02_mvp_150_eval_colab.ipynb`：Colab/Google Drive MVP 150 baseline 评测 notebook。
- `notebooks/03_train_lora_colab.ipynb`：Colab/Google Drive LoRA 探测和 PEFT smoke training 入口。
- `notebooks/04_train_lora_mvp_colab.ipynb`：Colab/Google Drive 正式 LoRA MVP bootstrap 训练入口，默认 600 step。
- `notebooks/05_eval_lora_mvp_colab.ipynb`：Colab/Google Drive LoRA MVP held-out 评测入口，对比 base 与 LoRA。
- `notebooks/06_train_lora_mvp_v2_colab.ipynb`：Colab/Google Drive LoRA MVP v2 快速 ablation 入口，默认 attention-only、noise/reverb-only、长短句均衡采样。
- `configs/baseline/qwen3_asr_baseline.yaml`：baseline smoke 配置。

## 快速开始

### 0. Colab 拉取或更新工程

打开并按顺序执行：

```text
notebooks/00_clone_github_colab.ipynb
```

默认 Google Drive 项目路径：

```text
/content/drive/MyDrive/qwen3-asr
```

执行完成后应看到当前 commit、最新 commit message，以及关键修复标记检查通过。

### 1. 本地生成 smoke 音频

该脚本依赖 macOS `say`，用于没有真实音频时快速生成 clean/noise 两条测试音频。

```bash
python3 scripts/create_smoke_audio.py --force
```

默认输出：

- `data/local_smoke/audio/clean_0001.wav`
- `data/local_smoke/audio/noise_0001.wav`
- `data/jsonl/baseline_smoke.local.jsonl`

这些生成物只用于本地 smoke test，已被 `.gitignore` 排除。

### 2. Colab 跑 Qwen3-ASR baseline

打开并按顺序执行：

```text
notebooks/01_baseline_colab.ipynb
```

默认 Google Drive 项目路径：

```text
/content/drive/MyDrive/qwen3-asr
```

如遇 Hugging Face 下载权限或限流问题，在 notebook 中登录后重试。

### 3. 生成并评测 MVP 150 条音频

本地生成：

```bash
python3 scripts/create_mvp_eval_audio.py --profile hard --force
```

默认输出：

- `data/mvp_eval/audio/`
- `data/jsonl/baseline_mvp_150.local.jsonl`
- `data/jsonl/baseline_mvp_150_stats.local.json`

当前仓库已经提交一份 MVP 150 音频，Colab 中 `git pull` 后应能直接得到 `data/mvp_eval/audio/`。如需重新生成本地版本，再把新文件同步到 Google Drive 项目目录后执行：

```text
notebooks/02_mvp_150_eval_colab.ipynb
```

### 4. 脚本化推理与评测

推理：

```bash
python3 inference/qwen3_asr_base_infer.py \
  --manifest data/jsonl/baseline_smoke.local.jsonl \
  --output-jsonl outputs/baseline/predictions.qwen3_asr_base.smoke.jsonl \
  --model-id Qwen/Qwen3-ASR-1.7B \
  --dtype float16 \
  --device-map cuda:0 \
  --max-inference-batch-size 1 \
  --max-new-tokens 128 \
  --language English \
  --limit 2
```

评测：

```bash
python3 evaluation/eval_wer.py \
  --predictions-jsonl outputs/baseline/predictions.qwen3_asr_base.smoke.jsonl \
  --scored-jsonl outputs/baseline/predictions.qwen3_asr_base.smoke.scored.jsonl \
  --metrics-json outputs/baseline/metrics.qwen3_asr_base.smoke.json \
  --metrics-by-scenario-csv outputs/baseline/metrics_by_scenario.qwen3_asr_base.smoke.csv
```

### 5. Colab 探测 LoRA target 并跑 smoke training

打开并按顺序执行：

```text
notebooks/03_train_lora_colab.ipynb
```

默认 Google Drive 项目路径：

```text
/content/drive/MyDrive/qwen3-asr
```

探测主要输出：

- `outputs/lora_probe/qwen3_asr_1_7b/module_snapshot.json`
- `outputs/lora_probe/qwen3_asr_1_7b/module_summary.csv`
- `outputs/lora_probe/qwen3_asr_1_7b/lora_target_candidates.json`
- `outputs/lora_probe/qwen3_asr_1_7b/lora_target_candidates.md`

训练主要输出：

- `checkpoints/qwen3-asr-1.7b-lora/target_modules.json`
- `checkpoints/qwen3-asr-1.7b-lora/training_config.json`
- `checkpoints/qwen3-asr-1.7b-lora/loss_log.jsonl`
- `checkpoints/qwen3-asr-1.7b-lora/summary.json`
- `checkpoints/qwen3-asr-1.7b-lora/adapter/`

### 6. 生成 LoRA MVP bootstrap train/val

正式 LoRA MVP 不直接使用 MVP 150 held-out test 训练。先生成独立 train/val：

```bash
python3 scripts/create_lora_mvp_dataset.py \
  --profile medium \
  --train-items-per-scenario 120 \
  --val-items-per-scenario 30 \
  --scenarios clean,noise,reverb \
  --force
```

默认输出：

- `data/lora_mvp/audio/`
- `data/jsonl/lora_mvp_train.local.jsonl`
- `data/jsonl/lora_mvp_val.local.jsonl`
- `data/jsonl/lora_mvp_stats.local.json`

第一版训练配置为：

```text
configs/train/qwen3_asr_lora_mvp_train.yaml
```

Colab 中正式训练入口为：

```text
notebooks/04_train_lora_mvp_colab.ipynb
```

### 7. 评测 LoRA MVP adapter

正式训练完成后，先在固定 MVP 150 held-out test 上跑 LoRA always-on：

```bash
python3 inference/qwen3_asr_lora_infer.py \
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

Colab 中推荐直接执行：

```text
notebooks/05_eval_lora_mvp_colab.ipynb
```

## 目录结构

```text
configs/      配置文件
data/         JSONL 示例与数据说明
docs/         中文架构、开发、测试、进度和路线图文档
evaluation/   WER/CER 与错误分析工具
inference/    Qwen3-ASR baseline/LoRA/router 推理入口
notebooks/    Colab 优先 notebook
router/       后续音频质量 router
scripts/      数据准备和 smoke 工具脚本
train/        后续 LoRA/QLoRA 训练入口
```

## 参考工程

原 Mega-ASR 上游工程应放在本地忽略目录：

- `references/mega-asr-upstream/`

该目录只用于查阅和对照，不作为新工程运行时依赖，不在其中继续开发 Qwen3-ASR 功能。`references/` 已被 `.gitignore` 排除，不进入 git。

## 下一步

按路线图继续执行：

1. 执行 [05 LoRA 训练 MVP](docs/qwen3-asr/roadmap/05_lora_training_mvp.md) 的 `05E`：跑 v2 attention-only、noise/reverb-only、长短句均衡采样短实验。
2. 用固定 MVP 150 held-out test 集比较 base、v1 和 v2。
3. 只有当 LoRA always-on 至少在 noise 或 reverb 优于 base 后，再进入 [07 Router MVP](docs/qwen3-asr/roadmap/07_router_mvp.md)。
