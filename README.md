# Gemma 4 Robust ASR

这是一个基于 Gemma 4 12B 的独立鲁棒 ASR 项目。

项目目标是完成一个类似 Mega-ASR 能力形态的产品雏形：具备鲁棒 ASR LoRA、音频质量 router、统一推理入口、数据增强管线、评测体系和发布文档。但本项目不以 Mega-ASR 代码作为实现底座，所有新功能都应基于 Gemma 4 的真实 API 和我们自己的工程结构开发。

## 当前状态

当前已完成第一版 baseline smoke 闭环：

```text
clean/noise 音频
  -> JSONL manifest
  -> Gemma 4 12B baseline 推理
  -> prediction JSONL
  -> WER/CER 评测
  -> overall 与 scenario-level 指标
```

已在 Colab 中跑通 clean/noise 各 1 条样本，smoke 结果 error_rate 为 0.0。该结果只证明流程可运行，不代表模型鲁棒性已达标。

## 文档入口

- [项目方案](docs/gemma4-mega-asr/README.md)
- [开发进度](docs/gemma4-mega-asr/00_progress.md)
- [路线图总览](docs/gemma4-mega-asr/roadmap/OVERVIEW.md)
- [执行路线图](docs/gemma4-mega-asr/roadmap/README.md)
- [文档验收与追踪矩阵](docs/gemma4-mega-asr/07_document_acceptance.md)
- [协作规范](AGENTS.md)

## 已落地内容

- `scripts/create_smoke_audio.py`：本地生成 clean/noise smoke 音频和本地 manifest。
- `inference/gemma4_base_infer.py`：读取 JSONL manifest，调用 Gemma 4 baseline 生成 ASR prediction JSONL。
- `evaluation/eval_wer.py`：计算 WER/CER、overall 指标和 scenario-level 指标。
- `notebooks/01_baseline_colab.ipynb`：Colab/Google Drive baseline smoke notebook。
- `configs/baseline/gemma4_baseline.yaml`：baseline smoke 配置。

## 快速开始

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

### 2. Colab 跑 Gemma 4 baseline

打开并按顺序执行：

```text
notebooks/01_baseline_colab.ipynb
```

默认 Google Drive 项目路径：

```text
/content/drive/MyDrive/gemma-mega-asr
```

需要 Hugging Face 账号已接受 Gemma 4 模型条款，并在 notebook 中登录。

### 3. 脚本化推理与评测

推理：

```bash
python3 inference/gemma4_base_infer.py \
  --manifest data/jsonl/baseline_smoke.local.jsonl \
  --output-jsonl outputs/baseline/predictions.gemma4_base.smoke.jsonl \
  --model-id google/gemma-4-12B-it \
  --dtype float16 \
  --device-map auto \
  --max-new-tokens 128 \
  --limit 2
```

评测：

```bash
python3 evaluation/eval_wer.py \
  --predictions-jsonl outputs/baseline/predictions.gemma4_base.smoke.jsonl \
  --scored-jsonl outputs/baseline/predictions.gemma4_base.smoke.scored.jsonl \
  --metrics-json outputs/baseline/metrics.gemma4_base.smoke.json \
  --metrics-by-scenario-csv outputs/baseline/metrics_by_scenario.gemma4_base.smoke.csv
```

## 目录结构

```text
configs/      配置文件
data/         JSONL 示例与数据说明
docs/         中文架构、开发、测试、进度和路线图文档
evaluation/   WER/CER 与错误分析工具
inference/    Gemma 4 baseline/LoRA/router 推理入口
notebooks/    Colab 优先 notebook
router/       后续音频质量 router
scripts/      数据准备和 smoke 工具脚本
train/        后续 LoRA/QLoRA 训练入口
```

## 参考工程

原 Mega-ASR 上游工程应放在本地忽略目录：

- `references/mega-asr-upstream/`

该目录只用于查阅和对照，不作为新工程运行时依赖，不在其中继续开发 Gemma 4 功能。`references/` 已被 `.gitignore` 排除，不进入 git。

## 下一步

按路线图继续执行：

1. 扩大 [02 Baseline 评估](docs/gemma4-mega-asr/roadmap/02_baseline_eval.md)：从 2 条 smoke 样本扩展到更多 clean/degraded 样本。
2. 执行 [03 数据 MVP](docs/gemma4-mega-asr/roadmap/03_data_mvp.md)：覆盖 clean、noise、reverb、far_field、dropout。
3. 执行 [04 音频增强](docs/gemma4-mega-asr/roadmap/04_audio_augmentation.md)：实现可复现的退化增强管线。
4. 执行 [05 LoRA 训练 MVP](docs/gemma4-mega-asr/roadmap/05_lora_training_mvp.md)：在 Colab 中跑通第一版 QLoRA/LoRA smoke training。
