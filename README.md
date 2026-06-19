# Qwen3-ASR Robust ASR

这是一个基于 Qwen3-ASR-1.7B 的独立鲁棒 ASR 项目。

项目目标是完成一个类似 Mega-ASR 能力形态的产品雏形：具备鲁棒 ASR LoRA、音频质量 router、统一推理入口、数据增强管线、评测体系和发布文档。但本项目不以 Mega-ASR 代码作为实现底座，所有新功能都应基于 Qwen3-ASR 的真实 API 和我们自己的工程结构开发。

## 当前状态

当前已完成第一版 baseline smoke 闭环、150 条本地合成 MVP 评测集生成与评测入口，并进入 LoRA 训练前探测阶段：

```text
clean/noise 音频
  -> JSONL manifest
  -> Qwen3-ASR-1.7B baseline 推理
  -> prediction JSONL
  -> WER/CER 评测
  -> overall 与 scenario-level 指标
```

Qwen3-ASR-1.7B 已在 Colab GPU runtime 中完成 MVP 150 hard profile baseline 评测。当前下一步不是直接训练，而是在 Colab 中导出 Qwen3-ASR 的真实模块结构和 LoRA target 候选，避免复用外部工程的 target 规则。

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
- `inference/qwen3_asr_base_infer.py`：读取 JSONL manifest，调用 Qwen3-ASR baseline 生成 ASR prediction JSONL。
- `evaluation/eval_wer.py`：计算 WER/CER、overall 指标和 scenario-level 指标。
- `evaluation/analyze_errors.py`：分析 scored prediction JSONL，输出 worst cases、场景聚合和错误标签。
- `train/inspect_qwen3_asr_modules.py`：训练前探测 Qwen3-ASR 模块结构，并生成 LoRA target 候选。
- `notebooks/01_baseline_colab.ipynb`：Colab/Google Drive baseline smoke notebook。
- `notebooks/02_mvp_150_eval_colab.ipynb`：Colab/Google Drive MVP 150 baseline 评测 notebook。
- `notebooks/03_train_lora_colab.ipynb`：Colab/Google Drive LoRA 训练前探测入口。
- `configs/baseline/qwen3_asr_baseline.yaml`：baseline smoke 配置。

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

把这些文件同步到 Google Drive 项目目录后，在 Colab 中执行：

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

### 5. Colab 探测 LoRA target 候选

打开并按顺序执行：

```text
notebooks/03_train_lora_colab.ipynb
```

默认 Google Drive 项目路径：

```text
/content/drive/MyDrive/qwen3-asr
```

主要输出：

- `outputs/lora_probe/qwen3_asr_1_7b/module_snapshot.json`
- `outputs/lora_probe/qwen3_asr_1_7b/module_summary.csv`
- `outputs/lora_probe/qwen3_asr_1_7b/lora_target_candidates.json`
- `outputs/lora_probe/qwen3_asr_1_7b/lora_target_candidates.md`

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

1. 执行 [05 LoRA 训练 MVP](docs/qwen3-asr/roadmap/05_lora_training_mvp.md) 的 `05A`：导出模块快照和候选 LoRA target。
2. 根据探测结果实现第一版 QLoRA/LoRA smoke training。
3. 用同一套 MVP 150 test 集比较 base 与 LoRA，重点检查 noise/reverb 改善和 clean regression。
4. 若 LoRA 有收益，再进入 [07 Router MVP](docs/qwen3-asr/roadmap/07_router_mvp.md)。
