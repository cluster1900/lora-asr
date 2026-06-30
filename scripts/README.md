# scripts

存放数据准备、音频增强和辅助命令脚本。

规则：

- 脚本必须面向新工程目录结构。
- 不调用 `references/mega-asr-upstream/` 中的脚本作为主流程。

## 当前脚本

- `create_smoke_audio.py`：在本地生成 baseline smoke test 所需的 clean/noise 音频和 JSONL manifest。默认使用 macOS 自带 `say` 生成英文语音，再叠加噪声生成 degraded 样本。
- `create_mvp_eval_audio.py`：在本地生成 baseline MVP 评测集，覆盖 clean、noise、reverb、far_field、dropout 各 30 条，总计 150 条。默认使用 macOS `say` 合成 clean 语音，再用标准库生成退化版本。
- `create_lora_mvp_dataset.py`：生成正式 LoRA MVP 启动用 bootstrap train/val manifest。默认覆盖 clean、noise、reverb，不使用固定 MVP 150 held-out test 作为训练集。
- `create_v6a_hard_profile_dataset.py`：从已有 `lora_mvp` clean 音频派生 v6A hard-profile train/val manifest，默认覆盖 clean、noise、reverb、noise_reverb、far_field、dropout、far_field_noise。
- `run_qwen3_asr_base_recheck.py`：一键复核 Qwen3-ASR base，在同一 MVP 150 manifest 上完成 base 推理、WER/CER、错误分析和 historical base/LoRA 对比。

## 本地生成物

默认输出：

- `data/local_smoke/audio/clean_0001.wav`
- `data/local_smoke/audio/noise_0001.wav`
- `data/jsonl/baseline_smoke.local.jsonl`

这些文件用于本地测试，已被 `.gitignore` 排除。

MVP 150 条评测集默认输出：

- `data/mvp_eval/audio/`
- `data/jsonl/baseline_mvp_150.local.jsonl`
- `data/jsonl/baseline_mvp_150_stats.local.json`

生成命令：

```bash
python3 scripts/create_mvp_eval_audio.py --profile hard --force
```

默认 `hard` profile 会主动降低 degraded 场景质量，用于压出 baseline 错误；clean 场景仍保持清晰，用于观察 clean speech regression。该数据集用于工程闭环和 baseline 批量评测，不作为真实 benchmark。

默认 30 条文本中，前 15 条为短文本，后 15 条为长文本。manifest 会写入 `text_length_bucket` 和 `reference_word_count`，方便后续按短句/长句分析错误。

LoRA MVP bootstrap 默认输出：

- `data/lora_mvp/audio/`
- `data/jsonl/lora_mvp_train.local.jsonl`
- `data/jsonl/lora_mvp_val.local.jsonl`
- `data/jsonl/lora_mvp_stats.local.json`

生成命令：

```bash
python3 scripts/create_lora_mvp_dataset.py \
  --profile medium \
  --train-items-per-scenario 120 \
  --val-items-per-scenario 30 \
  --scenarios clean,noise,reverb \
  --force
```

该数据集用于启动第一版 LoRA MVP 训练闭环；正式评测仍使用固定 MVP 150 held-out test。

v6A hard-profile 默认输出：

- `data/v6a_hard_profile/audio/`
- `data/jsonl/v6a_hard_profile_train.local.jsonl`
- `data/jsonl/v6a_hard_profile_val.local.jsonl`
- `data/jsonl/v6a_hard_profile_stats.local.json`

生成命令：

```bash
python3 scripts/create_v6a_hard_profile_dataset.py \
  --config configs/data/v6a_hard_profile.yaml \
  --force
```

该脚本只读取 `lora_mvp` 的 clean train/val 源音频，不使用固定 MVP 150 held-out test。
Notebook 10 使用它生成 v6A 数据；Notebook 11 再做 base WER difficulty scoring。

Base recheck 默认输出：

- `outputs/base_recheck_mvp_150/predictions.qwen3_asr_base_recheck.mvp_150.jsonl`
- `outputs/base_recheck_mvp_150/metrics.qwen3_asr_base_recheck.mvp_150.json`
- `outputs/base_recheck_mvp_150/error_analysis/`
- `outputs/base_recheck_mvp_150/comparison.json`
- `outputs/base_recheck_mvp_150/comparison_by_scenario.csv`

复核命令：

```bash
python scripts/run_qwen3_asr_base_recheck.py \
  --config configs/baseline/qwen3_asr_base_recheck_mvp_150.yaml
```

该脚本默认使用 4bit 加载 base，目的是和 LoRA v1/v2 的评测加载方式对齐；输出不会覆盖历史 `outputs/baseline_mvp_150/`。
