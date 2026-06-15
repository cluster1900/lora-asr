# 02 Baseline 评估

最后更新：2026-06-15

## 背景

训练前必须先知道 Qwen3-ASR-1.7B 原始 ASR 能力。没有 baseline，后续 LoRA、数据增强和 router 的效果都无法判断。

## 目标

在 Colab 上跑通 Qwen3-ASR-1.7B baseline 推理，并输出可复现的 WER/CER 指标。

## 范围

本步骤只评估 base model，不训练 LoRA，不做 router。

## 输入

- `data/jsonl/baseline_smoke.jsonl`
- baseline 配置文件。
- 少量 clean/degraded 音频。

评测 JSONL 格式：

```json
{
  "audio": "/content/drive/MyDrive/qwen3-asr/audio/test/000001.wav",
  "answer": "REFERENCE TEXT",
  "language": "en",
  "scenario": "clean"
}
```

## 输出

- `outputs/baseline/predictions.jsonl`
- `outputs/baseline/metrics.json`
- `outputs/baseline/metrics_by_scenario.csv`
- baseline 失败样本记录。

## Colab Notebook 位置

正式 Colab 入口放在 `notebooks/01_baseline_colab.ipynb`。该 notebook 负责挂载 Google Drive、检查 smoke 音频和 manifest、安装 `qwen-asr` 与评测依赖、按需登录 Hugging Face、运行 baseline 推理并计算 WER/CER。

`colab/` 不作为主工程 notebook 目录，避免和已有 `notebooks/` 规划重复。

## 需要实现的文件

- `scripts/create_smoke_audio.py`
- `scripts/create_mvp_eval_audio.py`
- `notebooks/01_baseline_colab.ipynb`
- `notebooks/02_mvp_150_eval_colab.ipynb`
- `inference/qwen3_asr_base_infer.py`
- `evaluation/eval_wer.py`
- `configs/baseline/qwen3_asr_baseline.yaml`

## 执行步骤

1. 本地生成 smoke 音频和 manifest，或准备真实 clean/degraded 音频。
2. 在 Colab 挂载 Google Drive。
3. 安装 `qwen-asr`、音频处理、评测依赖。
4. 加载 `Qwen/Qwen3-ASR-1.7B`。
5. 读取 baseline smoke JSONL。
6. 对每条音频生成转写。
7. 保存 prediction JSONL。
8. 计算 WER/CER 和 scenario-level 指标。
9. 抽样记录失败样本。
10. 更新进度文档。

## 本地 smoke 音频生成

如果暂时没有真实音频，可以先用本地脚本生成一条 clean 语音和一条 noise 语音：

```bash
python scripts/create_smoke_audio.py --force
```

默认输出：

- `data/local_smoke/audio/clean_0001.wav`
- `data/local_smoke/audio/noise_0001.wav`
- `data/jsonl/baseline_smoke.local.jsonl`

这些文件只用于本地 smoke test，不进入 git。正式 baseline 仍应使用真实语音或公开数据集样本。

## MVP 150 条本地合成评测集

为了在正式数据源接入前先扩大 baseline 评估闭环，本阶段新增一个本地合成的 MVP 评测集：

- clean：30 条。
- noise：30 条。
- reverb：30 条。
- far_field：30 条。
- dropout：30 条。
- 总计：150 条。
- 每个场景内前 15 条为短文本，后 15 条为长文本。

生成命令：

```bash
python scripts/create_mvp_eval_audio.py --force
```

默认输出：

- `data/mvp_eval/audio/`
- `data/jsonl/baseline_mvp_150.local.jsonl`
- `data/jsonl/baseline_mvp_150_stats.local.json`

该数据集使用 macOS `say` 合成 clean 音频，再用标准库生成 noise、reverb、far_field、dropout 退化版本。它只用于工程闭环、推理稳定性和评测输出验证，不作为真实 benchmark。

默认退化强度为 `hard`。设计目的不是“听起来自然”，而是尽快暴露 baseline 在噪声、混响、远场和丢包场景下的错误。clean 场景仍保持清晰，用于量化 clean speech regression；四个 degraded 场景会主动降低可懂度。

文本长度设计：前 15 条短句用于基础识别和 clean 回归检查，后 15 条长句用于暴露漏词、重复输出、幻觉补全、长上下文顺序错误和 dropout 后恢复失败。

可选强度：

```bash
python scripts/create_mvp_eval_audio.py --profile hard --force
python scripts/create_mvp_eval_audio.py --profile medium --force
python scripts/create_mvp_eval_audio.py --profile mild --force
```

Colab 入口：

```text
notebooks/02_mvp_150_eval_colab.ipynb
```

## 推理接口

baseline 直接使用 Qwen3-ASR 官方 `Qwen3ASRModel.transcribe(audio=..., language=...)` 接口。当前 MVP 音频为英文，默认强制 `language=English`；后续多语言 manifest 可以改为 `--language manifest` 或 `--language auto`。

## 测试标准

- 本地 smoke 生成脚本能产出 clean/noise 两条 wav 和 JSONL manifest。
- MVP 150 生成脚本能产出 5 个场景各 30 条 wav 和 JSONL manifest。
- MVP 150 manifest 每行包含 `audio`、`answer`、`language`、`scenario`、`utterance_id`、`is_degraded`。
- MVP 150 manifest 每行包含 `text_length_bucket` 和 `reference_word_count`。
- MVP 150 manifest 中所有音频路径存在，场景计数准确。
- MVP 150 stats 必须记录 `profile`、`degradation_stats`，用于确认 noise/far_field 的近似 SNR、dropout 静音占比等退化强度。
- MVP 150 stats 必须记录短文本/长文本分布，默认每个场景 short/long 各 15 条。
- 至少 1 条 clean 音频和 1 条 degraded 音频推理成功。
- 输出 JSONL 每行包含 `audio`、`answer`、`prediction`、`scenario`。
- 评测脚本能计算 WER/CER。
- 推理失败时记录错误，不中断整批任务。

## 验收标准

- 至少 50 条样本完成 baseline 评估；资源不足时先完成 10 条 smoke eval。
- 输出 overall WER/CER。
- 输出 scenario-level WER/CER。
- 记录空输出率和明显幻觉样本。
- 文档和进度已更新。

## 当前 Baseline 结果

2026-06-07 曾在 Colab 中完成 2 条样本 smoke eval，用于验证推理与评测链路：

- clean：1 条，error_rate 0.0，empty_output_rate 0.0。
- noise：1 条，error_rate 0.0，empty_output_rate 0.0。

该结果只证明历史 baseline 推理和评测闭环可运行。2026-06-11 切换到 `Qwen/Qwen3-ASR-1.7B` 后，必须重新执行 smoke 和 MVP 150 评测，新的 WER/CER 才能作为当前 baseline。

2026-06-08 已新增 MVP 150 条评测集生成脚本和 Colab 评测 notebook。本地生成与 oracle 评测链路已通过：

- manifest：150 条，clean/noise/reverb/far_field/dropout 各 30 条。
- 音频路径缺失：0 条。
- oracle overall error_rate：0.0。
- oracle scenario-level error_rate：五个场景均为 0.0。

oracle 结果只验证评测链路。真实 Qwen3-ASR baseline 结果需在 Colab 执行 `notebooks/02_mvp_150_eval_colab.ipynb` 后记录。

收到音频质量过高反馈后，已将 MVP 150 默认 profile 调整为 `hard` 并重新生成。当前 hard 退化统计：

- noise：平均近似 SNR -2.2575 dB。
- reverb：平均近似 SNR -0.4661 dB。
- far_field：平均近似 SNR 0.3233 dB，平均 RMS ratio 0.3047。
- dropout：活跃语音接近静音比例 0.6417。

这版数据应更容易压出 baseline 的漏识别、空输出、重复补全或幻觉式输出。

收到长文本覆盖不足反馈后，默认文本已调整为前 15 条短句、后 15 条长句。长句用于扩大 reference word count，并更容易暴露长音频转写中的漏词、重复和幻觉问题。

当前短/长文本校验结果：

- 每个场景：short 15 条，long 15 条。
- short 参考词数：6-8，平均 7.13。
- long 参考词数：21-27，平均 24.73。
- overall oracle ref_len：2395。

2026-06-15 已在 Colab GPU runtime 中完成 `Qwen/Qwen3-ASR-1.7B` MVP 150 hard profile 全量评测。

场景级结果：

| scenario | samples | num_edits | ref_len | WER | empty_output_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 30 | 5 | 479 | 0.010438 | 0.0 |
| noise | 30 | 161 | 479 | 0.336117 | 0.0 |
| reverb | 30 | 199 | 479 | 0.415449 | 0.0 |
| dropout | 30 | 364 | 479 | 0.759916 | 0.0 |
| far_field | 30 | 430 | 479 | 0.897704 | 0.0 |

整体 WER 约 0.483925。degraded-only WER 约 0.602296。所有场景 empty output rate 均为 0.0，说明失败主要来自错误替换、漏词、插入和幻觉式补全，而不是完全空转写。

短/长文本拆分结果：

| scenario | bucket | samples | edits | ref_len | WER |
| --- | --- | ---: | ---: | ---: | ---: |
| clean | short | 15 | 0 | 107 | 0.000000 |
| clean | long | 15 | 5 | 372 | 0.013441 |
| noise | short | 15 | 21 | 107 | 0.196262 |
| noise | long | 15 | 140 | 372 | 0.376344 |
| reverb | short | 15 | 31 | 107 | 0.289720 |
| reverb | long | 15 | 168 | 372 | 0.451613 |
| dropout | short | 15 | 77 | 107 | 0.719626 |
| dropout | long | 15 | 287 | 372 | 0.771505 |
| far_field | short | 15 | 92 | 107 | 0.859813 |
| far_field | long | 15 | 338 | 372 | 0.908602 |

验收判断：

- 已满足 baseline MVP 的样本数、场景级指标、空输出率和失败样本记录要求。
- clean baseline 正常，可以作为后续 clean regression 对照。
- degraded 场景退化足够明显，尤其 far_field/dropout，可作为 LoRA 和 router 的第一批优化目标。
- 下一步应进入错误分析文档化，并准备 `05 LoRA 训练 MVP` 的训练数据与 target 探测。

## 风险

- Qwen3-ASR 模型下载可能需要 Hugging Face 登录或稳定网络。缓解：notebook 中保留登录步骤和错误提示。
- Colab 显存不足。缓解：先用小样本、`float16`、`max_inference_batch_size=1`，必要时使用更高规格 runtime。
- Colab 预装依赖被升级导致冲突。缓解：notebook 中不安装未固定版本的 `pandas`，如需修复则固定 `pandas==2.2.2`。
- 输出包含解释文本或重复文本。缓解：记录为 baseline failure，后续通过数据、微调和输出解析处理。
