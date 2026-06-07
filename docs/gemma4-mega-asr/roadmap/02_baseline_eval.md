# 02 Baseline 评估

最后更新：2026-06-07

## 背景

训练前必须先知道 Gemma 4 12B 原始 ASR 能力。没有 baseline，后续 LoRA、数据增强和 router 的效果都无法判断。

## 目标

在 Colab 上跑通 Gemma 4 12B baseline 推理，并输出可复现的 WER/CER 指标。

## 范围

本步骤只评估 base model，不训练 LoRA，不做 router。

## 输入

- `data/jsonl/baseline_smoke.jsonl`
- baseline 配置文件。
- 少量 clean/degraded 音频。

评测 JSONL 格式：

```json
{
  "audio": "/content/drive/MyDrive/gemma-mega-asr/audio/test/000001.wav",
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

## 需要实现的文件

- `notebooks/01_baseline_colab.ipynb`
- `inference/gemma4_base_infer.py`
- `evaluation/eval_wer.py`
- `configs/baseline/gemma4_baseline.yaml`

## 执行步骤

1. 在 Colab 挂载 Google Drive。
2. 安装 Transformers、音频处理、评测依赖。
3. 加载 `google/gemma-4-12B-it`。
4. 读取 baseline smoke JSONL。
5. 对每条音频生成转写。
6. 保存 prediction JSONL。
7. 计算 WER/CER 和 scenario-level 指标。
8. 抽样记录失败样本。
9. 更新进度文档。

## Prompt 初版

```text
Transcribe the following speech segment in its original language. Only output the transcription.
```

## 测试标准

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

## 风险

- Gemma 4 模型需要授权或登录 Hugging Face。缓解：notebook 中加入登录步骤和错误提示。
- Colab 显存不足。缓解：先用小样本、量化加载，必要时使用 A100 runtime。
- 输出包含解释文本。缓解：记录为 baseline failure，后续通过 prompt 和 SFT 处理。

