# 测试方案

最后更新：2026-06-15

## 目标

评估 Qwen3-ASR ASR LoRA 是否能提升退化音频识别，同时不明显损害 clean speech。

## 测试模式

1. Base Qwen3-ASR-1.7B。
2. Qwen3-ASR-1.7B + LoRA always on。
3. Qwen3-ASR-1.7B + router conditional LoRA。
4. 可选外部 baseline：Mega-ASR 发布模型、Whisper、商业 ASR API。

外部 baseline 只作为对比对象，不作为实现依赖。

## 核心指标

### ASR 质量

- 英文 WER。
- 中文 CER。
- 按 scenario 聚合的 WER/CER。
- clean audio regression。
- 相对 base 的提升。

### 鲁棒性失败指标

- empty output rate。
- repeated output rate。
- hallucination-like output rate。
- excessive length ratio。
- 关键样本 missing keyword rate。

### 运行指标

- load time。
- inference time。
- real-time factor。
- peak VRAM，如果可获取。
- average tokens generated。

## 评测 JSONL

输入：

```json
{
  "audio": "/path/to/audio.wav",
  "answer": "reference transcript",
  "language": "en",
  "scenario": "noise_reverb",
  "is_degraded": true
}
```

输出：

```json
{
  "audio": "/path/to/audio.wav",
  "answer": "reference transcript",
  "prediction": "model transcript",
  "language": "en",
  "scenario": "noise_reverb",
  "metric": "wer",
  "wer": 0.1234,
  "num_edits": 3,
  "ref_len": 24,
  "empty_output": false,
  "length_ratio": 0.95,
  "mode": "lora"
}
```

## 场景桶

MVP 至少包含：

- clean
- noise
- reverb
- far_field
- clipping
- dropout
- noise_reverb
- far_field_noise

## MVP 成功标准

第一版 MVP 成功条件：

- 至少一个 degraded 场景 WER 相对下降 10%。
- router 模式下 clean WER 相对退化小于 5%。
- degraded 音频空输出率下降。
- 评测脚本可复现。

当前 Qwen3-ASR base 对照：

- clean WER：0.010438。
- noise WER：0.336117。
- reverb WER：0.415449。
- dropout WER：0.759916。
- far_field WER：0.897704。
- degraded-only WER：约 0.602296。
- empty output rate：所有场景均为 0.0。

LoRA MVP 的第一版成功门槛应以 degraded-only WER 或单场景 WER 相对 base 下降为准。由于 base 没有空输出，短期内更应关注错误替换、漏词、插入、重复和幻觉式补全是否减少。

## Scale-Up 成功标准

扩大规模版本成功条件：

- degraded 平均 WER 相对 Qwen3-ASR base 下降 15%-25%。
- router 模式下 clean WER 相对退化小于 3%。
- 在混合 clean/degraded 测试集上，router 模式优于 always-base 和 always-LoRA。
- 结果能在至少两个源数据集上成立。

## 人工错误分析

每次评测后抽样：

- 20 个提升最大样本。
- 20 个退化最严重样本。
- 20 个空输出或近似空输出样本。
- 20 个长幻觉样本。
- 20 个 clean speech regression 样本。

每个样本记录：

- 音频路径。
- reference。
- base prediction。
- LoRA prediction。
- scenario。
- suspected failure type。

## 回归测试

接受新训练结果前必须检查：

1. adapter 可成功加载。
2. clean 样本可正常推理。
3. degraded 样本可正常推理。
4. 10 条 mini eval 可计算 WER。
5. 无输出解析错误。
6. checkpoint metadata 存在。
