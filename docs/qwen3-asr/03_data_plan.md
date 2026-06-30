# 数据方案

最后更新：2026-06-30

## 目标

构建一个覆盖真实鲁棒性场景、同时适合 Colab MVP 训练的小规模语音数据集。Mega-ASR 的场景设计可作为灵感，但我们的 manifest、增强代码和场景体系都应由本项目维护。

## 数据格式

训练 JSONL：

```json
{
  "audio": "/content/drive/MyDrive/qwen3-asr/audio/train/000001.wav",
  "answer": "THE TRANSCRIPT TEXT",
  "language": "English",
  "scenario": "clean",
  "source": "librispeech",
  "is_degraded": false
}
```

评测 JSONL：

```json
{
  "audio": "/content/drive/MyDrive/qwen3-asr/audio/test/000001.wav",
  "answer": "THE TRANSCRIPT TEXT",
  "language": "en",
  "scenario": "noise_reverb",
  "source": "librispeech",
  "is_degraded": true
}
```

## MVP 数据规模

建议第一版规模：

- Train：10k-20k 条。
- Validation：1k-2k 条。
- Test：1k-2k 条。

音频长度：

- MVP：3-20 秒。
- 后续：加入 20-120 秒长音频评测。

## 候选数据源

Clean speech：

- LibriSpeech
- Common Voice
- AISHELL-1，中文
- WenetSpeech，后续中文扩展

噪声与环境素材：

- MUSAN
- DNS Challenge noise
- ESC-50
- UrbanSound8K
- room impulse response datasets

## 声学场景

### 原子条件

1. Clean speech
2. 背景噪声
3. 远场语音
4. 遮挡或闷声
5. 回声与混响
6. 录音设备伪影
7. 电子失真
8. 传输丢包或 dropout

Mega-ASR 描述了 7 类原子声学条件。我们的实现中会把 dropout 从录音/电子伪影中拆出来，方便生成和分析。

### 复合条件

示例：

- noise + reverb
- far-field + noise
- far-field + reverb
- obstruction + noise
- clipping + noise
- codec artifact + dropout
- restaurant noise + far-field + reverb

## 增强配方

### 噪声

参数：

- SNR：-5、0、5、10、15 dB。
- 噪声类型：人声嘈杂、街道、音乐、咖啡馆、办公室、车辆。

### 远场

参数：

- 音量衰减。
- RIR 卷积。
- 低通滤波。
- 必要时做轻微 stereo-to-mono collapse。

### 遮挡

参数：

- 低通截止频率：1.5-4 kHz。
- 高通截止频率：100-300 Hz。
- 随机 EQ notch。

### 回声与混响

参数：

- RIR 卷积。
- 合成 echo delay：80-250 ms。
- echo decay：0.2-0.6。

### 录音伪影

参数：

- 重采样到 8k/12k/16k/24k 后再回到 16k。
- 可用时加入 MP3/Opus 压缩。
- 麦克风 EQ。
- 错误增益归一化。

### 电子失真

参数：

- clipping 阈值。
- bit depth reduction。
- saturation。
- quantization noise。

### Dropout

参数：

- 随机静音片段：20-300 ms。
- packet loss rate：1%-15%。
- burst dropout。

## 切分规则

避免泄漏：

- 同一条原始 clean utterance 不得同时出现在 train 和 test 中，即使增强方式不同。
- 如果可行，同一 speaker 应只出现在一个 split。
- test 使用的噪声素材不应在 train 中复用。

## 元数据要求

每条生成样本应记录：

- 原始音频路径。
- 生成音频路径。
- 转写文本。
- 源数据集。
- speaker id，如果有。
- 语言。
- scenario。
- 增强参数。
- 随机种子。

## 第一版推荐

MVP 建议：

- 先做英文。
- 使用 LibriSpeech clean subset。
- 5 类退化：noise、reverb、far-field、clipping、dropout。
- 每条 clean utterance 生成 2-3 条 degraded 样本。
- 保留固定 held-out test set。

## LoRA MVP Bootstrap 数据

### 背景

`05C` 已经证明 Qwen3-ASR 可以挂载 PEFT LoRA 并完成 20 step smoke training。
正式 LoRA MVP 不能继续直接使用 MVP 150 评测集训练，否则后续 base-vs-LoRA
对比会失去 held-out 意义。因此需要新增一批独立 train/val manifest，用来启动
第一版监督训练闭环。

### 范围

本批 bootstrap 数据只用于 LoRA MVP 启动：

- 做 train/val，不做最终 test。
- 默认覆盖 clean、noise、reverb。
- 使用本项目合成音频和退化代码。
- 保留固定随机种子、文本 index、base utterance id 和增强参数。

本批数据不用于声明产品级鲁棒性，不替代真实语音数据或真实 noisy holdout。

### 默认配置

- train：clean、noise、reverb 各 120 条。
- val：clean、noise、reverb 各 30 条。
- profile：`medium`，避免一开始只学习 hard profile 的极端伪影。
- seed：`20260629`。
- held-out test：继续使用 `data/jsonl/baseline_mvp_150.local.jsonl` 和 `outputs/baseline_mvp_150/` 中的 base 指标。

默认输出：

- `data/lora_mvp/audio/train/`
- `data/lora_mvp/audio/val/`
- `data/jsonl/lora_mvp_train.local.jsonl`
- `data/jsonl/lora_mvp_val.local.jsonl`
- `data/jsonl/lora_mvp_stats.local.json`

### 数据格式

每条样本至少包含：

```json
{
  "audio": "data/lora_mvp/audio/train/noise/noise_0001.wav",
  "answer": "Please confirm the meeting room before the weekly review begins.",
  "language": "en",
  "scenario": "noise",
  "split": "train",
  "source": "macos_say_plus_noise",
  "is_degraded": true,
  "utterance_id": "train_utt_0001_noise",
  "base_utterance_id": "train_utt_0001",
  "degradation": "noise",
  "profile": "medium",
  "seed": 20260730,
  "sample_rate": 16000,
  "text_length_bucket": "short",
  "reference_word_count": 9
}
```

### 切分与泄漏规则

- train 与 val 的 `base_utterance_id` 不得重叠。
- train/val 不得包含 `baseline_mvp_150` 的音频路径。
- 同一个 split 内允许同一 base utterance 生成 clean、noise、reverb 多个场景，因为这是训练鲁棒映射的目标；但同一个 base utterance 不跨 split。
- held-out MVP 150 不进入训练和调参，只用于最终 base-vs-LoRA 对比。

### 生成命令

```bash
python3 scripts/create_lora_mvp_dataset.py \
  --profile medium \
  --train-items-per-scenario 120 \
  --val-items-per-scenario 30 \
  --scenarios clean,noise,reverb \
  --force
```

### 测试

- JSONL 每行可解析。
- 每条样本的音频路径存在。
- train/val scenario counts 与命令行参数一致。
- train/val `base_utterance_id` 无交集。
- stats 文件记录 seed、profile、split counts、scenario counts、duration 和退化质量统计。

### 验收

- Colab/本地可复现生成同一批 manifest。
- LoRA 训练配置引用 train manifest，验证或后续 early stopping 使用 val manifest。
- 固定 MVP 150 held-out test 与 bootstrap train/val 明确分离。

## Scale-Up 数据设计

### 背景

v3/v4/v5 的结果说明，当前 medium-profile bootstrap 数据无法支撑目标鲁棒性。
v5 增加 audio MLP target 后仍未超过 v3，说明下一步应优先补数据难度和场景覆盖，
再扩大 target 或训练 text decoder。

### 数据分层

Tier 1：hard-profile MVP train/val

- train：2k-5k 条。
- val：300-500 条。
- 场景：clean、noise、reverb、dropout、far_field、noise_reverb、far_field_noise。
- 用途：v6A/v6B，验证 hard-profile 数据对齐和 mixed constraint。

Tier 2：多源真实语音 + 合成退化

- train：20k-50k 条。
- val/test：2k-5k 条。
- clean 来源：LibriSpeech、Common Voice；中文阶段再加入 AISHELL/WenetSpeech。
- noise/RIR 来源：MUSAN、DNS Challenge、ESC-50、UrbanSound8K、公开 RIR。
- 用途：降低 macOS TTS 和固定合成伪影过拟合。

Tier 3：复合场景扩展

- train：100k+ 条。
- 场景：从 7 类原子条件扩展到 20-50 个复合场景。
- 用途：向 Mega-ASR 的 full-scenario robust ASR 能力形态靠近。

### 难度标注

从 v6 开始，训练 manifest 需要先经过 base inference 生成 difficulty manifest。
每条样本新增：

- `base_prediction`
- `base_wer`
- `difficulty_bucket`
- `failure_tags`
- `approx_snr_db`
- `rms_ratio`
- `active_near_silence_ratio`

推荐 difficulty bucket：

- `wer_0_10`
- `wer_10_30`
- `wer_30_50`
- `wer_50_70`
- `wer_70_plus`

### 采样原则

- v6 以 `wer_10_30` 和 `wer_30_50` 为主。
- `wer_50_70` 少量加入，用于提高 hard robustness。
- `wer_70_plus` 默认只做观察或小比例 hard negative，避免小数据阶段学到幻觉补全。
- clean retention 不低于 10%-15%。
- dropout/far_field 作为约束样本加入，不再只放在 held-out 观察中。

## 测试标准

- JSONL 每行都是合法 JSON。
- 每条样本的 `audio` 路径存在。
- 对于 Colab 训练，manifest 中的音频路径必须在 Google Drive 项目目录中真实存在；`outputs/baseline_mvp_150/baseline_mvp_150.colab.jsonl` 只保存路径和标签，不包含 wav 音频本体。
- 训练样本包含 `text`，评测样本包含 `answer`。
- 每条样本包含 `language`、`scenario`、`source`、`is_degraded`。
- train/val/test 不存在同一原始 utterance 泄漏。
- scenario 分布、source 分布和总时长能生成统计报告。

## 验收标准

- smoke、train、val、test 四类 manifest 都已生成。
- MVP 至少覆盖 clean、noise、reverb、far_field、dropout。
- 每个 split 都有数据统计文件。
- 数据构建配置和随机种子已保存。
- 测试集固定，不被训练或调参使用。
