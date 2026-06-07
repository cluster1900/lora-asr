# 数据方案

最后更新：2026-06-07

## 目标

构建一个覆盖真实鲁棒性场景、同时适合 Colab MVP 训练的小规模语音数据集。Mega-ASR 的场景设计可作为灵感，但我们的 manifest、增强代码和场景体系都应由本项目维护。

## 数据格式

训练 JSONL：

```json
{
  "audio": "/content/drive/MyDrive/gemma-mega-asr/audio/train/000001.wav",
  "text": "language English<asr_text>THE TRANSCRIPT TEXT",
  "prompt": "Transcribe the speech accurately.",
  "language": "English",
  "scenario": "clean",
  "source": "librispeech",
  "is_degraded": false
}
```

评测 JSONL：

```json
{
  "audio": "/content/drive/MyDrive/gemma-mega-asr/audio/test/000001.wav",
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

## 测试标准

- JSONL 每行都是合法 JSON。
- 每条样本的 `audio` 路径存在。
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
