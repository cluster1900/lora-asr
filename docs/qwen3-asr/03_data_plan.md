# 数据方案

## 背景与范围

第一轮使用公开数据快速形成可复现闭环，不生成自有训练语音。训练覆盖 clean、noise、reverb、
far_field、dropout 及 Voices-in-the-Wild 的其余场景；固定 benchmark 不进入训练。

## 来源与切分

- Robust train/validation：`zhifeixie/Voices-in-the-Wild-2M`，160k/8k。
- Robust test：`zhifeixie/Voices-in-the-Wild-Bench`，5k。
- English clean：LibriSpeech，train 20k、validation 1k，另保留官方 test-clean。
- Chinese clean：AISHELL-1，train 20k、validation 1k，另保留官方 test。

revision、license、split 和配额固定在 `configs/data/public_robust_200k.yaml`。总计 200k train、
10k validation、512 canary、30k curriculum 和 5k robust test。

## Manifest

每行 JSON 必须含音频路径、目标文本、语言、场景、真实/合成来源、数据集/revision/split/index、
source utterance ID、时长、license、seed 和音频 SHA-256。选择由固定 seed 决定，与输入行顺序无关。

音频存 Colab 本地 SSD；Google Drive 保存 shard、manifest、hash 和结果。大音频、缓存和生成物不
提交 Git。

## 质量检查

- 音频存在、可读、0.5-30 秒，目标文本非空。
- train/validation/test 的 source ID 和 audio hash 不重叠。
- 行数、语言和 scenario 配额完全匹配配置。
- 正式 `build/validate` 必须检查音频存在或可解码，不提供跳过行数合同的开关。
- 单条错误进入 rejects；正式 build 有任何硬错误即失败。
- curriculum 只使用 `base_error_rate < 0.70` 的 BF16 base 评分，不用预测文本替代 gold label。

## 测试与验收

先用 128-row smoke 覆盖全部 robust split 与两种语言，再运行正式 build。验收产物包括各 manifest、
sources、stats、validation report、rejects 和 SHA-256。未完成真实音频 staging 前，数据阶段不算完成。

## 影响

本次精简删除历史本地 TTS/MVP 数据。后续测试不得依赖已提交音频，测试 fixture 应在临时目录构造。
