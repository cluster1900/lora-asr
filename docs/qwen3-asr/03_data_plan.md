# 数据方案

## 背景与范围

第一轮使用公开数据快速形成可复现闭环，不生成自有训练语音。训练覆盖 clean、noise、reverb、
far_field、dropout 及 Voices-in-the-Wild 的其余场景；固定 benchmark 不进入训练。

## 来源与切分

- Robust train/validation：`zhifeixie/Voices-in-the-Wild-2M`，160k/8k。
- Robust test：`zhifeixie/Voices-in-the-Wild-Bench`，5k。
- English clean：LibriSpeech，train.100 取 20k、validation 取 1k。
- Chinese clean：AISHELL-1 的 pinned Parquet 重打包，train 20k、validation 1k；底层语料为
  OpenSLR SLR33 Apache-2.0。

revision、license、split 和配额固定在 `configs/data/public_robust_200k.yaml`。总计 200k train、
10k validation、512 canary、30k curriculum 和 5k robust test。

## 唯一 staging 合同

`scripts/prepare_public_robust_manifests.py stage` 是唯一远端数据入口。它使用 pinned revision 和
`datasets` streaming，关闭自动音频解码，只下载达到配额所需的行并把音频物化到
`/content/qwen3-asr-runtime/data`。不下载完整 Hub snapshot，不生成第二套 shard 格式。

- `--mode smoke`：每个 robust split 按语言各 1 条、两种 clean 各 10 条、Bench 的
  language x real/synthetic 各 1 条；shuffle buffer 固定为 64，避免为 smoke 预取数万条音频。
- `--mode full`：只 staging 168k robust candidate、42k clean candidate 和完整 5k Bench。
- Robust 使用跨 scenario 的 base source identity 做固定 90/10 train/validation 分区，再按
  scenario x language 配额取样，避免同一原始话语跨 train/validation。
- Clean train 只来自配置中的官方 `train_split`，clean validation 只来自官方
  `validation_split`，不从混合池重新切分。
- 候选音频路径相对 `data_root`；候选 JSONL、rejects 和 `stage_report.json` 写到指定
  candidate directory。每 100 条及每个 split 结束时 flush+fsync；重跑时仅复用音频存在且
  SHA-256 一致的行。

Colab 本地 SSD 在 runtime 重启后会清空。此时重跑同一命令会删除 candidate manifest 中已经失去
本地音频的进度行并重新下载，不会把缺失路径当作已完成。该语义保证结果正确；它不承诺跨 runtime
免下载恢复。

## Manifest

每行 JSON 必须含音频路径、目标文本、语言、场景、真实/合成来源、数据集/revision/split/index、
source utterance ID、时长、license、seed 和音频 SHA-256。选择由固定 seed 决定，与输入行顺序无关。

音频存 Colab 本地 SSD；Google Drive 保存 candidate manifest、正式 manifest、hash 和结果。大音频、
缓存和生成物不提交 Git。

## 质量检查

- 音频存在、可读、0.5-30 秒，目标文本非空。
- train/validation/test 的 source ID 和 audio hash 不重叠；Robust 的 source ID 不包含 scenario。
- 行数、语言和 scenario 配额完全匹配配置。
- 正式 `build/validate` 必须检查音频存在或可解码，不提供跳过行数合同的开关。
- 单条错误进入 rejects；正式 build 有任何硬错误即失败。
- curriculum 只使用 `base_error_rate < 0.70` 的 BF16 base 评分，不用预测文本替代 gold label。

## 测试与验收

先用 128-row smoke 覆盖全部 robust split 与两种语言，再运行正式 build。staging 单测必须覆盖
中断续跑、丢失音频修复、跨 scenario source identity 和官方 clean split 隔离。验收产物包括四份
candidate、stage report、各正式 manifest、sources、stats、validation report、rejects 和 SHA-256。
未完成真实音频 staging 前，数据阶段不算完成。

## 影响

本次精简删除历史本地 TTS/MVP 数据。后续测试不得依赖已提交音频，测试 fixture 应在临时目录构造。
