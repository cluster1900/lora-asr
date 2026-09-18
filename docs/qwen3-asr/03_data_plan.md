# 数据方案

## 背景与范围

完整后训练使用同一批公开语音的隔离角色，按 `source_utterance_id` 和固定 seed 分成 SFT、DPO、RL、
validation 和 test。任何阶段都不得读取 test；DPO/RL 的训练输入不能跨角色重复。

## 固定数据源

| source_dataset | revision | split | 角色 | 许可证记录 |
|---|---|---|---|---|
| `zhifeixie/Voices-in-the-Wild-2M` | `a8a35d3319737190d6fd3d39157b258eaab35980` | pinned robust train/validation | degraded SFT/DPO/RL | 保存源 license 和原始字段 |
| `openslr/librispeech_asr` | `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1` | `train.100`、`validation` | English clean SFT/DPO/RL | CC-BY-4.0 |
| `knoveleng/aishell1-mandarin` | `c6dde006238091dc7c81cf5888208140bba33cba` | `train`、`validation` | Chinese clean SFT/DPO/RL | Apache-2.0，保留 OpenSLR SLR33 来源 |
| `zhifeixie/Voices-in-the-Wild-Bench` | `788f5d72c6b0e9091b5c2e432370923b6f9f0660` | pinned test | 最终评测 | 永不进入训练 |

Robust 使用固定 90/10 source identity 分区；clean 使用官方 train/validation split 严格隔离供给：
官方 train split（LibriSpeech `train.100`、AISHELL-1 `train`）专供训练角色（`sft_train` 16k + `dpo_train_pool` 10k + `rl_train_pool` 1k = 27,000 条，LibriSpeech `train.100` 留 1,539 条清洗缓冲）；
官方 validation split（LibriSpeech `validation`、AISHELL-1 `validation`）专供验证角色（`validation` 1,000 + `dpo_val_pool` 1,000 + `rl_val_pool` 200 = 2,200 条），二者物理隔离、严禁串用。
数据 builder 尚未实现，编码时必须先完成 pinned revision、source identity、hash 和恢复合同，再开始训练代码。

## 下载后的处理链

原始下载完成后必须经过以下顺序，原始文件不可覆盖：

```text
raw source snapshot
  -> integrity/license/source report
  -> audio normalization: readable, mono, 16 kHz, WAV/PCM, 0.5–30 s
  -> transcript/language/scenario normalization
  -> source identity, duplicate and leakage checks
  -> canonical processed rows + audio_sha256
  -> disjoint SFT/DPO/RL/validation/Bench role pools
  -> model-dependent DPO pairs and RL rollouts
```

- Robust：解析 source utterance identity、scenario、real/synthetic origin 和原始 transcript；同一原始话语的不同 degraded scenario 必须共享 identity。
- LibriSpeech：只使用 `train.100`/`validation` 规定 split，保留 speaker/chapter identity，统一 English 文本规范化。
- AISHELL-1：只使用 `train`/`validation` 规定 split，保留 utterance identity，统一 Chinese 文本规范化。
- Bench：执行同样的音频可读性和 schema 处理，但只写入 `bench_test`，不能参与 role pool、DPO pair 或 RL rollout。
- 音频处理必须记录原始 hash、处理后 hash、采样率、声道、时长和处理版本；失败行写入 rejects，不得静默丢弃。
- 文本清洗不能改变 gold 语义；normalized text 用于评测和 pair 选择，原始 text 必须保留在审计产物中。

处理阶段产物必须有 `RAW_COMPLETE.json`、`PROCESSED_COMPLETE.json`、每个 role 的 `COMPLETE.json` 和总的
`DATASET_COMPLETE.json`。只有总门禁存在，才允许下载模型并进入 base smoke。

## 角色配额

| 角色 | Robust degraded | English clean | Chinese clean | 用途 |
|---|---:|---:|---:|---|
| `sft_train` | 120,000 | 16,000 | 16,000 | SFT 训练 |
| `dpo_train_pool` | 32,000 source → 16,000 pairs | 10,000 source → 2,000 pairs | 10,000 source → 2,000 pairs | DPO 训练 pairs，源音频不重叠 |
| `dpo_val_pool` | 3,200 source → 1,600 pairs | 1,000 source → 200 pairs | 1,000 source → 200 pairs | 独立 DPO 验证 pairs，源音频不重叠 |
| `rl_train_pool` | 16,000 | 1,000 | 1,000 | RL rollout 训练（16,000 robust + 1,000 en + 1,000 zh） |
| `rl_val_pool` | 1,600 | 200 | 200 | 独立 RL rollout 验证（1,600 robust + 200 en + 200 zh） |
| `validation` | 8,000 | 1,000 | 1,000 | 全阶段固定 ASR 评测 |
| `bench_test` | 5,000 | — | — | 只做最终 test |

注：Clean 语音因为模型识别准确率高，Base 与 SFT 预测文本存在较高的完全一致率（平局率）。
Clean source pool 扩大到目标 pair 数的 5 倍，Robust source pool 至少扩大到 2 倍；每条源音频可生成多个受控候选。
过滤后有效 pair 不足表中目标时，数据阶段失败，不得从 validation/test 补行。

Pilot 子集固定为：SFT `5,000+1,000+1,000`，DPO `2,000+500+500`（验证集 500 对），RL `2,000+500+500`（验证音频 500 条）。
角色之间 source identity 不重叠；不足配额时阶段失败，不允许从 validation/test 补行。

## SFT 输入

每行至少包含：

```json
{
  "sample_id": "stable-id",
  "audio": "audio.wav",
  "text": "gold transcript",
  "language": "en|zh",
  "scenario": "clean|noise|reverb|far_field|dropout|...",
  "condition_group": "clean|degraded",
  "audio_origin": "real|synthetic",
  "source_dataset": "dataset-id",
  "source_revision": "revision",
  "source_split": "split",
  "source_index": 123,
  "source_utterance_id": "stable-source-id",
  "duration_s": 3.2,
  "license": "license-id",
  "seed": 20260722,
  "audio_sha256": "sha256"
}
```

`sample_id` 唯一，音频可解码，文本非空，时长 0.5–30 秒。旧 `answer` 只允许一次性转换成 `text`，
不得让两个字段长期并存。

## DPO 输入

DPO pair 从 `dpo_train_pool` 与 `dpo_val_pool` 生成；`dpo_val_pool` 只用于 preference validation，不进入 DPO train。候选来自 base、SFT 和受控 negative；gold transcript 只用于离线
WER/CER 选 pair，不作为 DPO prompt 输入。每行至少包含 `sample_id`、`audio`、`language`、`prompt`、
`chosen`、`rejected`、`preference_source`、`judge`、双方 error rate、来源 revision、source identity、
音频 hash 和 seed，并支持离线预计算的 `ref_chosen_logp` 与 `ref_rejected_logp`。预计算文件还必须包含参考模型 revision/hash、tokenizer/chat-template hash、
logp 配置 hash；任一 provenance 不匹配不得进入 DPO。
ties、空候选、两边相同、音频失败或泄漏行必须进入 rejects。

full 目标为 20,000 pairs 训练集，另有独立不重叠的 2,000 pairs preference validation。DPO 训练不得读取 gold 或
error-rate 审计字段。

## RL 输入

RL 只从 `rl_train_pool` 与 `rl_val_pool` 取音频和 reference text，以 DPO 合并模型为起点，采用 GRPO 算法（候选组大小 $G=4$，采样 `temperature=0.7`, `top_p=0.9`）生成 rollouts。每条 rollout
记录 policy checkpoint、`group_id`、`group_size=4`、`rollout_rank`、rollout seed、prediction、reference error rate、
reward components、final reward、KL 和 error；advantage 计算前必须完成同组 all-gather。reward 固定为：

- `asr reward = 1 - min(WER/CER, 1.0)`；
- empty `-0.25`；repeat `-0.25`；too-long `-0.15`；hallucination `-0.25`；
- final reward clip 到 `[-1, 1]`；
- 策略损失引入针对冻结参考策略（`dpo_release`）的标准 KL 正则约束 $\beta D_{KL}(\pi_\theta || \pi_{\text{ref}})$。

full RL 目标为 18,000 audio 训练集，另有独立不重叠的 2,000 audio rollout validation。reward、rollout、KL 和 policy checkpoint
任一缺失，RL 阶段不算完成。

## 存储与恢复

所有物化音频、manifest、rejects、缓存和结果位于 `/data/mega-asr/`；不读取 `/data/mini-k3`。
每份 manifest 必须配套行数、source revision、source identity hash、audio hash、seed 和 manifest sha256。
中断后只能复用 hash 一致的行；损坏或缺失音频必须重新 staging。

## 数据阶段验收

数据阶段通过必须同时满足：

1. 四个 source revision 和许可证记录完整；
2. SFT、DPO、RL、validation、Bench 角色配额准确；
3. source identity、audio hash 和 sample_id 无跨角色泄漏；
4. 所有音频可解码，文本非空，时长在范围内；
5. DPO pair 可审计，chosen/rejected 不相同且无 ties；
6. RL rollout schema fixture 能记录 reward、KL 和错误；
7. manifest、rejects、sources、stats、hash 和恢复报告全部存在。

数据 builder 未完成以上产物前，不得开始 SFT、DPO 或 RL 正式训练。
