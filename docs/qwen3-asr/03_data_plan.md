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

Robust 使用固定 90/10 source identity 分区；clean 使用官方 train/validation split。数据 builder 尚未
实现，编码时必须先完成 pinned revision、source identity、hash 和恢复合同，再开始训练代码。

## 角色配额

| 角色 | Robust degraded | English clean | Chinese clean | 用途 |
|---|---:|---:|---:|---|
| `sft_train` | 120,000 | 16,000 | 16,000 | SFT |
| `dpo_pool` | 16,000 | 2,000 | 2,000 | 生成 chosen/rejected |
| `rl_pool` | 16,000 | 1,000 | 1,000 | RL rollout/reward |
| `validation` | 8,000 | 1,000 | 1,000 | 全阶段固定评测 |
| `bench_test` | 5,000 | — | — | 只做最终 test |

Pilot 子集固定为：SFT `5,000+1,000+1,000`，DPO `2,000+500+500`，RL `2,000+500+500`。
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

DPO pair 只从 `dpo_pool` 生成。候选来自 base、SFT 和受控 negative；gold transcript 只用于离线
WER/CER 选 pair，不作为 DPO prompt 输入。每行至少包含 `sample_id`、`audio`、`language`、`prompt`、
`chosen`、`rejected`、`preference_source`、`judge`、双方 error rate、来源 revision、source identity、
音频 hash 和 seed。ties、空候选、两边相同、音频失败或泄漏行必须进入 rejects。

full 目标为 20,000 pairs，另有不重叠的 2,000 pair preference validation。DPO 训练不得读取 gold 或
error-rate 审计字段。

## RL 输入

RL 只从 `rl_pool` 取音频和 reference text，由 `dpo_release` 生成 on-policy rollouts。每条 rollout
记录 policy checkpoint、rollout seed、prediction、reference error rate、reward components、final reward、
KL 和 error。reward 固定为：

- `1 - min(WER/CER, 1.0)`；
- empty `-0.25`；repeat `-0.25`；too-long `-0.15`；hallucination `-0.25`；
- final reward clip 到 `[-1, 1]`。

full RL 目标为 18,000 audio，另有不重叠 rollout validation。reward、rollout、KL 和 policy checkpoint
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
