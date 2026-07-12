# 数据方案

最后更新：2026-07-12

## 背景

历史数据只有 120 条独立 TTS train source，增加退化副本不能替代 speaker、文本和真实
声学多样性。第一轮正式训练直接使用公开鲁棒数据，不再开发自建 noise/RIR 工厂。

## 范围

本轮只准备：

- 200k train。
- 10k validation。
- Voices-in-the-Wild-Bench 5k fixed test。
- LibriSpeech test-clean 与 AISHELL-1 test clean regression。

不做全量 base WER difficulty scoring，不调用 GPT-5.5 teacher，不生成新的合成退化。

## 数据源

| 用途 | 数据源 | 当前事实 |
| --- | --- | --- |
| robust train/val | `zhifeixie/Voices-in-the-Wild-2M` | 645,925 条，54 split，约 197.5 GB，Apache-2.0，非 gated |
| English clean | LibriSpeech | train-clean 子集用于 retention，test-clean 用于固定 test |
| Chinese clean | AISHELL-1 | train/dev 用于 retention/validation，test 用于固定 test |
| robust test | `zhifeixie/Voices-in-the-Wild-Bench` | 5,000 条，中英、real/synthetic、8 类 condition |

每个数据源必须 pin revision 或发布版本，并把 license、revision 和 source split 写入
manifest 与统计文件。

## 固定配比

### Train 200k

- Robust 160k：7 个 atomic condition 各 16k，共 112k；compound 48k。
- English clean 20k。
- Chinese clean 20k。
- Robust 各 condition 内 English/Chinese 尽量 1:1，任何配额不足必须硬失败。

### Validation 10k

- Robust 8k：按 language 与 8 类 condition 分层。
- English clean 1k。
- Chinese clean 1k。

Validation 必须包含 clean，因为 50%/100% checkpoint 要在不触碰固定 test 的前提下检查
clean regression。

## Canonical JSONL

训练、validation 和 test 统一使用一个 schema：

```json
{
  "sample_id": "vitw2m:noise:123",
  "audio": "audio/train/shard-0001/sample-000123.flac",
  "answer": "reference transcript",
  "language": "en",
  "scenario": "noise",
  "condition_group": "atomic",
  "audio_origin": "synthetic",
  "source_dataset": "zhifeixie/Voices-in-the-Wild-2M",
  "source_revision": "a8a35d3319737190d6fd3d39157b258eaab35980",
  "source_split": "noise",
  "source_index": 123,
  "source_utterance_id": "derived-stable-id",
  "speaker_id": null,
  "duration_s": 7.42,
  "license": "apache-2.0",
  "seed": 20260711,
  "audio_sha256": "..."
}
```

字段约定：

- `answer` 是唯一 reference 字段；不再同时维护 `text` 和 `answer`。
- `language` 只允许 `en` 或 `zh`。Trainer 在运行时构造
  `language English<asr_text>...` 或 `language Chinese<asr_text>...`。
- `scenario` 使用标准 condition 名称；`condition_group` 只允许
  `clean|atomic|compound`。
- `audio_origin` 在 Bench 中使用 `real|synthetic`，clean test 使用 `clean`。
- `source_dataset` 取代历史 `source` 字段。
- `audio` 是相对 `data_root` 的路径；训练前解析后的真实文件必须存在。

Voices-in-the-Wild-2M 当前没有显式 language 字段。语言由 gold transcript 使用固定规则
推断；中英文混合或无法确定的样本写入 rejects。不得调用 teacher 猜语言或改写 transcript。

## Source 分组与防泄漏

同一 clean source 的多个退化版本必须先归并成一个 `source_utterance_id`，再按 group 切分。
派生优先级：公开 source id/name -> 可解析的 file/audio path -> 数据集稳定 index 规则。
若无法可靠派生，样本不得进入正式 train/validation。

硬门槛：

- train/validation/test 的 `sample_id` overlap 为 0。
- train/validation 的 `source_utterance_id` overlap 为 0。
- train/validation/Bench 的相同公开 name、source path、benchmark id 和 audio SHA256 overlap
  为 0。
- LibriSpeech/AISHELL-1 按官方 split，能获得 speaker id 时额外报告 speaker overlap。

归一化 transcript overlap 只报告并抽查，不作为通用硬失败条件；不同 speaker 说同一句话
不等于数据泄漏。若 transcript overlap 同时伴随相同 source metadata，则按 source 泄漏处理。

## 时长与质量过滤

第一轮只保留 0.5-30 秒音频。以下样本写入 rejects：

- 空 transcript 或语言不明。
- 音频不存在、无法解码、时长异常、NaN/Inf。
- 峰值或静音比例明显异常。
- source identity 缺失，无法保证 group split。
- 与 fixed test 存在硬 overlap。

每个 reject 保存 `reason` 和原始 source locator，不能静默丢弃。

## 缓存与 Colab I/O

公开 robust 数据压缩体积很大，160k 预计仍为几十 GB。为避免 Google Drive 小文件 I/O：

1. Drive 或其他持久盘只缓存原始 parquet/打包 tar shard 和构建状态。
2. Notebook 启动后把需要的 shard staging 到 `/content/mega-asr-runtime/`。
3. 在本地 SSD 解包/物化音频并生成 resolved manifest。
4. Trainer 只读本地 SSD。
5. Drive 只持续写 checkpoint、manifest、stats 和 prediction 增量。

不得把 200k 个独立 wav/flac 逐个写入或逐个读取 Google Drive。

## 输出

- `data/jsonl/public_robust_200k_train.jsonl`
- `data/jsonl/public_robust_10k_val.jsonl`
- `data/jsonl/vitw_bench_5k_test.jsonl`
- `data/jsonl/public_clean_tests.jsonl`
- `data/jsonl/public_robust_200k_stats.json`
- `data/jsonl/public_robust_200k_validation.json`
- `data/jsonl/public_robust_200k_rejects.jsonl`
- `data/jsonl/public_robust_200k_sources.json`

## 测试

- `--help` 和 `--smoke` 可执行。
- Smoke 覆盖 en/zh、clean、7 atomic、compound 和 Bench real/synthetic。
- Full 行数精确等于 200k/10k/5k。
- 所有 resolved audio 路径存在且可解码。
- 固定 seed/revision/config 重跑得到相同 sample id 顺序和 manifest hash。
- 硬 overlap 全部为 0，transcript overlap 单独报告。
- 每个 language/condition 抽听至少 10 条并记录结果。

## 验收与影响

`public_robust_200k_validation.json` 所有硬检查通过才允许训练。旧 TTS、MVP 150 和 v6A
保留为 smoke/history，不再参与正式模型选择，也不能用于证明数据规模。
