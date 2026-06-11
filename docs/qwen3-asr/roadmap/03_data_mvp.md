# 03 数据 MVP

最后更新：2026-06-08

## 背景

鲁棒 ASR 的效果取决于数据。我们需要先构建小规模、可复现、带场景标签的数据集，用于 baseline、LoRA 训练和评测。

## 目标

构建 MVP 级别 train/val/test JSONL，覆盖 clean 和核心 degraded 场景。

## 范围

本步骤负责数据 manifest 和初始音频收集，不追求百万级规模。

## 输入

- clean speech 数据源，例如 LibriSpeech 或 Common Voice。
- 少量真实 degraded 音频，如果已有。
- 数据配置文件。

## 输出

- `data/jsonl/train.jsonl`
- `data/jsonl/val.jsonl`
- `data/jsonl/test.jsonl`
- `data/jsonl/dataset_stats.json`
- 数据质量检查报告。

## MVP 规模

- train：10k-20k 条。
- val：1k-2k 条。
- test：1k-2k 条。
- smoke：10-50 条。
- baseline_mvp_150：本地合成评测集，clean/noise/reverb/far_field/dropout 各 30 条，总计 150 条，用于正式数据源接入前验证 baseline 扩展评测闭环。

## 必需字段

训练样本：

```json
{
  "audio": "...",
  "answer": "REFERENCE TEXT",
  "language": "English",
  "scenario": "noise",
  "source": "librispeech",
  "is_degraded": true
}
```

评测样本：

```json
{
  "audio": "...",
  "answer": "REFERENCE TEXT",
  "language": "en",
  "scenario": "noise",
  "source": "librispeech",
  "is_degraded": true
}
```

## 执行步骤

1. 确定第一数据源和授权条件。
2. 生成 clean speech manifest。
3. 选择固定 smoke/val/test 样本。
4. 为每条样本记录 source、speaker、utterance id。
5. 检查音频路径、时长、采样率。
6. 输出 dataset stats。
7. 更新数据文档和进度文档。

## 本地合成 MVP 评测集

在真实数据源接入前，先通过 `scripts/create_mvp_eval_audio.py` 生成可复现的本地合成评测集。该数据集用于：

- 验证 manifest 规模从 2 条扩展到 150 条后，推理脚本是否能稳定批处理。
- 验证评测脚本是否能按 `scenario` 聚合 clean/noise/reverb/far_field/dropout。
- 记录 Qwen3-ASR baseline 在典型退化场景下的初始失败模式。
- 同时覆盖短文本和长文本，观察模型在长句上的漏词、重复输出和幻觉补全。

生成物默认不进入 git：

- `data/mvp_eval/audio/`
- `data/jsonl/baseline_mvp_150.local.jsonl`
- `data/jsonl/baseline_mvp_150_stats.local.json`

注意：该数据集由系统 TTS 和规则退化生成，不能替代真实 benchmark。

默认使用 `--profile hard` 生成强退化版本：

- noise：低 SNR 背景噪声。
- reverb：更长延迟、更强多段回声。
- far_field：更明显的音量衰减、低通、房间反射和环境噪声。
- dropout：更长、更频繁的短时静音/强衰减。

clean 场景不做退化，用于后续比较 degraded 改善是否引入 clean regression。

文本长度设计：

- 每个场景前 15 条为短文本。
- 每个场景后 15 条为长文本。
- manifest 写入 `text_length_bucket` 和 `reference_word_count`，便于后续按长度聚合错误。

## 切分规则

- 同一原始 utterance 不得跨 split。
- 同一 speaker 尽量不跨 split。
- test 集固定，不参与训练和参数选择。
- 增强后的版本必须跟随原始 utterance 所属 split。

## 测试标准

- JSONL 每行合法。
- 每条音频路径存在。
- 必需字段齐全。
- `baseline_mvp_150` 场景计数必须为 clean/noise/reverb/far_field/dropout 各 30 条。
- 每个场景必须包含 short 15 条、long 15 条。
- `baseline_mvp_150_stats.local.json` 必须记录 `profile` 和各场景退化统计。
- 时长在 MVP 限制内，默认 3-20 秒。
- split 泄漏检查通过。

## 验收标准

- smoke、train、val、test 四类 manifest 均存在。
- dataset stats 包含样本数、总时长、scenario 分布、source 分布。
- 至少覆盖 clean、noise、reverb、far_field、dropout。
- 数据构建命令和配置可复现。

## 风险

- 数据路径在 Colab 和本地不一致。缓解：统一使用 Drive 根目录变量。
- 数据源授权不明确。缓解：只使用允许研究和训练的数据源，并记录来源。
- 测试集泄漏。缓解：先按原始 utterance 切分，再做增强。
