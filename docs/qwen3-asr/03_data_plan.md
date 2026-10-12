# 数据方案

最后更新：2026-10-12

## 背景与范围

MVP 以中文语音为主，覆盖 `clean`、`noise`、`reverb`、`far_field`、`dropout`，验证模型是否改善退化语音而不损害 clean speech。本阶段建立可审计、可复现的 JSONL 数据合同，不把未审计的外部数据直接送入训练，也不在 manifest 内保存训练 seed。

## Manifest schema

每行至少包含以下字段：

```json
{"schema_version":"qwen3-asr-manifest-v1","sample_id":"utt0001__noise","source_utterance_id":"utt0001","audio":"wavs/utt0001__noise.wav","text":"纯中文转写","language":"zh","scenario":"noise","split":"train"}
```

字段约束：

- `sample_id` 在 train/val/test 全局唯一；`source_utterance_id` 标识同一原始语句的所有增强版本。
- `audio` 是相对 manifest 所在目录的 POSIX 路径，不能使用 notebook 临时路径、Drive 绝对路径或 `..` 穿越。
- `text` 是纯中文转写，不含 `language Chinese<asr_text>` 前缀、特殊 token 或 reward；前缀和 `<|im_end|>` 由 trainer 按固定 `qwen-asr` 版本添加。
- `language` 当前固定为 `zh`；`scenario` 只能是五类 MVP 值；`split` 必须与文件名一致。
- 可选 `augmentation` 对象记录噪声文件、SNR、混响、远场距离、dropout 位置和处理版本；不能覆盖原始 audio 或 text。

manifest 目录必须同时保存 `train.jsonl`、`val.jsonl`、`test.jsonl`、`manifest.json` 和 `sha256.json`。`manifest.json` 记录原始数据版本、生成器 Git SHA、音频处理版本、行数、每个 split/scenario 计数、固定 seed 和质量检查摘要；`sha256.json` 记录每个文件的 SHA-256、字节数和生成时间。

## 来源、物化与音频规范

原始音频、标注和增强器版本必须能追溯到来源 ID；下载或增强失败写入审计，不用空音频占位。训练前把音频物化为 16 kHz、单声道、PCM/WAV，建议时长范围为 0.5–30 秒；超过上限的样本在 manifest 审计中统计，是否截断或拒绝必须由配置决定并记录。训练 step 只读取 Colab 本地缓存中的物化文件，Drive 保存原始 manifest、hash 和结果。

## 切分、配额与泄漏检查

先按 `source_utterance_id` 分组，再切 train/val/test；同一 source 的 clean 和所有增强版本只能进入一个 split。test 不参与 SFT、DPO pair 构造、RL rollout、checkpoint 选择和任何阈值调参，阶段门禁只读 val。

每个正式 pilot 的 train 至少包含五类 scenario；val 必须包含 clean 与四类 degraded，且用来宣称场景改善的 cell 至少 50 条。脚本输出每个 split × scenario 的行数、唯一 source 数、空文本数、坏音频数、重复 ID 数、时长分位数和音频 hash 冲突。缺少配额时状态为 `insufficient_data`，不能用复制行冒充新样本。

## 固定生成与质量检查

数据脚本从固定输入版本、配置和 seed 生成；同一命令第二次执行必须得到相同的 ID 顺序、计数和摘要。检查至少包括：

1. JSON 可解析，必填字段存在，类型和枚举正确。
2. 音频路径存在、可解码、采样率/声道/编码符合规范；文本非空且 UTF-8 有效。
3. `sample_id` 不重复，source 不跨 split，train/val/test 的 audio/text/source hash 交集为空。
4. 五类 scenario 覆盖和每个 gate cell 的最小行数满足配置。
5. manifest、物化音频和生成器版本写入 hash；改动任一输入都会改变 manifest SHA-256。

坏样本不应让批处理崩溃：写入 `rejected.jsonl`（含 `sample_id`、原因、路径和处理版本），汇总中的 rejected/shortage 数必须与文件一致。只有通过校验的 manifest 才能交给推理或训练。

## 训练接口影响

SFT 读取 `text` 并在运行时构造监督前缀；DPO pair builder 可读取 train/val manifest 和 SFT greedy predictions，但必须保留 source ID 和 pair 审计；RL 读取 train manifest，不把参考转写作为候选。任何新字段都必须更新 schema 版本、`configs/README.md`、训练方案和测试夹具。

## 验收

固定输入、配置和 seed 能重建三个 JSONL、`manifest.json` 和 `sha256.json`；所有音频路径存在且可读；split 无明显同源泄漏；五类 scenario 覆盖；坏样本有明确审计；train/val/test 行数、source 数和摘要可被第二次运行复核。未通过任一项时，Phase 1 保持未完成，后续训练不得启动。

## 影响

采用相对路径、文件级 split、纯文本 `text` 和显式 source ID 后，旧的绝对路径 manifest、混入特殊 token 的标注和未带版本 hash 的结果不能直接复用；必须重新物化或提供迁移报告。该合同同时支持本地单测和 Colab Drive 恢复，不改变外部数据原始文件。
