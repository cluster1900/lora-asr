# data/jsonl

存放 JSONL manifest。

正式训练与评测样本至少使用 `audio`、`answer`、`language`、`scenario`、`split`、
`source_dataset`、`source_id` 和 `duration`。完整 schema 以
`docs/qwen3-asr/03_data_plan.md` 为准。

评测样本使用 `audio`、`answer`、`language`、`scenario` 字段。

30k curriculum 在原字段外增加：

- `base_prediction`
- `base_error_rate`
- `base_metric`：English 为 `wer`，Chinese 为 `cer`
- `curriculum_bucket`：`lt_030`、`lt_050` 或 `lt_070`

以下字段只属于历史 LoRA MVP bootstrap：

- `split`：`train` 或 `val`。
- `base_utterance_id`：用于检查 train/val 不泄漏。
- `utterance_id`：单条增强样本 id。
- `is_degraded`：是否为退化音频。
- `seed`、`profile`：用于复现合成退化。

固定 MVP 150 manifest 只用于历史回归，不进入公开 200k A2S 训练或正式验收。
