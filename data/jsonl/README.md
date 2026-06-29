# data/jsonl

存放 JSONL manifest。

训练样本暂定使用 `audio`、`answer`、`language`、`scenario` 字段；LoRA 阶段如需额外 target 字段，必须先在训练文档中确认格式。

评测样本使用 `audio`、`answer`、`language`、`scenario` 字段。

LoRA MVP bootstrap 训练样本额外要求：

- `split`：`train` 或 `val`。
- `base_utterance_id`：用于检查 train/val 不泄漏。
- `utterance_id`：单条增强样本 id。
- `is_degraded`：是否为退化音频。
- `seed`、`profile`：用于复现合成退化。

固定 MVP 150 manifest 只作为 held-out test，不作为正式 LoRA MVP 训练输入。
