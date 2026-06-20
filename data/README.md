# data

存放新工程的数据 manifest 和轻量示例。

规则：

- 大规模原始音频、增强音频和缓存不提交到仓库。
- `data/mvp_eval/audio/` 是例外：当前提交 clean、noise、reverb、far_field、dropout 各 30 条 MVP smoke 音频，用于 Colab baseline 和 LoRA smoke training 复现。
- JSONL manifest 可以提交示例或小型 smoke 文件。
- 真实训练数据默认放在 Google Drive 或外部数据目录。
