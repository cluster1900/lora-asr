# data

存放 manifest 和轻量 fixture。当前正式数据合同见
`docs/qwen3-asr/03_data_plan.md`。

规则：

- 大规模原始音频、增强音频和缓存不提交到仓库。
- `data/mvp_eval/audio/` 是例外：仓库中的 150 条历史 MVP 音频只用于回归旧通路。
- JSONL manifest 可以提交示例或小型 smoke 文件。
- 公开 200k 正式音频不提交；大 shard 放 Google Drive，训练前物化到 `/content`。
- 正式 manifest、30k curriculum、stats、rejects 和 validation report 必须保存 config、revision、seed 和 hash。
- 当前训练不使用 teacher 生成或重写 transcript。
