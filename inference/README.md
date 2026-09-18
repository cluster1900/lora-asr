# inference 目录说明

## 目录职责

本目录预留 V100 FP16 base/adapter 批量推理入口。旧的 BF16 推理实现已经删除，新的实现必须使用
官方 `qwen-asr`/Transformers API、单卡 batch 1、eager attention，并将每条结果持久化为 JSONL。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `README.md` | 说明当前目录为空、未来推理 runner 的输入输出合同和维护要求。 |

## 计划输入

manifest 每行至少需要 `sample_id`、`audio`、`text`、`language`。`scenario`、`condition_group`、
`audio_origin` 等元数据应原样保留。base 和 adapter 必须使用相同模型 revision、dtype、attention、
设备和解码上限。

## 计划输出与恢复

未来 runner 必须逐行记录 `prediction`、`predicted_language`、model contract、`resolved_audio`、
`infer_seconds` 和 `error`。每行写入后 flush+fsync；单条失败写入 `error` 并继续；resume 不能重复
已落盘样本。

## 维护要求

新增推理 runner 时，必须同步更新本 README、根 README、架构/开发/测试文档和对应测试。实现必须
先通过单条 clean、单条 degraded、错误恢复和新进程 adapter 加载测试，再进入 pilot。
