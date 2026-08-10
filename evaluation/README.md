# evaluation 目录说明

## 目录职责

本目录负责把推理产生的逐条 JSONL 预测转换成可复现的 ASR 指标与失败样本报告。评测只消费结果，
不加载模型、不执行训练，也不修改输入预测文件。

英文与中文必须分开计量：英文使用 WER，中文使用 CER。包含两种语言的集合不生成伪造的混合
WER/CER，而是通过 `by_language` 和 language macro 汇总。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `eval_wer.py` | 唯一评测入口。执行文本归一化、WER/CER、失败输出检测、scenario 聚合和固定 32-cell Bench 聚合。 |
| `README.md` | 说明本目录边界、文件职责、输入输出和维护要求。 |

`__pycache__/` 是 Python 自动生成的本地缓存，已被 Git 忽略，不是项目产物，可随时删除。

## 输入

`eval_wer.py` 接收 `inference/qwen3_asr_infer.py` 生成的 prediction JSONL。

每行至少需要：

- `answer`：gold transcript，不能为空。
- `prediction`：模型预测文本；推理失败时可为空。
- `language`：只接受 `en` 或 `zh`，用于选择 WER/CER。

推荐保留 `sample_id`、`scenario`、`condition_group`、`audio_origin`、`error` 等字段，以便生成
分场景指标和失败统计。标记了 `error` 的行按空预测计分，不能因为推理失败而获得虚假的低错误率。

## 输出

指定 `--output-dir` 后生成：

| 产物 | 内容 |
| --- | --- |
| `scored.jsonl` | 原始/归一化文本、metric、edit 数、错误率、长度比和失败标签。 |
| `metrics.json` | overall、by-language、by-scenario、by-origin、by-condition 和 32-cell 汇总。 |
| `by_language.csv` | 英文 WER 与中文 CER。 |
| `by_scenario.csv` | 按语言和场景拆分的指标。 |
| `by_cell.csv` | language x real/synthetic x scenario 的固定 Bench cell。 |

关键失败指标包括 inference error、空输出、重复输出、过长输出和幻觉式输出。训练阶段 canary
直接读取 `metrics.json` 的 robust/clean language macro 与失败率。

## 使用方式

```bash
python evaluation/eval_wer.py \
  --predictions-jsonl /path/to/predictions.jsonl \
  --output-dir /path/to/evaluation
```

对应测试：`tests/test_eval_wer.py`。

## 维护要求

新增、重命名或删除本目录文件时，必须同步更新“文件清单”。修改指标定义、输入字段、输出产物或
CLI 参数时，必须在同一提交更新本 README 和对应测试；不得把中英文错误率合并成一个 WER/CER。
