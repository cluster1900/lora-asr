# inference 目录说明

## 目录职责

本目录负责用固定 BF16 合同批量运行 Qwen3-ASR base 或单个标准 PEFT adapter，并把每条结果持久化
为 JSONL。它只做推理，不选择训练数据、不计算最终指标，也不训练或合并 adapter。

base 与 adapter 共用同一模型 revision、dtype、设备和解码上限，保证后续评测是同口径对比。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `qwen3_asr_infer.py` | 唯一批量推理入口。支持 base/adapter、相对音频路径、逐条持久化、失败不中断和 `--resume`。 |
| `README.md` | 说明本目录边界、文件职责、输入输出和维护要求。 |

`__pycache__/` 是 Python 自动生成的本地缓存，已被 Git 忽略，不是推理结果，可随时删除。

## 输入

`--manifest` 指向 JSONL，每行至少需要：

- `sample_id`：稳定且唯一，用作恢复键。
- `audio`：绝对路径，或相对 `--audio-root`/manifest 目录的路径。
- `answer`：gold transcript，原样带入输出供评测使用。
- `language`：`en` 或 `zh`。

`scenario`、`condition_group`、`audio_origin` 等元数据会原样保留，供评测聚合。使用
`--adapter-dir` 时加载训练产出的标准 PEFT adapter；省略时运行固定 base。

## 输出与恢复

`--output-jsonl` 逐行记录原 manifest 字段及：

- `inference_key`、`manifest_index`：稳定恢复和顺序信息。
- `prediction`、`predicted_language`：转写结果。
- `model_id`、`model_revision`、`dtype`、`decoding`、`adapter_dir`：完整推理合同。
- `resolved_audio`、`infer_seconds`、`error`：运行诊断。

每行写入后执行 flush 和 fsync。单条音频失败会写入 `error` 并继续下一条；带 `--resume` 重跑时按
`inference_key` 跳过已落盘行。如果没有待处理行，程序不会加载模型或占用 GPU。

## 使用方式

```bash
# 固定 BF16 base
python inference/qwen3_asr_infer.py \
  --manifest /path/to/manifest.jsonl \
  --output-jsonl /path/to/base.predictions.jsonl \
  --audio-root /path/to/audio \
  --resume

# 单 adapter
python inference/qwen3_asr_infer.py \
  --manifest /path/to/manifest.jsonl \
  --output-jsonl /path/to/adapter.predictions.jsonl \
  --adapter-dir /path/to/release/adapter \
  --audio-root /path/to/audio \
  --resume
```

对应测试：`tests/test_infer_resume.py`。

## 维护要求

新增、重命名或删除本目录文件时，必须同步更新“文件清单”。修改模型 revision、adapter 挂载位置、
解码参数、输入输出 schema 或恢复语义时，必须在同一提交更新本 README、评测合同和对应测试。
