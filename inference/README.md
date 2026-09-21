# inference 目录说明

## 目录职责

本目录维护 V100 服务器 FP16 base 与 adapter 模型的批量推理 Runner。推理必须采用官方 `qwen-asr` / Transformers API、单卡 batch 1、eager attention，并将每条推理结果持久化为 JSONL，支持断点续推与容错写盘。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `run_inference.py` | E2 阶段推理 Runner：单卡 batch 1、FP16、eager attention，支持加载 base 或 PEFT adapter，支持断点续推、逐行 flush+fsync 容错写入，并可直接联动 `evaluation/eval_wer.py` 计算 WER/CER。 |
| `parallel_inference.py` | 多卡并行推理 Runner：自动将输入 Manifest 均匀分片至指定多张 GPU（如 4 卡），并行拉起 `run_inference.py` 执行推理，分片日志独立持久化至分片日志文件杜绝 OS 管道阻塞，支持断点续推与自动汇聚，按原始 Manifest 严格顺序合并预测产物并校验样本数一致性。 |
| `README.md` | 说明当前目录职责、文件清单、CLI 参数与维护要求。 |

## 使用入口与 CLI 参数

```bash
# 1. 运行 128 条 Base Smoke 推理（使用默认官方 revision 与 FP16/eager）
python3 inference/run_inference.py \
    --manifest /data/mega-asr/manifests/smoke.jsonl \
    --output /data/mega-asr/results/base_smoke_predictions.jsonl \
    --model-id Qwen/Qwen3-ASR-1.7B \
    --revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
    --device cuda:0 \
    --dtype float16 \
    --eval

# 2. 运行加载 Adapter 的推理与评测
python3 inference/run_inference.py \
    --manifest /data/mega-asr/manifests/validation.jsonl \
    --output /data/mega-asr/results/sft_pilot_predictions.jsonl \
    --model-id Qwen/Qwen3-ASR-1.7B \
    --adapter-dir /data/mega-asr/runs/sft_pilot/adapter \
    --eval
# 3. 运行合并权重底座（Merged Base）的推理与评测
python3 inference/run_inference.py \
    --manifest /data/mega-asr/manifests/validation.jsonl \
    --output /data/mega-asr/runs/eval_validation_sft_pilot/predictions.jsonl \
    --model-id /data/mega-asr/runs/sft_pilot/merged_base \
    --method sft_merged \
    --eval

# 4. 运行 4 卡并行分片推理与自动合并
python3 inference/parallel_inference.py \
    --manifest /data/mega-asr/manifests/pilot_dpo.jsonl \
    --output /data/mega-asr/runs/dpo_candidates/base_pilot_dpo_predictions.jsonl \
    --model-id Qwen/Qwen3-ASR-1.7B \
    --revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
    --gpus 0 1 2 3 \
    --method base
```

### 主要输入与输出 Schema

- **输入 Manifest**：每行至少包含 `sample_id`、`audio`、`text`、`language`、`scenario`、`condition_group`、`audio_origin`。
- **输出 Prediction**：逐行写入 JSONL，包含：
  - `sample_id`：样本唯一标识
  - `text`：gold transcript 文本
  - `prediction`：模型解码输出（若失败为空字符串）
  - `language`：语言标识 `en|zh`
  - `scenario`：测试场景
  - `model_id`：模型名称或本地合并权重路径
  - `model_revision`：模型 commit SHA
  - `dtype`：`float16`
  - `attention`：`eager`
  - `method`：推理方法标识（`base`、`lora`、`sft_merged`、`dpo_merged`、`rl_merged`；支持 `--method` CLI 显式指定，未指定时按 `adapter_dir` 与 `model_id` 自动智能推导）
  - `adapter_dir`：Adapter 路径（若无则为空）
  - `infer_seconds`：单条解码耗时（秒）
  - `error`：异常信息（若成功则为空字符串）

### 容错与恢复机制

- 逐条写入后执行 `flush()` 与 `os.fsync()`，保证进程被中断时已推理数据不丢失；
- 默认支持断点续推（读取输出文件中的已存 `sample_id` 并自动跳过）；
- 单条音频解码或文件异常时捕获异常并记录 `error` 字段，绝对不中断整批任务。

## 维护要求

新增推理 runner 时，必须同步更新本 README、根 README、架构/开发/测试文档和对应测试。实现必须
先通过单条 clean、单条 degraded、错误恢复和新进程 adapter 加载测试，再进入 pilot。
