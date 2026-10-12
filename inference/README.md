# inference

## 目录职责

封装面向官方 `qwen-asr`/Transformers API 的单条与批量 ASR 推理。负责加载 base 模型或特定 checkpoint adapter，执行统一文本归一化（`zh-cer-v1`）。评测推理严格执行 `decode.eval`（greedy），RL 候选生成严格执行 `decode.rollout`（温度/top-p 采样）。推理异常样本写入结果文件而不中断整批任务；公开 `model.generate` 的 token IDs、EOS/stop reason 和 `generation_raw`，`transcribe()` 仅用于 Phase 0 对照。

## 当前文件清单

- `README.md`：本说明文档。推理主入口 `run_inference.py` 和解码 helper 将在 Phase 0 实现并登记。

## 主要输入输出

- **输入**：`Qwen/Qwen3-ASR-0.6B` 模型权重（或附带 LoRA adapter 路径）、音频 JSONL manifest、推理配置（含 `decode.eval` 或 `decode.rollout` 参数）、输出文件路径。
- **输出**：预测 JSONL 文件，每行包含 `sample_id`、`prediction_raw`、`prediction_normalized`、`generation_raw`、`token_ids`、`stop_reason`、`error` 及模型/adapter、processor、decode、seed 和输入音频 hash 元数据。

## 使用入口

在 Phase 0 实现后提供：
```bash
# 评估推理（greedy）
python3 inference/run_inference.py --model Qwen/Qwen3-ASR-0.6B --manifest manifests/val.jsonl --output predictions/val.jsonl --language zh --mode eval

# 挂载 LoRA adapter 推理
python3 inference/run_inference.py --model Qwen/Qwen3-ASR-0.6B --adapter runs/<run_id>/checkpoints/step_000010/adapter --manifest manifests/val.jsonl --output predictions/val_sft.jsonl --language zh --mode eval
```

正式评测优先传 `--config configs/inference/base_eval.yaml`，其中固定 model revision、processor hash、language、context、max_new_tokens、EOS policy 和 `decode.eval`；显式 CLI 参数只用于 Phase 0 探针。单条失败必须 flush 到 JSONL 后继续，批处理返回非零码并在汇总中报告失败数；LoRA adapter 在新进程加载后至少复核一条 clean 和一条 degraded。

## 维护要求

推理参数、解码配置或 CLI 接口发生变更，以及新增、重命名或删除文件时，必须在同一提交同步更新对应目录的 `README.md`、开发文档和测试。
