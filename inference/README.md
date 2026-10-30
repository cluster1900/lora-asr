# inference

## 目录职责

封装面向官方 `qwen-asr`/Transformers API 的单条与批量 ASR 推理。负责加载 base 模型或特定 checkpoint adapter，执行统一文本归一化（`zh-cer-v1`）。评测推理严格执行 `decode.eval`（greedy），RL 候选生成严格执行 `decode.rollout`（温度/top-p 采样）。推理异常样本写入结果文件而不中断整批任务；公开 `model.generate` 的 token IDs、EOS/stop reason 和 `generation_raw`，`transcribe()` 仅用于 Phase 0 对照。

## 当前文件清单

- `README.md`：本说明文档。
- `config.py`：读取 `qwen3-asr-infer-v1`，拒绝未知字段，校验 `scoring`（含可选锁定 `jieba` WER 的 `wer_tokenizer`、`wer_tokenizer_version`、`wer_dictionary_sha256`、`wer_hmm`），并仅对 `decode.eval` 计算 `decode_eval_sha256`。
- `run_inference.py`：greedy `model.generate` 主路径，旁路记录官方 `transcribe()` 文本。单条失败写入 `error` 后继续。

LoRA adapter 加载留到 SFT checkpoint 出现之后。Phase 0 只跑官方 base。

## 主要输入输出

- **输入**：`Qwen/Qwen3-ASR-0.6B` 模型权重（或附带 LoRA adapter 路径）、音频 JSONL manifest、推理配置（含 `decode.eval` 或 `decode.rollout` 参数）、输出文件路径。
- **输出**：预测 JSONL 文件，每行包含 `sample_id`、`prediction_raw`、`prediction_normalized`、`generation_raw`、`token_ids`、`stop_reason`、`error` 及模型/adapter、processor、decode、seed 和输入音频 hash 元数据。

## 使用入口

```bash
python3 inference/run_inference.py --config configs/inference/base_eval.yaml \
  --manifest manifests/phase0/smoke.jsonl \
  --output runs/<run_id>/predictions/smoke.jsonl \
  --mode eval --seed 42
```

正式评测传 `--config configs/inference/base_eval.yaml`。CLI `--seed` 必须和配置里的 seed 相同。单条失败 flush 到 JSONL 后继续，进程返回码为 1；模型加载失败返回码为 2，不写半截成功结果。`prediction_raw` 不做重复折叠。`transcribe_text` 是官方 `transcribe()` 的文本，两者不一致时写入同一行的 `transcribe_matches_greedy=false`。

## 维护要求

推理参数、解码配置或 CLI 接口发生变更，以及新增、重命名或删除文件时，必须在同一提交同步更新对应目录的 `README.md`、开发文档和测试。
