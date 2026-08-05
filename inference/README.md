# inference

当前正式入口是待实现的 `qwen3_asr_infer.py`，统一支持 BF16 base 和可选单 LoRA adapter：

```bash
python inference/qwen3_asr_infer.py \
  --manifest data/jsonl/public_robust_test.jsonl \
  --output-jsonl outputs/eval/base.predictions.jsonl \
  --model-id Qwen/Qwen3-ASR-1.7B \
  --model-revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --dtype bfloat16 \
  --resume
```

增加 `--adapter-dir` 即切换为 LoRA；不提供 router 模式。入口必须逐条增量写入、支持 resume，
并将失败样本写为带 `error` 的 JSONL 行，不能让单条失败中断整批。语言按 manifest 每条
`language` 选择，禁止用全局 English 默认值覆盖中文。

`qwen3_asr_base_infer.py` 与 `qwen3_asr_lora_infer.py` 只保留历史 4bit MVP 复现，不用于
BF16 正式比较。不得复用 Mega-ASR 私有推理 wrapper。
