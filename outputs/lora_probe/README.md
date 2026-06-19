# LoRA Probe Outputs

本目录用于保存 `05A LoRA 训练前探测` 的受控输出。

推荐输出目录：

```text
outputs/lora_probe/qwen3_asr_1_7b/
```

该目录下应包含：

- `module_snapshot.json`
- `module_summary.csv`
- `lora_target_candidates.json`
- `lora_target_candidates.md`
- `unsloth_compatibility.json`

这些文件用于复核 Qwen3-ASR 当前版本的真实模块结构、第一版 LoRA target 候选，以及 Unsloth 是否可作为训练 backend。
