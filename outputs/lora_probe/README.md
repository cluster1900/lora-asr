# LoRA Probe Outputs

本目录保存 Qwen3-ASR pinned revision 的受控模块探测输出。历史候选不直接定义新 A2S target；
正式 runner 必须重新验证 revision、模块类型、分组数量和 target-map hash。

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

这些文件用于复核 Qwen3-ASR 真实模块结构。Unsloth 兼容性和旧 target 候选只属于历史记录；
当前 backend 固定为官方 Transformers Trainer + PEFT。
