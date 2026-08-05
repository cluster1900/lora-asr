# train

当前正式训练入口是待实现的 `train_qwen3_asr_a2s.py`。它基于 Qwen3-ASR 官方
`finetuning/qwen3_asr_sft.py` 的 prompt、collator 和 Trainer 结构，只增加项目 JSONL、
PEFT 与阶段切换；完整合同见 `docs/qwen3-asr/02_development_plan.md`。

## 唯一正式流程

```bash
python train/train_qwen3_asr_a2s.py \
  --config configs/train/qwen3_asr_public_200k_a2s.yaml \
  --smoke-steps 10

python train/train_qwen3_asr_a2s.py \
  --config configs/train/qwen3_asr_public_200k_a2s.yaml \
  --resume auto
```

Runner 在同一个 adapter 中依次执行：

1. Phase I：30k base-error curriculum，2 epoch，只训练 upper-4 audio + projection。
2. Phase II：200k，1 epoch，只训练 decoder。
3. Phase III：200k，1 epoch，联合训练全部 343 个 LoRA target。

固定使用 BF16、LoRA r=8/alpha=16/dropout=0.05、effective batch 128。阶段间只切换
`requires_grad` 和 optimizer groups，不合并或叠加多个 adapter。

## Target 合同

`inspect_qwen3_asr_modules.py` 和受控 module snapshot 用于从 pinned Qwen revision 独立生成
target map。正式 runner 必须按分组校验 343 个 Linear target，排除 `lm_head`、embedding、
norm 和 Conv2d；revision、分组数量或 target-map hash 不符时立即终止。

## 历史入口

`train_qwen3_asr_lora.py`、旧 YAML、`lora_targets.py` 和 `peft_targets.py` 只用于复现
batch-size-1 v1-v5 smoke/MVP。它们不支持正式 Trainer validation、scheduler、完整 resume
或新 decoder target 合同，不得扩建或用于 200k 训练。

禁止复用 Mega-ASR 的 wrapper、训练入口、target regex 或 adapter merge 逻辑。
