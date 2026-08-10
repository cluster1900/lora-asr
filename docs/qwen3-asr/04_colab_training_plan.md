# Colab 训练方案

## 背景与范围

Colab 是第一优先训练环境。第一轮只训练一个 BF16 LoRA adapter，不做量化训练、direct SFT
对照、target/LR sweep、teacher 或 RL。

## 环境与配置

依赖由 `requirements-colab.txt` 固定，保留 Colab 自带 CUDA torch。模型、Qwen 源码和数据集均
使用 40 字符 revision。新产物统一写入 Drive 的 `qwen3-asr-public-a2s/` 独立命名空间，训练时
音频放 `/content/qwen3-asr-runtime/` 本地 SSD，避免读取旧 Mega-ASR 实验目录。

训练配置：LoRA r=8、alpha=16、dropout=0.05，effective batch 128。Phase I/II 学习率
`1e-6`；Phase III audio/projection `5e-7`、decoder `1e-6`。

## 执行

```bash
python train/train_qwen3_asr_a2s.py \
  --config configs/train/qwen3_asr_public_200k_a2s.yaml \
  --smoke-steps 10

python train/train_qwen3_asr_a2s.py \
  --config configs/train/qwen3_asr_public_200k_a2s.yaml \
  --resume auto
```

正式训练按 Phase I -> II -> III 顺序执行。阶段边界保存 adapter 和完整 Trainer checkpoint；
canary 不通过时不得继续下一阶段。新 run 不接受旧 adapter，也不能通过 CLI 跳过阶段。

## 测试与验收

- 10 step 后保存，重新加载并继续 2 step；global step 必须连续。
- resolved config、model revision、target map hash 和 manifest hash 随 run 保存。
- 每阶段 loss、gradient、learning rate 均为有限值。
- 每阶段 canary 报告 clean/degraded、空输出、重复、过长、幻觉式输出和 inference error。
- 最终 adapter/processor 可从新进程加载并推理 clean/degraded 各一条。

## 影响与停止条件

OOM 时先降低 per-device batch 并等比增加 gradient accumulation，保持 effective batch 128。
target 合同漂移、resume 不连续、非有限训练状态或 canary 超阈值时立即停止，不通过跳过门禁来继续。
