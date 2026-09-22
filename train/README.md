# train 目录说明

## 目录职责

本目录维护 V100 服务器（4 × Tesla V100-SXM2-32GB）上的模型训练 Runner，覆盖 SFT（监督微调）、DPO（直接偏好优化）与 RL（强化学习，GRPO）三个阶段。所有训练必须面向官方 `qwen-asr` / Transformers API，基于底层 `thinker` 模块并精确注入 199 个 Linear LoRA targets，采用单机 4 卡 DDP（或单卡模式）、FP16 精度、eager attention 与梯度检查点，严格落地 10+2 步断点续训与 `merge_and_unload()` 权重接力生命周期。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 包声明与初始化文件。 |
| `train_sft.py` | SFT 阶段训练 Runner：支持单机 4 卡 DDP 与 `--single-gpu` 调试模式；注入 199 个 Linear LoRA 目标（Projection 3 + Decoder 196）；实现标签掩码 Cross-Entropy 损失与分层均衡 validation loss 评估；支持 10+2 检查点保存与断点恢复（optimizer、scheduler、RNG、training_state）；提供 `--export-merged` 执行 `merge_and_unload()` 权重导出。 |
| `train_dpo.py` | DPO 阶段训练 Runner：支持单机 4 卡 DDP 与 `--single-gpu` 调试模式；继承 199 个 Linear LoRA 目标；基于预计算的参考对数概率（或在线计算）实现 Bradley-Terry 偏好损失函数，记录隐式奖励差与 preference accuracy；提供全状态断点保存/恢复与 `--export-merged` 权重导出。 |
| `train_rl.py` | RL (GRPO) 阶段训练 Runner：支持单机 4 卡 DDP 与 `--single-gpu` 调试模式；以 DPO 合并模型为初始 Policy 与冻结 Reference Model，注入 199 个 Linear LoRA 目标；每条样本生成 $G=4$ 个在线 Rollout 采样，支持温度调节（默认 0.85）与 rank/candidate 独立 RNG 种子偏移；结合 ASR 准确率（WER/CER）与 empty/repeat/too_long/hallucination 惩罚计算序列 Reward 并标准化 Group Advantage；组内方差为零优势置 0，若连续 2 步批次内零方差组比例超过阈值（默认 0.80）硬中止并标记 `FAILED_ZERO_VARIANCE`；四卡独立落盘并汇聚全量 `rollouts.jsonl`（15,360 行）与审计；强制在 Step 0 记录冻结底座全量 held-out 基准并全程采用统一验证口径；提供 10+2 全状态断点恢复与 `--export-merged` 权重导出。 |
| `README.md` | 说明当前目录职责、文件清单、主要输入输出、CLI 参数与维护要求。 |

## 使用入口与 CLI 参数

```bash
# 1. 运行单机 4 卡 DDP SFT Pilot 训练（7,000 条子集，指定 --allow-subset）
torchrun --standalone --nproc_per_node=4 train/train_sft.py \
    --manifest /data/mega-asr/manifests/pilot_sft.jsonl \
    --config configs/train/qwen3_asr_v100.yaml \
    --output-dir /data/mega-asr/runs/sft_pilot \
    --max-steps 500 \
    --save-steps 50 \
    --allow-subset

# 2. 运行受控参数 SFT Pilot 训练（300 steps, lr=1e-5, warmup=50, linear decay, balanced 采样）
torchrun --standalone --nproc_per_node=4 train/train_sft.py \
    --manifest /data/mega-asr/manifests/pilot_sft.jsonl \
    --config configs/train/qwen3_asr_sft_controlled.yaml \
    --output-dir /data/mega-asr/runs/sft_pilot_controlled \
    --max-steps 300 \
    --save-steps 50 \
    --learning-rate 1e-5 \
    --warmup-steps 50 \
    --lr-scheduler linear \
    --sample-strategy balanced \
    --allow-subset

# 3. 运行全量正式 SFT 训练（严格要求 DATASET_COMPLETE.json 状态为 PASSED）
torchrun --standalone --nproc_per_node=4 train/train_sft.py \
    --manifest /data/mega-asr/manifests/sft_train.jsonl \
    --config configs/train/qwen3_asr_v100.yaml \
    --output-dir /data/mega-asr/runs/sft_full \
    --max-steps 10000 \
    --save-steps 500

# 4. 单机 4 卡 DDP Smoke 训练 10 步并保存检查点
torchrun --standalone --nproc_per_node=4 train/train_sft.py \
    --manifest /data/mega-asr/manifests/smoke.jsonl \
    --config configs/train/qwen3_asr_v100.yaml \
    --output-dir /data/mega-asr/runs/sft_smoke \
    --max-steps 10 \
    --save-steps 10

# 5. 单机 4 卡 DDP 从 step 10 恢复并继续训练 2 步至 step 12
torchrun --standalone --nproc_per_node=4 train/train_sft.py \
    --manifest /data/mega-asr/manifests/smoke.jsonl \
    --config configs/train/qwen3_asr_v100.yaml \
    --output-dir /data/mega-asr/runs/sft_smoke \
    --resume-from-checkpoint /data/mega-asr/runs/sft_smoke/checkpoints/step_10 \
    --max-steps 12 \
    --save-steps 2

# 6. 导出合并权重（执行 merge_and_unload）
python3 train/train_sft.py \
    --export-merged \
    --checkpoint-dir /data/mega-asr/runs/sft_smoke/checkpoints/step_12 \
    --output-dir /data/mega-asr/runs/sft_smoke/merged_base

# 7. 运行单机 4 卡 DDP DPO Pilot 训练（使用纯真实偏好对与预计算参考对数概率，支持全额 held-out 验证）
torchrun --standalone --nproc_per_node=4 train/train_dpo.py \
    --manifest /data/mega-asr/manifests/pilot_dpo_pairs.jsonl \
    --val-manifest /data/mega-asr/manifests/val_dpo_pairs.jsonl \
    --config configs/train/qwen3_asr_dpo.yaml \
    --output-dir /data/mega-asr/runs/dpo_pilot \
    --max-steps 150 \
    --save-steps 25 \
    --eval-steps 25

# 8. 导出 DPO 合并权重作为 RL 阶段底座
python3 train/train_dpo.py \
    --export-merged \
    --checkpoint-dir /data/mega-asr/runs/dpo_pilot/checkpoints/step_150 \
    --output-dir /data/mega-asr/runs/dpo_pilot/merged_base

# 9. 运行单机 4 卡 DDP RL (GRPO) Pilot 训练（Group Size G=4, temp=0.7, top_p=0.9, kl_beta=0.04）
torchrun --standalone --nproc_per_node=4 train/train_rl.py \
    --manifest /data/mega-asr/manifests/pilot_rl.jsonl \
    --val-manifest /data/mega-asr/manifests/rl_val_pool.jsonl \
    --config configs/train/qwen3_asr_rl.yaml \
    --output-dir /data/mega-asr/runs/rl_pilot \
    --max-steps 60 \
    --save-steps 10 \
    --eval-steps 10

# 10. 导出 RL 合并发布权重（执行 merge_and_unload）
python3 train/train_rl.py \
    --export-merged \
    --checkpoint-dir /data/mega-asr/runs/rl_pilot/checkpoints/step_60 \
    --output-dir /data/mega-asr/runs/rl_pilot/merged_base
```


### 主要输入与输出

- **输入 Manifest**：符合合同规范的 JSONL，每行包含 `sample_id`、`audio`、`text`、`language` 等。
- **输出 Run 目录**：
  - `resolved_config.yaml`：解析所有默认参数后的最终运行配置；
  - `environment.json`：Python、PyTorch、CUDA、GPU 驱动与库版本；
  - `manifest_sha256.json`：输入 manifest 哈希、条数与元数据；
  - `pipeline_state.json`：当前阶段、global step 与最后有效 checkpoint；
  - `loss_log.jsonl`：每步 loss、learning rate、耗时日志；
  - `checkpoints/step_<N>/`：包含 `adapter/`、`optimizer.pt`、`scheduler.pt`（若配置了学习率调度器；若 `warmup_steps=0` 且无调度器时，`training_state.json` 中显式记录 `"scheduler": null`，不落盘 `scheduler.pt`）、`rng_state_rank_<rank>.pt`、`training_state.json`。

## 维护要求

新增或修改训练 runner 时，必须同步更新本 README、根 README、架构/开发/测试文档和对应测试。训练器必须确保：
1. LoRA target 必须与合同指定的 199 个 Linear 完全一致，严禁侵入 audio encoder 内部层；
2. 训练配置、随机种子与输入 manifest 散列必须随 run 落盘；
3. 断点恢复必须保证 global step、优化器与调度器状态无缝连续；
4. 提交前必须运行单元测试与目录 README 合同测试。
