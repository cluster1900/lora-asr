# train 目录说明

## 目录职责

本目录维护 V100 服务器（4 × Tesla V100-SXM2-32GB）上的模型训练 Runner，覆盖 SFT（监督微调）、DPO（直接偏好优化）与 RL（强化学习，GRPO）三个阶段。所有训练必须面向官方 `qwen-asr` / Transformers API，基于底层 `thinker` 模块。默认精确注入 199 个 Linear LoRA targets；`lora.train_audio_projections: false` 时去掉 3 个 audio tower projection，保留 196 个 decoder target。采用单机 4 卡 DDP（或单卡模式）、FP16 精度、eager attention 与梯度检查点，严格落地 10+2 步断点续训与 `merge_and_unload()` 权重接力生命周期。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `__init__.py` | 包声明与初始化文件。 |
| `train_sft.py` | SFT 阶段训练 Runner：支持单机 4 卡 DDP 与 `--single-gpu` 调试模式；注入 199 个 Linear LoRA 目标（Projection 3 + Decoder 196）；实现标签掩码 Cross-Entropy 损失与分层均衡 validation loss 评估；支持 10+2 检查点保存与断点恢复（optimizer、scheduler、RNG、training_state）；提供 `--export-merged` 执行 `merge_and_unload()` 权重导出。 |
| `train_dpo.py` | DPO 阶段训练 Runner：支持单机 4 卡 DDP 与 `--single-gpu` 调试模式；继承 199 个 Linear LoRA 目标；基于预计算的参考对数概率（或在线计算）实现 Bradley-Terry 偏好损失函数，记录隐式奖励差与 preference accuracy；提供全状态断点保存/恢复与 `--export-merged` 权重导出。 |
| `train_rl.py` | RL (GRPO) 阶段训练 Runner：支持单机 4 卡 DDP 与 `--single-gpu`。Policy 与冻结 reference 都从 DPO 合并模型注入 LoRA。缺省 199 个 Linear；服务器副本在 `lora.train_audio_projections: false` 时经 `filter_lora_targets` 去掉 3 个音频投影，剩下 196 个。v10/v11 第一轮是 1 条贪心加 3 条 `temperature=1.0`、`top_p=0.95`、`top_k=50` 采样；未高出贪心 0.02 时再抽 8 条，反向长度固定为 2。优势以贪心为锚。服务器副本读取 `grpo.advantage.mode`：`unit` 为 1，`raw_gap` 为原始奖励差，`fixed` 为 `fixed_value`，`capped_gap` 为不超过 `cap` 的原始奖励差。配置里有 `local_max_relative` 时，过远的获胜样本不进反向。`train.apply_learning_rate_on_resume: true` 时，服务器副本在加载 `scheduler.pt` 之后调用 `apply_configured_learning_rate`，把学习率写成 YAML 的值。`grpo.policy_token_mask: changes_only` 时，服务器副本只在获胜句相对贪心句的替换和插入 token 上累积策略梯度，KL 仍用整句；缺省 `all` 时策略项覆盖全部 response token。掩码为空或全是 0 时，服务器副本在 `changes_only` 的更新上直接报错。`grpo.policy_token_mask: signed_edits` 时，服务器副本调用 `signed_edit_loss_args`：序列优势是 `(-gap, gap)`，贪心掩码只留删除，获胜掩码只留替换和插入；token 相同则这次不更新。本地这份文件还没有这段调用，不要覆盖服务器副本。KL 为 Schulman K3，β=0.04，策略项为序列求和。`sample_strategy: degraded` 时 clean 行不进优化器。服务器副本在 `sample_strategy: degraded_skip_regressed` 时调用 `degraded_skip_regressed_indices`，再去掉场景 `noise` 和 `recording`。`--sample-strategy` 的可选值包含这个名字。本地这份文件还没有这次调用，不要覆盖服务器副本。Rollout 按组审计。Step 0 写入贪心 Full Held-out。只有 `gate.json` 为 PASSED 时才 `--export-merged` 到该 run 自己的目录。本地这份文件还没有服务器上的锚定选择逻辑，不要覆盖服务器副本。`grpo.include_reference_candidate: true` 时，服务器副本在 `local_max_relative` 有值的选择里调用 `append_reference_candidate`，再交给 `local_winner_index`。缺省 false 时不追加参考文本。本地这份文件还没有这次调用，不要覆盖服务器副本。 |
| `rl_resume_lr.py` | 恢复检查点之后，把优化器参数组和调度器 `base_lrs` 写成配置里的学习率。不改 `last_epoch`。非正学习率直接拒绝。 |
| `rl_lora_targets.py` | 按 `train_audio_projections` 过滤 LoRA 模块名。false 去掉以 `audio_tower.` 开头的名字，true 原样保留。 |
| `rl_pair_advantage.py` | 长度 2 的获胜优势：`unit`、`raw_gap`、`fixed`、`capped_gap`。非法模式、越界的固定值或越界的上限直接拒绝。 |
| `rl_local_winner.py` | 局部改正过滤。英文按词、中文按字。编辑距离超过 `max(2, floor(0.35 × 较长一边))` 的样本不训练。平局保留较小下标。 |
| `rl_policy_mask.py` | 长度 2 的策略梯度掩码。`changed_token_mask` 把与贪心 token 对齐的位置标成 0，替换和插入标成 1。`signed_edit_update` 在这些获胜 token 上放 `raw_gap`，在被删除的贪心 token 上放 `-raw_gap`；token 完全相同则 `skip`。只有删除时仍是 `update`，不抛错。`signed_edit_loss_args` 把这个结果收成序列优势 `(-gap, gap)` 和两条掩码。`scatter_keep` 把掩码放到移位后的 response label 上。KL 不在这里被掩掉。 |
| `rl_sample_strategy.py` | v26 的选行。`degraded_skip_regressed_indices` 保留退化行，去掉场景名 `noise` 和 `recording`，返回清单顺序的下标。一行都没有时抛错。打乱仍由训练器按种子做。 |
| `rl_reference_candidate.py` | v27 的候选追加。`append_reference_candidate` 在没有规范化相同的候选时，把参考文本和调用方给的奖励加到末尾。空参考文本不追加。是否训练仍由 `local_winner_index` 决定。 |
| `rl_v14_decision.py` | v14/v15 门禁写成之后的动作：`passed`、`continue`、`switch`、`stop`。不改训练器的停止表。`BLOCKED_TRANSFER` 且贪心增量仍 ≥ 0、调用方允许切换时返回 `switch`。 |
| `rl_v16_decision.py` | v16 门禁写成之后的动作：`passed`、`continue`、`stop`。Step 8 起 Robust 仍大于 0 就停止。`BLOCKED_TRANSFER` 在奖励非负且 Robust 已不高于 DPO 时返回 `continue`。 |
| `rl_v17_decision.py` | v17 门禁写成之后的动作：`passed`、`continue`、`stop`。Robust 比上一块回升、增量 ≥ `0.0005` 或贪心转负时停止。`BLOCKED_SEARCH` 在累计奖励质量 ≥ `8.50` 时返回 `continue`。horizon 48。上一块固定按 `step - 4` 取门禁。 |
| `rl_v19_decision.py` | v19 沿用 v17 的动作表。上一块 Robust 改读当前步之前最近一份 `gate_step_*.json`，这样落在 4 的倍数之外的 KL 停止仍能写成决定。 |
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

# 9. 正式入口是 scripts/run_rl_pilot_v11.sh。下面这条只说明训练器参数。
# v11：lr=2e-5、constant、最多 12 步、degraded。第一轮 1 条贪心 + 3 条 temperature=1.0 采样。
# 损失为 sequence_sum。验证集是 rl_val_pool.jsonl 的全部贪心行。
# 块未走完 horizon 时传 --no-export-merged-on-finish。不要改成 balanced。
torchrun --standalone --nproc_per_node=4 train/train_rl.py \
    --manifest /data/mega-asr/manifests/pilot_rl.jsonl \
    --val-manifest /data/mega-asr/manifests/rl_val_pool.jsonl \
    --wer-manifest /data/mega-asr/manifests/validation.jsonl \
    --config configs/train/qwen3_asr_rl_v11.yaml \
    --output-dir /data/mega-asr/runs/rl_pilot_v11 \
    --sample-strategy degraded \
    --no-export-merged-on-finish

# 10. 只有该步 gate.json 为 PASSED 时，才把合并权重写到这一 run 自己的目录。
# 把 step_<N> 换成那份 PASSED gate 的步数。held-out reward 最高本身不构成导出条件。
# 不写入 /data/mega-asr/runs/dpo_pilot_v2/merged_base。
python3 train/train_rl.py \
    --export-merged \
    --config configs/train/qwen3_asr_rl_v11.yaml \
    --checkpoint-dir /data/mega-asr/runs/rl_pilot_v11/checkpoints/step_<N> \
    --output-dir /data/mega-asr/runs/rl_pilot_v11/merged_base
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
