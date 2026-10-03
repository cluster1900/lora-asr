# scripts 目录说明

## 目录职责

本目录维护 V100 服务器的数据 builder、数据集准备脚本，以及 RL pilot 训练结束后的异步评测脚本。数据脚本必须从固定 source、音频 staging、manifest、泄漏检查和恢复语义出发，确保训练、验证、评测角色物理隔离无泄漏。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `download_sources.py` | E1 阶段原始数据源拉取与缓存：锁定 4 个 pinned revision，支持 HF 镜像（`hf-mirror.com`）与 OpenSLR 镜像断点续传，产出 `RAW_COMPLETE.json`。 |
| `stage_parquet_sources.py` | E1 阶段 Parquet 数据流式解码与音频转码标准化：从 HF Parquet 文件直接流式解包音频 bytes，通过 `ffmpeg` 转码为 16kHz mono PCM 16-bit WAV，校验时长并产出 `staged_<source>.jsonl` 与 `PROCESSED_COMPLETE.json`。 |
| `build_robust_manifests.py` | E1 阶段角色划分与门禁构建器：摄入各源 staged 样本，执行哈希分区无泄漏角色分配（7 个角色）、抽取 128 条平衡 smoke 子集与 pilot 子集、rejects 记录、SHA-256 校验和以及各级完成门禁（含 `DATASET_COMPLETE.json`）。 |
| `build_dpo_pairs.py` | E5 阶段 DPO 偏好对构建与参考对数概率预计算器：基于真实的 Base 预测与 SFT Step 50 预测（必须显式提供 `--base-predictions` 与 `--sft-predictions`），经 WER/CER 排序构造真实的 `(chosen, rejected)` 偏好对；正式模式严禁使用 gold 答案或合成负例兜底（杜绝 gold 泄漏）；严格过滤平局（`cand_sft == cand_base` 或 `err_sft == err_base`）；使用冻结 SFT 合并底座离线预计算 `ref_chosen_logp` 与 `ref_rejected_logp`，记录完整 Provenance 哈希并生成 `dpo_pair_audit.json`。 |
| `finish_rl_pilot_v7.sh` | 在 V100 上等待已启动的 `rl_pilot_v7` 训练结束，再按 held-out 最优检查点优先的顺序，对每个已保存 step 跑 2,867 条 `validation.jsonl` 推理和 `verify_gate.py`。状态写到 run 目录的 `pipeline_followup.json`。 |
| `run_rl_pilot_v10.sh` | v10 的探针和 4/8/12 步退化集锚定 GRPO 驱动，学习率 `1e-5`，配置 `configs/train/qwen3_asr_rl.yaml`。该运行已在 Step 8 结束，状态 `BLOCKED_TRANSFER`。本仓库副本默认拒绝重跑；服务器上 2026-09-29 执行的那份没有这道门闩。不写入 DPO Champion。 |
| `run_rl_pilot_v11.sh` | v11 的探针和 4/8/12 步锚定 GRPO 驱动，学习率 `2e-5`，配置 `configs/train/qwen3_asr_rl_v11.yaml`。该运行已在 Step 7 结束，状态 `STOPPED_KL`，`gate_step_7.json` 为 FAILED。本仓库副本没有重跑门闩：再执行会先跑探针，然后进入 `run_chunk 4`。不要再启动。设计内停止时评 `pipeline_state.global_step`。不写入 DPO Champion。 |
| `run_rl_pilot_v11_recover.sh` | 等 `run_rl_pilot_v11.sh` 退出后做一次恢复：先给 `pipeline_state.global_step` 的完整检查点补贪心 held-out 和 2,867 条门禁。训练进程异常退出且上一块是 `CHUNK_DONE` 时，从最后一份含 optimizer 的检查点恢复下一块，同一块只恢复一次。设计内停止（Robust、KL、奖励下跌、`BLOCKED_*`、门禁 PASSED）不恢复训练，也不从 Step 0 重开。 |
| `score_rl_greedy_held_out.py` | 给一个已保存的 RL 检查点追加一条贪心 Full Held-out 记录。调用训练器的 `evaluate_rl_validation(decode_mode=greedy)`，不创建优化器，不导出合并权重。合格记录已存在时不重复写；`assigned` 或 `rollouts` 不是 1,698 时拒绝写入。`--resolve-step` 从 `pipeline_state.json` 的 `global_step` 找出要评的检查点。 |
| `score_rl_pilot_v11_checkpoint.sh` | 对 `rl_pilot_v11` 的某一个已保存 step 依次补贪心 held-out、2,867 条 `validation.jsonl` 预测和 `gate_step_<N>.json`。不调用 `train_rl.py`。门禁 FAILED 仍视为评测完成；预测行数不足或贪心行不合格时退出非 0。结束行 `SCORE_DONE` 从 `metrics.held_out_reward_improvement` 读取奖励增量。Step 7 已于 2026-09-29 22:33 CST 写成 FAILED。 |
| `run_rl_pilot_v12.sh` | 从 DPO Champion 启动 `rl_pilot_v12`（学习率 `1e-5`，获胜优势为原始奖励差，最多 8 步）。Step 4 门禁之后，若裁剪前 `grad_norm` 中位数仍大于 1.5，再从 Champion 启动 `rl_pilot_v13`（固定优势 `0.10`）。不写 v10、v11 和 DPO Champion。v12 已有 `pipeline_state.json` 但没有 `switch_decision.json` 时拒绝重跑。该运行已在 2026-09-30 11:18 CST 停在 Step 4，动作 `stop`。不要再启动。 |
| `run_rl_pilot_v14.sh` | 从 `rl_pilot_v12/checkpoints/step_4` 续训到新目录 `rl_pilot_v14`，学习率 `1e-5`，优势仍是原始奖励差。v14 已在 Step 8 写成 `switch`，v15 已在 Step 4 写成 `stop`。两道门禁都是 FAILED。再执行会读到已有决定并退出。不要靠删决定来重跑。 |
| `run_rl_pilot_v16.sh` | 从 DPO Champion 启动 `rl_pilot_v16`。学习率 `1e-5`，优势是原始奖励差，获胜样本还必须是贪心文本的局部改正。该运行已在 2026-10-01 11:44 CST 停在 Step 8，动作 `stop`，门禁 FAILED。再执行会读到已有决定并退出。不要靠删决定来重跑。 |
| `run_rl_pilot_v17.sh` | 从 DPO Champion 启动 `rl_pilot_v17`。学习率 `1e-5`，局部改正的阈值仍是 `0.35`，获胜优势改为 1。该运行已在 2026-10-01 13:13 CST 停在 Step 4，动作 `stop`，门禁 FAILED。再执行会读到已有决定并退出。不要靠删决定来重跑。 |
| `run_rl_pilot_v18.sh` | 只读恢复 v16 Step 8 到新目录 `rl_pilot_v18`。该运行已在 2026-10-01 14:08 CST 停在 Step 10，训练状态 `STOPPED_KL`，门禁 FAILED。跟步时缺少 `gate_step_6.json`，没有写成决定。目录已存在，再执行会拒绝。不要靠删目录来重跑。 |
| `run_rl_pilot_v19.sh` | 只读恢复 v16 Step 8 到新目录 `rl_pilot_v19`。学习率 `1e-5`，获胜优势是不超过 `0.05` 的奖励差。该运行已在 2026-10-01 15:34 CST 停在 Step 12，动作 `stop`，门禁 FAILED。再执行会读到已有决定并退出。不要靠删决定来重跑。 |
| `run_rl_pilot_v20.sh` | 从 DPO Champion 新开 `rl_pilot_v20`。学习率 `5e-6`，优势仍是原始奖励差，局部阈值 `0.35`。2026-10-01 17:11 CST 在 Step 4 停止，动作 `stop`，门禁 FAILED。目录已存在，再执行会读到决定并退出。 |
| `run_rl_pilot_v21.sh` | 从 DPO Champion 新开 `rl_pilot_v21`。学习率 `1e-5`，原始奖励差，局部阈值 `0.35`，不训练 3 个音频投影。2026-10-01 19:12 CST 前在 Step 4 停止，动作 `stop`，门禁 FAILED。目录已存在，再执行会读到决定并退出。 |
| `run_rl_pilot_v22.sh` | 只读恢复 `rl_pilot_v21` 的 Step 4 到新目录 `rl_pilot_v22`。学习率、优势和音频投影开关与 v21 相同。复制 v21 损失日志。不写 v21 或 DPO Champion。2026-10-01 21:02 CST 在 Step 11 停止，动作 `stop`，门禁 FAILED。目录已存在，再执行会读到决定并退出。 |
| `run_rl_pilot_v23.sh` | 只读恢复 `rl_pilot_v22` 的 Step 8 到新目录 `rl_pilot_v23`。恢复后把学习率写成 `5e-6`。优势、β、KL 天花板和音频投影开关不变。损失日志只保留到 Step 8。不写 v22 或 DPO Champion。2026-10-01 22:15 CST Step 12 动作 `stop`，门禁 FAILED。目录已存在，再执行会读到决定并退出。 |
| `run_rl_pilot_v24.sh` | 从 DPO Champion 新开 `rl_pilot_v24`。学习率 `1e-5`，原始奖励差，局部阈值 `0.35`，音频投影仍开，199 个目标。策略梯度只打在相对贪心句发生变化的 token 上，KL 仍按整句。2026-10-02 00:13 CST Step 4 门禁 FAILED，动作 `continue`。00:19 CST 续到 Step 8 的那一块退出码 1，原因是一次更新的获胜句没有替换或插入 token。目录已存在，决定是 `continue`，再执行会拒绝。不要删决定，不要重跑。不写 v10 到 v23 或 DPO Champion。 |
| `run_rl_pilot_v25.sh` | 从 DPO Champion 新开 `rl_pilot_v25`。学习率 `1e-5`，原始奖励差，局部阈值 `0.35`，音频投影仍开，199 个目标。替换和插入仍用正的 `raw_gap`，被删除的贪心 token 用负的 `raw_gap`，token 完全相同则跳过。2026-10-02 02:52 CST Step 4 门禁 FAILED，动作 `stop`。目录已存在，再执行会读到决定并退出。不要删决定，不要重跑。不写 v10 到 v24 或 DPO Champion。 |
| `run_rl_pilot_v26.sh` | 从 DPO Champion 新开 `rl_pilot_v26`。学习率、优势、掩码和 199 个目标与 v25 相同。`--sample-strategy degraded_skip_regressed` 让 `noise` 和 `recording` 不进优化器。2026-10-02 04:39 CST Step 4 门禁 FAILED，动作 `stop`，贪心 `−0.0005`，Robust `+0.000359`。目录已存在，再执行会读到决定并退出。不要删决定，不要重跑。不写 v10 到 v25 或 DPO Champion。 |
| `run_rl_pilot_v27.sh` | 从 DPO Champion 新开 `rl_pilot_v27`。学习率、优势、掩码和 199 个目标与 v25 相同。采样策略回到 `degraded`。`include_reference_candidate: true` 时，参考文本只有落在局部编辑距离内才进入更新。准备阶段要求训练器含 `append_reference_candidate(`，并保留 v26 的 `stop` 决定。2026-10-02 05:13 CST 启动，驱动 PID 133766。07:10 CST Step 7 门禁 FAILED，动作 `stop`：贪心 `+0.0002`，Robust `+0.000014`，训练状态 `STOPPED_KL`。不写 v10 到 v26 或 DPO Champion。不要从 Step 7 续训。 |
| `run_rl_pilot_v28.sh` | 只读恢复 v27 Step 4 到新目录 `rl_pilot_v28`。`apply_learning_rate_on_resume: true` 把学习率写成 `5e-6`。参考文本、`signed_edits`、`raw_gap`、局部阈值 `0.35` 和 199 个目标保持不变。准备阶段截断损失日志到 Step 4，要求 v27 的决定是 `stop`。不恢复 Step 7，不写 v27 或 DPO Champion。 |
| `score_rl_pilot_checkpoint.sh` | 给 `RL_RUN_DIR` 和 `RL_CONFIG` 指定的检查点补贪心 held-out、2,867 条预测和 gate。默认写 `gate_step_<N>.json`；设置 `RL_GATE_PROFILE=pilot_feasibility` 时写独立的 `pilot_gate_step_<N>.json`。拒绝把 v10、v11、v12 或 DPO Champion 当作目标。不调用 `train_rl.py`，不导出权重。 |
| `sync_rl_pilot_v7_results.sh` | 把 V100 上的 v7 日志、gate、预测和 adapter 配置拉回本地 `results/rl_pilot_v7/`。不拉取 optimizer 和合并后的 safetensors。 |
| `README.md` | 说明当前目录职责、文件清单、CLI 参数与维护要求。 |

## 使用入口与 CLI 参数

```bash
# 1. 下载原始数据源并记录 RAW_COMPLETE.json
python3 scripts/download_sources.py \
    --config configs/data/public_robust_v100.yaml \
    --raw-dir /data/mega-asr/data/raw \
    --endpoint https://hf-mirror.com

# 2. 流式解包 Parquet 并通过 ffmpeg 转码为 16kHz mono WAV（支持指定 source 或 all）
python3 scripts/stage_parquet_sources.py \
    --config configs/data/public_robust_v100.yaml \
    --source all \
    --endpoint https://hf-mirror.com

# 3. 哈希隔离分区并产出全套 7 角色 manifest、smoke.jsonl、pilot 子集与 DATASET_COMPLETE.json
# （在数据未达全额配额时使用 --no-strict-quotas 标记 NON_STRICT_SUBSET，使用 --freeze-validation 保持核心 validation.jsonl 绝对冻结）
python3 scripts/build_robust_manifests.py \
    --config configs/data/public_robust_v100.yaml \
    --staged-dir /data/mega-asr/manifests \
    --output-dir /data/mega-asr/manifests \
    --data-dir /data/mega-asr/data \
    --mode full \
    --no-strict-quotas \
    --freeze-validation

# 4. 深度核验已有 manifest 门禁与音频完整性
# 非全量模式校验子集门禁（预期 VERIFIED (NON_STRICT_SUBSET)）
python3 scripts/build_robust_manifests.py --config configs/data/public_robust_v100.yaml --mode verify-only --verify-audio
# 严格模式校验全额配额（全量达标前判为 FAILED）
python3 scripts/build_robust_manifests.py --config configs/data/public_robust_v100.yaml --mode verify-only --strict-quotas

# 5. 构建 DPO 偏好对并预计算参考模型对数概率（强制要求传入真实 Base 与 SFT 预测）
python3 scripts/build_dpo_pairs.py \
    --candidate-manifest /data/mega-asr/manifests/pilot_dpo.jsonl \
    --base-predictions /data/mega-asr/runs/dpo_candidates/base_pilot_dpo_predictions.jsonl \
    --sft-predictions /data/mega-asr/runs/dpo_candidates/sft_step50_pilot_dpo_predictions.jsonl \
    --output-manifest /data/mega-asr/manifests/pilot_dpo_pairs.jsonl \
    --rejects /data/mega-asr/manifests/rejects_dpo.jsonl \
    --audit-output /data/mega-asr/runs/dpo_pair_audit.json \
    --ref-model-dir /data/mega-asr/runs/sft_pilot_controlled/merged_base \
    --target-pairs 3000

# 6. v7 已经在 V100 上启动后，用这份脚本收尾。它固定 balanced 采样和采样 held-out。
#    v10/v11 使用各自的驱动。只改 RL_RUN_DIR 不能把本脚本拿去跑后续 pilot。
bash scripts/finish_rl_pilot_v7.sh

# 7. v7 收尾状态变为 DONE 后，在本地拉回 gate 和预测
bash scripts/sync_rl_pilot_v7_results.sh

# 8. v10 已结束（Step 8，BLOCKED_TRANSFER）。下面这条命令会直接退出。
bash scripts/run_rl_pilot_v10.sh

# 9. v11 已结束（Step 7，STOPPED_KL，gate_step_7.json 为 FAILED）。
#    run_rl_pilot_v11.sh 没有重跑门闩，再执行会进入 run_chunk 4。不要启动。
#    run_rl_pilot_v11_recover.sh 见到 STOPPED_KL 不加步。Step 7 已评完，不要再补评。

# 10. v12 已结束（Step 4，动作 stop，gate_step_4.json 为 FAILED）。
#     run_rl_pilot_v12.sh 再执行会读到已有决定并退出，不要靠删决定来重跑。

# 11. v14 和 v15 都已结束，门禁 FAILED。再执行会读到已有决定并退出。
bash scripts/run_rl_pilot_v14.sh

# 12. v16 已结束（Step 8，动作 stop，门禁 FAILED）。再执行会读到已有决定并退出。
bash scripts/run_rl_pilot_v16.sh

# 13. v17 已结束（Step 4，动作 stop，门禁 FAILED）。再执行会读到已有决定并退出。
bash scripts/run_rl_pilot_v17.sh

# 14. v18 已结束（Step 10，STOPPED_KL，门禁 FAILED）。目录已存在，再执行会拒绝。
bash scripts/run_rl_pilot_v18.sh

# 15. v19 已结束（Step 12，动作 stop，门禁 FAILED）。再执行会读到已有决定并退出。
bash scripts/run_rl_pilot_v19.sh

# 16. v20 已结束（Step 4，动作 stop，门禁 FAILED）。再执行会读到已有决定并退出。
bash scripts/run_rl_pilot_v20.sh

# 17. v21 已结束（Step 4，动作 stop，门禁 FAILED）。再执行会读到已有决定并退出。
bash scripts/run_rl_pilot_v21.sh

# 18. v22 已结束（Step 11，动作 stop，门禁 FAILED）。再执行会读到已有决定并退出。
bash scripts/run_rl_pilot_v22.sh

# 19. v23 已结束（Step 12，动作 stop，门禁 FAILED）。再执行会读到已有决定并退出。
bash scripts/run_rl_pilot_v23.sh

# 20. v24 已结束。Step 4 门禁 FAILED，决定是 continue，随后 chunk 8 退出码 1。再执行会拒绝。不要删决定。
bash scripts/run_rl_pilot_v24.sh

# 21. v25 已于 2026-10-02 02:52 CST 停在 Step 4，动作 stop，门禁 FAILED。目录已存在时再执行会退出。不要删决定。
bash scripts/run_rl_pilot_v25.sh

# 22. v26 已于 2026-10-02 04:39 CST 停在 Step 4，动作 stop，门禁 FAILED。目录已存在时再执行会退出。不要删决定。
bash scripts/run_rl_pilot_v26.sh

# 23. v27 已于 2026-10-02 07:10 CST 停在 Step 7，动作 stop，门禁 FAILED。目录已存在时再执行会退出。不要删决定，不要从 Step 7 续训。
bash scripts/run_rl_pilot_v27.sh

# 24. v28 只读恢复 v27 Step 4，恢复后的学习率是 5e-6。目录已存在且决定为 continue 时再执行会拒绝。
bash scripts/run_rl_pilot_v28.sh
```

### RL 检查点可行性门禁

```bash
# 默认执行正式 release profile，输出 gate_step_<N>.json
RL_RUN_DIR=/data/mega-asr/runs/rl_pilot_v28 \
RL_CONFIG=/data/mega-asr/repo/configs/train/qwen3_asr_rl_v28.yaml \
bash scripts/score_rl_pilot_checkpoint.sh 4

# 只确认训练闭环和 checkpoint 可用，输出 pilot_gate_step_<N>.json
RL_GATE_PROFILE=pilot_feasibility \
RL_RUN_DIR=/data/mega-asr/runs/rl_pilot_v28 \
RL_CONFIG=/data/mega-asr/repo/configs/train/qwen3_asr_rl_v28.yaml \
bash scripts/score_rl_pilot_checkpoint.sh 4
```


### 主要输入输出与 Schema

- **输入配置**：`configs/data/public_robust_v100.yaml`（定义 4 个 pinned 数据源与 7 个数据角色）。
- **主要输出**：
  - `manifests/RAW_COMPLETE.json`：原始数据下载完成标记。
  - `manifests/PROCESSED_COMPLETE.json`：音频转码与标准化完成标记。
  - `manifests/smoke.jsonl`：128 条平衡 smoke 数据（32 en + 32 zh + 64 degraded），供 Base 推理与续训 Smoke 使用。
  - `manifests/<role>.jsonl`：7 个角色（`sft_train`, `dpo_train_pool`, `dpo_val_pool`, `rl_train_pool`, `rl_val_pool`, `validation`, `bench_test`）的 JSONL manifest。
  - `manifests/pilot_<name>.jsonl`：pilot 子集 manifest。
  - `manifests/rejects.jsonl`：校验未通过或清洗丢弃的样本记录（含丢弃原因）。
  - `manifests/<role>_COMPLETE.json`：各角色完成指标与 hash。
  - `manifests/manifest_sha256.json`：所有 manifest 的 sha256 汇总。
  - `manifests/DATASET_COMPLETE.json`：E1 阶段总验收门禁文件（确认无跨角色泄漏与配额达成）。

## 维护要求

新增、重命名或删除脚本时，必须同步更新“文件清单”和仓库根 README。修改数据源、配额、schema、
输出文件、恢复策略或泄漏规则时，必须同步更新本 README、`docs/qwen3-asr/03_data_plan.md`、
开发/测试/进度文档和对应测试。
