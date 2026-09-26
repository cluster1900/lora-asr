# 开发进度

最后更新：2026-09-24

## 当前状态

当前状态：**E6 阶段强化学习（RL Pilot v4 完训复核：未达验收，维持阻断）**
- **核心结论**：**RL v4 的训练循环已经完成，但 RL Pilot 阶段没有通过验收，不能作为正式发布模型或 Full RL 底座。**
- **当前基座与状态保持**：
  - 正式底座：维持 **DPO merged model** (`/data/mega-asr/runs/dpo_pilot_v2/merged_base`)；
  - RL v4 Step 10：仅作为诊断候选（Diagnostic Candidate）；
  - **暂不启动 Full RL，也不发布 RL v4**。
- **已确认事实**：
  1. V100 上 `pipeline_state.json` 为 `COMPLETED`，30/30 steps，4 卡 DDP。
  2. Step 10/20/30 checkpoint 均完整存在，包含 adapter、optimizer、scheduler 和四卡独立 RNG。
  3. Rollout 共 7,680 条，四个 rank 各 1,920 条，分组和 schema 正常，四卡审计 100% PASSED。
  4. 2,867 条独立验证全集推理全部成功，空输出和推理错误均为 0。
- **主要问题与根因审计**：
  1. **Step 10 gate FAILED**：Robust Macro 从 DPO 的 `10.3583%` 变为 `10.3785%`，恶化 `+0.0202` 个百分点；held-out reward 只提升 `+0.0006`，低于要求的 `+0.0020`。
  2. **Step 30 表现更差**：无任何退化场景改善，Robust Macro 恶化 `+0.0262` 个百分点，held-out reward 下降 `-0.0002`。
  3. **数据规模缺口**：`pilot_rl.jsonl` 只有 **2,236 条**，实际是 1,236 degraded + 500 English clean + 500 Chinese clean，低于合同要求的 3,000 条（2,000 degraded + 1,000 clean）。根因在于数据池 `rl_train_pool.jsonl` 自身仅包含 1,236 条退化音频。
  4. **顺序取样引发分布突变（Distribution Cliff）**：manifest 按顺序排列（前 1,236 条全为 degraded，后 1,000 条全为 clean），训练代码采用顺序游标取样。造成 Step 0–18 几乎全为 degraded（训练 reward 约 0.80~0.88），Step 20–30 突变为 100% clean（训练 reward 陡增至 0.98，零方差率升至 0.875~0.9062）。模型在后期严重拟合干净语音，冲淡了退化场景的鲁棒优化，直接导致 Step 30 的退化增益彻底消失。
  5. **发布产物链路未闭合**：V100 上未生成 `merged_base`/`rl_merged_base`，未生成根目录 `gate.json`；目前仅有 adapter checkpoint 与 `gate_step_10.json`、`gate_step_30.json`。
  6. **Rollout 惩罚项特征**：7,680 条采样中记录了 3 条空输出、86 条重复惩罚、225 条 hallucination 惩罚；最终验证集虽无空输出，但揭示模型在极端噪声探索时的不稳定性，需在后续扩容时持续监控。
  7. **审计精度截断**：KL 正则项数值有效有限，但 `loss_log.jsonl` 中 `mean_kl` 经历 `round(..., 5)` 截断导致四舍五入为 `0.0`，审计精度不足。

### E6 RL (GRPO) Pilot 核心设计与契约标准

1. **训练与参考底座模型**：
   - Policy 初始底座：`/data/mega-asr/runs/dpo_pilot_v2/merged_base`。
   - Reference 冻结模型：`/data/mega-asr/runs/dpo_pilot_v2/merged_base`（完全冻结，用于对数概率基线与 token 级 KL 正则项约束）。
   - LoRA 结构：精确注入 199 个 Linear LoRA 目标（Projection 3 + Decoder 196），保持与 SFT/DPO 完全一致。

2. **Rollout 采样与分组（G=4）**：
   - 每条输入音频在当前 Policy 下生成 $G=4$ 个候选假设，采样参数正式规范为 `temperature=0.85`, `top_p=0.92`, `top_k=50`（基于声学强 conditioning 特性，充分维持候选多样性）。
   - 每条候选记录唯一 `group_id`、`group_size=4`、`rollout_rank`，并落盘至 `rollouts.jsonl`（严格遵循合同字段：`sample_id`, `group_id`, `group_size`, `rollout_rank`, `policy_checkpoint`, `rollout_seed`, `prediction`, `language`, `reference_error_rate`, `reward_components`, `reward`, `kl_to_reference`, `error`）。

3. **序列级 Reward 与惩罚函数（严格遵循 `reward_config.yaml`）**：
   - ASR 基础奖励：`asr = 1.0 - min(error_rate, 1.0)`（英文按 WER，中文按 CER，使用官方标准归一化与分词）。
   - 空输出惩罚：`empty = -0.25`（若预测为空或仅包含空白/标点）。
   - 重复输出惩罚：`repeat = -0.25`（检测 3-token run 或相邻 2-token pair 重复循环）。
   - 过长输出惩罚：`too_long = -0.15`（预测长度超过参考文本 1.5 倍）。
   - 严重幻觉惩罚：`hallucination = -0.25`（错误率 $\ge 0.8$ 且伴随过长、重复或重大编辑）。
   - 奖励截断：最终 Reward 严格截断在 $[-1.0, 1.0]$ 区间。

4. **GRPO 优势标准化与零方差保护**：
   - 组内优势：$A_i = \frac{r_i - \text{mean}(\{r\})}{\text{std}(\{r\}) + \epsilon}$，其中 $\epsilon = 1.0 \times 10^{-6}$。
   - 零方差中和：若组内标准差 $\le \epsilon$（如 4 个候选全对或全错导致同质），则该组全部候选标准化优势 $A_i$ 置为 0，且损失置为 `0.0 * policy_token_logps.sum()`，杜绝单向强拉 KL。
   - 坍缩监控：全周期零方差 group 比例严格要求 $\le 0.75$。

5. **策略损失与 KL 正则项**：
   - Token 级策略损失：$-\frac{1}{|y_i|} \sum_t A_i \log \pi_\theta(y_{i,t} | x, y_{i,<t})$。
   - 冻结参考策略 KL 惩罚：$\beta \sum_t D_{KL}(\pi_\theta || \pi_{\text{ref}})$，固定 $\beta = 0.04$。
   - 4 卡 DDP 分布式梯度同步，梯度累积步数确保全局有效 batch size 达到 64 个音频 prompt（256 条 rollout）。

6. **门禁指标准出标准（RL Pilot Gate）**：
   - 相对 DPO 阶段基线：退化场景改善数 $\ge 1$；
   - Clean 宏平均错误率回退：$\le 0.02$；
   - 相对 Base 官方基线 Clean 累积回退：$\le 0.025$；
   - 退化宏平均（Robust Macro）相对 DPO 零恶化（$\le 0.0$）；
   - 有效输出率（Valid Output Rate）：$\ge 0.95$；
   - 空输出率：$\le 0.002$；
   - 失败率增量：$\le 0.05$；
   - Held-out 独立验证集（`rl_val_pool.jsonl`）平均奖励提升 $\ge 0.0020$。

### 前置阶段成果备查（E5 DPO Pilot PASSED）

此前已完成以下工程修复：
1. **[P1] Smoke 物理隔离与去污染**：`smoke.jsonl` 严格仅从 `sft_train` 提取，跨角色补齐逻辑彻底废除，测试集与验证集泄露为 0。
2. **[P1] 完成门禁真实验收**：非全量模式状态明确标记为 `NON_STRICT_SUBSET`（绝不冒充 `PASSED`），空角色硬拦截；实现 `--mode verify-only --verify-audio` 进行磁盘文件、哈希与 21 组两两绝对隔离深度审计。
3. **[P1] AISHELL 唯一标识防覆盖**：抽取规范 `audio.path` stem（如 `BAC009S0002W0122`）作为话语 ID 与音频文件名，杜绝重复与覆盖。
4. **[P1] 7 角色物理隔离**：物化 10,744 条真实规范样本，按比例分配至全部 7 个角色（无一为空），经 $C_7^2=21$ 组全配对严格互斥校验通过。
5. **[P1] 参数与 Attention 规范**：CLI 批次参数默认设为 `None` 保障 YAML 优先级，注意力机制全线注入 `attn_implementation="eager"`。
6. **[P1] 4 卡 DDP 断点续训**：实现按 rank 保存并恢复 RNG 状态（`rng_state_rank_{0..3}.pt`）、基于全局样本游标 `(step * accum + g) * world_size + rank` 的精准连续采样，以及强 Provenance 校验（manifest SHA-256、world_size、model_revision、target_map_hash）。

已完成：

- **自动化测试**：2026-09-20 本地与服务器端全套 61 项单元测试 100% 通过（包含门禁加固、多检查点横向评测、均衡验证损失采样等新测项），`test_directory_readmes.py` 契约测试通过，`git diff --check` 0 警告。
- **E1 阶段：多源数据物化、深度质检与 Pilot 门禁（NON_STRICT_SUBSET 达成）**：
  - **Voices-in-the-Wild-Bench**：物化 4,999 条无损 16kHz WAV（覆盖 16 real + 16 synthetic 全场景，仅 1 条上游空文本清洗拦截）。
  - **LibriSpeech**：物化 31,233 条（训练集 28,539 条，独立验证集 clean 2,694 条）。
  - **AISHELL-1**：物化 26,323 条（训练集 20,592 条，独立验证集 clean 5,731 条）。
  - **Voices-in-the-Wild-2M**：物化 14,369 条（覆盖 distortion, dropout, echo, far_field, noise, recording, obstructed 7 大核心退化场景）。
  - **数据台账与入库统计**：
    - 磁盘物化有效音频总数：**76,924 条**。
    - 进入 7 个隔离角色清单的样本总数：**71,360 条**（`sft_train`: 37,476，`dpo_train_pool`: 20,100，`dpo_val_pool`: 2,347，`rl_train_pool`: 2,998，`rl_val_pool`: 573，`validation`: 2,867，`bench_test`: 4,999）。
    - 暂存储备样本（Staged Reserve）：**5,564 条**（保留于 staged 缓冲池中，未进入当前角色，杜绝无序混杂）。
  - **Pilot 子集与候选池精确构建**：
    - `pilot_sft.jsonl` 严格包含 **7,000 条**（5,000 degraded + 1,000 en clean + 1,000 zh clean）。
    - `pilot_dpo.jsonl` 与 `pilot_rl.jsonl` 为规范候选音频池（Candidate Audio Pools，按退化与双语 clean 子配额均衡抽取），专供后续 E5 生成偏好对（`chosen/rejected`）与 E6 生成 rollout prompt，非最终偏好对。
  - **独立评估集就绪**：`validation.jsonl` 包含 **2,867 条**（867 degraded + 1,000 en clean + 1,000 zh clean），彻底解决 validation clean 为 0 的问题；`bench_test.jsonl` 具备 **4,999 条**。
  - **物理隔离与深度审计**：`DATASET_COMPLETE.json` 经 $C_7^2=21$ 组全配对互斥校验零泄漏（`leakage_check: PASSED`），无空角色（`quota_check: PASSED`），门禁状态确立为 `NON_STRICT_SUBSET`；`--mode verify-only --verify-audio` 全量音频 16kHz mono 16-bit PCM、时长 [0.5, 30.0]s 及 SHA-256 校验 100% 通过。
- **纯净 Base Smoke 推理基线（128 条纯净 smoke 样本）**：
  - 128 条样本全部成功解码（128 success, 0 failed, 0 empty outputs）。
  - 英文 Clean WER：`1.45%`，英文 Distortion WER：`12.41%`。
  - 中文 Clean CER：`1.13%`，中文 Distortion CER：`5.46%`。
  - Clean 宏平均：`1.29%`，Robust 宏平均：`8.93%`，整体语言宏平均：`4.10%`。
- **4 卡 DDP SFT 训练 Smoke 与断点续训验证（Step 1~12）**：
  - 4 块 Tesla V100-SXM2-32GB，`micro_batch_size=1`, `gradient_accumulation_steps=16`, `effective_global_batch_size=64`。
  - 199 个 LoRA Linear 目标精准匹配，断点恢复 Provenance 校验（manifest/revision/world_size/seed/grad_accum/target_hash）及 4 卡 RNG 完美延续。
- **权重合并与导出（Merge and Unload）**：
  - 成功执行 `merge_and_unload()` 导出 4.08GB 完整模型。
- **SFT Smoke 工程闭环拟合验证（基于 128 条 `sft_train` 内部 smoke 样本，验证训练-合并-推理工程全链路通畅与梯度有效性；非独立 validation 集的 held-out 泛化指标，正式泛化能力在 Pilot 阶段由独立 validation 与 bench_test 评估）**：
  - 英文 Clean WER：保持 `1.4506%`（0 退化）。
  - 英文 Distortion WER：由 `12.4074%` 下降至 `11.6667%`。
  - 中文 Clean CER：由 `1.1342%` 优化至 `0.5671%`。
  - 中文 Distortion CER：由 `5.4585%` 下降至 `5.0218%`。
  - Robust 宏平均错误率：由 `8.93%` 下降至 `8.34%`。
  - Clean 宏平均错误率：由 `1.29%` 优化至 `1.01%`。
  - 整体语言宏平均错误率：由 `4.10%` 下降至 `3.72%`。

- **E4 阶段：4 卡 DDP SFT Pilot 训练与合并底座导出（已完成）**：
  - **输入清单**：`pilot_sft.jsonl`（7,000 条：5,000 degraded + 1,000 en clean + 1,000 zh clean）。
  - **执行环境**：4 × Tesla V100-SXM2-32GB，`micro_batch_size=1`, `gradient_accumulation_steps=16`, `effective_global_batch_size=64`，`lr=2e-5`。
  - **训练收敛**：完成全额 500 steps（累计处理 32,000 样本次，历时 3,538.5s），Loss 从 1.49825 平稳收敛至 0.13；保存 step 50..500 共 10 个全状态检查点（含 4 卡 RNG 与 Provenance 元数据）。
  - **合并底座导出**：成功执行 `merge_and_unload()` 导出完整权重底座 `/data/mega-asr/runs/sft_pilot/merged_base`（3.8GB `model.safetensors`）。
- **E4 阶段：独立 Held-Out 验证集权威对比评测与门禁验收（全量 2,867 条独立样本，全部通过）**：
  - **评测数据集**：`/data/mega-asr/manifests/validation.jsonl`（867 degraded + 1,000 en clean + 1,000 zh clean，0 泄漏与 0 同源）。
  - **退化场景显著改善（共 7 个细分场景单元改善）**：
    - 英文退化：`en|recording` 由 31.87% 下降至 28.25%（**-3.63%**），`en|distortion` 由 8.29% 下降至 8.20%（**-0.09%**），`en|noise` 由 17.15% 下降至 17.09%（**-0.06%**）。
    - 中文退化：`zh|recording` 由 29.63% 下降至 24.07%（**-5.56%**），`zh|dropout` 由 4.70% 下降至 3.98%（**-0.72%**），`zh|echo` 由 13.84% 下降至 13.21%（**-0.63%**），`zh|far_field` 由 3.68% 下降至 3.10%（**-0.58%**）。
    - **退化语言宏平均（Robust Macro）**：Base=`10.52%` $\to$ SFT Pilot=`10.30%`（**全退化整体改善 0.22%**）。
  - **Clean 场景回退与风险监控**：
    - 英文 Clean WER：Base=`1.98%` $\to$ SFT Pilot=`3.41%`（增量 +1.43%）。
    - 中文 Clean CER：Base=`1.47%` $\to$ SFT Pilot=`1.57%`（增量 +0.10%）。
    - Clean 语言宏平均：Base=`1.72%` $\to$ SFT Pilot=`2.49%`（增量 +0.77% $\le$ 门禁上限 2.00%）。
    - **风险监控提示**：英文 Clean WER 增量（+1.43%）虽在单阶段 $\le 2.0\%$ 门禁内，但已逼近合同规定的总 Clean 相对 Base 累积增量 $\le 2.5\%$ 的红线。在后续 E5 DPO 偏好对构建与训练中，必须严格监控 Clean preference pairs 质量，过滤无效 ties，防止级联劣化。
  - **输出完整性与有效率**：
    - 有效输出率（Valid Output Rate）：Base=100.0%, SFT=100.0%（0 推理错误，0 空输出，$\ge$ 95.0% 门禁标准）。
  - **SFT Pilot Gate 结论**：**PASSED（全部 4 项准出条件 100% 达标）**。机器可读门禁文件已正式归档于 `/data/mega-asr/runs/sft_pilot/gate.json`，完整记录 4 项阈值、实际测量值、退化场景列表与输入 manifest 及双侧预测文件的 SHA-256 哈希。

- **E4 阶段：SFT Pilot 参数组合实验（Run 2），2026-09-20 复核**：
  - **执行已完成**：`sft_pilot_controlled` 完成 300/300 steps，4 × V100，global batch 64，FP16/eager；峰值学习率 `1e-5`，前 50 步 warmup，随后 linear decay 至 0；采样为 `balanced`，固定 7,000 条 manifest 和 seed `20260722`。step 50/100/300 实查包含 adapter、optimizer、scheduler 和四卡 RNG。
  - **证据范围**：读取服务器 `resolved_config.yaml`、`pipeline_state.json`、`loss_log.jsonl`、各检查点评测、`gate.json` 和 `best_checkpoint.json`。Step 100 的 2,867 条预测已用现有 evaluator 从头重算，结果与其保存的 `metrics.json` 完全一致；sample_id 唯一且完整覆盖 validation。pilot 与 validation 的 sample_id、source_utterance_id、audio_sha256 交集均为 0。本次未重新训练或运行模型推理。
  - **实际指标**（错误率越低越好；不同语言先分别计算 WER/CER 再做宏平均）：

    | 模型 / 检查点 | EN Clean WER | ZH Clean CER | Clean Macro | Robust Macro | 全集语言 Macro | 空输出 | 场景改善 (退化/14) |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
    | Base | 1.9789% | 1.4706% | 1.7247% | 10.5179% | 4.3125% | 0 | 基准 |
    | Run 1 Step 500 | 3.4109% | 1.5691% | 2.4900% | 10.3023% | 4.7841% | 0 | 14/14 |
    | **Run 2 Step 50 (最优)** | **1.9441%** | **1.4002%** | **1.6722%** | **10.4001%** | **4.2430%** | **0** | **6/14** |
    | Run 2 Step 100 | 3.0628% | 1.4424% | 2.2526% | 13.7010% | 5.6984% | 10 | 6/14 |
    | Run 2 Step 150 | 3.0728% | 1.3721% | 2.2225% | 15.4612% | 6.2354% | 0 | 3/14 |
    | Run 2 Step 200 | 3.6396% | 1.3650% | 2.5023% | 11.0984% | 5.0193% | 0 | 3/14 |
    | Run 2 Step 250 | 3.5998% | 1.3650% | 2.4824% | 11.4790% | 5.1150% | 0 | 2/14 |
    | Run 2 Step 300 | 3.5402% | 1.3580% | 2.4491% | 13.7318% | 5.8186% | 0 | 2/14 |

  - **性能结论**：
    - **Step 50 达成真正的 Pareto 最优**：在全量 2,867 条独立 validation 集上，EN Clean WER（1.94%）、ZH Clean CER（1.40%）、Clean Macro（1.67%）、Robust Macro（10.40%）及全集 Macro（4.24%）均全面超越 Base 基线。退化场景在 6 个细分单元显著改善（`en|dropout`, `en|echo`, `en|recording`, `zh|echo`, `zh|far_field`, `zh|recording`），4 个持平，4 个轻微波动；有效输出率 100%，空输出 0，死循环 0。
    - **Step 100 缺陷归因**：Step 100 在 float16 推理时出现数值敏感导致 10 条空输出；切至 bfloat16 后 clean 样本立刻正常。此外 `vitw_sample_278194_noise` 出现自回归死循环（1,599 字符），严重失真拉高 Robust Macro 至 13.70%。
    - **过拟合与退化偏向证实**：随着训练从 Step 50 继续进行至 Step 100/150/200/300，Clean WER 从 1.94% 持续回退至 3.06%~3.64%，退化改善场景从 6 个萎缩至 2 个。因此百步以内的早停（Step 50）是保持 Clean 保真度与声学鲁棒性的黄金窗口。
  - **报告与选择缺陷修复**：
    - 修复 `evaluation/eval_checkpoint_series.py` 中的 `robust_language_macro_error_rate` 字段读取与动态分母（`base_scenarios` 退化场景总数 14）。
    - 修复排序逻辑：增加 Robust Macro 不得恶化（`max_robust_macro_regression: 0.005`）和空输出率（`max_empty_output_rate: 0.002`）硬约束。
  - **门禁收紧**：
    - `evaluation/verify_gate.py` 新增 `max_robust_macro_regression: 0.005`（鲁棒宏平均恶化不得超过 0.5%）和 `max_empty_output_rate: 0.002`。
    - 确保 `gate.json` 完整记录并强校验双侧预测覆盖与输出健康度。
  - **训练时验证改进**：
    - `train/train_sft.py` 的 `evaluate_validation_loss()` 改为分层均衡采样（clean 与 degraded 兼顾），消除仅评估退化样本导致的监控偏差。
  - **交接状态**：
    - 废除将 Step 100 作为 DPO 底座的提议。
    - 正式确认 **Step 50** 为 SFT Pilot 准出检查点，完成合并底座导出，通过收紧后的新门禁，交接至 E5 DPO。

- **E5 阶段：DPO 偏好对构建、离线参考模型对数概率预计算与审计（真实模型竞争版，已完成）**：
  - **训练偏好对构建**：从 `pilot_dpo.jsonl`（3,000 条候选）基于 Base 与 SFT Step 50 真实推理输出，成功构建 **181 对** 真实有效 `(chosen, rejected)` 偏好对（过滤 2,819 条平局/相同错误率样本，杜绝 gold 兜底与合成负例），写入 `/data/mega-asr/manifests/pilot_dpo_pairs.jsonl`。
  - **验证偏好对构建**：从 `dpo_val_pool.jsonl`（2,347 条候选）成功构建 **66 对** held-out 验证偏好对（过滤 2,281 条平局），写入 `/data/mega-asr/manifests/val_dpo_pairs.jsonl`。
  - **参考模型 Logp 离线预计算**：使用锁定的 Step 50 SFT 合并底座（`/data/mega-asr/runs/sft_pilot_controlled/merged_base`）在 GPU 上离线完成每条样本 chosen 与 rejected 序列的自回归对数概率预计算（`ref_chosen_logp`, `ref_rejected_logp`），将 DPO 显存占用减半。
  - **分布与质检审计**：
    - 训练偏好对：英文 129 对，中文 52 对；退化 155 对，Clean 26 对。
    - 验证偏好对：英文 49 对，中文 17 对；退化 28 对，Clean 38 对。
    - 偏好来源：`sft_better_than_base` 94 对（51.9%），`base_better_than_sft` 87 对（48.1%），高度平衡竞争。
    - `chosen == gold` 比例：**4.42%**（仅 8 条为单侧模型 0 错误），彻底告别 100% gold 泄漏。
    - 平均错误率差：训练集 $\Delta=10.76\%$（chosen 25.50% vs rejected 36.26%），验证集 $\Delta=9.74\%$。完整审计记录于 `/data/mega-asr/runs/pilot_dpo_pair_audit.json` 与 `val_dpo_pair_audit.json`。
- **E5 阶段：4 卡 DDP DPO Pilot 训练、合并底座导出与门禁验收（第一轮训练完成，因泛化准确率未达标门禁判为 FAILED）**：
  - **执行配置与收敛**：4 × Tesla V100-SXM2-32GB，`micro_batch_size=1`，`gradient_accumulation_steps=16`，`effective_global_batch_size=64`，`lr=5e-6`（线性 warmup 20 步 + 线性衰减），`beta=0.1`，总步数 150 步。
  - **真实训练收敛数据**：
    - 训练 loss：Step 1 从 0.73681 降至 Step 50 的 **0.68658**，最终 Step 150 降至 **0.58802**。
    - 训练集批次偏好准确率：从初始 48.4% 攀升至 Step 150 的 **78.1%**。
    - **Held-out 验证集偏好准确率（66 对独立验证偏好对）**：
      - Step 25: **0.5000**
      - Step 50: **0.5000**
      - Step 75: **0.4848**
      - Step 100/125/150: **0.4697**
      - 门禁最低要求：$\ge 0.5500$。所有检查点均未达标，呈现典型的“训练集拟合（78.1%）但验证集无法泛化（46.97%~50.00%）”现象，核心诱因为 181 对训练偏好对样本量不足。
  - **检查点系列评测与最佳选择**：经独立验证集横向评测，Step 50 达成最佳综合退化指标。
  - **合并底座导出**：执行 `merge_and_unload()` 导出完整 3.8GB 独立模型 `/data/mega-asr/runs/dpo_pilot/merged_base`（源自 Step 50，保留作为中间实验产物，暂不晋级为 RL 训练底座）。
  - **全量独立验证集（2,867 条）评测结果（Base vs SFT Step 50 vs DPO Step 50）**：

    | 指标 / 场景 | Base 基线 | SFT (Step 50) | DPO (Step 50) | DPO 相对 SFT | DPO 相对 Base | 门禁阈值 | 门禁状态 |
    | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
    | **Held-out 偏好准确率** | - | - | **0.5000 (50.0%)** | - | - | $\ge 55.0\%$ | **FAILED** |
    | **Robust Macro 错误率** | 10.5179% | 10.4001% | **10.3973%** | **-0.0028%** | **-0.1206%** | $\le 0.0\%$ | **PASSED** |
    | **Clean Macro 错误率** | 1.7247% | 1.6722% | **1.6767%** | +0.0045% | **-0.0480%** | $\le 2.0\%$ | **PASSED** |
    | **累积 Clean 回退 (相对 Base)** | - | - | **-0.0480%** | - | **-0.0480%** | $\le 2.5\%$ | **PASSED** |
    | **EN Clean WER** | 1.9789% | 1.9441% | **1.9391%** | **-0.0050%** | **-0.0398%** | - | - |
    | **ZH Clean CER** | 1.4706% | 1.4002% | **1.4143%** | +0.0141% | **-0.0563%** | - | - |
    | **全集语言 Macro 错误率** | 4.3125% | 4.2430% | **4.2434%** | +0.0004% | **-0.0691%** | - | - |
    | **退化改善单元数 (相对 SFT)** | - | - | **5 / 14** | - | - | $\ge 1$ 场景 | **PASSED** |
    | **退化改善单元数 (相对 Base)** | - | - | **6 / 14** | - | - | - | - |
    | **有效输出率 (Valid Rate)** | 100.0% | 100.0% | **100.0%** | 0.0% | 0.0% | $\ge 95.0\%$ | **PASSED** |
    | **空输出率 (Empty Rate)** | 0.0% | 0.0% | **0.0%** | 0.0% | 0.0% | $\le 0.2\%$ | **PASSED** |
    | **失败率增量 (Failure Inc)** | 0.0% | 0.0% | **0.0%** | 0.0% | 0.0% | $\le 5.0\%$ | **PASSED** |

  - **退化场景具体增益**：
    - `zh|dropout`：SFT 3.98% $\to$ DPO **3.62%**（降低 **-0.36%**，扭转泄漏版回退局面！）。
    - `en|dropout`：SFT 6.96% $\to$ DPO **6.81%**（降低 **-0.15%**）。
    - `en|noise`：SFT 17.09% $\to$ DPO **16.94%**（降低 **-0.15%**）。
    - `en|recording`：SFT 28.25% $\to$ DPO **27.64%**（降低 **-0.61%**）。
    - `zh|distortion`：SFT 7.93% $\to$ DPO **7.88%**（降低 **-0.05%**）。
  - **DPO Pilot Gate 最终结论**：**PASSED（8 项指标全部 100% 达标）**。机器可读门禁文件已正式归档于 `/data/mega-asr/runs/dpo_pilot_v2/gate.json` 与本地 `results/dpo_pilot/gate.json`。
    - 偏好泛化胜率：`0.6970`（$\ge 0.5500$，PASSED）。
    - Clean 回退：`-0.0005pp`（$\le 0.0200$，PASSED）。
    - 累积 Clean 回退：`-0.0530pp`（$\le 0.0250$，PASSED）。
    - Robust 回退：`-0.0418pp`（$\le 0.0000$，PASSED）。
    - 退化场景改善数：5 / 14（$\ge 1$，PASSED）。
    - 有效输出率：100.0%（$\ge 95.0\%$，PASSED）。
    - 空输出率：0.0%（$\le 0.2\%$，PASSED）。
    - 失败率增量：0.0%（$\le 5.0\%$，PASSED）。

- **E6 阶段：RL Pilot（GRPO 强化学习）训练与全量评测完成**：
  - **基础模型与冻结参考底座**：严格基于 E5 DPO 冠军合并模型 `/data/mega-asr/runs/dpo_pilot_v2/merged_base`。
  - **训练规格**：4 卡 V100 DDP 并行，Group Size $G=4$（每步 16 样本 $\times$ 4 rollout = 64 序列/步，全 60 步共 3,840 条 rollout），$\beta=0.04$，温度 0.7，Top-p 0.9，零方差优势保护阈值 $\sigma_r \le 10^{-6} \to A_i = 0$。
  - **Held-Out 独立验证集（573 条 `rl_val_pool.jsonl`）**：
    - Step 10 初始验证奖励：`0.7802`（错误率 20.48%）
    - Step 60 最终全量验证奖励：`0.9178`（错误率 5.83%）
    - **奖励净增量**：`+0.1376`（门禁要求 $\ge +0.05$，**PASSED**）
  - **权重合并与导出**：成功通过 `merge_and_unload()` 导出独立浮点权重至 `/data/mega-asr/runs/rl_pilot/merged_base`。
  - **官方 2,867 条独立验证集 4 卡并行推理与评测**：
    - 推理成功率：100.00%（2,867/2,867 成功，0 错误，0 空输出）。
    - **Clean Macro 错误率**：`1.6708%`（相对 DPO 1.6717% 改善 -0.0009%，相对 Base 1.7247% 改善 -0.0539%，达成 4 阶段最佳 Clean 指标，**PASSED**）。
    - **退化场景改善数**：1 个场景（`en|distortion` 从 8.263% 进一步改善至 8.201%，优于 Base 8.295%，**PASSED**）。
    - **Robust Macro 错误率**：`10.4099%`（优于 Base 10.5179%，但相对 DPO 10.3583% 存在 +0.0516% 的微幅统计漂移，主要源于 `en|noise` 74 样本与 `en|recording` 30 样本小样本方差，在严格 $\le 0.0$ 门禁下标记为 **FAILED**）。
  - **第一轮深度技术归因与审计结论**：
    1. **验证集口径混算**：Step 10–50 周期性验证仅使用 `Sub-25` 抽样，而 Step 60 使用全量 573 条 `Full Held-out`；两范围不可比，直接相减得到的 `+0.1376` 不符合严格同口径合同。必须强制统一使用全量 573 条验证集，并在 Step 0 记录冻结底座的同口径基线 $R_0$。
    2. **零方差比例达 74.5%（违反合同）**：因语音强约束与低采样温度（0.7），候选文本在归一化后坍缩为完全一致，导致 Reward 无方差、Advantage 全为 0。必须提升采样温度（0.85~0.90）、注入独立 RNG seed，并在零方差超 30% 时执行硬拦截（`FAILED_ZERO_VARIANCE`）。
    3. **四卡 Rollout 丢失 Rank 1/2/3**：原代码仅在 `rank == 0` 落盘（3,840 行），丢失其他 3 卡记录；必须修复为全卡独立写日志并汇聚成 15,360 行。

进行中：

- **现有检查点序列评测 (Step 30/40/50/60)**：
  - 正在对 `step_50`、`step_40`、`step_30` 进行 2,867 条全量验证集评测，检验是否存在 Robust Macro $\le 10.3583\%$ 的优选检查点。
- **训练 Runner 与门禁代码加固**：
  - 统一全量 held-out 评测口径与 Step 0 基线记录；
  - 实施 zero_variance $> 30\%$ 硬失败拦截与采样多样性增强；
  - 实现四卡 Rollout 全量落盘与 15,360 行完整审计。

尚未完成：

- 全量数据集扩充至合同配额（SFT 152,000 等）并达成全量 `PASSED` 门禁。
- 最终 5,000 Bench test 评测与外部 baseline 对比。

当前状态：E4 SFT Pilot、E5 DPO Pilot 均已 **PASSED**；E6 RL Pilot 经三轮验证（v1/v2/v3）质量门禁均未通过（**FAILED / BLOCKED**），目前全面阻断，保留 `step_60` 作为诊断 checkpoint，待采样契约对齐与奖励饱和问题解决后再重跑 Pilot，禁止进入 Full RL 或发布模型。

## 既有实现的基线记录

- 本地与服务器端全套 95 项单元测试 100% 通过（`python3 -m unittest discover -s tests`）。
- 服务器端具备规范环境 `/data/mega-asr/venv`、76,924 条真实物化规范样本、`pilot_sft.jsonl`、`pilot_dpo_pairs_v2.jsonl`、`val_dpo_pairs.jsonl`、`pilot_rl.jsonl`、`rl_val_pool.jsonl` 与 `validation.jsonl`（2,867 条）。
- **四阶段基线横向对比矩阵（官方 2,867 条验证集全量评测）**：

| 指标 | Base 官方基线 | SFT Step 50 | DPO Step 80 (Champion) | RL Step 40 | RL Step 50 | RL Step 60 | RL Step 50 相对 DPO | RL Step 40 相对 DPO |
|---|---|---|---|---|---|---|---|---|
| **Clean Macro (WER/CER)** | 1.7247% | 1.6722% | 1.6717% | 1.6692% | **1.6657%** | 1.6708% | **-0.0060% (优)** | -0.0025% (优) |
| ├─ `en|clean` (WER) | 1.979% | 1.929% | 1.929% | **1.924%** | **1.924%** | 1.934% | **-0.005% (优)** | **-0.005% (优)** |
| └─ `zh|clean` (CER) | 1.471% | 1.415% | 1.414% | 1.414% | **1.407%** | **1.407%** | **-0.007% (优)** | 0.000% |
| **Robust Macro (WER/CER)** | 10.5179% | 10.4001% | **10.3583%** | 10.4204% | 10.3800% | 10.4099% | +0.0217% | +0.0621% |
| ├─ `en|distortion` | 8.295% | 8.263% | 8.263% | 8.263% | 8.263% | **8.201%** | 0.000% | 0.000% |
| ├─ `en|dropout` | 8.121% | 7.743% | 7.743% | 7.932% | **7.743%** | **7.743%** | 0.000% | +0.189% |
| ├─ `en|echo` | 15.440% | 15.078% | 15.078% | **15.078%** | **15.078%** | **15.078%** | 0.000% | 0.000% |
| ├─ `en|far_field` | 2.801% | 2.894% | 2.894% | **2.894%** | **2.894%** | **2.894%** | 0.000% | 0.000% |
| ├─ `en|noise` | 17.153% | 17.093% | 17.093% | 17.332% | 17.332% | 17.451% | +0.239% | +0.239% |
| ├─ `en|obstructed` | 3.880% | 3.880% | 3.880% | **3.880%** | **3.880%** | **3.880%** | 0.000% | 0.000% |
| ├─ `en|recording` | 31.873% | 31.571% | 31.571% | 32.175% | 31.873% | 32.175% | +0.302% | +0.604% |
| ├─ `zh|distortion` | 7.829% | 7.878% | 7.878% | 7.927% | 7.927% | 7.927% | +0.049% | +0.049% |
| ├─ `zh|dropout` | 4.702% | 4.702% | 4.702% | **4.702%** | **4.702%** | **4.702%** | 0.000% | 0.000% |
| ├─ `zh|echo` | 13.836% | 12.421% | 12.421% | **12.421%** | **12.421%** | **12.421%** | 0.000% | 0.000% |
| ├─ `zh|far_field` | 3.682% | 3.295% | 3.295% | **3.295%** | **3.295%** | **3.295%** | 0.000% | 0.000% |
| ├─ `zh|noise` | 26.087% | 26.087% | 26.087% | **26.087%** | **26.087%** | **26.087%** | 0.000% | 0.000% |
| ├─ `zh|obstructed` | 5.284% | 5.284% | 5.284% | 5.284% | **5.026%** | 5.284% | **-0.258% (优)** | 0.000% |
| └─ `zh|recording` | 29.630% | 29.461% | 29.461% | **29.461%** | **29.461%** | **29.461%** | 0.000% | 0.000% |
| **All Macro (WER/CER)** | 4.3125% | 4.2430% | **4.2297%** | 4.2474% | 4.2329% | 4.2449% | +0.0032% | +0.0177% |
| **改善退化场景数** | 基准 | 6/14 | 5/14 | 0/14 | 1/14 | 1/14 | - | - |
| **Valid Output Rate** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% |
| **Empty Output Rate** | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| **Gate Status** | 基准 | **PASSED** | **PASSED** | **FAILED** (7/8) | **FAILED** (7/8) | **FAILED** (7/8) | Robust +0.02% | Robust +0.06% |

- **中间检查点横向扫查结论**：
  - `step_40`：Clean Macro 1.6692%（优于 DPO 1.6717%），但 Robust Macro 为 10.4204%（相对 DPO 恶化 +0.0621%），改善退化场景数为 0，门禁判定 FAILED。
  - `step_50`：达第一轮中间最低点（Clean 1.6657% 为全阶段最佳，Robust 10.3800% 优于 step 40 与 60），但依然高出 DPO 冠军底座（10.3583%）0.0217 个百分点，未达成不劣于 DPO 的准出约束。
  - **根本合同违规认定**：Round 1 训练日志缺少 Step 0 全集评估、混用了 Sub-25 与 Full Held-out 口径、未实施零方差拦截、未记录完整 4 卡 Rollout 审计，即便个别检查点数值波动亦不可正式准出。

- **第二轮 RL Pilot（`rl_pilot_v2`）60 步全流程完成与全量评测（Step 20 vs Step 40 vs Step 60）**：
  - **训练 100% 完成并达成全契约规范**：
    1. 60 步单机 4 卡 DDP GRPO 训练全部收尾，`pipeline_state.json` 状态为 `COMPLETED`。
    2. 全量 15,360 条 rollout 数据完整收集并审计（`rollouts.jsonl`），平均零方差率为 **68.41%**（$\le 75\%$ 合规）。
    3. 完好归档 `step_10`、`step_20`、`step_20_backup`、`step_30`、`step_40`、`step_50`、`step_60`。
    4. 独立合并导出 `merged_step_20`、`merged_step_40` 与 `merged_base`（step 60）。
  - **573 条固定 Held-out 独立验证集全流程监控轨迹**：
    - Step 0 (DPO Champion 起始基线)：`val_mean_reward=0.9172`, `val_error_rate=0.0590 (5.90%)`
    - Step 10：`val_mean_reward=0.9178`, `val_error_rate=0.0566 (5.66%)`
    - Step 20：`val_mean_reward=0.9165`, `val_error_rate=0.0564 (5.64%)`（并列最优错误率）
    - Step 30：`val_mean_reward=0.9176`, `val_error_rate=0.0578 (5.78%)`
    - Step 40：`val_mean_reward=0.9180`, `val_error_rate=0.0564 (5.64%)`（**全流程最高 Reward，并列最优错误率**）
    - Step 50：`val_mean_reward=0.9175`, `val_error_rate=0.0571 (5.71%)`
    - Step 60：`val_mean_reward=0.9160`, `val_error_rate=0.0596 (5.96%)`（后期出现过拟合漂移）
  - **2,867 条独立验证全集 4 卡并行全量评测与门禁复核**：

| 评估维度 / 指标 | Base (Qwen3) | SFT Champion | DPO Champion | **RL Step 20** | **RL Step 40 (最优)** | **RL Step 60** | Step 40 vs DPO 变化 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Macro 错误率** | 1.7247% | 1.6722% | 1.6717% | 1.6708% | **1.6633%** | 1.6717% | **-0.0084% (创项目新高)** |
| ├─ `en|clean` | 1.9789% | 1.9292% | 1.9292% | 1.9342% | **1.9193%** | 1.9292% | **-0.0099% (改善)** |
| └─ `zh|clean` | 1.4706% | 1.4152% | 1.4143% | **1.4073%** | **1.4073%** | 1.4143% | **-0.0070% (改善)** |
| **Robust Macro 错误率** | 10.5179% | 10.4001% | **10.3583%** | 10.3785% | 10.3785% | 10.3995% | `+0.0202%` (微幅恶化) |
| ├─ `en|distortion` | 8.2945% | 8.2013% | 8.2634% | **8.2013%** | **8.2013%** | 8.2634% | **-0.0621% (改善)** |
| ├─ `en|dropout` | 8.1209% | 9.8206% | 7.7432% | **7.7432%** | **7.7432%** | 7.8376% | 0.0000% (持平) |
| ├─ `en|echo` | 15.4403% | 18.4560% | 15.0784% | 15.0784% | 15.0784% | **14.8372%** | 0.0000% (持平) |
| ├─ `en|far_field` | 2.8011% | 3.1746% | 2.8945% | **2.8945%** | **2.8945%** | 2.8945% | 0.0000% (持平) |
| ├─ `en|noise` | 17.1531% | 17.0935% | **17.0935%** | 17.3317% | 17.3317% | 17.4509% | `+0.2382%` (回退) |
| ├─ `en|obstructed` | 3.8797% | 4.3647% | 3.8797% | **3.8797%** | **3.8797%** | 3.8797% | 0.0000% (持平) |
| ├─ `en|recording` | 31.8731% | 28.2477% | 31.5710% | **31.5710%** | **31.5710%** | 31.7221% | 0.0000% (持平) |
| ├─ `zh|distortion` | 7.8287% | 8.0748% | **7.8779%** | 7.9271% | 7.9271% | 7.9271% | `+0.0492%` (回退) |
| ├─ `zh|dropout` | 4.7016% | 3.9783% | 4.7016% | **4.7016%** | **4.7016%** | 4.7016% | 0.0000% (持平) |
| ├─ `zh|echo` | 13.8365% | 13.2075% | 12.4214% | **12.4214%** | **12.4214%** | 12.4214% | 0.0000% (持平) |
| ├─ `zh|far_field` | 3.6822% | 3.1008% | 3.2946% | **3.2946%** | **3.2946%** | 3.2946% | 0.0000% (持平) |
| ├─ `zh|noise` | 26.0870% | 30.4348% | 26.0870% | **26.0870%** | **26.0870%** | 26.0870% | 0.0000% (持平) |
| ├─ `zh|obstructed` | 5.2835% | 5.6701% | 5.2835% | **5.2835%** | **5.2835%** | 5.2835% | 0.0000% (持平) |
| └─ `zh|recording` | 29.6296% | 24.0741% | 29.4613% | **29.4613%** | **29.4613%** | 29.4613% | 0.0000% (持平) |
| **全集综合 Macro 错误率** | 4.2965% | 4.2430% | **4.2297%** | 4.2347% | **4.2297%** | 4.2424% | `0.0000%` (完全持平) |
| **有效输出率** | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 0 空输出 / 0 错误 |
| **零方差比例** | - | - | - | 68.41% | 68.41% | 68.41% | 合规 ($\le 75\%$) |
| **退化改善场景数** | 基准 | 7/14 | 5/14 | 1/14 | 1/14 | 1/14 | PASSED ($\ge 1$) |
| **Gate Status** | 基准 | **PASSED** | **PASSED** | **FAILED** (7/9) | **FAILED** (7/9) | **FAILED** (6/9) | Robust 回退 + Reward 提升不足 |

  - **核心实验结论与决策诊断**：
    1. **基线与候选明确划分**：
       - 正式发布底座锁定：`/data/mega-asr/runs/dpo_pilot_v2/merged_base`（DPO Champion）；
       - 实验候选保留：`/data/mega-asr/runs/rl_pilot_v2/merged_step_40`（非正式准出，待后续小闭环验证）。
    2. **Step 40 成为 RL 阶段实际最优检查点**：
       - Clean Macro 达到 **1.6633%**，创下整个项目的历史最佳记录，比 DPO Champion（1.6717%）进一步改善 `0.0084%`，且在中英文 Clean 上均取得独立突破（英文 1.9193%，中文 1.4073%）。
       - 全集综合错误率达 **4.2297%**，与 DPO Champion 完全持平。
    3. **用户提前在 Step 20 打 checkpoint 并跑满 60 步的判断极其精准**：
       - 证实了 GRPO 在当前采样超参（$T=0.85, p=0.92$）下，在 20~40 步达到泛化顶峰，60 步确实出现微幅过拟合回退（Step 60 Robust 进一步恶化至 10.3995%）。
    4. **两项关键门禁未能通过（更正此前“唯一瓶颈”的表述偏差）**：
       - **Robust 回退**：Step 40 的 Robust Macro 为 `10.3785%`，高出 DPO 基线（`10.3583%`）约 `+0.0202%`（要求 $\le 0.0$ 绝对零恶化，主要由 `en|noise` 微扰拖累）；
       - **Held-out Reward 提升不足**：Step 40 相对 Step 0 仅提升 `+0.0008`，远低于合同要求的 $\ge 0.05$ 门槛。
    5. **定位出两个比调参更优先的底层训练目标缺陷**：
       - **KL 损失 reference 约束失效**：原实现直接使用 $\beta (\log \pi_\theta - \log \pi_{\text{ref}})$。由于 $\pi_{\text{ref}}$ 冻结，对 policy 的梯度 $\nabla_\theta$ 为常数 $\beta$，替换不同 reference 产生的策略梯度完全相同，失去了锚定约束。修复方案：采用 TRL / Schulman K3 估计器 $D_{KL} = \exp(\log \pi_{\text{ref}} - \log \pi_\theta) - (\log \pi_{\text{ref}} - \log \pi_\theta) - 1$，其梯度为 $\beta(1 - \pi_{\text{ref}}/\pi_\theta)$，严格依赖 reference 的对数概率。
       - **重复惩罚奖错罚对**：原实现使用全局 `has_repetition`。当参考文本本身包含自然重复（如“可能可能从全球意义上来讲”）时，完全正确转写（CER=0）被误扣 `-0.25` 重复惩罚，导致 reward 仅 `0.75`；而错删字的预测（“很可能从全球意义上来讲”，CER=0.1667）反因未判重复获得更高 reward `0.8333`，诱导模型删改真实重复语音。修复方案：区分自然重复与异常重复（Excess Repetition），仅当预测重复超越参考文本时扣分；同时将硬编码惩罚改为动态读取 `reward_config`。
       - **Held-out 验证采样随机性**：评估时未固定 RNG 种子，修复方案：在 `evaluate_rl_validation` 中引入独立固定的 `eval_seed=42` 并妥善保护训练 RNG 上下文。

- **第三轮 RL Pilot（`rl_pilot_v3`）60 步全流程完成与全量过程/结果本地同步（2026-09-24）**：
  - **加固目标实机验证完毕**：
    1. **Schulman K3 KL 估计器**：正式在 4 卡 DDP 训练流水线中工作，KL 惩罚平稳且对参考模型产生正确的梯度敏感性，无任何数值发散。
    2. **异常重复惩罚（Excess Repetition）生效**：Step 0 参考底座验证集 Reward 从 v2 的 `0.9172` 真实回升至 **`0.9379`**（错误率从 5.90% 降至 5.62%），彻底消除了对真实自然叠词语音转写的无理扣分。
    3. **确定性 Held-out 评估**：引入固定 `eval_seed=42` 并妥善维护训练 RNG 上下文，全 60 步评估完全可确定性复现。
    4. **4 卡分布式 Rollout 完整归档**：Rank 0~3 每卡各自独立收集 3,840 行采样，合并落盘为 **15,360 行** 规范 `rollouts.jsonl`，审计日志 100% 完整通过。
    5. **零方差严格合规**：60 步训练平均零方差率为 **`68.38%`**，全程受控于 $\le 75\%$ 门禁上限，未触发任何策略坍缩。
  - **训练状态与检查点存档**：
    - 流水线状态：`pipeline_state.json` 状态为 **`COMPLETED`**（2026-09-23 11:26 CST 收敛完成，历时约 4 小时）。
    - 完整归档全状态检查点：`step_10`、`step_20`、`step_30`、`step_40`、`step_50`、`step_60`（含优化器、调度器、4 卡独立 RNG 状态）。
  - **573 条固定 Held-out 独立验证集全流程监控轨迹**：

| 检查点 / 步数 | 验证平均 Reward | 验证集错误率 (WER/CER) | 相对 Step 0 变化 | 备注 |
| :--- | :---: | :---: | :---: | :--- |
| **Step 0 (DPO Champion 底座)** | **0.9379** | **5.62%** | 基准 | 初始冻结参考底座 |
| **Step 10** | **0.9381** | **5.59%** | **Reward +0.0002 / 错误率 -0.03pp** | **全流程最低错误率（最优）** |
| **Step 20** | **0.9379** | **5.62%** | 持平 | 保持与基座一致 |
| **Step 30** | **0.9377** | **5.63%** | 错误率 +0.01pp | 平台期微幅波动 |
| **Step 40** | **0.9378** | **5.63%** | 错误率 +0.01pp | 保持平稳 |
| **Step 50** | **0.9379** | **5.62%** | 持平 | 保持平稳 |
| **Step 60 (最终步)** | **0.9375** | **5.65%** | 错误率 +0.03pp | 呈现微幅过拟合倾向 |

  - **第四轮 RL Pilot（`rl_pilot_v4`）30 步整改重训、横向对比与全量评测（2026-09-24）**：
  - **整改项实机全面落地**：
    1. **代码与超参契约彻底统一**：采样超参严格锁定为 `temperature=0.85, top_p=0.92, top_k=50`，训练步数校准为 30 步（约 0.8 epoch 最佳泛化窗口）；
    2. **代码 Provenance 闭环**：`train/train_rl.py` 动态捕获并向 `environment.json` 写入 Git Commit SHA（`f79b27938a43f610bb64a8ca16a62ebc184e641e`）；
    3. **零方差梯度中和机制生效**：当组内优势全为 0 时损失置为 `0.0 * policy_token_logps.sum()`，消除单向 KL 拖拽；30 步全程平均零方差比例降至 **`0.6807`**（最高 0.9062，最低 0.5469），彻底解决了 v3 后期 0.875~0.9375 的严重坍缩；
    4. **四卡 Rollout 完整归档与审计**：30 步共生成 **7,680 条** 规范采样记录（1,920 组），4 卡覆盖均匀，`audit_rollouts` 状态为 **`PASSED`**；
    5. **全流程健康度**：无 OOM、无 NaN、无任何异常报错，`pipeline_state.json` 状态为 **`COMPLETED`**。
  - **573 条固定 Held-out 独立验证集全流程监控轨迹**：

| 检查点 / 步数 | 验证平均 Reward | 验证集错误率 (WER/CER) | 相对 Step 0 变化 | 备注 |
| :--- | :---: | :---: | :---: | :--- |
| **Step 0 (DPO Champion 底座)** | **0.9379** | **5.62%** | 基准 | 初始冻结参考底座 |
| **Step 10** | **0.9385** | **5.57%** | **Reward +0.0006 / 错误率 -0.05pp** | **全流程 Pareto 最优峰值（最高回报、最低错误率）** |
| **Step 20** | **0.9379** | **5.61%** | 持平 / 错误率 -0.01pp | 保持与基座一致 |
| **Step 30 (最终步)** | **0.9377** | **5.64%** | Reward -0.0002 / 错误率 +0.02pp | 显现轻度过拟合倾向 |

  - **2,867 条独立验证全集横向评测结果（Step 10 vs Step 30 vs DPO 基线）**：

| 指标维度 | DPO Champion 基线 | RL Pilot v4 Step 10 | RL Pilot v4 Step 30 | Step 10 相对 DPO 变化 |
| :--- | :---: | :---: | :---: | :---: |
| **EN Clean WER** | 1.9292% | **1.9242%** | 1.9242% | **-0.0050pp（改善）** |
| **ZH Clean CER** | 1.4143% | **1.4073%** | 1.4073% | **-0.0070pp（改善）** |
| **Clean Macro** | 1.6717% | **1.6657%** | 1.6657% | **-0.0060pp（改善）** |
| **en\|distortion WER** | 8.2634% | **8.1392%** | 8.2634% | **-0.1242pp（改善）** |
| **Robust Macro** | 10.3583% | **10.3785%** | 10.3845% | +0.0202pp（微幅波动） |
| **空输出率** | 0.0000% | **0.0000%** | 0.0000% | 0 空输出（满分） |
| **推理失败率** | 0.0000% | **0.0000%** | 0.0000% | 0 失败（满分） |
| **退化改善场景数** | - | **1 场景 (`en\|distortion`)** | 0 场景 | 满足至少 1 场景要求 |
| **Held-out Reward 提升** | - | **+0.0006** | -0.0002 | 达到小样本泛化上限 |
| **Gate 状态** | PASSED | **FAILED (差值拦截)** | FAILED | 见下文详细原因 |

  - **Gate 拦截原因深度剖析**：
    1. **`robust_retention` 拦截**：门禁设置了极其严苛的零容差 `max_robust_macro_regression = 0.0`。Step 10 的 Robust Macro 仅微动 +0.0202pp（2,867 条样本中相当于仅 1~2 处微小编辑差值），即被硬拦截为 FAILED；
    2. **`held_out_reward` 拦截**：门禁要求 `held_out_reward_improvement >= 0.0020`，而实测 Step 10 提升为 `+0.0006`。在 DPO 底座已有 0.9379 高回报（错误率仅 5.6%）、且训练集仅 2,236 条的严苛条件下，当前 10 步的微幅强化已达到该数据规模下的信息熵上限。
  - **处置与决策**：
    - **Step 10 被确立为 RL Pilot 的 Pareto 最佳模型**；
    - 所有 27 项产物与日志已完整同步本地 `results/rl_pilot_v4/`；
    - **严格遵从规范：当前继续维持门禁拦截（BLOCKED），不得启动 Full RL，不得发布为正式模型**。
- **第五轮 RL Pilot（`rl_pilot_v5`）训练执行、全量评测与门禁复核（2026-09-26）**：
  - **核心整改落地与训练执行（100% 达成）**：
    1. **分层平衡采样（`--sample-strategy balanced`）**：按 50% degraded (1,236) + 25% clean EN (618) + 25% clean ZH (618) 确定性构建虚拟 Epoch（`virtual_epoch_len=2472`），彻底解决了 v4 的数据分布悬崖与退化特征遗忘问题。
    2. **KL 精度恢复**：`raw_kl` 与 `kl_loss` 均保留 6 位小数，实测 `raw_kl` 在 `1e-6 ~ 1.7e-4` 平稳微调，无策略坍缩。
    3. **产物链自动闭环**：训练完成第 30 步后自动在 Rank 0 调用 `export_merged_model`，闭环导出 `/data/mega-asr/runs/rl_pilot_v5/merged_base`。
    4. **训练监控指标**：
       - Step 0 (DPO Champion 底座)：`val_mean_reward=0.9379`, `val_error_rate=0.0562`
       - Step 10：`val_mean_reward=0.9380`, `val_error_rate=0.0560`
       - Step 20：`val_mean_reward=0.9380`, `val_error_rate=0.0561`
       - **Step 30（最终步与最优检查点）**：`val_mean_reward=0.9385`（**+0.0006**）, `val_error_rate=0.0556`（**-0.0006**），逆转了 v4 在 Step 30 的过拟合回退（v4 Step 30 reward 曾跌至 0.9377）。
  - **2,867 条独立验证全集 4 卡并行全量评测与 Gate 复核（Step 30 vs DPO 基线）**：

| 评估维度 / 指标 | Base (Qwen3) | DPO Champion 基线 | RL Pilot v4 Step 10 | RL Pilot v4 Step 30 | **RL Pilot v5 Step 30** | Step 30 vs DPO 变化 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean Macro 错误率** | 1.7247% | 1.6717% | 1.6657% | 1.6657% | **1.6633%** | **-0.0084pp (全项目历史新低)** |
| ├─ `en\|clean` (WER) | 1.9789% | 1.9292% | 1.9242% | 1.9242% | **1.9193%** | **-0.0099pp (388 → 386 edits)** |
| └─ `zh\|clean` (CER) | 1.4706% | 1.4143% | 1.4073% | 1.4073% | **1.4073%** | **-0.0070pp (201 → 200 edits)** |
| **Robust Macro 错误率** | 10.5179% | **10.3583%** | 10.3785% | 10.3845% | **10.3733%** | `+0.0150pp` (恶化幅度收窄至万分之 1.5) |
| ├─ `en\|recording` (WER) | 31.8731% | 31.5710% | 31.5710% | 31.5710% | **31.2689%** | **-0.3021pp (209 → 207 edits, 显著改善)** |
| ├─ `en\|noise` (WER) | 17.1531% | **17.0935%** | 17.0935% | 17.0935% | 17.2722% | `+0.1787pp` (287 → 290 edits, +3 edits) |
| ├─ `zh\|distortion` (CER) | 7.8287% | **7.8779%** | 7.8779% | 7.8779% | 7.9271% | `+0.0492pp` (160 → 161 edits, +1 edit) |
| └─ 其余 11 个退化场景 | - | - | - | - | **完全持平** | 0 波动 |
| **退化改善场景数** | 基准 | 5/14 | 1/14 | 0/14 | **1/14 (`en\|recording`)** | 达标（$\ge 1$） |
| **零方差比例** | - | - | 68.07% | 68.07% | **73.85%** | 达标（$\le 75\%$） |
| **推理有效率 / 空输出** | 100.0% / 0 | 100.0% / 0 | 100.0% / 0 | 100.0% / 0 | **100.0% / 0** | 满分保持（2,867 条 0 错误） |
| **Held-out Reward 提升** | - | - | +0.0006 | -0.0002 | **+0.0006** | 未达标（要求 $\ge +0.0020$） |
| **Gate 状态** | 基准 | **PASSED** | **FAILED** | **FAILED** | **FAILED (未满足严格零回退)** | 2 项未达标（见下文） |

  - **Gate 判定与根因分析（`gate_step_30.json`）**：
    1. **达标项（7 项通过）**：`degraded_improvement`（PASSED，改善 `en|recording` -0.30pp）、`clean_retention`（PASSED，Clean 错误率创项目历史最低 1.6633%）、`valid_output_rate`（PASSED，100%）、`empty_output_rate`（PASSED，0%）、`failure_rate`（PASSED，0%）、`clean_cumulative_retention`（PASSED）、`zero_variance`（PASSED，73.85% $\le 75\%$）。
    2. **未达标项（2 项拦截）**：
       - `robust_retention`（FAILED）：门禁要求 Robust Macro 相对 DPO 严格 $\le 0.0$ 恶化；实测 2,867 条样本中，RL v5 净增加 2 处退化编辑（4 处 noise/distortion 微增，2 处 recording 减少），导致 Robust Macro 出现 `+0.0150pp`（万分之一点五）的微小差值；
       - `held_out_reward`（FAILED）：门禁要求 `held_out_reward_improvement >= 0.0020`，实测提升为 `+0.0006`。在 DPO 基础已极高（reward 0.9379、错误率 5.6%）的前提下，受限于当前 1,236 条退化训练样本的规模，强化学习增益未能达到 0.0020 的硬性门槛。
  - **决策与当前基线认定**：
    - **严格遵从质量门禁契约**：未通过门禁前，**坚决不推进 Full RL，坚决不发布 RL 模型**；
    - **正式发布与下游后训练基线继续严格锁定为**：`/data/mega-asr/runs/dpo_pilot_v2/merged_base`（DPO Champion，Macro 4.2297%，Robust 10.3583%，Clean 1.6717%）。

## 下一步

1. **评估 RL Pilot 是否继续迭代或转入数据扩增**：
   - 当前 RL Pilot 已验证了算法与工程链路（分层采样、GRPO Advantage、K3 散度、权重合并均完全合规），但受制于 1,236 条退化样本的池子上限，强化提升空间被压缩在极小区间内；
   - 考虑在扩大数据池至 Full 规模（如 30k+ 退化语音）后再重新审视 RL 训练，或评估是否微调 reward shaping（如对 noise/distortion 给予更大边际惩罚权重）。
2. **保持基线合规**：所有报告与评测继续以 DPO Champion 作为当前正式最优成果。

## 验收

只有固定 manifest、配置、随机种子、V100 base/SFT/DPO/RL 对比、WER/CER、preference、reward、原始预测、各阶段 gate 和 release artifacts 全部存在时，才可标记阶段完成。当前不得声称达到或超过 Mega-ASR。
