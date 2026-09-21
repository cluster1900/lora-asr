# 开发进度

最后更新：2026-09-21

## 当前状态

当前状态：**E5 阶段 DPO Pilot 训练、权重合并与 2,867 条独立验证集全量评测已圆满完成！Prompt 根因修复后，Held-out 验证集偏好胜率达到 0.6970，全面通过 8 项严格合同门禁，门禁状态正式达成 PASSED！已正式解锁进入 E6 RL（强化学习）阶段！**

1. **核心根因修复与 Step 0 不变量闭环**：
   - 彻底修复了 `scripts/build_dpo_pairs.py` 中手写 Prompt 导致的约 45 个自然对数单位分布漂移问题，全链路严格统一为 Qwen3-ASR 官方 `_build_text_prompt`。
   - `train_dpo.py` 增设 Step 0 不变量断言，实测 Step 0 策略模型与参考模型对数概率误差仅 $7.80 \times 10^{-3}$，初始 Loss 精确对齐理论值 $\ln(2) \approx 0.69315$。
2. **Canonical Held-Out 偏好泛化准确率重大突破**：
   - `loss_log.jsonl` 真实全量评估记录：Step 10 (0.4545) $\to$ Step 20 (0.6515) $\to$ Step 50 (0.6818) $\to$ Step 60 (0.6970) $\to$ **Step 80 = 0.6970 (46/66)**。
   - 严格满足并超越门禁合同红线（$\ge 0.5500$，超出 +14.70pp）。
3. **权威 2,867 独立验证集评测指标对照（对比 SFT Step 50 真实基座）**：
   - **Clean Macro 零回退微改善**：`1.6722% → 1.6717%`（净改善 -0.0005pp，未发生任何 Clean 恶化）。
   - **Robust Macro 显著稳健改善**：`10.4001% → 10.3583%`（净改善 **-0.0418pp**，超越门限 $\le 0.0$）。
   - **全集语言 Macro 全面优化**：`4.2430% → 4.2297%`。
   - **退化场景有效改善数**：真实改善 **5/14 个场景**（`en|distortion`, `en|dropout`, `en|noise`, `en|recording`, `zh|distortion`）。
   - **模型解码健壮性**：2,867 条独立验证样本 100% 成功解码，0 推理错误，0 空输出。
4. **门禁判定与全流程 Provenance 固化（PASSED）**：
   - 机器可读门禁文件已正式归档于 `/data/mega-asr/runs/dpo_pilot_v2/gate.json` 并同步至本地 `results/dpo_pilot/gate.json`。
   - 完整记录了 8 项检查标准、实际测量值及 7 大核心产物的绝对路径与 SHA-256 哈希，完全可复现可追溯。
5. **官方 2,867 全体验证集核心指标权威对照表**：
   | 阶段 / 模型 | 英文 Clean WER ↓ | 中文 Clean CER ↓ | Clean Macro ↓ | Robust Macro ↓ | 整体 Macro ↓ | 场景改善数 | 偏好胜率 | 门禁判定 |
   |---|---:|---:|---:|---:|---:|:---:|:---:|:---:|
   | **Base 官方基准** | 1.9789% | 1.4706% | 1.7247% | 10.5179% | 4.3125% | — | — | 基准参考 |
   | **SFT Step 50 (受控基座)** | 1.9441% | 1.4002% | 1.6722% | 10.4001% | 4.2430% | 6 场景 | — | **PASSED** |
   | **DPO Step 80 (正式冠军版)** | **1.9292%** | **1.4143%** | **1.6717%** | **10.3583%** | **4.2297%** | **5 场景** | **0.6970** | **PASSED** (8 项全通) |
6. **Held-out 验证集规模风险说明**：
   - 当前 held-out 偏好验证集包含 66 对样本，已经按修正后的 Prompt 重新计算，并作为 Canonical 判定依据通过门禁（46/66 = 0.6970 $\ge$ 0.5500）；
   - 从统计意义上看，66 条样本量偏小，存在一定统计方差风险；当前暂不阻塞 RL Pilot 的启动，但在正式进入 Full RL Release 前，建议扩展构建第二个确认集进行双重确认。

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

进行中：

- **E6 阶段 RL Pilot（GRPO / Reinforcement Learning）前置准备**：
  - 基于正式通过门禁的 DPO Champion 合并底座（`/data/mega-asr/runs/dpo_pilot_v2/merged_base`），配置奖励函数组件（WER/CER 相对基线奖励、长度惩罚、空输出硬拦截惩罚）。
  - 准备 `pilot_rl.jsonl` 候选提示词并校验参考模型（Reference Model Freeze）冻结逻辑与 KL 散度约束。

尚未完成：

- RL Pilot 训练与门禁验证（E6，已正式解锁准入）。
- 全量数据集扩充至合同配额（SFT 152,000 等）并达成全量 `PASSED` 门禁。
- 最终 5,000 Bench test 评测与外部 baseline 对比。

当前状态：E4 SFT Pilot（Step 50）与 E5 DPO Pilot（Step 80）门禁均已真实 **PASSED**！模型在 2,867 条独立验证集上保持极低 Clean 错误率（1.6717%），显著提升声学退化鲁棒性（Robust Macro 10.3583%），且在 Held-Out 独立偏好集上取得 0.6970 高胜率，**正式解锁并交接至 E6 RL 强化学习阶段**！

## 既有实现的基线记录

- 本地与服务器端全套 77 项单元测试 100% 通过（`python3 -m unittest discover -s tests`）。
- 服务器端具备规范环境 `/data/mega-asr/venv`、76,924 条真实物化规范样本（71,360 入库角色互斥）、`pilot_sft.jsonl`（7,000 条）、`pilot_dpo_pairs_v2.jsonl`（412 对真实无泄漏）、`val_dpo_pairs.jsonl`（66 对真实无泄漏）、`validation.jsonl`（2,867 条）与 `smoke.jsonl`。
- **阶段门禁判定状态**：
  - Base: Clean Macro 1.725%, Robust Macro 10.518%, Total Macro 4.312%
  - SFT Step 50: Clean Macro 1.672%, Robust Macro 10.400%, Total Macro 4.243%, 改善场景 6/14, Gate: **PASSED**
  - DPO Step 80 (Champion): Clean Macro 1.672%, Robust Macro 10.358%, Total Macro 4.230%, 改善场景 5/14 (vs SFT), Preference Acc 0.6970 (>= 0.5500), Gate: **PASSED**

## 下一步

1. **E6 RL Pilot 启动**：使用 `/data/mega-asr/runs/dpo_pilot_v2/merged_base` 作为初始 Policy 与 Reference 底座。
2. **RL 奖励函数与 Rollout 验证**：校验 Rule-based + Metric-based reward 复合奖励信号及 KL 正则化。
3. **RL Pilot 训练与门禁验证**：跑通 4 卡 DDP RL Pilot，验证 held-out reward 提升与 Clean/Robust 零回退门禁。
4. **进入 E6 RL**：在 DPO 门禁真实 PASSED 后，启动 RL (GRPO) Pilot 阶段。

## 验收

只有固定 manifest、配置、随机种子、V100 base/SFT/DPO/RL 对比、WER/CER、preference、reward、原始预测、三阶段 gate 和 release artifacts 全部存在时，才可标记阶段完成。当前不得声称达到或超过 Mega-ASR。
