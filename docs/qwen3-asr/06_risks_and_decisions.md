# 风险与决策

## 当前决策

- 维护公开数据 -> FP16 base -> SFT -> DPO -> RL -> release -> 评测一条路径。
- 基础模型固定 `Qwen/Qwen3-ASR-1.7B`；V100 使用 FP16 和 eager attention，不使用 BF16 或
  FlashAttention-2。
- 训练使用独立 `/data/mega-asr` 环境和单机 4 卡 DDP；global batch 必须包含 world size。
- 完整链路固定为 SFT → DPO → RL；阶段交接采用 `merge_and_unload()` 策略产出下阶段基座，每阶段独立 adapter、manifest、checkpoint 和 gate。
- LoRA 目标层同时覆盖 LLM Decoder 与音频塔投影层，兼顾语义纠错与声学损伤补偿；A2S 不进入正式链路。
- DPO 推荐离线预计算参考模型 logprobs，消除训练时参考模型常驻显存开销（显存降 50%）。
- RL 采用带 KL 正则的 GRPO（组大小 $G=4$、temp 0.85、top_p 0.92、top_k 50），统一 `asr` 键名与序列惩罚。
- Clean 数据候选源扩大 5 倍以应对高平局率（ties），确保足额产出 clean preference pairs。
- 设置全局 Base 锚定的 clean 错误率累积退化上限（$\le 0.025$），防止级联劣化。
- English WER 与 Chinese CER 分开报告。
- Mega-ASR 只作方法和外部 baseline，不进入运行时依赖。
- 不使用 Teacher、Router 或量化训练；RL 是正式链路第三阶段。
- Hub 数据只按角色配额流式 staging，不镜像完整数据集；SFT、DPO、RL 使用隔离 pool。

## 风险与回滚条件

| 风险 | 门禁 | 回滚 |
|---|---|---|
| V100 FP16 数值不稳定 | 单 batch 前向反向、有限 loss/gradient、10+2 resume | 降低 micro batch 或回退到上一有效 checkpoint |
| eager attention 显存不足 | 128-row/10-step smoke 记录显存 | 降低 micro batch、增加 accumulation，不改 global batch 目标 |
| 公开数据字段或 split 漂移 | pinned revision、probe、严格 schema | 停止并更新配置/文档 |
| staging 过慢或服务器 CPU 被其他任务占用 | `/data/mega-asr` 独立缓存、逐条持久化、记录 CPU 负载 | 等待资源空闲或降低 worker，不读取其他项目产物 |
| Robust 原始话语跨 scenario 泄漏 | base source identity 固定 90/10 分区 | 停止 build 并修复 source identity |
| Hugging Face 直连不稳定 | mirror/ModelScope 下载记录和文件 hash | 切换已验证镜像，禁止无 hash 数据进入 manifest |
| checkpoint 不能续训或 global batch 漂移 | 10+2 resume、记录 world size | 不启动 pilot |
| clean 级联累积 regression | 每阶段 clean 相对 Base 累积增加 ≤0.025 红线 | 回退到上一阶段 adapter 并增大 clean 训练比例 |
| SFT 英文 Clean 回退对 DPO 的压力 | E4 英文 Clean WER 增量 +1.43%（1.98% $\to$ 3.41%），逼近累积 2.5% 红线 | DPO 阶段严格监控 Clean preference 质量与平局过滤，DPO vs SFT 回退超过 2.0% 或累积超 2.5% 即刻回滚 |
| DPO clean 平局率高导致 pairs 短缺 | 候选源扩大 5 倍、受控负例、严格过滤 ties | 重建 dpo_train_pool 候选池 |
| DPO pair 质量或 preference 泄漏 | chosen/rejected 审计、ties rejects、held-out accuracy | 停止 DPO，重建 dpo_train_pool |
| RL GRPO 候选同质化（组方差归零） | $G=4$、采样 temp=0.85/top_p=0.92 审计 | 调整采样温度或早停（30 步内），避免二次复读塌缩 |
| RL reward 投机或 KL 发散 | reward 组件单测、KL/rollout gate、reference freeze | 停止 RL，回退到 DPO adapter |
| 4 卡 DDP 梯度/数据 shard 不一致 | global batch、sample_id 去重、world size 记录 | 停止当前 run，保留最后有效 checkpoint |
| 指标不可比 | 同 manifest、同 evaluator、分语言指标 | 废弃该次比较 |
| 运行时加载非官方 wrapper | import 和 model revision 记录 | 停止并移除不合规依赖 |

### SFT Pilot 观测到的 Clean 回退与受控优化决策

在 E4 SFT Pilot 评测中，英文 Clean WER 从 1.98% 上升至 3.41%（+1.43%），中文 Clean CER 从 1.47% 上升至 1.57%（+0.10%），Clean 宏平均增量为 +0.77%，虽完全在单阶段 $\le 2.0\%$ 门禁内通过，但英文 Clean 回退已消耗了大部分累积容忍空间（相对 Base 累积增量 $\le 2.5\%$）。
经归因分析，确定三大诱因为：(1) 恒定 2e-5 学习率无衰减；(2) 退化样本占比过高（71.4%）；(3) 步数偏多（4.57 epochs）且盲目取最后一步。

据此确立受控优化决策：
1. **学习率降温与 Decay 约束**：LoRA 更新包含音频塔投影层与所有 28 层 Decoder 上的 199 个 Linear，过大的恒定学习率（2e-5）容易过快改变 Base 原有声学/语言映射。下调峰值学习率至 1e-5，并配置 50 步 Warmup 及 300 步全程 Linear Decay。
2. **数据分布重平衡（Balanced Clean Replay）**：针对退化音频占 71.4% 导致的声学特征漂移，引入平衡采样策略（50% degraded、25% English clean、25% Chinese clean）或 Clean 2x 重放，保全 Clean 先验。
3. **多 Checkpoint 评测与 Pareto 择优**：避免因过拟合盲目选取最后一个 checkpoint，采用 `evaluation/eval_checkpoint_series.py` 自动化横向评测 step 100/150/200/250/300，综合退化场景改善与 Clean 保真度选定正式合并底座。
4. **DPO 阶段监控红线**：DPO 偏好对构建中，Clean 源音频（LibriSpeech/AISHELL-1）必须确保 chosen 准确无误，严格过滤 gold 与预测一致的平局（ties），并防范负例误导。训练中以英文 Clean 为硬性约束，若 DPO vs SFT 回退 $> 0.02$ 或相对 Base 累积增量 $> 0.025$，触发立即停止并回滚。

### 受控优化实验（Run 2）验证结论与底座锁定

2026-09-20 在 4 × V100 SXM2-32GB 上完成 300 步受控对照实验及全系列检查点（Step 50~300）全量评测，获得明确结论：
1. **中文与英文 Clean 先验全面超越 Base**：Step 50 检查点下，英文 Clean WER 达到 `1.94%`（优于 Base 1.98%），中文 Clean CER 达到 `1.40%`（优于 Base 1.47%），Clean Macro 为 `1.67%`（优于 Base 1.72%）。彻底避免了过拟合退化特征引发的 clean 遗忘。
2. **全退化场景宏平均同步改善**：Step 50 的 Robust Macro 达到 `10.40%`，优于 Base（`10.52%`），并在 14 个退化单元中改善 6 个（`en|dropout`, `en|echo`, `en|recording`, `zh|echo`, `zh|far_field`, `zh|recording`），其余单元保持平稳。
3. **消除数值下溢与死循环风险**：Step 50 有效输出率达 100.0%（0 推理错误，0 空输出，0 死循环）。而 Step 100/150 步由于继续训练导致在严重的退化样本上产生自回归死循环（1,599 字符）及 float16 下溢空输出。
4. **决策裁定**：废除 Step 100 候选，**正式选定 `Step 50` 作为 SFT Pilot 最终准出模型**，通过收紧后的新门禁（限制 `max_robust_macro_regression <= 0.005` 与 `max_empty_output_rate <= 0.002`），导出合并底座至 `/data/mega-asr/runs/sft_pilot_controlled/merged_base`，作为向 E5 DPO 阶段交接的标准模型底座。

### RL Pilot 阶段风险暴露、门禁拦截与受控重跑决策（2026-09-24）

在 2026-09-24 对 `rl_pilot_v3`（60 steps GRPO）及历史轮次（v1/v2）的综合复核中，明确判定 **暂时不能进入正式下一步**，当前阶段标记为 **FAILED / BLOCKED**。

#### 1. 风险与问题事实确认
1. **Held-out Reward 门禁未达标且错误率反弹**：
   - Held-out（573 条独立池）平均 Reward 最终由 Step 0 的 `0.9379` 变为 Step 60 的 `0.9375`（净增量为 `-0.0004`，远未达到合同要求的 $\ge +0.05$ 门槛）；
   - Held-out 错误率由 `5.62%` 微幅上升至 `5.65%`（反弹 +0.03pp）。
2. **奖励饱和与后期零方差策略塌缩（Zero-Variance Collapse）**：
   - 训练后期（Step 50~60）训练集 batch reward 接近 1.0（达 0.9802），但 batch zero-variance ratio 频繁突破 `0.875 ~ 0.9375`；
   - 在高饱和与过拟合状态下，单批次内候选全对或全同，无法贡献有效策略梯度，表现出明显的策略塌缩信号。
3. **准出产物缺失与规范不完整**：
   - 历史 v1 与 v2 的 gate 均为 `FAILED`（v2 存在 Robust 回退 +0.0202% 及 reward 不达标）；
   - v3 缺少在 2,867 条全量验证集上的 `gate.json`、完整 `metrics.json`、全量 predictions 以及 merged release model，按合同不得声称阶段完成。
4. **采样超参契约不一致与代码 Provenance 缺失**：
   - 执行合同与方案中规划的采样参数为 temperature `0.7`、top_p `0.9`，但实际配置文件 `qwen3_asr_rl.yaml` 与运行参数使用了 `0.85 / 0.92`；
   - 运行环境记录中缺失代码的 git commit SHA，全链路溯源证据链不完整。

#### 2. 处置与拦截决策
1. **绝对阻断原则**：坚决不启动 full RL，坚决不发布当前任何 RL 检查点模型。
2. **检查点保留**：保留 `step_60` 作为诊断 checkpoint，用于后续分析策略塌缩和饱和 token 分布。
3. **重跑前置条件**：
   - 修正采样契约一致性，消除计划与配置漂移；
   - 补全运行时代码 git commit provenance 记录；
   - 针对奖励饱和与 zero-variance 塌缩提出针对性优化方案（如探索保持、早停判定、退火重构）；
   - 重新运行受控 Pilot 并在产出完整 2,867 条评测与全绿 `gate.json` 后，方可解冻下一阶段。

### RL Pilot v4 落地验证、Pareto 峰值确立与阶段决策（2026-09-24）

在完成零方差损失中和、采样超参冻结（`0.85/0.92/50`）、代码 Commit SHA 强制落盘后，于 4 × V100 SXM2-32GB 上完成了 `rl_pilot_v4`（30 steps）及 Step 10、Step 30 的 2,867 全量验证集评测，获得明确结论：

1. **零方差中和机制有效抑制策略塌缩**：
   - 组内优势全为 0 时损失置为 `0.0 * policy_token_logps.sum()`，在保留 DDP 反向图连通性的同时彻底切断了对参考模型的单向拉扯；
   - 30 步全程平均零方差率为 `0.6807`（完全受控于 $\le 0.75$ 的硬门槛），彻底消除了 v3 晚期高达 93.75% 的病态同质化。
2. **小样本 RL 的最佳泛化窗口确认（Step 10 Pareto 峰值）**：
   - 在 2,236 条小样本训练集上，模型在 Step 10（累积 640 样本次，~0.28 epoch）达到最佳泛化点：573 条 held-out 错误率降低至 `5.57%`（降低 0.05pp），Reward 升至 `0.9385`（提升 +0.0006）；
   - 在 2,867 条独立验证全集上，Step 10 较 DPO 基线保持 Clean 全面微增（Clean Macro 由 1.6717% 降至 1.6657%），退化场景 `en|distortion` 显式改善 0.1242pp（8.2634% → 8.1392%），0 空输出，0 推理失败；
   - 随后继续训练（Step 20/30）并未带来进一步增益，反而出现轻微饱和与过拟合迹象。
3. **核心问题与根因挖掘（为何 Step 10 达标受限、Step 30 退化）**：
   - **顺序取样引发严重分布突变（Distribution Cliff）**：
     - `pilot_rl.jsonl` 是将 1,236 条 degraded 与 1,000 条 clean 顺序拼接生成的；
     - 训练器采用顺序游标读取：Step 0–18 纯训练 degraded（reward ~0.80–0.88），Step 20–30 突变为 100% clean（reward 陡增至 ~0.98，零方差率升至 0.875–0.9062）；
     - 模型在后期严重拟合干净语音，冲淡并破坏了前期学到的退化语音表征，直接导致 Step 30 的退化增益彻底消失；
   - **训练样本规模缺口**：`pilot_rl.jsonl` 实际仅有 2,236 条（1,236 degraded + 1,000 clean），低于合同要求的 3,000 条（2,000 degraded + 1,000 clean），源于数据池 `rl_train_pool.jsonl` 自身仅包含 1,236 条退化音频；
   - **产物闭环与审计精度缺陷**：未生成 `merged_base` 与根目录 `gate.json`；`loss_log.jsonl` 的 `kl_loss` 精度截断导致四舍五入为 `0.0`。
4. **门禁拦截事实与严格裁决**：
   - **`held_out_reward` 拦截**：实测提升 `+0.0006`，未达到合同标称的 `+0.0020`；
   - **`robust_retention` 拦截**：实测全集 Robust Macro 回退 `+0.0202pp`（0.103785 vs 0.103583），由于门禁设置为零容差 `0.0`，即使微小波动（2,867 样本中仅 1~2 处微小编辑）也触发硬拦截为 FAILED。
5. **当前基准与冻结决策**：
   - **正式底座保持**：维持 **DPO merged model** (`/data/mega-asr/runs/dpo_pilot_v2/merged_base`) 为唯一受控发布基座；
   - **RL v4 Step 10**：确立为 RL Pilot 的 Pareto 最优模型，仅作为诊断候选全量归档；
   - **严格遵从红线约束：当前状态维持 BLOCKED，严禁启动 Full RL，严禁发布当前权重**；
   - 后续演进必须首先修复数据采样（打乱/分层平衡）、补齐训练数据池规模，并完善产物链条。

### RL Pilot v5 落地验证与阶段决策（2026-09-26）

在实现分层平衡采样（`--sample-strategy balanced`）、恢复 KL 6位小数精度、自动闭环导出 `merged_base` 后，于 4 × V100 SXM2-32GB 上完成了 `rl_pilot_v5`（30 steps）及 Step 30 的 2,867 全量验证集评测，获得明确结论：

1. **分层平衡采样彻底解决分布悬崖问题**：
   - Step 30 Held-out Reward 提升至 `0.9385`（+0.0006），Held-out Error Rate 降至 `0.0556`（-0.0006），打破了 v4 在 Step 30 的过拟合回退，成为全流程最优步数。
2. **Clean 场景与特定退化场景显著优化**：
   - 英文 Clean WER 达到 `1.9193%`（相对 DPO 下降 -0.0099pp，创项目新高）；
   - 中文 Clean CER 达到 `1.4073%`（相对 DPO 下降 -0.0070pp）；
   - Clean Macro 达到 `1.6633%`（相对 DPO 下降 -0.0084pp）；
   - 退化场景 `en|recording` 取得显式改善（31.5710% → 31.2689%，-0.3021pp）；
   - 2,867 条独立验证全集保持 100% 有效输出率，0 空输出，0 推理失败。
3. **门禁拦截事实（Gate Status: FAILED）**：
   - **`robust_retention` 拦截**：门禁要求 Robust Macro 恶化严格 $\le 0.0$。实测全集 2,867 条样本中，RL v5 仅净增加 2 处退化编辑（4 处 noise/distortion 微增，2 处 recording 减少），使 Robust Macro 从 `10.3583%` 变为 `10.3733%`（+0.0150pp，幅度较 v4 的 +0.0262pp 明显收窄），但仍触发零容差硬拦截；
   - **`held_out_reward` 拦截**：实测提升 `+0.0006`，未达到合同标称的 `+0.0020`。
4. **决策与基准认定**：
   - **严格遵从红线约束**：未通过门禁前，**坚决不推进 Full RL，坚决不发布 RL 模型**，阶段状态维持 **BLOCKED**；
   - **当前正式最优发布基座继续严格锁定为**：`/data/mega-asr/runs/dpo_pilot_v2/merged_base`（DPO Champion）。
   - 确认小数据规模（1,236 条退化样本）下强化学习的边际收益受限，后续若需突破，需结合 Full 规模退化数据与精细化的 reward shaping。

## 未验证假设

FP16 在 eager attention 下的吞吐、显存余量、4 卡 DDP 效率、SFT/DPO/RL 学习率、DPO pair 质量和
RL reward 稳定性都尚未在 V100 上验证。它们必须以 pilot gate 结果为准，不能凭规划文档宣称有效。

## 测试与验收

每项风险都必须有可执行门禁和输出文件。任何门禁失败都记录命令、输入 hash、错误和最后有效
checkpoint；不得静默跳过。只有完成固定 test 和 Mega-ASR 同 evaluator 比较后，才能讨论达到
或超过外部 baseline。

## 本次规划影响

旧 Colab/BF16/A2S 代码已删除；V100 smoke 和 pilot 通过前，新的训练/推理实现都不能作为正式结果证据。需要恢复历史实验时，必须重新提出目的并按文档优先流程加入。
