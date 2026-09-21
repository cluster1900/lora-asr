# 风险与决策

## 当前决策

- 维护公开数据 -> FP16 base -> SFT -> DPO -> RL -> release -> 评测一条路径。
- 基础模型固定 `Qwen/Qwen3-ASR-1.7B`；V100 使用 FP16 和 eager attention，不使用 BF16 或
  FlashAttention-2。
- 训练使用独立 `/data/mega-asr` 环境和单机 4 卡 DDP；global batch 必须包含 world size。
- 完整链路固定为 SFT → DPO → RL；阶段交接采用 `merge_and_unload()` 策略产出下阶段基座，每阶段独立 adapter、manifest、checkpoint 和 gate。
- LoRA 目标层同时覆盖 LLM Decoder 与音频塔投影层，兼顾语义纠错与声学损伤补偿；A2S 不进入正式链路。
- DPO 推荐离线预计算参考模型 logprobs，消除训练时参考模型常驻显存开销（显存降 50%）。
- RL 采用带 KL 正则的 GRPO（组大小 $G=4$、temp 0.7、top_p 0.9），统一 `asr` 键名与序列惩罚。
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
| RL GRPO 候选同质化（组方差归零） | $G=4$、采样 temp=0.7/top_p=0.9 审计 | 微调采样温度或增大候选组规模 |
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

## 未验证假设

FP16 在 eager attention 下的吞吐、显存余量、4 卡 DDP 效率、SFT/DPO/RL 学习率、DPO pair 质量和
RL reward 稳定性都尚未在 V100 上验证。它们必须以 pilot gate 结果为准，不能凭规划文档宣称有效。

## 测试与验收

每项风险都必须有可执行门禁和输出文件。任何门禁失败都记录命令、输入 hash、错误和最后有效
checkpoint；不得静默跳过。只有完成固定 test 和 Mega-ASR 同 evaluator 比较后，才能讨论达到
或超过外部 baseline。

## 本次规划影响

旧 Colab/BF16/A2S 代码已删除；V100 smoke 和 pilot 通过前，新的训练/推理实现都不能作为正式结果证据。需要恢复历史实验时，必须重新提出目的并按文档优先流程加入。
