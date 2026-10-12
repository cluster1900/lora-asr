# 开发进度

最后更新：2026-10-12

## 当前状态

- 已完成：清理旧的版本化 RL 实现、历史方案文档、旧配置和旧合同测试。
- 已完成：建立中文 Qwen3-ASR-0.6B、单张 A100、Colab + Drive 执行合同，以及 SFT → DPO → RL 新计划和目录骨架。
- 已完成：审核并修正训练合同，收束到 `04_training_plan.md`。
- 进行中：Phase 0，尚未实现 baseline runner 或启动训练。
- 下一步：先实现环境探针、单条 Qwen3-ASR-0.6B 推理、JSONL 输出和 CER evaluator。

## 结论记录

目前没有新的训练指标，不对 SFT、DPO、RL 或外部系统做效果结论。

2026-10-12 的设计审核修正了这些合同冲突与技术隐患：
1. **监督文本与停机**：明确在转写后必须添加并监督终止符 `<|im_end|>`，前缀 mask 为 -100，避免解码停机失败造成失控长幻觉。
2. **LoRA 目标层**：明确限定于解码器注意力层，冻结音频编码器同名层，避免非预期修改音频前置特征提取。
3. **Reward 与健康度**：Reward 增加 `[-1.0, 1.0]` 截断，避免极端插入造成无界负 advantage 引发梯度爆炸；明确了空输出与重复输出判定规则。
4. **Drive Checkpoint 状态机**：明确了本地写入 → 生成校验码 → 复制到 `.incoming/` → 就位至正式目录 → 读回校验 → 原子更新 `latest.json` 的闭环流程。
5. **DPO 合同收束**：确立了离线预计算 reference logprob 作为单卡 A100 标准模式（节省显存），并修正了 held-out preference accuracy 的数学定义（基于隐式 reward 差值）。
6. **RL 稳定性**：明确了全组 advantage 为 0 时整组损失（含 KL 项）均归零，防止纯 KL 拉回 reference；把混合采样上的 `k3_proxy_train` 与固定 val 前缀、词表求和的 held-out `raw_kl` 分开，避免阈值量纲冲突。
7. **目录文档合同**：统一了所有 7 个受管目录 README 的五大规范章节，并重建了 `tests/test_directory_readmes.py` 自动化测试。

## 2026-10-12 训练文档完备性审计

本轮审计检查了训练方案、数据方案、开发计划、测试方案、Colab/Drive 执行合同、Training Gym 映射、风险决策和 7 个受管目录 README。已补齐：

1. `04_training_plan.md` 的 `qwen3-asr-train-v1` 配置模板、A100 资源预算、环境锁和确定性规则、stage adapter 衔接、DPO pair 与 RL rollout schema、累积梯度与 scheduler 语义、checkpoint/state 文件 schema、恢复状态机、CLI 形状、run report 和发布选择规则。
2. DPO 的数据构造：gold 是 chosen、SFT greedy 是 rejected，删除“假设优于 gold”这一恒不成立的审计；sequence-sum logprob、token mask、离线 cache 复核和 held-out preference ties 规则固定下来。
3. RL 的实验命名和边界：`group_pg_mixed_v1` 明确不是标准 GRPO/PPO；训练 `k3_proxy` 与固定前缀、词表求和的 held-out `raw_kl` 分开，避免把混合采样估计量误称为无偏 KL；全组无信号时 optimizer/scheduler 不推进。
4. `05_testing_plan.md` 的测试矩阵、失败样本分析、恢复/异常验收和阶段 gate；`03_data_plan.md` 的字段约束、来源/物化、配额、泄漏、拒收审计；`07/08` 的 Colab 状态、Drive staging、观测和影响；`06` 的风险触发与回滚。
5. 官方 Qwen3-ASR 接口约束：processor/chat template、语言前缀、动态 EOS、训练 forward 与 `transcribe()` 的边界、解码器注意力 LoRA 目标和 checkpoint 可加载性。

同时为配置示例声明的 `jieba` WER 辅助分词器补入依赖范围；它仍需在 Phase 0 锁定版本并记录字典/配置 hash，不能直接当作已完成评测环境。

审计结论：**文档合同已达到可实现级别，但项目阶段仍为 Phase 0 进行中**。当前不存在 baseline、SFT、DPO 或 RL 的新 runner、GPU smoke、gate.json、release 或指标，不能标记任何后训练阶段完成。下一步按 `02_development_plan.md` 实现环境探针、锁定依赖、单条推理和 evaluator，并把真实结果回写本文件。
