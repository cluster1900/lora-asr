# Qwen3-ASR 文档

本目录描述 V100 服务器主线和迁移前的实际状态，不保存历史实验流水账。

- `00_progress.md`：当前完成度、下一步和实验结论。
- `01_architecture.md`：模块边界、数据流和交付物。
- `02_development_plan.md`：开发顺序、接口和完成条件。
- `03_data_plan.md`：公开数据、manifest 和质量门禁。
- `05_testing_plan.md`：本地、V100 smoke、数据和评测验收标准。
- `06_risks_and_decisions.md`：V100 运行时风险、决策和回滚条件。
- `07_v100_server_training_plan.md`：当前正式的服务器训练总方案。
- `08_execution_contract.md`：SFT、DPO、RL 的输入输出、数据角色、执行顺序和验收门禁。
- `09_rl_v10_design.md`：v10 退化集两轮锚定 GRPO 的设计与 Step 8 `BLOCKED_TRANSFER` 结局。
- `10_rl_v11_design.md`：v11 把学习率改为 `2e-5` 的方案、恢复边界，以及 Step 7 `STOPPED_KL` 后的门禁（FAILED，只未通过贪心 held-out）。
- `11_rl_v12_design.md`：v12 用学习率 `1e-5` 和原始奖励差；Step 4 门禁 FAILED，没有启动 v13。
- `12_rl_v14_design.md`：v14 续到 Step 8，门禁 FAILED；v15 在 Step 4 因贪心转负停止。
- `13_rl_v16_design.md`：v16 保持 `1e-5` 和原始奖励差，只训练局部改正；Step 8 门禁 FAILED 后停止。
- `14_rl_v17_design.md`：v17 把局部获胜优势改为 1；Step 4 贪心转负，门禁 FAILED。
- `15_rl_v18_design.md`：v18 从 v16 Step 8 继续原始奖励差；Step 10 因 `STOPPED_KL` 停止，门禁 FAILED。
- `16_rl_v19_design.md`：v19 把奖励差截断在 `0.05`；Step 12 门禁 FAILED，贪心 `−0.0001`，Robust `+0.000014`。
- `17_rl_v20_design.md`：v20 从 DPO Champion 新开，学习率改为 `5e-6`。Step 4 门禁 FAILED，贪心 `−0.0003`。
- `18_rl_v21_design.md`：v21 关掉 3 个音频投影。Step 4 门禁 FAILED，Robust `−0.000090`，贪心 `−0.0002`。
- `19_rl_v22_design.md`：v22 只读恢复 v21 Step 4。Step 11 门禁 FAILED，动作 `stop`，贪心 `−0.0004`，Robust `−0.000143`。
- `20_rl_v23_design.md`：v23 从 v22 Step 8 恢复，学习率 `5e-6`。Step 12 门禁 FAILED，贪心 `−0.0004`，Robust `−0.000090`。
- `21_rl_v24_design.md`：v24 从 DPO Champion 新开，策略梯度只打在改动 token 上。Step 4 门禁 FAILED，贪心 `+0.0001`，Robust `+0.000217`。续块在 Step 6 前退出。
- `22_rl_v25_design.md`：v25 把被删除的贪心 token 标成负的 `raw_gap`。2026-10-02 02:52 CST Step 4 门禁 FAILED，贪心 `−0.0003`，Robust `+0.000209`。
- `23_rl_v26_design.md`：v26 从 DPO Champion 新开，优化器跳过 `noise` 和 `recording`。2026-10-02 04:39 CST Step 4 门禁 FAILED，贪心 `−0.0005`，Robust `+0.000359`。
- `24_rl_v27_design.md`：v27 回到完整退化集，只在局部编辑距离内加入参考文本。Step 7 门禁 FAILED，贪心 `+0.0002`，Robust `+0.000014`，`STOPPED_KL`。
- `25_rl_v28_design.md`：v28 只读恢复 v27 Step 4，恢复后的学习率是 `5e-6`。参考文本和其余杠杆不变。
- `26_rl_pilot_gate_profiles.md`：RL Pilot 的 `release` 与 `pilot_feasibility` 分层门禁、阈值、输出和验收规则。

当前正式执行入口规划为服务器上的 CLI。旧 Colab notebook、依赖和训练入口已经删除，待按 V100
方案重新实现。

项目的目标路径为：服务器环境 -> 固定数据 manifest -> FP16 base -> SFT -> DPO -> RL -> release ->
统一推理和 WER/CER 评测。旧的 Colab/BF16/A2S 实现已删除，训练和推理 runner 待按该路径重新实现。
