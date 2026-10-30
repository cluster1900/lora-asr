# Qwen3-ASR 文档

本目录按功能记录从 baseline 到 SFT、DPO、RL 的新实现。文档先于功能代码更新，所有结论必须关联固定 manifest、配置、随机种子和输出指标。当前只有设计合同和目录合同已建立，尚无本仓库新实现的训练效果结论。

- `00_reset_plan.md`：本次清理范围、阶段计划和 Colab + Drive 合同。
- `01_architecture.md`：模块边界、输入输出和阶段关系。
- `02_development_plan.md`：实现顺序、接口和完成条件。
- `03_data_plan.md`：数据来源、五类 MVP 场景、manifest schema 和切分规则。
- `04_training_plan.md`：SFT、DPO、RL 的完整训练合同，包括官方监督接口、配置 schema、资源预算、阶段衔接、pair/rollout schema、损失、KL、停止、checkpoint、命令和验收。
- `05_testing_plan.md`：测试矩阵、夹具、指标、命令、恢复异常和 gate。
- `06_risks_and_decisions.md`：Colab、Drive、显存、恢复和数据质量风险。
- `07_colab_execution_contract.md`：环境启动、持久化、恢复和运行目录结构。
- `08_colab_drive_training_gym_design.md`：Colab/Drive 生命周期、单 A100 调度和 Training Gym 思想映射。
- `09_phase0_run_log.md`：Phase 0 每次命令的脚本、退出码和结果，用来回看失败发生在哪一步。
- `10_phase1_run_log.md`：Phase 1 manifest 构建和校验的命令、退出码和结果。
- `11_dataset_candidates.md`：公开中文语料候选比较、官方来源、首批抽样建议与接入缺口；候选研究不代表数据审计通过。
- `12_phase2_run_log.md`：Phase 2 SFT smoke 的命令、退出码和恢复结果。
- `00_progress.md`：当前进度、实验结果和阻塞项。

## 使用顺序

先读 `00_reset_plan.md` 了解范围，再读 `01_architecture.md`、`03_data_plan.md` 和 `04_training_plan.md`；实现时按 `02_development_plan.md` 分阶段，执行 Colab 前阅读 `07_colab_execution_contract.md`，验收按 `05_testing_plan.md`。`04_training_plan.md` 是算法和训练字段的单一来源；`07` 只补充 Colab/Drive 生命周期；`05` 只补充测试方法，不自行改写阈值。
