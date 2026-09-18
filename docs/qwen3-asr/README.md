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

当前正式执行入口规划为服务器上的 CLI。旧 Colab notebook、依赖和训练入口已经删除，待按 V100
方案重新实现。

项目的目标路径为：服务器环境 -> 固定数据 manifest -> FP16 base -> SFT -> DPO -> RL -> release ->
统一推理和 WER/CER 评测。旧的 Colab/BF16/A2S 实现已删除，训练和推理 runner 待按该路径重新实现。
