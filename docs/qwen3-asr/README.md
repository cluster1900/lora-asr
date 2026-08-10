# Qwen3-ASR 文档

本目录只描述当前可执行主线，不保存历史实验流水账。

- `00_progress.md`：当前完成度、下一步和实验结论。
- `01_architecture.md`：模块边界、数据流和交付物。
- `02_development_plan.md`：开发顺序、接口和完成条件。
- `03_data_plan.md`：公开数据、manifest 和质量门禁。
- `04_colab_training_plan.md`：Colab 环境与 A2S 训练合同。
- `05_testing_plan.md`：测试命令、指标和验收标准。
- `06_risks_and_decisions.md`：当前风险、决策和回滚条件。

唯一执行入口：`notebooks/12_fast_finetune_colab.ipynb`。Notebook 只编排文档中的正式 CLI。

项目只有一条正式路径：公开数据准备 -> BF16 base 评测 -> 单 adapter A2S -> 统一推理 ->
WER/CER 评测。历史 v1-v6A、router、旧 notebook、checkpoint 和结果不属于当前仓库。
