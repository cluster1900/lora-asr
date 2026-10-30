# notebooks

## 目录职责

保存 Colab 运行编排入口、训练监视、指标可视化和小规模诊断 notebook。notebook 仅负责流程编排与调用，所有可复用的核心训练、推理与评测逻辑必须位于 `scripts/`、`train/`、`inference/` 或 `evaluation/` 中。

## 当前文件清单

- `README.md`：本说明文档。
- `00_bootstrap_baseline.ipynb`：只调用 `scripts/run_phase0_baseline.py`，不在单元格里加载模型。
- `01_qwen3_asr_post_training_pipeline.ipynb`：Phase 0 → Phase 1 → Phase 2 (SFT) → Phase 3 (DPO) → Phase 4 (RL) → Phase 5 (Sealed `test.jsonl` 四阶段终局评测) 主 Colab 编排 notebook，所有步骤仅调用 `scripts/`、`train/`、`inference/`、`evaluation/` CLI，不在单元格里保存隐藏状态或加载模型。

## 主要输入输出

- **输入**：Google Drive 挂载授权、项目仓库 Git 分支/SHA、运行配置参数。
- **输出**：在 Drive `runs/<run_id>/` 下生成的日志、训练状态、评估报告与图表展示。

## 使用入口

在 Google Colab 的普通内存 A100 runtime 中打开 `00_bootstrap_baseline.ipynb`。运行前把仓库放在 `/content/lora-asr`，并挂载 Drive。终端等价命令是 `python3 scripts/run_phase0_baseline.py`。

## 维护要求

notebook 必须支持从全新空白 runtime 完整运行；新增、重命名或删除 notebook 时，必须在同一提交同步更新对应目录的 `README.md` 与相关文档。
