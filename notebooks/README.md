# notebooks

## 目录职责

保存 Colab 运行编排入口、训练监视、指标可视化和小规模诊断 notebook。notebook 仅负责流程编排与调用，所有可复用的核心训练、推理与评测逻辑必须位于 `scripts/`、`train/`、`inference/` 或 `evaluation/` 中。

## 当前文件清单

- `README.md`：本说明文档。Phase 0 完成后加入单一主入口 notebook `00_bootstrap_baseline.ipynb`。

## 主要输入输出

- **输入**：Google Drive 挂载授权、项目仓库 Git 分支/SHA、运行配置参数。
- **输出**：在 Drive `runs/<run_id>/` 下生成的日志、训练状态、评估报告与图表展示。

## 使用入口

在 Google Colab 中打开：
- `00_bootstrap_baseline.ipynb`（Phase 0 完成后提供）：从全新 runtime 运行环境探针、依赖安装、基线推理与指标评测。

## 维护要求

notebook 必须支持从全新空白 runtime 完整运行；新增、重命名或删除 notebook 时，必须在同一提交同步更新对应目录的 `README.md` 与相关文档。
