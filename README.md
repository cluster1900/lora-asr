# Qwen3-ASR Robust ASR

这是一个基于 `Qwen/Qwen3-ASR-0.6B` 官方 `qwen-asr`/Transformers API 的独立中文鲁棒 ASR 后训练项目。当前仓库已经重置为最小骨架，正式执行环境采用 Colab + Drive，训练使用单张 A100，顺序为 SFT → DPO → RL。

## 当前状态

已完成 Phase 0 环境与 baseline 验证、Phase 1 AISHELL-1 多声学场景 manifest 构建、Phase 2 SFT、Phase 3 DPO 与 Phase 4 RL pilot 训练与 release 发布。2026-10-26 已完成 Phase 5 封存 `test.jsonl`（2500 条）四阶段终局评测（Base: 2220 / SFT: 2204 / DPO: 2200 / RL: 2213 编辑）。详细实验进展与指标见 [`docs/qwen3-asr/00_progress.md`](docs/qwen3-asr/00_progress.md) 与 [`docs/qwen3-asr/12_phase2_run_log.md`](docs/qwen3-asr/12_phase2_run_log.md)。

## 目录

```text
configs/             训练和数据配置（重新实现后才加入具体配置）
docs/qwen3-asr/      架构、数据、训练、测试、风险和进度文档
evaluation/          WER/CER、场景聚合和 gate 评测
inference/           Qwen3-ASR baseline 与批量推理
notebooks/           Colab 编排入口；可复用逻辑必须放在 Python 模块
scripts/             manifest、环境、Drive 同步和阶段启动脚本
train/               SFT、DPO、RL runner
tests/               数据、恢复、训练和评测合同测试
```

## 新的工作方式

每个阶段先更新 `docs/qwen3-asr/` 中对应文档，再实现代码；先跑 5–20 step smoke，再扩大规模。所有 run 保存固定配置、manifest hash、随机种子、环境信息、checkpoint 状态和原始预测。只有同时具备可复现命令、输出指标、baseline 对比和文档记录，阶段才算完成。

## 起步计划

1. 在 Colab 从空 runtime 安装官方依赖并记录环境指纹。
2. 用固定 clean/degraded manifest 跑通 base 推理和 WER/CER。
3. 建立中文 clean、noise、reverb、far_field、dropout 五类数据 MVP。
4. 依次实现并验收 SFT、DPO、RL pilot。

详细范围、验收标准和 Drive checkpoint 合同见 [`docs/qwen3-asr/00_reset_plan.md`](docs/qwen3-asr/00_reset_plan.md)；训练实现、配置 schema、恢复、门禁和 run report 以 [`docs/qwen3-asr/04_training_plan.md`](docs/qwen3-asr/04_training_plan.md) 为准。可安装锁是根目录 `requirements.lock`。runner、训练 GPU smoke 和任何后训练效果仍未完成。

## 可恢复历史

重置前仓库状态保存在本地 Git 标签 `pre-reset-20261012`，不作为当前实现或实验结果来源。
