# 文档与阶段验收

最后更新：2026-07-12

## 当前文档权威顺序

1. `02_development_plan.md`：唯一执行合同。
2. `03_data_plan.md`：数据 schema、配比、缓存和防泄漏。
3. `04_colab_training_plan.md`：唯一 Colab 执行方式。
4. `05_testing_plan.md`：测试与指标门槛。
5. `06_risks_and_decisions.md`：当前决策和回滚条件。
6. `00_progress.md`：实际完成度与实验结果。

`roadmap/`、v1-v6 notebook 和旧配置是历史资料，不得覆盖当前合同。

## 文档完整性

每个功能改动交付前必须覆盖：背景、范围、设计、测试、验收和影响。

## 快速主线追踪

| 阶段 | 文档 | 实现 | 测试 | 结果 | 状态 |
| --- | --- | --- | --- | --- | --- |
| 公开 200k 数据 | 已完成 | 未实现 | 未运行 | 无 | 0/1 |
| 官方 Trainer + 10+2 | 已完成 | 未实现 | 未运行 | 无 | 0/1 |
| 200k 正式训练与发布 | 已完成 | 未实现 | 未运行 | 无 | 0/1 |

当前快速主线总状态：0/3。文档完成不等于阶段完成。

## 阶段硬门槛

### 数据完成

- 200k/10k/5k manifest 与 stats/rejects/hash 齐全。
- Resolved audio 存在、可解码。
- Source group、配额、防泄漏和人工抽听通过。

### Trainer 完成

- Golden batch 通过。
- 343 target 分组和禁止模块通过。
- 10+2 optimizer-step 真 resume 通过。
- 新进程 base+adapter 四条推理通过。

### 正式训练完成

- 100-step canary 通过。
- 50%/100% validation 完成并选出唯一候选。
- Bench 5k 和 clean fixed test 只对候选运行一次。
- Product MVP 指标通过。
- Release adapter、manifest 和新进程加载通过。

### Mega-ASR 目标完成

- Mega-ASR 发布模型在同一 evaluator 下完成本项目复测。
- 本项目 Bench macro error 不高于其 1.10 倍。
- Clean 与失败率门槛通过。

没有上述外部 baseline，不得声称达到或接近 Mega-ASR。

## 修改流程

1. 先更新本目录对应文档。
2. 实现最小代码改动。
3. 运行对应层级测试。
4. 把实际结果写入 `00_progress.md`。
5. 对照本文件更新阶段状态。

若测试或产物缺失，状态保持未完成，不使用“基本完成”“预计通过”等替代说法。
