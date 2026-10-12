# 开发计划

最后更新：2026-10-12

## 背景与范围

本计划把训练合同拆成可交付的小闭环，先完成 baseline 和数据门禁，再实现 SFT、DPO、RL。目标是每个阶段都能在全新 Colab runtime 用固定 manifest、配置、seed 和 Git SHA 重跑；本阶段不把未实现的 runner 或 notebook 当作完成。

## 顺序

1. Phase 0：环境探针、官方模型加载、单条推理和 evaluator。
2. Phase 1：manifest builder、质量检查和固定切分。
3. Phase 2：SFT runner 与恢复合同。
4. Phase 3：DPO pair builder、reference logprob 和 preference gate。
5. Phase 4：RL rollout、reward、KL 和 held-out gate。
6. Phase 5：Colab notebook 串联和规模化运行。

## 阶段交付物与完成条件

| 阶段 | 必须交付 | 通过条件 | 失败后的动作 |
| --- | --- | --- | --- |
| Phase 0 | 环境探针、base 推理、CER evaluator、最小 Colab 编排 | clean/degraded 各 1 条成功；失败行可记录；Drive 产物能读回 | 修复依赖/API/音频前端，不启动训练 |
| Phase 1 | manifest builder/validator、五类 scenario、固定切分 | schema、音频、source 泄漏、hash 和配额全通过 | 标记 `insufficient_data` 或 `data_failed` |
| Phase 2 | SFT runner、LoRA、checkpoint/resume、val gate | 5–20 step smoke 恢复通过；相对 base 的 gate 写出结果 | 保留失败 run，调整新配置新 run |
| Phase 3 | DPO pair builder、离线 reference logprob、preference gate | pair 审计和 held-out preference、CER gate 全通过 | 不生成 DPO release |
| Phase 4 | RL rollout/reward/GRPO、reference freeze、held-out gate | reward、raw KL、reference checksum、clean/degraded gate 全通过 | 不生成 RL release |
| Phase 5 | 主 notebook、规模化启动和最终 test 报告 | run report、test 指标、baseline 对比和文档结论齐全 | 阶段保持未完成 |

每个阶段实现前先更新对应专项文档和目录 README；实现后先跑本地合同测试，再跑 Colab preflight 和 smoke，最后才允许扩大数据或步数。

## 设计与接口约定

- 训练、评测和编排 CLI 必须支持 `--config`、`--output-dir` 和 `--seed`；数据/推理工具可用显式参数替代，但必须把最终解析配置写入结果。训练和正式评测还支持 `--run-id`；trainer 另支持 `--manifest`、`--val-manifest` 和 `--resume`。DPO 另支持 `--pair-manifest`。
- `--resume` 只恢复 `04_training_plan.md` 中核对通过的 checkpoint。推理和评测按配置重跑，不恢复 optimizer。
- 代码不得依赖 notebook 中的全局变量。
- manifest 路径、模型路径和 Drive 根目录由配置或参数传入。
- 错误样本写入 JSONL；批处理默认继续执行，并在汇总中报告失败数。

所有入口必须返回非零退出码表示合同失败，并同时写结构化事件；单条数据失败只写该行 `error`，不能悄悄吞掉异常。CLI 的 `--help`、schema 版本和输出文件名属于公共接口，变更时同步更新训练/测试文档。

## 测试

每一阶段先用合成 fixture 验证 schema、hash、错误退出码和 README 合同，再进行 Colab preflight。训练阶段额外执行 5–20 step smoke、保存/中断/恢复和固定 val 评测；DPO/RL 的算法单测不能以 loss 下降代替数学合同。所有测试命令、夹具和通过线记录在 `05_testing_plan.md`。

## 完成条件

实现前先补充本目录和 `05_testing_plan.md`；实现后完成 smoke、恢复、固定 manifest 评测和目录 README 合同测试；结果写入 `00_progress.md`。每个 run 还必须生成 `run_report.md`（或等价 JSON），列出环境、config、manifest、父版本、指标、gate、错误和恢复记录。

## 验收与影响

完成开发阶段要求代码、配置、测试、产物和文档五者一致；只有文档更新或本地 mock 通过不能标记阶段完成。新增文件、接口、CLI 参数或产物路径时，必须同步更新 `configs/`、`train/`、`evaluation/`、`inference/`、`scripts/`、`notebooks/` 或 `tests/` 的 README（适用者）以及 `docs/qwen3-asr/00_progress.md`。
