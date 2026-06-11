# 文档验收与追踪矩阵

最后更新：2026-06-07

## 验收结论

当前文档体系已达到“可指导后续实现”的标准。

文档已经覆盖：

- 项目定位和边界。
- 最终产品目标。
- 架构模块和模块达成标准。
- 开发步骤和执行顺序。
- 数据格式、增强策略和切分规则。
- Colab 训练路径。
- 测试指标和验收标准。
- 风险、决策和待定事项。
- 每一步 roadmap 的输入、输出、测试和验收。

## 本次验收发现的问题

### P1: 开发方案与 roadmap 步骤编号不一致

问题：

- 旧版开发方案使用 Phase 0-6，roadmap 使用 00-08，容易导致执行时不知道以哪个为准。

处理：

- 已重写 `02_development_plan.md`，以 roadmap 00-08 为唯一执行主线。

状态：

- 已解决。

### P2: 缺少产品能力追踪矩阵

问题：

- 文档有架构和路线图，但没有把最终产品能力映射到模块、步骤和验收文档。

处理：

- 新增本文件，补充追踪矩阵。

状态：

- 已解决。

### P3: 待定决策没有挂到执行步骤

问题：

- 风险文档列出了 D006-D009，但没有说明在哪一步必须解决。

处理：

- 已补充待定决策处理点。

状态：

- 已解决。

### P4: 顶层文档存在 Phase 与 roadmap 编号混用

问题：

- README 和架构文档中使用 Phase 表述，容易与 roadmap 的 00-08 执行步骤混淆。

处理：

- 已将 README 改为 roadmap 表述。
- 已将架构文档中的阶段验收改为产品里程碑验收。

状态：

- 已解决。

### P5: 数据方案和 Colab 训练方案缺少显式测试/验收标题

问题：

- 两份专项文档有质量要求，但没有统一的 `测试标准` 和 `验收标准` 标题。

处理：

- 已为 `03_data_plan.md` 和 `04_colab_training_plan.md` 补充测试标准与验收标准。

状态：

- 已解决。

### P6: 原 Mega-ASR 工程与新工程主路径混杂

问题：

- 根目录原本保留了 Mega-ASR 的 `src/MegaASR`、`scripts`、`assets`、`examples` 和推理入口，容易让后续开发误以为要继续在上游工程中改造。

处理：

- 已将原 Mega-ASR 上游工程隔离到本地忽略目录 `references/mega-asr-upstream/`。
- 已新增根目录 README，并在文档中说明 `references/` 不进入 git。
- 已更新 `AGENTS.md`、项目原则和 `01_project_scaffold.md`，明确参考工程固定隔离路径。

状态：

- 已解决。

### P7: 新工程根目录缺少最小骨架

问题：

- 仅隔离上游工程还不够，根目录需要立即呈现新工程的主结构，否则后续仍缺少明确落点。

处理：

- 已创建 `configs/`、`data/`、`evaluation/`、`inference/`、`notebooks/`、`router/`、`scripts/`、`train/`。
- 已创建基础配置和 `baseline_smoke.example.jsonl`。
- 已验证 JSONL 和 YAML 可解析。

状态：

- 已解决。

## 产品能力追踪矩阵

| 产品能力 | 架构模块 | 主要路线图步骤 | 测试/验收来源 | 当前状态 |
| --- | --- | --- | --- | --- |
| Qwen3-ASR base ASR 推理 | 基础模型、推理管线 | 02 Baseline 评估 | `02_baseline_eval.md`、`05_testing_plan.md` | 已规划 |
| 鲁棒 ASR LoRA adapter | ASR LoRA、训练管线 | 05 LoRA 训练 MVP | `05_lora_training_mvp.md`、`01_architecture.md` | 已规划 |
| 音频质量 router | Router、推理管线 | 07 Router MVP | `07_router_mvp.md`、`05_testing_plan.md` | 已规划 |
| clean/degraded 动态推理 | Router、部署模式 | 07 Router MVP | `07_router_mvp.md`、`01_architecture.md` | 已规划 |
| 数据 manifest | 数据管线 | 03 数据 MVP | `03_data_mvp.md`、`03_data_plan.md` | 已规划 |
| 声学退化增强 | 数据管线、音频预处理 | 04 音频增强 | `04_audio_augmentation.md`、`03_data_plan.md` | 已规划 |
| Colab 训练闭环 | 训练管线 | 02、03、04、05、06 | `04_colab_training_plan.md`、roadmap | 已规划 |
| WER/CER 评测 | 评测管线 | 06 评测与错误分析 | `06_eval_and_error_analysis.md`、`05_testing_plan.md` | 已规划 |
| scenario-level 分析 | 评测管线、数据 metadata | 03、06 | `06_eval_and_error_analysis.md` | 已规划 |
| 产品发布候选 | 部署模式、评测体系 | 08 规模化与发布 | `08_scale_up_and_release.md` | 已规划 |

## 文档完整性检查

| 检查项 | 结果 |
| --- | --- |
| 是否明确 Mega-ASR 仅作为参考 | 通过 |
| 是否明确最终目标是类似 Mega-ASR 的产品能力 | 通过 |
| 是否有架构模块说明 | 通过 |
| 是否有每个模块的目的和达成标准 | 通过 |
| 是否有路线图总览 | 通过 |
| 是否有每一步细化文档 | 通过 |
| 每一步是否有测试标准 | 通过 |
| 每一步是否有验收标准 | 通过 |
| 是否有数据格式和切分规则 | 通过 |
| 是否有 Colab 训练规划 | 通过 |
| 是否有风险与决策记录 | 通过 |
| 是否有功能到步骤的追踪矩阵 | 通过 |
| 是否隔离原 Mega-ASR 上游工程 | 通过 |
| 是否创建新工程最小骨架 | 通过 |

## 衔接关系检查

### 文档治理到项目骨架

状态：通过。

说明：

- `AGENTS.md` 和 `00_document_governance.md` 已规定文档先行。
- `01_project_scaffold.md` 负责把规范落到目录结构。

### 项目骨架到 Baseline

状态：通过。

说明：

- `01_project_scaffold.md` 规划了 `inference/`、`evaluation/`、`configs/baseline/`。
- `02_baseline_eval.md` 使用这些目录实现 baseline。

### Baseline 到数据与增强

状态：通过。

说明：

- baseline 需要 smoke/test manifest。
- `03_data_mvp.md` 负责正式 manifest。
- `04_audio_augmentation.md` 负责 degraded 数据生成。

### 数据与增强到 LoRA 训练

状态：通过。

说明：

- `05_lora_training_mvp.md` 依赖 train/val manifest 和增强样本。
- 训练输出进入统一评测。

### LoRA 训练到评测

状态：通过。

说明：

- `06_eval_and_error_analysis.md` 对 base、LoRA、router 使用统一评测。
- 验收标准要求至少一个 degraded 场景改善。

### 评测到 Router

状态：通过。

说明：

- Router 必须在确认 LoRA 有收益且 clean regression 已量化后进行。
- `07_router_mvp.md` 验收要求 router 模式优于 always-base。

### Router 到产品发布

状态：通过。

说明：

- `08_scale_up_and_release.md` 负责扩大规模、benchmark、model card 和限制说明。

## 当前仍需在执行中确认的事项

这些不是文档缺陷，而是执行阶段必须做出的实验决策：

- MVP 语言选择：英文、中文或双语。
- 首个源数据集。
- 训练框架选择：Unsloth、Transformers Trainer、TRL 或自定义循环。
- 是否发布中间 adapter。

对应处理位置见 `06_risks_and_decisions.md` 的待定决策处理点。
