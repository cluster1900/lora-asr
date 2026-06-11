# AGENTS.md

## 项目定位

本项目是独立的 Qwen3-ASR Robust ASR 项目，基础模型统一使用 `Qwen/Qwen3-ASR-1.7B`。Mega-ASR 只作为方法参考和外部 baseline，不作为最终代码底座。

## 开发约束

- 不 fork 或魔改 Mega-ASR 作为主实现。
- 可以依赖 Qwen3-ASR 官方 `qwen-asr` API，但不得复用 Mega-ASR 上游工程中的私有 Qwen3-ASR wrapper、训练入口或 LoRA target 规则。
- 原 Mega-ASR 工程只允许放在本地忽略目录 `references/mega-asr-upstream/` 作为参考。
- 新工程代码不得写入 `references/mega-asr-upstream/`。
- `references/` 必须保持在 `.gitignore` 中，不提交其中任何文件。
- 新代码应面向 Qwen3-ASR 官方 `qwen-asr`/Transformers API 设计。
- Colab 是第一优先训练环境，脚本和 notebook 要默认支持 Google Drive 路径。
- 数据、训练、推理、router、评测应尽量解耦。
- 所有实验应能通过 JSONL manifest、配置文件和固定随机种子复现。

## 文档规范

- 默认先更新文档，再修改代码。除拼写、格式化、明显小 bug 外，任何功能性改动都必须先有对应文档。
- 重要架构、训练、数据、测试和风险变更必须同步到 `docs/qwen3-asr/`。
- 文档默认使用中文。
- 新增实验结论要记录到 `00_progress.md` 或相关方案文档。

## 文档组织标准

文档按功能维度组织，而不是只按时间或阶段堆叠：

- 架构文档：说明为什么做、模块目的、输入输出、依赖关系、达成标准。
- 开发文档：说明功能拆分、开发步骤、文件位置、接口约定、完成条件。
- 数据文档：说明数据来源、格式、增强策略、切分规则、质量检查。
- 训练文档：说明训练目标、配置、资源要求、checkpoint 策略、停止条件。
- 测试文档：说明测试集、指标、命令、通过标准、失败样本分析方式。
- 进度文档：记录已完成、进行中、下一步、阻塞、关键实验结论。
- 风险文档：记录技术风险、决策原因、替代方案和回滚条件。

每个新功能至少要更新对应的开发、测试、进度文档；涉及架构、数据或训练策略时，还必须更新相应专项文档。

## 文档质量标准

文档不能只写结论，必须包含：

- 背景：为什么要做这个改动。
- 范围：本次做什么、不做什么。
- 设计：模块、接口、输入输出、数据格式。
- 测试：如何验证、通过标准是什么。
- 验收：达到什么指标或行为才算完成。
- 影响：对已有数据、训练、推理、评测的影响。

没有测试标准和验收标准的功能文档视为不完整。

## 修改流程

功能性修改必须遵循：

1. 先更新或新增相关文档。
2. 再实现代码或 notebook。
3. 运行对应测试。
4. 更新进度文档和实验结果。
5. 最后总结改动、测试结果和未解决风险。

如果因为探索性质需要先写原型，必须在提交或交付前补齐文档。

## 质量要求

- 先做小闭环，再扩大规模。
- 每个训练改动都应有 baseline 对比和 WER/CER 评测。
- 必须关注 clean speech regression、空输出、重复输出和幻觉式输出。
- 不得在没有自有评测结果前声称达到或超过 Mega-ASR。

## 测试标准

代码或 notebook 变更后，至少完成对应层级测试：

- 数据脚本：能生成 JSONL；每条样本包含 `audio` 和目标文本；音频路径存在；train/val/test 无明显泄漏。
- 推理脚本：至少跑通 1 条 clean 音频和 1 条 degraded 音频；输出可写入 JSONL；失败样本要记录错误而不是中断整批任务。
- 训练脚本：先跑 5-20 step smoke test；checkpoint 可保存、加载、继续训练；训练配置必须随 checkpoint 保存。
- 评测脚本：能计算 WER/CER；能按 `scenario` 聚合；保存原始预测、归一化预测和指标。
- Router：必须报告 clean/degraded accuracy、precision、recall，并记录阈值来源。

## 验收标准

一个阶段只有同时满足以下条件才算完成：

- 有可复现命令或 Colab notebook。
- 有固定输入 manifest、配置文件和随机种子。
- 有输出结果文件和指标摘要。
- 有 baseline 对比。
- 文档已同步更新到 `docs/qwen3-asr/`。

MVP 最低验收：

- Qwen3-ASR baseline 可运行并产出 WER/CER。
- 数据 MVP 至少覆盖 clean、noise、reverb、far_field、dropout。
- LoRA MVP 至少在一个 degraded 场景上相对 base 改善 WER。
- Router MVP 在混合测试集上优于 always-base，且 clean regression 有量化记录。

未满足验收标准时，不应标记阶段完成。
