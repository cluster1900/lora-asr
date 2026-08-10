# 风险与决策

## 当前决策

- 只维护公开数据 -> A2S -> 推理 -> 评测一条路径。
- 基础模型固定 `Qwen/Qwen3-ASR-1.7B`，训练和正式 base 对照均为 BF16。
- 使用一个 343-target adapter，阶段切换 `requires_grad`，不合并多个 adapter。
- English WER 与 Chinese CER 分开报告。
- Mega-ASR 只作方法和外部 baseline，不进入运行时依赖。

## 风险与回滚条件

| 风险 | 门禁 | 回滚 |
| --- | --- | --- |
| 公开数据字段或 split 漂移 | pinned revision、probe、严格 schema | 停止并更新配置/文档 |
| 约 200 GB staging 过慢 | 本地 SSD、可恢复 shard、Drive 只存产物 | 先跑 128-row smoke |
| Qwen 模块结构漂移 | 343 target 数量、分组和 hash | 固定旧 revision 或重新评审 target |
| checkpoint 不能续训 | 10+2 resume | 不启动正式训练 |
| clean regression/幻觉增加 | 512 canary 与 BF16 base 相对阈值 | 回退到上一阶段 adapter |
| 指标不可比 | 同 manifest、同 evaluator、分语言指标 | 废弃该次比较 |

## 未验证假设

A2S 的阶段顺序、27/196/343 target 范围和学习率是工程设计，不是已完成的消融结论。公开数据
candidate 下载、解码、完整训练时间和最终增益仍未在 GPU 上验证。

## 测试与验收

每项风险都必须有可执行门禁和输出文件。任何门禁失败都记录命令、输入 hash、错误和最后有效
checkpoint；不得静默跳过。只有完成固定 test 和 Mega-ASR 同 evaluator 比较后，才能讨论达到
或超过外部 baseline。

## 本次精简影响

历史 v1-v6A 代码和结果从工作树删除，但仍可从 Git 历史恢复。它们不再是当前接口、测试或结论
的一部分；如需恢复某个实验，必须重新提出目的并按文档优先流程加入。
