# 风险与决策

## 当前决策

- 只维护公开数据 -> A2S -> 推理 -> 评测一条路径。
- 基础模型固定 `Qwen/Qwen3-ASR-1.7B`，训练和正式 base 对照均为 BF16。
- 使用一个 343-target adapter，阶段切换 `requires_grad`，不合并多个 adapter。
- English WER 与 Chinese CER 分开报告。
- Mega-ASR 只作方法和外部 baseline，不进入运行时依赖。
- 首轮不使用 Teacher：gold transcript 已满足监督和 curriculum 评分；GPT-5.5 只有文本/图片输入，
  不承担音频伪标注。
- Hub 数据只按配额流式 staging，不镜像完整数据集；curriculum base 推理按 60k/100k/160k/200k
  逐级扩容。

## 风险与回滚条件

| 风险 | 门禁 | 回滚 |
| --- | --- | --- |
| 公开数据字段或 split 漂移 | pinned revision、probe、严格 schema | 停止并更新配置/文档 |
| AISHELL-1 使用第三方 Parquet 重打包 | pinned revision、OpenSLR license URL、schema probe | 换回可验证的 SLR33 物化器 |
| staging 过慢或 runtime 中断 | 本地 SSD、逐条持久化、按配额停止 | 先跑 128-row smoke；丢盘后重下缺失行 |
| Robust 原始话语跨 scenario 泄漏 | base source identity 固定 90/10 分区 | 停止 build 并修复 source identity |
| Teacher/API 增加成本且改写标签 | 首轮配置和 notebook 无 Teacher/API key | 保留 gold transcript 和 BF16 base 评分 |
| Qwen 模块结构漂移 | 343 target 数量、分组和 hash | 固定旧 revision 或重新评审 target |
| checkpoint 不能续训 | 10+2 resume | 不启动正式训练 |
| clean regression/幻觉增加 | 512 canary：robust 相对阈值、clean 绝对阈值、失败率 | 回退到上一阶段 adapter |
| 指标不可比 | 同 manifest、同 evaluator、分语言指标 | 废弃该次比较 |

## 未验证假设

A2S 的阶段顺序、27/196/343 target 范围和学习率是工程设计，不是已完成的消融结论。公开数据
candidate 下载、解码、90/10 分区容量、完整训练时间和最终增益仍未在 GPU 上验证。

## 测试与验收

每项风险都必须有可执行门禁和输出文件。任何门禁失败都记录命令、输入 hash、错误和最后有效
checkpoint；不得静默跳过。只有完成固定 test 和 Mega-ASR 同 evaluator 比较后，才能讨论达到
或超过外部 baseline。

## 本次精简影响

历史 v1-v6A 代码和结果从工作树删除，但仍可从 Git 历史恢复。它们不再是当前接口、测试或结论
的一部分；如需恢复某个实验，必须重新提出目的并按文档优先流程加入。
