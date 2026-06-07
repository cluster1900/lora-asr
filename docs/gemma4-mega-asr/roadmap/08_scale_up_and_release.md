# 08 规模化与发布

最后更新：2026-06-07

## 背景

MVP 证明可行后，才进入更大数据、更长训练和发布候选阶段。规模化不是简单加数据，而是要保证可复现、可比较、可回滚。

## 目标

把 MVP 扩展为可复现实验和 release candidate。

## 范围

本步骤包括数据扩大、训练策略升级、benchmark 扩展、model card 和发布检查。

## 输入

- MVP adapter 和评测结果。
- 扩展数据源。
- scale-up 训练配置。
- 固定 benchmark。

## 输出

- release candidate adapter。
- 完整训练日志。
- benchmark 报告。
- model card 草稿。
- 发布 checklist。

## 执行步骤

1. 复盘 MVP 结果，确定最有效数据和 LoRA target。
2. 扩展训练数据到 100k+。
3. 加入更多真实录音和长音频。
4. 运行渐进式 SFT。
5. 在固定 benchmark 上评测。
6. 做错误分析和 clean regression 检查。
7. 准备 model card、使用说明和限制说明。
8. 决定是否发布 adapter。

## 测试标准

- scale-up 数据集 manifest 可复现。
- 训练可从 checkpoint 恢复。
- 至少两个数据源上完成评测。
- 结果能与 MVP 和 baseline 对比。

## 验收标准

- degraded 平均 WER 相对 base 下降 15%-25%，或有明确实验解释。
- clean regression 在 router 模式下小于 3% 相对退化。
- router 模式优于 always-base 和 always-LoRA 的混合场景表现。
- 文档包含训练配置、数据版本、评测结果和已知限制。

## 风险

- 数据规模扩大但质量下降。缓解：先扩高质量场景，再扩数量。
- 训练成本过高。缓解：分阶段训练，保留中间 checkpoint。
- 发布声明过度。缓解：明确适用场景、限制和未覆盖语言/场景。

