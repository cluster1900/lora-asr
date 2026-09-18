# 风险与决策

## 当前决策

- 维护公开数据 -> FP16 base -> SFT -> DPO -> RL -> release -> 评测一条路径。
- 基础模型固定 `Qwen/Qwen3-ASR-1.7B`；V100 使用 FP16 和 eager attention，不使用 BF16 或
  FlashAttention-2。
- 训练使用独立 `/data/mega-asr` 环境和单机 4 卡 DDP；global batch 必须包含 world size。
- 完整链路固定为 SFT → DPO → RL；每阶段独立 adapter、manifest、checkpoint 和 gate。
- target contract 必须从官方模型运行时发现；A2S 不进入正式链路。
- English WER 与 Chinese CER 分开报告。
- Mega-ASR 只作方法和外部 baseline，不进入运行时依赖。
- 不使用 Teacher、Router 或量化训练；RL 是正式链路第三阶段。
- Hub 数据只按角色配额流式 staging，不镜像完整数据集；SFT、DPO、RL 使用隔离 pool。

## 风险与回滚条件

| 风险 | 门禁 | 回滚 |
|---|---|---|
| V100 FP16 数值不稳定 | 单 batch 前向反向、有限 loss/gradient、10+2 resume | 降低 micro batch 或回退到上一有效 checkpoint |
| eager attention 显存不足 | 128-row/10-step smoke 记录显存 | 降低 micro batch、增加 accumulation，不改 global batch 目标 |
| 公开数据字段或 split 漂移 | pinned revision、probe、严格 schema | 停止并更新配置/文档 |
| staging 过慢或服务器 CPU 被其他任务占用 | `/data/mega-asr` 独立缓存、逐条持久化、记录 CPU 负载 | 等待资源空闲或降低 worker，不读取其他项目产物 |
| Robust 原始话语跨 scenario 泄漏 | base source identity 固定 90/10 分区 | 停止 build 并修复 source identity |
| Hugging Face 直连不稳定 | mirror/ModelScope 下载记录和文件 hash | 切换已验证镜像，禁止无 hash 数据进入 manifest |
| checkpoint 不能续训或 global batch 漂移 | 10+2 resume、记录 world size | 不启动 pilot |
| clean regression/幻觉增加 | 每阶段 robust/clean/失败率阈值 | 回退到上一阶段 adapter |
| DPO pair 质量或 preference 泄漏 | chosen/rejected 审计、ties rejects、held-out accuracy | 停止 DPO，重建 dpo_pool |
| RL reward 投机或 KL 发散 | reward 组件单测、KL/rollout gate、reference freeze | 停止 RL，回退到 DPO adapter |
| 4 卡 DDP 梯度/数据 shard 不一致 | global batch、sample_id 去重、world size 记录 | 停止当前 run，保留最后有效 checkpoint |
| 指标不可比 | 同 manifest、同 evaluator、分语言指标 | 废弃该次比较 |
| 运行时加载非官方 wrapper | import 和 model revision 记录 | 停止并移除不合规依赖 |

## 未验证假设

FP16 在 eager attention 下的吞吐、显存余量、4 卡 DDP 效率、SFT/DPO/RL 学习率、DPO pair 质量和
RL reward 稳定性都尚未在 V100 上验证。它们必须以 pilot gate 结果为准，不能凭规划文档宣称有效。

## 测试与验收

每项风险都必须有可执行门禁和输出文件。任何门禁失败都记录命令、输入 hash、错误和最后有效
checkpoint；不得静默跳过。只有完成固定 test 和 Mega-ASR 同 evaluator 比较后，才能讨论达到
或超过外部 baseline。

## 本次规划影响

旧 Colab/BF16/A2S 代码已删除；V100 smoke 和 pilot 通过前，新的训练/推理实现都不能作为正式结果证据。需要恢复历史实验时，必须重新提出目的并按文档优先流程加入。
