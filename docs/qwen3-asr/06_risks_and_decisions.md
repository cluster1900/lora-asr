# 风险与决策

## 背景与范围

训练在可中断的 Colab 单卡上运行，数据、模型 API、偏好缓存、rollout、指标和 Drive 恢复任一环节出错都可能制造不可复现实验。本文件记录已冻结的工程决策、可观测触发信号、缓解方案和回滚条件；不把风险表当作训练结果，也不重复旧仓库的历史 gate。

## 设计原则

风险处置必须有结构化状态、可定位事件和不可变输入 hash。优先停止并保留证据，再用新 run 修复；不覆盖 base/父 release，不从不同 run 拼接 step 0、reward 或 checkpoint。具体算法字段以 `04_training_plan.md`、测试夹具以 `05_testing_plan.md` 为准。

## 已定决策

- 模型底座固定为官方 `Qwen/Qwen3-ASR-0.6B`。
- 数据主线固定为中文，主指标为 `zh-cer-v1` CER。reward 由 CER 换算，两者分别报告。
- Colab 单张 A100 是正式执行环境；Drive 只做持久化和共享存储，训练使用本地 runtime 磁盘。
- 训练顺序固定为 SFT → DPO → RL。父版本依次是 base、SFT release、DPO release。
- 所有阶段使用固定 manifest、配置和随机种子。门禁只读 val，test 只在发布时评一次。
- 评测解码固定为 greedy。RL 采样参数只用于训练候选。
- 初始 LoRA 只训练语言模型注意力投影，音频编码器和 projector 冻结。
- 损失对转写 token 取平均。RL advantage 只做组内减均值，之后不再改写成常数。
- `latest.json` 只指向最近的完整 checkpoint。release 只复制 val 门禁为 `passed` 的 step。
- 训练细节以 `04_training_plan.md` 为准。重置前 RL pilot 的数值门禁不带入新 run。
- 训练配置使用 `qwen3-asr-train-v1` schema；未知字段、非 greedy eval、错误 world size 和 LoRA 目标层在 preflight 拒绝。
- `micro_cer` 是 gate 的主聚合，`macro_cer` 只作补充；不足 50 条的 scenario 不能单独构成改善结论。
- 训练状态、恢复指针、事件日志和 release 使用不同文件，不能用 `latest.json` 代替 gate 或 release。

## 风险

- Colab runtime 重启：通过频繁 checkpoint、恢复 smoke 和启动探针缓解。
- Drive I/O 不稳定：训练前复制到本地，完成后 checksum 同步。
- GPU 型号或显存变化：启动时记录并执行 batch/seq 长度门禁。
- 依赖漂移：锁定版本并保存 `pip freeze`。
- 退化改善伴随 clean regression：统一 clean/degraded gate，禁止只看单一 reward。
- DPO/RL 数据污染：保存 source ID 和交集审计，test 不进入训练、配对、rollout 和选 checkpoint。
- 评测和解码混用：greedy 与 rollout 分属两份配置，hash 不同则结果为 `incomparable`。
- Drive 写入不完整：恢复只接受读回 checksum 与 `latest.json` 一致的 checkpoint。
- 无更新被当成提升：RL 通过线要求相对本 run step 0 的 reward 提升，CER 不退化单独不够。

## 风险矩阵与触发动作

| 风险 | 触发信号 | 缓解 | 回滚/处置 |
| --- | --- | --- | --- |
| 官方 API/监督格式漂移 | 模型加载或 token mask 合同失败 | 固定 `qwen-asr`/模型 revision，记录版本，运行 label-mask contract | 标记 `blocked_api`，不启动训练 |
| A100 显存不足 | preflight/OOM 或峰值显存无余量 | micro batch 1、长度上限、梯度累积和显存报告 | 状态 `oom`，新配置新 run |
| 标签前缀/终止符错误 | prefix 被计入 loss、`<|im_end|>` 缺失、空输出上升 | mask 单测与 smoke 解码 | 丢弃 checkpoint，回到父 release |
| 数据/增强泄漏 | source/audio/text hash 跨 split | 分组切分、交集审计、manifest hash | 状态 `data_failed`，重新生成 manifest |
| DPO preference 污染 | tie、空 pair、source 交集、reference hash 不一致 | 离线 pair audit 与 held-out schema | 不生成 DPO release |
| RL 无信号/发散 | zero-advantage 过高、非有限 loss、raw KL 超阈值 | 全组归零、梯度裁剪、KL 停止规则 | 回到 DPO release，新 run |
| Drive 半成品 | checksum 不一致、`.incoming` 被选中 | staging、读回校验、原子 `latest.json` | 保留旧指针，状态 `sync_failed` |
| 恢复重复 step | global step 倒退或重复、日志尾部不一致 | 只从完整 checkpoint 恢复，按 step 截断事件 | `resume_rejected`，人工检查后重试 |
| clean regression/健康度恶化 | clean CER、空/重复率超阈值 | 同一 val gate 与立即停止 | 回到上一通过 release |
| 依赖或硬件漂移 | environment hash/GPU/dtype 不同 | 启动探针、锁定 requirements、记录环境 | 标记 `incomparable` 或新 baseline |

## 测试

风险合同必须用 fixture 或 preflight 模拟：坏音频和 source 泄漏、未知配置字段、错误 LoRA target、BF16/OOM、非有限 loss、Drive 截断/错 hash、恢复 step 重复、reference 参数变化和 clean 健康度越界。每个触发器都应产生预期状态、事件和无 release 结果；测试命令及断言见 `05_testing_plan.md`。

## 回滚

任一阶段 gate 失败时回到上一个通过的 checkpoint；不覆盖 base、SFT 或 DPO 目录。旧仓库仅作为 Git 标签 `pre-reset-20261012` 的恢复来源。

## 验收与影响

风险合同的验收是每个高风险项都有可观察触发信号、结构化状态和可执行回滚；测试方案覆盖模拟 OOM、坏 checksum、恢复 hash 不匹配、source 泄漏、非有限 loss 和 reference 变化。新增风险或替代方案必须同步到 `04_training_plan.md`、`05_testing_plan.md` 和 `00_progress.md`，否则不视为完成。
