# Qwen3-ASR 项目重置计划

## 背景

仓库当前包含多轮 RL 方案、服务器专用 runner、历史门禁和实验记录。它们混合了不同的数据合同、解码预算和训练假设，继续在原结构上迭代会增加恢复和复现实验的成本。本次重置把项目收敛到一个最小、可复现的中文 Qwen3-ASR 后训练闭环，并将 Colab + Drive、单张 A100 作为正式执行环境。

## 范围

保留：`AGENTS.md`、Git 配置、项目根说明、目录接口和本计划。

删除：旧的 SFT/DPO/RL runner、版本化 RL helper、历史启动脚本、旧配置、历史实验设计文档和与旧实现绑定的合同测试。数据、模型权重和结果目录不纳入仓库，也不在本次重置中删除。

重新建立：

1. Qwen3-ASR-0.6B 官方 `qwen-asr`/Transformers API 的最小推理与评测基线；
2. 中文 JSONL manifest 和 clean/noise/reverb/far_field/dropout 五类 MVP 数据闭环；
3. SFT → DPO → RL 三阶段训练接口；
4. Colab 环境、Drive checkpoint、断点恢复和固定随机种子合同；
5. 每阶段的 smoke test、WER/CER、clean/degraded gate 和进度记录。

## 新的阶段计划

### Phase 0：环境与基线

建立一个可重复启动的 Colab notebook/脚本，记录 Python、PyTorch、CUDA、`qwen-asr`、Transformers、GPU 和 Git revision。用固定的 clean/degraded 小 manifest 跑通 1 条推理、JSONL 输出和 CER 评测。

验收：能从空 runtime 重新安装依赖；环境指纹、predictions 和 metrics 写入 Drive 后可以读回；失败样本写入结果文件而不是中断批处理。Phase 0 不恢复 optimizer，训练 checkpoint 的保存与续训从 Phase 2 开始。

### Phase 1：数据 MVP

生成包含 `sample_id`、`audio`、`text`、`language`、`scenario`、`source_utterance_id` 和可选音频元数据的 JSONL。固定 train/val/test 切分，检查音频存在、字段完整、source 不泄漏和五类场景覆盖。

验收：每个 manifest 可独立校验并输出摘要；固定 seed 重建结果一致；test 不参与训练或偏好构造。

### Phase 2：SFT pilot

只实现 LoRA SFT 的最小闭环。smoke 使用同时包含 clean 和一个 degraded scenario 的小切片，跑 5–20 step；pilot 使用包含五类场景的 train split。checkpoint 必须保存 adapter、optimizer、scheduler、RNG、global step、配置和 manifest hash。

验收：按 `04_training_plan.md` 相对 base 通过 val CER 门禁。smoke 只证明保存和续训，不单独构成发布。

### Phase 3：DPO pilot

从固定的 SFT greedy 结果和参考转写构造 chosen/rejected 对，并完成 `04_training_plan.md` 中的偏好审计。reference 冻结。

验收：DPO 相对 SFT 通过 held-out preference，并同时通过 clean/degraded CER 门禁。

### Phase 4：RL pilot

以 DPO release 为 policy 初始值，冻结 reference，按 `04_training_plan.md` 固定候选数、组内 advantage、`group_pg_mixed_v1` 损失、reward 和两种 KL 的定义。RL 只在 DPO gate 通过后启动。held-out 使用 greedy，不用 rollout 采样参数。

验收：同一 val 上的 greedy reward 相对本 run step 0 达到配置中的最小提升，raw KL、reference checksum 和 clean/degraded CER 门禁同时通过。未满足时保留 checkpoint，不晋级。

### Phase 5：扩大规模与发布

只对通过 pilot 的阶段增加数据和步数。发布时统一 base/SFT/DPO/RL 的 test manifest、`decode.eval` 和 evaluator，保存原始预测、归一化预测、CER 和按 scenario 聚合指标。配置写明分词器时同时保存 WER。

验收：有固定 manifest、配置、seed、运行记录、输出结果、baseline 对比和文档结论，才可以标记阶段完成。

## Colab + Drive 执行合同

- 训练在 Colab 本地磁盘执行，Drive 只作为输入缓存和 checkpoint/结果持久化层。
- 每次启动先执行环境探针并保存 `environment.json`、`resolved_config.yaml` 和 `manifest_sha256.json`。
- checkpoint 先写入本地临时目录，checksum 通过并读回核对后，才写新的 `latest.json`。Drive 改名不作为原子性保证。
- 恢复前校验 manifest hash、模型名、LoRA 配置、`world_size == 1`、eval 解码 hash 和 scheduler；不一致则拒绝恢复。采样 token 逐位相同不是恢复验收。
- 从 Phase 2 起，每个训练阶段至少验证一次“保存 → 断开 runtime → 重新加载 → 继续训练”。
- notebook 只负责编排；可复用逻辑放在 `scripts/`、`train/`、`inference/` 和 `evaluation/`，避免状态藏在单元格中。

## 测试与验收证据

重置后的每个阶段都必须有对应的本地合同测试、Colab preflight、固定 manifest 的原始 predictions、聚合 metrics、gate 和 run report。Phase 0 只验环境/推理产物读回；Phase 2–4 另验 5–20 step smoke、checkpoint 中断恢复、父版本对比和阶段特有 gate。没有这些文件时，进度只能写 `in_progress` 或 `blocked`，不能以旧仓库的结果替代新合同。

## 本次重置后的影响

旧 RL 版本、历史指标和服务器专用入口不再作为当前代码的一部分；它们仍可从 Git 标签 `pre-reset-20261012` 恢复。重新实现后，所有结论必须基于新 manifest、新 evaluator 和新 checkpoint 合同，不沿用旧实验的 gate 或指标。

