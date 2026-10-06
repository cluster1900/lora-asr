# RL v30 可行性整改方案

## 背景

`rl_pilot_feasibility_v29` 已经证明训练链路可用，但没有达到正式质量门禁：4 步后 held-out greedy reward 只提升 `+0.0005`，Robust macro 回退 `+0.000269`。Probe、rollout、checkpoint、2,867 条推理和零方差检查均通过；问题集中在训练信号覆盖与验证集泛化。

v29 的 2,354 条 rollout 中只有 108 条标记为实际训练，`policy_keep_ratio` 约为 `0.21–0.23`。2,867 条验证预测只有 17 条规范化文本发生变化，其中 6 条改善、5 条恶化；`en|noise` 和 `en|recording` 的恶化抵消了 `dropout`、`obstructed` 的小收益。因此下一轮先改样本覆盖，再单独验证参数范围，不能同时改学习率、KL 和 reward。

## 目标与范围

目标是把当前 pilot 从“能跑但更新过少且方向不稳”推进到“有效更新覆盖 hard degraded 场景，Robust 回退显著收窄”。本方案只做 4-step 可行性实验，不进入 Full RL，不导出发布模型，也不修改正式 release 门禁。

## 方案

### v30A：场景均衡采样

新增 `degraded_balanced` 采样策略，保留 clean 不进入优化器，并按 `language × scenario` 统计和固定 seed 生成每个 virtual epoch 的索引。每个 2,000 行 virtual epoch 先按语言各取 1,000 行；每种语言内 7 个 degraded 场景（`noise`、`recording`、`dropout`、`obstructed`、`distortion`、`echo`、`far_field`）**等权**，每个 cell 约 143 行（余数按最大余数法分配）。原方案曾把 `noise`、`recording` 各提到 20%，2026-10-03 复核后撤销：这两个场景正是 v25/v29 回退最多的场景，没有证据支持加倍采样（见 `28_rl_v30_trainer_sync_and_stats.md`）。稀有 cell 允许有放回采样，但单个原始样本最多重复 4 次；达到上限后从同语言其他 degraded cell 确定性回填，并记录实际计数、回填原因和最大重复倍数。不得读取 `rl_val_pool` 或 `validation` 的样本。

v30A 只改变采样策略，保持以下变量与 v29 相同：

- DPO Champion 作为 policy/reference 起点；
- seed `20260722`、学习率 `5e-6`、warmup `2`、β=`0.04`；
- `signed_edits`、`raw_gap`、`local_max_relative=0.35`、`min_improvement=0.02`；
- 199 个 LoRA targets、audio projections 开启；
- 4 steps、每 4 步保存和评测、禁止自动导出 merged model。

这样可以把改进或恶化归因到场景覆盖，而不是参数同时漂移。

### v30B：冻结音频投影的对照组

只有 v30A 的 Robust 回退仍大于 `+0.0001` 时才运行 v30B。v30B 使用完全相同的 manifest、seed、采样和优化器，只把 `train_audio_projections` 改为 `false`，用来判断共享音频投影是否造成少数 hard 样本的全局错误迁移。v30B 不与 v30A 合并结果，也不从 v30A checkpoint 续训，避免混淆因果。

## 执行步骤

0. **前置（2026-10-03 新增，见 `28_rl_v30_trainer_sync_and_stats.md`）**：仓库 `train/train_rl.py` 已改为以 V100 训练器为基线并叠加 v30A 补丁；配置契约测试通过后，才可把仓库同步到服务器。服务器训练器的 1-step smoke 通过前，不启动 v30A。
1. 在本地 `train/rl_sample_strategy.py` 新增并测试 `degraded_balanced` 的确定性索引生成器；`train.sample_strategy_options` 固定 `virtual_epoch_rows=2000`、`max_repeat=4` 和上述等权场景权重，训练日志输出每个 `language|scenario` 的采样计数、最大重复倍数和 manifest hash。
2. 用训练 manifest 与 validation manifest 做 sample-id 交集检查，交集必须为零；确认不读取 held-out 的 gold 或预测。
3. 先跑 sampler 单元测试和 1-step smoke，确认 checkpoint、resume、rollout 审计字段完整。
4. **seed 噪声基线**：用 v29 配置（`sample_strategy=degraded`）再换 2 个 seed 各跑 4 步并评测，得到 3 个 seed 的贪心 held-out 增量与 Robust 增量；极差作为判读噪声下限。若极差本身 `>= 0.001`，说明 4 步量级无法区分配置，v30A 的数值线不再解释为“有效”，只作为工程闭环验证。
5. 在 V100 从 DPO Champion 新开 `rl_pilot_v30a`，完成 4 steps；自动评测 1,698 条 `rl_val_pool` 和 2,867 条 validation。
6. 运行两份门禁：默认 `release` 只作诊断，独立 `pilot_feasibility` 决定该 pilot 是否可行。不得覆盖 DPO Champion 或历史 v29 产物。
7. 用 `evaluation/paired_significance.py` 对 DPO Champion 与 v30A 的 2,867 条 `scored.jsonl` 做配对比较，结果写入 run 目录。
8. 只有 v30A 达到 v30A 验收线但 Robust 仍不稳定时，才运行 v30B；若 v30A 已改善，则不做 v30B。

## 测试与验收

### 工程验收

- `degraded_balanced` 在相同 seed 下生成完全相同的索引；不同 epoch 生成不同但可复现的索引。
- 优化器不含 clean 行；所有 14 个 `language|scenario` degraded cell 都有计数，稀有 cell 的最大重复倍数不超过配置上限。
- train/val/validation sample-id 交集为零；rollout 每组完整，checkpoint 可加载并可继续训练。
- 4-step smoke、完整 rollout、2,867 条推理和 gate 输出均成功；失败样本写入 JSONL，不中断整批。
- `tests/test_rl_config_contract.py` 通过，且训练日志中的 `degraded_balanced_summary=` 与配置一致。

### Pilot 验收线

相对 DPO Champion：

- held-out greedy reward 增量 `>= +0.0010`；
- Robust macro 回退 `<= +0.0001`；
- 至少两个 degraded scenario 改善；
- Clean 增量 `<= +0.005`，有效输出率 `>= 0.99`，空输出率 `= 0`；
- 零方差比例 `<= 0.60`，KL、reward、gradient 均有限；
- `pilot_gate.json` 为 `PILOT_PASSED`，且 `release_eligible=false`；
- **统计条件（新增）**：`paired_significance.py` 在 `degraded` 条件组上的 bootstrap 95% 区间上界 `< 0`（逐样本 `error_rate` 差，负为改善），且 `clean` 条件组均值差 `<= +0.005`。只满足数值线、区间跨 0 的结果记为“与噪声无法区分”，不得写成“有效改善”。

如果 v30A 未达到 reward `+0.0010` 或 Robust `+0.0001`，停止扩大步数，先分析 rollout 和 changed predictions；不通过降低门禁来判定成功。正式 release 仍要求 reward `+0.002` 且 Robust `<=0`。

## 风险与回滚

- 均衡采样可能放大稀有 `zh|noise` 的重复，使用最大重复倍数限制并在审计中记录；若训练 reward 下降或 hard cell 恶化，删除该策略配置即可回退到 v29 的 `degraded`。
- 4 步只消耗 256 个 prompt，分到 14 个 cell 约每格 18 个，参与训练的 rollout 约每格 8 条；均衡采样改变的只是这约 100 条样本的组成，效果可能小于噪声，这是需要 seed 基线来判读的原因。
- 冻结音频投影可能降低 reward 增益；v30B 只作对照，不作为默认方案。
- 4 steps 仍然只验证方向，不代表 Full RL 或发布质量；任何 pilot 通过都不能写入 DPO Champion。

## 可行性检查

开跑前必须确认服务器训练器已经包含策略解析、有限重复采样、独立 gate profile 和 scorer 输出路径。2026-10-03 发现服务器训练器（3,542 行）与仓库旧版（1,957 行）不一致，已把服务器版本纳入仓库并叠加补丁；之后以仓库为唯一来源，从仓库更新服务器，不在 V100 上临时手改脚本后直接训练。

## 2026-10-03 复核结论：v30A 暂停，不建议启动

用服务器真实 manifest 与 v29/v28 产物做了只读复核（未训练、未写服务器）：

1. **“覆盖不足”的前提不成立。** `pilot_rl.jsonl` 的 2,000 条 degraded 本来就接近均衡：12 个 cell 在 121–189 行，只有 `en|noise` 233 行、`zh|noise` 18 行偏离。v29 前 4 步的 256 个 prompt 已覆盖全部 14 个 cell。按训练器真实入口复算，v30A（等权）与 v29 在 4 步内的逐 cell prompt 数只差 0–11 个，`zh|noise` 从 2 变 7；两者前 256 个 prompt 有 32 个是同一批源样本。v30A 的输入与 v29 几乎相同。
2. **真正稀疏的是中文的可学习信号，不是采样。** v29 rollout 中带 `trained` 标记的组：英文 7 个 cell 合计 77/146（53%），中文 7 个 cell 合计 31/110（28%），`zh|far_field` 2/14、`zh|dropout` 3/21、`zh|obstructed` 3/14。DPO 模型在中文上已接近正确，组内很少有比贪心好 `0.02` 的候选。均衡采样不会增加这类样本的赢家。
3. **v29、v28 与 DPO 在验证集上统计不可区分。** 用 `evaluation/paired_significance.py` 对 2,867 条 `scored.jsonl` 做配对比较（候选减 DPO，负为改善）：
   - v29 Step 4：degraded 867 条均值 `+0.000284`，95% 区间 `[−0.000311, +0.001105]`，预测变化 14 条，变好/变差 4/5，符号检验 p=1.0；clean 均值 `−0.000122`，区间 `[−0.000327, 0]`。
   - v28 Step 7：degraded 均值 `+0.000433`，区间 `[−0.000516, +0.001483]`，变化 28 条，变好/变差 10/9，p=1.0；clean 均值 `−0.000167`。
   - 区间半宽约 `0.0007–0.001`。要在 degraded 上判定改善，需要数十条样本同向变好；4 步 RL 只改变 14–28 条预测，且方向各半。
4. **v28 结果此前未记入进度。** `rl_pilot_v28` Step 7 `STOPPED_KL`，`gate_step_7.json` FAILED：贪心 `+0.0004`，Robust `+0.000007`（+1 次编辑），clean `−0.000109`，`switch_decision.json` 动作 `stop`。
5. **执行记录缺口。** v29 的启动脚本与 release 门禁只在服务器 `/tmp`（`/tmp/rl_pilot_feasibility_v29.sh`、`/tmp/rl_pilot_feasibility_v29_release_gate.json`）；脚本已归档为 `scripts/run_rl_pilot_feasibility_v29.sh`，release 门禁 JSON 仍只在 `/tmp`，应复制到 run 目录。进度文档写的“V100 正被其它训练占用”已不成立：2026-10-03 22:25 CST 四张卡均为 4 MiB、0% 占用，没有训练或推理进程。
6. **单轮成本。** v29：probe 约 6 分钟，4 步训练约 44 分钟，2,867 条推理与评测约 23 分钟，共约 73 分钟。

**决定：** v30A 预期结果与 v29 无法区分，跑一轮只消耗约 73 分钟 GPU 和一次文档往返，不能回答任何问题，暂停启动。v30B 依赖 v30A，一并暂停。代码和配置保留，契约测试继续有效。下一步不再以“改一个超参再跑 4 步”的方式推进 RL，由项目负责人在以下两条路中选择：

- **A. 收口 RL 阶段**：记录“在当前数据规模（2,000 条 degraded）与 KL 预算（`5e-4`）下，RL 对 DPO Champion 没有可测收益”，正式发布底座保持 DPO Champion，进入 bench_test 终评与 Mega-ASR 同 evaluator 对比。
- **B. 换实验量级再做 RL**：只有能让验证集上数十条 degraded 样本同向变化时才值得跑，例如扩大 degraded 训练池、按语言分开评估中文是否还有可学空间、或允许更大的 KL 预算并单独评估 clean 回退。每一项都必须先写设计文档，并以配对区间而不是点估计验收。

**2026-10-03 项目负责人选择 B。** 设计、实现与测试见 `29_rl_v31_scale_design.md`；v30A/v30B 不再启动。
