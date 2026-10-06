# 29 RL v31：换实验量级（方案 B）设计

状态：方案修复已完成，本地 `.venv` `pytest -q` 为 `337 passed, 9 skipped, 39 subtests passed`，Python 编译与 shell 语法检查通过；此前 V100 unittest `313 OK`、服务器 preflight 通过（判定 B 清单已生成 2,599 行）。v31 已完成一次同预算运行，但因训练池是非严格子集而只覆盖了目标量级的约四分之一，且 Step 32 的单次负增益规则提前停止；本次关闭数据契约、可恢复执行、场景兼容和磁盘容量四个启动阻断点，不启动训练。

## 背景

v10–v29 共 20 个 checkpoint 的贪心 held-out 增益都在 `−0.0006 ~ +0.0005`，配对比较与 DPO Champion 统计不可区分（见 `27_rl_v30_feasibility_plan.md` 末节、`28_rl_v30_trainer_sync_and_stats.md`）。2026-10-03 项目负责人选择方案 B：不再“改一个超参跑 4 步”，而是换实验量级，并用配对置信区间验收。

本设计先用已有产物回答“量级应该放大到多少、放大哪一项”，再给出一次可以回答问题的实验。

## 证据（只读分析，2026-10-03，服务器 CPU）

| 观察 | 数字 | 含义 |
|---|---|---|
| 每次 pilot 训练的 prompt 数 | 4–7 步 × 64 = 256–448 | 远低于常规 GRPO 的数百步；4 步只改变 14–28 条验证预测 |
| 停止原因 | v27/v28 都是 `STOPPED_KL`，`raw_kl` 上限 `5e-4` 在 Step 7 被触发 | KL 上限而不是数据决定了训练长度 |
| `raw_kl` 增长（lr `5e-6`，v28） | Step 4 `6.7e-5` → 5 `9.7e-5` → 6 `2.1e-4` → 7 `6.2e-4` | 每步约 ×2–3；`beta=0.04` 时 KL 罚项（≈`2.5e-5`）相对策略损失（≈`0.5`）可忽略，上限只起“截停”作用 |
| 采样候选的上限增益（v29 step_0 rollout，组内最优减贪心） | 英文 `0.117`（64% 组有 ≥0.02 的赢家）；中文 `0.026`（21%） | 英文有大量可学信号；中文的采样候选几乎和贪心一样 |
| probe best-of-11 平均上限增益 | `0.044`（128 prompt） | 策略分布内确有更好的转写 |
| DPO Champion 验证集错误分布 | 英文 degraded 59% 行有错，中文 36%；中文前 10% 的行占 65% 错误量（如“不求有功”→“嗯”） | 中文错误集中在短句整句失败，采样难以触达 |
| RL 训练池 | 当前合同已扩大到 160,000 条，但现有 `rl_train_pool` 只有 4,165 条 degraded；旧 v31 Step 32 只消费 2,048 个 prompt | 现有 manifest 是 `NON_STRICT_SUBSET`，正式运行不能继续按完整量级解释；需要大规模补齐独立 Voices-in-the-Wild 数据 |
| 判定集 | `validation` 只有 867 条 degraded；`dpo_val_pool` 另有 2,599 条 degraded，只用于 DPO preference 验证（66 对），从未进入 DPO 训练或 RL | 合并后 3,466 条 degraded，配对区间半宽约从 `0.001` 降到 `0.0005` |
| 单步耗时 | 稳态约 225 s/步；1,698 条 held-out 贪心评估约 14 min | 2,560 步约 160 h（不含最终 gate） |
| VITW 已暂存音频 | 48,580 条全部已分配到 7 个角色 | 再扩大训练池需要新下载（B2） |

## 范围

做（B1，本轮）：

1. 训练池从 `pilot_rl`（2,000 degraded）扩大到达到数据合同的完整 `rl_train_pool`（目标 160,000 degraded，是原合同的 10 倍），一轮使用约 2,560 步 × 64 prompt；正式启动器拒绝低于最低行数的非严格子集。
2. 新增 `scale` 停止档：去掉为 4–12 步短 pilot 设计的 Step 4/8/12 搜索/迁移截停；`raw_kl` 上限 `5e-4 → 2e-2`；Step 640 起只有连续两次 held-out 贪心增益低于 0 才触发无效截停，单次微小负增益不再截断完整量级训练。正式启动器按 320 步 checkpoint 分块，并可从最后一份完整检查点续跑。
3. 判定集扩大到 `validation` 全量 + `dpo_val_pool` degraded（共 3,466 degraded + 2,000 clean），用配对 bootstrap 区间验收，并分语言报告。
4. 正式 release 门禁（`verify_gate.py --gate-profile release`）**不放宽**，作为附加条件同时要求。

不做：

- 不改奖励、优势、采样温度、LoRA 目标和 gate。10x 量级修订追加一个稳定性补丁：学习率 `5e-6 → 2e-6`、warmup `2 → 64`、KL `beta 0.04 → 0.08`，用于控制已观测的早期 KL 快速上升；这仍属于 v31，不新增方案编号。
- 不下载新数据（原 B1 设计；本次 10x 修订必须先扩充并重新 staging 数据）。
- 不在判定集上挑 checkpoint：只评最后一个 checkpoint（到达 horizon 或停止步）。
- 不改 DPO Champion、不覆盖任何历史 run 目录。

## 设计

### 数据

| 角色 | 文件 | 行数 | 用途 |
|---|---|---|---|
| 训练 | `/data/mega-asr/manifests/rl_train_pool.jsonl`，策略 `degraded` | 目标 162,000（160,000 degraded + 2,000 clean 审计行） | rollout 与更新 |
| held-out（过程监控） | `rl_val_pool.jsonl` | 1,698 | Step 0/320/640/960/.../2,560 贪心奖励；停止规则与 release 门禁 |
| 判定 A | `validation.jsonl` | 2,867 | release 门禁 + 配对比较 |
| 判定 B | `rl_verdict_extra_degraded.jsonl`（由 `scripts/build_rl_verdict_manifest.py` 从 `dpo_val_pool` 取 degraded 生成） | 2,599 | 配对比较 |

`build_rl_verdict_manifest.py` 校验：判定 B 与 `sft_train`、`dpo_train_pool`、`rl_train_pool`、`rl_val_pool`、`validation`、`bench_test` 在 `sample_id`、`source_utterance_id`、`audio_sha256` 上零交集；所有参与隔离检查的行都必须填写且不能重复这三种身份键；只保留 `condition_group == degraded`；写出 `<output>.COMPLETE.json`（行数、sha256、来源 sha256、按语言/场景计数）。任何交集或身份字段缺失即退出非零。

`rl_train_pool` 名称不含 `pilot`，`DATASET_COMPLETE.json` 为 `NON_STRICT_SUBSET`，启动需显式 `--allow-subset`（与数据文档一致：本项目所有角色都是子集）。

### 训练预算与配置

配置 `configs/train/qwen3_asr_rl_v31_scale.yaml`，相对 v28 只改：

| 项 | v28 | v31 |
|---|---|---|
| `train.max_steps` / `save_steps` / `eval_steps` | 24 / 4 / 4 | 2,560 / 320 / 320 |
| `train.apply_learning_rate_on_resume` | true（从 v27 Step 4 恢复） | false（从 DPO Champion 新开） |
| `train.learning_rate` / `warmup_steps` / `grpo.kl_regularization.beta` | 旧 pilot 配置 | `2e-6` / `64` / `0.08` |
| `train.stop_profile` | （默认 `pilot`） | `scale` |
| 停止阈值副本 | pilot 阈值 | 只写 scale 档的键（见下）；pilot 专用键出现即被拒绝 |

### 停止档 `scale`

`train_rl.py` 把阈值表改为 `STOP_PROFILES = {"pilot": ..., "scale": ...}`，`STOP_CONTRACT` 保留为 `pilot` 档的别名，pilot 行为逐位不变。`decide_rl_stop(..., profile=)`、`plan_driver_action(..., stop_profile=)`、`driver_action_from_records(..., stop_profile=)` 默认 `pilot`。

| 规则 | pilot | scale |
|---|---|---|
| 零方差崩塌 `FAILED_ZERO_VARIANCE` | 有 | 有（同） |
| 贪心增益 < `−0.005` `STOPPED_REWARD_DROP` | 有 | 有（同） |
| `raw_kl` 上限 `STOPPED_KL` | `5e-4` | `2e-2` |
| 无赢家 `BLOCKED_NO_WINNERS` | 有 | 有（同） |
| Step 4/8/12 搜索质量与迁移截停 | 有 | **无** |
| 无效截停 `BLOCKED_TRANSFER` | — | Step ≥ 640、贪心增益 < `0.0` 且连续两次评估均未达线 |
| `STOPPED_ROBUST` | 有 | 有（同） |
| probe 门槛 `probe_mass_min` | 17.0 | 17.0 |

`check_config_contract` 读取 `train.stop_profile`（缺省 `pilot`）：未知档名、YAML 副本与该档不一致、或出现**其他档专有**的阈值键都会报错。

KL 上限 `2e-2` 的理由：`5e-4` 没有经验依据，它只是截停了训练；真实的风险是 clean 回退和输出退化，由 held-out 奖励下降规则（每 320 步）和最终 release 门禁直接测量。若 `raw_kl` 在 Step 640 前就超过 `2e-2`，结论是“该目标发散快于改进”，同样回答了问题。

### 判定与统计验收

`evaluation/paired_significance.py` 扩展：

- `--baseline` / `--candidate` 接受多个文件，按 `sample_id` 合并（跨文件重复即报错）；
- 新增 `by_language_condition`（如 `en|degraded`）分组；
- `--gate`：输出 `gate` 段，`PASSED` 当且仅当 `degraded` 的 95% 区间上界 < 0（显著改善），且 `clean` 的区间上界 ≤ `--clean-margin`（默认 `0.002`，非劣）。分语言结果只报告，不作为通过条件。

B1 通过 = 以下全部满足：

1. `paired_verdict_step_<N>.json` 的 `gate.status == PASSED`（判定 A+B，DPO Champion 为基线）；
2. `gate_step_<N>.json`（release 档，未放宽）`PASSED`；
3. 训练状态不是 `FAILED_ZERO_VARIANCE` / `STOPPED_REWARD_DROP`。

最小可检出效应：degraded 合并 3,466 条，按 v28 的逐样本差标准差约 `0.015` 估算，区间半宽约 `0.0005`，即 degraded 平均错误率需下降约 `0.0005` 以上（相对 DPO 的 `0.13` 约 0.4%）。

### 执行与产物

启动器 `scripts/run_rl_scale_v31.sh`（`RL_RUN_DIR` 只能位于 `/data/mega-asr/runs/rl_scale_v31*`；默认拒绝覆盖已存在的 run 目录；只有显式 `RL_RESUME=1 RL_RUN_DIR=<run>` 才允许从完整 checkpoint 继续；通过 `nvidia-smi` 拒绝在 GPU 有任何 CUDA compute 进程时启动；无法查询 GPU 时也拒绝启动）：

1. 生成判定 B manifest（已存在且 sha 一致则跳过）；
2. probe（128 prompt，`rl_train_pool`）→ 必须 `GO_GRPO`；
3. 训练 2,560 步（按 checkpoint 可恢复）；正式模式要求 RL degraded 行数达到 160,000。由于总角色门禁仍记录为 `NON_STRICT_SUBSET`，启动器保留 `--allow-subset` 作为总门禁兼容参数，但不会降低 RL 的 160,000/640/20%/零泄漏硬门禁；
4. 用 `score_rl_pilot_checkpoint.sh`（`RL_GATE_PROFILE=release`）评分最后一步 → `gate_step_<N>.json`；
5. DPO Champion 在判定 B 上推理一次，写入新目录 `/data/mega-asr/runs/eval_dpo_champion_verdict_extra/`（已存在则复用）；
6. 候选在判定 B 上推理 → `predictions_step_<N>_extra(_eval)`；
7. 配对比较 → `paired_verdict_step_<N>.json`；
8. 汇总 → `scale_verdict.json`（release 门禁状态、配对门禁状态、训练停止状态、分语言区间）。

run 目录：`/data/mega-asr/runs/rl_scale_v31/`。

### 成本

probe 6 min + Step 0 评估 14 min + 2,560 步 × 225 s ≈ 160 h + 8 次过程评估约 112 min + 判定 A 23 min + 判定 B 两次约 42 min ≈ **164 h（约 7 天）**，必须拆成可恢复的异步块运行。对比：v10–v29 共约 20 轮 × 73 min ≈ 24 h，没有回答问题。

## 2026-10-05：v31 失败归因与当前方案内修复

本节覆盖一次完整的只读复盘，不代表新增方案编号。远端运行 `/data/mega-asr/runs/rl_scale_v31_20261005T093505` 的事实是：配置目标 64 步，但训练池通过 `--allow-subset` 接受了 4,165 条 degraded；Step 32 已消费 `32 × 64 = 2,048` 个 prompt，约占数据合同 16,000 条的 12.8%，随后因 `greedy_gain = −0.0002` 写入 `BLOCKED_TRANSFER`。`raw_kl=0.010715 < 0.02`、`zero_variance_ratio=0.3125`、累计 winners `887`，因此不是 KL 或零方差先把训练打断。

训练池不足不是抽象风险，而是 manifest 构建配额的结果：当前 staged robust 数据在 90/10 身份分区后，SFT 与 DPO 先分走了大部分训练组，RL 只剩 4,165 条；`rl_train_pool_COMPLETE.json` 为非严格子集。正式启动器现在必须检查 degraded 行数和语言×场景最小覆盖，低于合同直接退出，不能再把子集当成量级实验。

有效更新也不是完全没有：2,048 个 group 中 887 个真正反向更新（43.3%），728 个 identical、433 个 no-improvement；降低候选改善阈值到 `0.01` 的离线重算只增加 4 个更新组，说明 `min_improvement=0.02` 不是当前主瓶颈。样本池的语言×场景分布却明显不均，当前 `zh|noise` 只有 30 条、`en|noise` 有 478 条，扩充独立数据并保持身份隔离是必要条件。

当时的修复保持 v31 的 DPO→RL、reward、LoRA、学习率和 release gate 不变，只做三件事；该段是历史记录，当前有效执行合同以文末的 10x 修复为准：

1. 将正式训练量与当时的 16,000 条 degraded 合同对齐为 256 步，并拒绝低于最低样本量的 `NON_STRICT_SUBSET`；
2. 将 scale 无效截停改为 Step 64 起连续两次负增益才触发，保留 `STOPPED_REWARD_DROP`、`STOPPED_KL`、zero-variance 和最终 release/paired/tail gate；
3. 训练日志继续记录 update/identical/no-improvement、winners、KL 和 held-out reward，扩充后的 manifest 必须通过三种身份键零交集检查。

这不能预先保证 RL 一定超过 DPO。它消除了“样本不足 + 单次微小波动提前停止”这两个已证实的阻断因素；如果完整量级仍在 Step 64/96/128 的同口径评估中没有增益，才可以把问题归因到 reward 与策略更新本身，而不是训练预算。

## 2026-10-05：RL 训练池再扩大 10 倍

本次继续沿用 v31，不新增方案编号。用户要求在当前方案上把样本再扩大 10 倍，因此把 RL optimizer 的 degraded 合同从 16,000 提到 160,000；SFT、DPO、验证集、reward、LoRA、采样规则和 release gate 保持不变，学习率只按下述稳定性补丁调整，确保扩大样本后仍能和 DPO 做同口径对比。

基于旧 run 的 `raw_kl` 早期上升和 `policy_keep_ratio` 平均 `0.216`，本次在同一 v31 内增加稳定性补丁：学习率降到 `2e-6`，warmup 提到 64 步，KL `beta` 提到 `0.08`。这三项只降低更新步幅、延缓早期漂移，不改变 reward、候选阈值、采样分布或 gate；如果 held-out 仍无正增益，结果仍会按原 gate 判失败。

对应调整如下：

1. 正式 manifest 至少 160,000 条 degraded，16 个 language×scenario cell（含 `mixed`）各至少 640 条；不能用 virtual epoch 重复或从 SFT/DPO 回填。
2. `max_steps` 从 256 提到 2,560，保持 global batch 64，约覆盖 163,840 个 prompt；保存和完整 held-out 评估从每 32 步改为每 320 步。
3. futility 起点从 Step 64 同步到 Step 640，仍要求连续两次负增益；KL、zero-variance、严重 reward drop 和最终 gate 不放宽。
4. 当前远端 4,165 条 degraded 距离新合同还差 155,835 条，且 `zh|noise` 只有 30 条，距离 cell 最低线还差 610 条。必须先重新 staging、分配和生成完整 manifest，再做 CPU/preflight；本轮不训练。

按旧的约 225 秒/step 估算，完整 2,560 步约需 160 小时，连同评估和收尾约 7 天。这个成本是数据量扩大 10 倍的直接结果，后续必须依靠 checkpoint 异步续跑，不能把短 smoke 当作正式结果。

## 2026-10-05：10x 样本的效果预估（只读）

扩大样本对“统计稳定性”和“覆盖不足”有明确帮助，但不能据此保证 RL 超过 DPO。已有 v31 证据如下：

- 旧 run 的更新率为 `887 / 2,048 = 43.3%`。若更新率和奖励分布保持不变，2,560 步 × 64 prompt 约有 163,840 个 prompt，预计约 70,960 个更新组；更新率估计的标准误会从约 `0.01095` 降到 `0.00122`，因此中文稀疏 cell 和场景差异会更容易被测出来。
- 但旧 run 的 Step 0→32 held-out reward 是 `0.8773 → 0.8771`，增益 `−0.0002`；相对 DPO 的 paired degraded mean delta 是 `+0.008069`（候选错误率上升），95% 区间上界 `+0.024239`，尚未出现正向 dose response。
- `raw_kl` 在 Step 32 已为 `0.010715`，线性外推现有 Step 1–32 日志，达到 `0.02` 的粗略位置约在 Step 58–72。该外推受噪声影响，不能当作精确预测，但它说明当前 10x 配置可能先触发 KL 停止，再到达 Step 640 的 futility 检查点。

因此当前判断是：**10x 样本大概率会改善统计可靠性和场景覆盖，但实际 WER/reward 改善的把握仍然偏低到中等；如果不先验证 KL 剂量，不能声称会改善。** 10x 训练只有在新增 manifest 通过门禁、并且小规模 CPU/离线 rollout 显示 KL 不会在早期快速上升时，才值得占用 V100。即使 KL 风险解除，仍需用固定 DPO 对照和 release gate 判断是否真正改善。

## 2026-10-05：10x 修订的对抗性运行审计

在占用 V100 之前，当前方案还有以下阻断风险：

| 级别 | 风险 | 触发方式 | 必须处理的门槛 |
|---|---|---|---|
| P0 | 数据仍不足 | 当前远端只有 4,165 条 degraded，距离 160,000 还差 155,835 条；`zh|noise` 只有 30 条 | 重新 staging、分配并生成完整角色 manifest；不能用 subset 或 virtual epoch 冒充正式数据 |
| P0 | 7 天任务无法自动续跑 | 启动器正式模式不传 `--max-steps`，只启动一个 2,560 步 `torchrun`；run 目录已存在时直接拒绝，进程中断后不会自动从最近 checkpoint 续跑 | 已修复：正式模式固定按 320 步 chunk 运行；异常退出后用 `RL_RESUME=1 RL_RUN_DIR=<run>`，只从 `pipeline_state.json` 的最后完整 checkpoint 续跑 |
| P1 | scenario 契约不一致 | 数据配置允许 `mixed`，但正式 validator 只接受 7 个 cell 场景；新增数据出现 `mixed` 会在 GPU 前直接 `REFUSE` | 已修复：正式 validator 接受 16-cell（含 `mixed`）合同；pilot 的七场景均衡 sampler 保持不变，正式 v31 仍使用 `degraded` 全量策略 |
| P1 | KL 仍可能先于收益停止 | 稳定性补丁尚未有实测证据；旧日志的 KL 外推在 Step 58–72 已接近上限，Step 640 futility 可能根本到不了 | 先做不占 GPU 的配置/数据检查，并在正式日志中要求 Step 320/640 的 KL 与 held-out 同时可见；`STOPPED_KL` 仍按原 gate 记失败 |
| P1 | smoke 不能证明正式数据可用 | `RL_SCALE_SMOKE=1` 把最低行数和 cell 门槛降到 1，当前 4,165 条清单也能通过 smoke | smoke 只能验证 checkpoint 闭环；正式数据门禁必须单独通过，不能把 smoke 通过当作正式启动许可 |
| P1 | rollout 产物会放大 | 旧实际 2,048 prompt 约 464 MB/run；新 2,560 步约 163,840 prompt，rollout、预测和合并审计可能达到数十 GB，并增加 merge 内存和耗时 | 已修复：启动器在 GPU/probe 前检查 run 所在文件系统至少 100 GiB 可用；完整文件仍必须通过 merge/audit |
| P2 | 数据仍可能偏斜 | 只要求每个 cell 至少 640 条，但 optimizer 仍使用 `sample_strategy=degraded` 全量打散，不保证 16 个 cell 的训练权重接近 | 已修复：正式 manifest 增加单 cell ≤20% 硬门禁；仍保留按 cell 分布写入 `source.json` 的审计 |
| P2 | 稳定性补丁可能过度保守 | `lr` 降到 `2e-6`、KL beta 加倍后可能减少回退，也可能让 held-out 增益长期接近 0 | 不放宽 gate；用 Step 320/640 的更新率、KL、policy_keep_ratio 和 reward 同时判断是否还有有效学习信号 |

这份审计的结论是：**当前方案只有在数据达到 160,000/640/20%/零泄漏契约后才允许占用 V100；可恢复 chunk、`mixed` 兼容和磁盘门禁已经写入启动器。KL 剂量仍由 Step 320/640 日志与原有安全门禁验证，不能在无实测时宣称成功。**

## 2026-10-05：对抗性检查后的修复方案与成功目标

本节是本轮执行合同，仍属于 v31，不新增方案。目标不是把某个门槛改到“必然通过”，而是保证训练在数据、进程中断和资源边界上能够完整执行，并让“RL 是否真正超过 DPO”只由固定门禁回答。

### 修复项

1. **数据门禁固定化**：正式启动器硬编码 degraded 总量 `160,000`、每个 `en|zh × 8` 场景 cell 至少 `640`、单 cell 占比不超过 `20%`，并在 GPU 前对 RL train 与 validation、SFT、DPO、WER、bench 六个角色的 `sample_id`、`source_utterance_id`、`audio_sha256` 做完整性、重复和零交集检查。validator 接受 `mixed`，并把 clean 行只作为审计行，不计入优化器。`sample_strategy=degraded` 保持当前方案，不引入新的训练策略。
2. **可恢复执行**：正式训练固定以 `320` 步为一个进程块，块结束必须写入 `pipeline_state.json` 和 `checkpoints/step_<N>`。进程异常退出时，设置 `RL_RESUME=1 RL_RUN_DIR=<原目录>`；启动器校验已有 `source.json`、探针和检查点，从最后 `CHUNK_DONE` 继续，不重跑已经完成的块。设计内的 `BLOCKED_*`、`STOPPED_*` 和 `FAILED_*` 状态不会被强行续训。
3. **资源门禁**：在探针和 GPU 检查前，run 所在文件系统可用空间必须至少 `100 GiB`；训练过程不删除 rollout/checkpoint 审计产物。
4. **稳定性与停止**：继续使用 `lr=2e-6`、warmup `64`、KL beta `0.08`；保留 `raw_kl=0.02`、严重 reward drop、zero-variance、release、paired 和 tail gate。Step 320/640 日志必须同时记录 raw KL、held-out reward、update ratio、policy keep ratio 和 rollout 审计，不能只看单一 reward。

### 成功目标与失败定义

- **执行成功**：正式 manifest 通过 160,000/640/20%/零泄漏检查；probe 为 `GO_GRPO`；训练到 horizon 2,560 或由安全停止明确结束；每个已完成块都有可加载 checkpoint；中断后能从最近块恢复；没有因目录覆盖、磁盘不足、`mixed` 行或单次微小负增益而异常退出。
- **训练成功**：最后评估步骤同时满足 release gate、paired degraded 显著改善、clean 非劣和 tail gate；训练状态不是 `FAILED_ZERO_VARIANCE` 或 `STOPPED_REWARD_DROP`。这才算 RL 相对 DPO 的成功，不能用 smoke、probe 或单个 held-out 点代替。
- **可解释失败**：若 raw KL 越过 `0.02`、严重 reward drop、zero-variance、数据门禁、checkpoint 恢复或任一最终 gate 失败，保留完整日志并标记 FAILED；不删除 run、不把失败结果发布为新底座。

## 测试

CPU 单测（本地与服务器）：

- `tests/test_rl_config_contract.py`：pilot 行为不变；scale 档 KL 用 `2e-2`；scale 档 Step 12 低增益不触发 `BLOCKED_*`；Step 640 的第一次负增益不触发、Step 960 的连续第二次才触发 `BLOCKED_TRANSFER`；未知档名、跨档阈值键、改动的 scale 阈值被拒绝；v31 配置通过契约；`driver_action_from_records` 透传档名。
- `tests/test_paired_significance.py`：多文件合并、跨文件重复报错、`by_language_condition`、门禁 PASSED/FAILED 条件、CLI `--gate`。
- `tests/test_build_rl_verdict_manifest.py`：只取 degraded；与六个排除角色任一键交集、身份字段缺失或身份键重复即失败；COMPLETE 文件字段。
- 目录 README 合同测试。

服务器 preflight（不占 GPU）：`train_rl.py --help`；v31 配置过契约；正式 RL manifest degraded 行数至少 160,000、16 个 language×scenario cell（含 `mixed`）各至少 640 且单 cell 不超过 20%；manifest 可保留 clean 审计行，但 `sample_strategy=degraded` 的 optimizer epoch 必须是 clean 0；生成判定 B manifest 并核对 2,599 行与零交集；`bash -n` 启动器。

GPU smoke（需确认后执行）：同一启动器加 `RL_SCALE_SMOKE=1` 时 run 目录改为 `rl_scale_v31_smoke`、horizon 用 `--max-steps 5 --save-steps 5 --eval-steps 5`、跳过判定与配对，只验证 probe→训练→checkpoint 保存，约 35 min。

## 验收

- 本设计：上述 CPU 测试与服务器 preflight 通过；文档与目录 README 同步。
- B1 实验：按“判定与统计验收”三条全部满足才算 RL 阶段通过；否则记录配对区间与分语言结果，按 B2 规则决定扩大数据或收口。

## 风险与回滚

| 风险 | 处理 |
|---|---|
| KL 加速增长，提前 `STOPPED_KL` | 记录停止步与该步 held-out 增益；这本身是结论；不在本轮调学习率 |
| clean 回退 | release 门禁 + 配对 clean 非劣（`+0.002`）双重约束 |
| 只对英文有效 | 分语言区间报告；degraded 合并显著即通过，中文结果写入结论，供 B2/中文专项决策 |
| 判定 B 曾用于 DPO preference 验证 | 只用于 66 对 preference accuracy，未训练；若有偏差也偏向 DPO 基线，对 RL 更保守 |
| 约 7 天占满 4 卡 | 启动器检测 GPU 进程；可随时 kill，checkpoint 每 320 步保存 |
| 回滚 | 新档默认 `pilot`，旧配置与历史行为不变；scale 档只由 v31 配置启用 |

## 影响

- 数据：只新增一个派生 manifest（判定 B），不修改已有角色文件。
- 训练：pilot 档逐位不变；所有旧配置照旧通过契约。
- 评测：`paired_significance.py` 输出新增字段，旧字段不变。
- 发布：B1 通过前，发布底座仍是 DPO Champion。

## 2026-10-04：当前 v31 的解码合同修复（覆盖上述历史启动细节）

背景：官方推理默认 512 个新 token，旧训练/held-out 固定 128；额外判定集中出现高 WER 的重复长输出。508 是编辑数，不能等同生成 token 数。该差异的因果贡献待验证；本次不预先承诺 RL 能通过质量验收。

范围：以后继续在此方案、同一配置和启动器中修改，不新增方案编号。保持 signed_edits、raw_gap、reference candidate、学习率、KL 与原 release 通过线；修改解码预算与实验审计，不修改奖励权重，不裁剪预测文本或移除异常样本。

设计与接口：

- `decoding: {max_new_tokens: 512}` 为当前 v31 唯一预算来源；`inference/decoding.py` 解析正整数预算并配置官方 model 的贪心设置（do_sample=false、num_beams=1、num_return_sequences=1）。训练采样继续显式覆盖 do_sample 和采样参数。未声明 decoding 的历史训练保持 128。
- 单卡/多卡推理新增 `--max-new-tokens`，缺省沿用官方 512；输出每行增加 `decoding` 合同。断点续推拒绝混用旧合同/不同模型或 adapter。RL checkpoint 同时保存 resolved_config 与 decoding，恢复时必须一致。
- 训练、held-out、probe 使用同一模型预算；held-out 日志记录预算。既有 reward 对长序列仍可能饱和，先保留这一变量以便判断解码修复效果；异常长度和错误率完整记录，不仅看平均 reward。
- 当前运行在独立 `rl_scale_v31_<时间戳>` 目录，`RL_RUN_DIR` 可显式指定但必须位于 `/data/mega-asr/runs/rl_scale_v31*` 且不存在。这是同一方案的运行产物，不是新方案。
- 先以 `RL_SCALE_SMOKE=1` 训练到 Step 5，校验 adapter/optimizer/scheduler/RNG/config 完整及训练状态；通过后用同一配置从 DPO Champion 新开正式 run 到 2,560，不续训旧的 v31 Step 32。确认日志、参数预算和 rollout 审计后才保留正式结果。任何进程或 smoke 验证失败都停止，不自动重开。
- 当前 run 内重新推理 base validation、DPO validation 和 DPO extra（全部 512、同 dtype/greedy），不复用缺乏 decoding 元数据的旧预测。最终 release 对比和 paired 对比均使用这些对照。
- `paired_significance.py --tail-gate --require-max-new-tokens 512` 验证每行解码合同、有限非负错误率，并报告全量/逐语言场景的 repeat_like、too_long、hallucination_like 计数和最大错误率。若任一候选 `error_rate >= 2` 且 too_long/hallucination_like 为真、同时错误率高于同一 DPO 样本，则尾部门禁 FAILED；这是额外的防回退条件，不取代宏平均与置信区间。

测试与验收：CPU 测试覆盖预算解析、官方构造参数、rollout/held-out 预算传递、旧合同恢复拒绝、推理续跑合同拒绝、多卡参数传递、非有限错误率拒绝，以及即使均值改善仍拦截单条严重异常；目录 README 合同测试、完整 pytest、shell 语法检查。V100 上先跑 1 条 clean + 1 条 degraded 推理并计算 WER/CER，随后异步执行 5+1 步 checkpoint 恢复 smoke；只有完整成功才继续同一轮训练。release、paired 和 tail 全部通过才可标记 RL 验收成功。

影响：旧实验文件不覆盖，DPO Champion 不改；512 rollout 可能增加耗时与显存，smoke 用于实测。阈值不放宽。最终以新 Step 0 作为同预算 held-out 基线，不能和旧 128-token Step 0 直接计算增益。

## 2026-10-05：分块恢复不能重放未提交步

背景：10x 运行按 320 步拆进程，预计约 7 天，主机维护或进程被杀掉是正常路径。检查点只在块结束时提交，但 `loss_log.jsonl` 和 `rollouts_rank_*.jsonl` 每个 optimizer step 都追加。从上一份完整检查点再启动时，训练器会把已经写过的步再跑一遍。`load_run_dose` 会把重复的 `reward_mass_in_step` 加两次；rollout 的 `group_id` 重复后，审计要求的组行数对不上，而合并失败只被记成警告。另外，`pipeline_state.json` 原先写在四卡 RNG 落盘之前，崩溃窗口里的状态会指向一份不完整检查点，启动器因此拒绝恢复。

范围：仍是 v31，不改奖励、采样、学习率、KL、horizon 或门禁。只修恢复的提交点。不删除已提交的检查点，不重跑已经完整结束的块。

设计：

- 一次 optimizer step 先把 rollout 写成 `policy_checkpoint=step_{N-1}`，再把日志写成 `global_step=N`，最后才保存 `checkpoints/step_N`。恢复到已提交步 `N` 时，只保留 `global_step <= N` 的损失行，以及 `policy_checkpoint` 步号 `< N` 的 rollout 行。没有 `policy_checkpoint` 的历史行保留。裁剪、`pipeline_state.json` 和 `training_state.json` 都先写临时文件再替换，避免半截 JSON 被当成完整检查点。
- `pipeline_state.json` 改到 adapter、optimizer、scheduler 和四卡 RNG 都落盘之后再写。启动器调用 `select_resume_step`：设计内的 `BLOCKED_*` / `STOPPED_*` / `FAILED_*` / `COMPLETED` / `CANDIDATE` 只有在该步检查点完整时才停止续训；如果 pipeline 原子替换前进程退出，恢复器也读取完整检查点 `training_state.json` 中的状态，避免把设计内停止误当作 `CHUNK_DONE`，并原子补写缺失或过期的 pipeline 状态后再进入最终评分。训练自然到达 2,560 步时把 `CHUNK_DONE` 收口为 `COMPLETED`，否则从最后一份完整检查点继续。完整指 adapter 目录、`optimizer.pt`、`scheduler.pt`、`training_state.json` 和 `rng_state_rank_{0..3}.pt` 都在。
- 恢复时核对 manifest、validation、WER 清单的 sha256，以及 world size、模型 revision、seed、gradient accumulation、scheduler 名称和 LoRA target hash。旧检查点没有的新字段不追溯拒绝；manifest sha、world size 和当前配置需要的 `scheduler.pt` 缺失则直接拒绝。scale 档的 futility 连续次数以裁剪后的损失日志重算，不单独信任内存里的旧计数。

测试：`truncate_resume_artifacts` 去掉未提交步后，剂量不再翻倍，剩余 rollout 能通过组审计。`assert_resume_checkpoint` 拒绝 manifest 不一致和缺 scheduler。`select_resume_step` 在 pipeline 超前且 RNG 缺失时回到上一份完整检查点，在更新的完整检查点存在时不重跑已提交块，在停止状态已完整提交时不再续训。`driver_action_from_records` 对 Step 640 的第一次负增益继续，对 Step 960 的连续第二次返回 `BLOCKED_TRANSFER`。`bash -n scripts/run_rl_scale_v31.sh`。

验收：上述 CPU 测试通过；全新启动不带 `--resume-from-checkpoint` 时若输出目录已有训练产物会直接拒绝，显式恢复时才按最后完整检查点裁剪；中断后的下一次 `RL_RESUME=1` 不重放已提交步，也不把半成品检查点当成终态。数据仍须先达到 160,000/640/20%/零泄漏，本轮不启动 V100。2026-10-05 本地 `.venv` `pytest -q` 为 `337 passed, 9 skipped, 39 subtests passed`；`py_compile train/train_rl.py`、`bash -n scripts/run_rl_scale_v31.sh` 和 `git diff --check` 通过。

影响：pilot 的停止档和已提交日志不变。恢复会改写当前 run 里超出最后完整检查点的损失行和 rollout 行；这是取消未提交步，不是清理已提交审计。发布底座仍是 DPO Champion。
