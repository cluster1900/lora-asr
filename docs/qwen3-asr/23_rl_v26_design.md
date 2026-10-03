# RL v26：优化器跳过 noise 和 recording

| 项 | 值 |
| --- | --- |
| 日期 | 2026-10-02 |
| 状态 | 2026-10-02 04:39 CST Step 4 整门 FAILED，动作 `stop`。贪心 `−0.0005`，Robust `+0.000359`。不重跑。 |
| 运行名 | `rl_pilot_v26` |
| 前序 | v25 Step 4 门禁 FAILED。贪心 `−0.0003`，Robust `+0.000209`，动作 `stop` |
| 发布底座 | DPO Champion：`/data/mega-asr/runs/dpo_pilot_v2/merged_base` |
| 配置 | `configs/train/qwen3_asr_rl_v26.yaml` |
| 入口 | `scripts/run_rl_pilot_v26.sh` |

## 背景

v25 从 DPO Champion 新开，学习率 `1e-5`，优势 `raw_gap`，音频投影 199 个，`policy_token_mask=signed_edits`。2026-10-02 02:52 CST 的 Step 4 整门 FAILED，动作 `stop`，原因是贪心奖励低于 Step 0。

实测：

- 1,698 条贪心 held-out：`0.8773 → 0.8770`（`−0.0003`）。同一份池，SHA-256 `b0701db2…`。
- Robust macro `0.103792`，相对 DPO `0.103583` 增加 `0.000209`，编辑差 `+4`。
- clean 增加 `0.0`。有效输出 `1.0`。空输出和失败增量都是 `0`。
- 变好的场景是 `en|distortion`（`−0.000621`）和 `en|dropout`（`−0.000945`）。
- 变差的场景是 `en|noise`（`+0.002978`）和 `en|recording`（`+0.003021`）。其余场景在六位小数上没有移动。
- 9 条退化语音的编辑数相对 DPO 有变化，净 `+4`。其中 `vitw_sample_042621_noise` 单独 `+4`，预测仍是 “In the difficult moments, we recognize our thirst for fulfillment.”。这条不在 `pilot_rl.jsonl` 里，也不在 1,698 条奖励池里。其余八条加总为 0。
- 四步 `raw_kl` 最高 `4e-6`。累计奖励质量 `7.432122`，获胜组 64。`policy_keep_ratio` 是 `0.1449`、`0.1661`、`0.1458`、`0.1981`。
- 没有根目录 `gate.json`，没有 `merged_base`。Champion 仍是 4,076,190,936 字节，mtime unix `1790002044`。

`noise` 和 `recording` 是这份门禁里错误率上升的两个场景。优化器目前用 `sample_strategy: degraded`，这两个场景都在 2,000 条退化语音里。v26 让它们不进优化器。学习率、优势、掩码、音频投影和通过线都保持 v25。

## 范围

新目录 `/data/mega-asr/runs/rl_pilot_v26`。从 DPO Champion 新开。复制 v12 的 128 条探针，不复制损失日志。清单文件仍是完整的 `pilot_rl.jsonl`，探针哈希才能对上。过滤发生在 `build_epoch_sample_indices`。

保留 v25 的 `stop` 决定和 v24 的 `continue` 决定。不重跑 `scripts/run_rl_pilot_v25.sh`，也不删除这两个决定。不写 Champion。不恢复 v22、v23、v24 或 v25 的检查点。

学习率保持 `1.0e-5`。优势保持 `raw_gap`。局部阈值保持 `0.35`。`β` 保持 `0.04`。KL 天花板保持 `5.0e-4`。音频投影保持打开，目标数 199。掩码保持 `signed_edits`。跟步仍用 `train/rl_v16_decision.py`。horizon 24。第一块到 Step 4。

不做这些事：不改学习率；不把优势改成 `unit`、`fixed` 或 `capped_gap`；不关音频投影；不把掩码改回 `all` 或 `changes_only`；不提高 `β`；不放宽 `5e-4`；不改通过线；不把那条验证集噪声样本写进训练集。

服务器训练器在策略名 `degraded_skip_regressed` 时调用 `degraded_skip_regressed_indices`。`degraded` 分支保持原样。驱动在训练器源码里没有这次调用时拒绝启动。四张卡空闲、v25 的决定仍是 `stop` 时，才往新目录启动。不覆盖已有训练器备份，也不用本地 `train/train_rl.py` 替换服务器副本。

## 设计

`train/rl_sample_strategy.py` 的 `degraded_skip_regressed_indices(rows)` 返回清单顺序的下标：

- `condition_group` 是 `degraded`，或者场景名里没有 `clean`，才算退化行。这和服务器上 `degraded` 策略的入选规则相同。
- 场景名去掉空白并转成小写后，等于 `noise` 或 `recording` 的行不进优化器。`noise_like` 这种不相等的名字保留。中英文都按场景名判断，清单里的场景字段本身不带语言前缀。
- clean 行不进优化器。
- 一个都没选中时抛 `ValueError`，避免未知策略落到全文件打乱。
- 函数不打乱。服务器在这条分支里用 `random.Random(seed + epoch)` 打乱，种子仍是 `20260722`。

`pilot_rl.jsonl` 共 3,000 行：clean 1,000，noise 251，recording 266。这条策略应选出 1,483 行，其中包含 distortion、dropout、echo、far_field、obstructed。准备阶段按清单实数核对；数量不对就只删除刚建的 v26 目录。

配置标量：`learning_rate` `1.0e-5`，`max_steps` 24，`loss_reduction` `sequence_sum`，`sample_strategy` `degraded_skip_regressed`，`mode` `raw_gap`，`local_max_relative` `0.35`，`beta` `0.04`，`max_raw_kl` `5.0e-4`，`policy_token_mask` `signed_edits`，`train_audio_projections` `true`，目标数 199。命令行 `--sample-strategy` 传同一个名字，因为训练器优先采用命令行。argparse 的可选值同时包含 `balanced`、`standard`、`degraded` 和 `degraded_skip_regressed`。缺了最后这个名字时，进程在读配置之前就退出。

准备阶段还核对：v16、v20、v21、v22、v23 的决定是 `stop`；v24 的决定仍是 `continue`；v25 的决定是 `stop`；探针是 128 条 `GO_GRPO`；Champion 仍是 4,076,190,936 字节；训练器源码含 `degraded_skip_regressed_indices(`、`signed_edit_loss_args(`，以及 argparse 可选值里的 `degraded_skip_regressed`。

已有 v26 决定且动作不是 `continue` 时，再执行直接退出。动作是 `continue` 的重入直接拒绝。目录已存在但没有决定时也拒绝。

## 测试

`tests/test_rl_sample_strategy.py` 调用已发出的 `degraded_skip_regressed_indices`。noise、recording、大小写不同的 `Noise` 和 clean 都离开下标列表。distortion、dropout、echo、far_field、obstructed 和 `noise_like` 按清单顺序留下。只剩被排除的行时抛 `ValueError`。

`tests/test_rl_v26_contract.py` 读取发出的配置文本里的标量，确认采样策略是 `degraded_skip_regressed`，学习率是 `1.0e-5`，掩码是 `signed_edits`，优势是 `raw_gap`，音频投影打开，目标数 199。文件里不能出现 `5.0e-6`、`2.0e-5`、`mode: unit`、`mode: capped_gap`、`mode: fixed`、`train_audio_projections: false`、`policy_token_mask: all` 或 `policy_token_mask: changes_only`。

驱动脚本必须保护 v10 到 v25 和 Champion，失败清理只指向 v26 目录，要求 v25 的决定是 `stop`，v24 的决定仍是 `continue`，并且命令行采样策略是 `degraded_skip_regressed`。

`bash -n scripts/run_rl_pilot_v26.sh`。`scripts/README.md` 和 `train/README.md` 列出新文件。

## 验收

通过线不变：贪心 held-out 相对这次运行自己的 Step 0 ≥ `+0.002`，Robust 六位小数 ≤ 0，clean ≤ `+0.02`，累计 clean ≤ `+0.025`，至少一个退化场景变好，有效输出 ≥ `0.95`。只有整道门禁 `PASSED` 才把权重合到 `rl_pilot_v26/merged_base`。

v10 到 v25 的决定保持不动。DPO Champion 的大小和时间戳不变。门禁没 PASSED 之前，不把这次运行说成通过。本地单测只说明选行函数的行为。

启动后的日志要出现 `sample_strategy='degraded_skip_regressed'`，并且 `virtual_epoch_len=1483`。四张卡都是 `LoRA targets: 199`，掩码仍是 `signed_edits`。

## 结果

2026-10-02 03:27 CST 的第一次启动在 argparse 退出：`--sample-strategy` 当时只接受 `balanced`、`standard`、`degraded`。进程没有写出检查点，也没有决定文件。空目录已删除。argparse 补上 `degraded_skip_regressed` 之后，03:29 CST 重新启动。驱动 PID 130104，PPID 1。torchrun PID 130113。

准备阶段打印 `v26 baseline ok 1483`。日志是 `Loaded 3000 training samples (sample_strategy='degraded_skip_regressed', virtual_epoch_len=1483)`，四张卡都是 `LoRA targets: 199 train_audio_projections=True`，启动行带 `policy_token_mask=signed_edits`。`source.json` 记录 `epoch_rows` 1483，`excluded_scenarios` 是 noise 和 recording，`copied_loss_log` 为 false。

Step 0 贪心 Full Held-out 是 `0.8773`，1,698 行，清单 sha 与奖励池一致。四步 `policy_loss` 是 `0.01763`、`0.02502`、`0.03116`、`0.03748`。`raw_kl` 是 0、0、`8e-6`、`6e-6`。批平均奖励是 `0.9256`、`0.8926`、`0.9048`、`0.8378`。`policy_keep_ratio` 是 `0.1301`、`0.1385`、`0.1529`、`0.1676`。累计奖励质量 `6.459451`，获胜组 56。学习率 Step 1 是 `5e-6`，之后是 `1e-5`。

2026-10-02 04:39:18 CST 写成 `gate_step_4.json`，时间戳 `2026-10-01T20:39:18.682169+00:00`。`gate_status` 是 FAILED。贪心 `0.8773 → 0.8768`（`held_out_reward_improvement` `−0.0005`）。Robust macro `0.103942`，相对 DPO `0.103583` 的 `robust_error_rate_increase` 是 `0.000359`，`robust_edit_delta` 是 6。clean `−2.5e-05`。有效输出 `1.0`。`improved_scenarios` 是空的，`degraded_scenario_improvements_count` 是 0。未通过的是 `degraded_improvement`、`robust_retention` 和 `held_out_reward`。`clean_retention`、`valid_output_rate`、`empty_output_rate`、`failure_rate`、`clean_cumulative_retention`、`zero_variance` 通过。

`switch_decision.json` 动作 `stop`，原因是 greedy reward fell below step 0，`scored_step` 4，训练状态 `CHUNK_DONE`。没有根目录 `gate.json`，没有 `merged_base`。驱动随后退出。04:54 CST 四张卡各 4 MiB。Champion 仍是 4,076,190,936 字节，mtime unix `1790002044`。v25 的决定仍是 `stop`，v24 的决定仍是 `continue`。

决定保留为 `stop`。再执行 `scripts/run_rl_pilot_v26.sh` 会读到这个决定并退出。不要删决定。下一轮回到完整退化集，见 `24_rl_v27_design.md`。

服务器训练器备份：策略分支之前是 `/data/mega-asr/logs/train_rl.py.bak-20261002-v26`（140,776 字节），argparse 可选值之前是 `/data/mega-asr/logs/train_rl.py.bak-20261002-v26-choices`（141,005 字节）。更早的 v16、v19、v21、v23、v24、v25 备份都还在。

## 影响

v26 保持终态，决定是 `stop`。跳过 `noise` 和 `recording` 之后，训练批奖励更高，held-out 更差，而且没有任何退化场景变好。v25 保持终态。发布底座仍是 DPO Champion。通过线不变。这一轮没有改 v25 的损失形式。不把 v26 说成通过。
