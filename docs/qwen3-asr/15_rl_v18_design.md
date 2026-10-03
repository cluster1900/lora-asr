# RL v18：从 v16 Step 8 继续原始奖励差

| 项 | 值 |
| --- | --- |
| 日期 | 2026-10-01 |
| 状态 | 2026-10-01 14:08 CST 在 Step 10 停止。训练状态 `STOPPED_KL`，门禁 FAILED。没有 `switch_decision.json`，没有 `merged_base` |
| 运行名 | `rl_pilot_v18` |
| 前序 | v17 Step 4 动作 `stop`，贪心 `−0.0003`，Robust `+0.000359`，门禁 FAILED。v16 Step 8 是贪心 `+0.0004`、Robust `+0.000164` |
| 发布底座 | DPO Champion：`/data/mega-asr/runs/dpo_pilot_v2/merged_base` |
| 配置 | `configs/train/qwen3_asr_rl_v18.yaml` |
| 入口 | `scripts/run_rl_pilot_v18.sh` |

## 背景

v16 用学习率 `1e-5`、原始奖励差和局部改正，Step 4 到 Step 8 的贪心增量从 0 升到 `+0.0004`，Robust 从 `+0.000321` 降到 `+0.000164`。驱动因为「Step 8 起 Robust 仍大于 0」停止。剩下的 +4 次编辑全部来自 `vitw_sample_042621_noise`，其余退化格子加总是 0。

v17 只把这批局部获胜样本的优势改成 1。Step 4 的获胜组还是 64，累计奖励质量还是 `7.432122`，和 v16 Step 4 相同，所以选中的样本没有变。裁剪前梯度变成 `4.31`、`4.87`、`3.52`、`4.50`，策略损失从 v16 的 `0.06`–`0.20` 升到 `0.60`–`1.13`。贪心 held-out 变成 `0.8770`（`−0.0003`），Robust 增量 `+0.000359`（+6 次编辑）。两条门禁都比 v16 同期更差。驱动按贪心低于 Step 0 停止。单位优势不再继续。

因此下一轮回到 v16 已经在变好的那条信号，并把 Step 8 的水平停止换成「相对上一块不回升才继续」。训练器在 Step 12 以后、质量低于 17 时会返回 `BLOCKED_SEARCH`。v16 Step 8 的质量是 `11.725`，下一块可能过不了 17。这个绝对质量线是按未过滤的奖励差标定的。质量仍 ≥ `8.50` 时，驱动把这次退出当成可续的块。

## 范围

新目录 `/data/mega-asr/runs/rl_pilot_v18`。只读恢复 `/data/mega-asr/runs/rl_pilot_v16/checkpoints/step_8`，并复制 v16 的 `loss_log.jsonl` 和 `gate_step_8.json`。Step 0 贪心奖励保持日志里的 `0.8773`，累计质量从 `11.724898` 接着记。不在 v16 目录里续写，不删除 v16 的 `switch_decision.json`。不恢复 v17。不写 DPO Champion。

训练配置与 v16 相同：学习率 `1e-5`，优势 `raw_gap`，`local_max_relative: 0.35`，`β=0.04`，`max_raw_kl=5e-4`，种子 `20260722`。horizon 改为 48。跟步函数沿用 `train/rl_v17_decision.py`：Robust 比上一块高出 `1e-6` 以上、增量 ≥ `0.0005`、贪心转负，或质量 < `8.50` 的 `BLOCKED_SEARCH`，都停止。否则续块。

不做这些事：不把优势改回 1 或固定 `0.10`；不提高学习率、β 或 `5e-4`；不改通过线；不改服务器训练器的停止表。

## 设计

`configs/train/qwen3_asr_rl_v18.yaml` 的 `train.max_steps` 是 48。驱动第一块的 `--max-steps` 是 12，`--resume-from-checkpoint` 指向 v16 的 Step 8。之后每块加 4，检查点改从 v18 自己的目录读。

准备阶段核对：v16 决定是 `stop`；Step 8 检查点有 adapter 和 optimizer；复制来的 Step 0 奖励是 `0.8773`；日志末尾质量是 `11.724898`；复制来的 Step 8 门禁是贪心 `+0.0004`、Robust `+0.000164`；Champion 仍是 4,076,190,936 字节。任一核对失败时只删除刚建的 v18 目录。

已有 v18 决定且动作不是 `continue` 时，再执行直接退出。动作是 `continue` 的重入直接拒绝。

## 测试

`tests/test_rl_v18_contract.py` 读取发出的 yaml，确认模式是 `raw_gap`、没有 `unit`、学习率 `1.0e-5`、局部阈值 `0.35`、horizon 48。同一测试调用已有的 `followup_action`：v16 Step 8 的数字继续；Step 12 的 Robust 从 `0.000164` 升到 `0.0002` 时，即使质量是 16 也停止。

`bash -n scripts/run_rl_pilot_v18.sh`。目录 README 要列出新脚本。

启动前四张卡没有 `train_rl.py`、`parallel_inference.py` 或 held-out 打分。

## 验收

通过线不变：贪心 held-out 相对这份日志里的 Step 0（`0.8773`）≥ `+0.002`，Robust 六位小数 ≤ 0，clean ≤ `+0.02`，累计 clean ≤ `+0.025`，至少一个退化场景变好，有效输出 ≥ `0.95`。只有整道门禁 `PASSED` 才把权重合到 `rl_pilot_v18/merged_base`。

v16 的决定保持 `stop`，v17 的决定保持 `stop`。DPO Champion 的大小和时间戳不变。

## 影响

v17 说明把局部获胜样本的优势放大到 1，会在四步内把贪心打负，也不能改善 Robust。v18 因此不再放大优势。它只是让 v16 已经观察到的下降趋势继续。发布底座在整道门禁 PASSED 之前仍是 DPO Champion。

## 结果

2026-10-01 13:18 CST 启动，驱动 PID 105112。日志确认从 v16 Step 8 恢复，本块目标是 Step 12。Step 9 的 `raw_kl` 是 `0.000402`，Step 10 是 `0.000529`，训练器在 Step 10 返回 `STOPPED_KL`。学习率两步都是 `1e-5`。Step 9、Step 10 的获胜组是 17 和 18，奖励差中位数是 `0.083` 和 `0.092`，单步质量是 `1.682` 和 `2.947`。Step 8 的中位数是 `0.030`，单步质量是 `0.999`，`raw_kl` 是 `5e-5`。

Step 10 贪心 held-out 是 `0.8774`，相对 Step 0 的 `0.8773` 只剩 `+0.0001`（Step 8 是 `+0.0004`）。Robust 增量 `+0.000014`（+2 次编辑），`pilot_robust_macro` `0.103597`。clean `−0.000124`，有效输出 `1.0`，变好的场景是 `en|distortion`、`en|echo`、`zh|obstructed`。未通过的是 `robust_retention` 和 `held_out_reward`。累计质量 `16.354781`。没有 `merged_base`。

打分之后，跟步脚本按 Step 10 去找 `gate_step_6.json`。上一份已经打分的门禁是复制来的 Step 8，所以脚本以 `missing prior gate` 退出，没有写成 `switch_decision.json`。`pipeline_followup.json` 的状态是 `FAILED`。目录已经存在，再执行 `scripts/run_rl_pilot_v18.sh` 会拒绝。不要删目录重跑，也不要从 Step 10 继续：那份检查点已经越过 `5e-4`，贪心也比 Step 8 更低。

下一步把超过 `0.05` 的奖励差截断，仍从 v16 Step 8 只读恢复。见 `16_rl_v19_design.md`。
