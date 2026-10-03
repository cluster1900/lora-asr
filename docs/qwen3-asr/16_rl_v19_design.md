# RL v19：把过大的奖励差截断在 0.05

| 项 | 值 |
| --- | --- |
| 日期 | 2026-10-01 |
| 状态 | 2026-10-01 15:34 CST 在 Step 12 停止，动作 `stop`，门禁 FAILED。没有 `merged_base` |
| 运行名 | `rl_pilot_v19` |
| 前序 | v18 Step 10 训练状态 `STOPPED_KL`，门禁 FAILED。贪心 `+0.0001`，Robust `+0.000014`。v16 Step 8 仍是这条轨迹上更好的一点：贪心 `+0.0004`，Robust `+0.000164`，`raw_kl` `5e-5` |
| 发布底座 | DPO Champion：`/data/mega-asr/runs/dpo_pilot_v2/merged_base` |
| 配置 | `configs/train/qwen3_asr_rl_v19.yaml` |
| 入口 | `scripts/run_rl_pilot_v19.sh` |

## 背景

v18 只读恢复 v16 Step 8，学习率、β、`5e-4` 和局部阈值都没变，优势仍是没有上限的原始奖励差。Step 8 的奖励差中位数是 `0.030`，单步质量 `0.999`，`raw_kl` `5e-5`，贪心增量 `+0.0004`。接下来两步的中位数升到 `0.083` 和 `0.092`，Step 10 的单步质量升到 `2.947`（均值约 `0.164`）。`raw_kl` 在 Step 9 变成 `0.000402`，Step 10 变成 `0.000529`，训练器返回 `STOPPED_KL`。贪心增量退回 `+0.0001`。Robust 降到 `+0.000014`（+2 次编辑），仍然大于 0。

学习率两步都是 `1e-5`，裁剪前梯度是 `0.385` 和 `0.613`，没有重新进入 v17 那种大于 1 的单位步。变大的是奖励差本身。Step 8 那些还没把 KL 推过天花板的更新，中位数低于 `0.05`。Step 9 之后的中位数已经高于 `0.05`。

因此这一轮只改优势尺度：获胜优势改成 `min(奖励差, 0.05)`。不超过 `0.05` 的局部改正保持原来的权重。更大的奖励差不再按全额进入反向。学习率、β、`5e-4`、局部阈值 `0.35` 和种子都不改。不从 v18 Step 10 继续，因为那份权重已经越过 KL 上限，贪心也低于 Step 8。

## 范围

新目录 `/data/mega-asr/runs/rl_pilot_v19`。只读恢复 `/data/mega-asr/runs/rl_pilot_v16/checkpoints/step_8`，并复制 v16 的 `loss_log.jsonl` 和 `gate_step_8.json`。Step 0 贪心奖励保持 `0.8773`，累计质量从 `11.724898` 接着记。不写 v16、v17、v18 或 DPO Champion。不删除已有的 `switch_decision.json`。v18 没有这道决定，也不补写。

跟步规则仍是 v17 的 `followup_action`。上一块 Robust 改从「当前步之前最近一份门禁」读取。v18 在 Step 10 停止时去找 `gate_step_6.json`，那份文件不存在，驱动没写成决定。KL 停止不保证落在 4 的倍数上，所以 v19 用最近一份更早的门禁。`STOPPED_KL` 仍然停止。

不做这些事：不把优势改回 1 或固定 `0.10`；不提高学习率、β 或 `5e-4`；不改通过线；不改 `decide_rl_stop` 的阈值。

## 设计

`configs/train/qwen3_asr_rl_v19.yaml` 的 `grpo.advantage.mode` 是 `capped_gap`，`cap` 是 `0.05`。`train.max_steps` 是 48。驱动第一块的 `--max-steps` 是 12，`--resume-from-checkpoint` 指向 v16 的 Step 8。之后每块加 4，检查点改从 v19 自己的目录读。

`winner_advantage` 在 `capped_gap` 下返回 `min(max(0, 奖励差), cap)`。`cap` 必须落在 `(0, 1]`。`raw_gap`、`unit`、`fixed` 的返回值不变。服务器训练器把 `capped_gap` 加入允许的模式，并把配置里的 `cap` 传给这个函数。本地 `train/train_rl.py` 仍然不是服务器那份，不要覆盖。

准备阶段核对：v16 决定是 `stop`；v18 的 `pipeline_followup.json` 不是 `TRAINING`；Step 8 检查点有 adapter 和 optimizer；复制来的 Step 0 奖励是 `0.8773`；日志末尾质量是 `11.724898`；复制来的 Step 8 门禁是贪心 `+0.0004`、Robust `+0.000164`；Champion 仍是 4,076,190,936 字节。任一核对失败时只删除刚建的 v19 目录。

已有 v19 决定且动作不是 `continue` 时，再执行直接退出。动作是 `continue` 的重入直接拒绝。目录已存在但没有决定时也拒绝。

## 测试

`tests/test_rl_pair_advantage.py` 调用发出的 `winner_advantage`：奖励差 `0.03` 在上限 `0.05` 下保持 `0.03`；奖励差 `0.15` 变成 `0.05`；负差变成 0；上限 `0` 或 `1.5` 拒绝。同一函数的 `raw_gap` 不吃这个上限。

`tests/test_rl_v19_contract.py` 读取发出的 yaml，确认模式是 `capped_gap`、上限 `0.05`、没有 `unit` 或 `raw_gap` 模式行、学习率 `1.0e-5`、局部阈值 `0.35`、horizon 48。它调用 `latest_prior_robust`：只有 Step 8 门禁时，Step 10 读到 `0.000164`；Step 8 和 Step 10 都在时，Step 12 读到 Step 10 的值。它调用已有的 `followup_action`：`STOPPED_KL` 停止。

`bash -n scripts/run_rl_pilot_v19.sh`。`scripts/README.md` 和 `train/README.md` 列出新文件。

启动前四张卡没有 `train_rl.py`、`parallel_inference.py` 或 held-out 打分。

## 验收

通过线不变：贪心 held-out 相对这份日志里的 Step 0（`0.8773`）≥ `+0.002`，Robust 六位小数 ≤ 0，clean ≤ `+0.02`，累计 clean ≤ `+0.025`，至少一个退化场景变好，有效输出 ≥ `0.95`。只有整道门禁 `PASSED` 才把权重合到 `rl_pilot_v19/merged_base`。

v16 和 v17 的决定保持 `stop`。v18 的目录保持不写。DPO Champion 的大小和时间戳不变。

## 影响

v18 说明没有上限的奖励差在 Step 8 之后会把单步质量拉高，两步内越过 `5e-4`，并把已经拿到的贪心增量退回去。截断只削掉高于 `0.05` 的那一部分。发布底座在整道门禁 PASSED 之前仍是 DPO Champion。

## 结果

2026-10-01 14:39 CST 启动，驱动 PID 107939。日志确认从 v16 Step 8 恢复，本块到 Step 12，horizon 48。解析后的配置是 `mode: capped_gap`、`cap: 0.05`、学习率 `1e-5`。

`raw_kl` 在反向更新之前、按当前策略在这一批数据上计算。Step 9 因此和 v18 的 Step 9 相同，都是 `0.000402`：两轮的起点都是 v16 Step 8，第一批数据也相同。真正被截断改变的是梯度。v18 Step 9 的 `grad_norm` 是 `0.385`，v19 是 `0.159`。Step 10 从 `0.613` 降到 `0.193`。奖励差中位数仍是 `0.083`，高于 `0.05`，所以大多数获胜样本都被截断了。

四步 `raw_kl` 为 `0.000402`、`0.000464`、`0.000384`、`0.000640`。Step 12 返回 `STOPPED_KL`。贪心 held-out `0.8773 → 0.8772`（`−0.0001`）。Robust 增量 `+0.000014`（+2 次编辑），`pilot_robust_macro` `0.103597`，与 v18 Step 10 相同。clean `−9.9e-05`，有效输出 `1.0`，变好的场景是 `en|distortion`、`en|echo`、`en|obstructed`、`zh|obstructed`。未通过的是 `robust_retention` 和 `held_out_reward`。累计质量 `20.414682`。驱动动作 `stop`，原因是贪心低于 Step 0。没有 `merged_base`。

不要再启动 `scripts/run_rl_pilot_v19.sh`。不要从 Step 12 继续。把优势再缩一档、或把学习率再降一档，都不能越过已经测到的上限：v14 Step 8 的贪心 `+0.0005` 和 v16 Step 8 的 `+0.0004` 是这条 GRPO 信号的高点，离 `+0.002` 还差约四到五倍；从那个高点再走，KL 会在 `5e-4` 附近把贪心退回 Step 0，Robust 停在 `+0.000014`。
