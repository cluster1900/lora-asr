# RL v14：沿 v12 的权重继续，必要时把学习率升到 2e-5

| 项 | 值 |
| --- | --- |
| 日期 | 2026-09-30 |
| 状态 | 已跑完。v14 Step 8 为 `BLOCKED_TRANSFER`，门禁 FAILED。v15 Step 4 贪心转负，动作 `stop`，门禁 FAILED。都没有 `merged_base` |
| 运行名 | `rl_pilot_v14`；切换时才有 `rl_pilot_v15` |
| 前序 | `rl_pilot_v12` Step 4 `CHUNK_DONE`，动作 `stop`，`gate_step_4.json` FAILED |
| 发布底座 | DPO Champion：`/data/mega-asr/runs/dpo_pilot_v2/merged_base` |
| 配置 | `configs/train/qwen3_asr_rl_v14.yaml`，切换时 `configs/train/qwen3_asr_rl_v15.yaml` |
| 入口 | `scripts/run_rl_pilot_v14.sh` |

## 背景

v12 把获胜优势从 1 改成原始奖励差，学习率保持 `1e-5`。Step 4 贪心奖励 `0.8773 → 0.8775`（`+0.0002`），2,867 条 Robust `+5.9e-05`（+2 次编辑）。四步 `raw_kl` 是 0、0、`5e-6`、`2e-6`。裁剪前 `grad_norm` 是 `0.607`、`1.435`、`1.964`、`1.433`，中位数 `1.434`。policy loss 约 `0.12–0.27`。驱动按事先写好的规则停止，没有启动固定优势 `0.10` 的 v13。

同一条种子上，v10 Step 4 用优势 1 和 `1e-5`，贪心增量是 `−0.0001`，Robust `+0.000306`（+5 次编辑）。v11 Step 4 用优势 1 和 `2e-5`，贪心增量也是 `−0.0001`，Robust +1 次编辑。v12 的获胜组数是 19、19、21、18，累计奖励质量 `10.603235`；v11 Step 4 的质量是 `10.59`。采样找到的差距几乎没变，变的是写进损失的优势。原始奖励差是目前同一个 Step 4 上更好的方向。

Step 2 到 Step 4 的裁剪前范数仍然在 1 附近或以上，Adam 在这些步上看到的步长仍由学习率决定。v12 只走了 4 步就因为增量小于 `+0.001` 停下。v13 的固定 `0.10` 是给“范数仍然大于 1.5”准备的，这次中位数是 `1.434`，再把优势缩小没有对应的测量。β 和 `5e-4` 这一轮也不动。

## 范围

做两段互不覆盖的运行。v12 目录只读：复制 `loss_log.jsonl` 和 `search_probe.json`，从 `checkpoints/step_4` 恢复。不删除 v12 的 `switch_decision.json`，不重写它的门禁，也不再执行 `scripts/run_rl_pilot_v12.sh`。

v14：学习率 `1e-5`，优势 `raw_gap`，`β=0.04`，`max_raw_kl=5e-4`，种子 `20260722`，温度和锚与 v12 相同。输出目录 `/data/mega-asr/runs/rl_pilot_v14`。horizon 12。第一块从 Step 4 训到 Step 8。Step 0 基线用复制来的 `0.8773`，不在已更新的权重上重测 Step 0。

v15：只在 v14 的 Step 8 被训练器写成 `BLOCKED_TRANSFER`，且贪心增量仍 ≥ 0、Robust 增量 < `0.0005` 时，从 DPO Champion 重新开始。学习率 `2e-5`，优势仍是 `raw_gap`。不恢复 v14 的 optimizer，因为那份状态里的学习率是 `1e-5`。复用 v12 的探针文件。没有 v16。

不做这些事：不把优势改回 1 或固定 `0.10`；不试 `1.5e-5`；不提高 β；不放宽 `5e-4`；不恢复 v10、v11；不把 v12 目录接着写下去；不写入 DPO Champion。奖励增量变成负数、Robust 增量 ≥ `0.0005`、KL 停止或搜索停止时，不启动 v15。

## 设计

v14 复制 v12 的 `loss_log.jsonl` 之后，Step 0 行必须是贪心 Full Held-out，奖励 `0.8773`，1,698 行。四步 `reward_mass_in_step` 之和必须是 `10.603235`。训练器按这个日志累加奖励质量，不读取某一行里已经写好的累计值。Step 8 时质量若仍按每四步大约 `10.6` 增长，会到 21 左右，高于 Step 8 的转移线 `11.33`。贪心增量仍 < `+0.001` 时，训练器返回 `BLOCKED_TRANSFER`。

`train/rl_v14_decision.py` 在门禁写成之后选择动作：

| 条件 | 动作 |
| --- | --- |
| 整道门禁 PASSED | 导出到该 run 自己的 `merged_base`，停止 |
| Robust 增量 ≥ `0.0005` | 停止，不切 v15 |
| 贪心增量 < 0 | 停止，不切 v15 |
| `STOPPED_KL`、`STOPPED_REWARD_DROP`、`STOPPED_ROBUST`、`FAILED_ZERO_VARIANCE`、`BLOCKED_NO_WINNERS`、`BLOCKED_SEARCH` | 停止，不切 v15 |
| `BLOCKED_TRANSFER`，增量 ≥ `+0.001`，步数 < 12 | 同一 run 继续到下一步 |
| `BLOCKED_TRANSFER`，增量 ≥ 0 且 < `+0.001`，步数 ≥ 8，并且这是 v14 | 从 Champion 启动 v15 |
| `BLOCKED_TRANSFER`，但这是 v15 | 停止 |
| `CHUNK_DONE` 且步数 < 12 | 同一 run 继续 |
| `CHUNK_DONE` 且步数已经到 12 | 停止 |

v15 的第一步评 Step 4。增量 ≥ 0 且状态是 `CHUNK_DONE` 时继续到 Step 8；增量 < 0 或 Robust 增量 ≥ `0.0005` 时停止。Step 8 再被 `BLOCKED_TRANSFER` 拦住就停止，不再开新学习率。增量已经 ≥ `+0.001` 时可以继续到 Step 12。

通过线不变：贪心 held-out 相对该 run 日志里的 Step 0 ≥ `+0.002`；Robust 六位小数 ≤ 0；clean ≤ `+0.02`；累计 clean ≤ `+0.025`；至少一个退化场景变好；有效输出 ≥ `0.95`。只有整道 `gate.json` 为 PASSED 才导出，目标必须落在该 run 的 `merged_base`。

## 测试

`tests/test_rl_v14_decision.py` 覆盖通过、Robust 停止、奖励转负、KL 停止、搜索停止、v14 的转移切换、v15 的转移停止、门禁增量已经到 `+0.001` 时继续、Step 4 的 `CHUNK_DONE` 继续、Step 12 停止，以及不认识的状态停止。

`bash -n scripts/run_rl_pilot_v14.sh`。目录 README 合同测试要能看到新脚本和新的 `train/rl_v14_decision.py`。

启动前确认四张卡没有 `train_rl.py`、`parallel_inference.py` 或 held-out 打分进程，并且 v12 的 Step 4 检查点含 adapter、optimizer、scheduler 和四份 RNG。复制后的日志对不上 `0.8773` 或 `10.603235` 时，驱动在构造优化器之前退出。

## 验收

v14 目录里有自己的 `loss_log.jsonl`、探针、检查点和所评步的 `gate_step_<N>.json`。v12 的 `switch_decision.json` 仍是 `stop`，`gate_step_4.json` 仍是 FAILED，v12 不新增 checkpoint。DPO Champion 的 `model.safetensors` 时间戳仍是 2026-09-21 22:47 CST。v15 只有在切换动作出现后才有目录。PASSED 之前两个运行都没有 `merged_base`。

## 结果

v14 于 2026-09-30 23:42 CST 写完 Step 8，00:06 CST 门禁完成。贪心奖励 `0.8773 → 0.8778`（`+0.0005`）。累计奖励质量 `18.255`，高于 `11.33`，增量低于 `+0.001`，训练器状态是 `BLOCKED_TRANSFER`。Step 5–8 的裁剪前 `grad_norm` 是 `0.499`、`0.829`、`0.725`、`0.837`，这四步已经低于裁剪阈值。`raw_kl` 最高是 Step 7 的 `9.2e-5`。2,867 条 Robust `+0.000247`（+3 次编辑），clean `+1e-05`，有效输出 `1.0`。变好的场景是 `en|distortion` 和 `en|echo`。未通过的是 `held_out_reward` 和 `robust_retention`。

v15 于 2026-10-01 01:18 CST 停在 Step 4。贪心奖励 `0.8773 → 0.8771`（`−0.0002`），动作是 `stop`，没有进入 Step 8。Robust `+0.00015`（+2 次编辑），clean 为 0，有效输出 `1.0`。Step 4 的 `raw_kl` 是 `4.8e-5`。Step 1 的 `policy_loss` 仍是 `0.22666`，和 v12 的同一步一致。

两条运行都没有 `merged_base`。v12 的 `switch_decision.json` 仍是 Step 4 的 `stop`，检查点仍只有 `step_4`。DPO Champion 的权重时间戳仍是 2026-09-21 22:47 CST。

## 影响

v10、v11、v12 继续只作诊断。v13 的配置文件保留，但不启动。发布底座在整道门禁 PASSED 之前仍是 DPO Champion。v15 若启动，它的 Step 0 是 Champion 上新测的贪心奖励，不沿用 v14 已经更新过的权重。
