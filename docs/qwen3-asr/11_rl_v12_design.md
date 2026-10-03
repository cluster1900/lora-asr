# RL v12：原始奖励差优势，失败时切到固定 0.10

| 项 | 值 |
| --- | --- |
| 日期 | 2026-09-30 |
| 状态 | 已执行。2026-09-30 11:18 CST 停在 Step 4，动作 `stop`，门禁 FAILED。没有启动 v13。后续见 `12_rl_v14_design.md` |
| 运行名 | `rl_pilot_v12`；只有切换条件成立时才有 `rl_pilot_v13` |
| 前序 | `rl_pilot_v10` Step 8 `BLOCKED_TRANSFER`；`rl_pilot_v11` Step 7 `STOPPED_KL` |
| 发布底座 | DPO Champion：`/data/mega-asr/runs/dpo_pilot_v2/merged_base` |
| 配置 | `configs/train/qwen3_asr_rl_v12.yaml`，切换时 `configs/train/qwen3_asr_rl_v13.yaml` |
| 入口 | `scripts/run_rl_pilot_v12.sh` |

## 背景

v10 和 v11 的获胜优势都是 1。裁剪前梯度范数在 3 到 5，`clip_grad_norm_(max_norm=1.0)` 之后 AdamW 的步长由学习率决定。两条运行的 Step 1 `policy_loss` 都是 `2.13505`。v10 用 `1e-5`，Step 8 的 `raw_kl` 是 `7.1e-5`，2,867 条 −6 次编辑，贪心奖励 `−0.0002`。v11 用 `2e-5`，Step 7 的 `raw_kl` 是 `5.68e-4`，编辑数仍是 −6，贪心奖励 `−0.0006`。β 的 KL 项比策略项小约三个数量级，提高 β 或放宽 `5e-4` 没有对应的测量支持。

探针里获胜差距的中位数是 `0.075`。优势从 1 改成这个差距后，裁剪前范数有机会降到 1 以下，Adam 才会看到被缩小的一步。2026-09-30 决定由这一轮直接测量这件事，并在范数仍然大于 1.5 时立刻换一组更小的固定优势。

## 范围

做两次互不覆盖的新运行，都从 DPO Champion 重新开始。

v12：学习率 `1e-5`，获胜优势等于原始奖励差，`β=0.04`，`max_raw_kl=5e-4`，温度、锚、`min_improvement=0.02` 和定长 2 条反向沿用 v10。最多 8 步。

v13：只在 v12 的训练日志里，裁剪前 `grad_norm` 的中位数仍大于 1.5 时启动。学习率仍是 `1e-5`，获胜优势固定为 `0.10`。β 和 KL 天花板不变。v13 不再往下切。

不做这些事：不恢复 v10 或 v11；不写入这两个目录；不把学习率改回 `2e-5` 或 `1.5e-5`；不提高 β；不放宽 `5e-4`；不把已更新话语送进 DPO；不写入 DPO Champion。

## 设计

优化器仍读 3,000 行 `pilot_rl.jsonl`，`sample_strategy: degraded`。验证池是 1,698 行 `rl_val_pool.jsonl`，SHA-256 `b0701db2ec3735222179e21fb4bf3c49c256d8da5d3a85cb7e61bfa6b7d9cd99`。WER 用 2,867 行 `validation.jsonl`。种子 `20260722`。

`train/rl_pair_advantage.py` 的 `winner_advantage` 有三种模式。`unit` 返回 1，这是 v10/v11 的行为，缺省仍是它。`raw_gap` 返回 `max(0, r_winner − r_greedy)`。`fixed` 返回配置里的 `fixed_value`，并且该值必须在 `(0, 1]`。服务器 `train/train_rl.py` 的 `select_anchored_training_pair` 按 YAML `grpo.advantage.mode` 调用它。本地 `main` 的 `train_rl.py` 仍不接受 `degraded`，这次不把本地那份覆盖到服务器，只在服务器文件上补这个调用。

v12 先做 128 条退化探针。决策不是 `GO_GRPO` 就不构造优化器。v13 复制 v12 的 `search_probe.json`。接收函数会按 `update_rate_k11`、`median_winning_gap_k11` 和 manifest SHA 重算质量，两轮的起点都是同一个 Champion。

驱动 `scripts/run_rl_pilot_v12.sh` 先跑到第 4 步，再给 `pipeline_state.global_step` 补贪心 held-out 和 2,867 条门禁。然后写 `switch_decision.json`：

| 条件 | 动作 |
| --- | --- |
| 整道门禁 PASSED | 导出到该 run 自己的 `merged_base`，停止 |
| Robust 增量 ≥ `0.0005` | 停止，不切 v13 |
| 裁剪前 `grad_norm` 中位数 > `1.5` | 停止 v12 的后续步，从 Champion 启动 v13 |
| 范数 ≤ `1.5`，状态是 `CHUNK_DONE`，步数 < 8，贪心奖励增量 ≥ `+0.001` | v12 继续到第 8 步 |
| 范数已经 ≤ `1.5`，奖励增量仍 < `+0.001`，或训练已被 KL 等设计内状态拦住 | 停止，不切 v13 |

v13 使用同一张表，但 `allow_switch=no`，所以范数仍然大于 1.5 时也只停止。已有 v12 的 `switch_decision.json` 且动作是 `switch`、v13 还没有 `pipeline_state.json` 时，再执行驱动只补 v13。v12 已有 `pipeline_state.json` 但还没有切换决定时，驱动拒绝，避免重写第 4 步。

通过线与 v11 相同：贪心 held-out 相对本次 Step 0 ≥ `+0.002`；Robust 六位小数 ≤ 0；clean ≤ `+0.02`；累计 clean ≤ `+0.025`；至少一个退化场景变好；有效输出 ≥ `0.95`。只有该 run 的 `gate.json` 为 PASSED 才导出到该 run 目录。

## 测试

`tests/test_rl_pair_advantage.py` 覆盖 `unit`、`raw_gap`、`fixed` 和非法值。`bash -n` 检查两个新脚本。服务器补丁之后对 `train/train_rl.py` 做 `py_compile`，并用同一组样例调用 `winner_advantage`。

启动前确认没有 `train_rl.py`、`parallel_inference.py` 或 held-out 打分进程。驱动在这些进程还在时退出。评测脚本要求预测正好 2,867 行；`verify_gate` 因 FAILED 返回 1 时，只要 `gate_step_<N>.json` 已写成，评测就算完成。

## 验收

v12 的 `switch_decision.json` 必须能读出动作、中位 `grad_norm`、贪心奖励增量和所评步数。若动作是 `switch`，v13 目录里要有自己的探针副本、检查点和门禁，并且 v12 的检查点还在。两个运行都不写 DPO Champion，也不写 v10/v11。PASSED 才在对应 run 下出现 `merged_base`。

## 影响

v10 与 v11 继续只作诊断。这次测量如果范数降到 1.5 以下而贪心奖励仍不够 `+0.002`，v12 驱动停止，不在这个驱动里改学习率、β 或 KL 天花板。测完之后的续训是 `rl_pilot_v14`，见 `12_rl_v14_design.md`。发布底座在整道门禁 PASSED 之前仍是 DPO Champion。
