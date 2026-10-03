# RL v17：局部改正用单位优势，Robust 不回升就继续

| 项 | 值 |
| --- | --- |
| 日期 | 2026-10-01 |
| 状态 | 2026-10-01 13:13 CST 在 Step 4 停止，动作 `stop`，门禁 FAILED。没有 `merged_base` |
| 运行名 | `rl_pilot_v17` |
| 前序 | v16 Step 8 动作 `stop`，贪心 `+0.0004`，Robust `+0.000164`（+4 次编辑），门禁 FAILED |
| 发布底座 | DPO Champion：`/data/mega-asr/runs/dpo_pilot_v2/merged_base` |
| 配置 | `configs/train/qwen3_asr_rl_v17.yaml` |
| 入口 | `scripts/run_rl_pilot_v17.sh` |

## 背景

v16 保持学习率 `1e-5` 和原始奖励差，只训练相对贪心文本的局部改正。Step 8 的贪心 held-out 从 `0.8773` 到 `0.8777`（`+0.0004`）。Robust macro 从 Step 4 的 `+0.000321` 降到 `+0.000164`，编辑差从 +7 降到 +4。驱动按「Step 8 起 Robust 仍大于 0 就停止」退出。`raw_kl` 最高 `5.0e-5`，Step 5–8 的裁剪前梯度是 `0.273` 到 `0.351`。β 和 `5e-4` 没有顶到。

Step 8 剩下的 +4 次编辑全部来自 `en|noise` `vitw_sample_042621_noise`。其余退化格子加总是 0。这一条从 v10 到 v16 都写成同一句无关错文本，`pilot_rl.jsonl` 里没有它的参考文本。v10 用单位优势时，别的格子多改善了大约 10 次编辑，所以 Robust 仍能是 −6；v16 的原始奖励差把别的格子收到了和 DPO 持平，但没有再往下。

奖励质量按原始奖励差累计，不按优势权重累计。v16 Step 8 的质量是 `11.725`。按 Step 4 到 Step 8 的增速，Step 12 会落在 17 下面。训练器在 Step ≥ 12、贪心增量仍小于 `+0.002`、质量小于 17 时返回 `BLOCKED_SEARCH`。这道绝对质量线是按未过滤的原始奖励差标定的。局部过滤把质量压低之后，它会在搜索其实没有塌掉的时候停掉进程。

## 范围

新目录 `/data/mega-asr/runs/rl_pilot_v17`。从 DPO Champion 重新开始，不恢复 v16 的检查点，也不复制 v16 的损失日志。复制 v12 的 `search_probe.json`，不重跑探针。不写 v10、v11、v12、v13、v14、v15、v16，也不写 DPO Champion。

训练信号只改一件事：局部改正仍然用 `local_max_relative: 0.35`，被选中的获胜样本优势从原始奖励差改为 `unit`（固定为 1）。学习率仍是 `1e-5`。`β=0.04`。`max_raw_kl=5e-4`。种子 `20260722`。温度、锚和 `min_improvement=0.02` 不变。

跟步规则也换成能让这条信号走完的版本，写在 `train/rl_v17_decision.py`：

1. 整道门禁 `PASSED` 才导出，目录只限 `rl_pilot_v17/merged_base`。
2. Robust 增量 ≥ `0.0005`、贪心增量 < 0，或训练器返回 KL、奖励下跌、`STOPPED_ROBUST`、零方差、没有获胜组，都停止。
3. `BLOCKED_SEARCH` 且累计奖励质量 < `8.50` 停止。质量 ≥ `8.50` 时，这一状态和 `BLOCKED_TRANSFER`、`CHUNK_DONE` 一样可以续块。`8.50` 是训练器自己的 Step 8 搜索地板。
4. Step ≥ 8 时要有上一块门禁。当前 Robust 比上一块高出 `1e-6` 以上就停止。下降或持平都继续。Step 4 没有上一块，沿用 v16 的继续条件。
5. horizon 是 48。到点仍未 `PASSED` 就停止。

不做这些事：不恢复 v16；不把优势改回原始奖励差或固定 `0.10`；不试 `1.5e-5` 或 `2e-5`；不提高 β；不放宽 `5e-4`；不把通过线改松；不改服务器 `train/train_rl.py` 的停止表。服务器训练器已经会在局部过滤之后按配置里的 `advantage.mode` 计算优势。

## 设计

配置 `configs/train/qwen3_asr_rl_v17.yaml` 的 `train.max_steps` 是 48，这是训练器的 horizon。驱动传给进程的 `--max-steps` 只是当前块的结尾，依次为 4、8、12，直到 48。`grpo.advantage.mode` 是 `unit`，`local_max_relative` 是 `0.35`。

`followup_action` 是纯函数。命令行从 `pipeline_state.json` 读训练状态和累计奖励质量，从当前 `gate_step_<N>.json` 读贪心增量和 Robust 增量，Step ≥ 8 时再读上一块门禁的 Robust 增量。写出的 `switch_decision.json` 只含动作、原因、步数、训练状态、两项增量、门禁状态、上一块 Robust 增量和累计质量。动作词只有 `passed`、`continue`、`stop`。

驱动 `scripts/run_rl_pilot_v17.sh` 在目录已存在时拒绝新建。已有决定且动作不是 `continue` 时直接退出；动作是 `continue` 的重入直接拒绝。准备失败时只删除刚建的 v17 目录。四张卡上已有 `train_rl.py`、`parallel_inference.py` 或 held-out 打分时拒绝启动。

## 测试

`tests/test_rl_v17_decision.py` 锁住这些行为：门禁 PASSED 返回 `passed`；Robust 天花板、贪心转负、KL 停止；Step 4 没有上一块时继续；v16 Step 8 那种 Robust 从 `0.000321` 降到 `0.000164` 的情况继续；Robust 回升停止；质量 `16` 的 `BLOCKED_SEARCH` 继续，质量 `8.49` 停止，`8.50` 继续；质量够高但 Robust 回升仍然停止；Step 48 未通过时停止；Step 24 在 horizon 48 下继续。同一测试用已有的 `local_winner_index` 和 `winner_advantage` 确认近处样本被选中且单位优势是 1，并读取发出的 yaml，确认模式是 `unit`、局部阈值是 `0.35`、学习率是 `1.0e-5`、horizon 是 48。

`bash -n scripts/run_rl_pilot_v17.sh`。目录 README 合同测试要能看到新脚本和 `train/rl_v17_decision.py`。

启动前确认四张卡空闲，v16 的 `switch_decision.json` 仍是 `stop`，服务器训练器含 `local_max_relative`，复制的探针是 128 条 `GO_GRPO`。

## 验收

v17 目录里有自己的探针、损失日志、检查点和所评步的 `gate_step_<N>.json`。通过线不变：贪心 held-out 相对本次 Step 0 ≥ `+0.002`，Robust 六位小数 ≤ 0，clean ≤ `+0.02`，累计 clean ≤ `+0.025`，至少一个退化场景变好，有效输出 ≥ `0.95`。采样平均奖励不参与通过。`robust_edit_delta` 不单独投票。只有整道门禁 `PASSED` 才把权重合到 `rl_pilot_v17/merged_base`。

v16 的决定保持 `stop`。DPO Champion 的 `model.safetensors` 保持 4,076,190,936 字节，时间戳保持 2026-09-21 22:47 CST。

## 影响

v10 到 v16 继续只作诊断。发布底座在整道门禁 PASSED 之前仍是 DPO Champion。

## 结果

2026-10-01 13:13 CST，驱动在 Step 4 写出 `stop` 后退出。原因是贪心奖励低于 Step 0。`gate_step_4.json` 为 FAILED。没有 `merged_base`。

贪心 held-out `0.8773 → 0.8770`（`−0.0003`）。Robust macro `0.103942`，相对 DPO `+0.000359`（+6 次编辑）。变好的场景是 `en|dropout`。clean `−0.000109`。有效输出 `1.0`。未通过的是 `held_out_reward` 和 `robust_retention`。累计奖励质量 `7.432122`，获胜组 64，和 v16 Step 4 相同。四步裁剪前梯度是 `4.312`、`4.869`、`3.520`、`4.499`。Step 4 `raw_kl` 是 `3.9e-5`。

同一批局部样本在原始奖励差下，Step 4 的贪心增量是 0、Robust 增量是 `+0.000321`。单位优势把两条门禁都推得更差。不要再启动 `scripts/run_rl_pilot_v17.sh`。下一步回到原始奖励差，从 v16 Step 8 继续，见 `15_rl_v18_design.md`。
