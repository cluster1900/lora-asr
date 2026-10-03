# RL v16：只强化局部改正，并允许正增量走过 Step 8

| 项 | 值 |
| --- | --- |
| 日期 | 2026-10-01 |
| 状态 | 2026-10-01 11:44 CST 在 Step 8 停止，动作 `stop`，门禁 FAILED。没有 `merged_base` |
| 运行名 | `rl_pilot_v16` |
| 前序 | v14 Step 8 `BLOCKED_TRANSFER`，贪心 `+0.0005`，Robust +3 次编辑，门禁 FAILED。v15 Step 4 贪心 `−0.0002`，动作 `stop` |
| 发布底座 | DPO Champion：`/data/mega-asr/runs/dpo_pilot_v2/merged_base` |
| 配置 | `configs/train/qwen3_asr_rl_v16.yaml` |
| 入口 | `scripts/run_rl_pilot_v16.sh` |

## 背景

v14 是目前贪心 held-out 最好的一点：`0.8773 → 0.8778`（`+0.0005`）。学习率 `1e-5`，优势是原始奖励差。Step 5–8 的裁剪前梯度已经低于 1，`raw_kl` 最高 `9.2e-5`。同一优势把学习率改成 `2e-5` 的 v15，在 Step 4 把贪心打回 `0.8771`。所以这一轮不提高学习率，不提高 β，也不放宽 `5e-4`。

两道未通过的门禁对应两件已经量到的事。

奖励增量只有通过线的四分之一。v12 的 Step 0 到 Step 4 是 `+0.0002`，v14 的 Step 4 到 Step 8 是再 `+0.0003`。Step 8 的累计奖励质量是 `18.255`，高于转移线 `11.33`，增量仍小于 `+0.001`，训练器返回 `BLOCKED_TRANSFER`。这个停止看的是「质量已经很高，但 held-out 还没到 `+0.001`」。原始奖励差的步子更小，四步大约只增加 `0.0003`，到不了这条线。驱动因此没有把仍为正的斜率再往下走。

Robust 多出来的编辑不是大面积变差。v14 相对 DPO 只有 18 条预测发生变化。其中 `vitw_sample_042621_noise` 一条就多了 4 次编辑，假设从 DPO 的错句换成了另一句无关的错句。v10、v11、v12、v14、v15 对这一条给出的是同一句错文本。其余格子加起来大约是 −1 次编辑。v10 能把 Robust 做成 −6 次编辑，是因为另外 15 条里有一批一词修正（`a`/`the`、`dr`/`doctor`、`for ever`、`漂亮`）。v14 把其中 11 条退回了 DPO 的文本。

v14 在 Step 5–8 新产生的 59 个获胜组里，和贪心文本的相对编辑距离小于 0.3 的有 43 个，奖励差质量 `4.039`；相对距离 ≥ 0.6 的整句替换有 9 个，质量 `2.282`，约占 30%。整句替换是「用另一句错文本换掉当前错文本」。一词修正才是 v10 把 Robust 拉回来的那种更新。

## 范围

新目录 `/data/mega-asr/runs/rl_pilot_v16`。从 DPO Champion 重新开始，不恢复 v14 的 optimizer。v14 的权重里已经有那条噪声错句。复制 v12 的 `search_probe.json`，不重跑探针。不写 v10、v11、v12、v13、v14、v15，也不写 DPO Champion。

学习率仍是 `1e-5`。优势仍是 `raw_gap`。`β=0.04`。`max_raw_kl=5e-4`。种子 `20260722`。温度、锚和 `min_improvement=0.02` 不变。

这一轮新增的只有两件事：

1. 获胜样本还必须是贪心文本的局部改正。英文按词、中文按字，归一化和 `evaluation/eval_wer.py` 一致。编辑距离不超过 `max(2, floor(0.35 × 较长的一边))`。更远的样本即使奖励差更大也不进反向；还有更近的合格样本时，用其中奖励差最大的一条。第一轮都不合格时，第二轮 8 条仍会抽。
2. 驱动在贪心增量 ≥ 0、Robust 增量 < `0.0005`、并且 Step 还小于 8 时，允许 Robust 暂时大于 0，继续到 Step 8。Step 8 及以后 Robust 仍大于 0 就停止。增量已经 ≥ 0 且 Robust ≤ 0 时，同一目录继续，horizon 是 24。训练器在 Step 8 和 Step 12 仍可能因为 `BLOCKED_TRANSFER` 退出进程；检查点会保存，驱动把这种退出当成可续的块，不另开学习率。

不做这些事：不把优势改回 1 或固定 `0.10`；不试 `1.5e-5` 或 `2e-5`；不提高 β；不放宽 `5e-4`；不把通过线改松；不把 v10 到 v15 的样本挖进 DPO。

## 设计

`train/rl_local_winner.py` 的 `local_winner_index` 返回获胜下标和状态。状态是 `update`、`no_improvement` 或 `identical`。平局保留更小的样本下标。`0.02` 的比较带 `1e-9` 绝对误差，和训练器里的 `_clears_min_improvement` 一致。

服务器 `train/train_rl.py` 的 `select_anchored_training_pair` 增加可选参数 `local_max_relative`。配置里没有这个键时，选择逻辑与现在相同。v16 的配置写 `local_max_relative: 0.35`。优势数值仍由 `winner_advantage(..., "raw_gap")` 计算。旧配置不传这个参数，不会改到已结束运行的行为；那些运行不再启动。

`train/rl_v16_decision.py` 在门禁写成之后选择动作：

| 条件 | 动作 |
| --- | --- |
| 整道门禁 PASSED | 导出到该 run 自己的 `merged_base`，停止 |
| Robust 增量 ≥ `0.0005` | 停止 |
| 贪心增量 < 0 | 停止 |
| `STOPPED_KL`、`STOPPED_REWARD_DROP`、`STOPPED_ROBUST`、`FAILED_ZERO_VARIANCE`、`BLOCKED_NO_WINNERS`、`BLOCKED_SEARCH` | 停止 |
| Step ≥ 8 且 Robust 增量 > 0 | 停止 |
| Step 已经到 24 | 停止 |
| `CHUNK_DONE` 或 `BLOCKED_TRANSFER`，且上面的停止都没碰上 | 同一 run 继续到下一块 |
| 其他状态 | 停止 |

下一块的终点是当前 Step 加 4，不超过 24。评测仍是贪心 1,698 条加上 2,867 条 `validation.jsonl`。

通过线不变：贪心 held-out 相对该 run 的 Step 0 ≥ `+0.002`；Robust 六位小数 ≤ 0；clean ≤ `+0.02`；累计 clean ≤ `+0.025`；至少一个退化场景变好；有效输出 ≥ `0.95`。只有整道 `gate.json` 为 PASSED 才导出。

按 v14 后四步大约 `+0.0003` 的斜率，24 步量级才接近 `+0.002`。局部过滤会拿掉大约 30% 的奖励差质量，斜率可能更慢。Step 8 的 Robust 若仍大于 0，这一轮会停，不会为了把奖励凑到通过线而继续推高 Robust。

## 测试

`tests/test_rl_local_winner.py` 覆盖一词改正入选、整句替换被拒、更近的样本胜过更远但奖励差更大的样本、平局取较小下标、中文一字改正、差距不足 `0.02`、差距刚好 `0.02`，以及奖励完全相同。

`tests/test_rl_v16_decision.py` 覆盖通过、Robust `0.0005`、奖励转负、KL、搜索停止、Step 4 的小幅 Robust 继续、Step 8 的 Robust 仍大于 0 停止、Step 8 的 Robust ≤ 0 在 `BLOCKED_TRANSFER` 下继续、horizon 停止，以及不认识的状态停止。

`bash -n scripts/run_rl_pilot_v16.sh`。目录 README 合同测试要能看到新脚本、`train/rl_local_winner.py` 和 `train/rl_v16_decision.py`。

启动前确认四张卡没有 `train_rl.py`、`parallel_inference.py` 或 held-out 打分进程。服务器训练器必须已经能读 `local_max_relative`。复制的探针必须是 `GO_GRPO`。

## 验收

v16 目录里有自己的探针、损失日志、检查点和所评步的 `gate_step_<N>.json`。v12 的 `switch_decision.json` 仍是 `stop`，v14 的仍是 `switch`，v15 的仍是 `stop`。这三个目录不新增 checkpoint。DPO Champion 的 `model.safetensors` 时间戳仍是 2026-09-21 22:47 CST。PASSED 之前没有 `merged_base`。

已有 `switch_decision.json` 且动作不是 `continue` 时，再执行驱动直接退出。动作是 `continue` 的重入直接拒绝，避免从半块又开一个训练进程。

## 影响

v10 到 v15 继续只作诊断。发布底座在整道门禁 PASSED 之前仍是 DPO Champion。v16 已经在 Step 8 停止。局部过滤没有在 8 步内把 Robust 拉回 ≤ 0。不要再启动 `scripts/run_rl_pilot_v16.sh`。

## 结果

2026-10-01 11:44 CST，驱动写完 Step 8 的决定后退出。动作 `stop`，原因是 Robust 仍高于 DPO。`gate_step_8.json` 为 FAILED。没有 `merged_base`。DPO Champion 的 `model.safetensors` 仍是 4,076,190,936 字节。

| 项 | Step 4 | Step 8 |
| --- | --- | --- |
| 训练状态 | `CHUNK_DONE`，动作 `continue` | `BLOCKED_TRANSFER`，动作 `stop` |
| 贪心 held-out | `0.8773`，增量 `0` | `0.8777`，增量 `+0.0004` |
| Robust 增量 | `+0.000321`（+7 次编辑） | `+0.000164`（+4 次编辑） |
| Robust macro | `0.103904` | `0.103747` |
| clean 增量 | `−7.4e-05` | `−7.4e-05` |
| 变好的退化场景 | `zh\|obstructed` | `en\|echo`、`zh\|obstructed` |
| 有效输出 | `1.0` | `1.0` |
| 累计奖励质量 / 获胜组 | `7.432122` / 64 | `11.724898` / 110 |
| 该步 `raw_kl` | `1.5e-5` | `5.0e-5` |
| 该步 `grad_norm` | `1.308` | `0.351` |

Step 5–7 的 `grad_norm` 是 `0.273`、`0.278`、`0.303`，都低于裁剪线 1。`raw_kl` 最高是 Step 8 的 `5.0e-5`。学习率、β 和 `5e-4` 都没有成为停止原因。

Step 8 的 +4 次编辑全部来自 `en|noise` `vitw_sample_042621_noise`。其余退化格子加总是 0：`en|echo` −4，`en|dropout` +2，`en|distortion` +2，`en|recording` +1，`zh|distortion` +1，`zh|obstructed` −2。这一条噪声样本从 v10 到 v16 都写成同一句无关错文本，训练集 `pilot_rl.jsonl` 里没有这句参考文本。

和 v14 Step 8 比，贪心增量接近（v14 是 `+0.0005`），Robust 增量更小（v14 是 `+0.000247`、+3 次编辑），奖励质量也更低（v14 是 `18.255`）。过滤把其余格子收到了和 DPO 持平，但没有抵消那条固定的 +4。下一轮改的是局部获胜样本的优势尺度，见 `14_rl_v17_design.md`。
