# RL v20：学习率降到 5e-6，从 Champion 新开

| 项 | 值 |
| --- | --- |
| 日期 | 2026-10-01 |
| 状态 | 2026-10-01 17:11 CST 在 Step 4 停止，动作 stop，门禁 FAILED。没有 `merged_base` |
| 运行名 | `rl_pilot_v20` |
| 前序 | v19 Step 12 动作 `stop`，贪心 `−0.0001`，Robust `+0.000014`，`raw_kl` `0.000640`，门禁 FAILED |
| 发布底座 | DPO Champion：`/data/mega-asr/runs/dpo_pilot_v2/merged_base` |
| 配置 | `configs/train/qwen3_asr_rl_v20.yaml` |
| 入口 | `scripts/run_rl_pilot_v20.sh` |

## 背景

v16 在学习率 `1e-5`、原始奖励差和局部改正下，Step 8 的贪心增量是 `+0.0004`，Robust 是 `+0.000164`。v18 从这份权重继续，两步后 `raw_kl` 越过 `5e-4`，贪心退到 `+0.0001`。v19 不改学习率，只把奖励差截断在 `0.05`。Step 9 的梯度从 `0.385` 降到 `0.159`，Step 12 的 `raw_kl` 仍是 `0.000640`，贪心变成 `−0.0001`，Robust 停在 `+0.000014`。

优势缩放没有换掉更新方向。v19 的小步仍然从 Step 8 的权重往贪心更低的一侧走。那份权重在下一批数据上的 `raw_kl` 已经是 `0.000402`，是 `1e-5` 走完八步之后留下的。再从它恢复，只是把同一段路走慢一点。

这一轮改学习率，不改优势。从 DPO Champion 重新开始，学习率用 `5e-6`，优势仍是没有截断的原始奖励差，局部阈值仍是 `0.35`，`β` 仍是 `0.04`，KL 上限仍是 `5e-4`。不加载 v16、v18 或 v19 的优化器，避免调度器把学习率恢复成 `1e-5`。

v16 的八步一共拿到 `+0.0004`，大约每步 `+0.00005`。学习率减半之后，如果每步斜率也减半并且能保持下去，80 步才碰到 `+0.002`。这个斜率没有被证明能保持。贪心转负、`raw_kl` 超过 `5e-4`，或 Robust 比上一块回升，驱动都停止。

## 范围

新目录 `/data/mega-asr/runs/rl_pilot_v20`。只复制 v12 的 `search_probe.json`。不复制任何损失日志，Step 0 由这次运行自己重算。不写 v16、v17、v18、v19 或 DPO Champion。不删除已有的 `switch_decision.json`。

跟步仍用 `train/rl_v19_decision.py` 里的 v17 动作表。上一块 Robust 取当前步之前最近一份门禁。horizon 是 80。第一块到 Step 4，之后每次加 4。

不做这些事：不把优势改成 1、固定 `0.10` 或再截一档；不把学习率升到 `1e-5` 以上；不提高 `β`；不放宽 `5e-4`；不改通过线；不改 `decide_rl_stop`。

## 设计

`configs/train/qwen3_asr_rl_v20.yaml` 的 `train.learning_rate` 是 `5.0e-6`，`train.max_steps` 是 80。`grpo.advantage.mode` 是 `raw_gap`，`local_max_relative` 是 `0.35`。服务器训练器在新建运行时用配置里的学习率构造 AdamW。前两步是 warmup，日志里的第一步学习率是 `2.5e-6`，之后是 `5e-6`。

准备阶段核对：v12 探针是 128 条 `GO_GRPO`；v16 和 v19 的决定都是 `stop`；v18 的跟步状态不是 `TRAINING`；Champion 仍是 4,076,190,936 字节；局部获胜函数仍会拒绝那条过远的替换。任一核对失败时只删除刚建的 v20 目录。

已有 v20 决定且动作不是 `continue` 时，再执行直接退出。动作是 `continue` 的重入直接拒绝。目录已存在但没有决定时也拒绝。

## 测试

`tests/test_rl_v20_contract.py` 读取发出的配置文本里的标量，确认学习率是 `5.0e-6`、模式是 `raw_gap`、局部阈值是 `0.35`、`β` 是 `0.04`、`max_raw_kl` 是 `5.0e-4`、horizon 是 80。文件里不能出现 `1.0e-5`、`mode: unit` 或 `mode: capped_gap`。同一测试调用已有的 `followup_action`：Step 4 贪心为正时继续；贪心为负时停止。

`bash -n scripts/run_rl_pilot_v20.sh`。`scripts/README.md` 列出新脚本。

启动前四张卡没有 `train_rl.py`、`parallel_inference.py` 或 held-out 打分。

## 验收

通过线不变：贪心 held-out 相对这次运行自己的 Step 0 ≥ `+0.002`，Robust 六位小数 ≤ 0，clean ≤ `+0.02`，累计 clean ≤ `+0.025`，至少一个退化场景变好，有效输出 ≥ `0.95`。只有整道门禁 `PASSED` 才把权重合到 `rl_pilot_v20/merged_base`。

v16、v17、v19 的决定保持 `stop`。DPO Champion 的大小和时间戳不变。

## 影响

v19 已经说明，从 Step 8 把步长缩小并不能把贪心保持在正的一侧。v20 因此不恢复那份权重。它只检验更小的学习率从 Champion 出发时，正的贪心斜率能不能维持到 `+0.002`。若 Step 4 贪心已经为负，或中途越过 `5e-4`，本轮停止，不把未通过的检查点说成候选。发布底座在整道门禁 PASSED 之前仍是 DPO Champion。

## 结果

2026-10-01 16:00 CST 启动，驱动 PID 110855。17:11 CST 跟步写成 `stop`，原因是贪心低于 Step 0。训练状态是 `CHUNK_DONE`，不是 KL 停止。

Step 0 贪心奖励 `0.8773`。Step 1 日志学习率 `2.50e-06`，Step 2 起是 `5.00e-06`。四步 `raw_kl` 是 0、0、`4e-6`、`3e-6`。`grad_norm` 是 `0.328`、`0.941`、`0.351`、`1.319`。获胜组 15、17、17、14，累计奖励质量 `7.355199`。

Step 4 门禁 FAILED。贪心 `0.8773 → 0.8770`（`−0.0003`）。Robust `+0.000202`（`robust_edit_delta` 3），clean `−4.9e-05`，有效输出 `1.0`。变好的场景是 `en|dropout`、`en|echo`。未通过的是 `held_out_reward` 和 `robust_retention`。

退化集上只有 6 条预测和 DPO 不同，净编辑 `+3`。`vitw_sample_042621_noise` 从 9 次编辑变成 13 次，预测又是 “In the difficult moments, we recognize our thirst for fulfillment.”。这句不在 2,784 条 rollout 里。其余退化格子净 `−1`。把学习率减半没有避开这条句子，也没有维持 v16 的正贪心。

没有 `merged_base`。不要再启动 `scripts/run_rl_pilot_v20.sh`。不要从 Step 4 恢复。下一轮不改学习率、优势或 KL 上限，见 `18_rl_v21_design.md`。
