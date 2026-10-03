# RL v22：从 v21 Step 4 再走四步

| 项 | 值 |
| --- | --- |
| 日期 | 2026-10-01 |
| 状态 | 2026-10-01 21:02 CST 在 Step 11 停止，动作 `stop`，门禁 FAILED。贪心 `−0.0004`，Robust `−0.000143`。训练状态 `STOPPED_KL`。没有 `merged_base` |
| 运行名 | `rl_pilot_v22` |
| 前序 | v21 Step 4 动作 `stop`。贪心 `−0.0002`，Robust `−0.000090`，只有奖励项失败 |
| 发布底座 | DPO Champion：`/data/mega-asr/runs/dpo_pilot_v2/merged_base` |
| 配置 | `configs/train/qwen3_asr_rl_v22.yaml` |
| 入口 | `scripts/run_rl_pilot_v22.sh` |

## 背景

v21 关掉 3 个音频投影后，Step 4 的 Robust macro 是 `0.103493`，比 DPO 的 `0.103583` 低 `0.000090`，净编辑是 0。clean、有效输出和退化场景也都过了。整道门禁只败在贪心 `−0.0002`。1,698 条上一条样本的平均奖励大约是 `0.0006`，所以这次停止落在一条样本以内。

v16 在同样的学习率和原始奖励差下，Step 4 的贪心增量是 0，Step 8 才到 `+0.0004`。v21 没有被允许看到 Step 8。它的 Step 4 权重是这一族里第一份 Robust 六位小数不高于 DPO、同时音频投影没有参与更新的检查点。

## 范围

新目录 `/data/mega-asr/runs/rl_pilot_v22`。只读恢复 `/data/mega-asr/runs/rl_pilot_v21/checkpoints/step_4`。复制 v21 的损失日志，使 Step 0 仍是 `0.8773`、奖励质量从 `7.462425` 继续。复制 v12 探针。不复制 v21 的门禁文件。不写 v21 目录，不删除 v21 的 `stop` 决定，不重跑 `scripts/run_rl_pilot_v21.sh`。

学习率、优势、局部阈值、`β`、`5e-4` 和 `train_audio_projections: false` 都与 v21 相同。跟步仍用 `train/rl_v16_decision.py`。第一块到 Step 8。Step 8 贪心仍为负，或 Robust 回到 0 以上，就停。两项都没碰到就继续，horizon 24。

不做这些事：不把学习率改回 `5e-6` 或升到 `2e-5`；不重新打开音频投影；不改优势；不提高 `β`；不放宽 `5e-4`；不改通过线。

## 设计

`configs/train/qwen3_asr_rl_v22.yaml` 与 v21 的训练杠杆相同：`learning_rate` `1.0e-5`，`max_steps` 24，`mode` `raw_gap`，`local_max_relative` `0.35`，`β` `0.04`，`max_raw_kl` `5.0e-4`，`train_audio_projections` `false`，目标数 196。

准备阶段核对：v21 决定是 `stop`；v21 Step 4 门禁的贪心增量是 `−0.0002`、Robust 增量小于 0；检查点有 `adapter/adapter_model.safetensors` 和 `optimizer.pt`；复制后的损失日志 Step 0 是 `0.8773`、累计质量是 `7.462425`；探针仍是 128 条 `GO_GRPO`；Champion 仍是 4,076,190,936 字节；服务器训练器仍调用 `filter_lora_targets`。核对失败时只删除刚建的 v22 目录。

恢复时调度器会带上 v21 Step 4 已经升到 `1e-5` 的学习率。日志应出现 `LoRA targets: 196`，以及从 v21 Step 4 恢复、下一块到 Step 8。

已有 v22 决定且动作不是 `continue` 时，再执行直接退出。动作是 `continue` 的重入直接拒绝。目录已存在但没有决定时也拒绝。

## 测试

`tests/test_rl_v22_contract.py` 读取发出的配置，确认学习率、模式、局部阈值、`β`、KL 上限、horizon、音频投影开关和目标数与 v21 相同。文件里不能出现 `5.0e-6`、`mode: unit`、`mode: capped_gap` 或 `train_audio_projections: true`。

同一测试调用已有的 v16 `followup_action`：Step 8 贪心仍为负时停止；Step 8 贪心为正且 Robust 仍低于 DPO 时继续；Step 8 的 Robust 回到 0 以上时停止。驱动脚本必须包含 v21 Step 4 检查点路径，并且失败清理只指向 v22 目录。

`bash -n scripts/run_rl_pilot_v22.sh`。`scripts/README.md` 列出新脚本。

启动前四张卡没有 `train_rl.py`、`parallel_inference.py` 或 held-out 打分。

## 验收

通过线不变：贪心 held-out 相对这次运行自己的 Step 0 ≥ `+0.002`，Robust 六位小数 ≤ 0，clean ≤ `+0.02`，累计 clean ≤ `+0.025`，至少一个退化场景变好，有效输出 ≥ `0.95`。只有整道门禁 `PASSED` 才把权重合到 `rl_pilot_v22/merged_base`。

启动日志必须仍是 196 个目标，并且恢复的是 v21 Step 4，不是 Champion 的 Step 0。

v21 的决定保持 `stop`。DPO Champion 的大小和时间戳不变。

## 结果

2026-10-01 20:13 CST 写成 `gate_step_8.json`，状态 FAILED。`switch_decision.json` 的动作是 `continue`，原因是局部改正运行可以再走一块。训练状态 `BLOCKED_TRANSFER`。贪心从 Step 0 的 `0.8773` 到 `0.8774`（`+0.0001`）。Robust macro `0.103545`，相对 DPO 的 `0.103583` 是 `−0.000038`，编辑差 `+1`。clean macro `0.016643`，增量 `−7.4e-05`。有效输出 `1.0`。变好的场景是 `en|echo`、`zh|obstructed`。九项检查里只有 `held_out_reward` 失败，`robust_retention` 通过。累计奖励质量 `11.838534`，获胜组 112。

Step 5 到 Step 8 的学习率都是 `1.00e-05`。`raw_kl` 依次是 `0.000021`、`0.000030`、`0.000048`、`0.000070`。裁剪前梯度范数依次是 `0.2066`、`0.2313`、`0.2272`、`0.2665`。

同一驱动在 20:13 CST 开始下一块，日志是 `global_step=8 -> chunk_end=12 horizon=24`。Step 9 的 `raw_kl` 升到 `0.000476`，Step 10 是 `0.000500`，Step 11 是 `0.000604`。停止条件是 `raw_kl > 5e-4`，所以 Step 11 返回 `STOPPED_KL`，没有走到 Step 12。Step 9 到 Step 11 的裁剪前梯度范数是 `0.3013`、`0.4563`、`0.5058`。

2026-10-01 21:02 CST 写成 `gate_step_11.json`，状态 FAILED。动作 `stop`，原因是贪心低于 Step 0。贪心 `0.8773 → 0.8769`（`−0.0004`）。Robust macro `0.103440`，相对 DPO `−0.000143`，编辑差 `−1`。clean macro `0.016518`，增量 `−0.000199`。有效输出 `1.0`。变好的场景是 `en|echo`、`en|recording`、`zh|obstructed`。未通过的仍只有 `held_out_reward`。累计奖励质量 `18.228518`，获胜组 159。没有 `merged_base`。四张卡随后空闲。不要删除这份 `stop` 决定，不要再启动 `scripts/run_rl_pilot_v22.sh`。Step 8 检查点仍在，贪心是这一轮的高点。下一轮从那里把学习率改成 `5e-6`，见 `20_rl_v23_design.md`。

## 影响

这一轮不改更新规则，只让已经通过 Robust 的解码器权重再训练四步。若 Step 8 的贪心仍为负，或 Robust 重新高于 DPO，驱动停止。不要把未通过的检查点说成候选。发布底座在整道门禁 PASSED 之前仍是 DPO Champion。
