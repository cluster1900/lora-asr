# RL v23：从 v22 Step 8 把学习率降到 5e-6

| 项 | 值 |
| --- | --- |
| 日期 | 2026-10-01 |
| 状态 | 2026-10-01 22:15 CST Step 12 门禁 FAILED，动作 `stop`。贪心 `−0.0004`，Robust `−0.000090`。训练状态 `STOPPED_KL`。没有 `merged_base`。不再开下一轮 |
| 运行名 | `rl_pilot_v23` |
| 前序 | v22 Step 11 动作 `stop`。贪心 `−0.0004`，Robust `−0.000143`，`STOPPED_KL`。Step 8 仍是 `+0.0001` 且 Robust 低于 DPO |
| 发布底座 | DPO Champion：`/data/mega-asr/runs/dpo_pilot_v2/merged_base` |
| 配置 | `configs/train/qwen3_asr_rl_v23.yaml` |
| 入口 | `scripts/run_rl_pilot_v23.sh` |

## 背景

v22 从 v21 Step 4 续到 Step 8 时，贪心从 `−0.0002` 变成 `+0.0001`，Robust 仍是 `−0.000038`。这是关掉音频投影之后，第一份两项符号都对的检查点。同一学习率再走三步，`raw_kl` 从 Step 8 的 `0.000070` 升到 Step 9 的 `0.000476`、Step 11 的 `0.000604`。训练器在 Step 11 返回 `STOPPED_KL`。贪心变成 `−0.0004`，Robust 进一步降到 `−0.000143`。

Step 5 到 Step 8 的裁剪前梯度范数不超过 `0.27`，`raw_kl` 不超过 `7e-5`。Step 9 起范数和 KL 一起变大，held-out 贪心奖励往回走。v20 的 `5e-6` 是从 Champion 新开，四步就把贪心打成 `−0.0003`，不能当成这一步的结果。这一步测的是：已经同时满足正贪心和 Robust ≤ 0 的 Step 8 权重，在更短的步长下会不会停在 KL 天花板以内。

## 范围

新目录 `/data/mega-asr/runs/rl_pilot_v23`。只读恢复 `/data/mega-asr/runs/rl_pilot_v22/checkpoints/step_8`，包括适配器、优化器和调度器。不恢复 Step 11。复制 v22 的探针。损失日志只保留 `global_step <= 8` 的行，使 Step 0 仍是 `0.8773`、奖励质量从 `11.838534` 继续，并且不把 Step 11 的负奖励写进新日志。不复制 v22 的门禁文件。不写 v22 目录，不删除 v22 的 `stop` 决定，不重跑 `scripts/run_rl_pilot_v22.sh`。

只改学习率。恢复之后写成 `5e-6`。优势仍是 `raw_gap`，局部阈值 `0.35`，`β` `0.04`，KL 天花板 `5e-4`，音频投影仍然关闭，目标数 196。跟步仍用 `train/rl_v16_decision.py`。第一块到 Step 12。贪心为负、Robust 回到 0 以上，或训练器返回 `STOPPED_KL`，就停。否则继续，horizon 24。

不做这些事：不把学习率留在 `1e-5`，也不升到 `2e-5`；不重新打开音频投影；不改优势；不提高 `β`；不放宽 `5e-4`；不改通过线；不从 Champion 重开。

调度器的 `load_state_dict` 会把学习率盖回检查点里的 `1e-5`。因此 `train.apply_learning_rate_on_resume: true` 时，训练器在加载 `scheduler.pt` 之后调用 `apply_configured_learning_rate`，把优化器参数组和 `base_lrs` 写成 YAML 里的学习率。缺这个键时不调用，旧配置的恢复行为不变。Step 8 的调度器已经过了 2 步 warmup，所以下一小步的日志应直接是 `5.00e-06`，而不是再打印半个 warmup。

## 设计

`configs/train/qwen3_asr_rl_v23.yaml` 的训练杠杆：`learning_rate` `5.0e-6`，`apply_learning_rate_on_resume` `true`，`max_steps` 24，`mode` `raw_gap`，`local_max_relative` `0.35`，`β` `0.04`，`max_raw_kl` `5.0e-4`，`train_audio_projections` `false`，目标数 196。

`train/rl_resume_lr.py` 的 `apply_configured_learning_rate` 接收优化器、调度器和学习率。学习率必须是正的有限数。它改每个参数组的 `lr`，若组里已有 `initial_lr` 也一起改。调度器有 `base_lrs` 时按同样的值重写。不改 `last_epoch`。

准备阶段核对：v22 决定是 `stop`，贪心增量是 `−0.0004`，Robust 增量小于 0；v22 Step 8 门禁的贪心增量是 `+0.0001`，Robust 增量小于 0；检查点有适配器、`optimizer.pt` 和 `scheduler.pt`；截断后的日志最大步数是 8，Step 0 是 `0.8773`，Step 8 贪心是 `0.8774`，最后的累计质量是 `11.838534`；探针仍是 128 条 `GO_GRPO`；Champion 仍是 4,076,190,936 字节；服务器训练器含 `apply_configured_learning_rate`。核对失败时只删除刚建的 v23 目录。

已有 v23 决定且动作不是 `continue` 时，再执行直接退出。动作是 `continue` 的重入直接拒绝。目录已存在但没有决定时也拒绝。

## 测试

`tests/test_rl_resume_lr.py` 用假的优化器和调度器调用已发出的 `apply_configured_learning_rate`：`1e-5` 被写成 `5e-6`，`base_lrs` 和 `initial_lr` 一起改；没有调度器时仍改优化器；非正学习率被拒绝。

`tests/test_rl_v23_contract.py` 读取发出的配置，确认学习率是 `5.0e-6`、恢复后写回开关为 `true`，其余杠杆与 v22 相同。文件里不能出现 `learning_rate: 1.0e-5`、`mode: unit`、`mode: capped_gap`、`train_audio_projections: true` 或把该开关写成 `false`。

同一测试调用已有的 v16 `followup_action`：Step 12 贪心为负时停止；Step 12 贪心为正、Robust 仍低于 DPO 且状态可继续时继续；Robust 回到 0 以上时停止；贪心非负但状态是 `STOPPED_KL` 时停止。驱动脚本必须包含 v22 Step 8 检查点路径，并且失败清理只指向 v23 目录。

`bash -n scripts/run_rl_pilot_v23.sh`。`scripts/README.md` 和 `train/README.md` 列出新文件。

启动前四张卡没有 `train_rl.py`、`parallel_inference.py` 或 held-out 打分。启动日志必须出现 `configured_learning_rate=5.00e-06`、`LoRA targets: 196`，以及从 v22 Step 8 恢复、下一块到 Step 12。

## 验收

通过线不变：贪心 held-out 相对这次运行自己的 Step 0 ≥ `+0.002`，Robust 六位小数 ≤ 0，clean ≤ `+0.02`，累计 clean ≤ `+0.025`，至少一个退化场景变好，有效输出 ≥ `0.95`。只有整道门禁 `PASSED` 才把权重合到 `rl_pilot_v23/merged_base`。

v22 的决定保持 `stop`。DPO Champion 的大小和时间戳不变。

## 结果

2026-10-01 21:19 CST 启动，驱动 PID 119458。日志有 `configured_learning_rate=5.00e-06`、`LoRA targets: 196`，以及从 v22 Step 8 恢复、`global_step=8 -> chunk_end=12`。

Step 9 的 `raw_kl` 是 `0.000476`，和 v22 Step 9 相同，因为 KL 在更新前按进入该步的权重计算。Step 10 降到 `0.000427`（v22 是 `0.000500`），Step 11 是 `0.000446`，Step 12 是 `0.000600`。训练器在 Step 12 返回 `STOPPED_KL`。贪心 `0.8773 → 0.8769`（`−0.0004`），和 v22 Step 11 的奖励相同。累计质量 `20.112178`。

2026-10-01 22:15 CST 写成 `gate_step_12.json`，状态 FAILED。动作 `stop`，原因是贪心低于 Step 0。Robust macro `0.103493`，相对 DPO 的 `0.103583` 是 `−0.000090`，编辑差 0。clean macro `0.016593`，增量 `−0.000124`。有效输出 `1.0`。变好的场景是 `en|echo`、`zh|obstructed`。九项检查里只有 `held_out_reward` 失败，`robust_retention` 通过。没有 `merged_base`。四张卡随后空闲。Champion 仍是 4,076,190,936 字节，mtime unix `1790002044`。

这一轮到这里停止。不删除 v23 目录，不重跑 `scripts/run_rl_pilot_v23.sh`。后面换了一个不同的杠杆，写在 `21_rl_v24_design.md`。v23 的结果不变。

## 影响

这一轮只缩短从 Step 8 出发的步长。Step 8 的 `+0.0001` 离 `+0.002` 还差一个量级，更短的步长不一定补上这个差距。若 Step 12 的贪心为负或 KL 再次越线，驱动停止，不把未通过的检查点说成候选。发布底座在整道门禁 PASSED 之前仍是 DPO Champion。
