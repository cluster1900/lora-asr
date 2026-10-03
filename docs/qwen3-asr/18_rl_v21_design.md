# RL v21：不训练音频投影，其余回到 v16

| 项 | 值 |
| --- | --- |
| 日期 | 2026-10-01 |
| 状态 | 2026-10-01 19:12 CST 前已在 Step 4 停止，动作 stop，门禁 FAILED。Robust 这一项通过。没有 `merged_base` |
| 运行名 | `rl_pilot_v21` |
| 前序 | v20 Step 4 动作 `stop`，贪心 `−0.0003`，Robust `+0.000202`，门禁 FAILED |
| 发布底座 | DPO Champion：`/data/mega-asr/runs/dpo_pilot_v2/merged_base` |
| 配置 | `configs/train/qwen3_asr_rl_v21.yaml` |
| 入口 | `scripts/run_rl_pilot_v21.sh` |

## 背景

v16 在学习率 `1e-5`、原始奖励差和局部改正下，Step 8 贪心 `+0.0004`，Robust 的 `+4` 次编辑全部来自 `vitw_sample_042621_noise`。v20 把学习率降到 `5e-6` 后，Step 4 贪心变成 `−0.0003`，同一条噪声样本仍从 9 次编辑变成 13 次，预测是同一句与参考无关的话。这句不在 v20 的 2,784 条 rollout 里，所以它不是被直接选成获胜样本的。

DPO Champion 对这条的预测还是另一句。四步很小的更新就把解码推进了这句。音频塔的 `conv_out`、`proj1`、`proj2` 会改整段语音嵌入，三个 Linear 就足以让一条难噪声样本跳到一句记忆文本。解码器上的局部改正不能解释“训练文本里从未出现、验证时整句替换”。

v20 已经否定“把学习率减半就能保住正贪心”。这一轮把学习率、优势、局部阈值、`β` 和 `5e-4` 放回 v16，只关掉三个音频投影的 LoRA。

## 范围

新目录 `/data/mega-asr/runs/rl_pilot_v21`。从 DPO Champion 的 Step 0 新开。只复制 v12 的 `search_probe.json`。不复制损失日志。不写 v16 到 v20，也不写 DPO Champion。不删除已有决定。不恢复 v20 的 Step 4。

跟步用已有的 `train/rl_v16_decision.py`。Step 8 起 Robust 仍大于 0 就停。贪心为负也停。horizon 24。第一块到 Step 4。

不做这些事：不把学习率留在 `5e-6` 或升到 `2e-5`；不改优势模式；不提高 `β`；不放宽 `5e-4`；不改通过线；不改 `decide_rl_stop`；不把验证集那一条加进训练集。

## 设计

`configs/train/qwen3_asr_rl_v21.yaml` 的 `train.learning_rate` 是 `1.0e-5`，`train.max_steps` 是 24。`grpo.advantage.mode` 是 `raw_gap`，`local_max_relative` 是 `0.35`，`β` 是 `0.04`，`max_raw_kl` 是 `5.0e-4`。`lora.train_audio_projections` 是 `false`。`expected_total_targets` 写成 196。

服务器 `train/train_rl.py` 仍用原来的正则找出 199 个 Linear。`train/rl_lora_targets.py` 的 `filter_lora_targets` 在标志为 false 时去掉以 `audio_tower.` 开头的名字。缺省标志仍是 true，旧配置的行为不变。注入前检查数量：开着音频投影必须是 199 且含有 `audio_tower.`，关掉必须是 196 且一个都不留。日志打出 `LoRA targets: 196 train_audio_projections=False`。

准备阶段核对：v12 探针是 128 条 `GO_GRPO`；v16、v19、v20 的决定都是 `stop`；Champion 仍是 4,076,190,936 字节；过滤函数会丢掉三个音频投影并保留解码器目标；服务器训练器已经调用 `filter_lora_targets`。任一核对失败时只删除刚建的 v21 目录。

已有 v21 决定且动作不是 `continue` 时，再执行直接退出。动作是 `continue` 的重入直接拒绝。目录已存在但没有决定时也拒绝。

## 测试

`tests/test_rl_v21_contract.py` 读取发出的配置文本，确认学习率是 `1.0e-5`、模式是 `raw_gap`、局部阈值是 `0.35`、`β` 是 `0.04`、`max_raw_kl` 是 `5.0e-4`、horizon 是 24、`train_audio_projections` 是 `false`、目标数是 196。文件里不能出现 `5.0e-6`、`mode: unit`、`mode: capped_gap` 或 `train_audio_projections: true`。

同一测试调用已有的 `filter_lora_targets`：false 时去掉 `audio_tower.`，true 时原样保留。并调用已有的 v16 `followup_action`：Step 4 贪心为负时停止；Step 8 的 Robust 仍大于 0 时停止。

`bash -n scripts/run_rl_pilot_v21.sh`。`scripts/README.md` 和 `train/README.md` 列出新文件。

启动前四张卡没有 `train_rl.py`、`parallel_inference.py` 或 held-out 打分。

## 验收

通过线不变：贪心 held-out 相对这次运行自己的 Step 0 ≥ `+0.002`，Robust 六位小数 ≤ 0，clean ≤ `+0.02`，累计 clean ≤ `+0.025`，至少一个退化场景变好，有效输出 ≥ `0.95`。只有整道门禁 `PASSED` 才把权重合到 `rl_pilot_v21/merged_base`。

启动日志必须出现 196 个目标。若仍是 199，这次运行作废，不把它的门禁当成音频投影已经关掉。

v16、v19、v20 的决定保持 `stop`。DPO Champion 的大小和时间戳不变。

## 影响

这一轮检验的是整句替换来自音频投影，而不是来自学习率。解码器 LoRA 仍可做局部改正。若 Step 4 贪心为负，或 Step 8 的 Robust 仍大于 0，驱动停止。不要把未通过的检查点说成候选。发布底座在整道门禁 PASSED 之前仍是 DPO Champion。

## 结果

2026-10-01 17:45 CST 启动，驱动 PID 113578。四张卡都是 `LoRA targets: 196 train_audio_projections=False`。Step 1 学习率 `5.00e-06`，Step 2 起 `1.00e-05`。19:12 CST 之前跟步已写成 `stop`，原因是贪心低于 Step 0。训练状态 `CHUNK_DONE`。

Step 0 贪心 `0.8773`。Step 4 贪心 `0.8771`（`−0.0002`）。Robust macro `0.103493`，相对 DPO `−0.000090`，`robust_edit_delta` 是 0。clean `−4.9e-05`，有效输出 `1.0`。变好的场景是 `en|distortion`、`zh|obstructed`。门禁里只有 `held_out_reward` 失败，`robust_retention` 通过。四步 `raw_kl` 最高 `1.1e-5`。`grad_norm` 是 `0.268`、`0.722`、`0.250`、`0.742`。获胜组 15、17、18、15，累计奖励质量 `7.462425`。

关掉音频投影没有去掉那句无关预测。`vitw_sample_042621_noise` 仍是 9 次编辑变成 13 次。退化集净编辑是 0，所以宏平均过了线。`−0.0002` 小于一条样本在 1,698 条上的平均奖励。

没有 `merged_base`。不要再启动 `scripts/run_rl_pilot_v21.sh`，也不要删掉这份 `stop` 决定。下一步只读恢复 Step 4，见 `19_rl_v22_design.md`。
