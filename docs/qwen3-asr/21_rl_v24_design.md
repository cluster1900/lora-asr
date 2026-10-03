# RL v24：策略梯度只打在改动 token 上

| 项 | 值 |
| --- | --- |
| 日期 | 2026-10-01 |
| 状态 | 2026-10-02 00:19 CST 已结束。Step 4 门禁 FAILED。跟步写成 `continue` 之后，下一块在 Step 6 前退出。没有 `merged_base`。本目录不重跑。下一杆在 `22_rl_v25_design.md`。 |
| 运行名 | `rl_pilot_v24` |
| 前序 | v23 Step 12 门禁 FAILED，动作 `stop`。贪心 `−0.0004`，Robust `−0.000090`，`STOPPED_KL` |
| 发布底座 | DPO Champion：`/data/mega-asr/runs/dpo_pilot_v2/merged_base` |
| 配置 | `configs/train/qwen3_asr_rl_v24.yaml` |
| 入口 | `scripts/run_rl_pilot_v24.sh` |

## 背景

v16 用学习率 `1e-5`、原始奖励差和局部改正，从 Champion 走到 Step 8。贪心是 `+0.0004`，`raw_kl` 只有 `5e-5`，KL 预算还在。Robust 仍高于 DPO，多出来的 4 次编辑全部来自 `vitw_sample_042621_noise`，其余退化格子加总为 0。那句无关预测不在训练 rollout 里。v21 关掉音频投影之后，这句仍然出现，所以它不是音频 LoRA 单独写进去的。

v17 到 v23 接着试过的是单位优势、截断优势、把学习率改成 `5e-6`，以及继续已经靠近 KL 天花板的权重。这些都没有让整道门禁通过。v22 Step 8 进入下一步时，更新前的 `raw_kl` 已经是 `0.000476`。从那里再改步长，走不了几步就会碰到 `5e-4`，贪心奖励也被交回去。

这一轮不改这三样。长度 2 的反向现在对获胜句的每个 response token 都加同一份优势。局部改正里，多数 token 和贪心句是对齐的，真正要改的只有少数替换和插入。整句一起推，会把共享 LoRA 往「整句更像这条获胜样本」的方向带，held-out 上就会冒出那句无关文本。v24 让策略项只累积这些改动 token。KL 仍按整句的 Schulman K3 计算，停止线仍是 `raw_kl > 5e-4`。

## 范围

新目录 `/data/mega-asr/runs/rl_pilot_v24`。从 DPO Champion 新开，不恢复 v16、v21、v22 或 v23 的检查点。复制 v12 的 128 条探针，不复制任何一份损失日志，这样 Step 0 由这次运行自己重算。不写 Champion，不删除 v10 到 v23 的决定文件，不重跑那些目录。

学习率保持 `1e-5`。优势保持 `raw_gap`。局部阈值保持 `0.35`。`β` 保持 `0.04`。KL 天花板保持 `5e-4`。音频投影保持打开，目标数 199，和 v16 一样。跟步仍用 `train/rl_v16_decision.py`。第一块到 Step 4。贪心为负就停。Step 8 起 Robust 仍大于 0 就停。奖励非负且 Robust 已不高于 DPO 时继续，horizon 24。

不做这些事：不把学习率改成 `5e-6` 或 `2e-5`；不把优势改成 `unit`、`fixed` 或 `capped_gap`；不关掉音频投影；不提高 `β`；不放宽 `5e-4`；不改通过线；不把验证集那条噪声样本写进训练集。

缺省 `grpo.policy_token_mask` 是 `all`。旧配置没有这个键，服务器训练器不会给策略项加掩码。只有 v24 写成 `changes_only`。

## 设计

`train/rl_policy_mask.py` 的 `changed_token_mask` 对贪心 token id 和获胜 token id 做编辑距离对齐。获胜句上与贪心对齐的位置是 0，替换和插入是 1。删除没有获胜位置，所以不会给留下来的相同 token 打上 1。空的获胜序列返回空掩码。代价相同时优先认作匹配。

`scatter_keep` 把这条掩码放到移位后的 label 上。label 的第 0 位不参与预测。`ignore_index` 之外的后续 label 按顺序接收掩码。长度对不上就拒绝，不静默退回整句。

`policy_keep_ratio` 是掩码里 1 的比例。更新组才记入这一步的平均值，写到损失日志的 `policy_keep_ratio`。没有更新组时这个字段是空。

服务器 `compute_grpo_group_loss` 增加可选的 `policy_keep`。传入时，策略项乘 `token_mask * policy_keep`，KL 项仍只乘原来的 `token_mask`。不传时两条路径与补丁前相同。贪心那一行的优势本来就是 0，掩码保持全 1。获胜那一行在 `changes_only` 下使用 `changed_token_mask`。一次更新的掩码如果全是 0，训练器直接报错，避免把「没有 token 差异」当成一次更新。

配置标量：`learning_rate` `1.0e-5`，`max_steps` 24，`loss_reduction` `sequence_sum`，`mode` `raw_gap`，`local_max_relative` `0.35`，`beta` `0.04`，`max_raw_kl` `5.0e-4`，`policy_token_mask` `changes_only`，`train_audio_projections` `true`，目标数 199。

准备阶段核对：v16、v20、v21、v22、v23 的决定都是 `stop`；探针是 128 条 `GO_GRPO`；Champion 仍是 4,076,190,936 字节；服务器上的 `changed_token_mask([10, 11, 12, 13], [10, 99, 12, 13])` 返回 `[0.0, 1.0, 0.0, 0.0]`；训练器源码含 `changed_token_mask` 和 `policy_keep`。核对失败时只删除刚建的 v24 目录。

已有 v24 决定且动作不是 `continue` 时，再执行直接退出。动作是 `continue` 的重入直接拒绝。目录已存在但没有决定时也拒绝。

启动日志要出现 `policy_token_mask=changes_only` 和 `LoRA targets: 199 train_audio_projections=True`。损失日志里，有获胜组的步应写出小于 1 的 `policy_keep_ratio`。比例一直是 1，说明掩码没有把整句拆开。

## 测试

`tests/test_rl_policy_mask.py` 调用已发出的三个函数：相同 id 全 0；中间一个替换只留 1 个 1；插入加替换；删除不标记留下来的 token；空锚全 1；空获胜返回空；代价相同时优先匹配；`scatter_keep` 把掩码放到 response 后缀上，prompt 位置保持 0；长度不一致被拒绝；空掩码不能算比例。

`tests/test_rl_v24_contract.py` 读取发出的配置文本里的标量，确认学习率是 `1.0e-5`、掩码是 `changes_only`、优势是 `raw_gap`、音频投影打开、目标数 199。文件里不能出现 `5.0e-6`、`2.0e-5`、`mode: unit`、`mode: capped_gap`、`mode: fixed`、`train_audio_projections: false` 或 `policy_token_mask: all`。

同一测试调用已有的 v16 `followup_action`：Step 4 贪心为负时停止；Step 8 Robust 仍大于 0 时停止；Step 8 贪心为正且 Robust 低于 DPO 时继续；同一点若训练状态是 `STOPPED_KL` 则停止。驱动脚本必须保护 v10 到 v23 和 Champion，失败清理只指向 v24 目录，并且不恢复 v22 或 v23 的检查点。

`bash -n scripts/run_rl_pilot_v24.sh`。`scripts/README.md` 和 `train/README.md` 列出新文件。

## 验收

通过线不变：贪心 held-out 相对这次运行自己的 Step 0 ≥ `+0.002`，Robust 六位小数 ≤ 0，clean ≤ `+0.02`，累计 clean ≤ `+0.025`，至少一个退化场景变好，有效输出 ≥ `0.95`。只有整道门禁 `PASSED` 才把权重合到 `rl_pilot_v24/merged_base`。

v10 到 v23 的决定保持不动。DPO Champion 的大小和时间戳不变。`policy_keep_ratio` 在有更新的步上小于 1，才算这个杠杆真的生效。门禁没 PASSED 之前，不把这次运行说成通过。

## 结果

2026-10-01 22:59 CST 启动，驱动 PID 122215，PPID 1。准备阶段打印 `v24 baseline ok`。四张卡都是 `LoRA targets: 199 train_audio_projections=True`。启动行是 `global_step=0 -> chunk_end=4 horizon=24`，并且带 `policy_token_mask=changes_only`。`source.json` 记录：从 DPO Champion 新开，复制 v12 探针，`copied_loss_log` 为 false，学习率 `1e-5`，优势 `raw_gap`，`policy_token_mask` `changes_only`，音频投影打开，目标数 199，horizon 24。

这一轮实际用的参数：

| 项 | 值 |
| --- | --- |
| 起点 | DPO Champion，不恢复 v22 或 v23 |
| 学习率 | `1.0e-5`，`warmup_steps` 2，`constant`。日志是 Step 1 `5.00e-06`，Step 2 起 `1.00e-05` |
| 优势 | `raw_gap`。`local_max_relative` `0.35`，`min_improvement` `0.02` |
| KL | β `0.04`，停止线 `raw_kl > 5.0e-4`，整句 Schulman K3 |
| 策略掩码 | `changes_only`。策略项只累积替换和插入 token |
| LoRA | 音频投影打开，199 个目标（3 + 112 + 84），r=16，alpha=32 |
| 损失 | `sequence_sum`。优化器只吃退化语音。种子 `20260722` |
| 采样 | 组大小 4，未高出贪心 0.02 时再抽 8 条。温度 `1.0`，`top_p` `0.95`，`top_k` 50 |
| 步数 | horizon 24，每 4 步保存并评测。全局 batch 64 |
| 通过线 | 贪心 ≥ `+0.002`，Robust 六位小数 ≤ 0，clean ≤ `+0.02`，累计 clean ≤ `+0.025`，至少一个退化场景变好，有效输出 ≥ `0.95` |

Step 0 贪心是 `0.8773`（1,698 条，sha `b0701db2…`）。Step 1 到 Step 5 的 `policy_keep_ratio` 是 `0.1449`、`0.1661`、`0.144`、`0.1981`、`0.1378`，都小于 1，掩码把句子拆开了。这五步的 `raw_kl` 是 0、0、`8e-6`、`4e-6`、`2e-6`。Step 4 的累计奖励质量是 `7.462425`，获胜组 65。Step 5 写进了损失日志，质量变成 `8.520615`，获胜组 78。没有 Step 5 的门禁。

2026-10-02 00:13 CST 写成 `gate_step_4.json`，状态 FAILED。贪心 `0.8773 → 0.8774`（`+0.0001`）。Robust macro `0.1038`，相对 DPO 的 `0.103583` 是 `+0.000217`，编辑差 +5。clean macro `0.016692`，增量 `−2.5e-05`。有效输出 `1.0`，空输出 0。变好的场景只有 `zh|obstructed`。九项里 `held_out_reward` 和 `robust_retention` 失败，其余通过。`vitw_sample_042621_noise` 的 Step 4 预测仍是 “In the difficult moments, we recognize our thirst for fulfillment.”。

跟步读的是已有的 v16 规则。Step 4 贪心为正，Robust `+0.000217` 还没到 `0.0005`，步数也还没到 8，所以 `switch_decision.json` 写成 `continue`，原因是 “local-correction run can take another chunk”。下一块从 Step 4 续到 Step 8，启动行仍是 `policy_token_mask=changes_only`。

Step 5 打完之后，rank 2 在 `train_rl.py` 的 `length2_loss` 第 2692 行抛出 `RuntimeError: changes_only update has no changed response tokens`。这是 `changes_only` 的更新遇到一条获胜句，它的 token 掩码是空的或全是 0，也就是相对贪心句没有替换、没有插入。日志没有把那条样本的 token id 打出来。2026-10-02 00:19:45 CST，elastic 向其他 rank 发了 SIGTERM，torchrun 退出码 1。`pipeline_followup.json` 状态 FAILED，phase `chunk_8`，时间 00:19 CST。没有 Step 8 门禁，没有根目录 `gate.json`，没有 `merged_base`。Step 4 检查点还在，含 adapter、optimizer、scheduler、四份 RNG 和 `training_state.json`。

四张卡随后各 4 MiB，没有训练进程。Champion 仍是 4,076,190,936 字节，mtime unix `1790002044`。决定文件保留为 `continue`。再执行 `scripts/run_rl_pilot_v24.sh` 会因为已有 `continue` 而拒绝。不删除这份决定，不把 Step 4 说成候选。本目录不重跑。下一杆已在 2026-10-02 01:40 CST 启动，见 `22_rl_v25_design.md`。

## 影响

这一轮只改了策略梯度覆盖的 token。优势仍是原始奖励差，学习率仍是 `1e-5`，音频投影仍是 199 个。掩码在有更新的五步上都小于 1，所以杠杆确实作用在句子的一部分上。四步之后贪心只动了 `+0.0001`，离 `+0.002` 还差一个量级，Robust 往高的方向走了 `+0.000217`。同一句噪声幻觉还在。KL 预算到 Step 5 几乎没用。进程随后死在「获胜句没有可打梯度的 token」这条保护上，所以没有更后面的门禁。发布底座仍是 DPO Champion。v24 目录不重跑。下一条杠杆写在 `22_rl_v25_design.md`。
