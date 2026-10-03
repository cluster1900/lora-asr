# RL v25：被删除的贪心 token 用负的奖励差

| 项 | 值 |
| --- | --- |
| 日期 | 2026-10-02 |
| 状态 | 2026-10-02 02:52 CST Step 4 整门 FAILED，动作 `stop`。贪心 `−0.0003`，Robust `+0.000209`。不重跑。 |
| 运行名 | `rl_pilot_v25` |
| 前序 | v24 Step 4 门禁 FAILED。贪心 `+0.0001`，Robust `+0.000217`。续块在 Step 5 之后退出 |
| 发布底座 | DPO Champion：`/data/mega-asr/runs/dpo_pilot_v2/merged_base` |
| 配置 | `configs/train/qwen3_asr_rl_v25.yaml` |
| 入口 | `scripts/run_rl_pilot_v25.sh` |

## 背景

v24 从 DPO Champion 新开，学习率 `1e-5`，优势 `raw_gap`，音频投影 199 个。策略项只打在获胜句相对贪心句的替换和插入 token 上。Step 1 到 Step 5 的 `policy_keep_ratio` 是 `0.1449`、`0.1661`、`0.144`、`0.1981`、`0.1378`，掩码确实把句子拆开了。`raw_kl` 最高 `8e-6`。

2026-10-02 00:13 CST 的 Step 4 门禁 FAILED：贪心 `0.8773 → 0.8774`（`+0.0001`），Robust `+0.000217`（+5 次编辑），clean `−2.5e-05`，有效输出 `1.0`，只有 `zh|obstructed` 变好。同一条噪声样本的预测仍是 “In the difficult moments, we recognize our thirst for fulfillment.”。同样的学习率、`raw_gap` 和 199 个目标，v16 用整句策略项时是贪心 `+0.0004`、Robust `+0.000164`。拿掉未改动 token 之后，两条都更差。

崩溃是另一件事。删除在获胜句上没有位置，留下来的相同 token 掩码是 0。Step 5 之后一条获胜句的掩码是空的或全是 0，rank 2 抛出 `changes_only update has no changed response tokens`，00:19 CST 退出。`switch_decision.json` 仍是 `continue`。

删除掉的 token 只存在于贪心句。策略项如果只看获胜句，这些 token 没有梯度，进程还会因为掩码全 0 退出。v25 把同一个 `raw_gap` 的负号打在被删掉的贪心 token 上。

## 范围

新目录 `/data/mega-asr/runs/rl_pilot_v25`。从 DPO Champion 新开，不恢复 v22、v23 或 v24 的检查点。复制 v12 的 128 条探针，不复制任何一份损失日志。不写 Champion。不删除 v10 到 v24 的决定文件。不重跑 `scripts/run_rl_pilot_v24.sh`，也不删除 v24 的 `continue` 决定。

学习率保持 `1.0e-5`。优势保持 `raw_gap`。局部阈值保持 `0.35`。`β` 保持 `0.04`。KL 天花板保持 `5.0e-4`。音频投影保持打开，目标数 199。跟步仍用 `train/rl_v16_decision.py`。第一块到 Step 4。horizon 24。

不做这些事：不把学习率改成 `5e-6` 或 `2e-5`；不把优势改成 `unit`、`fixed` 或 `capped_gap`；不关掉音频投影；不提高 `β`；不放宽 `5e-4`；不改通过线；不把验证集那条噪声样本写进训练集；不把 v24 的 `changes_only` 原样再跑一遍。

服务器训练器在 `policy_token_mask: signed_edits` 时调用 `signed_edit_loss_args`。`changes_only` 遇到全 0 掩码仍抛错，v24 不重跑。驱动在训练器源码里没有 `signed_edit_loss_args(` 时拒绝启动。四张卡空闲、v24 的决定仍是 `continue` 时，才往新目录启动。不覆盖已有的训练器备份，也不用本地 `train/train_rl.py` 替换服务器副本。

## 设计

`train/rl_policy_mask.py` 的 `token_edit_masks` 对贪心 token id 和获胜 token id 做编辑距离对齐。代价相同时优先认作匹配，和 v24 的 `changed_token_mask` 同一条回溯。

- 获胜句上，对齐位置是 0，替换和插入是 1。`changed_token_mask` 仍只返回这一条，旧测试不变。
- 贪心句上，被删除的位置是 1，其余是 0。删除没有获胜位置，所以不会给留下来的相同 token 打上 1。
- 回溯结束后，贪心句还没走完的前缀也算删除。

`signed_edit_loss_args(anchor_ids, winner_ids, winner_advantage)` 调用上面的函数，返回长度 2 损失要用的参数。`winner_advantage` 就是这一组已经选好的获胜优势。`raw_gap` 下它等于奖励差，贪心序列的原优势是 0。

- `action` 是 `skip` 时，序列优势是 `(0, 0)`，这一组不更新。
- `action` 是 `update` 时，序列优势是 `(-winner_advantage, winner_advantage)`。贪心掩码只保留删除位置，所以这些位置的策略项等于 `-winner_advantage`，其余贪心位置是 0。获胜掩码只保留替换和插入。只有删除时获胜掩码全是 0，函数仍返回 `update`。

策略项仍是 `-(优势 * log pi) * 掩码`。KL 仍乘整句 `token_mask`。这个组合和 `signed_edit_update` 的逐 token 优势一致。

`signed_edit_update(anchor_ids, winner_ids, raw_gap)` 返回 `SignedEditUpdate`：

- 替换、插入、删除都没有时，`action` 是 `skip`。两边的 token 优势都是 0。这一组不进入策略更新。
- 有任一编辑时，`action` 是 `update`。获胜 token 的优势是 `raw_gap` 乘获胜掩码：替换和插入等于 `raw_gap`，对齐位置是 0。被删除的贪心 token 优势是 `-raw_gap`，没被删除的贪心位置是 0。
- 只有删除时，`action` 仍是 `update`。获胜句优势全是 0。空的获胜句也是 `update`，贪心句每个位置都是 `-raw_gap`。函数不抛错。
- `winner_keep_ratio` 在获胜句非空时是掩码里 1 的比例。只有删除且获胜句为空时，这个字段是空，因为空掩码本来就不能算比例。

KL 仍按整句。贪心句原来的序列优势是 0；v25 只改被删除的那些位置，不把整句贪心优势改成负的。`raw_gap` 必须是有限数。

配置标量：`learning_rate` `1.0e-5`，`max_steps` 24，`loss_reduction` `sequence_sum`，`mode` `raw_gap`，`local_max_relative` `0.35`，`beta` `0.04`，`max_raw_kl` `5.0e-4`，`policy_token_mask` `signed_edits`，`train_audio_projections` `true`，目标数 199。

准备阶段核对：v16、v20、v21、v22、v23 的决定都是 `stop`；v24 的决定仍是 `continue`，并且不删除它；探针是 128 条 `GO_GRPO`；Champion 仍是 4,076,190,936 字节。服务器上的 `signed_edit_update([1, 2, 3], [1, 3], 0.25)` 必须是 `update`，获胜优势 `(0.0, 0.0)`，贪心优势 `(0.0, -0.25, 0.0)`。相同 token 的一对必须是 `skip`，优势全是 0。训练器源码含 `signed_edit_loss_args(`。核对失败时只删除刚建的 v25 目录。

已有 v25 决定且动作不是 `continue` 时，再执行直接退出。动作是 `continue` 的重入直接拒绝。目录已存在但没有决定时也拒绝。

## 测试

`tests/test_rl_policy_mask.py` 调用已发出的 `signed_edit_update` 和 `signed_edit_loss_args`。损失参数把序列优势乘上掩码之后，必须等于逐 token 优势。相同 id 的 `action` 是 `skip`，优势全是 0。中间一个替换只在该获胜 token 上得到正的 `raw_gap`，贪心优势全是 0。只有中间被删掉时，`action` 是 `update`，获胜优势全是 0，被删位置是 `-raw_gap`，函数不抛错。获胜句为空时，贪心句每个位置都是 `-raw_gap`。代价相同时优先匹配，被删的是贪心句前缀。贪心 `[1, 2, 3]` 对获胜 `[1, 9]` 的两种对齐代价相同，回溯先认末尾的替换，中间那个贪心 token 记成删除：获胜优势 `(0.0, 0.25)`，贪心优势 `(0.0, -0.25, 0.0)`。

`tests/test_rl_v25_contract.py` 读取发出的配置文本里的标量，确认学习率是 `1.0e-5`、掩码是 `signed_edits`、优势是 `raw_gap`、音频投影打开、目标数 199。文件里不能出现 `5.0e-6`、`2.0e-5`、`mode: unit`、`mode: capped_gap`、`mode: fixed`、`train_audio_projections: false`、`policy_token_mask: all` 或 `policy_token_mask: changes_only`。

同一测试调用 `signed_edit_update` 核对删除和相同对。驱动脚本必须保护 v10 到 v24 和 Champion，失败清理只指向 v25 目录，不删除 v24 的决定，不恢复 v22、v23 或 v24 的检查点，并且在训练器源码含 `signed_edit_loss_args(` 之后才允许往下走。

`bash -n scripts/run_rl_pilot_v25.sh`。`scripts/README.md` 和 `train/README.md` 列出新文件。

## 验收

通过线不变：贪心 held-out 相对这次运行自己的 Step 0 ≥ `+0.002`，Robust 六位小数 ≤ 0，clean ≤ `+0.02`，累计 clean ≤ `+0.025`，至少一个退化场景变好，有效输出 ≥ `0.95`。只有整道门禁 `PASSED` 才把权重合到 `rl_pilot_v25/merged_base`。

v10 到 v24 的决定保持不动。DPO Champion 的大小和时间戳不变。门禁没 PASSED 之前，不把这次运行说成通过。本地单测只说明纯函数行为。

## 结果

2026-10-02 01:40 CST 启动，驱动 PID 126231，PPID 1。准备阶段打印 `v25 baseline ok`。四张卡都是 `LoRA targets: 199 train_audio_projections=True`。启动行是 `global_step=0 -> chunk_end=4 horizon=24`，并且带 `policy_token_mask=signed_edits`。`source.json` 记录从 DPO Champion 新开，`copied_loss_log` 为 false，学习率 `1e-5`，优势 `raw_gap`，掩码 `signed_edits`，音频投影打开，目标数 199。

2026-10-02 02:52 CST，`gate_step_4.json` 写成 FAILED。训练状态 `CHUNK_DONE`。`switch_decision.json` 动作 `stop`，原因是贪心奖励低于 Step 0。

- 贪心 held-out `0.8773 → 0.8770`（`−0.0003`），1,698 行，SHA-256 `b0701db2ec3735222179e21fb4bf3c49c256d8da5d3a85cb7e61bfa6b7d9cd99`。
- Robust 增加 `0.000209`，编辑差 `+4`。pilot Robust macro `0.103792`，DPO 是 `0.103583`。
- clean 增加 `0.0`。有效输出 `1.0`。`zero_variance_ratio` `0.4648`。
- 变好：`en|distortion`、`en|dropout`。变差：`en|noise` `+0.002978`、`en|recording` `+0.003021`。
- `vitw_sample_042621_noise` 仍是 “In the difficult moments, we recognize our thirst for fulfillment.”，相对 DPO 多 4 次编辑。这句话不在 rollout 和 `pilot_rl.jsonl` 里，也不在 1,698 条奖励池里。
- Step 1 到 Step 4：`policy_keep_ratio` `0.1449`、`0.1661`、`0.1458`、`0.1981`；`raw_kl` 最高 `4e-6`；累计奖励质量 `7.432122`；获胜组 64；`grad_norm` `0.3029`、`0.7807`、`0.3546`、`1.3473`。
- 没有根目录 `gate.json`，没有 `merged_base`。Champion 仍是 4,076,190,936 字节，mtime unix `1790002044`。v24 的决定仍是 `continue`。03:10 CST 四张卡各 4 MiB。驱动已退出。

服务器训练器在补丁前备份为 `/data/mega-asr/logs/train_rl.py.bak-20261002-v25`（139,178 字节）。`changes_only` 的空掩码报错还在。补丁后的文件调用 `signed_edit_loss_args(`。

决定保留为 `stop`。再执行 `scripts/run_rl_pilot_v25.sh` 会读到这个决定并退出。下一轮改采样，见 `23_rl_v26_design.md`。

## 影响

v25 停在 Step 4。v24 不重跑。发布底座仍是 DPO Champion。通过线不变。这次整门是 FAILED，不能当作通过。
