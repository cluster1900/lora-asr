# RL v27：局部距离内加入参考文本

| 项 | 值 |
| --- | --- |
| 日期 | 2026-10-02 |
| 状态 | Step 7 门禁 FAILED，动作 `stop`。贪心 `+0.0002`，Robust `+0.000014`。训练状态 `STOPPED_KL`。没有 `merged_base`。 |
| 运行名 | `rl_pilot_v27` |
| 前序 | v26 Step 4 门禁 FAILED。贪心 `−0.0005`，Robust `+0.000359`，动作 `stop` |
| 发布底座 | DPO Champion：`/data/mega-asr/runs/dpo_pilot_v2/merged_base` |
| 配置 | `configs/train/qwen3_asr_rl_v27.yaml` |
| 入口 | `scripts/run_rl_pilot_v27.sh` |

## 背景

v26 从 DPO Champion 新开，学习率 `1e-5`，优势 `raw_gap`，掩码 `signed_edits`，199 个目标。唯一改动是优化器跳过 `noise` 和 `recording`，清单仍是完整的 `pilot_rl.jsonl`，日志是 `virtual_epoch_len=1483`。

2026-10-02 04:39 CST 的 Step 4 整门 FAILED，动作 `stop`，原因是贪心奖励低于 Step 0。1,698 条贪心 `0.8773 → 0.8768`（`−0.0005`）。2,867 条 Robust macro `0.103942`，相对 DPO `0.103583` 增加 `0.000359`（+6 次编辑）。clean `−2.5e-05`。有效输出 `1.0`。变好的场景数是 0。`degraded_improvement`、`robust_retention`、`held_out_reward` 失败，其余检查通过。没有根目录 `gate.json`，没有 `merged_base`。Champion 仍是 4,076,190,936 字节，mtime unix `1790002044`。

训练批上的平均奖励是 `0.9256`、`0.8926`、`0.9048`、`0.8378`，比 v25 的退化全集更高，但 held-out 更差。四步累计奖励质量 `6.459451`，获胜组 56。`raw_kl` 最高 `8e-6`。`policy_keep_ratio` 是 `0.1301`、`0.1385`、`0.1529`、`0.1676`。去掉那两个场景之后，v25 里仅有的 `en|distortion` 和 `en|dropout` 改善也没了。

所以下一轮回到完整的退化集，不改学习率、优势、掩码或音频投影。改的是候选集合：贪心句附近如果没有够好的采样，就把清单里的参考文本放进同一次局部选择。

## 范围

新目录 `/data/mega-asr/runs/rl_pilot_v27`。从 DPO Champion 新开。复制 v12 的 128 条探针，不复制损失日志。不写 Champion。不删除 v10 到 v26 的决定文件。不重跑 `scripts/run_rl_pilot_v26.sh`，也不删除 v26 的 `stop` 决定。v24 的 `continue` 决定保留。

学习率保持 `1.0e-5`。优势保持 `raw_gap`。局部阈值保持 `0.35`。`min_improvement` 保持 `0.02`。`β` 保持 `0.04`。KL 天花板保持 `5.0e-4`。音频投影保持打开，目标数 199。掩码保持 `signed_edits`。采样策略回到 `degraded`，噪声和录音都回到优化器。跟步仍用 `train/rl_v16_decision.py`。第一块到 Step 4。horizon 24。

不做这些事：不把学习率改成 `5e-6` 或 `2e-5`；不把优势改成 `unit`、`fixed` 或 `capped_gap`；不关掉音频投影；不提高 `β`；不放宽 `5e-4`；不改通过线；不把验证集那条噪声样本写进训练集；不再用 `degraded_skip_regressed`；不把 `changes_only` 再跑一遍。

`grpo.include_reference_candidate` 缺省是 false。旧配置没有这个键，服务器训练器不会给它们追加参考文本。v27 写成 true。驱动在训练器源码里没有 `append_reference_candidate(` 时拒绝启动。四张卡空闲、v26 的决定仍是 `stop` 时，才往新目录启动。不覆盖已有的训练器备份，也不用本地 `train/train_rl.py` 替换服务器副本。

## 设计

`train/rl_reference_candidate.py` 的 `append_reference_candidate` 接收候选文本、奖励、参考文本和参考奖励。参考文本规范化之后是空的，或者已经有一条候选规范化后相同，就原样返回。否则把参考文本和调用方给的奖励加到末尾。

参考奖励由服务器上的 `compute_sequence_reward(reference, reference, language)` 计算。预测和参考相同时，错误率是 0，空输出、重复、过长和 hallucination 惩罚都是 0，裁剪后的奖励是 `1.0`。准备阶段用一条英文和一条中文核对这个值。函数本身不把 `1.0` 写死。

服务器的 `select_anchored_training_pair` 只在 `local_max_relative` 有值、并且调用方传入参考文本和参考奖励时，先调用 `append_reference_candidate`，再调用原来的 `local_winner_index`。锚点仍是下标 0。距离超过 `0.35` 的参考文本得到 `no_improvement`，第二轮采样照旧。距离内且奖励差达到 `0.02` 时，这一对进入 `signed_edits`。token 完全相同则跳过，只有删除时仍更新。

预览和正式选择都传入参考文本。预览已经能更新时，不再抽第二轮。参考文本赢得更新、但还不在采样列表里时，训练器把它补进行级 rollout，`decode_mode` 写成 `reference`，这样 `trained` 指到这一行，组大小仍和行数一致。采样列表里已经有同一句时不重复追加。

清单文件仍是完整的 `pilot_rl.jsonl`。`sample_strategy: degraded` 选出 2,000 条退化语音，其中包含 noise 和 recording，不包含 clean。命令行 `--sample-strategy degraded` 与配置相同。这个名字本来就在 argparse 里，不再新增可选值。

## 测试

`tests/test_rl_reference_candidate.py` 调用发出的 `append_reference_candidate` 和 `local_winner_index`。近距离的参考文本成为更新；远距离的参考文本保持 `no_improvement`；规范化后已经存在的参考文本不重复追加；空参考文本不追加；长度不一致抛错。

`tests/test_rl_v27_contract.py` 读取发出的配置文本里的标量。学习率是 `1.0e-5`，采样策略是 `degraded`，`include_reference_candidate` 是 `true`，掩码是 `signed_edits`，优势是 `raw_gap`，音频投影打开，目标数 199。文件里不能出现 `5.0e-6`、`2.0e-5`、`mode: unit`、`mode: capped_gap`、`mode: fixed`、`train_audio_projections: false`、`policy_token_mask: all`、`policy_token_mask: changes_only`、`degraded_skip_regressed` 或 `include_reference_candidate: false`。

驱动脚本必须保护 v10 到 v26 和 Champion，失败清理只指向 v27 目录，要求 v26 的决定是 `stop`，v24 的决定仍是 `continue`，命令行采样策略是 `degraded`，并且准备阶段核对 2,000 条退化行里仍有 noise 和 recording。

`bash -n scripts/run_rl_pilot_v27.sh`。`scripts/README.md` 和 `train/README.md` 列出新文件。

准备阶段在服务器上调用 `compute_sequence_reward`，确认参考文本对自身的奖励是 `1.0`、四项惩罚是 0，并且训练器源码含 `append_reference_candidate(` 和 `reference_text=reference_text`。数量或奖励不对就只删除刚建的 v27 目录。

## 验收

通过线不变：贪心 held-out 相对这次运行自己的 Step 0 ≥ `+0.002`，Robust 六位小数 ≤ 0，clean ≤ `+0.02`，累计 clean ≤ `+0.025`，至少一个退化场景变好，有效输出 ≥ `0.95`。只有整道门禁 `PASSED` 才把权重合到 `rl_pilot_v27/merged_base`。

v10 到 v26 的决定保持不动。DPO Champion 的大小和时间戳不变。门禁没 PASSED 之前，不把这次运行说成通过。本地单测只说明候选追加和局部选择的行为。

启动后的日志要出现 `sample_strategy='degraded'`，并且 `virtual_epoch_len=2000`。四张卡都是 `LoRA targets: 199`。启动行带 `policy_token_mask=signed_edits` 和 `include_reference_candidate=True`。`source.json` 记录 `epoch_rows` 2000、`include_reference_candidate` 为 true、`copied_loss_log` 为 false。

## 结果

2026-10-02 05:13 CST 启动。驱动 PID 133766，PPID 1。准备阶段打印 `v27 baseline ok 2000`。日志是 `Loaded 3000 training samples (sample_strategy='degraded', virtual_epoch_len=2000)`，四张卡都是 `LoRA targets: 199 train_audio_projections=True`，启动行是 `policy_token_mask=signed_edits include_reference_candidate=True`。`source.json` 记录 `epoch_rows` 2000，`copied_loss_log` 为 false。Champion 仍是 4,076,190,936 字节，mtime unix `1790002044`。

Step 0 贪心是 `0.8773`。四步 `policy_loss` 是 `0.57672`、`0.51676`、`0.50131`、`0.57144`。`raw_kl` 是 0、0、约 0、`6.7e-5`。`policy_keep_ratio` 是 `0.2322`、`0.215`、`0.2307`、`0.2288`。获胜组 25、27、26、31，累计 109。累计奖励质量 `16.727412`。`second_round_ratio` 是 `0.6406`、`0.6562`、`0.625`、`0.5312`。学习率 Step 1 是 `5e-6`，之后是 `1e-5`。

2026-10-02 06:21 CST 写成 `gate_step_4.json`，时间戳 `2026-10-01T22:21:26.135436+00:00`。`gate_status` 是 FAILED。贪心 `0.8773 → 0.8774`（`held_out_reward_improvement` `+0.0001`）。Robust macro `0.103576`，相对 DPO `0.103583` 的 `robust_error_rate_increase` 是 `−0.000007`，`robust_edit_delta` 是 −1。clean 是 `0.0`。有效输出 `1.0`。变好的场景是 `en|distortion` 和 `en|recording`。未通过的只有 `held_out_reward`。`switch_decision.json` 当时的动作是 `continue`，原因是 local-correction run can take another chunk。没有根目录 `gate.json`，没有 `merged_base`。

续块写到了 Step 7。Step 5 到 Step 7 的 `raw_kl` 是 `9.7e-5`、`0.000322`、`0.001151`。Step 7 超过 `5e-4`，`pipeline_state.status` 是 `STOPPED_KL`。累计奖励质量 `27.297916`，获胜组 184。损失日志里的 Step 7 贪心是 `0.8775`。

2026-10-02 07:10 CST 写成 `gate_step_7.json`，时间戳 `2026-10-01T23:10:53.147886+00:00`。`gate_status` 是 FAILED。贪心 `0.8773 → 0.8775`（`held_out_reward_improvement` `+0.0002`）。Robust macro `0.103597`，相对 DPO `0.103583` 的 `robust_error_rate_increase` 是 `+0.000014`，`robust_edit_delta` 是 +2。clean 增量 `−0.000144`。有效输出 `1.0`。变好的场景是 `en|distortion`、`en|obstructed`、`zh|obstructed`。未通过的是 `held_out_reward` 和 `robust_retention`。`switch_decision.json` 的动作改成了 `stop`，原因是 trainer stop STOPPED_KL，`scored_step` 是 7。没有根目录 `gate.json`，没有 `merged_base`。

Step 4 到 Step 7，2,867 条总编辑从 2,132 降到 2,130。去掉 clean 之后，退化集净增 3 次编辑。增加集中在 `en|recording`（+5），主要是 `vitw_sample_195337_recording`（+3）和 `vitw_sample_452136_recording`（+2）。这两条在 Step 4 已经远离参考文本。`vitw_sample_042621_noise` 两步都还是 “In the difficult moments, we recognize our thirst for fulfillment.”，没有变。clean 英语 −3、中文 −2。把 0.35 收成 0.10 时，前四步奖励质量只从 `16.727412` 降到 `9.7963`，不是这一次 KL 越过天花板的原因。

服务器训练器备份：这次调用之前是 `/data/mega-asr/logs/train_rl.py.bak-20261002-v27`（141,032 字节）。更早的 v16、v19、v21、v23、v24、v25、v26 备份都还在。

## 影响

v27 的 Step 4 是这一族里，从 Champion 新开后第一份贪心为正、Robust 不高于 DPO 的检查点。奖励增量 `+0.0001` 仍低于 `+0.002`。参考文本把四步奖励质量抬到 `16.727412`，接着在 Step 7 越过 KL 天花板。Step 7 的贪心只再增加 `0.0001`，Robust 从 −1 次编辑变成 +2 次编辑。v27 的 `stop`、v26 的 `stop` 和 v24 的 `continue` 都保留。不从 Step 7 续训。发布底座仍是 DPO Champion。通过线不变。下一轮是 `25_rl_v28_design.md`。
