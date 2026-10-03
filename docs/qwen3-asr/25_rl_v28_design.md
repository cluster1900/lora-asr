# RL v28：从 v27 Step 4 把学习率写成 5e-6

| 项 | 值 |
| --- | --- |
| 日期 | 2026-10-02 |
| 状态 | 2026-10-02 07:28 CST 已启动，驱动 PID 139274。只读恢复 v27 Step 4，恢复后的学习率是 `5e-6`。第一块目标是 Step 8。 |
| 运行名 | `rl_pilot_v28` |
| 前序 | v27 Step 7 门禁 FAILED，动作 `stop`。贪心 `+0.0002`，Robust `+0.000014`，`STOPPED_KL` |
| 发布底座 | DPO Champion：`/data/mega-asr/runs/dpo_pilot_v2/merged_base` |
| 配置 | `configs/train/qwen3_asr_rl_v28.yaml` |
| 入口 | `scripts/run_rl_pilot_v28.sh` |

## 背景

v27 从 DPO Champion 新开，学习率在 warmup 之后是 `1e-5`，优势 `raw_gap`，掩码 `signed_edits`，199 个目标，采样策略 `degraded`。唯一的新杠杆是 `include_reference_candidate: true`：参考文本只有落在局部编辑距离 `0.35` 内才进入更新。

2026-10-02 06:21 CST 的 Step 4 整门 FAILED，只差奖励。贪心 `0.8773 → 0.8774`（`+0.0001`）。Robust 增量 `−0.000007`（−1 次编辑）。clean `0.0`。有效输出 `1.0`。变好的场景是 `en|distortion` 和 `en|recording`。当时动作是 `continue`。四步累计奖励质量 `16.727412`，获胜组 109。进入下一步时的 `raw_kl` 是 `9.7e-5`。

续到 Step 7 时，`raw_kl` 变成 `9.7e-5`、`0.000322`、`0.001151`。训练器在 Step 7 返回 `STOPPED_KL`。07:10 CST 的整门 FAILED：贪心 `0.8775`（`+0.0002`），Robust macro `0.103597`（相对 DPO `+0.000014`，+2 次编辑），clean `−0.000144`，有效输出 `1.0`。未通过的是 `held_out_reward` 和 `robust_retention`。动作改成 `stop`。没有根目录 `gate.json`，没有 `merged_base`。

Step 4 到 Step 7，退化集净增 3 次编辑。增加集中在两条 Step 4 已经远离参考文本的英文录音。那条固定的噪声幻觉句没有变。clean 略好。把局部阈值收成 `0.10` 只能把前四步质量从 `16.727412` 降到 `9.7963`，大多数参考获胜本来就在 `0.15` 以内。质量大是因为近距离参考文本的奖励差，不是因为放进了远距离改写。

裁剪前梯度范数在这几步多数大于 1。Adam 看到的是单位向量，步长由学习率决定。`1e-5` 下，参考文本更新在 Step 4 之后两步就越过 `5e-4`。v23 也把恢复后的学习率写成 `5e-6`，但它恢复的是已经到 `0.000476` 的检查点，随后仍 `STOPPED_KL`，贪心掉到 `−0.0004`。v27 Step 4 进入下一步时的 `raw_kl` 只有 `9.7e-5`，KL 预算还在。v20 的 `5e-6` 是从 Champion 新开、而且没有参考文本，四步就把贪心打负。这次不新开，也不从 Step 7 续。

## 范围

新目录 `/data/mega-asr/runs/rl_pilot_v28`。只读恢复 `/data/mega-asr/runs/rl_pilot_v27/checkpoints/step_4`。复制 v27 的探针，损失日志只保留 `global_step <= 4`。不写 v27 目录，不写 Champion。不删除 v27 的 `stop`、v26 的 `stop` 或 v24 的 `continue`。不重跑 `scripts/run_rl_pilot_v27.sh`。

只改学习率。`train.learning_rate` 是 `5.0e-6`，`train.apply_learning_rate_on_resume` 是 true。调度器加载 `scheduler.pt` 之后，服务器训练器调用已有的 `apply_configured_learning_rate`，把参数组和 `base_lrs` 写成这个值。优势仍是 `raw_gap`。局部阈值仍是 `0.35`。参考文本仍加入候选。掩码仍是 `signed_edits`。采样策略仍是 `degraded`，epoch 仍是 2,000 条。音频投影仍打开，目标数 199。`β` 仍是 `0.04`。KL 天花板仍是 `5.0e-4`。跟步仍用 `train/rl_v16_decision.py`。第一块到 Step 8。horizon 24。

不做这些事：不恢复 Step 7；不把学习率留在检查点里的旧值，也不改成 `2e-5`；不把优势改成 `unit`、`fixed` 或 `capped_gap`；不关掉音频投影；不提高 `β`；不放宽 `5e-4`；不改通过线；不收紧 `0.35`；不再用 `degraded_skip_regressed`；不把 `changes_only` 再跑一遍。

Step 4 的累计质量已经是 `16.727412`，高于 Step 8 的 `11.33`。若 Step 8 的贪心增量仍低于 `+0.001`，训练器会返回 `BLOCKED_TRANSFER`。跟步把这个状态当成可以继续，前提是贪心没有低于 Step 0，而且 Robust 没有高于 DPO。贪心为负、Robust 回到 0 以上，或训练器返回 `STOPPED_KL`，就停。

## 设计

配置和 v27 相同，除了学习率和恢复钩子。`apply_learning_rate_on_resume` 缺这个键时，旧运行的恢复行为不变。v28 写成 true。Step 4 的调度器已经过了 2 步 warmup，所以下一块的学习率日志应直接是 `5.00e-06`。

准备阶段核对：v27 决定是 `stop`，训练状态是 `STOPPED_KL`；Step 7 贪心增量是 `+0.0002`，Robust 增量大于 0；Step 4 贪心增量是 `+0.0001`，Robust 增量小于 0；检查点有适配器、`optimizer.pt` 和 `scheduler.pt`；截断后的日志最大步数是 4，Step 0 贪心是 `0.8773`，Step 4 贪心是 `0.8774`，最后的累计质量是 `16.727412`；探针仍是 128 条 `GO_GRPO`；退化 epoch 仍是 2,000 条，含 noise 和 recording；近距离参考文本会被选中，远距离参考文本是 `no_improvement`；服务器训练器同时含 `apply_configured_learning_rate` 和 `append_reference_candidate(`。核对失败时只删除刚建的 v28 目录。

驱动把 v10 到 v27 和 Champion 列进拒绝名单。失败清理只指向 v28。目录已经存在且决定是 `continue` 时拒绝再跑；决定是别的动作时直接退出。

## 测试

`tests/test_rl_v28_contract.py` 读取发出的配置文本里的标量。学习率是 `5.0e-6`，`apply_learning_rate_on_resume` 是 `true`，`include_reference_candidate` 是 `true`，采样策略是 `degraded`，掩码是 `signed_edits`，优势是 `raw_gap`，局部阈值是 `0.35`，音频投影打开，目标数 199。文件里不能出现 `learning_rate: 1.0e-5`、`learning_rate: 2.0e-5`、`mode: unit`、`mode: capped_gap`、`mode: fixed`、`train_audio_projections: false`、`policy_token_mask: all`、`policy_token_mask: changes_only`、`degraded_skip_regressed`、`include_reference_candidate: false` 或 `apply_learning_rate_on_resume: false`。

同一测试调用已有的 `followup_action`：Step 8 贪心为负时停止；Step 8 贪心为正、Robust 仍低于 DPO、状态是 `BLOCKED_TRANSFER` 时继续；Robust 回到 0 以上时停止；贪心非负但状态是 `STOPPED_KL` 时停止。

驱动脚本必须包含 v27 Step 4 检查点路径，不能把 Step 7 检查点当作恢复源，失败清理只指向 v28 目录，并且保留 v27 的 `stop` 与 v24 的 `continue`。

`bash -n scripts/run_rl_pilot_v28.sh`。`scripts/README.md` 列出这个驱动。训练器源码这次不改，沿用 v27 已经接上的参考候选和 v23 已经接上的恢复学习率钩子。

## 验收

通过线不变：贪心 held-out 相对这次运行自己的 Step 0 ≥ `+0.002`，Robust 六位小数 ≤ 0，clean ≤ `+0.02`，累计 clean ≤ `+0.025`，至少一个退化场景变好，有效输出 ≥ `0.95`。只有整道门禁 `PASSED` 才把权重合到 `rl_pilot_v28/merged_base`。

v10 到 v27 的决定保持不动。DPO Champion 的大小和时间戳不变。门禁没 PASSED 之前，不把这次运行说成通过。

启动后的日志要出现从 v27 Step 4 恢复，`configured_learning_rate=5.00e-06`，`sample_strategy='degraded'`，`virtual_epoch_len=2000`，`LoRA targets: 199`，`include_reference_candidate=True`。`source.json` 记录恢复检查点、学习率 `5.0e-6`、`apply_learning_rate_on_resume` 为 true、复制的奖励质量 `16.727412`。

## 影响

v28 回答的是：从符号都对、KL 还低的 Step 4 把步长减半，参考文本更新能不能在 `5e-4` 之内把贪心继续往上推，同时 Robust 保持不高于 DPO。Step 7 的权重已经越过天花板，并且 Robust 已经高于 DPO，不作为恢复点。发布底座仍是 DPO Champion。通过线不变。

## 结果

2026-10-02 07:28 CST 启动。驱动 PID 139274，PPID 1。准备阶段打印 `v28 baseline ok 2000 16.727412`。日志是 `configured_learning_rate=5.00e-06`、`Loaded 3000 training samples (sample_strategy='degraded', virtual_epoch_len=2000)`、四张卡 `LoRA targets: 199`，启动行是 `global_step=4 -> chunk_end=8`、`policy_token_mask=signed_edits include_reference_candidate=True`。`source.json` 的恢复检查点是 v27 Step 4，`copied_reward_mass` 是 `16.727412`，`copied_loss_log` 是 true。Champion 仍是 4,076,190,936 字节，mtime unix `1790002044`。v27 的决定仍是 `stop`，v24 的决定仍是 `continue`。没有 `merged_base`。Step 8 门禁还没写成。
