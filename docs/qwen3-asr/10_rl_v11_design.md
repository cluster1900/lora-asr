# RL v11：同一套锚定 GRPO，学习率 2e-5

| 项 | 值 |
| --- | --- |
| 日期 | 2026-09-29 |
| 状态 | 已执行。训练在 Step 7 停止（`STOPPED_KL`）。`gate_step_7.json` 为 FAILED，只未通过 `held_out_reward` |
| 运行名 | `rl_pilot_v11` |
| 前序 | `rl_pilot_v10`，Step 8 `BLOCKED_TRANSFER`，见 `09_rl_v10_design.md` |
| 发布底座 | DPO Champion：`/data/mega-asr/runs/dpo_pilot_v2/merged_base` |
| 配置 | `configs/train/qwen3_asr_rl_v11.yaml` |
| 入口 | `scripts/run_rl_pilot_v11.sh`、`scripts/run_rl_pilot_v11_recover.sh`、`scripts/score_rl_pilot_v11_checkpoint.sh` |

## 背景

v10 把探针奖励质量做到 `17.10`，Step 8 的累计质量 `18.12`，已经高于 12 步地板 `17.0`。贪心 held-out 仍从 `0.8773` 降到 `0.8771`。LoRA B 最大绝对值停在 `6.56e-5`，`raw_kl` 为 `7.1e-5`。梯度和优势在 `clip_grad_norm_(max_norm=1.0)` 之后被 AdamW 按学习率缩放，所以长度 2、优势固定为 1 并没有放大步长。v8 在无锚点、`2e-5` 时 LoRA B 能到 `2.10e-4`，转写会变，但 Robust 变差。v11 要测的是：同一套贪心锚，在能够推动 `raw_kl` 的学习率上，更好的采样会不会转到贪心解码。测量已经做完，结果在文末「收口」：v10 在 `raw_kl=7.1e-5` 时 2,867 条已是 −6 次编辑；v11 把 KL 推过 `5e-4` 后编辑数仍是 −6，贪心奖励是 `−0.0006`。

## 范围

这次运行是从 DPO Champion 重新开始的锚定 GRPO，已经结束。学习率用的是 `2e-5`，warmup 2 步后保持常数。锚、β=`0.04`、温度 `1.0` / `top_p=0.95` / `top_k=50`、两轮采样、定长 2 条反向、奖励质量地板和停止表沿用 v10。

不做这些事：不从 v10 或本次 KL 停止之后恢复；不再提高学习率；不把已经更新过的话语送进 DPO；不新增人工标注；不写入 DPO Champion；不声称达到或超过 Mega-ASR。

## 设计

`pilot_rl.jsonl` 仍是 3,000 行（2,000 退化 + 500 英文 clean + 500 中文 clean）。优化器使用 `sample_strategy: degraded`，clean 行留在文件里做审计。验证池是 1,698 行 `rl_val_pool.jsonl`，SHA-256 `b0701db2ec3735222179e21fb4bf3c49c256d8da5d3a85cb7e61bfa6b7d9cd99`。贪心 WER 用冻结的 2,867 行 `validation.jsonl`。

训练前先跑 128 条退化语音探针。`projected_train_reward_mass >= 17.0` 且决策为 `GO_GRPO` 才构造优化器。前 32 条贪心 `generate` 与 `transcribe` 不一致则是 `BLOCKED_DECODE_MISMATCH`。

步数最多 12，按 4/8/12 分块。`--max-steps` 是这一块的终点。YAML 里的 12 是 horizon。每一块传 `--no-export-merged-on-finish`。`CHUNK_DONE` 才允许下一块。`COMPLETED` 由驱动在没有停止且走完 horizon 时书写。

设计内停止不加步：

| 条件 | 状态 |
| --- | --- |
| `raw_kl > 5e-4` | `STOPPED_KL` |
| 贪心奖励相对 Step 0 `< -0.005` | `STOPPED_REWARD_DROP` |
| 2,867 条 Robust 增量 `>= 0.0005` | `STOPPED_ROBUST` |
| Step 8 质量 `>= 11.33` 且贪心增益 `< +0.001` | `BLOCKED_TRANSFER` |
| Step 12 质量 `>= 17` 且贪心增益 `< +0.001` | `BLOCKED_TRANSFER` |
| 低质量且增益未到 `+0.002` | `BLOCKED_SEARCH` |
| 连续坍缩 | `FAILED_ZERO_VARIANCE` |
| 整道门禁 PASSED | 候选，留在本 run 目录 |

`scripts/run_rl_pilot_v11_recover.sh` 在驱动退出后最多做一次恢复。评测缺行或缺少 gate 时重评该检查点。torchrun 非 0 且上一块是 `CHUNK_DONE` 时，从最后一份同时含 adapter 与 optimizer 的检查点恢复下一块，同一块只恢复一次。上面的设计内停止不恢复，也不从 Step 0 重开。训练或评测进程还在时不启动第二份。

停止如果落在 4/8/12 之外，评的是 `pipeline_state.json` 的 `global_step`，只要该目录有 `adapter_model.safetensors` 和 `optimizer.pt`。`scripts/score_rl_greedy_held_out.py` 调用训练器的 `evaluate_rl_validation(decode_mode="greedy")` 追加一条 Full Held-out，不创建优化器。随后 `scripts/score_rl_pilot_v11_checkpoint.sh` 跑 2,867 条和 `verify_gate.py --held-out-decode greedy --max-robust-regression 0.0`。`assigned` 或 `rollouts` 不是 1,698 时不写 held-out 行。合格行已存在时不重复写。

导出：只有 `gate.json` 的 `gate_status` 为 `PASSED` 时，才允许把该检查点导出到 `/data/mega-asr/runs/rl_pilot_v11/`。held-out reward 最高本身不是导出条件。`promoted` 保持 false。Champion 目录不写入。

## 测试

- 贪心 held-out 行的字段、整数行数和重复写入由 `tests/test_score_rl_greedy_held_out.py` 检查。
- 目录 README 必须列出 `scripts/` 里的 v11 脚本和 `run_rl_pilot_v10.sh`。
- 服务器门禁用服务器上的 `evaluation/verify_gate.py`。本地旧副本没有贪心 provenance 检查，不能覆盖服务器文件。
- 2,867 条预测必须是 2,867 行。`verify_gate` 因门禁 FAILED 返回 1 时，只要 `gate_step_<N>.json` 已写成，评测就算完成。

## 验收

候选必须同时满足：贪心 held-out 相对本次 Step 0（`0.8773`）≥ `+0.002`；Robust 相对 DPO 六位小数 ≤ 0；clean ≤ `+0.02`；相对 base 的累计 clean ≤ `+0.025`；至少一个退化场景变好；有效输出 ≥ `0.95`。采样平均奖励不参与通过。`robust_edit_delta` 不单独投票。只有对应的 `gate.json` 为 `PASSED` 才是候选。

## 影响

2026-09-29 11:28 CST，v11 在第 8 步这一块的 Step 7 触发 `STOPPED_KL`。`raw_kl` 依次是 Step 4 `8.9e-5`、Step 5 `2.38e-4`、Step 6 `4.04e-4`、Step 7 `5.68e-4`。`checkpoints/step_7` 含 adapter 和 optimizer。探针与 v10 相同，质量 `17.10`。Step 4 的 2,867 条门禁 FAILED：Robust `+9.8e-5`（+1 次编辑），held-out `−0.0001`。训练停止线是 `0.0005`，所以 Step 4 之后脚本继续，随后被 KL 拦住。

21:56 CST 开始对 Step 7 补贪心 held-out 和 2,867 条门禁。22:33 CST 写成 `/data/mega-asr/runs/rl_pilot_v11/gate_step_7.json`，状态 FAILED。贪心奖励从 Step 0 的 `0.8773` 到 Step 7 的 `0.8767`（`−0.0006`，错误率 `0.1100 → 0.1102`）。2,867 条上 Robust `−0.00045`（−6 次编辑）、clean `−0.000139`、累计 clean 相对 base `−0.000669`、6 个退化场景变好、有效输出 `1.0`。这些 WER 项通过，奖励项没有达到 `+0.002`。没有导出 `merged_base`。v11 不是候选。发布底座仍是 DPO Champion。不从这次 KL 停止续训。

v11 已把 v10 动笔时的「不提高学习率」改成 `2e-5` 并跑完。贪心奖励是 `−0.0006`，2,867 条编辑数仍是 −6。这一结果不再作为继续改学习率的依据。v10 检查点保持诊断用途。`scripts/finish_rl_pilot_v7.sh` 继续服务 balanced、采样 held-out 的旧 run，不能靠改 `RL_RUN_DIR` 来跑 v11。

Step 4 之后的 `raw_kl` 增量大约是 `+1.49e-4`、`+1.66e-4`、`+1.64e-4`，不是每步翻倍。停机原因是 `raw_kl > 5e-4`。第 4 步的门禁已经 FAILED，训练停止线 `0.0005` 让那一块继续；`+0.002` 没有在中途掐掉这次运行。Step 7 的 2,867 条 WER 项通过，同一检查点的贪心 held-out 奖励下降。两条指标反向移动，所以 WER 通过不能代替奖励门禁，也不能单独成为导出条件。`5e-4` 来自无锚点 v8 在相近 KL 上把 Robust 推差的记录。v11 这一个锚点检查点还没有出现那种恶化。这一次测量不改 β、学习率调度和 KL 天花板。

执行 v11 的训练器在服务器 `/data/mega-asr/repo/train/train_rl.py`（3,288 行）。它接受 `--sample-strategy degraded`、`--probe-only`、`--probe-decision` 和 `--wer-manifest`。本地 `main` 的 `train/train_rl.py` 只有 `balanced`/`standard`，没有这几个参数。分支 `execute-plan/5f274847-pr-4-rl-smoke` 的 argparse 同样拒绝 `degraded`。合入该分支不能让本地 v11 脚本跑起来，也不要用本地这份覆盖服务器训练器。

## 收口（2026-09-29）

v10 Step 8 与 v11 Step 7 用的是同一套锚、`β=0.04`、温度和定长 2 条反向。学习率分别是 `1e-5` 和 `2e-5`。Step 1 的 `policy_loss` 都是 `2.13505`，前三步获胜组都是 19、19、20，裁剪前梯度范数都在 3 到 5。

| 运行 | 步 | `raw_kl` | 获胜组 | 2,867 Robust | 编辑 | 场景 | 贪心奖励增量 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v10 | 8 | `7.1e-5` | 135 | `−0.000404` | −6 | 5 | `−0.0002` |
| v11 | 7 | `5.68e-4` | 121 | `−0.00045` | −6 | 6 | `−0.0006` |

2,867 条来自 `parallel_inference.py` 的 `transcribe`。1,698 条奖励来自训练器的贪心 `generate`。v10 在 KL `7.1e-5` 时已经有 −6 次编辑。v11 把 KL 推过 `5e-4` 之后，编辑数没有增加，贪心奖励更低。`+0.002` 等于这 1,698 条上 `3.40` 个奖励点，不是 30～40 次编辑。

因此这一轮收口本身不再把学习率改成 `1.5e-5`、把 β 调到 `0.06` 或 `0.08`，或把 `max_raw_kl` 放到 `8e-4`。v10 与 v11 的检查点都只作诊断。发布底座仍是 DPO Champion。`scripts/run_rl_pilot_v11.sh` 没有重跑门闩，再执行会先跑探针并进入第 4 步那一块，所以不要启动。2026-09-30 另开的运行是 `rl_pilot_v12`：学习率回到 `1e-5`，获胜优势改为原始奖励差；只有裁剪前梯度中位数仍大于 1.5 才切到 `rl_pilot_v13` 的固定优势 `0.10`。v12 已停在 Step 4。见 `11_rl_v12_design.md` 和 `12_rl_v14_design.md`。
