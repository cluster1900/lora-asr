# RL Pilot 门禁分层方案

## 背景

当前 RL Pilot 的训练进程多数能够完成并保存 checkpoint，但正式门禁要求 held-out greedy reward 至少提升 `+0.002`，同时 Robust macro 绝对不能回退。历史结果显示，RL v6 Step 20 和 v27 Step 4 已经满足输出、Clean 与大部分 Robust 条件，却分别因为 reward `-0.0004` 和 `+0.0001` 未通过。若直接降低正式门禁，可能把诊断模型误当成发布模型。

## 范围

本方案只增加一个显式的 `pilot_feasibility` 门禁配置，不修改既有 `release` 门禁，也不改变模型发布条件。它用于回答“训练是否完成且没有明显伤害基座”，不能用于 Full RL 或正式发布。

## 设计

`evaluation/verify_gate.py` 增加 `--gate-profile` 参数：

| profile | 用途 | reward 要求 | Robust 要求 | 发布资格 |
| --- | --- | ---: | ---: | --- |
| `release` | 正式验收，默认值 | `>= +0.002` | `<= 0.0` | 可以进入 release 候选 |
| `pilot_feasibility` | 小闭环可行性验证 | `>= 0.0` | `<= +0.001` | 不可以发布 |

两种 profile 仍然要求：至少一个退化场景改善、Clean 回退不超过 `0.02`、相对 Base 的累计 Clean 回退不超过 `0.025`、有效输出率至少 `0.95`、空输出率不超过 `0.002`、失败率增量不超过 `0.05`、零方差比例不超过 `0.75`。

可行性门禁输出到独立的 `pilot_gate.json`，记录 `gate_profile=pilot_feasibility`、原始指标、阈值和完整 provenance，并写入 `release_eligible=false`。只有 `gate_profile=release` 且 `gate_status=PASSED` 时，脚本才可以把 checkpoint 视为正式候选。

服务器上的 `scripts/score_rl_pilot_checkpoint.sh` 通过 `RL_GATE_PROFILE` 选择 profile：默认值 `release` 写入 `gate_step_<N>.json`，设置为 `pilot_feasibility` 时写入独立的 `pilot_gate_step_<N>.json`。两种结果都必须保留，不能用 pilot 文件覆盖正式 gate。

## 输入输出

输入仍是 Base/DPO/RL 的 `metrics.json`、预测 JSONL、RL loss log 和 manifest。新增输出字段：

- `gate_profile`：`release` 或 `pilot_feasibility`；
- `release_eligible`：只有严格 release 通过时为 `true`；
- `gate_status`：严格门禁为 `PASSED/FAILED`，可行性门禁为 `PILOT_PASSED/FAILED`。

## 测试与验收

- 默认不带 `--gate-profile` 的命令行为保持原有 release 门禁；
- `pilot_feasibility` 能接受 reward `0.0001` 和 Robust `0.000014` 的 v27 Step 4 形态；
- `pilot_feasibility` 仍拒绝负 reward、无退化场景改善、空输出或明显 Robust 回退；
- 可行性门禁通过时 `release_eligible` 必须为 `false`；
- 每份输出都保留阈值、输入文件 SHA-256 和预测 provenance；
- scorer 的 `RL_GATE_PROFILE=pilot_feasibility` 能写出独立的 `pilot_gate_step_<N>.json`；
- 只有 release profile 的 `PASSED` 才满足 RL 阶段正式验收。

## 影响

该方案允许先确认训练链路和 checkpoint 是可用的，同时保留正式质量门禁的约束。它不会制造模型增益，也不会把当前 RL 结果宣称为超过 DPO Champion；若可行性门禁通过而 release 失败，正式底座仍保持 DPO Champion。
