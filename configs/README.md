# configs

## 目录职责

配置是数据、推理和训练的唯一声明来源。运行时必须保存解析后的配置副本；禁止在 notebook 或 runner 中隐藏关键超参数。学习率、LoRA 结构、`decode.eval`、`decode.rollout` 和门禁阈值都在开跑前写入配置，开跑后保持不可变。

## 当前文件清单

- `README.md`：本说明文档。
- `inference/base_eval.yaml`：Phase 0 greedy 评测配置，schema `qwen3-asr-infer-v1`，seed 42，`wer_tokenizer` 为 null。
- `data/manifest_mvp.yaml`：Phase 1 正式数据配置，schema `qwen3-asr-data-v1`，seed 42，val 每个 scenario 至少 50 条。`source_version` 是 `aishell1-openslr33-pilot-20261015`，`audited` 为 true。行数不够时仍是 `insufficient_data`。OpenSLR 正文里的 “free for academic use” 还没有单独接受。
- `train/sft_smoke.yaml`：SFT smoke 配置，schema `qwen3-asr-train-v1`，6 step，第 5 step 中断。`wer_tokenizer` 为 null。`allow_dirty` 为 true，因为当前工作区还有未提交改动。它不能发布。
- `train/sft_pilot.yaml`：已执行的 100 step SFT pilot，学习率 `1e-4`。每 10 step 保存并评固定 2500 条 val。`interrupt_after_step` 为 null。`keep_last` 为 10。`publishable` 在 release 读回之前保持 false。
- `train/sft_pilot_lr1e5.yaml`：下一轮 100 step SFT，学习率 `1e-5`。解码、数据和 gate 与 `sft_pilot.yaml` 相同。不恢复 run `20261015T104034_sft_06b_zh_seed42`。当前 run 是 `20261015T234939_sft_06b_zh_seed42`，step 30 的评测未完成。
- `train/aishell_val_base_20261015.json`：pilot 的父 baseline，schema `qwen3-asr-baseline-v1`。它钉住 val 文件 SHA-256 和五个 scenario 的 base micro CER。
- `train/aishell_val_sft_step30_20261016.json`：DPO 的 SFT 父指标，schema `qwen3-asr-parent-metrics-v1`。数字来自 step 30 的 2500 条 val。
- `train/dpo_pilot.yaml`：DPO pilot。学习率 `5e-6`，LoRA dropout `0.0`，累积 4，beta `0.1`，父 release 是 `sft_06b_zh_20261015T234939_step000030`。`audit_disposition` 保持 `unlistened`。
- `train/dpo_dropout_probe.yaml`：dropout probe。学习率 `5e-5`，warmup 0，最多 10 step，`train_scenarios` 只有 dropout。父 release 与 pilot 相同。`selected_step` 保持 null，`publishable` 保持 false。
- `train/dpo_gap_strip_probe.yaml`：gap-strip dropout probe。预算与 `dpo_dropout_probe.yaml` 相同，pair 换成 `pairs_gap_strip` 和 `gap_strip_ref_logprobs`。`launch_utterance` 是「开始吧」。`listen_waiver` 仍是 `user_20261017_skip_listen`，`audit_disposition` 仍是 `unlistened`。`selected_step` 保持 null，`publishable` 保持 false。不恢复 `20261018T052928` 或 `20261017T131802`。
- `train/dpo_pilot_nondropout.yaml`：非静音四场景 DPO pilot。预算与 `dpo_pilot.yaml` 相同（`profile: pilot`、学习率 `5e-6`、warmup 2、最多 100 step、beta `0.1`），`train_scenarios` 设为 `[clean, noise, reverb, far_field]`，直接复用 2026-10-17 冻结的 `ref_logprobs/` 和 `pairs/` 哈希（3640 train / 533 val pairs），全量 2500 条 val 门禁评测。`audit_disposition` 保持 `unlistened`，`publishable` 保持 false。不恢复 `20261018T132840`、`20261018T052928` 或 `20261017T131802`。
- `train/dpo_pilot_aligned.yaml`：无标点单字原位替换、去 Clean 同音拷贝与差异 token 梯度掩码 DPO pilot。保持 `profile: pilot`、学习率 `5e-6`、最多 100 step、beta `0.1`，开启 `train_scenarios: [noise, reverb, far_field]`、`val_scenarios: [clean, noise, reverb, far_field]`、`aligned_substitution_only: true`、`exclude_clean_identical_rejected: true`、`max_edit_count: 1`、`loss_token_scope: mismatched_only`、`ref_margin_min: -4.0`、`ref_margin_max: 0.0`（内存过滤后 164 train / 118 val pairs），并将优化器阻尼校准为 `warmup_steps: 10`、`eps: 0.000001`、`weight_decay: 0.0`。复用 2026-10-17 冻结的 `ref_logprobs/` 和 `pairs/` 哈希，全量 2500 条 val 门禁不变。`audit_disposition` 保持 `unlistened`，`publishable` 保持 false。不恢复 `20261019T130250`、`20261019T012838` 及前三次 DPO run。已执行的 run `20261020T014427_dpo_06b_zh_seed42` 在 step 10 风险停止，不恢复。
- `train/dpo_pilot_edit1_seq.yaml`：同一 164 条单字过滤和同一套冻结门禁，`loss_token_scope` 为 `sequence`，`warmup_steps` 为 2。step 1 的学习率是 `2.5e-6`，step 2 起是 `5e-6`。`eps` 仍是 `1e-6`，`weight_decay` 仍是 `0.0`。不恢复 `20261020T014427`。`publishable` 保持 false。run `20261020T024637_dpo_06b_zh_seed42` 已在 step 10 风险停止，不恢复。
- `train/dpo_pilot_edit1_anchor.yaml`：同一 164 条单字过滤、`mismatched_only`、`warmup_steps: 10` 和同一套冻结门禁。每个 optimizer step 另加 `dropout_anchor_weight: 0.05` 的 token 平均 NLL，教师是 train dropout 行的 `rejected_text`（SFT step 30 贪心），不是金标。不恢复 `20261020T014427` 或 `20261020T024637`。`publishable` 保持 false。run `20261020T041218_dpo_06b_zh_seed42` 的 step 40 通过，step 41 风险停止，不恢复。
- `train/dpo_pilot_edit1_anchor_stop40.yaml`：与 `dpo_pilot_edit1_anchor.yaml` 相同，只把 `max_steps` 改成 40，停在评测间隔上，不再走第 41 步。其他 pilot 的 `max_steps` 仍是 100。不恢复 `20261020T041218`。run `20261020T114647_dpo_06b_zh_seed42` 在 step 30 风险停止，`selected_step` 是 10。不恢复。`publishable` 保持 false。
- `train/dpo_pilot_edit1_anchor_step40_h100.yaml`：纯方案 1。在 `dpo_pilot_edit1_anchor_stop40.yaml` 基础上设置 `optim.scheduler_max_steps: 100`，使采样器在 `train.max_steps: 40` 准时停步，同时余弦学习率保持 100 步周期（step 20 `4.8643084e-6`、step 30 `4.4736e-6`、step 40 `3.875e-6`），每步仍取 1 条 `dropout` 锚（`dropout_anchor_samples_per_step: 1` 缺省），已在 run `20261022T133007_dpo_06b_zh_seed42` 上 100% 确定性复现 `20261020T041218` Step 10–40 过关结果并在 Step 40 无风险完赛（`selected_step = 40`）。`publishable` 保持 false。
- `train/dpo_pilot_edit1_anchor_step40_h100_a4.yaml`：方案 1 + 方案 3 对比配置。在 `dpo_pilot_edit1_anchor_step40_h100.yaml` 基础上开启 `dpo.dropout_anchor_samples_per_step: 4`（与 `gradient_accumulation_steps: 4` 对齐），每个 optimizer step 对 4 条确定性 `dropout` 锚样本做等权平均（每条权重 `0.05 / 4 = 0.0125`，逐条反传），保持 `max_grad_norm: 1.0` 与门禁阈值不变。run `20261022T161911_dpo_06b_zh_seed42` 在 step 10 因 `preference_accuracy=0.5` 风险停止，不恢复。`publishable` 保持 false。
- `train/aishell_val_dpo_step40_20261022.json`：RL 的 DPO 父指标，schema `qwen3-asr-parent-metrics-v1`。数字来自 `20261022T133007_dpo_06b_zh_seed42`（`dpo_06b_zh_20261022T133007_step000040`）step 40 的 2500 条 val（总计 micro CER `0.06297105406881485`）。
- `train/rl_smoke.yaml`：Phase 4 RL 6 step smoke 配置，schema `qwen3-asr-train-v1`，`stage: rl`，`profile: smoke`，学习率 `1e-5`，1 group / 累积 2，第 5 step 中断，父 release 为 `dpo_06b_zh_20261022T133007_step000040`。`publishable` 为 false。
- `train/rl_pilot.yaml`：Phase 4 RL 50 step pilot 配置，schema `qwen3-asr-train-v1`，`stage: rl`，`profile: pilot`，学习率 `1e-5`，LoRA dropout `0.0`，1 group（`G=4`：1 greedy + 3 sampled）/ 累积 2，`kl_beta: 0.04`，每 5 step 保存并评 2500 条 val（含固定 step 0 前缀词表条件 `raw_kl <= 0.001` 与相对 step 0 的 `min_reward_improvement: 0.002`），父 release 为 `dpo_06b_zh_20261022T133007_step000040`。已执行的 run `20261023T020835_rl_06b_zh_seed42` 在 step 5 风险停止，不恢复。
- `train/rl_pilot_aligned.yaml`：Phase 4 对齐版 RL pilot 配置，schema `qwen3-asr-train-v1`，`stage: rl`，`profile: pilot`。开启 `rl.train_scenarios: [noise, reverb, far_field]`、组内无标点等长替换对齐与差异 token 梯度掩码（`rl.loss_token_scope: mismatched_only`、`rl.max_edit_count: 2`）、仅非零信号 group 计入累积窗口（`rl.accumulate_signal_groups_only: true`、`gradient_accumulation_steps: 4`）、`rl.dropout_anchor_weight: 0.05`（以冻结 DPO step 40 贪心预测为教师锚定 train `dropout`），以及阻尼优化器与日程（`learning_rate: 1e-6`、`warmup_steps: 5`、`scheduler_max_steps: 50`、`max_steps: 20`、`save_steps: 5`、`eval_steps: 5`、`kl_beta: 0.10`、`min_reward_improvement: 0.00002`）。父 release 为 `dpo_06b_zh_20261022T133007_step000040`。已执行的 run `20261023T042034_rl_06b_zh_seed42` 在 step 5 一次性通过全部门禁（`selected_step = 5`），并封包发布为 `rl_06b_zh_20261023T042034_step000005`。`publishable` 保持 false。
- `train/aishell_val_rl_step5_20261023.json`：Phase 4 RL 选出 release（`rl_06b_zh_20261023T042034_step000005`，step 5）在 2500 条固定 `val` 上的固化指标，schema `qwen3-asr-parent-metrics-v1`（总计 micro CER `0.06275259421081376`）。

训练字段以 [`docs/qwen3-asr/04_training_plan.md`](../docs/qwen3-asr/04_training_plan.md) 为准。数据字段以 [`docs/qwen3-asr/03_data_plan.md`](../docs/qwen3-asr/03_data_plan.md) 为准。

## 主要输入输出

- **输入**：开发者或实验调度器指定的超参数与路径。
- **输出**：在运行启动时解析并固化至 `runs/<run_id>/resolved_config.yaml`，同时写语义 `config_sha256` 和 YAML 字节 hash；训练 checkpoint 必须携带该副本。

## 使用入口

由各 CLI 脚本通过 `--config` 参数读取，例如：
```bash
python3 train/train_sft.py --config configs/train/sft_pilot.yaml ...
```

开发阶段先用文档模板生成配置，配置提交前运行未知字段、有限数值、stage/parent release、`decode.eval.do_sample=false`、`world_size=1` 和必填 gate 的 preflight。恢复只能使用原 run 的 resolved config；改变学习率、预算、阈值或 LoRA 目标必须新建 run。

## 维护要求

新增配置模板或修改配置 schema 时，必须在同一提交同步更新对应目录的 `README.md` 和训练/数据方案文档。
