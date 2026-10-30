# train

## 目录职责

提供 Qwen3-ASR-0.6B 的 SFT、DPO、RL runner，面向单张 A100 与 Colab 执行合同，读取固定配置和中文 JSONL manifest，并保存可恢复的完整 checkpoint。

## 当前文件清单

- `README.md`：本说明文档。
- `train_sft.py`：SFT runner。只训练解码器注意力 LoRA，loss 按监督 token 加权，`--resume` 只读 `latest.json`。
- `build_sft_smoke_subset.py`：从已有 train/val manifest 抽 smoke 子集。10 个 train source、2 个 val source，每个 source 保留五类场景。
- `sft_schema.py`：`qwen3-asr-train-v1` 解析和语义 hash，支持可选 `optim.scheduler_max_steps` 解耦余弦周期与 `train.max_steps`。未知字段会在创建 optimizer 之前拒绝。
- `sft_contract.py`：LoRA 名称、label mask、token 加权 loss、sampler、smoke 抽样、运行时间上限和未完成评测记录。
- `sft_checkpoint.py`：checkpoint checksum 和 `latest.json`。`.incoming` 不能作为恢复来源。
- `sft_release.py`：把 gate 选出的 pilot checkpoint 收成不可变 SFT release，并在新进程里核对 greedy 文本。`latest.json` 不作为 release。
- `reference_logprob.py`：用已发布 SFT adapter 给 DPO pair 的 chosen 和 rejected 计算 reference 序列 logprob。打分前去掉 SFT 的 embedding hook 并关闭 gradient checkpointing。不更新权重，不下载额外模型到测试里。
- `dpo_contract.py`：DPO pair loss、held-out preference、长度分桶、相对 SFT 父版本和 base clean 上限的门禁、场景与无标点原位替换/去 `clean` 同音重复拷贝/最大编辑数/参考 margin 内存过滤（`aligned_substitution_only`、`exclude_clean_identical_rejected`、`max_edit_count`、`ref_margin_min`、`ref_margin_max`）、差异监督 token 掩码 `mismatched_token_mask`、确定性多样本 dropout 锚切片 `dropout_anchor_step_slice`，以及 dropout 静音段剥离 `strip_dropout_gap`。trainer 当前不调用剥离。
- `dpo_schema.py`：解析 stage `dpo`，支持可选 `train_scenarios`、`val_scenarios`、`aligned_substitution_only`、`exclude_clean_identical_rejected`、`max_edit_count`、`loss_token_scope`、`ref_margin_min`、`ref_margin_max`、`dropout_anchor_weight`、`dropout_anchor_target`、`dropout_anchor_samples_per_step` 以及锚点停步配置的 `optim.scheduler_max_steps: 100`。SFT loader 仍拒绝非 null 的 `dpo` 块。
- `train_dpo.py`：DPO runner。加载官方 base 和已发布 SFT adapter，用离线 reference 序列 logprob 做 pair loss，解耦训练与验证集的场景/同音/编辑数过滤，支持 `loss_token_scope: mismatched_only` 直通式差异 token 梯度反传与 `dropout_anchor_samples_per_step` 多样本锚等权平均，并在退出前将最终 `status.json`、`gate.json`、`run_report.md` 和 `listen_waiver.json` 同步到 Drive。policy 保留训练 hook。不把听音处置改成已听。
- `dpo_release.py`：把 gate 选出的 DPO pilot checkpoint（`20261022T133007_dpo_06b_zh_seed42` 的 `step_000040`）封包为不可变 DPO release（`dpo_06b_zh_20261022T133007_step000040`），并在独立新进程中核对 2500 条 val 的 greedy 文本，一致后标为 `published`（同时保留源审计 `unlistened` 记录与源 run `gate.json` 不变）。
- `rl_contract.py`：RL（`group_pg_mixed_v1`）确定性候选种子派生、单候选 reward 计算与 `[-1, 1]` 截断、G=4 组有效性与组均值中心化 advantage、全组同分零损失（含零 KL）、组内无标点等长替换对齐与差异 token 掩码 `aligned_group_mismatch_mask`、`k3` 逐 token 代理 KL、固定 step 0 前缀词表条件 `raw_kl`、reference 权重 tensor SHA-256 校验，以及相对 DPO 父版本/base/step 0 的 RL 门禁裁决。
- `rl_schema.py`：解析 `stage: rl`（`profile: smoke | pilot`），校验已发布 DPO 父 release、RL 预算与语义配置 hash，支持可选对齐字段（`train_scenarios`、`loss_token_scope`、`max_edit_count`、`accumulate_signal_groups_only`、`dropout_anchor_weight`）。
- `train_rl.py`：RL runner。加载官方 base 与已发布 DPO release adapter，维护独立冻结 reference LoRA 权重并在启动/评测/保存/退出时核对 checksum（先在 `no_grad` 下执行 reference 前向再执行 policy 前向并逐候选反传），执行 `1 greedy + 3 sampled` rollout、可选声学场景过滤/组内无标点等长替换差异 token 掩码/仅非零信号 group 累积窗口/`dropout` 父版本假设锚，以及固定 step 0 前缀 `raw_kl` 与 CER/reward 门禁评测。
- `rl_release.py`：把 gate 选出的 RL pilot checkpoint 封包为不可变 RL release（`rl_06b_zh_<timestamp>_step<selected>`），核对 `passed=true`、`risk_stop=false`、`reward_delta_vs_step0 >= min_reward_improvement`、`raw_kl <= max_raw_kl` 与 `reference_matched=true`，并在独立新进程中核对 2500 条 val 的 greedy 文本，一致后标为 `published`（同时保留 `audit_disposition: unlistened` 与源 run `gate.json` 不变）。

## 主要输入输出

- **输入**：`Qwen/Qwen3-ASR-0.6B` 官方权重、训练/验证 JSONL manifest、YAML 训练配置、随机种子、`run_id`、输出目录。DPO 阶段另输入包含离线 reference logprob 的 `pair_manifest.jsonl`。
- **输出**：遵循 `docs/qwen3-asr/04_training_plan.md` 的 run 目录，包含 `request.json`、`environment.json`、`resolved_config.yaml`、`manifest_sha256.json`、`metrics.jsonl`、`events.jsonl`、逐步 predictions、每次评测的 `eval/<step_name>/metrics.json` 与 `by_scenario.csv`、`gate.json`、`run_report.md`，以及完整 checkpoint 目录（含 `adapter/`、`optimizer.pt`、`scheduler.pt`、`rng_state.pt`、`sampler_state`、`training_state.json`、`sha256.json`）与原子更新的 `latest.json`。运行时间在评测中途用尽时，该 step 不写 metrics 和 gate，只写 `eval/<step_name>/runtime_stop.json`，`status.json` 为 `paused` / `max_runtime`。

## 使用入口

```bash
python3 train/build_sft_smoke_subset.py \
  --train-manifest results/phase1_aishell_pilot_20261015/manifest/train.jsonl \
  --val-manifest results/phase1_aishell_pilot_20261015/manifest/val.jsonl \
  --output-dir results/phase2_sft_smoke_20261015/manifest \
  --seed 42

python3 train/train_sft.py \
  --config configs/train/sft_smoke.yaml \
  --manifest results/phase2_sft_smoke_20261015/manifest/train.jsonl \
  --val-manifest results/phase2_sft_smoke_20261015/manifest/val.jsonl \
  --output-dir runs/<run_id> \
  --run-id <YYYYMMDDTHHMMSS_sft_06b_zh_seed42> \
  --seed 42

python3 train/train_sft.py \
  --config configs/train/sft_smoke.yaml \
  --manifest results/phase2_sft_smoke_20261015/manifest/train.jsonl \
  --val-manifest results/phase2_sft_smoke_20261015/manifest/val.jsonl \
  --output-dir runs/<run_id> \
  --run-id <YYYYMMDDTHHMMSS_sft_06b_zh_seed42> \
  --seed 42 \
  --resume
```

DPO pilot：

```bash
python3 train/train_dpo.py \
  --config configs/train/dpo_pilot.yaml \
  --manifest <train.jsonl> \
  --val-manifest <val.jsonl> \
  --pair-train <pairs_logprob/train_pairs.jsonl> \
  --pair-val <pairs_logprob/val_pairs.jsonl> \
  --reference-cache <ref_logprobs.jsonl> \
  --reference-report <ref_report.json> \
  --pair-metadata <pair_metadata.json> \
  --release-dir <releases/sft/sft_06b_zh_20261015T234939_step000030> \
  --audio-root <manifest-dir> \
  --null-pairs-dir <pairs-dir> \
  --output-dir <run-dir> \
  --run-id <YYYYMMDDTHHMMSS_dpo_06b_zh_seed42> \
  --seed 42
```

`run_id` 必须以 `_dpo_06b_zh_seed` 加 seed 结尾。填充后的 pair 哈希、reference cache 哈希和 `parent_release_sha256` 必须与配置一致。null `pairs/` 只读，用来确认原来的哈希和 null logprob 还在。配置里的 120 分钟不改。启动时设置 `QWEN3_ASR_DISABLE_RUNTIME_BUDGET=1`，日志必须出现 `DPO_RUNTIME_BUDGET disabled`。`publishable` 保持 false。

发布选出的 DPO step 40 并做独立进程读回校验：

```bash
python3 train/dpo_release.py prepare \
  --config configs/train/dpo_pilot_edit1_anchor_step40_h100.yaml \
  --run-dir <dpo_run_dir> \
  --output-dir releases/dpo/dpo_06b_zh_20261022T133007_step000040 \
  --release-id dpo_06b_zh_20261022T133007_step000040 \
  --sync-dir <drive>/releases/dpo/dpo_06b_zh_20261022T133007_step000040

python3 train/dpo_release.py readback \
  --config configs/train/dpo_pilot_edit1_anchor_step40_h100.yaml \
  --release-dir releases/dpo/dpo_06b_zh_20261022T133007_step000040 \
  --manifest <val.jsonl> \
  --sync-dir <drive>/releases/dpo/dpo_06b_zh_20261022T133007_step000040
```

RL smoke / pilot：

```bash
python3 train/train_rl.py \
  --config configs/train/rl_pilot_aligned.yaml \
  --manifest <train.jsonl> \
  --val-manifest <val.jsonl> \
  --release-dir releases/dpo/dpo_06b_zh_20261022T133007_step000040 \
  --audio-root <manifest-dir> \
  --output-dir runs/<run_id> \
  --run-id <YYYYMMDDTHHMMSS_rl_06b_zh_seed42> \
  --seed 42
```

发布选出的 RL pilot checkpoint 并做独立进程读回校验：

```bash
python3 train/rl_release.py prepare \
  --config configs/train/rl_pilot_aligned.yaml \
  --run-dir <rl_run_dir> \
  --output-dir releases/rl/rl_06b_zh_<timestamp>_step<selected> \
  --release-id rl_06b_zh_<timestamp>_step<selected> \
  --sync-dir <drive>/releases/rl/rl_06b_zh_<timestamp>_step<selected>

python3 train/rl_release.py readback \
  --config configs/train/rl_pilot_aligned.yaml \
  --release-dir releases/rl/rl_06b_zh_<timestamp>_step<selected> \
  --manifest <val.jsonl> \
  --sync-dir <drive>/releases/rl/rl_06b_zh_<timestamp>_step<selected>
```

dropout probe 用同一条命令，把 `--config` 换成 `configs/train/dpo_dropout_probe.yaml`。它在整份 pair 哈希和 cache 检查之后只训练 dropout。第一次评测打印 `DPO_LOCATE`，并把 `eval/<step>/scored.jsonl` 和 `eval_decisions.jsonl` 同步到 Drive。`selected_step` 保持 null。这个配置不恢复 `20261017T131802_dpo_06b_zh_seed42`。

gap-strip probe 把 `--config` 换成 `configs/train/dpo_gap_strip_probe.yaml`。`--null-pairs-dir` 指向 `pairs_gap_strip`，填充 pair、cache 和 report 指向 `gap_strip_ref_logprobs`。trainer 不调用 `strip_dropout_gap`。`listen_waiver.json` 仍记录 2026-10-17 的豁免，并加上 2026-10-18 的「开始吧」。`heard` 保持 false，审计处置保持 `unlistened`。`pair_metadata.json` 的 `trainable` 保持 false。`selected_step` 保持 null。这个配置不恢复 `20261018T052928_dpo_06b_zh_seed42`。

非静音四场景 DPO pilot 把 `--config` 换成 `configs/train/dpo_pilot_nondropout.yaml`。它保持 `profile: pilot` 和 `5e-6` 预算，直接复用 2026-10-17 冻结的 `pairs/` 与 `ref_logprobs/`，在整份哈希和 cache 核对后过滤为 `[clean, noise, reverb, far_field]`（3640 train / 533 val pairs），并在每次 `eval_steps` 对全部 2500 条 val 执行门禁。这个配置不恢复 `20261018T132840_dpo_06b_zh_seed42`、`20261018T052928_dpo_06b_zh_seed42` 或 `20261017T131802_dpo_06b_zh_seed42`。

无标点原位替换与去 Clean 同音拷贝 DPO pilot 把 `--config` 换成 `configs/train/dpo_pilot_aligned.yaml`。它保持 `profile: pilot` 和 `5e-6` 预算，直接复用 2026-10-17 冻结的 `pairs/` 与 `ref_logprobs/`，在整份哈希和 cache 核对后按 `train_scenarios: [noise, reverb, far_field]`、`val_scenarios: [clean, noise, reverb, far_field]`、`aligned_substitution_only: true`、`exclude_clean_identical_rejected: true`、`max_edit_count: 1`、`loss_token_scope: mismatched_only`、`ref_margin_min: -4.0`、`ref_margin_max: 0.0` 过滤为 164 train / 118 val pairs（优化器 `warmup_steps: 10`、`eps: 1e-6`、`weight_decay: 0.0`），并在每次 `eval_steps` 对全部 2500 条 val 执行门禁。164 条、累积 4、`max_epochs: 1` 时优化器在 step 41 结束；这一步也会保存并评测。日志里的 `lr` 是这一步实际使用的学习率：step 1 为 `5e-7`，step 10 为 `5e-6`。这个配置不恢复 `20261019T130250_dpo_06b_zh_seed42`、`20261019T012838_dpo_06b_zh_seed42` 及前三次 DPO run。它的 run `20261020T014427_dpo_06b_zh_seed42` 已在 step 10 风险停止，不恢复。`configs/train/dpo_pilot_edit1_seq.yaml` 使用同一 164 条单字过滤和同一套门禁，`loss_token_scope` 为 `sequence`，`warmup_steps` 为 2。step 1 的学习率是 `2.5e-6`，step 2 起是 `5e-6`。run `20261020T024637_dpo_06b_zh_seed42` 已在标准 A100 上跑到 step 10 并风险停止。step 1 日志是 `lr=2.5e-06`。preference 过了 `0.55`，degraded macro CER 上升。不恢复。`configs/train/dpo_pilot_edit1_anchor.yaml` 回到 `mismatched_only` 和 `warmup_steps: 10`，并在每个 optimizer step 用 `dropout_anchor_weight: 0.05` 对一条 train dropout 的父版本贪心文本做 token 平均 NLL。锚文本是 `rejected_text`。step 日志另写 `anchor_loss`。`configs/train/dpo_pilot_edit1_anchor_stop40.yaml` 只把 `max_steps` 改成 40。164 条、累积 4 时优化器停在 step 40，最后一批仍是 4 条。不恢复 `20261020T041218`。`configs/train/dpo_pilot_edit1_anchor_step40_h100.yaml`（纯方案 1）在 `max_steps: 40` 下设置 `optim.scheduler_max_steps: 100`，使 step 1–40 的余弦学习率保持 100 步周期并在 step 40 结束采样；其 run `20261022T133007_dpo_06b_zh_seed42` 已在标准 A100 上通过 step 10、20、30、40 全部四次门禁，以 `status=completed` / `reason=max_steps`（零 `gate_risk`）完赛，`selected_step` 与 `latest.json` 均为 `40`（`publishable: false`）。`configs/train/dpo_pilot_edit1_anchor_step40_h100_a4.yaml`（方案 1+3）在此基础上设置 `dpo.dropout_anchor_samples_per_step: 4`，每步对 4 条 `dropout` 锚样本做等权平均反传；其 run `20261022T161911_dpo_06b_zh_seed42` 在 step 10 因 `preference_accuracy=0.5` 风险停止（声学 CER 净减 6 个编辑），不恢复。不恢复 `20261020T114647` 或 `20261020T041218`。

SFT smoke 配置在第 5 step 保存并暂停，`--resume` 继续到第 6 step。smoke 的 `gate.json` 保持 `incomparable`。

已执行的 100 step pilot 用 `configs/train/sft_pilot.yaml`，学习率 `1e-4`。它训练完整 pilot train manifest，并在固定 2500 条 val 上和 `configs/train/aishell_val_base_20261015.json` 比较。pilot 没挂上 Drive 会在下载模型前退出。`interrupt_after_step` 为空。通过的 checkpoint 在脏工作区上仍然 `publishable: false`。

下一轮用 `configs/train/sft_pilot_lr1e5.yaml`，学习率 `1e-5`。解码 hash 与已执行 pilot 相同。`--resume` 不能指向 run `20261015T104034_sft_06b_zh_seed42`。run `20261015T234939_sft_06b_zh_seed42` 已在 step 50 以 `gate_risk` 停下。step 30 和 step 40 通过，选出的 checkpoint 是 step 30，`publishable` 仍是 false。`latest.json` 指向 step 50，那是最后保存的进度。

发布这份 step 30：

```bash
python3 train/sft_release.py prepare \
  --config configs/train/sft_pilot_lr1e5.yaml \
  --run-dir <run_dir> \
  --output-dir releases/sft/sft_06b_zh_20261015T234939_step000030 \
  --release-id sft_06b_zh_20261015T234939_step000030

python3 train/sft_release.py readback \
  --config configs/train/sft_pilot_lr1e5.yaml \
  --release-dir releases/sft/sft_06b_zh_20261015T234939_step000030 \
  --manifest <val.jsonl>
```

`prepare` 写出 `status=staged`。`readback` 只有在全部 `prediction_raw`、`stop_reason` 和 `error` 一致时才写成 `published`。读回进程在加载模型前使用与训练相同的 `CUBLAS_WORKSPACE_CONFIG=:4096:8`、确定性算法、关闭 TF32 和配置 seed，并每 250 条打印 `SFT_RELEASE_PROGRESS`。

已发布的 release 可以导出一份 manifest 的 greedy 假设。staged release 会被拒绝。输出 JSONL 已有内容时，只能从与 manifest 相同的 sample_id 前缀继续。

```bash
python3 train/sft_release.py hypotheses \
  --config configs/train/sft_pilot_lr1e5.yaml \
  --release-dir releases/sft/sft_06b_zh_20261015T234939_step000030 \
  --manifest <train.jsonl> \
  --output train_predictions.jsonl \
  --sync-path <drive>/dpo/sft_06b_zh_20261015T234939_step000030/train_predictions.jsonl
```

给已经写成的 pair 补 reference logprob：

```bash
python3 train/reference_logprob.py \
  --config configs/train/sft_pilot_lr1e5.yaml \
  --release-dir releases/sft/sft_06b_zh_20261015T234939_step000030 \
  --pairs-dir <pairs-dir> \
  --audio-root <manifest-dir> \
  --output-dir <logprob-dir> \
  --seed 42
```

缓存按 train、val 的现有顺序续写。token 数和 pair 不一致，或抽查 forward 的绝对误差大于 1e-4、相对误差大于 1e-5，都不写出 `pairs_logprob`。原来的 null logprob pair 文件保持不变。

运行时间上限在每个训练 step 开始前和每条 val 样本解码前检查。超过后暂停在最后一个已保存的 checkpoint。恢复时如果这个 step 还没有 decision，先补评测再继续训练。配置里的 120 分钟保持不变，语义 hash 才能接上原 checkpoint。续训启动设置 `QWEN3_ASR_DISABLE_RUNTIME_BUDGET=1` 时，日志必须出现 `SFT_RUNTIME_BUDGET disabled`，这次不按该时钟暂停。Colab 客户端不包住整个训练。sub2 用短连接保活，checkpoint 仍每 10 step 在评测前同步到 Drive。

每个 runner 必须先做 schema、模型/processor、LoRA target、dtype/world size、manifest/hash、显存和一 batch forward/backward preflight；只在 preflight 通过后创建 optimizer。`--resume` 只能加载通过 checksum 与语义 hash 核对的 `latest.json`，不能从新 run 或失败 run 拼接状态。

## 维护要求

训练器必须严格保证单张 A100、BF16 优先、smoke 支持、断点恢复合同和配置随 checkpoint 保存；新增、重命名或修改 runner 时必须在同一提交同步更新对应目录的 `README.md`、开发文档与测试。
