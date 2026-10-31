# 开发计划

最后更新：2026-10-30

## 背景与范围

本计划把训练合同拆成可交付的小闭环，先完成 baseline 和数据门禁，再实现 SFT、DPO、RL。目标是每个阶段都能在全新 Colab runtime 用固定 manifest、配置、seed 和 Git SHA 重跑；本阶段不把未实现的 runner 或 notebook 当作完成。

## 顺序

1. Phase 0：环境探针、官方模型加载、单条推理和 evaluator。
2. Phase 1：manifest builder、质量检查和固定切分。
3. Phase 2：SFT runner 与恢复合同。
4. Phase 3：DPO pair builder、reference logprob 和 preference gate。
5. Phase 4：RL rollout、reward、KL 和 held-out gate。
6. Phase 5：Colab notebook 串联和规模化运行。

## 阶段交付物与完成条件

| 阶段 | 必须交付 | 通过条件 | 失败后的动作 |
| --- | --- | --- | --- |
| Phase 0 | 环境探针、base 推理、CER evaluator、最小 Colab 编排 | clean/degraded 各 1 条成功；失败行可记录；Drive 产物能读回 | 修复依赖/API/音频前端，不启动训练 |
| Phase 1 | manifest builder/validator、五类 scenario、固定切分 | schema、音频、source 泄漏、hash 和配额全通过 | 标记 `insufficient_data` 或 `data_failed` |
| Phase 2 | SFT runner、LoRA、checkpoint/resume、val gate | 5–20 step smoke 恢复通过；相对 base 的 gate 写出结果 | 保留失败 run，调整新配置新 run |
| Phase 3 | DPO pair builder、离线 reference logprob、preference gate | pair 审计和 held-out preference、CER gate 全通过 | 不生成 DPO release |
| Phase 4 | RL rollout/reward/GRPO、reference freeze、held-out gate | reward、raw KL、reference checksum、clean/degraded gate 全通过 | 不生成 RL release |
| Phase 5 | 主 notebook、规模化启动和最终 test 报告 | run report、test 指标、baseline 对比和文档结论齐全 | 阶段保持未完成 |

每个阶段实现前先更新对应专项文档和目录 README；实现后先跑本地合同测试，再跑 Colab preflight 和 smoke，最后才允许扩大数据或步数。

## 设计与接口约定

- 训练、评测和编排 CLI 必须支持 `--config`、`--output-dir` 和 `--seed`；数据/推理工具可用显式参数替代，但必须把最终解析配置写入结果。训练和正式评测还支持 `--run-id`；trainer 另支持 `--manifest`、`--val-manifest` 和 `--resume`。DPO 另支持 `--pair-manifest`。
- `--resume` 只恢复 `04_training_plan.md` 中核对通过的 checkpoint。推理和评测按配置重跑，不恢复 optimizer。
- 代码不得依赖 notebook 中的全局变量。
- manifest 路径、模型路径和 Drive 根目录由配置或参数传入。
- 错误样本写入 JSONL；批处理默认继续执行，并在汇总中报告失败数。

所有入口必须返回非零退出码表示合同失败，并同时写结构化事件；单条数据失败只写该行 `error`，不能悄悄吞掉异常。CLI 的 `--help`、schema 版本和输出文件名属于公共接口，变更时同步更新训练/测试文档。

## 测试

每一阶段先用合成 fixture 验证 schema、hash、错误退出码和 README 合同，再进行 Colab preflight。训练阶段额外执行 5–20 step smoke、保存/中断/恢复和固定 val 评测；DPO/RL 的算法单测不能以 loss 下降代替数学合同。所有测试命令、夹具和通过线记录在 `05_testing_plan.md`。

## 完成条件

实现前先补充本目录和 `05_testing_plan.md`；实现后完成 smoke、恢复、固定 manifest 评测和目录 README 合同测试；结果写入 `00_progress.md`。每个 run 还必须生成 `run_report.md`（或等价 JSON），列出环境、config、manifest、父版本、指标、gate、错误和恢复记录。

## Phase 1 入口

- `configs/data/manifest_mvp.yaml` 保存正式配额和 seed 42。当前 `source_version` 是 `aishell1-openslr33-pilot-20261015`，`audited` 是 true。
- `scripts/build_manifest.py` 读取源 JSONL，写出五类场景、固定切分和 PCM16 WAV。没有官方身份的源仍按 source id 哈希切分。带 `speaker_id` 和 `official_split` 的源保留官方切分，`dev` 进入 `val.jsonl`。
- `scripts/manifest_validate.py` 独立检查 schema、音频、source、speaker、文本、音频泄漏和配额。说话人跨 split 时拒绝后出现的 source，不挪动样本。
- 只有校验退出码 0 的目录可以交给推理或训练。正式配置现在是 `source_version: aishell1-openslr33-pilot-20261015`、`audited: true`。P13 的 pilot manifest 在补上这个审阅标记后应退出码 0。行数不够的小样本仍是 `insufficient_data`。`scripts/sample_aishell_pilot.py` 只负责抽样，不改配置。

## Phase 2 入口

- `configs/train/sft_smoke.yaml` 是 6 step smoke，不是 100 step pilot。`publishable` 为 false。
- `train/train_sft.py` 是 SFT runner。`train/build_sft_smoke_subset.py` 从 pilot manifest 抽 10 个 train source 和 2 个 val source，每个 source 保留 clean、noise、reverb、far_field、dropout。
- 本地合同测试已通过。Colab run `20261015T080308_sft_06b_zh_seed42` 已做第 5 step 保存和第 6 step 恢复，记录在 `12_phase2_run_log.md`。smoke 的小 val 与 2500 条 val base 不是同一份 manifest，gate 记为 `incomparable`。
- 100 step pilot 的已执行配置是 `configs/train/sft_pilot.yaml`，学习率 `1e-4`。它使用完整 train manifest 和固定 2500 条 val，父数字在 `configs/train/aishell_val_base_20261015.json`。Drive 没挂上时 pilot 直接退出。smoke 才允许 `sync_skipped`。
- 下一轮配置是 `configs/train/sft_pilot_lr1e5.yaml`，学习率 `1e-5`。run `20261015T234939_sft_06b_zh_seed42` 从 step 30 恢复后跑到 step 50。step 30 和 step 40 通过 2500 条 val gate，选出的 checkpoint 是 step 30。step 50 因为 degraded macro CER 高于冻结上限 `0.0` 而 `risk_stop`。`publishable` 仍是 false。run `20261015T104034_sft_06b_zh_seed42` 不恢复。解码和 gate 与已执行的 pilot 相同。
- `train/sft_release.py` 把选出的 step 收成 SFT release，并在新进程里核对全部 val 的 greedy 文本。读回通过前 `status` 保持 `staged`。已发布的 release 可以用 `hypotheses` 导出一份 manifest 的 greedy 预测。2026-10-17 的 train 假设已经落地，19685 行且与 manifest 同序。`scripts/build_dpo_pairs.py` 已在 2026-10-17 写出 train 7456 对和 val 1022 对。`train/reference_logprob.py` 已在 2026-10-17 给这 8478 对补上父版本序列 logprob。复核 40 条，绝对误差和相对误差都是 0.0。打分前去掉训练 hook 并关闭 gradient checkpointing，没有更新权重。听音审计仍是 `unlistened`。2026-10-17 用户说「不听，直接开始训练」，处置不改成已听。`train/train_dpo.py` 记录这份豁免后，run `20261017T131802_dpo_06b_zh_seed42` 已在标准 A100 上跑到 step 10 并风险停止。Drive 上的决策是 `passed=false`，degraded macro CER 上升 `0.0006826870562534199`。`publishable` 保持 false。2026-10-18 `configs/train/dpo_dropout_probe.yaml` 的 run `20261018T052928_dpo_06b_zh_seed42` 已在 step 10 结束。`DPO_LOCATE` 是 `OTHER`，dropout CER 比父版本高 `0.029492080830147455`。`passed=false`。这一步不作为父版本。不恢复 `20261017T131802`。2026-10-18 的编辑分解见 `12_phase2_run_log.md`。dropout 多出的 216 次编辑里，静音段内的删除减少了 18 次，0.55 之后的删除增加了 130 次。冻结 pair 两边剥离静音段后，train 仍有 6213 对，val 仍有 843 对。`strip_dropout_gap` 已在 `train/dpo_contract.py`，`train_dpo.py` 不调用。`scripts/strip_dpo_gap_pairs.py` 已在 2026-10-18 把剥离后的 pair 写到 `pairs_gap_strip`。train 6213 对，val 843 对，tie 1422 条，超 token 0。不覆盖 `pairs/` 或 `pairs_logprob/`。logprob 已在 2026-10-18 重算完成，train 6213、val 843，复核 40 条，绝对误差和相对误差都是 0.0，写在 `gap_strip_ref_logprobs/`。`trainable` 仍是 false。2026-10-18 用户说「开始吧」之后，`configs/train/dpo_gap_strip_probe.yaml` 的 run `20261018T132840_dpo_06b_zh_seed42` 已在标准 A100 上启动。日志有 `DPO_PREFLIGHT_OK`，训练 pair 是过滤后的 2573 条 dropout，preference val 是 310 条。2026-10-18T14:12:33Z 结束。`DPO_LOCATE` 是 `OTHER`，dropout delta `0.006417258328782094`，0.55 之后的删除是 70（父版本 57，上一次 probe 187）。`passed=false`，唯一风险是 degraded macro CER 上升 `0.0015360458765701773`。`selected_step` 保持 null。`publishable` false。不恢复 `20261018T132840`、`20261018T052928` 或 `20261017T131802`。2026-10-19 核对发现字符 `[0.40, 0.55]` 剥离在 `50.3%` 的训练 `dropout` pair 上误删了有声匹配字、保留了 `83.5%` 的短 `rejected`，叠加 `lr=5e-5` 在 3 条静音段吐出 `humanrightswatch`、2 条提前 EOS；新增 `configs/train/dpo_pilot_nondropout.yaml`，保持正式 pilot 预算（`lr=5e-6`），将 `train_scenarios` 设为 `[clean, noise, reverb, far_field]`（3640 train / 533 val pairs），直接复用 2026-10-17 冻结的 `ref_logprobs/`，全量 2500 条 val 门禁不变。2026-10-19 run `20261019T012838_dpo_06b_zh_seed42` 在 step 10 因 `degraded_macro_cer` 上升 `+0.0006144183506280737`（净增 `+18` 次编辑：`dropout +20`、`noise -2`、`reverb 0`、`far_field 0`）风险停止；根因是 3640 条非 dropout 训练对中 `36.8%` 的 `rejected` 含标点（`chosen` 0% 标点）、`48.6%` token 数不等长导致序列和 DPO 配分函数 $\log Z$ 不抵消（压低 `'，'` 并抬高 `'Human'`，使 `BAC009S0748W0156__dropout` 单条产生 `10` 次 gap 插入与 `+14` 次净增编辑），叠加 `75.5%` 极端负参考 margin（`<-2.0` nats）、781 条原位替换对中 `64.9%`（`507/781`）为 `clean` 同音词/金标错字跨四场景重复拷贝，以及 AdamW 参数扰动。同日在 `train/dpo_contract.py`、`train/dpo_schema.py` 和 `train/train_dpo.py` 支持内存级无标点原位等长替换过滤（`aligned_substitution_only: true`）、去 `clean` 同音重复拷贝过滤（`exclude_clean_identical_rejected: true`）、参考 margin 窗口过滤（`ref_margin_min: -4.0`、`ref_margin_max: 0.0`）以及训练/验证场景过滤解耦（`train_scenarios: [noise, reverb, far_field]`、`val_scenarios: [clean, noise, reverb, far_field]`），修复 `SCENARIOS` 常量导入并补齐训练结束时 `status.json`/`gate.json`/`run_report.md`/`listen_waiver.json` 的 Drive 同步，更新 `configs/train/dpo_pilot_aligned.yaml`（`warmup_steps: 5`、`eps: 1e-6`、`weight_decay: 0.0`，过滤后 274 train / 118 val pairs，不改动磁盘缓存与任何门禁阈值）。2026-10-19 run `20261019T130250_dpo_06b_zh_seed42` 在 step 10 将 `locate_signature` 收敛至 `QUIET`（`gap_insertions=0`，消除英文幻觉）、`preference_accuracy` 达 `0.6101694915254238`、`reverb` 净减 1 次编辑，但因四退化场景净增 `+4` 次编辑（`degraded_macro_cer` 上升 `+0.00013653741125067842`：`dropout +4`、`noise +1`、`reverb -1`、`far_field 0`）风险停止；2026-10-20 定位根因为整句序列和反传在共享后缀位点（$t > t_0$，占全句平均 `58.9%` token、占非零梯度位点约 `88%`）上因前缀不同（$c_t^+ \ne c_t^-$）而对正确金标后缀字施加负梯度（触发前缀出错后的暴露偏差后缀级联崩塌）并由 $+\nabla_z \log Z(c_t^-)$ 在 21 条验证样本中凭空推高逗号 `'，'`，叠加 274 条训练对中 `40.1%`（`110` 条）为多字替换噪声对（`edit_count >= 2`）。同日在 `train/dpo_contract.py`、`train/dpo_schema.py` 和 `train/train_dpo.py` 新增 `max_edit_count` 单字原位替换过滤与 `loss_token_scope: mismatched_only` 直通式差异 token 梯度掩码（前向保持完整序列和，反向仅对 `chosen != rejected` 的差异 token 求导，使单字替换位点 $c_{t_0}^+ \equiv c_{t_0}^-$ 的配分函数 100% 精确抵消且共享后缀与 `<|im_end|>` 零梯度），并将 `configs/train/dpo_pilot_aligned.yaml` 校准为 `max_edit_count: 1`、`loss_token_scope: mismatched_only`、`warmup_steps: 10`（164 train / 118 val pairs）。164 条、累积 4 时采样在 step 41 结束，这一步保存并评测；DPO 在 Adam 之前推进学习率，step 1 的实际学习率是 `5e-7`，step 10 是 `5e-6`。该配置的 run `20261020T014427_dpo_06b_zh_seed42` 在 step 10 风险停止：preference `0.5`，degraded macro CER 上升 `0.00010240305843801922`。2026-10-20 用户说「开始吧」，门禁保持硬性条件。新增 `configs/train/dpo_pilot_edit1_seq.yaml`：同一 164 条单字对和同一套门禁，`loss_token_scope` 改回 `sequence`，`warmup_steps` 改为 2，`eps` 仍是 `1e-6`，`weight_decay` 仍是 `0.0`。step 1 的学习率是 `2.5e-6`，step 2 起是 `5e-6`。不恢复 `20261020T014427`。run `20261020T024637_dpo_06b_zh_seed42` 已在标准 A100 会话 `lora-asr-dpo-seq` 上跑到 step 10 并风险停止。preference `0.576271186440678` 过了 `0.55`，degraded macro CER 上升 `0.000341343528126703`。`selected_step` 是 null。不恢复。下一份书面配置是 `configs/train/dpo_pilot_edit1_anchor.yaml`：回到 `mismatched_only` 和 `warmup_steps: 10`，每个 optimizer step 用 `dropout_anchor_weight: 0.05` 对一条 train dropout 的 `rejected_text` 做 token 平均 NLL。门禁数字不变。该配置的 run `20261020T041218_dpo_06b_zh_seed42` 在 step 40 通过门禁，step 41 把 `BAC009S0748W0156__dropout` 翻成 Human Rights Watch。随后的停步配置是 `configs/train/dpo_pilot_edit1_anchor_stop40.yaml`，只把 `max_steps` 改成 40。2026-10-20 用户说「开始吧」。run `20261020T114647_dpo_06b_zh_seed42` 已在标准 A100 上跑到 step 30 并风险停止。step 10 和 step 20 通过，`selected_step` 是 10。step 30 的 macro 上升 `3.4134352812673074e-05`。不恢复 `20261020T114647`，也不恢复 `20261020T041218`。同日定位 `20261020T114647` 在 step 30 失败的根因是 `train.max_steps: 40` 同时把 `HorizonScheduler` 的余弦退火分母从 `90` 步压缩成 `30` 步（step 20 学习率跌至 `3.875e-6`、step 30 暴跌至 `1.625e-6`），叠加每步仅抽 1 条 `dropout` 锚样本存在单句 NLL 波动；在 `train/sft_schema.py`、`train/train_sft.py`、`train/dpo_contract.py`、`train/dpo_schema.py` 与 `train/train_dpo.py` 中支持 `optim.scheduler_max_steps`（解耦余弦周期与采样停步，方案 1）和 `dpo.dropout_anchor_samples_per_step`（每步多条 `dropout` 锚等权平均，方案 3），并拆分为两份对比配置：`configs/train/dpo_pilot_edit1_anchor_step40_h100.yaml`（纯方案 1：100 步余弦周期、step 40 停步、每步 1 条锚，100% 逐位复现 `20261020T041218` Step 40 过关结果）与 `configs/train/dpo_pilot_edit1_anchor_step40_h100_a4.yaml`（方案 1+3：100 步余弦周期、step 40 停步、每步 4 条锚平均）。本地测试通过后待审阅决定启动顺序。2026-10-22 在标准 A100 会话 `lora-asr-dpo-h100` 上先后完成这两份配置的对比运行：纯方案 1（`configs/train/dpo_pilot_edit1_anchor_step40_h100.yaml`，run `20261022T133007_dpo_06b_zh_seed42`）在 step 10、20、30、40 四次门禁全部通过（与 `20261020T041218` 对应步逐位一致），并在 step 40 干净完赛（`status=completed`、`reason=max_steps`、全程零 `gate_risk`，`selected_step=40`，`latest.json` 指向 `step_000040`，四退化场景净减 11 个编辑、`preference_accuracy=0.6440677966101694`）；方案 1+3（`configs/train/dpo_pilot_edit1_anchor_step40_h100_a4.yaml`，run `20261022T161911_dpo_06b_zh_seed42`）在 step 10 声学 CER 净减 6 个编辑（`dropout 0`、`noise -1`、`reverb -2`、`far_field -3`），但因 `preference_accuracy=0.5`（59/118）触发 `risk_stop`（`selected_step=null`）。纯方案 1 `20261022T133007_dpo_06b_zh_seed42` 的 `step_000040` 胜出。会话 `lora-asr-dpo-h100` 已停止，`publishable` 保持 false。2026-10-23 新增 `train/dpo_release.py`、`tests/test_dpo_release.py` 和 `configs/train/aishell_val_dpo_step40_20261022.json`，将 `20261022T133007_dpo_06b_zh_seed42` 选出的 `step_000040` 封包为不可变 DPO release `dpo_06b_zh_20261022T133007_step000040` 并在独立进程做 2500 条 val greedy 读回校验（`status=published`，`adapter_sha256=1d28d3119385c178c4b1b9a1e52a68d134f33a9bfd3de36659290a184ba5c51a`）；同时新增 Phase 4 RL 模块 `train/rl_contract.py`、`train/rl_schema.py`、`train/train_rl.py`、`train/rl_release.py`、`configs/train/rl_smoke.yaml`、`configs/train/rl_pilot.yaml`、`configs/train/rl_pilot_aligned.yaml` 和 `tests/test_rl_contract.py`。同日在标准 A100 会话 `lora-asr-dpo-rl` 上完成 6-step RL Smoke（run `20261023T020031_rl_06b_zh_seed42`，第 5 步中断、第 6 步 `--resume` 恢复通过）、初始 RL Pilot（run `20261023T020835_rl_06b_zh_seed42`，step 5 风险停止并完成根因定位）以及对齐版 RL Pilot（run `20261023T042034_rl_06b_zh_seed42`，step 5 一次性通过全部门禁，`selected_step=5`，2500 条 val 总编辑数降至 `2298` / `micro_cer=0.06275259`），并通过 `train/rl_release.py` 完成 `rl_06b_zh_20261023T042034_step000005` 的封包与 2500 条 val 独立进程读回发布（`status=published`，`adapter_sha256=79df11247728f149df4999447bbcd05ae66ae5dd1c5646f48d7a9515a94ff3a2`，`matched=true`，`compared=2500`，`mismatch_count=0`）。`configs/train/aishell_val_rl_step5_20261023.json` 固化 RL Step 5 的 2500 条 val 指标。
- 长任务挂在 sub2 上，日志写到 `/root/colab-keepalive/`。

## Phase 5 入口

- `scripts/run_phase5_test_benchmark.py` 是在 SFT、DPO、RL 三阶段 release 全部完成独立进程 `val` 读回发布后，对封存 2500 条 `test.jsonl`（SHA-256 `611051822cd6a3a11a1c4df63b2434b18ee43475a7fe7569aeb13426fc4cc588`）执行一次性四阶段（`base`、`sft_06b_zh_20261015T234939_step000030`、`dpo_06b_zh_20261022T133007_step000040`、`rl_06b_zh_20261023T042034_step000005`）同口径 greedy 评测与汇总对比的 CLI 入口。
- 启动前强制核对 `test.jsonl` SHA-256、2500 条音频文件字节哈希（若本地仅存放 500 条 `clean` 源音频，可通过 `scripts/manifest_audio.py` 的 `render_scenarios` 确定性重建其余四类退化波形并与 `sha256.json` 逐文件核对零差异）、三份已发布 release 的 `status == "published"` / `publishable == true` / `adapter_sha256` / `payload_sha256`，以及统一的 `decode.eval` 哈希（`4a4fee2b67884438faa196909749fcc8dcc6126b175554ad19992bf9b4f2f642`）。
- 支持按阶段与 `sample_id` 前缀断点续跑，输出各阶段的 `predictions/<stage>.jsonl`、`eval/<stage>/metrics.json`、`eval/<stage>/scored.jsonl`、`eval/<stage>/by_scenario.csv`，以及汇总对比报告 `benchmark_summary.json`、`run_report.md`、`status.json` 并同步到 Google Drive `runs/<run_id>/`。
- `notebooks/01_qwen3_asr_post_training_pipeline.ipynb` 是主 Colab 编排 notebook，按 Phase 0 → Phase 1 → Phase 2 (SFT) → Phase 3 (DPO) → Phase 4 (RL) → Phase 5 (Test Benchmark) 串联 CLI 调用，不在单元格内隐藏模型状态或超参数。

## Phase 6（Post-MVP）Track 1 收口入口

- `configs/inference/base_eval_wer.yaml` 在保持与 `base_eval.yaml` 完全一致的 `model` 与 `decode.eval` SHA-256（`4a4fee2b67884438faa196909749fcc8dcc6126b175554ad19992bf9b4f2f642`）前提下，锁定 `wer_tokenizer: "jieba"`、`wer_tokenizer_version: "0.42.1"`、`wer_dictionary_sha256: "7197c3211ddd98962b036cdf40324d1ea2bfaa12bd028e68faa70111a88e12a8"` 与 `wer_hmm: false`。
- `evaluation/zh_cer.py`（`load_jieba_tokenizer`、`score_wer_pair`）、`evaluation/eval_cer.py` 与 `inference/config.py` 支持隔离 `jieba.Tokenizer` 词典验签与词级 Levenshtein WER（`wer`/`micro_wer`/`macro_wer` 及分场景 WER），并在 `wer_tokenizer: null` 时保持 100% 向后兼容；`train/train_sft.py` 与 `scripts/run_phase5_test_benchmark.py` 同步透传可选 `scoring` WER 字段。
- Phase 5 `20261026T101800_phase5_test_06b_zh_seed42` 完整分场景与逐句错误归因、`val.jsonl` / `test.jsonl` 四阶段离线中文 WER 重算结果与 OpenSLR 33 非商业学术研究用途合规边界归档在 `12_phase2_run_log.md`、`03_data_plan.md` 与 `00_progress.md`。

## Phase 6（Post-MVP）Track 2 AliMeeting 真实远场 OOD 评测入口

- `configs/data/alimeeting_ood_eval.yaml`（`qwen3-asr-ood-data-v1`）声明 OpenSLR 119 `Eval_Ali.tar.gz`（`CC BY-SA 4.0`，只读 OOD 用途）的非重叠单说话人过滤、近场耳麦与 8 通道远场阵列 `ch0` 同步配对、纯中文清洗及会议轮转抽样参数（`seed: 42`、`max_paired_sources: 500`）。
- `scripts/prepare_alimeeting_ood_eval.py` 解析 Praat `.TextGrid`，剔除多说话人重叠与语码转换片段，同步切出 `near_field` 与 `far_field` 16 kHz 单声道 PCM16 WAV，写出 `ood_eval.jsonl`、`rejected.jsonl`、`manifest.json`（`trainable: false`）与 `sha256.json`，并支持与 Google Drive `data/alimeeting_eval/v1_seed42/` 双向验签缓存。
- `scripts/run_alimeeting_ood_benchmark.py` 联合读取 `configs/train/rl_pilot_aligned.yaml`（冻结模型/LoRA/reward 运行时）与 `configs/inference/base_eval_wer.yaml`（锁定 `zh-cer-v1` CER + `jieba==0.42.1` WER，核对 `decode_eval_sha256` 一致），校验 `ood_eval.jsonl` 全部音频哈希与三份已发布 release（`sft`、`dpo`、`rl`），按阶段与 `sample_id` 前缀续跑评测四阶段，输出含远近场退化差与 2000 次配对源簇 Bootstrap 95% CI 的 `benchmark_summary.json`、`run_report.md`、`status.json` 并同步至 Drive `runs/<run_id>/`。

## 验收与影响

完成开发阶段要求代码、配置、测试、产物和文档五者一致；只有文档更新或本地 mock 通过不能标记阶段完成。新增文件、接口、CLI 参数或产物路径时，必须同步更新 `configs/`、`train/`、`evaluation/`、`inference/`、`scripts/`、`notebooks/` 或 `tests/` 的 README（适用者）以及 `docs/qwen3-asr/00_progress.md`。
