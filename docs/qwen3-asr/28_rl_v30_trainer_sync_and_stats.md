# RL v30 前置整改：训练器同步、配置契约与统计判据

## 背景

2026-10-03 对 v10–v29 的复盘发现两类问题，且都发生在 v30A 启动之前：

1. **仓库训练器和 V100 实际训练器不是同一份。** V100 `/data/mega-asr/repo/train/train_rl.py` 为 3,542 行（md5 `60f5910b874b51a6b3497378d5c39183`），含 `signed_edits`、`raw_gap`、`local_max_relative`、`include_reference_candidate`、`second_round`、`apply_learning_rate_on_resume`、`train_audio_projections` 开关；仓库里的 `train/train_rl.py` 只有 1,957 行，这些逻辑都没有。`v30A` 的补丁原先打在仓库那份旧文件上，配置里的大部分键会被悄悄忽略，不满足“只改采样、其余与 v29 相同”的前提，也违背 AGENTS.md 的可复现要求。服务器目录不是 git 仓库，没有版本记录。
2. **验收判据低于噪声。** 20 个检查点的贪心 held-out 增量落在 `−0.0006…+0.0005`（均值约 `−0.00007`），且与学习率、掩码、优势形态没有剂量关系；Robust 零容差由单条噪声样本即可决定。点估计门禁无法区分“训练有效”与“随机波动”。

另有两处小问题：`max_raw_kl` 在 YAML 中声明但训练器把 KL 停止线硬编码为 `5e-4`；v30A 把 `noise`、`recording` 各提到 20%（合计 40%），而这正是 v25/v29 回退最多的两个场景，计划里没有理由。

## 范围

做：

- 以 V100 训练器为基线纳入仓库，再叠加 v30A 的 `degraded_balanced` 补丁。
- 新增配置契约测试：YAML 中每个叶子键必须在代码里有消费点，已知的“只声明未消费”键必须显式登记。
- v30A 场景权重改为 7 个场景等权。
- 新增 `evaluation/paired_significance.py`：对两份 `scored.jsonl` 做逐样本配对比较，输出符号检验与 bootstrap 置信区间。
- 在 27 号方案里补充“启动前置条件”和统计验收。

不做：不启动任何训练或评测；不修改 `verify_gate.py` 的门禁阈值；不覆盖服务器文件；不写 v30a 驱动脚本（等前置条件满足再写）。

## 设计

### 1. 训练器同步

- 服务器文件只读拉取到本地，作为 `train/train_rl.py` 的基线。服务器上其余 `train/*.py` 与本地一致，不动。
- 补丁点（服务器版本里已存在对应结构）：导入 `degraded_balanced_indices`、`degraded_balanced_summary`；`build_epoch_sample_indices` 增加 `strategy_options` 与 `degraded_balanced` 分支；`train_rl` 读取 `train.sample_strategy_options` 并在 rank 0 打印 `degraded_balanced_summary=` 与 manifest hash；`--sample-strategy` 增加 `degraded_balanced` 选项。
- 本地 `train/rl_sample_strategy.py` 是服务器旧版的超集，沿用本地版本。
- 服务器同步后，仓库成为唯一来源；服务器需从仓库更新，不再在服务器上手改。

### 2. 配置契约测试

`tests/test_rl_config_contract.py`：解析 `configs/train/qwen3_asr_rl_v30a.yaml`，收集 `train.*`、`grpo.*`、`lora.*` 的叶子键名，要求每个键名作为字符串字面量出现在 `train/*.py`、`scripts/*`、`evaluation/*.py` 之一。未被消费的键必须列入测试里的 `DECLARED_ONLY` 并注明原因；集合不一致即失败，因此新增死键或误删消费点都会被发现。

### 3. v30A 场景权重

等权 `1/7`。依据：覆盖不足假设只需要“每个 cell 都有样本”，不需要偏向任何场景；等权避免把已知回退场景的采样量翻倍。代码默认权重和既有单测不变，只改 YAML。

### 4. 配对显著性工具

输入两份 `scored.jsonl`（基线、候选，必须含 `sample_id`、`error_rate`、`scenario`、`language`、`prediction_normalized`），按 `sample_id` 内连接，要求两边集合完全一致。输出：

- 逐样本 `delta = candidate − baseline` 的均值与固定 seed 的 bootstrap 95% 置信区间；
- 预测文本发生变化的样本数，其中变好、变差、持平，及双侧精确符号检验 p 值；
- 按 `language|scenario` 分组的同样统计；
- 机器可读 JSON，`--output` 指定。

### 5. 启动前置条件与统计验收（写入 27 号方案）

- 先用 v29 配置换 2 个 seed 重跑 4 步，得到 seed 噪声基线；3 个 seed 的 held-out 增量极差作为后续判读的噪声下限。
- v30A 的“通过”除原有数值线外，还要求 `paired_significance.py` 在 `degraded` 条件组上的 bootstrap 95% 区间上界 `< 0`（口径是 2,867 条 validation 的逐样本 `error_rate` 差，负为改善，不是 RL 训练内的 reward），否则只能判为“与噪声无法区分”。

### 6. 未被读取的配置键（2026-10-03 已修复）

复核发现 12 个“YAML 声明、代码不读”的键，分三类处理，行为不变：

- **停止/探针阈值的合同副本**（`train.early_held_out_reward_drop`、`probe_mass_min`、`step12_mass_floor`、`step8_transfer_mass_floor`、`step8_search_mass_floor`、`step4_lcb_fraction`、`min_step8_greedy_reward_gain`、`max_raw_kl`）以及 `grpo.train_sequences`：阈值仍只在代码里定义，但集中到 `train_rl.py` 的 `STOP_CONTRACT` / `TRAIN_SEQUENCES`，`decide_rl_stop` 与探针默认值改为读这些常量；训练启动时 `check_config_contract` 比对 YAML 副本，不一致直接报错。此前改 `max_raw_kl` 不生效、也没有任何提示，现在会拒绝启动。全部 21 份 RL 配置的现有值与常量一致，已由测试覆盖。
- **`grpo.advantage.epsilon`**：锚定模式不调用组内标准化，该键从 v30A 配置删除。
- **`reward.clip`、`reward.penalties`**：训练器只从 `reward.config_path`（`reward_config.yaml`）读取奖励组件与截断，内联副本从未生效，从 v30A 配置删除，只保留 `config_path`。

`tests/test_rl_config_contract.py` 的 `DECLARED_ONLY` 因此为空，并新增：reward 段只允许 `config_path`；所有 RL 配置通过 `check_config_contract`；修改阈值会被拒绝；KL 与 reward-drop 停止使用常量值。

## 2026-10-03 复核修复（未训练）

本次代码复核补齐了五个会让结果失真或让启动器误判的边界：`verify_gate.py` 现在显式支持历史日志的 `--held-out-decode sample`（无 `val_decode` 的旧日志只在 sample 模式接受，greedy 必须明确标记）；v31 启动器使用 `nvidia-smi` 查询所有 CUDA compute 进程并在查询不可用时拒绝启动；判定 B 清单与六个既有角色（`sft_train`、`dpo_train_pool`、`rl_train_pool`、`rl_val_pool`、`validation`、`bench_test`）按三种身份键做完整隔离检查；配对统计要求两侧的语言、场景和条件组元数据一致；未知 `sample_strategy` 现在直接报错，不会静默退回全量随机采样。未启动训练，需在 V100 上重新执行 preflight 后再考虑 smoke。

## 测试

- `python3 -m unittest tests.test_rl_sample_strategy tests.test_rl_config_contract tests.test_paired_significance tests.test_directory_readmes`。
- 同步后的 `train/train_rl.py` 必须通过 `python3 -m py_compile`，并且服务器侧已有的 `tests/test_train_rl.py`、`tests/test_rl_v28_contract.py` 在装有依赖的环境里不得回退；本地缺少依赖时，用 AST 检查确认 `degraded_balanced` 补丁点存在，并明确记录“未在 GPU 环境执行”。
- 配对工具用合成数据覆盖：完全相同输入（delta 为 0、p 为 1）、全部变好、sample_id 不一致报错、固定 seed 可复现。

## 验收

- 仓库 `train/train_rl.py` 包含 `signed_edits`、`include_reference_candidate`、`degraded_balanced`，行数接近服务器版本。
- 契约测试通过，`DECLARED_ONLY` 与实际一致。
- 上述单测全部通过，目录 README 合同测试通过。
- 没有触发任何训练、没有修改服务器。
- 未验证项（需 GPU 环境）：同步后训练器的 1-step smoke、checkpoint 恢复、rollout 审计字段。这些仍是 v30A 启动前的必做项。

## 影响

- 已有 v10–v29 产物、门禁文件和决定文件不受影响。
- 仓库训练器行为将与服务器 v29 一致；此前依赖旧仓库训练器的本地单测可能需要随之调整，改动会在测试结果里如实记录。
- 实际影响（2026-10-03 本地 venv，无 GPU、无 torch、无 ffmpeg）：`tests/test_train_rl.py` 的两个 rollout 审计夹具没有 `decode_mode: greedy` 行，不满足服务器训练器“每组恰好一条贪心行”的合同，已补上夹具并新增“缺贪心行必须报错”的用例。除 `tests/test_rl_gradient_and_reward.py`（需要 torch，本地未运行）外，其余 266 项中 264 通过、6 跳过；仅有的 2 个失败是缺 ffmpeg 的转码测试，改动前就失败，与本次无关。同步后的训练器未在 GPU 环境执行。
- v30A 的结果解释从“点估计过线”改为“数值线 + 统计区间”，更难通过，但结论更可信。

## 风险

- 服务器训练器可能仍有未纳入仓库的运行期依赖或路径假设，需在 smoke 里确认。
- 等权采样仍是假设，不保证改善；若 seed 噪声基线显示极差本身已超过 `0.001`，应停止在 4 步量级上继续比较配置。
