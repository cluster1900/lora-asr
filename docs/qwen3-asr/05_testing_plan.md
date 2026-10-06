# 测试方案

## 范围

测试覆盖数据、训练合同、推理恢复和双语评测。单元测试不下载模型或公开数据；GPU smoke 在 V100
服务器单独执行。

## 本地测试

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile evaluation/eval_wer.py
python3 evaluation/eval_wer.py --help
```

通过标准：测试全部通过、CLI 可解析、无语法错误、`git diff --check` 无错误。

## v31 失败归因修复测试

在重新占用 GPU 之前，必须完成以下只读检查：

- 训练 manifest 的 degraded 行数至少 160,000；16 个 `language|scenario` cell（含 `mixed`）各至少 640 条且单 cell 不超过总量 20%；manifest 可以保留 clean 审计行，但 `sample_strategy=degraded` 生成的 optimizer epoch 必须不含 clean。
- `sample_id`、`source_utterance_id`、`audio_sha256` 与 RL validation、DPO train/validation、release validation 和 bench/test 零交集。
- scale 停止规则在 Step 640 的第一次负增益时继续，在下一次完整评估（Step 960）的连续第二次负增益时停止；KL、zero-variance 和严重 reward drop 规则仍然立即生效。
- v31 配置的 horizon 为 2,560、保存/评估间隔为 320；训练日志必须记录实际 manifest 行数、degraded 虚拟 epoch 长度和 update/identical/no-improvement 计数。
- v31 稳定性补丁的学习率、warmup 和 KL beta 必须分别为 `2e-6`、`64`、`0.08`；gate 阈值不得被放宽。
- 10x 正式启动器必须能从最近的 `step_320`、`step_640` 等完整 checkpoint 继续，不能因 run 目录已存在而把中断任务判为不可恢复；`mixed` 场景必须在数据门禁中有明确处理结果。
- 中断的 chunk 再恢复时，必须裁掉最后一份完整检查点之后的损失行和 rollout 行，剂量不能把同一 optimizer step 加两次；manifest sha、world size 或 `scheduler.pt` 对不上时必须拒绝恢复。`pipeline_state.json` 指向半成品检查点时，续跑步必须回到最后一份完整检查点。

本轮只修改文档、配置和代码并运行 CPU/preflight 测试，不启动训练。只有上述检查全部通过，才允许进入 GPU smoke。

## 数据测试

数据下载/物化是第一个可执行阶段。模型下载、base inference、SFT、DPO 和 RL 都必须等待
`DATASET_COMPLETE.json`。

128-row smoke 必须覆盖全部 robust split、English/Chinese clean，且路径、时长、hash、配额和泄漏
检查通过。SFT、DPO、RL pilot manifest 必须分别满足 08 号合同配额；只有三个 pilot 全部通过后才允许
构建 full role pools。Bench/test 数据不能进入任何训练阶段。

data builder 的 fixture 必须验证：

- 同一命令第二次执行不重复下载或追加；中断后能补齐剩余配额。
- candidate 指向的音频缺失或 hash 改变时，该行不计入 resume 进度并被重新物化。
- 原始音频不可覆盖；处理后音频必须可读、mono、16 kHz、PCM/WAV、0.5–30 秒，并同时记录原始/处理后 hash 和处理版本。
- Robust 同一个 source index 在不同 scenario 得到同一 source utterance ID，且只进入一个
  train/validation 分区。
- Clean train/validation 分别来自配置指定 split；Bench smoke 覆盖 en/zh x real/synthetic。
- stage report 的 requested/materialized/resumed/rejected/shortage 数量与文件一致。

目录 README 合同测试继续检查维护文件清单；它不能作为 V100 训练验收。V100 还必须记录 GPU 型号、
CUDA/PyTorch 版本、dtype、attention 实现、world size 和 manifest hash。

## V100 训练与推理测试

- FP16 单 batch 前向和反向成功，loss、gradient、learning rate 均为有限值。
- 4 卡 DDP 的 global batch 等于配置值；10+2 step resume 后 checkpoint、配置、world size、global
  step 和 adapter 可恢复。
- 缺少或重复 `sample_id`、缺少 `audio` 时必须明确报错或记录单条推理错误。
- clean 与 degraded 各至少一条成功；单条失败写 `error` 并继续。
- prediction 每完成一条即 flush+fsync；重跑 `--resume` 不重复样本。

## 评测

- English 计算 WER，Chinese 计算 CER，按 scenario 聚合。
- Voices-in-the-Wild-Bench 输出 language x origin x scenario 的 32-cell macro。
- 保存 raw/normalized reference、prediction、edit count 和所有指标。
- 空 reference 硬失败；inference error 按全删除计分并进入失败率。
- 评测支持官方 `text`（兼容历史 `answer`），条件分组支持 `clean` 与 `degraded`（兼容 `atomic`/`compound`）。
- 报告 clean regression、空输出、重复输出、过长和幻觉式输出。
- Canary gate 不得用 clean+robust 混合宏平均冒充 robust 指标。

## SFT/DPO/RL 阶段验收

SFT pilot 必须在同一 manifest、同一 FP16 base、同一 evaluator 下改善至少一个 degraded 场景，同时
clean 错误率绝对增加不超过 0.02，有效输出率不少于 0.95，失败率增加不超过 0.05，且 adapter 能在新进程加载并成功 `merge_and_unload` 作为下阶段底座。

DPO 测试必须验证：

- chosen/rejected 非空、不同、可追溯，ties 和坏音频进入 rejects，Clean 数据通过扩大候选源保障足额产出；
- DPO trainer 不读取 gold/error-rate 审计字段，支持预计算参考模型 logps；
- held-out preference accuracy ≥0.55；
- 相对 SFT 至少一个 degraded scenario 改善，clean 相对 SFT 回退不超过 0.02，且相对 Base 全局累积回退不超过 0.025；
- DPO adapter 能在新进程加载并成功 `merge_and_unload` 作为 RL 底座。

RL 测试必须验证：

- 第一轮是 1 条贪心解码加 3 条采样（`temperature=1.0`、`top_p=0.95`、`top_k=50`）；采样奖励高出贪心解码至少 `0.02` 才有正优势；未打过时第二轮 8 条，反向长度固定为 2；策略项是序列求和 $-A\sum_t\log\pi$；
- v10/v11 训练前的 128 条退化探针：`projected_train_reward_mass >= 17.0` 才是 `GO_GRPO`；前 32 条贪心 `generate` 与 `transcribe` 不一致则是 `BLOCKED_DECODE_MISMATCH`，不构造优化器；
- v12 的获胜优势是原始奖励差，v13 的获胜优势是固定 `0.10`。v13 只在 v12 裁剪前 `grad_norm` 中位数 > `1.5` 时启动。`unit` 仍返回优势 1，供缺省配置使用；
- v14 从 v12 Step 4 续训，学习率和原始奖励差不变。v15 只在 v14 Step 8 为 `BLOCKED_TRANSFER` 且贪心增量仍 ≥ 0 时从 Champion 新开，学习率 `2e-5`。奖励转负、Robust 增量 ≥ `0.0005` 或 KL 停止时不切换；
- v16 仍用 `1e-5` 和原始奖励差。获胜样本还要落在贪心文本的局部编辑距离内（`local_max_relative: 0.35`，至少允许 2 个 token）。Step 8 起 Robust 增量 > 0 停止；增量 ≥ 0 且 Robust ≤ 0 时可以续到 Step 24。`BLOCKED_TRANSFER` 在这个条件下由驱动续块，不另开学习率。该运行已在 Step 8 停止，门禁 FAILED；
- v17 的过滤和学习率与 v16 相同，获胜优势改为 `unit`。该运行已在 Step 4 因贪心转负停止，门禁 FAILED；
- v18 只读恢复 v16 Step 8，优势回到 `raw_gap`。跟步规则与 v17 相同。该运行已在 Step 10 因 `raw_kl` `0.000529` 停止，门禁 FAILED：贪心 `+0.0001`，Robust `+0.000014`；
- v19 仍只读恢复 v16 Step 8。获胜优势改为 `min(奖励差, 0.05)`。该运行已在 Step 12 停止，门禁 FAILED：贪心 `−0.0001`，Robust `+0.000014`，`raw_kl` `0.000640`。梯度比不截断时小，KL 和贪心仍过不了线；
- v20 从 DPO Champion 新开，学习率改为 `5e-6`。优势仍是 `raw_gap`，局部阈值 `0.35`，`β` 和 `5e-4` 不变。不恢复 v16 Step 8 的优化器。horizon 80。该运行已在 Step 4 因贪心 `−0.0003` 停止，门禁 FAILED：Robust `+0.000202`。通过线仍是贪心 ≥ `+0.002`、Robust 六位小数 ≤ 0；
- v21 回到 v16 的 `1e-5`、`raw_gap` 和局部阈值 `0.35`。`lora.train_audio_projections: false`，目标数 196。horizon 24。该运行已在 Step 4 停止，门禁 FAILED：贪心 `−0.0002`，Robust `−0.000090`，只有奖励项失败。通过线不变；
- v22 只读恢复 v21 Step 4。配置杠杆与 v21 相同。Step 8 贪心 `+0.0001`、Robust `−0.000038`，动作 `continue`。Step 11 因 `raw_kl` `0.000604` 停止，门禁 FAILED：贪心 `−0.0004`，Robust `−0.000143`。通过线不变；
- v23 只读恢复 v22 Step 8，恢复后的学习率是 `5e-6`。2026-10-01 22:15 CST Step 12 门禁 FAILED：贪心 `−0.0004`，Robust `−0.000090`，动作 `stop`，训练状态 `STOPPED_KL`。通过线不变；
- v24 从 DPO Champion 新开。学习率、`raw_gap`、局部阈值 `0.35` 和 199 个音频投影目标与 v16 相同。`grpo.policy_token_mask: changes_only` 只让策略梯度打在替换和插入 token 上，KL 仍按整句，天花板仍是 `5e-4`。有更新的步必须在损失日志里写出小于 1 的 `policy_keep_ratio`。2026-10-02 00:13 CST Step 4 门禁 FAILED：贪心 `+0.0001`，Robust `+0.000217`。Step 1 到 Step 5 的比例是 `0.1449`、`0.1661`、`0.144`、`0.1981`、`0.1378`。00:19 CST 续块退出，一条更新的掩码全是 0。通过线不变；
- v25 仍用 `1e-5`、`raw_gap`、局部阈值 `0.35` 和 199 个目标。`signed_edit_loss_args` 把序列优势设成 `(-gap, gap)`。2026-10-02 02:52 CST Step 4 门禁 FAILED：贪心 `−0.0003`，Robust `+0.000209`，动作 `stop`。通过线不变；
- v26 保持 v25 的学习率、优势、掩码和 199 个目标。`sample_strategy: degraded_skip_regressed` 让 `noise` 和 `recording` 不进优化器，清单仍是完整的 `pilot_rl.jsonl`。2026-10-02 04:39 CST Step 4 门禁 FAILED：贪心 `−0.0005`，Robust `+0.000359`，变好的场景数是 0，动作 `stop`。通过线不变；
- v27 回到 `sample_strategy: degraded`。学习率、优势、掩码和 199 个目标不变。`include_reference_candidate: true` 时，参考文本先经 `append_reference_candidate`，再由 `local_winner_index` 决定是否更新。距离超过 `0.35` 的参考文本不训练。2026-10-02 06:21 CST Step 4 门禁 FAILED，只差奖励：贪心 `+0.0001`，Robust `−0.000007`。07:10 CST Step 7 门禁 FAILED：贪心 `+0.0002`，Robust `+0.000014`，动作 `stop`，训练状态 `STOPPED_KL`。通过线不变；
- v28 只读恢复 v27 Step 4。`apply_learning_rate_on_resume: true` 把学习率写成 `5e-6`。参考文本、`signed_edits`、`raw_gap`、`0.35` 和 199 个目标不变。不恢复 Step 7。通过线不变；
- reward 组件（asr 及各项惩罚）和最终 reward 对空输出、重复、过长、hallucination 的单测；
- rollout 按组完整：每行包含 policy checkpoint、seed、prediction、reward、KL 和 error。一组可以是 4 行或 12 行。不把 7,680 或 15,360 这个固定总行数当作 v10/v11 的通过条件；
- reference policy 冻结，KL 正则与 gradient 有限；
- 贪心 held-out mean reward 相对本次运行的 Step 0 提升 ≥0.0020，`val_decode=greedy`，验证集行数与 sha 记入 loss log；
- Robust Macro 相对 DPO 零恶化（`max_robust_macro_regression <= 0.0`，按六位小数，没有 ±1 次编辑豁免）；
- 相对 DPO 至少一个 degraded scenario 改善，clean 相对 DPO 回退不超过 0.02，且相对 Base 全局累积回退不超过 0.025，有效输出率不少于 0.95；
- 只有整道 `gate.json` 为 `PASSED` 的检查点可以导出。设计内停止落在 4/8/12 之外时，仍要给 `pipeline_state.global_step` 的完整检查点补上贪心 held-out 和 2,867 条门禁；
- RL adapter 能在新进程加载并完成 clean/degraded 推理。

没有 SFT、DPO、RL 三个阶段的 gate.json、四组 prediction、合并 release 模型和固定 test 结果，不得标记完整后训练完成。
Router 不在当前范围。

## 影响

删除历史 fixture 后，所有本地测试只使用临时合成 JSON/对象，不依赖仓库内 checkpoint、prediction
或音频文件。

## 2026-10-04 当前 v31 修订

当前 v31 修复验证：预算解析和传递、checkpoint/预测续跑合同、tail gate 极端回退与非有限数值单测；完整 pytest 与 README 合同；V100 clean/degraded 各一条推理；异步 5+1 步保存/加载/继续 smoke。正式验收同时要求原 release、paired 与新增 tail 通过，详见 `29_rl_v31_scale_design.md`。
