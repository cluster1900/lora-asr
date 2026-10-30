# 测试方案

最后更新：2026-10-30

## 背景与范围

训练链路同时涉及中文数据、官方 Qwen3-ASR API、LoRA、DPO 离线参考 logprob、RL rollout 和 Colab + Drive 恢复。单看 loss 或一次成功启动不能证明链路可复现，因此测试必须覆盖数据、推理、评测、训练、持久化和阶段门禁。本文件规定测试夹具、命令、通过标准和失败样本分析方式；数值阈值以 [`04_training_plan.md`](04_training_plan.md) 中启动前冻结的配置为准。

本阶段不下载真实模型或公开数据运行本地单元测试。模型加载、A100 显存、Drive 读回和 5–20 step smoke 属于 Colab GPU/preflight；本地测试使用合成音频路径、合成中文文本、mock logits 和临时目录。

## 测试矩阵

| 层级 | 最小夹具/输入 | 固定命令 | 通过标准 | 产物 |
| --- | --- | --- | --- | --- |
| 目录合同 | 当前受管目录与 README | `python3 -m unittest tests/test_directory_readmes.py -v` | 清单无漏项、章节齐全 | 测试输出 |
| 数据合同 | 含五类 scenario、同源冲突、坏音频的 JSONL | `python3 scripts/manifest_validate.py --manifest <path> --check-audio` | schema、路径、采样率、split、source 无泄漏；坏行有明确错误 | `manifest.json`、`sha256.json`、审计报告 |
| 推理 | 1 条 clean、1 条 degraded、1 条失败样本 | `python3 inference/run_inference.py ...` | 成功行含 raw/normalized；失败行含 `error`，批处理继续 | predictions JSONL |
| 评测 | 固定预测、gold 与锁定 `jieba` 配置 | `python3 evaluation/eval_cer.py ...` / `python3 -m unittest tests.test_eval_cer -v` | CER 与锁定 `jieba` WER 可重算，分词器版本或词典哈希不符直接拒绝，按 scenario 聚合，健康度单独统计 | `metrics.json`、`by_scenario.csv`、逐样本评分 |
| 配置/preflight | 最小 YAML、未知字段、错误 dtype/world size | `python3 train/<stage>.py --config <path> ...` | 非法配置在建 optimizer 前失败；resolved config/hash 已写出 | `preflight` 事件 |
| SFT smoke | 5–20 step，含 clean+degraded | `python3 train/train_sft.py ...` | loss/grad/lr 有限；checkpoint 完整；可恢复并再训 1 step | checkpoint、events、metrics |
| SFT release | 选出的 passed step、一份 risk stop、两行 scored | `python3 -m unittest tests.test_sft_release -v` | 只复制选出 step 的 adapter；文本不一致不发布；staged release 不能导出 hypotheses；续写必须是 manifest 前缀；不下载模型 | `release.json`、`readback.json` |
| DPO pair | chosen/rejected、tie、空输出、超 token、跨 split source/音频 | `python3 -m unittest tests.test_build_dpo_pairs -v` | tie 和空输出进 `pair_rejected.jsonl`；超 token 进 `overlength_pairs.jsonl`；泄漏、坏音频和未发布 release 失败且不留 pair 文件；val 数量不足退出码 3；不下载模型，不算 logprob | `train_pairs.jsonl`、`val_pairs.jsonl`、`pair_audit.json` |
| DPO gap strip | 冻结 null pair、dropout 静音段、其他场景原样 | `python3 -m unittest tests.test_strip_dpo_gap_pairs -v` | 纯 gap 删除进 `gap_ties.jsonl`；留下的 dropout 行 chosen 等于剥离后的 reference；其他场景文本和 token 数不变；源哈希不符或输出目录名叫 `pairs` / `pairs_logprob` 时不写文件；审计保持 `unlistened`；不下载模型，不算 logprob | `pairs_gap_strip/train_pairs.jsonl`、`val_pairs.jsonl`、`gap_ties.jsonl` |
| DPO reference | 注入的 token NLL、缓存前缀、第二次 forward、训练 hook | `python3 -m unittest tests.test_reference_logprob -v` | logprob 是监督 token NLL 的序列和再取负；绝对误差 ≤1e-4 且相对误差 ≤1e-5；前缀不对不能续写；复核失败不发布带 logprob 的 pair；打分前去掉 embedding hook 并关闭 gradient checkpointing；不下载模型 | `ref_logprobs.jsonl`、`pairs_logprob/` |
| DPO loss / gate | 相等 logprob、preference 0.5、父版本 CER、base clean 上限、`unlistened`、`dpo_gap_strip_probe.yaml`、`dpo_pilot_nondropout.yaml`、`dpo_pilot_aligned.yaml`、`dpo_pilot_edit1_seq.yaml`、`dpo_pilot_edit1_anchor.yaml` | `python3 -m unittest tests.test_dpo_contract -v` | loss 按 pair 平均；step 0 相等时 accuracy 是 0.5；0.55 且严格高于 0.5 才过 preference；clean 同时受父版本 0.01 和 base 0.02 约束；豁免不把处置写成已听；gap-strip 配置只换 pair 哈希，预算与 dropout probe 相同；`dpo_pilot_nondropout.yaml` 与 `dpo_pilot_aligned.yaml` 保持 pilot 预算且复用 2026-10-17 冻结 logprob 哈希；`aligned_substitution_only`、`exclude_clean_identical_rejected`、`max_edit_count: 1`、`loss_token_scope: mismatched_only`、`ref_margin_min`/`ref_margin_max` 与解耦的 `train_scenarios`/`val_scenarios` 仅保留无标点、等长、同 token 数的单字原位替换且落在参考 margin 窗口内的纯声学退化训练对（本机冻结对过滤后为 164 train / 118 val），直通差异 token 掩码保持前向序列和不变且仅对差异 token 反传梯度；164 条样本在 step 41 结束采样，该步保存并评测，恢复时补跑缺失 decision；DPO 在 Adam 之前推进 scheduler，step 10 的学习率倍率是 1；`dpo_pilot_edit1_seq.yaml` 复用同一套冻结门禁、pair 哈希和 164 条单字过滤，`loss_token_scope` 是 `sequence`，`warmup_steps` 是 2，语义哈希与 `dpo_pilot_aligned.yaml` 不同；`dpo_pilot_edit1_anchor.yaml` 复用同一套冻结门禁和 164 条单字过滤，`loss_token_scope` 是 `mismatched_only`，`warmup_steps` 是 10，`dropout_anchor_weight` 是 `0.05`，教师是 `rejected_text`，权重为 0 时不改变旧配置的语义哈希；`dpo_pilot_edit1_anchor_stop40.yaml` 只把 `max_steps` 改成 40，其余与锚配置相同，别的 pilot 写成 40 会被拒绝；`dpo_pilot_edit1_anchor_step40_h100.yaml`（方案 1）设置 `train.max_steps: 40` 与 `optim.scheduler_max_steps: 100`，使 step 1–40 的余弦学习率与 `max_steps: 100` 逐位一致且采样在 step 40 结束；`dpo_pilot_edit1_anchor_step40_h100_a4.yaml`（方案 1+3）在此基础上设置 `dpo.dropout_anchor_samples_per_step: 4`，由 `dropout_anchor_step_slice` 每步確定性取 4 条 `dropout` 锚按 `anchor_weight / 4` 反传，且两项可选字段在默认值下不改变任何历史配置的语义哈希；`train_dpo.py` 正确导入 `SCENARIOS` 并在结束时把 `status.json`/`gate.json`/`run_report.md` 同步到 Drive；pilot 仍拒绝含 `dropout` 的场景过滤；旧 probe 和 pilot 仍能加载；不下载模型 | `gate.json` 里 `publishable=false` |
| DPO release | 选出的 passed step 40、一份 risk stop、preference 校验、两行 scored | `python3 -m unittest tests.test_dpo_release -v` | 只复制选出 step 的 adapter 与 source 元数据；risk_stop 或 preference 未过线拒绝；读回文本不一致保持 `staged`；全部一致才写 `published` 且不改 payload hash；不下载模型 | `release.json`、`readback.json` |
| RL | 固定 G=4 候选、空/重复/截断/同 reward 组、k3 与精确条件 KL、零信号窗口、reference 冻结、组内无标点等长替换对齐与 RL release | `python3 -m unittest tests.test_rl_contract -v` | reward 分量与 `[-1, 1]` 截断、组均值中心化 advantage、同分全组零损失（含零 KL）、截断/错误整组无效、零信号窗口不推进 optimizer/scheduler、step 0 固定前缀词表条件 `raw_kl` 与 `k3_proxy_train` 区分、reference 权重 checksum 校验、组内无标点等长替换差异 token 掩码 `aligned_group_mismatch_mask`、仅非零信号组累积窗口、`dropout` 父版本假设锚、`rl_smoke.yaml`/`rl_pilot.yaml`/`rl_pilot_aligned.yaml` 解析以及 `train/rl_release.py` 打包读回正确；不下载模型 | rollout、metrics、gate、`release.json` |
| Phase 5 终局评测 | 封存 test.jsonl、500 条 clean 重建 2500 条测试波形、四阶段候选（base/sft/dpo/rl）与主 notebook | `python3 -m unittest tests.test_phase5_benchmark -v` | 拒绝未发布 release 或错误 test.jsonl 哈希；500 条 clean 确定性生成 2500 条波形并与 `sha256.json` 零差异；四阶段顺序解码、前缀断点续跑、`benchmark_summary.json` 与 `run_report.md` 汇总及 Drive 读回一致；主 notebook JSON 与 CLI 合同通过；不下载模型 | `benchmark_summary.json`、`run_report.md` |
| Drive/恢复 | 截断 checkpoint、坏 checksum、runtime 中断 | `python3 scripts/drive_sync.py ...` | `.incoming` 不可恢复；`latest.json` 只指向完整目录；global step 不重复 | sync events、latest |

## 统一指标与门禁

中文主指标是 `zh-cer-v1` CER；WER 只有配置指定分词器时计算，否则为 `null` 并记录原因。每个结果保存 `prediction_raw`、`prediction_normalized`、编辑数、参考字符数、样本 CER、空输出、重复输出、超长输出和推理错误。`micro_cer` 为编辑数总和除以参考字符数总和，`macro_cer` 为样本 CER 平均；clean/degraded 和场景改善门禁使用 `micro_cer`。

### `zh-cer-v1` 与输出健康度

`zh-cer-v1` 只用于评分，不改写训练目标或 `prediction_raw`。对参考文本和假设文本分别做：

1. Unicode NFKC。
2. `casefold`。
3. 删除 Unicode 类别以 `P`（标点）或 `Z`（分隔符，含所有空白）开头的字符。
4. 删除类别 `Cc` 的控制符。
5. 剩余字符按顺序作为 CER token。编辑距离是替换、插入、删除代价都为 1 的 Levenshtein。`score_pair` 只返回编辑总数。操作回溯在 `align_edits`，不写进 `qwen3-asr-metrics-v1`。从终点往回走时，能还原当前代价的前驱按这个顺序取第一个：相等字符的对角线、替换、删除、插入。`xy` 对 `yx` 因此记成两次替换。插入锚点是参考下标除以参考长度，参考读完后的锚点是 1.0。

参考文本归一化后为空时，评测进程失败，不把该行记成 0 错误。`sample_cer = edit_count / reference_char_count`，允许大于 1。`micro_cer = sum(edit_count) / sum(reference_char_count)`，`macro_cer` 是样本 `sample_cer` 的算术平均。比较门禁时使用未四舍五入的浮点值；JSON 里同时保存编辑数和参考字符数，使 CER 可以重算。

健康度与 CER 分开计数：

- 空输出：实际预测归一化后长度为 0。
- 重复输出：归一化字符里同一字符连续出现不少于 4 次，或长度为 2–8 的片段连续重复不少于 3 次。
- 超长输出：假设字符数严格大于 `1.5 ×` 参考字符数。
- 截断：`stop_reason == "length"`，即生成长度达到 `max_new_tokens` 且最后一个 token 不是 EOS。
- 推理错误：该行 `error` 非空。错误行仍进入 CER，假设按空串计，避免失败被记成 0 错误；空输出标志仍看实际预测，不看这个替代假设。
- 有效输出：`error` 为空且归一化预测非空。重复和超长单独限流，不从有效输出里剔除。

`wer_tokenizer` 缺省或为 null 时，`wer` 为 null，`wer_reason` 为 `wer_tokenizer_not_configured`。Phase 0 不锁定 jieba 词典，因此 `configs/inference/base_eval.yaml` 默认不计算 WER。

### 锁定 `jieba` 中文 WER 辅助指标合同

- **背景**：MVP 验收标准要求在以中文 CER 为主指标的同时保留中文 WER 辅助指标；中文无天然空格分词，若不锁定分词器版本、内置词典字节哈希与新词发现开关（HMM），跨环境或自定义词典污染会导致 WER 漂移且不可复算。
- **范围**：仅在推理或训练配置的 `scoring` 显式启用 `wer_tokenizer: "jieba"`（如 `configs/inference/base_eval_wer.yaml`）时计算并输出中文 WER；不改变任何阶段的 CER 门禁裁决、不参与 RL reward 计算，也不改动已冻结的历史 YAML 与已发布 release 哈希。
- **设计**：
  1. 配置字段（`qwen3-asr-infer-v1` 与 `qwen3-asr-train-v1` 一致）：`wer_tokenizer`（`"jieba"` 或 `null`）、`wer_tokenizer_version`（锁定 `"0.42.1"`）、`wer_dictionary_sha256`（锁定内置 `jieba/dict.txt` 的 64 位十六进制 SHA-256 `"7197c3211ddd98962b036cdf40324d1ea2bfaa12bd028e68faa70111a88e12a8"`）、`wer_hmm`（布尔值，锁定 `false`，关闭 HMM 未登录词切分以保证纯词典确定性分词）。当 `wer_tokenizer` 为 `null` 时，版本与词典哈希必须为 `null`，`wer_hmm` 必须为 `false`（在 `qwen3-asr-infer-v1` 中后三项可缺省）。`decode_eval_sha256` 仅对 `decode.eval` 求哈希，因此 `base_eval_wer.yaml` 与 `base_eval.yaml` 共享完全相同的 `decode_eval_sha256`（`4a4fee2b67884438faa196909749fcc8dcc6126b175554ad19992bf9b4f2f642`）。
  2. 分词器加载与校验（`evaluation/zh_cer.py:load_jieba_tokenizer`）：仅接受 `wer_tokenizer == "jieba"`；核对 `importlib.metadata.version("jieba") == wer_tokenizer_version` 且 `jieba` 包目录下 `dict.txt` 的文件 SHA-256 等于 `wer_dictionary_sha256`，任一不符立即抛出 `ValueError`（CLI 退出码 2，不写出半截 `metrics.json`）；使用独立实例 `jieba.Tokenizer(dictionary=str(dict_path))` 初始化并按 `(version, dict_sha256)` 缓存，杜绝全局 `jieba.add_word` / `load_userdict` 状态污染。
  3. 计分与输出（`evaluation/zh_cer.py:score_wer_pair` 与 `evaluation/eval_cer.py:evaluate`）：分词输入固定为 `zh-cer-v1` 归一化后的 `reference_normalized` 与 `scored_prediction_normalized`（推理错误行与 CER 一致，按空词序列 `[]` 计分）。词级编辑距离采用替换/插入/删除代价均为 1 的 `edit_distance(reference_words, scored_prediction_words)`，`sample_wer = word_edit_count / reference_word_count`。启用 WER 时：`scored.jsonl` 每行新增 `reference_words`、`scored_prediction_words`、`word_edit_count`、`reference_word_count`、`sample_wer`；`by_scenario` 与 `by_scenario.csv` 新增 `word_edit_count`、`reference_word_count`、`micro_wer`、`macro_wer`；顶层 `metrics.json` 令 `wer = micro_wer`、`wer_reason = null`，并写出 `micro_wer`、`macro_wer`、`word_edit_count`、`reference_word_count` 与 `wer_tokenizer_info`（含 `name`、`version`、`dictionary_sha256`、`hmm`）。未启用时所有输出字段与历史版本逐字一致。
- **测试**：`python3 -m unittest tests.test_eval_cer -v` 验证锁定 `jieba` micro/macro/scenario WER、推理错误行空词计分、未知分词器拒绝、版本号不匹配拒绝、词典 SHA-256 不匹配拒绝、`configs/inference/base_eval_wer.yaml` CLI 运行及 `wer_tokenizer=null` 向后兼容。
- **验收**：全部单元测试通过；使用 `configs/inference/base_eval_wer.yaml` 对固定 `val.jsonl`（2500 行）和封存 `test.jsonl`（2500 行）的四阶段（`base`、`sft`、`dpo`、`rl`）预测离线重算 WER，重算所得 `edit_count` 与 `micro_cer` 与已发布评测产物 100% 一致，且四阶段中文 WER 归档至运行日志与进度文档。
- **影响**：只增量扩展可选 WER 评测配置与离线/在线评分字段，不改动任何已有 manifest、`decode_eval_sha256`、训练配置哈希或已发布 SFT/DPO/RL release。

Phase 0 的 smoke manifest 是 `manifests/phase0/smoke.jsonl`，`split` 固定为 `smoke`。它只证明一条 clean、一条 degraded 和一条失败行，不是 Phase 1 的 train/val/test，也不能拿来训练。

SFT 的本地合同测试是 `python3 -m unittest tests.test_train_sft -v`。它检查未知字段、seed 不一致、解码器 LoRA 名称、pad 与 EOS 相同时的 label、token 加权窗口、epoch 尾的短窗口、checkpoint 只认 `latest.json`、`configs/train/sft_smoke.yaml`、`configs/train/sft_pilot.yaml` 对固定 val base 的门禁，以及 `configs/train/sft_pilot_lr1e5.yaml` 只把学习率改成 `1e-5` 且不改 decode hash。它还检查运行时间上限的边界、未完成评测的 `runtime_stop` 记录不能当作 gate，以及恢复时要补评一个已保存但没有 decision 的 step。这些测试不下载 `Qwen/Qwen3-ASR-0.6B`。GPU 上的 6 step 保存和恢复、100 step pilot 另记在 `12_phase2_run_log.md`。smoke 小 val 的 CER 不能代替 2500 条 val gate。

Phase 1 的本地合同测试是 `python3 -m unittest tests.test_manifest tests.test_sample_aishell_pilot -v`。校验命令必须带 `--check-audio`。pilot 抽样夹具只证明选择、拒收和退出码，不是审阅通过的语料。退出码 0 是 `passed`；2 是配置、seed、混用切分模式或缺少 `--check-audio`；3 是 `insufficient_data`；4 是 `data_failed`；1 是 schema、泄漏、坏音频或 hash 冲突。`configs/data/manifest_mvp.yaml` 已是审阅后的 pilot 配置。一条合成音达不到 val 配额 50，正式配置对它仍应退出码 3，并且不再因 `source_not_audited` 短缺。合成夹具把 `audited` 写成 true 只用于证明退出码 0 的路径，不能当作训练集。官方切分夹具要证明 `dev` 留在 val、同一说话人跨 split 被拒绝且不被挪动，以及哈希行和官方行混用时退出码 2。

比较结果前必须相同：manifest SHA-256、`normalization_version`、`decode.eval` hash、模型 revision、音频前端和评测 schema。任一不同，`gate.json.status` 必须为 `incomparable`，不能用人工解释覆盖。改变 `decode.rollout` 不得改变同一预测在 eval 下的 CER。

阶段通过条件如下，具体数字只能从冻结配置读取：

- SFT：至少一个不少于 50 条的 degraded scenario 相对 base 严格改善；clean 回退、空输出率、重复率和有效输出率满足配置；smoke 保存和恢复通过。
- DPO：held-out preference accuracy 严格高于 0.5 且达到 `min_preference_accuracy`；相对 SFT 的 clean/degraded CER gate 通过；pair 审计和 reference logprob hash 通过。
- RL：相对本 run step 0 的 greedy reward 达到配置最小提升；`raw_kl` 不超过上限；reference checksum 未变；相对 DPO 的 clean/degraded 和健康度 gate 通过。

test 只在发布前以同一 `decode.eval` 评测一次，不参与训练、pair 构造、rollout 或 checkpoint 选择。没有同一 manifest、同一 evaluator 和父版本的 baseline，阶段不得标记完成。

## 训练恢复与异常验收

每个训练阶段至少做一次“保存 → 模拟 runtime 中断 → 重新启动 → `--resume` → 再训 1 step”。测试必须比较 `global_step`、optimizer/scheduler 参数个数、CPU/CUDA RNG 是否可加载、LoRA 目标配置、manifest/config/eval hash 和 reference checksum。恢复不是从任意目录寻找最近文件，而是先读取 `latest.json` 并核对 `checkpoint_manifest.json`。

以下情况必须停止当前 run、写 `events.jsonl` 和 `status.json`，保留最后一个完整 checkpoint，且不生成 release：OOM、非有限 loss/grad、坏音频、schema 错误、Drive checksum 不一致、reference checksum 变化、恢复 hash 不匹配、clean 或健康度停止条件触发。调整配置后新建 `run_id`；不得覆盖失败 run，也不得把失败 run 的指标拼接到新 run 的 step 0。

## 失败样本分析

批量推理和评测失败不应中断整批。分析脚本按 `error_type`、`scenario`、`sample_id` 和 `source_utterance_id` 汇总，保留原始音频路径、模型/adapter、decode hash、错误文本、raw/normalized prediction 和 traceback 摘要。至少抽查每类错误 20 条，区分数据不可读、模型生成空/重复、超长幻觉、归一化问题和 evaluator bug；修复后用同一 sample ID 重放并把前后指标写入 `run_report.md`。

## 本地与 Colab 命令

实现后本地先运行：

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile scripts/*.py train/*.py inference/*.py evaluation/*.py
git diff --check
```

Colab 再运行固定 manifest 的 preflight、clean/degraded 各一条推理和 5–20 step smoke。正式阶段必须额外保存 `environment.json`、`resolved_config.yaml`、`manifest_sha256.json`、原始 predictions、`metrics.json`、`gate.json` 和 `run_report.md`。RL 阶段除验证 `rl_smoke.yaml` / `rl_pilot.yaml` 外，还须通过单元测试验证 `configs/train/rl_pilot_aligned.yaml` 的声学场景过滤、组内无标点等长替换差异 token 掩码、仅非零信号 group 累积窗口、`dropout` 父版本假设锚以及 `use_reference_adapter` 先于 policy 前向执行的版本安全约束。目录文件清单发生变化时，先更新对应 README，再重跑目录合同测试。

## 环境锁重建

`requirements.lock` 的可安装行来自 `phase0_baseline_20261013b` 的完整冻结指纹，只保留 Phase 0 要钉住的项目包。`# verify:` 目前是 `torch==2.11.0+cu130`、`torchaudio==2.11.0+cu130`、`torchvision==0.26.0+cu130`、`torchcodec==0.11.0+cu130`。

第二份空 runtime 使用：

```bash
python3 scripts/rebuild_phase0_env.py \
  --lock requirements.lock \
  --output-dir /content/run/phase0_lock_rebuild_20261013
```

通过标准：

- 安装前，每个 verify 包的 `importlib.metadata` 版本与锁相同，并且 `torch.cuda.is_available()` 为 true。任一不符时退出码是 2，进程不调用 pip。
- pip 成功后，安装行和 verify 行的版本仍与锁相同，CUDA 仍可用。
- 输出目录里有 `rebuild_report.json`，schema 为 `qwen3-asr-lock-rebuild-v1`，记录安装前后版本、pip 退出码和 CUDA 状态。
- 这条重建不把 729 行完整冻结指纹当成已经复现。

退出码 2 或安装后版本偏离时，环境锁保持未关闭。命令、退出码和报告摘要写入 `09_phase0_run_log.md`。

## 验收与影响

完成一个阶段需要同时满足：测试矩阵中该阶段的所有项目通过；smoke checkpoint 能保存、加载、继续；val gate 写出 `passed`；test 结果、baseline 对比、原始预测和归一化预测已归档；结论写入 `00_progress.md`。失败测试只能把阶段保持为 `in_progress` 或 `blocked`，不能以文档结论替代实际产物。

本方案会约束后续代码的 CLI、日志字段、checkpoint 文件名和指标 schema。它不改变已经发布的权重，也不把当前仓库尚未实现的 runner 误标为可执行；Phase 0 完成前，命令中的 runner 仅是待实现接口。
