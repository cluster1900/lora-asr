# scripts

## 目录职责

提供环境探针、中文 manifest 构建与校验、Drive 同步管理、运行目录初始化以及阶段启动支持脚本。为各训练与评测环节提供可复现的自动化 CLI 工具。

## 当前文件清单

- `README.md`：本说明文档。
- `colab_bootstrap.py`：写入 `environment.json`、`resource_report.json`、`request.json` 和解析后的推理配置。不加载 ASR 模型。
- `prepare_phase0_audio.py`：下载官方中文样例，生成 5 dB 噪声副本和一条故意缺失的 manifest 行。
- `phase0_readback.py`：把指定产物复制到目标目录后再读回，按 SHA-256 核对。
- `run_phase0_baseline.py`：按顺序调用上面的脚本、推理和 CER。每次子命令追加到运行目录的 `command_log.md`。Drive 父目录存在时创建 Drive 根目录并读回；父目录不存在时仍做本地读回，退出码 3。
- `rebuild_phase0_env.py`：在第二份空 runtime 上安装仓库根目录 `requirements.lock` 的精确版本。`# verify:` 的 CUDA 包只核对；核对失败时退出码 2，不调用 pip。锁定的包安装成功后，如果预装 `torchao` 低于 0.16.0，就卸掉它，避免 `peft` 0.21 注入 LoRA 时抛错。不升级 `torchao`，也不替换 CUDA 版 torch。通过时写出 `rebuild_report.json`。
- `manifest_contract.py`：Phase 1 的 schema、配置解析、切分哈希、文本规则和退出码表。
- `manifest_audio.py`：把一条源音频做成 clean、5 dB noise、reverb、far_field、dropout 五条 16 kHz 波形。噪声种子来自 SHA-256，不用 Python `hash()`。
- `build_manifest.py`：读取源 JSONL，写出 `train.jsonl`、`val.jsonl`、`test.jsonl`、`rejected.jsonl`、`manifest.json`、`sha256.json` 和 `wavs/`。无官方身份时按 source id 哈希切分；整份源带 `speaker_id` 和 `official_split` 时保留官方切分。
- `manifest_validate.py`：按配置重算状态、配额、source、speaker、文本、音频泄漏和音频格式，并与 `manifest.json` 对照。
- `sample_aishell_pilot.py`：从 AISHELL-1 转写和时长表里抽出 pilot 源。train 目标 18000 秒且至少 50 名说话人，dev 和 test 各 500 条。全角拉丁字母、削波和重复文本写入拒收表。不改正式配置，不把 `audited` 设为 true。
- `build_dpo_pairs.py`：把已发布 SFT release 的 greedy 预测配成 DPO pair。chosen 是 manifest 原文，rejected 是 `prediction_raw`。tie、空输出和推理错误写入 `pair_rejected.jsonl`，超 token 写入 `overlength_pairs.jsonl`。不计算 reference logprob，不下载模型。
- `strip_dpo_gap_pairs.py`：读取冻结 null pair，只对 dropout 从两边去掉 0.40–0.55 的静音段，写到新目录 `pairs_gap_strip`。剥完相同或 rejected 为空的行写入 `gap_ties.jsonl`。不覆盖 `pairs/` 或 `pairs_logprob/`，不算 logprob，不下载模型。
- `run_phase5_test_benchmark.py`：Phase 5 封存 2500 条 `test.jsonl` 四阶段（`base`、`sft`、`dpo`、`rl`）同口径 greedy 终局评测与汇总脚本。强制核对 `test.jsonl` SHA-256、测试音频与 `manifest/sha256.json`（支持从 500 条 `clean` 测试源音频通过 `render_scenarios` 确定性重建其余 2000 条退化 WAV 并逐文件验签）以及三份已发布 release（`sft_06b_zh_20261015T234939_step000030`、`dpo_06b_zh_20261022T133007_step000040`、`rl_06b_zh_20261023T042034_step000005`），透传 `scoring` 可选锁定 WER 字段至 `evaluate`，支持阶段与 `sample_id` 前缀断点续跑，输出各阶段 `predictions/<stage>.jsonl`、`eval/<stage>/metrics.json`、`eval/<stage>/scored.jsonl`、`eval/<stage>/by_scenario.csv` 以及 `benchmark_summary.json`、`run_report.md`、`status.json` 并同步到 Drive。
- `prepare_alimeeting_ood_eval.py`：Phase 6 Track 2 AliMeeting Eval（OpenSLR 119，`CC BY-SA 4.0`）只读真实远场 OOD 评测集构建脚本。使用纯标准库解析近场与远场 Praat `.TextGrid`，跨全场会议说话人轨剔除多说话人重叠、副语言噪声标记与拉丁语码转换，按 `seed: 42` 跨会议轮转抽取配对源语句，同步切出 `near_field`（耳麦单声道）与 `far_field`（8 通道阵列 `ch0`）16 kHz PCM16 WAV，写出 `ood_eval.jsonl`、`rejected.jsonl`、`manifest.json`（`trainable: false`）与 `sha256.json`，支持与 Drive `data/alimeeting_eval/v1_seed42/` 验签缓存与恢复。
- `run_alimeeting_ood_benchmark.py`：Phase 6 Track 2 AliMeeting 真实远场近/远场配对 OOD 四阶段（`base`、`sft`、`dpo`、`rl`）同口径 greedy 评测与统计汇总脚本。联合读取 `configs/train/rl_pilot_aligned.yaml` 与 `configs/inference/base_eval_wer.yaml`（强制核对 `decode_eval_sha256` 一致、`trainable == false`、`ood_eval.jsonl` 与全部配对 WAV SHA-256 及三份已发布 release 谱系），同时计算 `zh-cer-v1` CER、锁定 `jieba==0.42.1` 中文 WER、远近场退化差与 2000 次配对源簇 Bootstrap 95% CI，支持阶段与 `sample_id` 前缀断点续跑，并将全套预测、评分、`benchmark_summary.json`、`run_report.md` 和 `status.json` 读回校验同步至 `<DRIVE_ROOT>/runs/<run_id>/`。

`drive_sync.py` 还没有实现。checkpoint 恢复属于 Phase 2。正式配置的 `source_version` 是 `aishell1-openslr33-pilot-20261015`。

## 主要输入输出

- **输入**：原始音频与标注、Drive 挂载路径、配置 YAML、随机种子以及待同步的本地 checkpoint 目录。
- **输出**：结构化环境报告 `environment.json`、`command_log.md`（每条子命令的命令行、退出码、stdout、stderr）、`run_report.md`。Phase 1 构建器另写四个 JSONL、`manifest.json`、`sha256.json` 和 `wavs/`。`sha256.json` 的 `generated_at` 不参与两次构建是否相同的比较。Drive checkpoint 和 DPO 审计还没有实现。

## 使用入口

```bash
python3 scripts/run_phase0_baseline.py \
  --config configs/inference/base_eval.yaml \
  --output-dir /content/run/phase0_baseline_20261013a \
  --run-id phase0_baseline_20261013a \
  --seed 42 \
  --drive-root /content/drive/MyDrive/qwen3-asr
```

退出码 0 表示 clean、noise、失败行、CER 和 Drive 读回都完成。退出码 3 表示转写和本地读回完成，但 Drive 根目录不存在。退出码 1 或 2 表示这条 baseline 没有跑完。

环境锁重建：

```bash
python3 scripts/rebuild_phase0_env.py \
  --lock requirements.lock \
  --output-dir /content/run/phase0_lock_rebuild_20261013
```

退出码 0 表示 verify 包、安装包和 CUDA 都与锁一致。退出码 2 表示安装前的 CUDA 包已经偏离，pip 没有运行。退出码 1 表示 pip 失败，或安装后的版本、CUDA 与锁不一致。

Phase 1 构建和校验：

```bash
python3 scripts/build_manifest.py \
  --config configs/data/manifest_mvp.yaml \
  --sources <source.jsonl> \
  --output-dir <manifest-dir> \
  --seed 42
python3 scripts/manifest_validate.py \
  --manifest <manifest-dir> \
  --config configs/data/manifest_mvp.yaml \
  --check-audio
```

pilot 抽样：

```bash
python3 scripts/sample_aishell_pilot.py catalog \
  --archive /content/aishell/data_aishell.tgz \
  --output-dir /content/aishell/pilot_catalog
python3 scripts/sample_aishell_pilot.py select \
  --transcript aishell_transcript_v0.8.txt \
  --durations /content/aishell/pilot_catalog/durations.jsonl \
  --output-dir /content/aishell/pilot_sample \
  --seed 42
```

`catalog` 只读整包，写出 `durations.jsonl` 和 `catalog.json`。`select` 写出 `source.jsonl`、`selected.jsonl`、`rejected.jsonl` 和 `selection.json`。配额不够时退出码 3，并且不留下 `source.jsonl`。输入缺失或说话人切分数量不对时退出码 2。默认期望 train/dev/test 说话人是 340/40/20，夹具用 `--expect-speakers` 改成小数量。

`--seed` 必须与配置里的 seed 相同。哈希模式的源 JSONL 每行只有 `source_utterance_id`、`audio`、`text`，可选 `language`。官方模式还要有 `speaker_id` 和 `official_split`（`train`、`dev` 或 `test`），可选 `source_package`。`dev` 写入 `val.jsonl`。两种模式不能出现在同一份源文件里，否则退出码 2。`audio` 相对源文件所在目录。构建器退出码与校验器相同：0 是 `passed`，2 是配置、seed 不匹配、源文件缺失或混用切分模式，3 是 `insufficient_data`，4 是 `data_failed`，1 是 schema、泄漏、坏音频或 hash 冲突。校验器缺少 `--check-audio` 时退出码 2。正式配置 `audited` 为 true。行数不够时退出码仍是 3。`audited: false` 的配置即使行数够也必须得到 3，不能交给 SFT。

DPO pair：

```bash
python3 scripts/build_dpo_pairs.py \
  --train-manifest results/phase1_aishell_pilot_20261015/manifest/train.jsonl \
  --train-predictions <train_predictions.jsonl> \
  --val-manifest results/phase1_aishell_pilot_20261015/manifest/val.jsonl \
  --val-predictions <val_scored_or_predictions.jsonl> \
  --release-dir <published-sft-release> \
  --tokenizer-dir <qwen3-asr-tokenizer> \
  --output-dir <pair-dir> \
  --seed 42 \
  --max-transcript-tokens 512
```

退出码 0 表示 val pair 至少 100 且每个场景至少 10，`pair_metadata.json` 的 `trainable` 仍是 false。退出码 3 表示文件已写出但 val 数量不足。退出码 1 表示 release 未发布、id 对不齐、音频损坏或跨 split 泄漏，不留下 pair 文件。`--contract-test-char-tokens` 只用在合同测试里，正式 pair 必须用 `--tokenizer-dir`。EOS 是 `<|im_end|>`，token 数是正文 token 再加 1。

dropout 静音段另写：

```bash
python3 scripts/strip_dpo_gap_pairs.py \
  --train-pairs <frozen-null/train_pairs.jsonl> \
  --val-pairs <frozen-null/val_pairs.jsonl> \
  --expect-train-sha256 d958d602dcc4a252ac75ce36d99536e8a53f48fa72929d1b69ead2647dba3f1b \
  --expect-val-sha256 a7591e86f5a33efb314e29304153fe55e2d24ffba9bdd8b2739f953568b116d2 \
  --tokenizer-dir <qwen3-asr-tokenizer> \
  --output-dir <pairs_gap_strip> \
  --seed 42 \
  --max-transcript-tokens 512
```

输出目录的最后一级必须叫 `pairs_gap_strip`。脚本核对源 SHA-256 后才写 `train_pairs.jsonl`、`val_pairs.jsonl`、`gap_ties.jsonl`、`overlength_pairs.jsonl`、`pair_audit.json`、`pair_metadata.json` 和 `pair_report.json`。dropout 的 `reference_text` 与 `chosen_text` 都是剥离后的归一化文本。其他场景的文本和 token 数不变。reference logprob 保持 null，`trainable` 是 false，审计处置是 `unlistened`。退出码 0 表示 val 数量够。退出码 3 表示文件已写出但 val 不足。退出码 2 表示参数不合法。退出码 1 表示哈希不符、目录名被拒绝或行内容不能剥，并且不留下 pair 文件。`--contract-test-char-tokens` 只用在合同测试里。

Phase 5 封存 `test.jsonl` 四阶段终局评测：

```bash
python3 scripts/run_phase5_test_benchmark.py \
  --config configs/train/rl_pilot_aligned.yaml \
  --test-manifest results/phase1_aishell_pilot_20261015/manifest/test.jsonl \
  --manifest-sha-index results/phase1_aishell_pilot_20261015/manifest/sha256.json \
  --data-config configs/data/manifest_mvp.yaml \
  --pilot-sample-tar /content/drive/MyDrive/qwen3-asr/data/aishell1/20261015/pilot_sample_audio.tar \
  --sft-release-dir <releases/sft/sft_06b_zh_20261015T234939_step000030> \
  --dpo-release-dir <releases/dpo/dpo_06b_zh_20261022T133007_step000040> \
  --rl-release-dir <releases/rl/rl_06b_zh_20261023T042034_step000005> \
  --output-dir runs/<run_id> \
  --drive-root /content/drive/MyDrive/qwen3-asr \
  --run-id <YYYYMMDDTHHMMSS_phase5_test_06b_zh_seed42> \
  --seed 42
```

脚本按顺序评测 `base`、`sft`、`dpo`、`rl` 四个阶段，若本地缺失测试音频且提供 `--pilot-sample-tar`，会先从 tar 包解出 500 条 `clean` 测试源音频，再确定性生成其余 2000 条退化音频（针对跨 CPU 架构 SIMD `float32` 归约导致的 `noise` 缩放因子 `±8 ULP` 尾数差自动对齐并校验）并对全部 2500 条 WAV 与 `sha256.json` 验签；支持按阶段与 `sample_id` 前缀断点续跑，并在 Drive 根目录存在时将预测、评分、`benchmark_summary.json`、`run_report.md` 和 `status.json` 同步至 `<DRIVE_ROOT>/runs/<run_id>/`。

Phase 6 Track 2 AliMeeting Eval 只读 OOD 物化与四阶段评测：

```bash
python3 scripts/prepare_alimeeting_ood_eval.py \
  --config configs/data/alimeeting_ood_eval.yaml \
  --output-dir /content/alimeeting_ood/v1_seed42 \
  --drive-cache-dir /content/drive/MyDrive/qwen3-asr/data/alimeeting_eval/v1_seed42 \
  --archive-path /content/alimeeting_raw/Eval_Ali.tar.gz \
  --download-if-missing \
  --seed 42

python3 scripts/run_alimeeting_ood_benchmark.py \
  --train-config configs/train/rl_pilot_aligned.yaml \
  --infer-config configs/inference/base_eval_wer.yaml \
  --ood-manifest /content/alimeeting_ood/v1_seed42/ood_eval.jsonl \
  --manifest-sha-index /content/alimeeting_ood/v1_seed42/sha256.json \
  --sft-release-dir /content/drive/MyDrive/qwen3-asr/releases/sft/sft_06b_zh_20261015T234939_step000030 \
  --dpo-release-dir /content/drive/MyDrive/qwen3-asr/releases/dpo/dpo_06b_zh_20261022T133007_step000040 \
  --rl-release-dir /content/drive/MyDrive/qwen3-asr/releases/rl/rl_06b_zh_20261023T042034_step000005 \
  --output-dir runs/<run_id> \
  --drive-root /content/drive/MyDrive/qwen3-asr \
  --run-id <YYYYMMDDTHHMMSS_phase6_alimeeting_ood_06b_zh_seed42> \
  --seed 42
```

`prepare_alimeeting_ood_eval.py` 退出码：0 为 `passed`（配对源语句不少于 `min_paired_sources` 且全部 WAV 与 `sha256.json` 验签通过）；2 为配置/种子或参数错误；3 为 `insufficient_data`；1 为归档损坏、目录缺失或哈希校验失败。`run_alimeeting_ood_benchmark.py` 退出码：0 为四阶段评测与 Drive 读回完成；2 为配置、非只读 manifest、音频验签或未发布 release 校验失败。

`drive_sync.py` 必须先写本地 staging 和 `sha256.json`，再复制 `.incoming`，读回核对后才更新 `latest.json`。manifest builder 必须同时写 schema 版本、source 泄漏审计和 rejected 行，不能只返回一个行数。

## 维护要求

脚本职责、CLI 参数、输入输出 schema 变更，以及新增、重命名或删除脚本时，必须在同一提交同步更新对应目录的 `README.md`、开发文档和测试；严禁将训练状态只记录在 shell 临时变量中。
