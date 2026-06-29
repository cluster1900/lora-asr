# 测试方案

最后更新：2026-06-29

## 目标

评估 Qwen3-ASR ASR LoRA 是否能提升退化音频识别，同时不明显损害 clean speech。

## 测试模式

1. Base Qwen3-ASR-1.7B。
2. Qwen3-ASR-1.7B + LoRA always on。
3. Qwen3-ASR-1.7B + router conditional LoRA。
4. 可选外部 baseline：Mega-ASR 发布模型、Whisper、商业 ASR API。

外部 baseline 只作为对比对象，不作为实现依赖。

## 核心指标

### ASR 质量

- 英文 WER。
- 中文 CER。
- 按 scenario 聚合的 WER/CER。
- clean audio regression。
- 相对 base 的提升。

### 鲁棒性失败指标

- empty output rate。
- repeated output rate。
- hallucination-like output rate。
- excessive length ratio。
- 关键样本 missing keyword rate。

### 运行指标

- load time。
- inference time。
- real-time factor。
- peak VRAM，如果可获取。
- average tokens generated。

## 评测 JSONL

输入：

```json
{
  "audio": "/path/to/audio.wav",
  "answer": "reference transcript",
  "language": "en",
  "scenario": "noise_reverb",
  "is_degraded": true
}
```

输出：

```json
{
  "audio": "/path/to/audio.wav",
  "answer": "reference transcript",
  "prediction": "model transcript",
  "language": "en",
  "scenario": "noise_reverb",
  "metric": "wer",
  "wer": 0.1234,
  "num_edits": 3,
  "ref_len": 24,
  "empty_output": false,
  "length_ratio": 0.95,
  "mode": "lora"
}
```

## 场景桶

MVP 至少包含：

- clean
- noise
- reverb
- far_field
- clipping
- dropout
- noise_reverb
- far_field_noise

## MVP 成功标准

第一版 MVP 成功条件：

- 至少一个 degraded 场景 WER 相对下降 10%。
- router 模式下 clean WER 相对退化小于 5%。
- degraded 音频空输出率下降。
- 评测脚本可复现。

当前 Qwen3-ASR base 对照：

- 当前 LoRA 对比应优先使用 `outputs/base_recheck_mvp_150/` 中的 4bit base recheck。
- clean WER：0.008351。
- noise WER：0.450939。
- reverb WER：0.544885。
- dropout WER：0.762004。
- far_field WER：0.985386。
- degraded-only WER：约 0.685804。
- empty output rate：所有场景均为 0.0。

历史 `outputs/baseline_mvp_150/` base 指标来自早期 baseline notebook，overall WER
为 0.483925，noise/reverb/far_field 明显强于 4bit base recheck。由于 LoRA v1/v2
评测使用 4bit base 后挂载 adapter，该历史 base 只作为旧环境参考，不再作为当前
LoRA 是否改善的主判断口径。

LoRA MVP 的第一版成功门槛应以 degraded-only WER 或单场景 WER 相对 4bit base
recheck 下降为准。由于 base 没有空输出，短期内更应关注错误替换、漏词、插入、
重复和幻觉式补全是否减少。

## Base Recheck 测试

背景：

- LoRA v1/v2 评测使用 4bit 加载 base 后挂载 adapter，而历史 `baseline_mvp_150`
  base 指标来自早期 baseline notebook。
- 如果 base 与 LoRA 使用的模型加载参数、qwen-asr 版本、音频路径或 manifest 不一致，
  LoRA 对比会被污染。
- 在继续 LoRA ablation 前，需要用同一套 held-out MVP 150 音频重新跑一个 base
  复核版本。

范围：

- 输入仍使用 `data/jsonl/baseline_mvp_150.local.jsonl`。
- 音频仍使用同一批 `data/mvp_eval/audio/`。
- 默认输出到 `outputs/base_recheck_mvp_150/`，不覆盖历史 `outputs/baseline_mvp_150/`。
- 默认使用 `quantization=4bit`、`dtype=float16`、`device_map=cuda:0`，与 LoRA
  v1/v2 评测加载方式保持一致。

执行入口：

```bash
python scripts/run_qwen3_asr_base_recheck.py \
  --manifest data/jsonl/baseline_mvp_150.local.jsonl \
  --audio-root . \
  --output-dir outputs/base_recheck_mvp_150 \
  --model-id Qwen/Qwen3-ASR-1.7B \
  --dtype float16 \
  --device-map cuda:0 \
  --quantization 4bit \
  --max-inference-batch-size 1 \
  --max-new-tokens 128 \
  --language English
```

输出：

- `predictions.qwen3_asr_base_recheck.mvp_150.jsonl`
- `predictions.qwen3_asr_base_recheck.mvp_150.scored.jsonl`
- `metrics.qwen3_asr_base_recheck.mvp_150.json`
- `metrics_by_scenario.qwen3_asr_base_recheck.mvp_150.csv`
- `error_analysis/`
- `comparison.json`
- `comparison_by_scenario.csv`

通过标准：

- prediction JSONL 行数等于输入 manifest 行数。
- 推理错误数为 0，或所有错误都保存在 `error` 字段中并进入评测。
- metrics JSON 包含 overall 和 clean/noise/reverb/dropout/far_field。
- `comparison.json` 至少比较 historical base 与 recheck base；如果 LoRA v1/v2 指标存在，也一起比较。
- 新 base recheck 结果要作为后续 LoRA 对比的新候选 base，而不是直接覆盖旧 baseline。

当前执行结果：

- base recheck 已完成，150 条推理错误 0，empty output 0。
- base recheck overall WER 为 0.550313，历史 base overall WER 为 0.483925。
- 差异主要来自 noise、reverb 和 far_field，说明历史 base 与当前 LoRA 4bit
  评测口径不一致。
- 以 base recheck 为准，LoRA v1 对 noise+reverb 有约 5.45% 相对改善，LoRA
  v2 有约 2.52% 相对改善，均未达到第一版 10% 改善门槛。

## Scale-Up 成功标准

扩大规模版本成功条件：

- degraded 平均 WER 相对 Qwen3-ASR base 下降 15%-25%。
- router 模式下 clean WER 相对退化小于 3%。
- 在混合 clean/degraded 测试集上，router 模式优于 always-base 和 always-LoRA。
- 结果能在至少两个源数据集上成立。

## 人工错误分析

每次评测后抽样：

- 20 个提升最大样本。
- 20 个退化最严重样本。
- 20 个空输出或近似空输出样本。
- 20 个长幻觉样本。
- 20 个 clean speech regression 样本。

每个样本记录：

- 音频路径。
- reference。
- base prediction。
- LoRA prediction。
- scenario。
- suspected failure type。

## 回归测试

接受新训练结果前必须检查：

1. adapter 可成功加载。
2. clean 样本可正常推理。
3. degraded 样本可正常推理。
4. 10 条 mini eval 可计算 WER。
5. 无输出解析错误。
6. checkpoint metadata 存在。

## LoRA 训练前探测测试

`05A LoRA 训练前探测` 不产出 adapter，因此测试重点不是 WER，而是确认模块快照可复现、候选 target 可复核。

本地测试：

- `train/inspect_qwen3_asr_modules.py --help` 可以正常执行。
- `train/lora_targets.py` 可以通过语法检查。
- `notebooks/03_train_lora_colab.ipynb` 是合法 notebook JSON，且无执行输出、无 token、无个人密钥。

Colab 测试：

- 使用 GPU runtime。
- 项目目录为 `/content/drive/MyDrive/qwen3-asr`。
- `notebooks/03_train_lora_colab.ipynb` 能完成依赖安装、配置读取和 GPU 检查。
- 探测脚本能加载 `Qwen/Qwen3-ASR-1.7B`。
- 输出目录中存在 `module_snapshot.json`、`module_summary.csv`、`lora_target_candidates.json`、`lora_target_candidates.md`。

验收标准：

- `module_snapshot.json` 中 `modules` 数量大于 0。
- `lora_target_candidates.json` 中至少有一个候选分组。
- `lora_target_candidates.md` 能用于人工复核第一版 target。
- 探测输出已记录到进度文档，并在需要排查时提交到仓库。

## Unsloth 兼容性测试

`05B Unsloth 兼容性检查` 的目标不是训练，而是确认训练 backend 是否可用。当前结果为不兼容，训练 backend 已回退到 Transformers + PEFT。

通过标准：

- Colab GPU runtime 能安装 `unsloth` 和 `unsloth_zoo`。
- Unsloth 安装不应使用 `--force-reinstall`；如果出现 Colab 预装依赖冲突，先修复 `cuda-python==12.9.4`、`cuda-bindings==12.9.4`、`pandas==2.2.2`、`requests==2.32.4`、`protobuf<6,>=3.20.2`、`jedi>=0.16`。
- `train/check_unsloth_qwen3_asr.py` 能写出 `unsloth_compatibility.json`。
- JSON 中必须包含 `compatible`、`checks`、`errors`、`matched_targets_preview`。
- 若 `compatible=true`，`matched_target_count` 必须等于配置中的 `expected_target_count`，当前为 99。
- matched target 不得包含 `model.thinker.model.layers.` 或 `lm_head`。

失败处理：

- 如果 Unsloth 不能加载 Qwen3-ASR，记录异常类型和 traceback。
- 如果 target 匹配不精确，不进入训练。
- 兼容失败时训练实现切换到 Transformers + PEFT，推理评测仍使用 qwen-asr。

## Transformers + PEFT Smoke 测试

`05C` 只验证训练闭环，不以指标提升为目标。

通过标准：

- LoRA target 正则匹配数量为 99。
- matched target 不包含 text decoder、speech conv 或 `lm_head`。
- 可训练参数量接近 1,683,456。
- 训练 batch 的 `labels` 只在 answer token 上非 `-100`，prompt 和 padding 不参与 loss。
- 5-20 step smoke training 能完成。
- loss 非 NaN。
- adapter 可保存到 `checkpoints/qwen3-asr-1.7b-lora/`。
- adapter 可重新加载。
- 至少 1 条 clean 和 1 条 degraded 音频能完成 LoRA 推理。

## LoRA MVP 正式训练测试

`05D` 开始验证 LoRA 是否真的改善 ASR 指标。它与 `05C` 的区别是：必须使用
独立 train/val 数据训练，并在固定 held-out MVP 150 test 上与 base 对比。

### 数据测试

- `scripts/create_lora_mvp_dataset.py --help` 可执行。
- 用小参数生成 smoke train/val 时，manifest 行数等于 `split_count * scenario_count`。
- 每条样本包含 `audio`、`answer`、`language`、`scenario`、`split`、`source`、`is_degraded`、`base_utterance_id`、`seed`。
- 每条样本的音频路径存在。
- train 与 val 的 `base_utterance_id` 无交集。
- train/val 不包含 MVP 150 held-out test 的音频路径。

### 训练测试

- `configs/train/qwen3_asr_lora_mvp_train.yaml` 可解析。
- 训练 target 匹配数量仍为 99。
- 可训练参数量以 PEFT 实际统计为准，约 1,683,456。
- MVP 训练 loss 全部有限。
- 输出目录包含 `adapter/`、`processor/`、`target_modules.json`、`training_config.json`、`loss_log.jsonl`、`summary.json`。

### 推理测试

- LoRA adapter 可重新加载。
- 至少 1 条 clean 和 1 条 noise/reverb 可完成推理。
- 推理失败记录到 JSONL 的 `error` 字段，不中断整批任务。
- LoRA prediction JSONL 可被 `evaluation/eval_wer.py` 评分。
- 推理输出必须保留与 base 相同的字段，并额外记录 `mode=lora` 与 `adapter_dir`，方便后续 base-vs-LoRA 对齐分析。

### LoRA MVP 评测执行

背景：`summary.json` 和 loss 只能证明训练过程完成，不能证明 ASR 质量提升。正式
LoRA MVP 必须把 adapter 加载到同一个 Qwen3-ASR official wrapper 中，在固定
MVP 150 held-out test 上跑 always-on LoRA 推理，再和 base 指标对比。

范围：

- 使用 `inference/qwen3_asr_lora_infer.py` 加载 `checkpoints/qwen3-asr-1.7b-lora-mvp/adapter/`。
- 使用 `data/jsonl/baseline_mvp_150.local.jsonl` 作为 held-out test。
- 使用 `evaluation/eval_wer.py` 和 `evaluation/analyze_errors.py` 复用同一评测口径。
- 本阶段不训练 router，不做动态切换。

本地静态测试：

- `python3 inference/qwen3_asr_lora_infer.py --help` 可执行。
- 脚本语法检查通过。
- `notebooks/05_eval_lora_mvp_colab.ipynb` 是合法 notebook JSON，且无执行输出、无 token、无个人密钥。

Colab 运行测试：

- 先用 `--limit 2` 跑 1 条 clean 和 1 条 degraded 或前 2 条样本，确认 adapter 加载和转写输出正常。
- 再跑完整 MVP 150，输出 `outputs/lora_mvp_eval/predictions.qwen3_asr_lora_mvp.mvp_150.jsonl`。
- 使用同一评测脚本输出 scored JSONL、metrics JSON 和 scenario CSV。
- 使用错误分析脚本输出 worst cases 与场景聚合。

通过标准：

- prediction JSONL 行数等于输入 manifest 行数。
- 所有行包含 `prediction`、`error`、`mode`、`adapter_dir`。
- 推理异常必须写入 `error`，不能中断整批评测。
- metrics JSON 同时包含 overall 和 scenario。
- scenario CSV 包含 clean、noise、reverb、dropout、far_field。

### 验收测试

- 在固定 MVP 150 held-out test 上跑 LoRA always-on。
- 与 `outputs/base_recheck_mvp_150/metrics.qwen3_asr_base_recheck.mvp_150.json`
  对比；历史 `outputs/baseline_mvp_150/metrics.qwen3_asr_base.mvp_150.json`
  只作为旧环境参考。
- 报告 clean、noise、reverb、dropout、far_field 五个场景。
- noise 或 reverb 至少一个场景 WER 相对 4bit base recheck 改善，第一版
  期望接近或达到 10% 相对 WER 下降。
- clean regression 量化记录；第一版警戒线为相对退化不超过 5%。
- dropout/far_field 即使不改善也要作为观察结果记录。

### v1 LoRA MVP 评测结论

v1 已完成 held-out MVP 150 评测。按历史 base 口径，v1 曾被判断为不满足
验收标准：

- overall WER：base 0.483925，LoRA v1 0.543215，相对变差 12.25%。
- degraded-only WER：base 0.602296，LoRA v1 0.676931，相对变差 12.39%。
- clean WER：base 0.010438，LoRA v1 0.008351，小幅改善。
- noise WER：base 0.336117，LoRA v1 0.419624，相对变差 24.84%。
- reverb WER：base 0.415449，LoRA v1 0.521921，相对变差 25.63%。

base recheck 完成后，该结论需要修正：历史 base 与当前 4bit LoRA 评测口径
不一致。以 4bit base recheck 为对照，LoRA v1 对 noise+reverb 有约 5.45%
相对改善，LoRA v2 有约 2.52% 相对改善；二者仍未达到 10% 改善门槛，也没有
改善 dropout/far_field，因此当前阶段仍不能进入 router。

下一轮测试对象：

- 继续使用固定 MVP 150 held-out test。
- 继续输出 historical base、base recheck、v1 和当前版本对比。
- 优先尝试 v3 target-focus：v1 风格 99 个 target、noise/reverb-only、长短均衡采样。
- 通过标准仍然是 noise 或 reverb 至少一个场景相对 4bit base recheck 有明确改善。

### v3 LoRA MVP 测试目标

v3 的目的不是证明训练通路，而是把已观察到的弱收益放大：

- base 对照：`outputs/base_recheck_mvp_150/metrics.qwen3_asr_base_recheck.mvp_150.json`。
- 目标场景：noise、reverb。
- 观察场景：dropout、far_field。
- clean regression：必须继续量化。

通过标准：

- `target_modules.json` 中 target count 等于 99。
- `loss_log.jsonl` 覆盖 noise/reverb × short/long 四个桶。
- prediction JSONL 行数等于 150。
- WER/CER、scenario CSV 和 error analysis 均生成。

验收标准：

- noise 或 reverb 至少一个场景相对 4bit base recheck 接近或达到 10% 相对 WER 下降。
- clean WER 不相对 4bit base recheck 退化超过 5%。
- 如果只有 noise+reverb 合并改善、但单场景未接近 10%，继续 ablation，不进入 router。
