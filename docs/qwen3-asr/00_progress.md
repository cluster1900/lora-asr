# 开发进度

最后更新：2026-06-30

## 当前状态

当前阶段：`Base recheck` 已完成，历史 base 口径被确认与当前 LoRA 评测口径不一致。`05A` 训练前探测、`05B` Unsloth 兼容性检查、`05C` Transformers + PEFT smoke training、正式 v1 bootstrap 训练、v1 held-out MVP 150 评测、v2 attention-only 短跑、v2 held-out MVP 150 评测、4bit base recheck 和 v3 target-focus 训练评测均已完成。

现在的执行重点是：继续优化 LoRA，把 noise/reverb 相对 4bit base recheck 的改善推到 10% 以上。v3 已验证“v1 target 组合 + noise/reverb-only + 长短均衡采样”是目前最有效方向：noise 相对改善约 8.33%，noise+reverb 合并改善约 6.71%，但仍未达到 10% 门槛。v4 checkpoint sweep 已完成，证明单纯延长 target-focus 训练不能稳定突破 10%；600 step 对 reverb 有小幅增益，但 far_field 明显回退，router 继续暂停。

我们已经把 Mega-ASR 作为参考项目进行了初步分析，并决定设计一个独立的 Qwen3-ASR-1.7B 鲁棒 ASR 训练路径。第一个里程碑是 Colab 友好的 MVP，包含我们自己的数据管线、训练代码、推理封装和评测工具。

## 已完成

- 阅读 Mega-ASR 项目结构。
- 确认 Mega-ASR 已发布的训练代码主要是 A2S-SFT；DG-WGPO RL 代码尚未发布。
- 确认 Mega-ASR 的推理方式：Transformers 后端通过 router 动态启停 LoRA；vLLM 后端会物化 LoRA，不做逐样本路由。
- 确认 Qwen3-ASR-1.7B 支持官方 `qwen-asr` 推理接口。
- 创建 `docs/qwen3-asr` 文档工作区。
- 明确最终项目必须独立开发，Mega-ASR 只作为方法参考。
- 创建执行路线图文档，覆盖文档治理、项目骨架、baseline、数据、增强、训练、评测、router 和规模化发布。
- 创建路线图总览，明确每一步的作用，以及最终目标是完成类似 Mega-ASR 的独立鲁棒 ASR 产品。
- 完成整体文档验收，修复开发方案与 roadmap 脱节问题，并新增产品能力追踪矩阵。
- 已将原 Mega-ASR 上游工程隔离到本地忽略目录 `references/mega-asr-upstream/`，根目录作为新工程主目录。
- 已创建新工程最小骨架：`configs/`、`data/`、`evaluation/`、`inference/`、`notebooks/`、`router/`、`scripts/`、`train/`。
- 已完成 `01 独立项目骨架` 验收：上游工程隔离完成，根目录无上游主入口残留，示例 JSONL/YAML 可解析，Markdown 链接检查通过。
- 已新增 `scripts/create_smoke_audio.py`，可在本地生成 baseline smoke test 的 clean/noise 音频和本地 manifest。
- 已新增 `inference/qwen3_asr_base_infer.py` 和 `evaluation/eval_wer.py`，用于 Qwen3-ASR baseline 推理与 WER/CER 评测。
- 已新增 `notebooks/01_baseline_colab.ipynb`，用于在 Colab/Google Drive 中跑通 baseline smoke 推理与评测。
- 已新增 `scripts/create_mvp_eval_audio.py` 和 `notebooks/02_mvp_150_eval_colab.ipynb`，用于生成并评测 clean/noise/reverb/far_field/dropout 各 30 条的 baseline MVP 集。
- 已提交 `data/mvp_eval/audio/` 下 150 条 MVP smoke 音频，方便 Colab 直接 `git pull` 后运行 baseline、评测和 LoRA smoke training。
- 已新增 `notebooks/00_clone_github_colab.ipynb`，用于在 Colab/Google Drive 中 clone 或更新 GitHub 工程，并确认当前 commit 与关键修复标记。
- 已完成 `05C Transformers + PEFT smoke training` 的 Colab 20 step 验收：99 个 audio tower target、loss 非 NaN、adapter/processor/summary 已保存到本地 checkpoint 目录。该结果只作为训练通路验证，不作为 LoRA MVP 指标。
- 已完成正式 LoRA MVP bootstrap 训练产物同步：`checkpoints/qwen3-asr-1.7b-lora-mvp/` 包含 adapter、processor、target_modules、training_config、loss_log 和 summary；`summary.json` 记录 `status=trained`、`steps=600`。
- 已完成 v1 LoRA always-on held-out 评测：`outputs/lora_mvp_eval/` 包含 prediction、scored、metrics、scenario CSV 和错误分析输出。
- 已完成 v2 LoRA attention-only ablation 训练与 held-out 评测：`checkpoints/qwen3-asr-1.7b-lora-mvp-v2-attn-noise-reverb/` 和 `outputs/lora_mvp_v2_eval/` 已同步，loss 有限、adapter 可保存、推理 empty output 为 0。
- 已新增 base recheck 入口：`scripts/run_qwen3_asr_base_recheck.py` 和 `configs/baseline/qwen3_asr_base_recheck_mvp_150.yaml`，用于不覆盖历史 baseline 的前提下重跑 base 并生成对比表。
- 已完成 base recheck 输出同步：`outputs/base_recheck_mvp_150/` 包含 prediction、scored、metrics、scenario CSV、error analysis 和 base/LoRA 对比表。
- 已完成 v3 target-focus 训练与 held-out 评测：`checkpoints/qwen3-asr-1.7b-lora-mvp-v3-target-focus/` 和 `outputs/lora_mvp_v3_eval/` 已同步。v3 是当前最优 LoRA：overall WER 0.540292，noise WER 0.413361，reverb WER 0.515658，clean WER 0.008351。

## 进行中

- `Base recheck`：
  - 使用同一 MVP 150 manifest 和音频。
  - 默认用 `quantization=4bit` 复核 base，与 LoRA v1/v2 评测加载方式对齐。
  - 输出到 `outputs/base_recheck_mvp_150/`，不覆盖历史 `outputs/baseline_mvp_150/`。
  - 已完成。新 base recheck overall WER 为 0.550313，明显差于历史 base 0.483925。
  - 后续 LoRA 对比应优先使用 base recheck 口径，历史 base 仅作为旧环境参考。
- `05E LoRA MVP v2 ablation`：
  - 不进入 router。
  - 固定同一 held-out MVP 150 作为 test。
  - v1/v2 在 base recheck 口径下对 noise/reverb 有小幅收益，但幅度不足 10%，仍需要继续快速迭代。
- `05F LoRA MVP v3 target-focus`：
  - 已完成训练和 MVP 150 held-out 评测。
  - target 使用 v1 的 99 个 audio tower attention + speech projection。
  - train scenario 只使用 noise/reverb，采样按 `scenario + text_length_bucket` 均衡轮转。
  - 实际 450 step 覆盖 noise long 113、noise short 113、reverb long 112、reverb short 112。
  - 相对 4bit base recheck，v3 overall 改善约 1.82%，noise 改善约 8.33%，reverb 改善约 5.36%，noise+reverb 合并改善约 6.71%；clean 无退化，dropout/far_field 仍未改善。
- `05G LoRA MVP v4 checkpoint sweep`：
  - 已完成 Colab 训练和 160/320/480/final 600 step held-out 评测。
  - 保留 v3 的 99 target、noise/reverb-only 和 `scenario + text_length_bucket` 均衡采样。
  - `max_steps=600`，`save_steps=160`，用于评测 160/320/480/final 600 step。
  - `save_steps=160` 是为了对齐 `gradient_accumulation_steps=16` 的完整 optimizer update，避免保存半个累积周期的 adapter。
  - 结果显示没有 checkpoint 达到 10% 改善门槛。v4_0600 的 reverb WER 最低，为 0.507307，相对 base recheck 改善约 6.90%；noise+reverb 合并 WER 为 0.463466，略好于 v3 的 0.464509，但 far_field WER 升到 1.135699，导致 overall 和 degraded-only 明显差于 v3。

## 下一批里程碑

1. 暂停单纯增加 step 的路线；v4 已证明该方向会牺牲 far_field，且仍不足 10%。
2. 下一轮优先做数据/损失/target 方向的 ablation，例如加入少量 hard far_field/dropout 作为负向约束，或调整训练数据难度以匹配 held-out hard profile。
3. 继续以 v3 作为当前最稳 LoRA 对照；v4_0600 只作为 reverb 略优但泛化风险更高的对照。
4. 只有 noise 或 reverb 达到 10% 相对 WER 改善、clean 无退化，并且错误分析没有新增明显重复/幻觉风险时，才进入 router。

## 待确认问题

- 首次训练使用哪种 Colab 资源：免费版、Pro、Pro+，还是外部 GPU runtime？
- MVP 优先语言：英文、中文，还是双语？
- 首个数据源使用 LibriSpeech、Common Voice、AISHELL，还是混合数据？
- MVP 是否只做短音频，还是尽早加入 30-120 秒长音频？

## 日志

### 2026-06-07

创建第一版规划文档。推荐第一版做小而完整的独立闭环：baseline、数据生成、LoRA SFT、评测，以及简单 router 决策。

补充执行路线图文档。后续开发应按 roadmap 顺序执行，每一步先满足文档、测试和验收要求，再进入下一步。

完成整体文档验收。当前文档体系已经可以支撑后续实现，下一步应执行 [01 独立项目骨架](./roadmap/01_project_scaffold.md)。

完成参考工程隔离。原 Mega-ASR 上游源码、脚本、资源和站点文档已移动到本地忽略目录 `references/mega-asr-upstream/`，根目录转为 Qwen3-ASR Robust ASR 新工程主路径。

创建新工程最小骨架和基础配置。根目录现在以 Qwen3-ASR Robust ASR 为主，参考工程仅位于 `references/mega-asr-upstream/`，且不会进入 git。

完成 `01 独立项目骨架` 验收：示例 JSONL 解析通过，YAML 配置解析通过，本地 Markdown 链接检查通过。下一步进入 baseline 评估。

新增本地 smoke 音频生成脚本。脚本已验证可生成 16kHz mono clean/noise WAV，并输出 `data/jsonl/baseline_smoke.local.jsonl`；生成物已被 `.gitignore` 排除。

新增 baseline 推理和评测脚本。`evaluation/eval_wer.py` 已用伪 prediction 数据验证；`inference/qwen3_asr_base_infer.py` 已做语法检查，真实模型推理需在 Colab/GPU 环境运行。

完成历史 baseline smoke 链路验证。`notebooks/01_baseline_colab.ipynb` 曾使用 Google Drive 路径完成 clean/noise 各 1 条样本的推理和评测闭环。切换到 `Qwen/Qwen3-ASR-1.7B` 后，该结果只保留为流程记录，不作为当前模型指标。

### 2026-06-08

开始扩展 baseline MVP 评测集：目标为 clean、noise、reverb、far_field、dropout 各 30 条，总计 150 条。该集合用于正式数据源接入前验证批量推理、场景聚合评测和失败样本记录。

完成本地 MVP 150 音频生成。`scripts/create_mvp_eval_audio.py --force` 已生成 `data/mvp_eval/audio/` 下 150 个 wav，并输出 `data/jsonl/baseline_mvp_150.local.jsonl` 与 stats。manifest 校验通过：clean、noise、reverb、far_field、dropout 各 30 条，缺失音频 0 条。

完成本地 oracle 评测链路验证。使用 `prediction = answer` 生成 oracle prediction JSONL 后运行 `evaluation/eval_wer.py`，overall error_rate 为 0.0，empty_output_rate 为 0.0；五个场景各 30 条的 scenario-level error_rate 均为 0.0。该结果只验证评测链路，不代表 Qwen3-ASR 真实模型结果。

收到音频质量过高的反馈，已将 `scripts/create_mvp_eval_audio.py` 默认退化强度调整为 `hard`，并在 stats 中记录退化强度与近似退化统计。clean 场景保持清晰，degraded 场景用于主动压出 baseline 错误。

重新生成 hard profile MVP 150 数据集。退化统计显示：noise 平均近似 SNR -2.2575 dB，reverb 平均近似 SNR -0.4661 dB，far_field 平均近似 SNR 0.3233 dB，dropout 活跃语音接近静音比例 0.6417。manifest 仍为 150 条，五个场景各 30 条，缺失音频 0 条。

收到长文本覆盖不足反馈，决定将默认 30 条文本调整为前 15 条短句、后 15 条长句，并在 manifest 中记录 `text_length_bucket` 与 `reference_word_count`。长句用于暴露漏词、重复输出、幻觉补全和长上下文顺序错误。

完成短/长文本重生成。当前 MVP 150：每个场景 short 15 条、long 15 条；short 参考词数 6-8，平均 7.13；long 参考词数 21-27，平均 24.73；overall ref_len 从 1075 提升到 2395。oracle 评测链路仍为 error_rate 0.0。

### 2026-06-11

按新决策将项目基础模型统一切换为 `Qwen/Qwen3-ASR-1.7B`。本次切换包括：文档目录改为 `docs/qwen3-asr/`，baseline 配置改为 `configs/baseline/qwen3_asr_baseline.yaml`，训练配置改为 `configs/train/qwen3_asr_lora_mvp.yaml`，推理入口改为 `inference/qwen3_asr_base_infer.py`。

baseline 推理逻辑从通用多模态聊天模板改为 Qwen3-ASR 官方 `qwen-asr` 包的 `Qwen3ASRModel.transcribe`。默认 Colab Free 参数为 `dtype=float16`、`device_map=cuda:0`、`max_inference_batch_size=1`、`language=English`，用于降低 OOM 风险并从头重跑 baseline。

本次切换只完成工程和文档收敛；真实 Qwen3-ASR WER/CER 仍需在 Colab GPU runtime 中重新执行 `notebooks/01_baseline_colab.ipynb` 或 `notebooks/02_mvp_150_eval_colab.ipynb` 后记录。

### 2026-06-15

完成 `Qwen/Qwen3-ASR-1.7B` 在 MVP 150 hard profile 上的全量 baseline 评测。评测集包含 clean、noise、reverb、far_field、dropout 各 30 条，短文本/长文本各 15 条。

场景级结果：

| scenario | samples | num_edits | ref_len | WER | empty_output_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 30 | 5 | 479 | 0.010438 | 0.0 |
| noise | 30 | 161 | 479 | 0.336117 | 0.0 |
| reverb | 30 | 199 | 479 | 0.415449 | 0.0 |
| dropout | 30 | 364 | 479 | 0.759916 | 0.0 |
| far_field | 30 | 430 | 479 | 0.897704 | 0.0 |

汇总结果：

- overall WER：约 0.483925，1159 edits / 2395 reference words。
- degraded-only WER：约 0.602296，1154 edits / 1916 reference words。
- clean WER：0.010438，说明 clean baseline 基本正常。
- empty output rate：所有场景均为 0.0，主要失败模式不是空输出，而是幻觉式替换、漏词、插入和语义补全。

短/长文本拆分：

| scenario | bucket | samples | edits | ref_len | WER |
| --- | --- | ---: | ---: | ---: | ---: |
| clean | short | 15 | 0 | 107 | 0.000000 |
| clean | long | 15 | 5 | 372 | 0.013441 |
| noise | short | 15 | 21 | 107 | 0.196262 |
| noise | long | 15 | 140 | 372 | 0.376344 |
| reverb | short | 15 | 31 | 107 | 0.289720 |
| reverb | long | 15 | 168 | 372 | 0.451613 |
| dropout | short | 15 | 77 | 107 | 0.719626 |
| dropout | long | 15 | 287 | 372 | 0.771505 |
| far_field | short | 15 | 92 | 107 | 0.859813 |
| far_field | long | 15 | 338 | 372 | 0.908602 |

初步错误观察：

- far_field 是最严重场景，短句和长句都出现大量幻觉式替换，例如将 “Could you send the report before noon?” 识别成无关句子。
- dropout 场景 WER 接近 0.76，说明断续音频会造成严重漏词和错误补全。
- noise/reverb 仍有明显退化，但比 far_field/dropout 更可训练，适合作为第一版 LoRA 改善目标。
- clean 仅 5 个 edit，后续 LoRA 和 router 必须把 clean regression 作为硬门槛。

### 2026-06-19

已在本地对 `outputs/baseline_mvp_150/predictions.qwen3_asr_base.mvp_150.scored.jsonl` 执行 `evaluation/analyze_errors.py`，生成 `outputs/baseline_mvp_150/error_analysis/` 下的分析文件。

为方便后续排查和复盘，本项目将 `outputs/baseline_mvp_150/` 作为第一批受控 baseline 输出提交到版本库。该目录包含 Qwen3-ASR baseline 的 prediction、scored JSONL、metrics、scenario CSV 和 error analysis 结果；其他通用 `outputs/` 子目录仍默认忽略。

进入 `05A LoRA 训练前探测`。本阶段先实现模块探测脚本和 Colab 入口，输出 Qwen3-ASR 当前版本的模块快照、候选 LoRA target 和人工复核摘要。未完成探测前，不开始正式训练循环，避免 LoRA target 来自未经验证的外部假设。

完成 `05A LoRA 训练前探测`。Colab 输出已保存到 `outputs/lora_probe/qwen3_asr_1_7b/`：

- root module：`model: Qwen3ASRForConditionalGeneration`。
- total modules：703。
- audio encoder layers：24。
- text decoder layers：28。
- attention candidates：208，其中 audio tower 96、text decoder 112。
- MLP candidates：132，其中 audio tower 48、text decoder 84。
- speech projection candidates：3，分别为 `conv_out`、`proj1`、`proj2`。
- `lm_head`：1 个，默认不训练。

第一版 smoke training target 决策：

- 训练 audio tower attention：`q_proj`、`k_proj`、`v_proj`、`out_proj`。
- 训练 speech projection：`conv_out`、`proj1`、`proj2`。
- 暂不训练 text decoder、audio tower MLP、speech conv 和 `lm_head`。
- `r=8` 预计 LoRA 可训练参数约 1,683,456。

该决策已经写入 `configs/train/qwen3_asr_lora_mvp.yaml`，策略名为 `audio_tower_attention_plus_projection_smoke`。

根据训练效率和 Colab Free 显存约束，曾将训练 backend 决策调整为 Unsloth 优先。已查阅官方 Unsloth/Qwen 文档：Unsloth 明确支持 Qwen3/Qwen3 MoE 高效微调，但 Qwen3-ASR 是音频 ASR 架构，不等同于普通 Qwen3 LLM。因此新增 `05B Unsloth 兼容性检查`，先确认能否加载模型并精确匹配 audio tower target，再进入真实 smoke training。

完成 `05B Unsloth 兼容性检查`。Colab 结果显示 `compatible=false`，失败原因为 Unsloth `FastModel.from_pretrained` 走标准 Transformers AutoConfig 路径，而当前 `transformers==4.57.6` 不识别 `model_type=qwen3_asr`。该结果已保存到 `outputs/lora_probe/qwen3_asr_1_7b/unsloth_compatibility.json`。当前训练 backend 回退为 `transformers_peft`，不再继续通过依赖 pinning 强推 Unsloth。

完成 `05C Transformers + PEFT smoke training` 的工程入口实现：

- 新增 `train/peft_targets.py`，复用配置中的 include/exclude regex 精确匹配 PEFT LoRA target。
- 新增 `train/train_qwen3_asr_lora.py`，通过官方 `Qwen3ASRModel.from_pretrained` 加载底层模型和 processor，挂载 PEFT LoRA，并构造只在 answer token 计算 loss 的训练 batch。
- 更新 `notebooks/03_train_lora_colab.ipynb`，主路径改为 Transformers + PEFT smoke training；Unsloth 兼容性检查已从 notebook 执行流程中移除，仅保留历史脚本、文档结论和 `unsloth_compatibility.json`。
- 默认 smoke manifest 使用已提交的 `outputs/baseline_mvp_150/baseline_mvp_150.colab.jsonl`；本地运行可通过 `--manifest data/jsonl/baseline_mvp_150.local.jsonl` 覆盖。该 manifest 仅用于训练闭环验证，不代表正式训练集。

下一步需要在 Colab Free GPU 中执行 5-20 step smoke training，检查 `target_modules.json`、`loss_log.jsonl`、`summary.json` 和 `adapter/`，确认 loss 非 NaN 且 adapter 可保存。

Colab 首次执行 smoke training 时出现 `CalledProcessError`，该异常只表示子进程非 0 退出，根因需要查看脚本 stdout/stderr。已对 notebook 做两项收敛：

- smoke 配置默认关闭 `gradient_checkpointing`，并在 k-bit PEFT 准备阶段同步关闭，避免 Qwen3-ASR 自定义音频架构与 checkpointing/input embedding 逻辑冲突。
- notebook 的模块探测和训练 cell 改为捕获并打印 stdout/stderr tail，后续失败时可以直接看到真实异常。

第二次 Colab smoke training 捕获到真实异常：`_forward_unimplemented() got an unexpected keyword argument 'input_ids'`。根因是脚本把 PEFT LoRA 包在最外层 `Qwen3ASRForConditionalGeneration` 上，而该外层主要实现 `generate()`，没有训练用的 `forward(input_ids=..., labels=...)`。已修复为训练内部 `model.thinker`，并让 target 匹配工具以 `model.thinker` 作为探测全路径前缀，同时把 PEFT 实际 target 保持为相对 thinker 的 raw module name。

### 2026-06-20

Colab 中临时拉取 notebook 只执行了 clone/pull，看到 `Already up to date` 时无法判断 Drive 里的代码是否包含最新训练修复。为此新增正式 `notebooks/00_clone_github_colab.ipynb`：

- 空目录时 clone `https://github.com/cluster1900/lora-asr.git` 到 `/content/drive/MyDrive/qwen3-asr`。
- 已存在 git 仓库时执行 `fetch + pull --ff-only`，默认不覆盖本地修改。
- 打印当前 `HEAD`、最新提交信息和 `git status --short`。
- 检查 `resolve_training_model`、`target_root_prefix`、`gradient_checkpointing: false` 等关键标记，作为继续执行训练 notebook 前的版本验收。

随后按“只保留一个标准入口”的要求删除多余的旧 notebook，后续 Colab 拉取和更新统一使用 `notebooks/00_clone_github_colab.ipynb`。

Colab `03_train_lora_colab.ipynb` 第 19 个运行单元再次失败，真实错误为：

```text
ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported
```

根因是 Colab runtime 预装旧版 `torchao`，PEFT 在 `get_peft_model()` 注入 LoRA 时会探测该包并因版本过旧中断。当前 Qwen3-ASR smoke training 不依赖 torchao，因此修复策略是在 notebook 安装依赖后默认卸载 `torchao`，并在训练脚本启动阶段加入环境预检，提前给出明确处理建议。

再次执行后，训练已经通过模型加载和 LoRA 注入，新的失败点是：

```text
FileNotFoundError: /content/drive/MyDrive/qwen3-asr/data/mvp_eval/audio/clean/clean_0001.wav
```

根因是 `outputs/baseline_mvp_150/baseline_mvp_150.colab.jsonl` 已提交到仓库，但 `data/mvp_eval/audio/` 是生成音频目录，被 `.gitignore` 排除，不会随 GitHub 拉取自动出现在 Google Drive。当前修复策略是在 notebook 和训练脚本中提前验证所选训练样本的音频路径，缺失时先报清楚缺失样例，再提示同步音频目录。

音频提交并重新执行后，训练已经通过输入路径检查、模型加载和 LoRA 注入，新的失败点是：

```text
RuntimeError: Input type (c10::Half) and bias type (float) should be the same
```

根因是训练脚本把 `input_features` 按模型首个参数转成 `float16`，但 Qwen3-ASR 的 `audio_tower.conv2d1` 在当前 4bit 加载路径中仍保留 `float32` bias。修复策略是让音频特征跟随 `audio_tower.conv2d1` 的 dtype/device，而不是跟随模型任意首个参数。

完成 `05C Transformers + PEFT smoke training` 的 Colab 20 step 验收。训练产物已同步到本地 `checkpoints/qwen3-asr-1.7b-lora/`，但 checkpoint 目录继续按 `.gitignore` 保持不提交：

- `summary.json` 中 `status=trained`。
- `loss_log.jsonl` 共 20 行，loss 全部为有限值，`loss_min=5.84125075420161e-07`，`loss_max=4.830900192260742`，`loss_last=3.1748266220092773`。
- `target_modules.json` 中 `count=99`，`target_root_prefix=model.thinker`。
- 实际可训练参数 `trainable_params=1,683,456`，总参数 `total_params=1,178,592,896`。
- 20 step 覆盖 clean、noise、reverb、far_field、dropout 各 4 条。
- `adapter/adapter_model.safetensors`、`adapter/adapter_config.json` 和 `processor/` 均已生成。

注意：`summary.json` 中的 `estimated_lora_params` 在 4bit `Linear4bit` packed weight shape 下会被放大，不作为验收指标；本阶段以 PEFT 实际统计的 `trainable_parameter_summary.trainable_params` 为准。

### 2026-06-29

重新界定 LoRA MVP 状态：`05C` 已完成的是训练通路 smoke，不是正式 LoRA MVP。正式 LoRA MVP 从 `05D` 开始，要求独立 train/val 数据、固定 held-out test、base-vs-LoRA 指标对比和 clean regression 记录同时成立。

本轮启动项：

- 固定 MVP 150 hard profile 为 held-out test，不直接作为训练集。
- 第一版 LoRA MVP 训练数据优先覆盖 clean、noise、reverb；dropout 和 far_field 暂作为测试观察场景。
- 新增 bootstrap 数据生成入口和 MVP 训练配置后，再进入 Colab 训练。
- 新增 `notebooks/04_train_lora_mvp_colab.ipynb`，将正式 600 step bootstrap 训练从 `03_train_lora_colab.ipynb` 中拆出，避免 smoke 与正式训练混用。

已完成正式 LoRA MVP bootstrap 训练并同步 checkpoint。当前 adapter 位于
`checkpoints/qwen3-asr-1.7b-lora-mvp/adapter/`，`summary.json` 显示
`status=trained`、`steps=600`、`loss_last=0.41244810819625854`。下一步进入
LoRA always-on held-out 评测，目标是产出 `outputs/lora_mvp_eval/` 下的
prediction、scored、metrics、scenario CSV 和错误分析输出。

完成 v1 LoRA always-on held-out 评测。评测文件位于 `outputs/lora_mvp_eval/`，
输入为固定 MVP 150 hard profile，共 150 条，推理错误 0 条，empty output 0 条。

base vs LoRA v1 结果：

| scenario | base WER | LoRA v1 WER | absolute delta | relative delta |
| --- | ---: | ---: | ---: | ---: |
| overall | 0.483925 | 0.543215 | +0.059290 | +12.25% |
| degraded-only | 0.602296 | 0.676931 | +0.074635 | +12.39% |
| clean | 0.010438 | 0.008351 | -0.002087 | -20.00% |
| noise | 0.336117 | 0.419624 | +0.083507 | +24.84% |
| reverb | 0.415449 | 0.521921 | +0.106472 | +25.63% |
| dropout | 0.759916 | 0.770355 | +0.010439 | +1.37% |
| far_field | 0.897704 | 0.995825 | +0.098121 | +10.93% |

结论：v1 不满足 LoRA MVP 验收标准。虽然 clean 小幅改善，但目标场景 noise 和
reverb 均明显变差，degraded-only 也变差。逐样本对齐显示 150 条中 17 条改善、
50 条变差、83 条持平；变差集中在 noise/reverb/far_field。当前不进入 router。

下一步：进入 `05E LoRA MVP v2 ablation`，先做更保守的 attention-only、
noise/reverb-only、小学习率、短步数实验。由于步数减少，训练采样必须显式平衡
short/long，不允许因为原始 manifest 顺序导致 150 step 主要覆盖短句。评测仍使用
同一 MVP 150 held-out test。

完成 v2 LoRA attention-only 短跑和 held-out 评测。当前 adapter 位于
`checkpoints/qwen3-asr-1.7b-lora-mvp-v2-attn-noise-reverb/adapter/`，`summary.json`
显示 `status=trained`、`steps=150`、`loss_last=0.5158780813217163`、训练耗时约
166.8 秒。训练实际采样按 `scenario + text_length_bucket` 轮转：noise long 38、
noise short 38、reverb long 37、reverb short 37，满足短长文本覆盖要求。

v2 相比 v1 的实验优化：

- LoRA target 从 99 个变为 96 个，只保留 audio tower attention，移除 speech projection。
- 训练场景从 clean/noise/reverb 改为 noise/reverb-only，避免 clean 样本稀释目标 degraded 场景。
- 学习率从 2e-5 降到 1e-5，训练步数从 600 降到 150，用于降低过训和破坏性更新风险。
- 采样从普通顺序训练改为 `scenario + text_length_bucket` 均衡轮转，避免短跑只覆盖某一类长度。

base vs LoRA v1 vs LoRA v2 结果：

| scenario | base WER | LoRA v1 WER | LoRA v2 WER | v2 - v1 | v2 - base |
| --- | ---: | ---: | ---: | ---: | ---: |
| overall | 0.483925 | 0.543215 | 0.545720 | +0.002505 | +0.061795 |
| clean | 0.010438 | 0.008351 | 0.008351 | +0.000000 | -0.002087 |
| noise | 0.336117 | 0.419624 | 0.438413 | +0.018789 | +0.102296 |
| reverb | 0.415449 | 0.521921 | 0.532359 | +0.010438 | +0.116910 |
| dropout | 0.759916 | 0.770355 | 0.764092 | -0.006263 | +0.004176 |
| far_field | 0.897704 | 0.995825 | 0.985386 | -0.010439 | +0.087682 |

逐样本对齐显示，v2 相比 v1：150 条中 12 条改善、14 条变差、124 条持平。
改善主要体现在 dropout/far_field 的少量样本和 clean 稳定性；noise/reverb 作为
训练目标场景没有改善，反而分别比 v1 更差 4.48% 和 2.00%。该结论基于历史
base，对当前 4bit LoRA 评测并不公平，后续以 base recheck 口径重算。

完成 4bit base recheck。输出位于 `outputs/base_recheck_mvp_150/`，推理 150 条，
错误 0 条，empty output 0 条。新 base recheck 与 LoRA v1/v2 使用相同的
`quantization=4bit`、`dtype=float16`、`device_map=cuda:0` 口径，更适合作为
当前 LoRA 对照。

historical base vs base recheck：

| scenario | historical base WER | base recheck WER | delta |
| --- | ---: | ---: | ---: |
| overall | 0.483925 | 0.550313 | +0.066388 |
| clean | 0.010438 | 0.008351 | -0.002087 |
| noise | 0.336117 | 0.450939 | +0.114822 |
| reverb | 0.415449 | 0.544885 | +0.129436 |
| dropout | 0.759916 | 0.762004 | +0.002088 |
| far_field | 0.897704 | 0.985386 | +0.087682 |

按 base recheck 口径重算 LoRA v1/v2：

| group | base recheck | LoRA v1 | LoRA v2 | v1 delta | v2 delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| overall | 0.550313 | 0.543215 | 0.545720 | -0.007098 | -0.004593 |
| degraded-only | 0.685804 | 0.676931 | 0.680063 | -0.008873 | -0.005741 |
| noise+reverb | 0.497912 | 0.470772 | 0.485386 | -0.027140 | -0.012526 |
| dropout+far_field | 0.873695 | 0.883090 | 0.874739 | +0.009395 | +0.001044 |
| clean | 0.008351 | 0.008351 | 0.008351 | +0.000000 | +0.000000 |

结论：历史 base 指标不能继续作为 LoRA v1/v2 的公平对照。以 4bit base recheck
为准，LoRA v1 对 noise+reverb 有约 5.45% 相对改善，v2 有约 2.52% 相对改善；
clean 无退化，dropout/far_field 未改善。因此状态从“LoRA 完全失败”修正为
“目标场景有弱收益，但未达到 MVP 10% 改善门槛，仍不进入 router”。
