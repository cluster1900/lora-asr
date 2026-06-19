# 开发进度

最后更新：2026-06-19

## 当前状态

当前阶段：`02 Baseline 评估` 已完成 MVP 150 全量 baseline，准备进入错误分析与 LoRA MVP 前置设计。

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

## 进行中

- `02 Baseline 评估`：`Qwen/Qwen3-ASR-1.7B` 已完成 MVP 150 全量评测。clean WER 为 1.04%，degraded-only WER 约 60.23%，empty output rate 为 0.0。下一步应做错误样本分析，并把这些结果作为 LoRA/Router 的 base 对照。
- 正在启动 `06 评测与错误分析`：先实现 scored JSONL 错误分析脚本，输出 worst cases、场景/文本长度拆分和疑似幻觉/重复/过短输出标签。

## 下一批里程碑

1. 按 [02 Baseline 评估](./roadmap/02_baseline_eval.md) 实现 Qwen3-ASR ASR baseline。
2. 按 [03 数据 MVP](./roadmap/03_data_mvp.md) 构建小规模 JSONL 数据。
3. 按 [04 音频增强](./roadmap/04_audio_augmentation.md) 实现退化增强。
4. 按 [05 LoRA 训练 MVP](./roadmap/05_lora_training_mvp.md) 实现第一版 QLoRA。
5. 按 [06 评测与错误分析](./roadmap/06_eval_and_error_analysis.md) 跑通 baseline 与微调评测。
6. 按 [07 Router MVP](./roadmap/07_router_mvp.md) 实现 clean/degraded 路由。

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
