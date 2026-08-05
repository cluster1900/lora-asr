# 文档与阶段验收

最后更新：2026-07-22

## 背景与范围

本文件把快速 A2S 主线转换为不可模糊解释的完成条件。目标是尽早阻断错误数据或错误训练，
同时避免为了“更稳妥”重复跑中间模型和全量评测。

首轮只验收公开数据、单-adapter A2S runner 和一次正式 A2S 发布。Teacher、direct SFT、router、
RL、自建增强、50k/100k 中间模型、LR/rank/target sweep 和 50%/100% checkpoint 比较不属于
验收项，也不得作为阶段完成的隐藏前置。

## 当前文档权威顺序

1. `02_development_plan.md`：唯一执行合同。
2. `03_data_plan.md`：数据 schema、配比、缓存和防泄漏。
3. `04_colab_training_plan.md`：唯一 Colab 执行方式。
4. `05_testing_plan.md`：测试与指标门槛。
5. `06_risks_and_decisions.md`：当前决策、假设、风险和停止条件。
6. `07_document_acceptance.md`：阶段状态硬门。
7. `00_progress.md`：实际完成度、命令和实验结果。

`roadmap/`、v1-v6 notebook、旧配置和旧输出是历史证据，不得覆盖当前合同。

## 状态定义

- **未开始**：没有实现或只有文档。
- **实现中**：代码存在，但任一必需 smoke、产物或报告缺失。
- **完成**：本阶段全部硬门通过，证据路径和可复现命令已写入 `00_progress.md`。
- **失败**：已运行但任一硬门失败；保存诊断，不能写“基本完成”或“预计通过”。

文档完成不等于阶段完成，canary 通过也不等于正式训练或产品验收完成。

## 快速主线追踪

| 阶段 | 文档 | 实现 | 必需测试 | 必需结果 | 当前状态 |
| --- | --- | --- | --- | --- | --- |
| 固定公开数据与 curriculum | 已更新 | 未实现 | 未运行 | 无 | 未开始 |
| 单-adapter A2S runner | 已更新 | 未实现 | 未运行 | 无 | 未开始 |
| 一次正式 A2S 与 release | 已更新 | 未实现 | 未运行 | 无 | 未开始 |

当前快速主线总状态：0/3。

## 阶段 1：固定公开数据与 curriculum

### 设计输入

- 固定 dataset revision、选样 seed 和 `configs/data/public_robust_200k.yaml`。
- 200k train、10k validation、Bench 5k、LibriSpeech test-clean、AISHELL-1 test。
- 从 robust train 派生 30k base-error curriculum。

### 硬门 A：Metadata-only probe

正式音频下载前必须生成 probe report，并同时证明：

- 配置 revision 与实际 revision 一致。
- 实际为 54 个总 split：7 atomic + 47 compound，而不是 7 + 54。
- 必需字段、row count、source identity 候选和 en/zh 配额满足固定选样规则。
- shard 下载量、解包量和 `/content` staging 峰值未超过空间预算。
- 各数据源 license、用途限制和 release 影响已记录；许可不明项为 0。

任一项失败时不得下载完整音频、不得自动缩小规模或调整配额。

### 硬门 B：128-row 数据 smoke

- JSONL 每条包含可解析的 `audio`、gold `text`、language、scenario、source id 和 provenance。
- 相对 `data_root` 的音频存在、可解码且时长在 0.5-30 秒。
- 拒绝行写入 rejects，单条坏数据不终止批次。
- 固定 seed 重跑得到相同 row id 顺序和 manifest hash。
- 人工抽听覆盖 en/zh、atomic/compound、clean/robust。

### 硬门 C：Full data gate

- `train=200,000`：160k robust + 20k English clean + 20k Chinese clean。
- `validation=10,000`：8k robust + 1k English clean + 1k Chinese clean。
- robust train 的 7 atomic 各 16k；47 compound 合计 48k，固定余数分配可复现。
- Validation 保持 70% atomic / 30% compound，并满足固定 en/zh 规则。
- Bench 5k、LibriSpeech test-clean 和 AISHELL-1 test manifest 固定且未进入 train/validation。
- source/name/path/benchmark id/audio hash 的 train/validation/test 硬 overlap 全部为 0。
- stats、rejects、overlap report、license report、revision、seed 和 manifest SHA256 齐全。

### 硬门 D：30k curriculum

- 使用 pinned BF16 base、同一 normalization 和 evaluator 生成。
- English 使用 WER，Chinese 使用 CER，统一写入 `base_error_rate` 和 `base_metric`。
- 30k 样本全部来自 robust train 且 `base_error_rate < 0.70`。
- `<0.30`、`<0.50`、`<0.70` 三个累计视图数量、分层分布和 hash 齐全。
- 抽样复算 prediction、error rate 和阈值归属通过。
- 未对 200k 全量运行 difficulty scoring，未调用 teacher。

### 阶段完成证据

Metadata probe、128-row smoke、200k/10k/5k manifest、30k curriculum、stats、rejects、
overlap/license report、人工抽听记录和可复现命令全部存在，阶段才从“实现中”改为“完成”。

## 阶段 2：单-adapter A2S runner

### 设计输入

- `train/train_qwen3_asr_a2s.py`。
- `configs/train/qwen3_asr_public_200k_a2s.yaml`。
- pinned Qwen runtime module snapshot 与 343-target map。
- 128 条 en/zh、clean/degraded 平衡 fixture。

### 硬门 A：官方边界与 golden batch

- prompt、collator、audio mask、label mask、padding 和 Trainer 行为与 pinned Qwen 官方结构一致。
- batch 中 gold transcript 与模型监督 token 一致，prompt/audio token 不进入 loss。
- 历史 trainer 和 Mega-ASR wrapper/入口/target/merge 代码不在新 runner import 路径中。

### 硬门 B：Target map 与阶段切换

- 全量 target 精确为 343：audio attention 96、audio MLP 48、projection 3、decoder
  attention 112、decoder MLP 84。
- Phase I 可训练 target 精确为 27，Phase II 为 196，Phase III 为 343。
- `lm_head`、embedding、norm 和三个 Conv2d frontend 不可训练。
- target path、运行时类型、模型 revision 和 target-map hash 全部保存。
- target-switch smoke 逐阶段证明：只有允许组存在梯度、optimizer group 与学习率正确、冻结组
  参数在 optimizer step 后不变化。
- 三阶段使用同一个已注入 adapter，不创建中间 adapter 或执行 merge。

### 硬门 C：10+2 真 resume

1. 在 128-row fixture 上运行 10 个 optimizer step 并保存 checkpoint。
2. 退出进程，由新进程加载 checkpoint 并继续到 step 12。
3. 验证 global step、active phase、adapter、optimizer、scheduler、RNG、Trainer state、
   pipeline state、resolved config、target map/hash 和 manifest hash 全部连续。

缺失任一状态、从 step 0 重启或只恢复 adapter 均判定失败。

### 硬门 D：新进程推理

新进程只通过 base + release-compatible adapter + processor 加载，对 English/Chinese 的
clean/degraded 各 1 条生成非空结果，并把 raw prediction、normalized prediction 和错误写入
JSONL。单条失败必须记录而不是终止整批。

### 阶段完成证据

Golden batch、343 target 快照、三阶段 target-switch、10+2 resume、新进程四条推理、固定配置
和命令全部通过，阶段才标记完成。仅“能开始训练”不算完成。

## 阶段 3：一次正式 A2S 与 release

### 唯一执行序列

```text
Phase I: 30k curriculum x 2 epoch
  -> fixed 512 validation canary
Phase II: 200k x 1 epoch
  -> fixed 512 validation canary
Phase III: 200k x 1 epoch
  -> fixed 512 validation canary
  -> full 10k validation once
  -> Bench 5k + clean fixed tests once
  -> release reload
```

### 硬门 A：三个固定 512 canary

每阶段边界使用相同 manifest、decode 和 normalization，必须同时满足：

- loss 和 gradient 为有限值，输出有效率至少 95%。
- robust macro 相对 BF16 base 不恶化超过 15%。
- empty、repeat-like、too-long 和 hallucination-like 任一失败率增幅不超过 5 个百分点。
- prediction/metrics 与 phase checkpoint hash 已保存。

任一 canary 失败立即停止并保留诊断。Canary 只判断是否继续，不选择 checkpoint，也不替代
full validation。

### 硬门 B：一次最终全量评测

- 正常路径只对 Phase III 运行一次完整 10k validation。
- Phase III full validation 通过后，只对该候选运行一次 Bench 5k、LibriSpeech test-clean 和
  AISHELL-1 test。
- 只有 Phase III 失败且 Phase II canary 正常时，才允许懒评估 Phase II full validation。
- 不运行 50%/100% checkpoint sweep，不为每个阶段重复 Bench 或 clean test。
- 所有评测保存 raw/normalized reference/prediction、English WER、Chinese CER、逐 scenario
  和 Bench 32-cell 指标，以及失败样本标签。

### 硬门 C：产品指标

- English robust WER、Chinese robust CER、Bench 32-cell macro error 均相对 BF16 base
  改善至少 10%。
- Bench 至少 24/32 cell 改善，real macro 与 synthetic macro 都改善。
- LibriSpeech test-clean WER 增幅不超过 `max(0.3 个百分点, base WER 的 5%)`。
- AISHELL-1 test CER 增幅不超过 `max(0.5 个百分点, base CER 的 5%)`。
- empty、repeat-like、too-long、hallucination-like 增幅均不超过 1 个百分点。

### 硬门 D：Release 可恢复与可加载

- 单一 adapter、processor、训练配置、target map/hash、manifest hash、模型/数据/依赖 revision、
  训练状态、prediction、metrics 和 release manifest 齐全。
- 新进程按 release manifest 加载 base + adapter，并复现四条 smoke inference。
- `00_progress.md` 记录实际命令、artifact 路径、hash、指标、失败项和最终结果分支。

### 结果分支与阶段状态

| 结果 | 阶段状态与动作 |
| --- | --- |
| 三项 robust 均 >=10%，clean/failure 通过 | 正式阶段完成；发布并停止训练 |
| 三项 robust 均 >=5% 但未全部到 10% | 训练执行完成但产品验收失败；保存实验 adapter，停止并归因 |
| 任一 robust <5% | 产品验收失败；停止并检查数据、curriculum、target、prompt、labels、evaluator |
| robust 通过但 clean/failure 失败 | 产品验收失败；停止并按失败类型决定是否另立后续项目 |

所有分支互斥，且都不自动触发 teacher、direct SFT、重跑、扩量、router、RL 或 sweep。

## Mega-ASR 目标验收

“产品验收完成”和“接近 Mega-ASR”是两个不同结论。只有同时具备以下证据，才允许写
“接近 Mega-ASR”：

- Mega-ASR 发布模型在相同固定 manifest、decode、normalization 和 evaluator 下完成复测。
- 本项目 Bench macro error 不高于其 1.10 倍。
- 本项目 clean regression 与四类失败率门槛全部通过。
- 对比报告包含两方模型/revision、prediction、metrics 和 evaluator hash。

没有上述自有同口径评测，不得声称达到、超过或接近 Mega-ASR。200k A2S 也不得直接等同于
论文包含更大数据和 RL 的完整系统。

## 修改与验收流程

1. 先更新对应功能、数据、训练、测试和风险文档。
2. 只实现当前阶段的最小代码和 fixture。
3. 运行本文件对应硬门，失败立即停止该阶段。
4. 把命令、结果、hash 和产物路径写入 `00_progress.md`。
5. 对照硬门逐项更新状态；证据缺失时保持“实现中”或“失败”。

该流程的影响是：下载可与 runner fixture 开发并行，但正式训练必须等待阶段 1 和阶段 2 全部
通过；后续实验只能在首轮结果归因后单独立项，不能插入当前快速主线。
