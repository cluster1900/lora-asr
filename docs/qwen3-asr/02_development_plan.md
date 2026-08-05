# 快速 A2S 微调执行合同

最后更新：2026-07-22

## 目标

用一条可恢复、无实验树的训练链路，得到可加载、可评测、可发布的
`Qwen/Qwen3-ASR-1.7B` 鲁棒 ASR LoRA，并尽量用最少训练成本接近 Mega-ASR-Base 的
A2S-SFT 收益。

本文件是当前唯一执行合同。其他文档解释数据、Colab、测试和风险，不得定义第二套步骤。

## 当前事实与设计修正

- 历史 baseline、推理、WER/CER、错误分析和 LoRA v1-v5 已跑通，但只证明旧通路可运行。
- 新主线的数据 CLI、A2S runner、统一推理和 32-cell evaluator 均未实现，当前为 0/3。
- Mega-ASR Table 5 中 direct SFT 约改善 7%，A2S-SFT 约改善 14%-15%，完整系统约
  18%-19%。因此“先跑 200k direct SFT，失败后再 A2S”会重复一次全量训练，现已删除。
- GPT-5.5 不支持音频输入，且训练数据已有 gold transcript；teacher 不进入设计或依赖。
- 论文正文、附录和公开脚本的学习率存在冲突。首轮以更完整的 Appendix E.1 Table 22
  及公开 `1e-6` 命令为依据，并把假设写入 resolved config。

## 范围

第一轮只交付三项：

1. 一份固定公开数据池、30k base-error curriculum 和固定 test。
2. 一个基于 Qwen 官方 Trainer 结构的单-adapter A2S runner。
3. 一次三阶段 A2S 训练、BF16 公平评测和 release adapter。

第一轮不实现 direct SFT 对照训练、teacher、router、RL、自建增强、全量 645,925 样本
difficulty scoring、target/LR/rank sweep 或 50k/100k 过渡模型。

## 步骤 1：固定公开数据

### 文件

- `scripts/prepare_public_robust_manifests.py`
- `configs/data/public_robust_200k.yaml`
- `inference/qwen3_asr_infer.py`
- `evaluation/eval_wer.py`

### 1A. Metadata-only probe

下载音频前先读取 dataset revision、split、字段、行数和 parquet shard metadata，验证：

- Hub revision 固定为配置值，实际仍有 54 个 split：7 atomic + 47 compound。
- `answer`、`name`、`index`、`file_name`、`subset` 可用，`name` 可作为 source identity
  的第一候选。
- 每个 split 的 en/zh 推断后配额足够，混合或语言不明样本单独统计。
- 预计下载、解包和 `/content` staging 空间不超过运行环境预算。

Probe 只产 report，不训练模型；失败时在下载几十 GB 前终止。

### 1B. 固定规模

| split | 数量 | 组成 |
| --- | ---: | --- |
| train | 200,000 | 160k robust + 20k English clean + 20k Chinese clean |
| validation | 10,000 | 8k robust + 1k English clean + 1k Chinese clean |
| curriculum | 30,000 | 从 robust train 派生，base error <70% |
| robust test | 5,000 | Voices-in-the-Wild-Bench 全量 |
| clean test | 官方固定集 | LibriSpeech test-clean + AISHELL-1 test |

Robust train 中 7 个 atomic split 各 16k，共 112k。47 个 compound split 合计 48k：
按 split 名排序，每个先取 `floor(48000/47)=1021`，余下 13 条依次给前 13 个 split。
Validation 沿用 70% atomic / 30% compound 比例。每个 split 内 en/zh 尽量各半；不足时
probe 硬失败，不静默改比例。

### 1C. Curriculum

从 robust train 以固定 seed 按 language/scenario 分层产生候选，使用统一 BF16 base 推理和
同一 evaluator 写入：

```json
{"base_prediction":"...","base_error_rate":0.42,"base_metric":"wer"}
```

English 使用 WER，Chinese 使用 CER；统一字段名为 `base_error_rate`。按候选顺序补足
30k 个 `<0.70` 样本，并生成三个累计视图 `<0.30`、`<0.50`、`<0.70`。不对 200k 全量
做 difficulty scoring。

### 1D. I/O 与质量门

- Drive 只保存大 parquet/tar shard、manifest、状态、checkpoint 和结果。
- 训练所需音频在 `/content` 本地 SSD 物化；禁止从 Drive 逐条读取 200k 小文件。
- Manifest 的 `audio` 使用相对 `data_root` 路径，正式训练前必须存在且可解码。
- 只保留 0.5-30 秒音频；失败行写 rejects，不中断整批。
- 同一 source utterance 的退化版本必须同 split；train/validation/test 的 source、name、
  path、benchmark id 和 audio hash 硬 overlap 为 0。
- 保存 seed、row index、dataset/model revision、license、config 和 manifest SHA256。

### 完成条件

Metadata probe、128-row smoke、200k/10k/5k full manifest、30k curriculum、stats、rejects、
validation report 和人工抽听全部通过。Trainer 可在 full 下载期间用 128-row fixture 并行开发，
但正式训练必须等待 full data gate。

## 步骤 2：单-adapter A2S Runner

### 文件

- `train/train_qwen3_asr_a2s.py`
- `configs/train/qwen3_asr_public_200k_a2s.yaml`
- `requirements-colab.txt`
- `notebooks/12_fast_finetune_colab.ipynb`

### 官方边界

复用 Qwen 官方 `finetuning/qwen3_asr_sft.py` 的 prompt、collator、label mask、Trainer、
scheduler、validation 和 resume 结构，只新增本项目 JSONL/YAML、PEFT、A2S scope 切换、
分组学习率、duration bucketing、generation canary 与 release 元数据。不得扩建历史逐样本
trainer，不得复制 Mega-ASR wrapper、训练入口、target regex 或 adapter merge 逻辑。

### Target 合同

从 pinned Qwen revision 的运行时模块快照独立生成 target map：

| 分组 | 数量 |
| --- | ---: |
| audio attention | 96 |
| audio MLP | 48 |
| speech projection | 3 |
| decoder attention | 112 |
| decoder MLP | 84 |
| 全部 | 343 |

Phase I 只启用 upper-4 audio attention 16、upper-4 audio MLP 8 和 projection 3，共 27；
Phase II 只启用 decoder 196；Phase III 启用全部 343。排除 `lm_head`、embedding、norm 和
三个 Conv2d frontend；`conv_out` 按运行时类型为 Linear 时纳入 projection。

所有 target 一次注入同一 adapter，阶段间只切换 `requires_grad` 和 optimizer parameter
groups。这样最终只发布一个 adapter，不需要自定义多 adapter loader。若分组数量、类型、
revision 或 target-map hash 不匹配，训练立即失败。

### 固定 A2S 配置

| 参数 | Phase I | Phase II | Phase III |
| --- | --- | --- | --- |
| role | acoustic curriculum | semantic adaptation | joint alignment |
| data | 30k cumulative error views | full 200k | full 200k |
| epoch | 2 | 1 | 1 |
| active target | upper-4 audio + projection | decoder | all 343 |
| audio/projection LR | `1e-6` | frozen | `5e-7` |
| decoder LR | frozen | `1e-6` | `1e-6` |
| warmup ratio | 0.05 | 0.05 | 0.03 |

共同配置：BF16、r=8、alpha=16、dropout=0.05、effective batch 128、weight decay 0.01、
max grad norm 1.0、linear scheduler、0.5-30 秒、duration bucketing、FlashAttention 2 和
gradient checkpointing。单卡 A100 40GB 参考 micro batch 4、grad accumulation 32；其他
GPU 只调整 micro batch/accumulation，保持 effective batch 128。

论文未披露 Phase I 两个 epoch 在三个累计 WER 门槛间的精确 step 分配。实现不得伪造
论文结论；首轮采用配置显式记录的等 optimizer-step 三段，并在 resolved config 标注为本项目
假设。三段总 sample exposure 仍等于 30k x 2。

### Smoke 与恢复

1. Golden batch 验证 prompt、audio mask、label mask、target 文本和 padding。
2. Target-switch smoke 验证每阶段只有允许的 LoRA 参数收到梯度。
3. 128 条平衡 fixture 训练 10 optimizer step，保存后由新进程 resume 到 12。
4. Checkpoint 包含 adapter、optimizer、scheduler、RNG、Trainer state、pipeline state、
   resolved config、target map/hash 和 manifest hash。
5. 新进程 base+adapter 对 en/zh clean/degraded 各 1 条推理成功。

### 完成条件

Golden batch、target switch、10+2 resume、新进程加载和四条推理全部通过；BF16 base
validation 与 curriculum scoring 已完成。

## 步骤 3：一次正式 A2S

### 单次执行序列

```text
Phase I: 30k curriculum x 2 epoch
  -> fixed 512 validation canary
Phase II: 200k decoder x 1 epoch
  -> fixed 512 validation canary
Phase III: 200k joint x 1 epoch
  -> fixed 512 validation canary
  -> full 10k validation once
  -> Bench 5k + clean tests once
  -> release reload
```

阶段 canary 检查 loss、梯度、输出有效率、empty/repeat/too-long 和 robust macro；相对 BF16
base 的 robust macro 不得恶化超过 15%，输出有效率至少 95%，任一失败率增幅不得超过
5 个百分点。失败立即停止并保存诊断。阶段 canary 不用于挑 checkpoint。

正式只对 Phase III 运行一次完整 10k validation。若 Phase III 不通过而 Phase II canary
正常，才懒评估 Phase II full validation；不预先跑 50%/100% checkpoint sweep。

### 产品验收

- English robust WER、Chinese robust CER、Bench 32-cell macro error 均相对 BF16 base
  改善至少 10%。
- Bench 至少 24/32 cell 改善，real 与 synthetic macro 都改善。
- LibriSpeech test-clean WER 增幅不超过 `max(0.3 个百分点, base WER 的 5%)`。
- AISHELL-1 test CER 增幅不超过 `max(0.5 个百分点, base CER 的 5%)`。
- empty、repeat-like、too-long、hallucination-like 增幅均不超过 1 个百分点。
- Adapter、processor、release manifest、批量推理命令和新进程加载齐全。

### 结果分支

| 结果 | 动作 |
| --- | --- |
| 三项 robust 均 >=10%，clean/failure 通过 | 发布 adapter，测 Mega-ASR 同 evaluator gap，停止训练 |
| 三项 robust 均 >=10%，但 clean/failure 未通过 | 停止并按失败类型记录 clean retention 需求，不自动重跑 |
| 三项 robust 均 >=5% 但未全部到 10% | 保存实验 adapter并停止；先做错误归因，不自动加训练 |
| 任一 robust <5% | 停止，检查数据、base-error 桶、target、prompt、labels 和 evaluator |

RL 只有在产品门已通过、且剩余差距明确集中于高错误率空输出/幻觉/漏句时才单独立项。
Router 只有在 robust 通过但 clean 冲突无法由 retention 解决时才单独立项。两者都不是本轮
验收依赖。

## Mega-ASR 口径

200k A2S 比论文数据规模小且不含 RL，不能预先承诺复现完整 Mega-ASR。只有运行其发布
模型，并使用相同 manifest、decode、normalization 和 evaluator 后，才比较：本项目 Bench
macro error 不高于 Mega-ASR 的 1.10 倍且 clean 门槛通过，才允许写“接近 Mega-ASR”。

## 交付物与影响

- 固定数据配置、manifest、curriculum、stats、rejects 和 hash。
- BF16 base、A2S LoRA 与 Mega-ASR 外部 baseline prediction/metrics。
- 单一可加载 adapter、processor、Trainer/pipeline state 和 release manifest。
- 一条 Colab A2S 入口和一条可恢复 JSONL 批量推理命令。
- `00_progress.md` 记录实际结果、失败类型和是否需要后续独立立项。

旧 v1-v6 trainer/config/notebook/output 全部保留为历史证据，但不进入新 runner 的 import、
配置或模型选择路径。
