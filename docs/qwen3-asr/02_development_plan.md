# 快速微调执行合同

最后更新：2026-07-12

## 目标

用最少的新代码和最少的训练分支，得到一版可加载、可评测、可继续迭代的
`Qwen/Qwen3-ASR-1.7B` 鲁棒 ASR LoRA。

本文件是当前唯一执行合同。其他文档说明背景、数据、Colab、测试和风险，但不重复定义
另一套步骤或超参数。

## 当前事实

- 历史 baseline、推理、WER/CER、错误分析和 LoRA v1-v5 已跑通。
- 历史最优 v3 只相对同口径 4bit base 在 noise+reverb 上改善 6.71%，不能证明相对
  BF16 base 的真实净收益。
- 历史正式 train 只有 120 条独立 TTS source；v6A 只是这些 source 的重复增强。
- 当前工作区没有可复载的 v3-v5 adapter 权重，历史实验只能作为方法和指标记录。
- 下面的新主线尚未实现，当前完成度为 0/3。

## 为什么不能承诺一次训练复现 Mega-ASR

Mega-ASR 的公开消融表明，直接 SFT、三阶段 A2S-SFT 和完整 RL 的收益不是同一量级：
直接 SFT 约带来 7% 相对改善，A2S-SFT 约 14%-15%，完整系统约 18%-19%。因此本项目
第一轮只验证“公开大规模数据 + joint broad LoRA”是否能得到明确净收益，不能预先声称
一次 200k SFT 足以达到 Mega-ASR 完整效果。

## 范围

第一轮只实现三项：

1. 一个公开数据准备入口。
2. 一个基于 Qwen 官方 finetuning 结构的 PEFT Trainer 入口。
3. 一次 200k joint broad LoRA 训练及 BF16 公平评测。

第一轮明确不实现：teacher、router、RL、自建增强工厂、全量 difficulty scoring、target
sweep、50k/100k 过渡模型和更多 notebook 分支。

## 步骤 1：固定公开数据

### 文件

- `scripts/prepare_public_robust_manifests.py`
- `configs/data/public_robust_200k.yaml`

### 固定规模

| split | 数量 | 组成 |
| --- | ---: | --- |
| train | 200,000 | 160k robust + 20k English clean + 20k Chinese clean |
| validation | 10,000 | 8k robust + 1k English clean + 1k Chinese clean |
| robust test | 5,000 | Voices-in-the-Wild-Bench 全量 |
| clean test | 官方固定集 | LibriSpeech test-clean + AISHELL-1 test |

Robust 数据来自 pinned revision 的 `zhifeixie/Voices-in-the-Wild-2M`。公开数据当前 card
为 645,925 条、约 197.5 GB，不把名称中的 2M 当作实际可下载条数。Bench 使用
`zhifeixie/Voices-in-the-Wild-Bench` 的 pinned revision。

160k robust 固定按语言与条件分层：7 个 atomic condition 各 16k，共 112k；compound
48k。每组 English/Chinese 尽量各半，配额不足必须失败并输出统计，不能静默改比例。

### 数据 I/O

- Google Drive 只保存 manifest、下载状态、较大的 parquet/tar shard、checkpoint 和结果。
- 训练前把所需 shard 或音频 staging 到 `/content` 本地 SSD，禁止从 Drive 逐条读取
  200k 小音频文件。
- 正式 manifest 的 `audio` 使用相对路径，由配置中的 `data_root` 解析；训练前所有解析后
  路径必须存在。
- 首轮只保留 0.5-30 秒音频，超限样本写入 rejects，以控制训练时长和显存波动。

### 数据硬门槛

- 统一 schema 见 `03_data_plan.md`，训练和评测都使用 `answer`、`language=en|zh` 和
  `source_dataset`。
- 同一 source utterance 的不同退化版本必须落在同一 split。
- train、validation、Bench 之间的 source id、name、source path、文件名、音频哈希和
  benchmark id 硬 overlap 必须为 0；归一化 transcript overlap 只报告并抽查。
- 语言、场景、时长、音频解码或 transcript 校验失败的行写入 rejects，不中断整批。
- 配置、seed、dataset revision、选中 row index、license 和 manifest SHA256 全部保存。

### 完成条件

生成 train/validation/test JSONL、stats、validation report 和 rejects；smoke 与 full
检查均通过，并完成每个语言/条件至少 10 条人工抽听。

## 步骤 2：官方 Trainer 薄适配

### 文件

- `train/train_qwen3_asr_lora_official.py`
- `configs/train/qwen3_asr_public_200k_broad_lora.yaml`
- `notebooks/12_fast_finetune_colab.ipynb`

### 实现边界

复用 Qwen 官方 `finetuning/qwen3_asr_sft.py` 的 prompt、collator、label mask、Trainer、
scheduler、validation 和 resume 结构，只增加：

- 项目 JSONL/schema 与 YAML 配置适配。
- PEFT LoRA 注入、分组校验和 adapter-only 保存/加载。
- duration bucketing、最大时长过滤、gradient checkpointing。
- validation generation、WER/CER 和失败率回调。
- 完整 resolved config、依赖版本、模型 revision 和 Trainer state 保存。

不得继续扩建历史逐样本 trainer；不得复制 Mega-ASR wrapper、训练入口或 target 规则。

### Broad LoRA

从本项目 pinned Qwen 模型快照独立生成 343 个线性 target：

| 分组 | 数量 |
| --- | ---: |
| audio attention | 96 |
| audio MLP | 48 |
| speech projection | 3 |
| decoder attention | 112 |
| decoder MLP | 84 |
| 合计 | 343 |

`r=8` 时预计 12,365,824 个 LoRA 参数。排除 `lm_head`、embedding、norm 和三个真实
Conv2d frontend。名称为 `conv_out` 的模块是 Linear，可以包含。若 pinned revision 下
各分组数量不匹配，训练直接失败并重新审计，不能只校验总数。

### 固定首轮配置

| 参数 | 值 |
| --- | --- |
| precision | BF16 |
| LoRA | r=8, alpha=16, dropout=0.05 |
| learning rate | 1e-6，全局统一 |
| epoch | 1 |
| effective batch | 64 |
| reference micro batch | A100 40GB 单卡 4，grad accumulation 16 |
| scheduler | linear，warmup ratio 0.03 |
| regularization | weight decay 0.01，max grad norm 1.0 |
| memory | FlashAttention 2 + gradient checkpointing |
| input | 0.5-30 秒，按 duration bucketing |

若 GPU 不同，只允许调整 micro batch 和 gradient accumulation，并保持 effective batch 64；
实际 resolved 值必须写入 checkpoint。首轮不做学习率、rank 或 target sweep。

### Smoke 与 golden batch

1. Golden batch 检查 prompt 文本、target 文本、audio mask、有效 label 数和 padding mask。
2. 128 条平衡样本训练 10 optimizer step，保存后 resume 2 optimizer step。
3. checkpoint 必须包含 adapter、optimizer、scheduler、RNG、Trainer state 和 resolved
   config。
4. 新进程加载 base+adapter，对中英 clean/degraded 各 1 条完成推理。

### 完成条件

Golden batch、10+2 resume、新进程加载和四条推理全部通过；BF16 base validation 评测
已启动或完成。

## 步骤 3：一次正式训练

### 单次执行序列

```text
正式 200k run
  -> optimizer step 100: 固定 512 条 validation canary
  -> 50%: 完整 10k validation
  -> 100%: 完整 10k validation
  -> validation 选出唯一候选
  -> Bench 5k + 双语 clean test 只评测一次
```

100-step canary 属于同一次正式 run，不是额外过渡模型。它只用于发现 label、prompt、
学习率、输出崩溃和数据吞吐问题；通过后继续训练，canary checkpoint 不作为发布候选。

Canary 通过标准：loss 全部有限；输出有效率至少 95%；empty、repeat-like、too-long 任一
指标相对 BF16 base 增幅不超过 5 个百分点；robust macro error 不得相对 base 恶化超过
15%。失败立即停止，不继续消耗剩余训练时间。

### 候选选择

50% 和 100% 只在 10k validation 上比较：

- English robust WER。
- Chinese robust CER。
- robust condition macro error。
- English/Chinese clean validation regression。
- empty、repeat-like、too-long、hallucination-like。

选择 robust macro 更低且 clean/failure 门槛通过的唯一 checkpoint。Bench 5k、
LibriSpeech test-clean 和 AISHELL-1 test 只对该 checkpoint 跑一次。

### 第一轮产品验收

- English robust WER、Chinese robust CER、Bench 32-cell macro error 均相对 BF16 base
  改善至少 10%。
- Bench 至少 24/32 个 `language x real/synthetic x condition` cell 改善，real 与
  synthetic macro 都改善。
- LibriSpeech test-clean WER 增幅不超过 `max(0.3 个百分点, base WER 的 5%)`。
- AISHELL-1 test CER 增幅不超过 `max(0.5 个百分点, base CER 的 5%)`。
- empty、repeat-like、too-long、hallucination-like 增幅均不超过 1 个百分点。
- adapter、processor、release manifest、批量推理命令和新进程加载验证齐全。

### Mega-ASR 目标口径

“达到 Mega-ASR”必须另行运行其发布模型作为外部 baseline，并用本项目同一 normalization
与 Bench evaluator 计算。只有当本项目 Bench macro error 不高于 Mega-ASR 的 1.10 倍，
且 clean 门槛通过时，才允许标记为接近其微调效果。论文表格只能作为参考，不能代替
本项目复测。

## 唯一结果分支

| 第一轮结果 | 下一动作 |
| --- | --- |
| 三项 robust 指标均改善 >=10%，clean 通过 | 立即交付 200k adapter；先比较 Mega-ASR gap，不自动扩全量 |
| 主要指标改善 5%-10%，clean 通过 | 在同一 200k 上做一次压缩 A2S；不做 target/LR sweep |
| 任一语言或 macro 改善 <5% | 停止训练，检查数据、prompt、labels、Trainer 和评测 |
| robust 明显改善但 clean 失败 | clean 比例只允许提高一次并重跑；仍失败才考虑 router |

压缩 A2S 不是第一轮依赖。只有触发时才实现：对固定 30k 样本做 base WER 分桶，然后在
一个编排任务内依次训练 upper audio+projection、decoder、joint 三个 scope。RL、teacher
和全量 645,925 继续后置。

## GPT-5.5 Teacher 边界

当前数据已有 gold transcript，第一轮不调用 teacher。未来只有未标注真实音频或标签冲突
才使用：

```text
TEACHER_API_KEY
TEACHER_BASE_URL
TEACHER_MODEL=gpt-5.5
```

官方 OpenAI Python SDK 支持 `gpt-5.5`、Responses API、显式 `api_key` 和 `base_url`/
`OPENAI_BASE_URL`。但任意兼容 base URL 是否支持音频输入必须运行 capability probe，不能
默认假设。Teacher 输出只能作为离线审校或伪标签候选，不能覆盖 gold transcript。

## 交付物

- 固定数据配置、manifest、stats、rejects 和 hash。
- BF16 base、LoRA 和 Mega-ASR 外部 baseline 的 prediction/metrics。
- 可加载 adapter、processor、Trainer state 和 release manifest。
- 一条 Colab 训练入口和一条 JSONL 批量推理命令。
- `00_progress.md` 中记录实际结果、未解决风险和是否触发下一分支。
