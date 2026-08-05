# 风险与决策

最后更新：2026-07-22

## 背景与范围

首轮目标不是穷举实验，而是用最短闭环验证并交付一个可恢复、可加载、可公平评测的
`Qwen/Qwen3-ASR-1.7B` 鲁棒 ASR LoRA。当前唯一主线是
`30k acoustic curriculum -> 200k semantic -> 200k joint` 的单次 A2S 训练。

本文件只记录会改变这条主线的决策、风险、停止条件和回滚条件。Teacher、direct-SFT
预训练、router、RL、自建增强、全量 difficulty scoring、超参数 sweep 和中间规模模型均不在
首轮范围内。

## 当前决策

### D001: 基础模型与依赖固定

- 基础模型统一使用 `Qwen/Qwen3-ASR-1.7B`。
- 正式训练、base scoring 和评测统一使用 BF16；4bit 只保留历史复现价值。
- 模型、Qwen3-ASR 源码、`qwen-asr`、Transformers、PEFT 和数据集均固定 revision/version。
- resolved config、版本、target map/hash、manifest hash 随 checkpoint 和 release 保存。

原因：模型结构、processor 或数据 revision 漂移都会让 target 数量、样本构成和结果失去可比性。

### D002: 第一轮直接运行 A2S，不先跑 direct SFT

- Phase I：30k base-error curriculum，2 epoch。
- Phase II：200k public train，1 epoch。
- Phase III：200k public train，1 epoch。
- 不额外训练一个 200k direct-SFT adapter，也不训练 50k/100k 过渡模型。

原因：Mega-ASR 消融中 direct SFT 的收益明显低于 A2S-SFT。先做 direct SFT、失败后再做
A2S 会重复一次 200k 全量训练，不符合快速交付目标。首轮直接选择更接近目标方法的唯一路径。

### D003: 公开数据固定为 200k/10k/5k

- Train：160k robust + 20k English clean + 20k Chinese clean。
- Validation：8k robust + 1k English clean + 1k Chinese clean。
- Robust test：Voices-in-the-Wild-Bench 5k 全量。
- Clean test：LibriSpeech test-clean + AISHELL-1 test。
- Curriculum：仅从 robust train 派生 30k，不对 200k 全量做 difficulty scoring。

固定数据 revision 当前应有 54 个总 split，即 7 atomic + 47 compound。旧文档中的
`7 atomic + 54 compound` 不再使用。下载音频前必须先运行 metadata-only probe，验证 revision、split、
字段、行数、语言配额、source identity 候选、shard 大小和本地磁盘预算；任一硬条件不满足即
停止，不静默调整配额。

### D004: 数据许可与可追溯性是训练前置条件

- 保存每个数据源的 license、dataset revision、原始 row index、选样 seed 和 manifest SHA256。
- 保存样本来源和拒绝原因，不提交或重新分发受限制的音频。
- 正式训练前必须人工确认各数据源的训练、衍生权重和发布用途与其许可相容。
- 许可不明或条款冲突时，移除该数据源并重新生成 manifest/hash，不凭经验默认可用。

原因：技术上可下载不等于允许训练或发布衍生模型；许可记录缺失会让 release 无法审计。

### D005: Trainer 以 Qwen 官方结构为边界

复用 Qwen 官方 `finetuning/qwen3_asr_sft.py` 的 prompt、collator、label mask、Trainer、
scheduler、validation 和 resume 结构；本项目只新增 JSONL/YAML、PEFT、A2S scope 切换、
分组学习率、duration bucketing、generation canary 和 release 元数据。

不得扩建历史逐样本 trainer，不得复用 Mega-ASR 私有 wrapper、训练入口、target regex 或
adapter merge 逻辑。

### D006: 所有 target 一次注入同一 adapter

从 pinned Qwen revision 的运行时模块快照独立生成 343-target map：96 audio attention、
48 audio MLP、3 speech projection、112 decoder attention 和 84 decoder MLP。

- Phase I 仅启用 upper-4 audio + projection，共 27 个 target。
- Phase II 仅启用 decoder，共 196 个 target。
- Phase III 启用全部 343 个 target。

阶段间只切换 LoRA 参数的 `requires_grad` 和 optimizer parameter groups，不合并或重建
adapter。这样只发布一个标准 adapter，避免多 adapter loader 和中间 merge。target 数量、类型、
revision 或 hash 不匹配时立即失败。

### D007: 学习率采用 Appendix E.1 Table 22 口径

论文正文、附录和公开命令的学习率存在冲突，无法把任一数值描述为无歧义复现。首轮采用
信息更完整的 Appendix E.1 Table 22，并与公开 `1e-6` 命令对齐：

| 阶段 | audio/projection LR | decoder LR | warmup ratio |
| --- | ---: | ---: | ---: |
| Phase I | `1e-6` | frozen | 0.05 |
| Phase II | frozen | `1e-6` | 0.05 |
| Phase III | `5e-7` | `1e-6` | 0.03 |

这是本项目的显式工程假设，不伪装成论文已完整披露的复现细节。首轮不做 LR/rank/target sweep，
实际 resolved config 必须记录该假设。

### D008: 首轮不需要 teacher

训练集已有 gold transcript，A2S curriculum 只需计算 BF16 base prediction 与 gold 的
English WER/Chinese CER；teacher 不会增加首轮必需标签。

GPT-5.5 当前只接受 text/image 输入，不接受 audio，因此不能充当声学 ASR teacher。首轮不配置
`TEACHER_API_KEY`、`TEACHER_BASE_URL` 或任何替代 teacher，也不为其保留调用步骤。以后只有
在独立的无标注音频项目中，才评估可接收音频并输出时间对齐 transcript 的模型；这不属于本轮
失败后的自动分支。

### D009: 只保留阶段 canary 和一次最终全量评测

- Phase I、II、III 结束后分别运行同一固定 512 validation canary。
- 仅 Phase III 通过 canary 后运行一次完整 10k validation。
- 仅最终候选运行一次 Bench 5k、LibriSpeech test-clean 和 AISHELL-1 test。
- 只有 Phase III 失败且 Phase II canary 正常时，才懒评估 Phase II full validation。
- 不预先生成或比较 50%/100% checkpoint，不用 canary 挑选 checkpoint。

原因：canary 用于及时发现训练崩溃，完整评测用于发布判定；重复全量评测不会改善 adapter，
只延长闭环。

### D010: 结果分支互斥，不自动扩实验

| 结果 | 固定动作 |
| --- | --- |
| 三项 robust 均改善 >=10%，clean/failure 通过 | 发布，复测 Mega-ASR gap，停止训练 |
| 三项 robust 均改善 >=5% 但未全部到 10% | 保存实验 adapter，停止并做错误归因 |
| 任一 robust 改善 <5% | 停止，检查数据、curriculum、target、prompt、labels 和 evaluator |
| robust 通过但 clean/failure 失败 | 停止并按失败类型决定是否另立 clean retention 或 router 项目 |

任何分支都不会自动触发 teacher、direct SFT、数据扩量、第二次正式训练、router 或 RL。

### D011: Drive 只保存大 artifact

Google Drive 保存大 parquet/tar shard、manifest、状态、checkpoint、prediction 和结果；训练音频
在 `/content` 本地 SSD 物化。禁止从 Drive 逐条读取 200k 小文件。

### D012: Mega-ASR 只作为外部 baseline

Mega-ASR 发布模型可以在本地忽略的 `references/` 环境中运行，但其代码不进入本项目依赖。
只有在相同 manifest、decode、normalization 和 evaluator 下复测后，才允许比较效果。

200k A2S 的数据规模小于论文完整系统且不含 RL，不能预先承诺达到完整 Mega-ASR。只有本项目
Bench macro error 不高于同评测 Mega-ASR 的 1.10 倍，并同时通过 clean/failure 门槛，才允许写
“接近 Mega-ASR”。

## 主要风险与缓解

### R001: 200k A2S 仍可能达不到目标收益

论文结果来自不同规模与完整训练条件，本项目单 adapter 和 Phase I 等 step 分配是简化实现。

缓解：先以三项 robust 均改善 10% 作为产品门，不预先声称复现；未通过时按 D010 停止并定位
瓶颈，不用连续试验掩盖失败。

### R002: 数据集 card、revision 与实际 split 不一致

公开描述可能与 pinned revision 的实际 54 个总 split 或字段不一致，直接按文档下载可能造成配额
错误和大量无效 I/O。

缓解：metadata-only probe 是硬门；report 必须记录 7 atomic + 47 compound、字段、行数、配额、
shard 字节数和 revision。probe 失败不得启动音频下载或训练。

### R003: 数据下载、解包或 Drive I/O 比训练更慢

公开 robust 数据总量约 197.5 GB，选定子集仍可能超过临时磁盘或 Colab session 预算。

缓解：probe 先估算空间；按 shard 选择、断点续传、大文件持久化、本地 SSD staging；空间不足时
停止并报告，不边训练边远程读取小文件。

### R004: 数据许可阻断 release

数据许可、衍生模型条款或再分发限制可能随数据源不同而变化。

缓解：训练前生成 license report 并人工确认；release manifest 记录来源与 revision。无法确认的
数据必须移除，不能等训练后再补许可判断。

### R005: Source leakage 与重复退化样本

同一 clean utterance 可能有多个退化版本，音频 hash 不同但 source 相同，导致 validation/test
指标虚高。

缓解：先派生 `source_utterance_id` 再 group split；source/name/path/benchmark id/audio hash
硬 overlap 为 0。Transcript overlap 单独报告和抽查，不作为唯一泄漏判断。

### R006: 语言推断或 curriculum 指标错误

数据可能没有可靠 language 字段；混合文本或错误归一化会把中文 CER 和英文 WER 混用，改变
30k curriculum。

缓解：固定语言推断和 normalization，混合或不确定样本进入 rejects；manifest 保存
`base_prediction`、`base_error_rate` 和 `base_metric`，抽样复算各阈值桶。

### R007: 模型结构漂移导致 target map 失效

同名模块不一定同类型，revision 更新也可能改变 343 个 target 的数量或路径。

缓解：从运行时类型生成 target map；明确排除 `lm_head`、embedding、norm 和 Conv2d；target
count/type/revision/hash 任一不匹配即失败。

### R008: 阶段切换出现错误梯度或 optimizer 状态污染

只修改 `requires_grad` 而未同步 optimizer groups，可能让冻结组继续更新，或丢失应训练参数状态。

缓解：target-switch smoke 逐阶段检查可训练参数集合、实际梯度、optimizer groups 和学习率；
checkpoint 保存 pipeline phase，resume 后再次验证 active target hash。

### R009: Resume 看似成功但训练状态未恢复

只重载 adapter 会丢失 optimizer、scheduler、RNG、global step 和 curriculum phase。

缓解：正式训练前必须通过 10 optimizer step 保存、新进程 resume 到 12 的集成测试；比较
global step、LR、phase、optimizer/scheduler/RNG 和 Trainer state，缺一项即失败。

### R010: 评测静默美化错误

空 reference、双语混算、失败样本被跳过或 normalization 不一致都可能虚增改善率。

缓解：空 reference 硬失败；English WER 与 Chinese CER 分开；保存 raw/normalized
reference/prediction、逐样本 edits、scenario 和失败标签，并聚合完整 Bench 32 cells。

### R011: 鲁棒收益伴随 clean regression 或异常输出

decoder/joint LoRA 可能改善退化音频，却增加 clean WER/CER、空输出、重复、过长或幻觉输出。

缓解：训练固定 20% clean retention；每阶段 512 canary 检查输出有效率；最终 clean/failure 使用
硬门槛。失败时停止，不自动重跑或用 router 隐藏回归。

### R012: Colab 中断导致长任务丢失

缓解：manifest 与选样固定；prediction 增量写入；checkpoint 原子复制；adapter、optimizer、
scheduler、RNG、Trainer/pipeline state 和 resolved config 均可恢复。阶段 canary 不是恢复点替代品。

### R013: 在缺少自有评测时过度宣称

缓解：文档、README 和 release note 只能记录实际运行的 BF16 base、A2S 和外部 baseline 指标。
没有同 evaluator 的 Mega-ASR 结果时，不得写“达到”“超过”或“接近 Mega-ASR”。

## 测试与验收影响

- 数据实现必须先通过 metadata probe、license/revision report、128-row smoke 和 full data gate。
- Trainer 必须通过 golden batch、target-switch、10+2 真 resume 和新进程加载。
- 正式训练仅运行三个固定 512 canary、一次 Phase III 10k validation 和一次固定 test。
- `07_document_acceptance.md` 是状态更新的硬门；任何风险缓解证据缺失，阶段保持未完成。

## 当前参考

- Qwen3-ASR 官方 finetuning：https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning
- Voices-in-the-Wild-2M：https://huggingface.co/datasets/zhifeixie/Voices-in-the-Wild-2M
- Voices-in-the-Wild-Bench：https://huggingface.co/datasets/zhifeixie/Voices-in-the-Wild-Bench
- Mega-ASR 论文：https://arxiv.org/abs/2605.19833
- GPT-5.5 模型页：https://developers.openai.com/api/docs/models/gpt-5.5
