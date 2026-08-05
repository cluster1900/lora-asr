# 快速微调策略

最后更新：2026-07-22

## 定位

本文件解释为什么首轮采用单-adapter A2S，以及训练过程中如何停止。所有文件、固定参数、
命令和完成条件以 `02_development_plan.md` 为唯一执行合同。

目标是在一次可恢复的正式训练中，尽量接近 Mega-ASR-Base 的 A2S-SFT 收益等级，并交付
可由 `Qwen/Qwen3-ASR-1.7B` + 标准 PEFT 加载的单一 adapter。本轮不承诺复现使用更大
数据和 RL 的 Mega-ASR 完整系统。

## 范围

首轮只有一条训练路径：

```text
30k base-error curriculum x 2 epoch
  -> 200k decoder x 1 epoch
  -> 200k joint x 1 epoch
  -> full validation/test once
```

不先跑 direct SFT，也不并行训练多个 rank、target、学习率或 checkpoint。Teacher、Router、
RL、自建增强和 645,925 全量扩展均不进入实现、依赖或验收。

## 为什么直接进入 A2S

Mega-ASR 公开消融中，direct SFT 相对 base 约改善 7%，A2S-SFT 约改善 14%-15%。如果先做
200k direct，再以同一数据执行 A2S，首轮总样本 exposure 约为 660k；直接 A2S 只需约
460k，同时更贴近目标方法。因此 direct-first 不是低风险 canary，而是一次已知可能不足的
冗余全量训练。

代码正确性由 128-row fixture 的 10+2 optimizer-step resume smoke 保证，训练稳定性由每个
阶段后的固定 512 validation canary 保证，不再用完整 direct run 充当测试。

## 数据与课程

训练集固定为 160k robust + 20k English clean + 20k Chinese clean；validation 固定为 8k
robust + 1k English clean + 1k Chinese clean。20% clean retention 是 decoder 和 joint
阶段的训练约束，用于降低 clean regression，不是额外实验分支。

Curriculum 只从 robust train 候选中派生。使用 pinned BF16 base 和正式 evaluator 生成：

```json
{"base_prediction":"...","base_error_rate":0.42,"base_metric":"wer"}
```

英文以 WER、中文以 CER 计算，但统一写入 `base_error_rate`。按固定 seed 的
language/scenario 分层候选顺序补足 30k 个 `<0.70` 样本，并生成累计 `<0.30`、`<0.50`、
`<0.70` 三个视图。只对候选池做 scoring，不扫描完整 200k，更不扫描 Hub 全量数据。

论文没有披露两个 epoch 在三个累计阈值间的精确 step 分配。首轮采用等 optimizer-step
三段，总 sample exposure 保持 30k x 2，并在 resolved config 中明确标记为本项目假设，
不把它描述成论文原始设置。

## 单 Adapter 设计

从 pinned Qwen revision 的运行时模块快照独立生成 target map，不复制 Mega-ASR 的 target
regex。所有 343 个 target 只注入一次：audio attention 96、audio MLP 48、speech
projection 3、decoder attention 112、decoder MLP 84。

| 阶段 | 活跃 target | 数量 | 目的 |
| --- | --- | ---: | --- |
| Phase I | upper-4 audio attention + upper-4 audio MLP + projection | 27 | acoustic curriculum |
| Phase II | decoder attention + decoder MLP | 196 | semantic adaptation |
| Phase III | 全部 target | 343 | joint alignment |

阶段切换只修改 LoRA 参数的 `requires_grad` 与 optimizer parameter groups，不 merge、复制或
串联 adapter。`lm_head`、embedding、norm 和三个 Conv2d frontend 始终排除；`conv_out`
仅在运行时类型为 Linear 时计入 projection。target 数量、模块类型、model revision 或
target-map hash 任一不符都立即终止训练。

## 固定训练配置

| 参数 | Phase I | Phase II | Phase III |
| --- | --- | --- | --- |
| data | 30k cumulative error views | full 200k | full 200k |
| epoch | 2 | 1 | 1 |
| active target | upper-4 audio + projection | decoder | all 343 |
| audio/projection LR | `1e-6` | frozen | `5e-7` |
| decoder LR | frozen | `1e-6` | `1e-6` |
| warmup ratio | 0.05 | 0.05 | 0.03 |

共同配置固定为 BF16、LoRA r=8、alpha=16、dropout=0.05、effective batch 128、weight
decay 0.01、max grad norm 1.0、linear scheduler、0.5-30 秒、duration bucketing、
FlashAttention 2 和 gradient checkpointing。

单卡 A100 40GB 参考 micro batch 4、gradient accumulation 32。更换 GPU 时只允许调整
micro batch 与 accumulation 的组合，effective batch 仍保持 128，避免把硬件适配变成新的
超参数实验。

论文正文、附录和公开脚本的学习率存在冲突。首轮采用信息更完整的 Appendix E.1 Table 22
和公开 `1e-6` 命令口径；该来源与差异必须写入 resolved config。没有产品结果前不发起
学习率 sweep。

## 官方实现边界

Runner 复用 Qwen 官方 `finetuning/qwen3_asr_sft.py` 的 prompt、collator、label mask、
Trainer、scheduler、validation 和 resume 结构，只加入 JSONL/YAML、PEFT、阶段 scope
切换、分组学习率、duration bucketing、generation canary 与 release metadata。

历史逐样本 trainer 不扩建，新 runner 不 import Mega-ASR 上游工程，也不引入其 wrapper、
训练入口、target 规则或 adapter merge 逻辑。最终产物必须能由 pinned Qwen base、processor
和单一 PEFT adapter 在新进程中加载。

## Smoke、Canary 与完整评测

正式训练前必须通过：

1. Golden batch：prompt、audio mask、label mask、padding 和目标文本正确。
2. Target switch：每阶段只有允许的 LoRA 参数收到梯度。
3. 128-row fixture：训练 10 optimizer step，保存后由新进程 resume 到 12。
4. Release reload：en/zh clean/degraded 各一条推理成功。

正式训练每个阶段后只评测固定 512 条 validation canary，检查 loss、梯度、有效输出率、
empty、repeat、too-long 与 robust macro。Phase III 完成后只运行一次 full 10k validation、
Bench 5k、LibriSpeech test-clean 和 AISHELL-1 test。

Phase III 失败但 Phase II canary 正常时，才懒评估 Phase II full validation。首轮不预先运行
50%/100% checkpoint sweep，也不用 canary 挑 checkpoint。

## 停止规则

| 结果 | 动作 |
| --- | --- |
| 三项 robust 均 >=10%，clean/failure 通过 | 发布 adapter，测 Mega-ASR 同 evaluator gap，停止训练 |
| 三项 robust 均 >=5% 但未全部到 10% | 保存实验 adapter 并停止，先做错误归因 |
| 任一 robust <5% | 停止，检查数据、base-error 桶、target、prompt、labels 和 evaluator |
| robust 通过但 clean/failure 失败 | 停止并按失败类型归因，不自动重跑 |

三项 robust 指 English robust WER、Chinese robust CER 与 Bench 32-cell macro error 的
相对改善。产品门还要求至少 24/32 cell 改善、real 与 synthetic macro 均改善，并通过
clean regression 和各类生成失败率上限。

本轮没有自动“下一阶段”。只有产品门已经通过、剩余差距明确集中于高错误率空输出、幻觉
或漏句时，才另立项评估 RL；只有 robust 通过但 clean 冲突无法通过 retention 解决时，才
另立项评估 Router。增加数据规模、学习率/rank/target sweep 同样必须基于错误归因重新立项。

## 验收与影响

一次首轮训练只有同时具备固定 manifest/config/seed、完整 checkpoint/resume 状态、BF16
base 对比、原始与归一化预测、按 scenario 聚合的 WER/CER、失败样本记录和新进程加载命令，
才算完成。

三项 robust 均改善至少 10% 且 clean/failure 门通过，表示达到本项目 200k A2S 发布标准。
是否“接近 Mega-ASR”仍需运行其发布模型，并在同一 manifest、decode、normalization 和
evaluator 下验证 macro error 不高于其 1.10 倍；在此之前不得作等效或超越声明。

该策略使第一轮只产生一个正式训练链、一个 release adapter 和一套最终完整评测。历史
v1-v6 配置与输出继续保留为证据，但不参与新 runner 的模型选择或默认命令。
