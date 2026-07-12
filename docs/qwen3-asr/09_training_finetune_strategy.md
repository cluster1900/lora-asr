# 快速微调策略

最后更新：2026-07-12

## 定位

本文件只解释训练决策，不再维护第二套执行步骤。所有文件、命令、固定参数和完成条件以
`02_development_plan.md` 为唯一合同。

## 首轮假设

第一轮验证一个问题：在 200k 双语公开数据上，Qwen3-ASR-1.7B 的 BF16 343-target
joint broad LoRA，能否相对同环境 BF16 base 获得至少 10% 鲁棒错误率下降，同时保持
clean 能力。

选择 joint broad LoRA 的原因：

- 历史 99/123-target 小样本 audio-side LoRA 只得到弱收益，继续 target sweep 信息价值低。
- 严重退化同时涉及 acoustic grounding 与 semantic recovery，只训练 audio tower 不足以
  覆盖全部失败模式。
- 343 target 来自本项目自己的 Qwen 模块快照，不复用 Mega-ASR target 规则。
- `r=8` 只有约 12.37M 可训练参数，仍属于轻量 PEFT。

首轮全局学习率固定为 `1e-6`。这是针对全 audio+decoder scope 的保守值；`5e-6` 在没有
target/LR sweep 的前提下风险偏高，容易把一次正式 run 变成昂贵排错。

## 为什么保留 20% clean

Train 固定 160k robust + 40k clean，validation 固定 8k robust + 2k clean。Broad LoRA
直接更新 decoder，而第一轮又不做 router，因此 clean retention 是训练约束，不是可选
装饰。10% clean 不足以支撑 clean checkpoint selection，20% 可降低只允许一次正式重跑
时的失败概率。

## 为什么需要 100-step canary

10+2 smoke 只能证明代码、保存和 resume 可运行，不能证明 prompt/label 正确、学习率安全
或生成没有崩溃。100-step canary 放在同一次正式 run 内，只评测固定 512 条 validation；
失败立即停止，通过后继续到 50% 和 100%。这不是新增实验分支，而是节省训练时间的硬
早停门。

## Mega-ASR 方法边界

公开消融显示：直接 SFT 的改善约 7%，A2S-SFT 约 14%-15%，完整 RL 约 18%-19%。因此：

- 一次 200k joint SFT 是最快的有效信号。
- 5%-10% 改善时，不应直接盲目扩到 645,925，也不应继续 target sweep。
- 唯一有方法证据的下一步是压缩 A2S：30k WER 分桶，acoustic、decoder、joint 三阶段
  在一个编排任务中完成。
- RL 只有 A2S 后仍存在明显高 WER 语义恢复问题时才讨论。

## Teacher 决策

第一轮没有 teacher：公开训练数据已有 gold transcript，GPT-5.5 不能增加 speaker、噪声、
RIR 或场景多样性。对现有 gold label 调 teacher 只增加成本、延迟和标签漂移。

未来引入未标注真实音频时，teacher 通过 API key、base URL 和 model 环境变量接入，输出
必须缓存并保存 prompt version。若实际 endpoint 未通过音频 capability probe，只能做文本
冲突仲裁，不能充当音频转写 teacher。

## Router 决策

第一轮不做 router。只有 robust 指标明显改善但 clean 回退仍超标时才恢复。未来 router
目标不是简单预测 clean/degraded，而是预测该样本使用 LoRA 是否优于 base，并仍需报告
clean/degraded accuracy、precision、recall 和阈值来源。

## 停止规则

- 首轮 <5%：先排查实现与数据，不扩大规模。
- 首轮 5%-10%：只允许压缩 A2S，不做超参数树。
- 首轮 >=10% 且 clean 通过：先交付 adapter，再测 Mega-ASR gap。
- clean 或幻觉门槛失败：提高 clean retention 后最多重跑一次。
- 没有同一 evaluator 下的外部 baseline，不声称达到 Mega-ASR。
