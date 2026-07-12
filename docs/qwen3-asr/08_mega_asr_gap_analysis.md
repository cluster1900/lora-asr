# Mega-ASR 差距分析

最后更新：2026-07-12

## 结论

当前项目已经具备历史 LoRA 工程闭环，但尚未得到相对 BF16 base 的正式净收益。一次
200k joint broad LoRA 是最快的有效验证，不等于能够复现 Mega-ASR 的完整 A2S+RL+router
效果。

## 公开方法证据

Mega-ASR 公开论文 Table 5 在 Voices/Noizeus 上的消融：

| 方案 | Voices | Noizeus | 相对 Qwen3-ASR |
| --- | ---: | ---: | --- |
| Qwen3-ASR | 8.94 | 9.45 | baseline |
| Direct SFT without A2S | 8.31 | 8.79 | 约 7% 改善 |
| Mega-ASR-Base A2S-SFT | 7.59 | 8.12 | 约 14%-15% 改善 |
| Mega-ASR full | 7.35 | 7.64 | 约 18%-19% 改善 |

Table 6 显示 rule-based reward 与 LLM judge 的 WER 基本持平，而单 step 时间约快 3.2
倍。因此首轮或后续 RL 都没有理由默认引入 GPT-5.5 judge。

## 当前项目与 Mega-ASR

| 维度 | Mega-ASR | 当前项目 |
| --- | --- | --- |
| 基础模型 | Qwen3-ASR-1.7B | Qwen3-ASR-1.7B |
| 公开训练数据 | 论文 2.4M；当前 Hub 645,925 | 正式 200k 尚未准备 |
| 场景 | 7 atomic + 54 compound | 历史只覆盖少量合成场景 |
| 正式 SFT | A2S 三阶段 | 未实现；计划先 direct joint SFT |
| RL | DG-WGPO | 不进入第一轮 |
| Router | 有 | 不进入第一轮 |
| 公平 BF16 base | 有 | 尚未生成新 benchmark 结果 |
| Release adapter | 有 | 当前没有新主线 adapter |

## 历史项目证据

历史 v3 在同口径 4bit base 下 overall 改善 1.82%，noise+reverb 改善 6.71%，clean
持平，dropout/far-field 没有改善。

但 4bit base 本身比历史非量化 baseline 差 13.72%，测试集只有 30 条独立 TTS source，
且大多数样本 edit 数不变。因此 v3 不能支撑“已接近 Mega-ASR”的判断。

## 最小必要组件

第一轮必须：

- 公开多场景、大量 source 的 200k 数据。
- BF16 base 与 BF16 base+LoRA 公平对照。
- Official Trainer 风格的 batch、validation、checkpoint 和 resume。
- Joint audio+decoder LoRA。
- Bench 5k、双语 clean test、失败率与 Mega-ASR 外部 baseline。

第一轮不需要：teacher、router、RL/LLM judge、自建增强、全量 difficulty scoring 或
target/LR/checkpoint sweep。

## 最短升级路径

1. Direct joint 200k SFT。
2. 若只有 5%-10% 改善，在同一 200k 上执行压缩 A2S：30k WER curriculum acoustic
   warm start -> decoder -> joint。
3. 只有 A2S 后仍有明显高 WER semantic failure 才评估 RL。
4. 只有 robust 好但 clean 冲突仍无法通过 retention 解决时才训练 router。
5. 全量 645,925 是后置 scale 选项，不是第一轮自动动作。

## “达到一样效果”的定义

必须运行 Mega-ASR 发布模型，使用相同 Bench manifest、normalization、WER/CER 和 32-cell
evaluator。本项目 macro error <= Mega-ASR macro error 的 1.10 倍，并同时通过 clean 与
失败率门槛，才允许写“接近 Mega-ASR 微调效果”。

## 参考

- Mega-ASR：https://github.com/xzf-thu/Mega-ASR
- 论文：https://arxiv.org/abs/2605.19833
- 数据：https://huggingface.co/datasets/zhifeixie/Voices-in-the-Wild-2M
- Benchmark：https://huggingface.co/datasets/zhifeixie/Voices-in-the-Wild-Bench
