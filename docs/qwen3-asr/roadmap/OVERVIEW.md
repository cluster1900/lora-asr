# 历史路线图总览

最后更新：2026-07-22

本文件仅解释项目早期如何从 baseline、合成数据和小规模 LoRA 逐步建立训练闭环。该路线已经
被公开数据 A2S 快速路径替代，不再要求按 00-08 顺序执行。

历史阶段包括：文档治理、独立骨架、baseline、数据 MVP、合成增强、LoRA v1-v6、错误分析、
router 设想和规模化设想。其中可继续复用的是独立 Qwen3-ASR 边界、JSONL 可复现约束以及
WER/CER 与失败样本质量门；历史 batch-size-1 trainer、4bit 对照、TTS bootstrap、target
sweep、checkpoint sweep 和 router 均不进入当前训练。

当前路线只有一条：

```text
metadata probe
  -> 200k train / 10k validation / 5k fixed test
  -> BF16 base score 30k curriculum
  -> Phase I 30k x 2
  -> Phase II 200k x 1
  -> Phase III 200k x 1
  -> final validation + Bench + clean test
  -> release or stop with one failure diagnosis
```

当前目标是用单个 adapter 尽快验证接近 Mega-ASR-Base A2S-SFT 量级的收益，不承诺复现
Mega-ASR 完整 2.4M 数据、RL 和 router 系统。详细参数和结果分支以
`../02_development_plan.md` 为准。
