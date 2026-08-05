# Qwen3-ASR Robust ASR

独立的 Qwen3-ASR-1.7B 鲁棒 ASR 微调项目。Mega-ASR 只作为方法与外部 baseline，
不作为主实现代码底座。

## 当前状态

历史 baseline、LoRA v1-v5、4bit base recheck、WER/CER 和错误分析已经跑通。历史最优 v3
只在 4bit 对照下对 noise+reverb 改善 6.71%，没有证明相对 BF16 base 的正式净收益。

新的快速主线已完成最小脚本实现，但正式运行仍为 0/3：

1. 已实现 metadata probe、固定配额/泄漏校验和 curriculum 构建；Hub 音频 staging 与 full
   manifest 尚未运行。
2. 已实现官方 Trainer 薄适配、单 adapter target 切换、统一推理和双语 evaluator；10+2
   GPU smoke 尚未运行。
3. 正式 A2S、阶段 canary、最终 validation/test 和 release adapter 尚未运行。

完整执行合同见 [docs/qwen3-asr/02_development_plan.md](docs/qwen3-asr/02_development_plan.md)。

## 固定方案

- Base model：`Qwen/Qwen3-ASR-1.7B`。
- Train：160k robust + 20k English clean + 20k Chinese clean。
- Validation：8k robust + 1k English clean + 1k Chinese clean。
- LoRA：单 adapter 预注入 343 个 Linear target，r=8，alpha=16，dropout=0.05，BF16。
- A2S：30k acoustic curriculum 2 epoch -> 200k decoder 1 epoch -> 200k joint 1 epoch。
- 学习率：前两阶段 `1e-6`；joint audio/projection `5e-7`、decoder `1e-6`。
- 正式 run：effective batch 128；每阶段边界只跑 512 canary，最终只跑一次完整 validation。
- Test：Voices-in-the-Wild-Bench 5k、LibriSpeech test-clean、AISHELL-1 test。
- 第一轮不做 direct SFT 前置实验、teacher、router、RL、自建增强、全量 difficulty scoring
  或 target/LR sweep。

## 仓库结构

```text
configs/      历史配置与固定 A2S 数据/训练配置
data/         JSONL 与历史 smoke/MVP 数据
docs/         当前方案、测试、进度和风险
evaluation/   双语 WER/CER、32-cell 与错误分析
inference/    统一可恢复入口；旧 base/LoRA 入口仅供历史复现
notebooks/    Colab notebook；00-11 为历史主线
outputs/      历史 prediction、metrics 和分析
scripts/      公开 manifest/curriculum 与历史辅助脚本
train/        单-adapter A2S runner、模块探测和历史 trainer
router/       历史占位，不是当前交付物
references/   本地忽略的外部参考
```

## 当前可运行的历史 smoke

```bash
python3 scripts/create_smoke_audio.py --force
python3 evaluation/eval_wer.py --help
python3 evaluation/analyze_errors.py --help
```

新主线的数据 CLI、A2S runner、统一推理和 32-cell evaluator 已实现并有本地合同测试；
Notebook 12、Hub candidate staging、真实 GPU smoke 和正式训练尚未完成。不要把历史 notebook
继续扩展为正式 200k 训练入口。

## 文档

- [文档首页](docs/qwen3-asr/README.md)
- [当前进度](docs/qwen3-asr/00_progress.md)
- [架构方案](docs/qwen3-asr/01_architecture.md)
- [唯一执行合同](docs/qwen3-asr/02_development_plan.md)
- [数据方案](docs/qwen3-asr/03_data_plan.md)
- [Colab 方案](docs/qwen3-asr/04_colab_training_plan.md)
- [测试方案](docs/qwen3-asr/05_testing_plan.md)
- [风险与决策](docs/qwen3-asr/06_risks_and_decisions.md)
- [Mega-ASR 差距](docs/qwen3-asr/08_mega_asr_gap_analysis.md)

未完成 BF16 自有评测、release adapter 和 Mega-ASR 同 evaluator 外部 baseline 前，不声称
达到或超过 Mega-ASR。
