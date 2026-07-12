# Qwen3-ASR Robust ASR

独立的 Qwen3-ASR-1.7B 鲁棒 ASR 微调项目。Mega-ASR 只作为方法与外部 baseline，
不作为主实现代码底座。

## 当前状态

历史 baseline、LoRA v1-v5、4bit base recheck、WER/CER 和错误分析已经跑通。历史最优 v3
只在 4bit 对照下对 noise+reverb 改善 6.71%，没有证明相对 BF16 base 的正式净收益。

新的快速主线尚未实现，当前 0/3：

1. 公开 200k train、10k validation 和固定 test。
2. Qwen 官方 Trainer 薄适配、343-target BF16 LoRA 和 10+2 resume smoke。
3. 一次 200k 正式训练、100-step canary、50%/100% validation 和 release adapter。

完整执行合同见 [docs/qwen3-asr/02_development_plan.md](docs/qwen3-asr/02_development_plan.md)。

## 固定方案

- Base model：`Qwen/Qwen3-ASR-1.7B`。
- Train：160k robust + 20k English clean + 20k Chinese clean。
- Validation：8k robust + 1k English clean + 1k Chinese clean。
- LoRA：343 Linear targets，r=8，alpha=16，dropout=0.05，BF16，`lr=1e-6`。
- 正式 run：1 epoch，effective batch 64，step 100 canary，50%/100% 候选。
- Test：Voices-in-the-Wild-Bench 5k、LibriSpeech test-clean、AISHELL-1 test。
- 第一轮不做 teacher、router、RL、自建增强、全量 difficulty scoring 或 target sweep。

## 仓库结构

```text
configs/      历史配置；新主线配置待实现
data/         JSONL 与历史 smoke/MVP 数据
docs/         当前方案、测试、进度和风险
evaluation/   WER/CER 与错误分析
inference/    历史 base/LoRA 推理入口
notebooks/    Colab notebook；00-11 为历史主线
outputs/      历史 prediction、metrics 和分析
scripts/      数据与辅助脚本
train/        模块探测和历史 PEFT trainer
router/       仅占位，第一轮不实现
references/   本地忽略的外部参考
```

## 当前可运行的历史 smoke

```bash
python3 scripts/create_smoke_audio.py --force
python3 evaluation/eval_wer.py --help
python3 evaluation/analyze_errors.py --help
```

新主线的数据 CLI、官方 Trainer、统一推理、32-cell evaluator 和 Notebook 12 尚未实现；
不要把历史 notebook 继续扩展为正式 200k 训练入口。

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
