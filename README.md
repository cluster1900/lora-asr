# Qwen3-ASR Robust ASR

基于 `Qwen/Qwen3-ASR-1.7B` 官方 API 的独立鲁棒 ASR 微调项目。Mega-ASR 仅作为方法参考和
外部 baseline，不是代码依赖。

## 当前主线

```text
公开数据 -> 固定 JSONL -> BF16 base -> 单 adapter A2S -> BF16 base/adapter 对比
```

- Train：160k robust + 20k English clean + 20k Chinese clean。
- Validation：8k robust + 1k English clean + 1k Chinese clean。
- A2S：30k curriculum x2 -> 200k decoder x1 -> 200k joint x1。
- LoRA：r=8、alpha=16、dropout=0.05，27/196/343 target 分阶段启用。
- Eval：English WER、Chinese CER、scenario 和 32-cell 聚合。

历史 v1-v6A、router、旧 notebook、已提交数据、checkpoint 和 prediction 已移除。正式数据物化、
GPU smoke 和训练结果尚未完成。

## 文件

```text
configs/      数据与训练的唯一配置
docs/         架构、开发、数据、训练、测试、进度和风险
scripts/      公开数据 manifest/curriculum
train/        单 adapter A2S runner
inference/    base/adapter 统一批量推理
evaluation/   WER/CER 与聚合评测
tests/        不依赖模型下载的合同测试
```

## 本地验证

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile scripts/prepare_public_robust_manifests.py \
  train/train_qwen3_asr_a2s.py inference/qwen3_asr_infer.py evaluation/eval_wer.py
```

## 入口

```bash
python scripts/prepare_public_robust_manifests.py --help
python train/train_qwen3_asr_a2s.py --help
python inference/qwen3_asr_infer.py --help
python evaluation/eval_wer.py --help
```

执行合同见 [docs/qwen3-asr/README.md](docs/qwen3-asr/README.md)。未完成 BF16 固定评测、release
adapter 和 Mega-ASR 同 evaluator baseline 前，不声称达到或超过 Mega-ASR。
