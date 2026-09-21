# Qwen3-ASR Robust ASR

基于 `Qwen/Qwen3-ASR-1.7B` 官方 API 的独立鲁棒 ASR 微调项目。Mega-ASR 仅作为方法参考和
外部 baseline，不是代码依赖。

## 当前主线

```text
V100 服务器 -> 固定数据 manifest -> SFT -> DPO -> RL -> release adapter
           -> base/SFT/DPO/RL 对比 -> 固定 test 验收
```

- 训练目标仍为 `Qwen/Qwen3-ASR-1.7B`，运行时使用官方 `qwen-asr`/Transformers API。
- 目标服务器为 4 × Tesla V100-SXM2-32GB；训练采用 FP16、eager attention 和单机 4 卡 DDP。
- SFT、DPO、RL 都是完整后训练必做阶段；每阶段先做 pilot，再按 gate 扩大到 full。
- A2S、Teacher、Router 和 FlashAttention 不在正式链路；RL 使用固定 reward/rollout 合同。
- English 使用 WER，Chinese 使用 CER，并保留 scenario、失败输出和 32-cell 聚合。

2026-09-18 已完成服务器只读盘点和从零训练规划同步；没有安装依赖、修改远端或启动训练。旧的 Colab/BF16/A2S 训练和推理实现已删除，V100 runner 尚未重建，不能据此声称 GPU 训练已可用。

## 文件

```text
configs/      数据与训练的唯一配置
docs/         架构、开发、数据、训练、测试、进度和风险
notebooks/    预留 Notebook 目录，当前为空，详见 notebooks/README.md
scripts/      V100 数据 builder 与数据集物化脚本，详见 scripts/README.md
train/        V100 模型训练 runner（SFT/DPO/RL），详见 train/README.md
inference/    预留 V100 推理 runner，详见 inference/README.md
evaluation/   WER/CER 与聚合评测，详见 evaluation/README.md
tests/        不依赖模型下载的合同测试
```

目录说明：[notebooks](notebooks/README.md) · [scripts](scripts/README.md) ·
[train](train/README.md) · [inference](inference/README.md) · [evaluation](evaluation/README.md)。这些目录的文件或职责变化时，
必须在同一提交维护对应 README。

## 本地验证

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile evaluation/eval_wer.py scripts/download_sources.py scripts/stage_parquet_sources.py scripts/build_robust_manifests.py
```

## 入口

当前保留的 CLI 入口：

- 数据源拉取与门禁验证：
```bash
python3 scripts/download_sources.py --help
python3 scripts/stage_parquet_sources.py --help
python3 scripts/build_robust_manifests.py --help
python3 scripts/build_robust_manifests.py --config configs/data/public_robust_v100.yaml --mode verify-only
```



- 评测工具：
```bash
python3 evaluation/eval_wer.py --help
```

训练和推理 runner 待按 V100 方案重新实现。

执行合同见 [docs/qwen3-asr/README.md](docs/qwen3-asr/README.md)。未完成 SFT、DPO、RL 三阶段 pilot、固定评测、
release adapter 和 Mega-ASR 同 evaluator baseline 前，不声称达到或超过 Mega-ASR。
