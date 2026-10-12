# train

## 目录职责

提供 Qwen3-ASR-0.6B 的 SFT、DPO、RL runner，面向单张 A100 与 Colab 执行合同，读取固定配置和中文 JSONL manifest，并保存可恢复的完整 checkpoint。

## 当前文件清单

- `README.md`：本说明文档。各阶段训练 runner（`train_sft.py`、`train_dpo.py`、`train_rl.py`）将按 SFT → DPO → RL 顺序逐步实现并登记。

## 主要输入输出

- **输入**：`Qwen/Qwen3-ASR-0.6B` 官方权重、训练/验证 JSONL manifest、YAML 训练配置、随机种子、`run_id`、输出目录。DPO 阶段另输入包含离线 reference logprob 的 `pair_manifest.jsonl`。
- **输出**：遵循 `docs/qwen3-asr/04_training_plan.md` 的 run 目录，包含 `request.json`、`environment.json`、`resolved_config.yaml`、`manifest_sha256.json`、`metrics.jsonl`、`events.jsonl`、逐步 predictions、每次评测的 `eval/<step_name>/metrics.json` 与 `by_scenario.csv`、`gate.json`、`run_report.md`，以及完整 checkpoint 目录（含 `adapter/`、`optimizer.pt`、`scheduler.pt`、`rng_state.pt`、`sampler_state`、`training_state.json`、`sha256.json`）与原子更新的 `latest.json`。

## 使用入口

在 Phase 2–4 实现后提供：
```bash
# SFT
python3 train/train_sft.py --config configs/train/sft_pilot.yaml --manifest manifests/train.jsonl --val-manifest manifests/val.jsonl --output-dir runs/<run_id> --run-id <run_id> --seed 42

# DPO
python3 train/train_dpo.py --config configs/train/dpo_pilot.yaml --pair-manifest manifests/dpo_train_pairs.jsonl --val-manifest manifests/dpo_val_pairs.jsonl --output-dir runs/<run_id> --run-id <run_id> --seed 42

# RL
python3 train/train_rl.py --config configs/train/rl_pilot.yaml --manifest manifests/train.jsonl --val-manifest manifests/val.jsonl --output-dir runs/<run_id> --run-id <run_id> --seed 42
```

每个 runner 必须先做 schema、模型/processor、LoRA target、dtype/world size、manifest/hash、显存和一 batch forward/backward preflight；只在 preflight 通过后创建 optimizer。`--resume` 只能加载通过 checksum 与语义 hash 核对的 `latest.json`，不能从新 run 或失败 run 拼接状态。

## 维护要求

训练器必须严格保证单张 A100、BF16 优先、smoke 支持、断点恢复合同和配置随 checkpoint 保存；新增、重命名或修改 runner 时必须在同一提交同步更新对应目录的 `README.md`、开发文档与测试。
