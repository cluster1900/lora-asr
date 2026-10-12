# scripts

## 目录职责

提供环境探针、中文 manifest 构建与校验、Drive 同步管理、运行目录初始化以及阶段启动支持脚本。为各训练与评测环节提供可复现的自动化 CLI 工具。

## 当前文件清单

- `README.md`：本说明文档。辅助脚本（`colab_bootstrap.py`、`drive_sync.py`、`manifest_validate.py`、`build_dpo_pairs.py` 等）将在各阶段按需实现并登记。

## 主要输入输出

- **输入**：原始音频与标注、Drive 挂载路径、配置 YAML、随机种子以及待同步的本地 checkpoint 目录。
- **输出**：结构化环境报告 `environment.json`、校验后的 `manifest_sha256.json`、符合 schema 的中文 train/val/test manifest、Drive 持久化 checkpoint、DPO pair/reference-logprob 审计、追加式 `events.jsonl` 和 `run_report.md`。

## 使用入口

在 Phase 0–3 实现后提供：
```bash
# 环境启动与探针
python3 scripts/colab_bootstrap.py --config configs/train/sft_pilot.yaml --drive-root /content/drive/MyDrive/qwen3-asr --run-id <run_id> --seed 42

# Manifest 校验
python3 scripts/manifest_validate.py --manifest manifests/train.jsonl --check-audio

# DPO Pair 构建与离线参考 logprob 预计算
python3 scripts/build_dpo_pairs.py --sft-release releases/sft/<sft_run_id> --train-manifest manifests/train.jsonl --val-manifest manifests/val.jsonl --output-dir manifests/dpo/<pair_id> --seed 42
```

`drive_sync.py` 必须先写本地 staging 和 `sha256.json`，再复制 `.incoming`，读回核对后才更新 `latest.json`；manifest/pair builder 必须同时写 schema 版本、source 泄漏审计和 rejected 行，不能只返回一个行数。

## 维护要求

脚本职责、CLI 参数、输入输出 schema 变更，以及新增、重命名或删除脚本时，必须在同一提交同步更新对应目录的 `README.md`、开发文档和测试；严禁将训练状态只记录在 shell 临时变量中。
