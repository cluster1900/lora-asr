# scripts 目录说明

## 目录职责

本目录维护 V100 服务器的数据 builder 与数据集准备脚本。所有实现必须从固定 source、音频 staging、manifest、泄漏检查和恢复语义出发，确保训练、验证、评测角色物理隔离无泄漏。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `download_sources.py` | E1 阶段原始数据源拉取与缓存：锁定 4 个 pinned revision，支持 HF 镜像（`hf-mirror.com`）与 OpenSLR 镜像断点续传，产出 `RAW_COMPLETE.json`。 |
| `stage_parquet_sources.py` | E1 阶段 Parquet 数据流式解码与音频转码标准化：从 HF Parquet 文件直接流式解包音频 bytes，通过 `ffmpeg` 转码为 16kHz mono PCM 16-bit WAV，校验时长并产出 `staged_<source>.jsonl` 与 `PROCESSED_COMPLETE.json`。 |
| `build_robust_manifests.py` | E1 阶段角色划分与门禁构建器：摄入各源 staged 样本，执行哈希分区无泄漏角色分配（7 个角色）、抽取 128 条平衡 smoke 子集与 pilot 子集、rejects 记录、SHA-256 校验和以及各级完成门禁（含 `DATASET_COMPLETE.json`）。 |
| `build_dpo_pairs.py` | E5 阶段 DPO 偏好对构建与参考对数概率预计算器：基于真实的 Base 预测与 SFT Step 50 预测（必须显式提供 `--base-predictions` 与 `--sft-predictions`），经 WER/CER 排序构造真实的 `(chosen, rejected)` 偏好对；正式模式严禁使用 gold 答案或合成负例兜底（杜绝 gold 泄漏）；严格过滤平局（`cand_sft == cand_base` 或 `err_sft == err_base`）；使用冻结 SFT 合并底座离线预计算 `ref_chosen_logp` 与 `ref_rejected_logp`，记录完整 Provenance 哈希并生成 `dpo_pair_audit.json`。 |
| `README.md` | 说明当前目录职责、文件清单、CLI 参数与维护要求。 |

## 使用入口与 CLI 参数

```bash
# 1. 下载原始数据源并记录 RAW_COMPLETE.json
python3 scripts/download_sources.py \
    --config configs/data/public_robust_v100.yaml \
    --raw-dir /data/mega-asr/data/raw \
    --endpoint https://hf-mirror.com

# 2. 流式解包 Parquet 并通过 ffmpeg 转码为 16kHz mono WAV（支持指定 source 或 all）
python3 scripts/stage_parquet_sources.py \
    --config configs/data/public_robust_v100.yaml \
    --source all \
    --endpoint https://hf-mirror.com

# 3. 哈希隔离分区并产出全套 7 角色 manifest、smoke.jsonl、pilot 子集与 DATASET_COMPLETE.json
# （在数据未达全额配额时使用 --no-strict-quotas 标记 NON_STRICT_SUBSET）
python3 scripts/build_robust_manifests.py \
    --config configs/data/public_robust_v100.yaml \
    --staged-dir /data/mega-asr/manifests \
    --output-dir /data/mega-asr/manifests \
    --data-dir /data/mega-asr/data \
    --mode full \
    --no-strict-quotas

# 4. 深度核验已有 manifest 门禁与音频完整性
# 非全量模式校验子集门禁（预期 VERIFIED (NON_STRICT_SUBSET)）
python3 scripts/build_robust_manifests.py --config configs/data/public_robust_v100.yaml --mode verify-only --verify-audio
# 严格模式校验全额配额（全量达标前判为 FAILED）
python3 scripts/build_robust_manifests.py --config configs/data/public_robust_v100.yaml --mode verify-only --strict-quotas

# 5. 构建 DPO 偏好对并预计算参考模型对数概率（强制要求传入真实 Base 与 SFT 预测）
python3 scripts/build_dpo_pairs.py \
    --candidate-manifest /data/mega-asr/manifests/pilot_dpo.jsonl \
    --base-predictions /data/mega-asr/runs/dpo_candidates/base_pilot_dpo_predictions.jsonl \
    --sft-predictions /data/mega-asr/runs/dpo_candidates/sft_step50_pilot_dpo_predictions.jsonl \
    --output-manifest /data/mega-asr/manifests/pilot_dpo_pairs.jsonl \
    --rejects /data/mega-asr/manifests/rejects_dpo.jsonl \
    --audit-output /data/mega-asr/runs/dpo_pair_audit.json \
    --ref-model-dir /data/mega-asr/runs/sft_pilot_controlled/merged_base \
    --target-pairs 3000
```


### 主要输入输出与 Schema

- **输入配置**：`configs/data/public_robust_v100.yaml`（定义 4 个 pinned 数据源与 7 个数据角色）。
- **主要输出**：
  - `manifests/RAW_COMPLETE.json`：原始数据下载完成标记。
  - `manifests/PROCESSED_COMPLETE.json`：音频转码与标准化完成标记。
  - `manifests/smoke.jsonl`：128 条平衡 smoke 数据（32 en + 32 zh + 64 degraded），供 Base 推理与续训 Smoke 使用。
  - `manifests/<role>.jsonl`：7 个角色（`sft_train`, `dpo_train_pool`, `dpo_val_pool`, `rl_train_pool`, `rl_val_pool`, `validation`, `bench_test`）的 JSONL manifest。
  - `manifests/pilot_<name>.jsonl`：pilot 子集 manifest。
  - `manifests/rejects.jsonl`：校验未通过或清洗丢弃的样本记录（含丢弃原因）。
  - `manifests/<role>_COMPLETE.json`：各角色完成指标与 hash。
  - `manifests/manifest_sha256.json`：所有 manifest 的 sha256 汇总。
  - `manifests/DATASET_COMPLETE.json`：E1 阶段总验收门禁文件（确认无跨角色泄漏与配额达成）。

## 维护要求

新增、重命名或删除脚本时，必须同步更新“文件清单”和仓库根 README。修改数据源、配额、schema、
输出文件、恢复策略或泄漏规则时，必须同步更新本 README、`docs/qwen3-asr/03_data_plan.md`、
开发/测试/进度文档和对应测试。
