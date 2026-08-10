# scripts 目录说明

## 目录职责

本目录保存训练前的数据准备工具。当前唯一脚本负责公开数据 metadata 探测、可恢复音频 staging、
固定配额 manifest、泄漏检查和 A2S curriculum；它不加载训练模型，也不执行推理或评测。

远端访问与本地构建严格分开：`probe`/`stage` 访问 pinned Hugging Face 数据源，`smoke`/`build`/
`validate` 只消费本地文件，`curriculum` 额外消费固定 BF16 base 的评分结果。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `prepare_public_robust_manifests.py` | 唯一公开数据入口，提供 `probe`、`stage`、`smoke`、`build`、`validate`、`curriculum` 六个子命令。 |
| `README.md` | 说明本目录边界、脚本子命令、输入输出和维护要求。 |

`__pycache__/` 是 Python 自动生成的本地缓存，已被 Git 忽略，不是数据产物，可随时删除。

## 子命令

| 子命令 | 是否访问远端 | 作用与主要产物 |
| --- | --- | --- |
| `probe` | 是，仅 metadata | 校验 dataset ID、40 位 revision、split、schema、行数和容量，不下载音频。 |
| `stage --mode smoke` | 是 | 按最小场景/语言配额物化 smoke 音频和 candidate JSONL。 |
| `stage --mode full` | 是 | 按正式配额物化音频；逐条追加、每 100 行持久化，并支持 hash 校验恢复。 |
| `smoke` | 否 | 从本地 candidates 生成 128-row train 和 4-row Bench smoke，执行音频/泄漏门禁。 |
| `build` | 否 | 生成 200k train、10k validation、512 canary、5k Bench 及统计/来源/校验报告。 |
| `validate` | 否 | 对指定 manifest 重新执行 schema、路径、计数、音频和跨切分泄漏检查。 |
| `curriculum` | 否 | 合并 train 与 base scored JSONL，生成 30k 选择和累计难度视图。 |

## 配置、输入与输出

唯一数据配置为 `configs/data/public_robust_200k.yaml`，其中固定数据源 revision、语言/场景配额、
时长范围、随机种子、输出文件名和 manifest 必需字段。

正式数据流：

```text
pinned Hub metadata/audio
  -> candidate JSONL + local audio + stage_report
  -> canonical train/validation/canary/test JSONL
  -> BF16 base scored JSONL
  -> 30k curriculum JSONL + cumulative views
```

所有选择使用固定 seed 和稳定 hash 排序。canonical manifest 每行至少包含 `sample_id`、`audio`、
`answer`、`language`、`scenario`、`condition_group`、来源 revision 和 selection role。

## 使用方式

```bash
python scripts/prepare_public_robust_manifests.py --help
python scripts/prepare_public_robust_manifests.py probe \
  --config configs/data/public_robust_200k.yaml
```

完整执行顺序由 `notebooks/12_fast_finetune_colab.ipynb` 统一编排。对应测试：
`tests/test_prepare_public_robust_manifests.py`。

## 维护要求

新增、重命名或删除本目录文件/子命令时，必须同步更新“文件清单”和“子命令”。修改数据源、配额、
schema、输出文件、恢复策略或泄漏规则时，必须在同一提交更新本 README、数据配置、
`docs/qwen3-asr/03_data_plan.md`、测试和进度文档。
