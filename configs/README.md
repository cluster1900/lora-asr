# configs

## 目录职责

配置是数据、推理和训练的唯一声明来源。运行时必须保存解析后的配置副本；禁止在 notebook 或 runner 中隐藏关键超参数。学习率、LoRA 结构、`decode.eval`、`decode.rollout` 和门禁阈值都在开跑前写入配置，开跑后保持不可变。

## 当前文件清单

- `README.md`：本说明文档。具体中文 Qwen3-ASR-0.6B YAML 配置将在 Phase 0–4 逐步加入并登记。

当前没有可运行的训练 YAML；`configs/train/sft_pilot.yaml`、`dpo_pilot.yaml`、`rl_pilot.yaml`、`configs/inference/base_eval.yaml` 和 `configs/data/manifest_mvp.yaml` 是待实现入口，不应在 Phase 0 尚未完成前被当作已存在文件。字段和默认值以 [`docs/qwen3-asr/04_training_plan.md`](../docs/qwen3-asr/04_training_plan.md) 的 `qwen3-asr-train-v1` 为准。

## 主要输入输出

- **输入**：开发者或实验调度器指定的超参数与路径。
- **输出**：在运行启动时解析并固化至 `runs/<run_id>/resolved_config.yaml`，同时写语义 `config_sha256` 和 YAML 字节 hash；训练 checkpoint 必须携带该副本。

## 使用入口

由各 CLI 脚本通过 `--config` 参数读取，例如：
```bash
python3 train/train_sft.py --config configs/train/sft_pilot.yaml ...
```

开发阶段先用文档模板生成配置，配置提交前运行未知字段、有限数值、stage/parent release、`decode.eval.do_sample=false`、`world_size=1` 和必填 gate 的 preflight。恢复只能使用原 run 的 resolved config；改变学习率、预算、阈值或 LoRA 目标必须新建 run。

## 维护要求

新增配置模板或修改配置 schema 时，必须在同一提交同步更新对应目录的 `README.md` 和训练/数据方案文档。
