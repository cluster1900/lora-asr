# 架构方案

## 背景与目标

项目目标是在 `Qwen/Qwen3-ASR-0.6B` 上建立可复现的中文鲁棒 ASR 后训练链路。训练固定使用单张 A100；新架构优先保证数据、训练、推理和评测解耦，使 Colab runtime 重启不会改变实验定义。

## 模块

- `scripts/`：环境探针、manifest 构建、Drive 同步和阶段编排。
- `inference/`：统一的 base/SFT/DPO/RL 推理接口，输出原始与归一化文本。
- `train/`：三个阶段的训练 runner，读取配置和 JSONL manifest，不在代码中隐含数据路径。
- `evaluation/`：WER/CER、scenario 聚合、输出健康度和 gate。
- `configs/`：唯一的配置来源；运行时保存解析后的副本。
- `notebooks/`：Colab 编排和可视化，不承载不可恢复的业务逻辑。

训练接口参考 Modal Training Gym 的拆分方式：dataset、rollout/generate、reward、eval 和 train config 各自独立；只借鉴接口思想和可观测性，不引入其训练后端或私有 wrapper。

## 数据流

```text
source audio/text
  -> manifest builder
  -> train/val/test split
  -> base inference
  -> SFT
  -> preference pairs
  -> DPO
  -> fixed rollout + reward
  -> RL
  -> common evaluator
  -> gate + release adapter
```

## 阶段边界与依赖

| 阶段 | 读取 | 写出 | 下游允许使用的条件 |
| --- | --- | --- | --- |
| baseline | 固定 val/test manifest、base 模型、`decode.eval` | predictions、metrics、environment | metrics 与 manifest/eval hash 齐全 |
| SFT | train/val manifest、base release、SFT config | adapter、完整 checkpoint、gate | val gate 为 `passed` 才能成为 DPO policy |
| DPO | SFT release、train/val pair manifest、离线 reference logprob | adapter、pair audit、preference/CER gate | gate 为 `passed` 才能成为 RL policy/reference |
| RL | DPO release、train manifest、rollout config、冻结 reference | rollout、adapter、reward/KL/CER gate | gate 为 `passed` 才能发布 RL release |

每个边界都传递 `schema_version`、Git SHA、模型 revision、配置 hash、manifest hash、seed 和 eval decode hash；缺任一项时下游只能诊断，不能训练或发布。`latest.json` 是 run 内恢复指针，release 是通过门禁的不可变副本，两者不互相替代。

## 运行控制面

训练计算只读取 Colab 本地 staging；Drive 保存 manifest、环境指纹、追加式事件/指标、checkpoint 和 release。`request.json` 描述意图，`resolved_config.yaml` 描述实际参数，`training_state.json` 描述可恢复状态，`gate.json` 描述裁决。状态机和文件 checksum 由 `04_training_plan.md`、`07_colab_execution_contract.md` 共同定义；notebook 只能调用 CLI，不能拥有额外状态。

## 输入输出

训练输入是 JSONL manifest、YAML 配置、模型标识和随机种子。训练输出是 `04_training_plan.md` 规定的 run 目录。推理输出每行包含 `sample_id`、`prediction_raw`、`prediction_normalized`、`error` 和运行元数据。

评测解码 `decode.eval` 固定为 greedy。RL 采样只存在于 `decode.rollout`。阶段门禁读 val，test 只在发布时评一次。CER 使用 `zh-cer-v1`；reward 由同一套 CER 换算，两者分别报告。

## 达成标准

每个阶段可单独运行、恢复和评测；同一 manifest 和配置能够复现输入摘要、checkpoint 元数据和指标计算；任何阶段失败都能定位到具体样本或 gate。

## 测试与影响

架构验收由目录合同、manifest/preflight、推理失败续跑、5–20 step smoke、断点恢复、DPO pair 审计、RL reward/KL 和 gate 测试组成。该分层让 SFT、DPO、RL 可以独立替换 trainer，同时保持 JSONL、checkpoint 和 evaluator 接口稳定；代价是每个阶段必须保存更多 hash、审计和中间产物。
