# 架构

## 背景与范围

目标是在 `Qwen/Qwen3-ASR-1.7B` 官方 API 上快速得到可评测的鲁棒 ASR adapter。Mega-ASR
仅作为方法和外部 baseline；不导入其 wrapper、训练入口或 target 规则。本轮不做 router、RL、
teacher、自建增强或多 adapter 编排。

## 模块

```text
公开数据 -> scripts/prepare_public_robust_manifests.py -> JSONL manifests
                                                    |
                                                    v
BF16 base/adapter -> inference/qwen3_asr_infer.py -> predictions.jsonl
          |                                         |
          |                                         v
          +-> train/train_qwen3_asr_a2s.py      evaluation/eval_wer.py
                     |                               |
                     v                               v
             adapter/checkpoint                 WER/CER/report
```

- 数据层只负责选择、校验和 curriculum，不加载训练模型。
- 训练层只消费固定 manifest/config，不负责下载数据。
- 推理层固定模型 revision、BF16、`cuda:0`、batch 1 和 manifest 语言，只允许切换 base/adapter；
  单条失败写入结果而不中断批次。
- 评测层只消费 prediction JSONL，英文和中文指标分开报告。

## 接口

manifest 每行至少包含 `sample_id`、`audio`、`answer`、`language`、`scenario`、来源信息和
`audio_sha256`。prediction 继承 `sample_id` 并增加 `prediction` 或 `error`。训练输出必须保存
resolved config、target map hash、pipeline state、checkpoint 和 adapter。

## A2S

单个 LoRA adapter 预注入 343 个 Linear target：audio attention 96、audio MLP 48、projection
3、decoder attention 112、decoder MLP 84。阶段仅切换可训练参数：

1. 30k curriculum x2：upper-4 audio + projection，共 27 target。
2. 200k x1：decoder，共 196 target。
3. 200k x1：全部 343 target 联合训练。

## 测试与验收

本地合同测试必须验证 target 数量/分组、确定性选择、resume、错误输出和 WER/CER 聚合。GPU
验收还必须覆盖 10+2 resume、clean/degraded 推理、三阶段 canary 与最终固定测试集。

## 影响

本次精简删除历史复现资产和可造成评测漂移的推理参数。历史指标只存在于 Git 历史，不能继续
作为正式产品证据。
