# 架构

## 背景与范围

目标是在 4 张 V100 上，用 `Qwen/Qwen3-ASR-1.7B` 官方 API 建立可复现的鲁棒 ASR 后训练闭环。
Mega-ASR 只作为方法参考和外部 baseline；不导入其 wrapper、训练入口或 target 规则。

完整后训练链路固定为 SFT → DPO → RL。SFT 先建立可加载的 adapter，DPO 用可审计 preference pairs，
RL 用固定 reward/rollout；三阶段都必须通过独立 gate。Router、Teacher、量化训练和 A2S 不在正式链路。

## 模块

```text
V100 独立环境 -> data builder（待实现） -> JSONL manifests
                                                           |
                                                           v
FP16 base -> inference runner（待实现） -> base predictions
                                      |
                                      v
                              single LoRA adapter（待实现）
                                      |
                                      v
                         adapter predictions -> evaluation/eval_wer.py
                                                        |
                                                        v
                                                WER/CER/gates/report
```

- 数据层未来负责 pinned source staging、音频物化、选择、校验和 manifest 输出，不加载训练模型。
- 训练层只消费固定 manifest/config，不负责下载数据；正式路径使用 `/data/mega-asr`，不读取
  `/data/mini-k3`。
- 运行时使用 FP16，不使用 BF16 或 FlashAttention-2；第一版 attention 为 eager，验证稳定后才
  评估 PyTorch SDPA。
- 训练使用单机 4 卡 DDP，global batch 必须显式包含 world size；推理固定单卡、batch 1 和
  解码上限，单条失败写入结果而不中断批次。
- 评测层只消费 prediction JSONL，英文和中文指标分开报告。

## 训练策略

SFT、DPO、RL 各使用一个独立 stage adapter 和独立输出目录。SFT 从 base 开始，DPO 从 SFT release 开始，
RL 从 DPO release 开始；任何阶段都不能跳过前一阶段。SFT pilot 使用 5k robust + 1k English clean +
1k Chinese clean；DPO 和 RL 使用各自不重叠的 pool，具体 schema、配额和 reward 见 08 号合同。

## 接口

SFT manifest 每行包含 `sample_id`、`audio`、`text`、语言/场景和完整 source/hash 信息；DPO manifest 额外
包含 `chosen`/`rejected`；RL rollout 额外包含 sampled prediction、reward components 和 KL。prediction
继承 `sample_id` 并增加 `prediction` 或 `error`。每个阶段保存 resolved config、manifest hash、world
size、checkpoint、adapter、metrics、rollouts（DPO/RL）和 gate。

## 测试与验收

本地合同测试必须验证 schema、确定性分区、resume、错误输出、DPO pair 审计、RL reward 和 WER/CER 聚合。
V100 验收必须覆盖 FP16 单 batch、4 卡 DDP、10+2 resume、SFT/DPO/RL pilot、clean/degraded 推理和最终固定 test。

## 影响

本次规划删除了 Colab/BF16/A2S 旧实现，不把历史代码或指标当作 V100 运行证据。历史指标只存在于
Git 历史，不能继续作为正式产品证据。远端数据未来只经 data builder 进入 `/data/mega-asr/data`；服务器 CLI
不保存第二份业务逻辑。
