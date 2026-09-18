# 开发计划

## 背景与范围

仓库只维护一个 V100 服务器闭环，避免 Colab、A100 和服务器配置并存。当前范围是把现有静态
合同迁移为 FP16/eager attention/4 卡 DDP，并完整实现 SFT → DPO → RL；不新增 router、sweep、
Teacher 或独立评测器。

## 唯一流程

1. 新建 data builder：完成 pinned source、staging、manifest、泄漏检查和恢复。
2. 新建 inference runner：生成 FP16 base smoke/baseline/pilot prediction。
3. 新建 train runner：实现 SFT smoke/pilot、DPO pair training 和 RL rollout training，三阶段均使用 4 卡 DDP。
4. `evaluation/eval_wer.py`：接收 prediction 和 output directory，固定生成 scored JSONL、metrics
   JSON 及 scenario/cell/language CSV。

核心配置必须覆盖：

- V100 训练配置：服务器路径、FP16、eager attention、world-size-aware global batch。
- 数据配置：pinned revision、smoke/pilot/full 配额、manifest schema 和输出路径。

当前接口只接受唯一 `sample_id` 和 `audio` 的 manifest；迁移时将官方训练字段统一为 `audio` +
`text`，旧 `answer` 映射必须在代码变更中明确记录。训练输出必须保存 resolved config、target map
hash、manifest hash、world size、pipeline state、checkpoint 和 adapter。

## 开发步骤

1. 先更新 V100 环境、配置、依赖和路径文档；正式目录固定为 `/data/mega-asr`。
2. 把模型加载、Trainer 参数和推理合同改为 FP16/eager attention，移除对 FlashAttention-2 和
   BF16 的硬依赖。
3. 运行 metadata probe、128-row smoke 和 base smoke；schema、音频或泄漏失败时停止。
4. 跑 10+2 checkpoint/resume；确认 optimizer、scheduler、RNG、world size 和配置可恢复。
5. 对 `sft_train` 子集运行 SFT pilot，完成 degraded/clean gate。
6. 从独立 `dpo_pool` 生成 preference pairs，运行 DPO pilot，完成 preference 和 ASR gate。
7. 从独立 `rl_pool` 运行 DPO→RL rollout pilot，完成 reward、KL、degraded/clean gate。
8. 三个 pilot 全部通过后，才按 full 配额依次运行 SFT、DPO、RL 和 release test。

## 测试

每次代码变更运行：

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile evaluation/eval_wer.py
```

数据或训练合同变化还要运行 schema fixture、config validation 和 smoke。正式执行必须记录命令、method、
revision、manifest hash、随机种子、dtype、attention、world size 和 gate.json。

## 完成条件

- V100 独立环境可从配置生成固定 manifest。
- FP16/eager attention 的 clean/degraded 推理均能逐条写 prediction。
- 10+2 resume 成功且 resolved config、world size、manifest hash 随 checkpoint 保存。
- SFT、DPO、RL 前后均产出 English WER、Chinese CER、scenario、reward/preference 和失败统计。
- release adapter 和 processor 可在新进程重新加载。

## 影响

新功能必须直接延伸上述四个入口。训练器不提供跳过 pilot gate 或命令行注入旧 adapter 的入口；
恢复只读取当前 output directory 的 pipeline state 和 checkpoint。需要第二套入口时，先证明现有
接口无法表达需求并更新本文件。
