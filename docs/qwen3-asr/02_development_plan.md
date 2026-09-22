# 开发计划

## 背景与范围

仓库只维护一个 V100 服务器闭环，避免 Colab、A100 和服务器配置并存。当前范围是把现有静态
合同迁移为 FP16/eager attention/4 卡 DDP，并完整实现 SFT → DPO → RL；不新增 router、sweep、
Teacher 或独立评测器。

## 唯一流程

1. 新建 data builder：完成 pinned source、staging、manifest、泄漏检查和恢复。
2. 新建 inference runner (`inference/run_inference.py`)：单卡 batch 1、FP16、eager attention，基于官方 `qwen-asr` / Transformers API，支持加载 Base 与 PEFT Adapter，具备断点续推（基于已有 `sample_id` 跳过）、逐行 flush+fsync 容错写入与 `--eval` 联动 `evaluation/eval_wer.py` 自动产出评测报告。
3. 新建 train runner (`train/train_sft.py`)：实现 SFT smoke/pilot 训练，支持单机 4 卡 DDP 与 `--single-gpu` 模式，操作底层 `model.model.thinker` 并开启 `gradient_checkpointing` 与 `enable_input_require_grads`；精确注入 199 个 Linear LoRA targets（Projection 3 + Decoder 196）；实现标签掩码 Cross Entropy 损失计算；具备规范 10+2 检查点保存与恢复机制（保存 adapter、optimizer、scheduler、RNG、training_state 与 resolved config）；提供 `--export-merged` 支持 `merge_and_unload()` 权重合并导出。后续以此为基础延伸 DPO 与 RL 训练。
4. 新建 DPO 偏好对构建器 (`scripts/build_dpo_pairs.py`)：从候选音频池基于真实的 Base 与 Step 50 SFT 独立预测生成真实的 `(chosen, rejected)` 偏好对；正式模式强制传入 `--base-predictions` 和 `--sft-predictions`，严禁使用 gold 答案兜底或合成负例（杜绝 gold 泄漏）；经 WER/CER 判定过滤平局（`cand_sft == cand_base` 或 `err_sft == err_base` 一律剔除）；离线预计算冻结参考模型的序列对数概率 `ref_chosen_logp` 与 `ref_rejected_logp`，记录完整 Provenance 哈希并生成 `dpo_pair_audit.json`。
5. 新建 DPO train runner (`train/train_dpo.py`)：实现 4 卡 DDP DPO 训练，基于冻结参考模型 logp 计算隐式奖励差与 Bradley-Terry 偏好损失，实时监控 `preference_accuracy` 与 `margin`；支持全量 held-out 验证集评估；遵循 10+2 检查点契约、保存 4 卡独立 RNG 并支持 `--export-merged`。
6. 新建 RL (GRPO) train runner (`train/train_rl.py`)：实现 4 卡 DDP GRPO 训练，从 `dpo_release` 合并底座起训；生成组大小 $G=4$ 的采样 rollout（`temperature=0.7, top_p=0.9`）；按 `reward_config.yaml` 严格计算 ASR 奖励与 empty、repeat、too_long、hallucination 惩罚；组内优势按均值与标准差标准化并具备零方差置零保护；计算策略梯度与针对冻结参考模型的 token 级 KL 正则惩罚（$\beta=0.04$）；保存 `rollouts.jsonl`、`loss_log.jsonl`、10+2 检查点并提供 `--export-merged` 导出合并权重。
7. `evaluation/eval_wer.py`：接收 prediction 和 output directory，固定生成 scored JSONL、metrics JSON 及 scenario/cell/language CSV。
8. `evaluation/verify_gate.py`：支持 SFT、DPO 与 RL 阶段机器可读门禁计算与归档，强校验 held-out preference/reward、退化改善数、Clean 回退与累积回退红线；DPO 与 RL 阶段门禁严格要求 Robust Macro 相对前一阶段零恶化（`max_robust_macro_regression <= 0.0`），并完整记录前序基线 metrics 与 predictions 的路径与 SHA-256 哈希。

核心配置必须覆盖：

- V100 训练配置：服务器路径、FP16、eager attention、world-size-aware global batch。
- 数据配置：pinned revision、smoke/pilot/full 配额、manifest schema 和输出路径。

当前接口只接受唯一 `sample_id` 和 `audio` 的 manifest；迁移时将官方训练字段统一为 `audio` +
`text`，旧 `answer` 映射必须在代码变更中明确记录。训练输出必须保存 resolved config、target map
hash、manifest hash、world size、pipeline state、checkpoint 和 adapter。

## 开发步骤

1. 先冻结 V100、SFT/DPO/RL、数据 schema、revision、seed 和 gate 合同；正式目录固定为 `/data/mega-asr`。
2. 运行完整数据下载/物化和角色池 builder，直到 `DATASET_COMPLETE.json` 存在；数据未完成前不下载模型、不跑 inference、不训练。
3. 创建独立 V100 环境，迁移 FP16/eager attention，移除 FlashAttention-2/BF16 依赖，并记录 environment.json。
4. 运行 128-row base smoke、10+2 checkpoint/resume；确认 4 卡 world size、optimizer、scheduler、RNG、adapter 和 merge 可恢复。
5. 对 `pilot_sft.jsonl`（7,000 条：5,000 degraded + 1,000 en clean + 1,000 zh clean）运行 4 卡 DDP SFT pilot（指定 `--allow-subset` 明确合规子集边界；全量训练强制要求 `DATASET_COMPLETE.json` 状态为 `PASSED`），在独立 `validation.jsonl` 上验证 degraded/clean gate，并通过 `merge_and_unload()` 产出 SFT Step 50 最优底座。
6. 从独立 `dpo_train_pool` 与 `dpo_val_pool` 分别运行 Base 和 SFT Step 50 真实推理，基于双模型实际预测生成纯真实 preference pairs（禁止 gold/synthetic fallback；严格过滤平局；离线预计算 Step 50 参考模型 logp），运行 DPO pilot 并执行全额 held-out 验证，通过 zero-regression 门禁后产出 RL 基座。
7. 从独立 `rl_train_pool` 与 `rl_val_pool` 运行 4 卡 GRPO rollout pilot（同组 G=4，advantage 前 all-gather），完成 reward、KL、clean 累积约束 gate。
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
- release adapter、合并发布模型和 processor 可在新进程重新加载。

## 影响

新功能必须直接延伸上述四个入口。训练器不提供跳过 pilot gate 或命令行注入旧 adapter 的入口；
恢复只读取当前 output directory 的 pipeline state 和 checkpoint。需要第二套入口时，先证明现有
接口无法表达需求并更新本文件。
