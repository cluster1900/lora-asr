# V100 服务器后训练方案

## 背景与范围

本方案把正式训练环境从 Colab/A100 假设切换到 `ai@v100-ssh.apexolab.com`。目标是在 4 张
Tesla V100-SXM2-32GB 上，用官方 `qwen-asr`/Transformers API 对 `Qwen/Qwen3-ASR-1.7B`
进行可复现的完整鲁棒 ASR 后训练，顺序为 SFT → DPO → RL。

本方案负责环境、数据、SFT、DPO、RL、评测和阶段门禁。Mega-ASR 只作为方法参考和
外部 baseline；不复用其私有 wrapper、训练入口或 target 规则。当前文档同步不代表已经安装环境
或启动训练。

## 服务器盘点

| 项目 | 结果 | 规划影响 |
|---|---|---|
| GPU | 4 × Tesla V100-SXM2-32GB，Compute Capability 7.0，NVLink 全互联 | 使用单机 4 卡 DDP |
| CPU/内存 | 56 逻辑线程，46 GiB RAM | dataloader worker 初始设为每进程 1-2 |
| 存储 | `/data` 可用约 5.5 TB | 独立保存数据、cache、runs 和结果 |
| 软件 | Driver 580.178.04，CUDA toolkit 12.8；现有 venv 有 PyTorch 2.5.1+cu121 | 新建项目 venv，不复用 `/data/mini-k3/venv` |
| 网络 | Hugging Face 直连不稳定，hf-mirror、ModelScope、PyPI 可访问 | 固定镜像变量和下载记录 |

当时检查 GPU 无训练进程，但 CPU 有其他 `/data/mini-k3` 数据任务。正式训练前必须确认 CPU
资源和 `/data` 配额，不得终止或读取其他项目任务。

## 运行时合同

- dtype：`float16`；`bf16=false`，`tf32=false`。
- attention：第一版 `eager`；只有在 smoke 稳定后才评估 PyTorch SDPA。
- 不安装或依赖 FlashAttention-2。V100 属于 SM70，而官方 FlashAttention-2 支持矩阵以
  Ampere/Ada/Hopper 为主。
- 训练：`torchrun --nproc_per_node=4`，DDP 不使用 `device_map`。
- LoRA 目标固定为 199 个 Linear：音频 `thinker.audio_tower.conv_out/proj1/proj2` 3 个，加上 28 层
  LLM Decoder 的 attention/MLP 196 个；完整 canonical allowlist、禁止模块和 `target_map_hash` 见 08 号合同；$r=16, \alpha=32$。
- 阶段权重交接：SFT、DPO、RL 各阶段完成并通过门禁后，统一执行 `merge_and_unload()` 产出合并基座作为下阶段起点。
- DPO 显存优化：支持离线预计算参考模型 logprobs，DPO 训练无需在 GPU 常驻参考模型，显存直降 50%。
- RL 算法：采用带 KL 正则的 GRPO，候选组大小 $G=4$，采样 `temperature=0.85`、`top_p=0.92`、`top_k=50`，基于声学强 conditioning 保证候选多样性，避免方差过早归零。
- 推理：固定单卡、batch 1、相同模型 revision 和解码上限；支持 4 进程按 manifest shard 分片并行推理后合并去重。
- gradient checkpointing：开启（PEFT 需配合 `enable_input_require_grads()`）。
- 初始 micro batch：每卡 1；初始 accumulation：16；global batch 为 `1 × 16 × 4 = 64`。
  只有显存和吞吐稳定后才提高到 128。
- 所有 run 保存 resolved config、代码 commit、model revision、method、manifest hash、world size、seed、
  reward/preference config（DPO/RL）和环境版本。

建议目录：

```text
/data/mega-asr/
├── repo/       # 代码 checkout
├── venv/       # 本项目独立环境
├── cache/      # HF/ModelScope/音频缓存
├── data/       # 物化音频
├── manifests/  # smoke/pilot/full JSONL
├── runs/       # checkpoint、pipeline state、adapter
├── results/    # prediction 和评测
└── logs/       # 命令、环境和阶段日志
```

## 从 0 的阶段

### S0：合同冻结

只更新配置、schema、数据角色、SFT/DPO/RL 合同和 gate；不下载模型、不启动训练。通过标准是所有
输入输出字段、revision、seed、阈值和停止条件已冻结。

### S1：完整数据下载/物化（第一个可执行步骤）

先完整下载或物化四个 pinned source 的原始音频、metadata、source identity 和许可证信息，再创建
`sft_train`、`dpo_train_pool`、`dpo_val_pool`、`rl_train_pool`、`rl_val_pool`、`validation` 和 `bench_test`
的 source manifest。不得在数据未完成前下载模型、跑 base inference 或启动任何训练。

通过标准：四个 source 有完整 hash；所有角色配额准确、source identity 不重叠、音频可解码、manifest
可恢复；写出 `DATASET_COMPLETE.json`。DPO pairs 和 RL rollouts 需要模型输出，分别在 S5/S6 从已经
准备好的 source pool 生成。

### S2：环境、模型加载和 base smoke

在 `DATASET_COMPLETE.json` 存在后创建独立环境，加载固定 model revision，以 FP16/eager attention
跑 128-row smoke 和 base baseline。保存 `environment.json`、base predictions、WER/CER、scenario 和
失败统计。

通过标准：4 卡 world size 正确；clean/degraded 至少各一条成功；无 OOM/NaN；所有失败有 `error`。

### S3：SFT checkpoint smoke

用 4 卡 DDP 运行 10 step，保存 checkpoint；新进程恢复并继续 2 step。检查 optimizer、scheduler、RNG、
adapter、global step、world size、`adapter.save_pretrained()` 和副本 `merge_and_unload()`。

通过标准：step 10→12 连续、所有训练状态可恢复、adapter 和 merged 权重均可加载。

### S4：SFT pilot

使用 `sft_train` 的 5k robust + 1k English clean + 1k Chinese clean（`pilot_sft.jsonl`，共 7,000 条）。

针对首轮实验发现的 Clean 回退（英文 Clean WER 从 1.98% 回退至 3.41%）及模型拟合现象，S4 阶段设立严格受控对照实验：
- **基线配置（Run 1）**：`lr=2e-5`, `warmup_steps=0`, 恒定学习率, 均匀无重放随机打乱（退化样本占比 71.4%），训练 500 steps（累计 32,000 样本次，~4.57 epochs）。
- **受控优化配置（Run 2）**：
  - 学习率下调至 `1e-5`，配合 `warmup_steps=50` 与 300 步全程 `linear decay`；
  - 采用平衡采样策略 `--sample-strategy balanced`（固定 50% degraded + 25% English clean + 25% Chinese clean，或 `--sample-strategy clean_2x` 进行 Clean 2 倍重放），纠正数据偏向；
  - 步数缩减至 300 steps（累计 19,200 样本次，~2.7 epochs），每 50 步保存检查点；
  - 采用 `evaluation/eval_checkpoint_series.py` 自动化批量评测 step 50/100/150/200/250/300，收紧门禁规则（要求 Robust Macro 不得恶化且空输出率 $\le 0.002$）；综合退化场景改善数与 Clean 保真度遴选 Pareto 最优检查点；
  - 经 2,867 条独立验证集全量实测，选定 **Step 50**（EN Clean 1.94%、ZH Clean 1.40%、Clean Macro 1.67%、Robust Macro 10.40% 全面优于 Base，0 空输出）执行 `merge_and_unload()`，产出正式进入 E5 DPO 的 SFT 合并底座并归档 `gate.json`。

### S5：DPO pair 与 DPO pilot

从已下载的 `dpo_train_pool`/`dpo_val_pool` 生成可审计 pairs，支持离线 reference logprobs；使用 4 卡
DDP 从 merged SFT 底座运行 DPO pilot。通过 preference、ASR、clean 累积退化和 merge gate 后保存 merged RL 底座。

### S6：RL rollout 与 RL pilot

从已下载的 `rl_train_pool`/`rl_val_pool` 生成 rollout；使用 4 卡 GRPO（G=4、temperature=0.85、top_p=0.92、top_k=50，推荐 30 步内防过拟合），
同一 `group_id` 的候选在 advantage 前完成 all-gather。通过 reward、KL、ASR、clean 累积退化和 merge gate 后保存最终 release。

### S7：full SFT → full DPO → full RL

SFT、DPO、RL 三个 pilot 全部通过后，才按 08 号合同的 full pool 依次运行。每阶段使用
`torchrun --standalone --nproc_per_node=4`，保留 adapter、merged 权重、manifest hash、checkpoint、rollout
和 gate；任何阶段失败都停止扩大并回退到最后有效版本。

### S8：最终 release 和外部比较

对 base、SFT、DPO、RL 四个版本在同一 validation 和 5,000 Bench test 上评测，保存四组 prediction、
WER/CER、scenario、clean/degraded、失败统计、DPO preference 和 RL reward 摘要。最终 release 包含
RL adapter 与合并发布模型（且 clean 相对 Base 全局回退 ≤0.025）；只有三阶段均通过，才可与 Mega-ASR 使用同一 evaluator 比较。

## 目录产物

每一阶段都必须有：

- `environment.json`：Python、PyTorch、CUDA、GPU、qwen-asr 和 Git commit；
- `resolved_config.yaml`：完整训练配置；
- `manifest_sha256.json`：输入 manifest hash；
- `predictions/`：base、SFT、DPO、RL 逐条结果；
- `rollouts/`：DPO pair 审计和 RL reward/KL 结果；
- `metrics.json`、`by_language.csv`、`by_scenario.csv`；
- `gate.json`：每阶段阈值、实际值和 status；
- `pipeline_state.json` 和最后有效 checkpoint；
- `failure_samples.jsonl`：空输出、重复、过长、hallucination 和 inference error。

## 停止条件

出现以下任一情况立即停止当前阶段：FP16 非有限 loss/gradient、OOM、DDP global batch 漂移、
resume 不连续、manifest 泄漏、clean regression 超阈值、有效输出率低于 0.95，或模型加载使用
了非官方 wrapper。停止时保留日志、输入 hash 和最后有效 checkpoint。

## 验收

V100 训练阶段只有同时满足以下条件才算完成：

1. 有固定命令、配置、manifest、seed 和环境记录；
2. 有 base、SFT、DPO、RL 四组 prediction；
3. 有 WER/CER、scenario、clean/degraded、失败统计；
4. 通过 10+2 checkpoint/resume、DPO preference gate、RL reward gate 和新进程加载；
5. SFT、DPO、RL 三个 pilot/full gate 有明确结果；
6. 文档、进度和风险记录已同步；
7. 有 release adapter 和可复现的输出路径。

在这些产物齐全前，不标记阶段完成，也不声称超过 Mega-ASR。
