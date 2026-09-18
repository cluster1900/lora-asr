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
- 推理：固定单卡、batch 1、相同模型 revision 和解码上限。
- gradient checkpointing：开启。
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

### S0：环境和代码迁移

固定 Python、PyTorch、`qwen-asr`、Transformers、Accelerate、PEFT、datasets 版本，设置
`HF_ENDPOINT`、`HF_HOME` 和 `/data/mega-asr` 路径。把训练和推理从 BF16/FlashAttention/Colab
合同迁移到 FP16/eager/server 合同。

完成标准：配置能解析；不加载模型的本地测试通过；依赖、路径、dtype、attention 和 world size
均可在 resolved config 中看到。

### S1：数据 smoke 和 base baseline

先生成 128-row smoke，覆盖 robust split、clean 两种语言和至少一个 degraded 音频。随后只在
同一 smoke/512-canary 上跑 FP16 base，保存原始 prediction、WER/CER、scenario 和失败统计。

完成标准：音频可读、无 source leakage、base 输出可恢复、空输出和推理错误均有记录。

### S2：LoRA checkpoint smoke

用 4 卡 DDP 运行 10 step，保存 checkpoint；新进程恢复并继续 2 step。检查 optimizer、scheduler、
RNG、adapter、global step、world size 和配置是否一致。

完成标准：无 OOM、loss/gradient/learning rate 有限、step 从 10 连续到 12、adapter 可重新加载。

### S3：SFT pilot

使用 `sft_train` 的 5k robust + 1k English clean + 1k Chinese clean，先训练 SFT adapter。

通过条件：至少一个 degraded scenario 改善；clean 错误率绝对增加不超过 0.02；有效输出率不少于
0.95；失败率增加不超过 0.05；新进程能加载 SFT adapter。

### S4：DPO pair 与 DPO pilot

从独立 `dpo_pool` 生成 chosen/rejected pairs，不能读取 validation/test。先运行 2k robust + 500
English clean + 500 Chinese clean 的 DPO pilot，起点为 SFT release。

通过条件：held-out preference accuracy ≥0.55；至少一个 degraded scenario 相对 SFT 改善；clean
回退不超过 0.02；有效输出率不少于 0.95；DPO 没有读取 gold/error-rate 审计字段。

### S5：RL rollout 与 RL pilot

从独立 `rl_pool` 取 2k robust + 500 English clean + 500 Chinese clean，以 DPO pilot 为 policy、
冻结 DPO 副本为 reference policy，使用固定 WER/CER、空输出、重复、过长、hallucination 和 KL reward。
四张 V100 同时生成 rollout 并训练，保存每条 reward、reward components、KL 和 sampled transcript。

通过条件：held-out mean reward 相对 DPO reference 提升 ≥0.05；至少一个 degraded scenario 相对 DPO
改善；clean 回退不超过 0.02；有效输出率不少于 0.95；reward、KL、gradient 均有限。

### S6：full SFT → full DPO → full RL

SFT、DPO、RL 三个 pilot 全部通过后，才按 08 号合同的 full pool 依次运行。每阶段使用 `torchrun --standalone --nproc_per_node=4`，保留上阶段 adapter、manifest hash、checkpoint 和 gate；任何阶段失败都停止
扩大并回退到最后有效版本。

### S7：release 和外部比较

对 base、SFT、DPO、RL 四个版本在同一 validation 和 5,000 Bench test 上评测，保存四组 prediction、
WER/CER、scenario、clean/degraded、失败统计、DPO preference 和 RL reward 摘要。最终 release 必须
是 RL adapter；只有三阶段均通过，才可与 Mega-ASR 使用同一 evaluator 比较。

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
