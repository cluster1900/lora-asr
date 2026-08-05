# Colab 快速 A2S 训练方案

最后更新：2026-07-22

## 背景与目标

首轮目标不是维护多条训练路线，而是在一个 Colab Notebook 中完成数据 staging、BF16 base、
单 adapter 三阶段 A2S、断点恢复和最终评测。训练顺序与参数以
`02_development_plan.md` 为唯一执行合同；本文件只说明 Colab 上如何可靠执行该合同。

唯一入口：

- `notebooks/12_fast_finetune_colab.ipynb`

## 范围

Notebook 只执行以下闭环：

1. 固定环境和数据 revision。
2. 完成 metadata probe、manifest 与本地 SSD staging。
3. 通过 golden batch、target switch、10+2 resume 和四条生成 smoke。
4. 生成 BF16 base 结果和 30k curriculum。
5. 在同一 adapter 上依次完成 Phase I、II、III。
6. 完成一次 full validation、一次固定 test 和 release reload。

不在 Notebook 中创建第二套 trainer、模型选择 sweep 或额外实验分支。

## 目录与 I/O

```text
/content/mega-asr                         # git 工作区
/content/mega-asr-runtime/                # 本地 SSD 音频、缓存和临时 checkpoint
/content/drive/MyDrive/mega-asr-artifacts # 持久 shard、manifest、checkpoint 和结果
```

- Git 仓库与大数据 artifact 分开，避免训练缓存进入版本库。
- Drive 只保存大 shard、manifest、状态、checkpoint 和结果，不逐条读取 200k 小音频。
- 训练音频先物化到 `/content/mega-asr-runtime/data`，正式训练开始后不依赖 Hugging Face
  网络。
- Checkpoint 先写本地临时目录，校验完整性后再原子复制到 Drive。

## 环境固定

Notebook 第一段必须写入 run metadata：

- 当前 git commit、`run_id` 和固定随机种子。
- GPU 型号、数量、显存以及 BF16、FlashAttention 2 支持情况。
- CUDA、Torch、qwen-asr、Transformers、PEFT、Datasets、FlashAttention 版本。
- Qwen 模型 revision、Qwen 官方 finetuning 源码 commit 和数据 revision。
- 数据配置、训练配置及其 SHA256。

`requirements-colab.txt` 固定首轮依赖。依赖或 revision 变化必须产生新的 `run_id`，不能在
原实验目录中覆盖继续。

## Notebook 执行顺序

### 1. 挂载与预检

- 挂载 Drive，拉取仓库并显示 commit。
- 检查 GPU 能力、Drive 空间和 `/content` 本地空间。
- 创建独立 run 目录，验证上次 checkpoint 是否可恢复。
- 先运行 metadata-only probe，确认固定 revision、54 个 split、字段、配额和预计磁盘占用；
  probe 失败时不下载音频。

### 2. 数据 staging

- 从固定 revision 下载或从 Drive 大 shard cache 恢复。
- 运行 `prepare_public_robust_manifests.py` 的 128-row smoke，再生成 full manifest。
- 把训练音频 staging 到本地 SSD。
- 验证 200k train、10k validation、5k Bench 的行数、音频解码、语言/场景配额、duration、
  rejects 和跨 split 零泄漏。
- 保存 manifest、stats、rejects、validation report 和 SHA256；抽听通过后锁定 manifest。

### 3. Trainer smoke 与恢复

- 加载 pinned `Qwen/Qwen3-ASR-1.7B`，验证 343 个 target 的分组、运行时类型和 hash。
- 运行 en/zh golden batch，检查 prompt、audio mask、label mask、padding 和目标文本。
- 逐阶段运行 target-switch smoke，确认只有当前 scope 的 LoRA 参数收到梯度。
- 用 128 条平衡 fixture 训练 10 个 optimizer step，保存完整 checkpoint。
- 释放进程，在新进程中 resume 到 step 12；global step、optimizer、scheduler 和 RNG 连续。
- 新进程加载 base+adapter，推理 en/zh clean/degraded 各 1 条。

任一 smoke 失败都不能进入正式训练。

### 4. BF16 base 与 curriculum

- 使用统一推理入口生成固定 10k validation 的 BF16 base prediction，作为阶段 canary 和最终
  validation 的共同 base。
- 只对 robust train 的分层候选做 BF16 base 推理，写入 `base_error_rate`，生成固定 30k
  curriculum 的 `<0.30`、`<0.50`、`<0.70` 累计视图。
- 固定 test 的 BF16 base prediction 可在另一 GPU runtime 并行生成；单 runtime 时在正式训练
  前后串行生成，但每个 test manifest 只生成一次。

### 5. 单 adapter 三阶段 A2S

正式训练只启动一次 runner，并在同一 adapter 上切换 `requires_grad` 与 optimizer parameter
groups：

| 阶段 | 数据与 exposure | 激活 target | 学习率 | warmup |
| --- | --- | --- | --- | --- |
| Phase I | 30k curriculum x 2 epoch | upper-4 audio + projection，共 27 | audio/projection `1e-6` | 0.05 |
| Phase II | full 200k x 1 epoch | decoder，共 196 | decoder `1e-6` | 0.05 |
| Phase III | full 200k x 1 epoch | all，共 343 | audio/projection `5e-7`；decoder `1e-6` | 0.03 |

共同配置：BF16、LoRA r=8、alpha=16、dropout=0.05、weight decay 0.01、max grad norm
1.0、linear scheduler、0.5-30 秒、duration bucketing、FlashAttention 2、gradient
checkpointing 和 effective batch 128。

单卡 A100 40GB 参考 per-device batch 4、gradient accumulation 32。其他 GPU 只调整这两个
参数的组合，并保持 effective batch 128。Phase I 三个累计难度视图按配置中的等 optimizer-step
三段执行；这是项目假设，必须写入 resolved config。三阶段总 sample exposure 为
`30k x 2 + 200k + 200k = 460k`。

### 6. 阶段门与最终评测

```text
Phase I 完成 -> 固定 512 validation canary
Phase II 完成 -> 同一固定 512 validation canary
Phase III 完成 -> 同一固定 512 validation canary
               -> full 10k validation 一次
               -> Bench 5k + 两个 clean test 一次
               -> release reload
```

每次 canary 检查 loss、梯度、输出有效率、robust macro、empty、repeat-like 和 too-long。
canary 失败立即停止并保存诊断；它只做安全门，不用于挑选 checkpoint。正式路径只对 Phase III
执行一次 full 10k validation。仅当 Phase III 未通过、且 Phase II canary 正常时，才按需对
Phase II 做一次 full validation 以定位回退点；该结果不触发自动续训。

### 7. 发布

- Phase III full validation 预检通过后，固定唯一 adapter，运行 Bench 5k、LibriSpeech test-clean 和
  AISHELL-1 test；不得根据 test 反向选择 checkpoint。
- 生成 prediction、metrics、32-cell comparison、failure summary 和 release manifest。
- 在全新 Python 进程中只使用 base model id、adapter path 和 release manifest，完成四条
  smoke 推理和一批可恢复 JSONL 推理。

## Checkpoint 与恢复合同

每个可恢复 checkpoint 必须包含：

- 单一 adapter 权重与配置。
- optimizer、scheduler、RNG、Trainer state 和 pipeline phase/state。
- processor/tokenizer 或 pinned source revision。
- resolved training config、target map/hash 和实际可训练参数量。
- manifest hash、数据/model revision、git commit、依赖版本和 `run_id`。

断线后只从最近完整 checkpoint 恢复，不能把 adapter-only load 当成 resume。恢复时必须校验
hash、当前 phase、global step 和 curriculum view；不重新采样数据。

## 失败处理

- OOM：减小 micro batch、等比例增加 accumulation，保持 effective batch 128 和其余参数不变。
- 下载中断：按 shard 状态继续，不重新随机选样。
- 训练断开：从最近完整 checkpoint resume。
- 阶段 canary 失败：停止并保存 predictions、failure rows、梯度和配置诊断。
- 批量推理单条失败：写入 `error` 并继续，不中断整批。

结果解释和停止动作严格采用 `05_testing_plan.md` 的四个互斥分支；Notebook 不自行追加训练。

## 验收与影响

Notebook 从干净 runtime 可按上述顺序执行，关键 cell 可重复运行，10+2 resume 和阶段切换
可验证，Drive 不成为逐样本 I/O 瓶颈。最终必须产出单一 adapter、完整恢复状态、resolved
config、release manifest、prediction、metrics 和实际运行命令。

这一路径删除重复全量训练与重复 full evaluation；其影响是首轮只有一个正式候选，所有收益和
失败均可直接归因到固定数据、A2S 阶段与单一 target map。
