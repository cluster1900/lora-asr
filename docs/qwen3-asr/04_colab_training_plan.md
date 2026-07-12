# Colab 快速训练方案

最后更新：2026-07-12

## 目标

用一个 Notebook 完成环境安装、数据 staging、smoke/resume、正式训练、checkpoint 恢复和
评测入口。Colab 是第一优先环境，但不依赖 Google Drive 逐条读取训练音频。

唯一入口：

- `notebooks/12_fast_finetune_colab.ipynb`

## 目录

```text
/content/mega-asr                         # git 工作区
/content/mega-asr-runtime/                # 本地 SSD 数据与临时输出
/content/drive/MyDrive/mega-asr-artifacts # 持久 manifest/shard/checkpoint/result
```

Git 仓库与大数据 artifact 分开，避免把训练缓存误提交到代码仓库。

## 环境固定

Notebook 第一段必须记录：

- 当前 git commit。
- GPU 型号、数量和显存。
- CUDA、Torch、qwen-asr、Transformers、PEFT、Datasets、FlashAttention 版本。
- Qwen 模型 revision 与 Qwen 官方 finetuning 源码 commit。
- 数据配置、训练配置和随机种子。

新增一个 `requirements-colab.txt` 固定首轮依赖。Notebook 不允许每次安装不同版本后继续
训练；依赖变化必须形成新实验 id。

## Notebook 顺序

### 1. 挂载与检查

- 挂载 Drive。
- 拉取/更新仓库并显示 commit。
- 检查 GPU 是否支持 BF16 和 FlashAttention 2。
- 检查 Drive 可用空间与 `/content` 本地空间。
- 生成 `run_id`，所有输出写入独立目录。

### 2. 数据 staging

- 根据 pinned revision 下载或从 Drive 大 shard cache 恢复。
- 运行 `prepare_public_robust_manifests.py` smoke/full。
- 把训练所需音频 staging 到 `/content/mega-asr-runtime/data`。
- 运行 manifest、解码、配额、防泄漏和人工抽听检查。
- 训练开始后不访问 Hugging Face 网络。

禁止将 200k 个小音频逐个写入 Drive。需要持久化时按大 shard/tar 保存。

### 3. Golden batch 与 10+2 smoke

- 加载 pinned `Qwen/Qwen3-ASR-1.7B`。
- 验证 343 target 分组数量和禁止模块。
- 打印一条 en/zh golden batch 的 prompt、label mask 摘要和有效 target token 数。
- 128 条平衡样本训练 10 optimizer step并保存。
- 释放进程后从 checkpoint resume 2 step。
- 新进程加载 adapter，推理 en/zh clean/degraded 各 1 条。

任何一项失败都不进入正式 200k。

### 4. BF16 base

至少先完成 10k validation 的 BF16 base prediction，供 canary 和 checkpoint 选择使用。

若有第二个 Colab/GPU runtime，Bench 5k、LibriSpeech test-clean、AISHELL-1 test 的 base
评测可与训练并行；只有一个 runtime 时按 validation base -> train -> fixed test base 的
顺序执行，避免同时占用显存。

### 5. 正式 200k run

参考单卡 A100 40GB：

| 参数 | 值 |
| --- | --- |
| per-device batch | 4 |
| gradient accumulation | 16 |
| effective batch | 64 |
| precision | BF16 |
| gradient checkpointing | on |
| max audio | 30 秒 |
| duration bucketing | on |
| learning rate | 1e-6 |
| epoch | 1 |

其他 GPU 只调整 per-device batch 与 accumulation，并保持 effective batch 64。Resolved
配置必须保存，不能只保存模板 YAML。

执行门：

- Step 100：固定 512 条 canary，失败自动停止。
- 50%：保存 adapter + 完整 Trainer state，跑 10k validation。
- 100%：保存 adapter + 完整 Trainer state，跑 10k validation。
- Canary 通过后可删除临时 checkpoint，只保留日志和 canary 指标。

### 6. 选择与固定测试

- 仅根据 10k validation 选择 50% 或 100% checkpoint。
- 使用统一推理入口，可选 `--adapter-dir`，base 与 LoRA 不再维护两份批处理实现。
- Prediction 每条完成后增量写入并 flush，支持 `--resume` 跳过已完成 sample id。
- 唯一候选跑 Bench 5k、LibriSpeech test-clean、AISHELL-1 test。
- 生成 comparison、32-cell 指标和错误分析。

## Checkpoint 与恢复

正式 checkpoint 必须包含：

- adapter 权重与配置。
- optimizer、scheduler、RNG 和 Trainer state。
- processor/tokenizer 或其 pinned source revision。
- resolved training config。
- manifest hash 与数据 revision。
- target_modules 分组、实际可训练参数量。
- git commit、依赖版本和 run id。

训练 checkpoint 先写本地临时目录，再原子复制到 Drive。恢复前校验文件大小和 hash。

## 产品产物

```text
checkpoints/qwen3-asr-public-200k-broad-lora/
  adapter/
  processor/
  trainer_state/
  release_manifest.json
  resolved_training_config.yaml
  target_modules.json
```

Release 验证必须在全新 Python 进程中只使用 base model id、adapter path 和 release
manifest 完成四条 smoke 推理及一批 JSONL 推理。

## 失败处理

- OOM：减小 micro batch、等比例增加 accumulation；不改 LR、target 或 effective batch。
- 数据下载中断：从 shard 状态继续；不重新随机选样。
- 训练断开：从最近正式 checkpoint resume；不得把 adapter-only reload 当作 resume。
- Canary 崩溃：停止并排查，不继续训练观察是否自行恢复。
- 固定 test 中单条失败：写入 `error` 并继续整批。

## 验收

Notebook 从干净 runtime 能按上述顺序完成；所有关键 cell 可重复运行；Drive 只承担大文件
持久化，不成为逐样本训练 I/O；最终给出 adapter、release manifest、prediction、metrics
和实际运行命令。
