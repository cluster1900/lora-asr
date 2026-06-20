# Colab 训练方案

最后更新：2026-06-20

## 目标

在 Colab 上跑通 Qwen3-ASR-1.7B 的 baseline 到 LoRA 训练完整闭环。

## Colab 资源现实约束

Qwen3-ASR-1.7B 比超大模型更适合 Colab，但全参训练仍不现实。现实路径是先跑 baseline，再做 QLoRA 或 LoRA，小 batch、梯度累积、短音频。

推荐硬件：

- Colab Free：baseline 脚本、数据准备、极小 smoke test。
- Colab Pro T4/L4：小规模 QLoRA 实验。
- Colab Pro+ A100：第一版更稳定的 LoRA/QLoRA 训练和较大 batch 评测。
- 外部 GPU runtime：适合数小时以上训练。

## Google Drive 目录

```text
/content/drive/MyDrive/qwen3-asr/
  data/
    raw/
    augmented/
    jsonl/
  checkpoints/
    qwen3-asr-1.7b-lora/
  outputs/
    baseline/
    eval/
  logs/
```

## Notebook 00: 拉取或更新工程

名称：

- `notebooks/00_clone_github_colab.ipynb`

目的：

- 在新的 Colab Free runtime 中挂载 Google Drive。
- 如果 `/content/drive/MyDrive/qwen3-asr` 不存在，则从 GitHub clone 项目。
- 如果目录已存在且是 git 仓库，则执行 `fetch + pull --ff-only`。
- 打印当前本地 commit、远端 commit 和关键修复标记，避免只看到 `Already up to date` 却无法判断代码版本。

为什么需要单独 notebook：

- Colab 的 Drive 目录会跨 runtime 保留，代码可能来自旧 clone。
- `git pull` 输出 `Already up to date` 只说明当前本地分支相对远端没有新提交，不说明当前 notebook 是否拿到了我们期望的修复。
- 训练 notebook 失败时，必须先确认 Drive 中的工程代码版本，再继续排查模型或依赖问题。

输入：

- GitHub 仓库：`https://github.com/cluster1900/lora-asr.git`
- 分支：`main`
- Drive 项目目录：`/content/drive/MyDrive/qwen3-asr`

输出：

- Drive 中的项目目录。
- 当前 `HEAD` 短 hash。
- 最新一条 commit message。
- 关键文件存在性和关键字符串检查结果。

测试标准：

- 空 Drive 项目目录时，可以 clone 到 `/content/drive/MyDrive/qwen3-asr`。
- 已存在 git 仓库时，可以 fast-forward 更新。
- notebook 必须打印 `git rev-parse --short HEAD` 和 `git log -1 --oneline`。
- notebook 必须检查 `resolve_training_model`、`target_root_prefix`、`gradient_checkpointing: false` 等关键修复标记。

验收标准：

- 执行完成后显示“版本验收通过”。
- 若 Drive 中代码不是 git 仓库，必须明确报错并提示用户先确认目录处理方式。
- 默认不得丢弃用户在 Drive 仓库里的本地修改；只有手动设置 `FORCE_RESET=True` 才允许强制对齐远端。

## Notebook 01: Baseline

名称：

- `notebooks/01_baseline_colab.ipynb`

目的：

- 加载 `Qwen/Qwen3-ASR-1.7B`。
- 在小评测集上运行 Qwen3-ASR `transcribe`。
- 计算 WER/CER。
- 保存预测结果。

主要步骤：

1. 挂载 Google Drive。
2. 安装依赖。
3. 登录 Hugging Face，如果需要。
4. 加载 Qwen3-ASR model。
5. 对每条音频生成转写。
6. 归一化预测。
7. 计算 WER/CER。

输出：

- `outputs/baseline/predictions.jsonl`
- `outputs/baseline/metrics.json`

## Notebook 02: 数据构建

名称：

- `notebooks/02_make_dataset_colab.ipynb`

目的：

- 下载或挂载 clean speech。
- 生成 degraded audio。
- 构建 train/val/test JSONL。

输出：

- `data/jsonl/train.jsonl`
- `data/jsonl/val.jsonl`
- `data/jsonl/test.jsonl`

## Notebook 03: LoRA 训练

名称：

- `notebooks/03_train_lora_colab.ipynb`

目的：

- 先完成 Qwen3-ASR 模块探测和 LoRA target 候选导出。
- 再训练第一版 ASR LoRA adapter。

当前 notebook 覆盖训练前探测和 Transformers + PEFT smoke training。Qwen3-ASR
官方 `qwen-asr` wrapper 的内部模块结构需要以当前环境真实加载结果为准，不能
复用 Mega-ASR 或其他工程的 target 规则。

历史 Unsloth 兼容性检查结果：

- `compatible=false`。
- 失败原因是 Transformers AutoConfig 不识别 `model_type=qwen3_asr`。
- 后续训练入口回退到 Transformers + PEFT，不继续在当前 MVP 中调试 Unsloth 依赖。
- 兼容性结论保留在 `outputs/lora_probe/qwen3_asr_1_7b/unsloth_compatibility.json`；当前 notebook 不再重复执行已知失败的 Unsloth 安装和检查。

依赖注意事项：

- 当前 notebook 固定 `qwen-asr==0.0.6`、`transformers==4.57.6`、`accelerate==1.12.0`，避免破坏 Qwen3-ASR 官方 wrapper 已验证过的依赖组合。
- Colab 预装包中 `pandas`、`requests` 容易被升级到冲突版本，因此 notebook 固定 `pandas==2.2.2`、`requests==2.32.4`。
- PEFT 注入 LoRA 时会探测 `torchao`。部分 Colab runtime 预装 `torchao==0.10.0`，而当前 PEFT 只支持 `torchao>0.16.0`，会在 `get_peft_model()` 阶段报错。当前 notebook 默认卸载 `torchao`，因为本 smoke training 不依赖 torchao。
- 训练 manifest 不包含音频本体；执行 LoRA smoke training 前，`data/mvp_eval/audio/` 必须已经同步到 Google Drive 项目目录。notebook 会在加载模型前检查前 20 条训练样本的音频路径。
- 如果安装后出现 pip resolver warning，先确认目标脚本是否能运行，再决定是否重启 runtime；warning 不一定等于 cell 失败。

初始配置：

```yaml
model_id: Qwen/Qwen3-ASR-1.7B
quantization: 4bit
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 2e-5
epochs: 1
max_audio_seconds: 20
max_new_tokens: 256
gradient_checkpointing: false
```

smoke 阶段默认关闭 gradient checkpointing。当前 LoRA target 位于 audio tower，
而 Qwen3-ASR 不是普通文本 LLM；在 k-bit PEFT 准备阶段启用 checkpointing 可能
引入自定义架构兼容问题。先用 4bit + batch size 1 跑通训练闭环，再决定是否单独
测试 checkpointing。

训练阶段：

0. Probe：加载模型并导出 `named_modules()`、LoRA target 候选和摘要。
1. PEFT compatibility：按正则精确匹配 99 个 audio tower target。
2. Smoke test：20 条样本，5-20 steps，当前入口为 `train/train_qwen3_asr_lora.py`。
3. MVP train：10k-20k 样本，1 epoch。
4. Scenario-balanced train：按声学场景做加权采样。

输出：

- `outputs/lora_probe/qwen3_asr_1_7b/module_snapshot.json`
- `outputs/lora_probe/qwen3_asr_1_7b/module_summary.csv`
- `outputs/lora_probe/qwen3_asr_1_7b/lora_target_candidates.json`
- `outputs/lora_probe/qwen3_asr_1_7b/lora_target_candidates.md`
- `outputs/lora_probe/qwen3_asr_1_7b/unsloth_compatibility.json`
- `checkpoints/qwen3-asr-1.7b-lora/target_modules.json`
- `checkpoints/qwen3-asr-1.7b-lora/training_config.json`
- `checkpoints/qwen3-asr-1.7b-lora/loss_log.jsonl`
- `checkpoints/qwen3-asr-1.7b-lora/summary.json`
- `checkpoints/qwen3-asr-1.7b-lora/adapter/`

## Notebook 04: 评测

名称：

- `notebooks/04_eval_colab.ipynb`

目的：

- 对比 base model、LoRA always-on，以及可用时的 router mode。

指标：

- overall WER/CER。
- scenario-level WER/CER。
- clean regression。
- empty output rate。
- long hallucination rate。
- latency per audio second。

输出：

- `outputs/eval/base.jsonl`
- `outputs/eval/lora.jsonl`
- `outputs/eval/metrics_by_scenario.csv`

## Notebook 05: Router

名称：

- `notebooks/05_router_colab.ipynb`

目的：

- 训练并校准 clean/degraded 音频分类器。

输出：

- router checkpoint。
- threshold config。
- routing evaluation report。

## 训练目标格式

baseline 推理不使用聊天 prompt，而是使用 `Qwen3ASRModel.transcribe(audio=..., language=...)`。训练阶段使用底层 forward + labels，因此需要显式构造 prompt 和 answer mask。

当前 smoke training target 格式：

```text
language English<asr_text>THE TRANSCRIPT TEXT
```

`labels` 中 prompt 和 padding 全部置为 `-100`，只有 `THE TRANSCRIPT TEXT` 和结束 token 参与 loss。后续仍可对比纯转写目标：

```text
THE TRANSCRIPT TEXT
```

最终格式由验证集 WER、空输出率、重复输出率和 clean regression 决定。

## Checkpoint 策略

- 每 200-500 optimizer steps 保存一次。
- Colab 中保留最近 3 个 checkpoint。
- 最终 adapter 同步到 Google Drive。
- 每个 adapter 旁边写入 `training_config.json`。

## 提前停止条件

出现以下情况应提前停止：

- loss 变成 NaN。
- 输出大多为空。
- 输出出现大量重复模板文本。
- validation WER 连续两次变差。
- Colab runtime 接近超时且近期没有保存 checkpoint。

## 测试标准

- 每个 notebook 能从空 Colab runtime 按顺序执行关键单元。
- `00_clone_github_colab.ipynb` 必须能确认本地代码版本和关键修复标记。
- notebook 使用 Google Drive 根目录变量，不写死个人本地路径。
- baseline notebook 至少能对 1 条 clean 和 1 条 degraded 音频推理。
- 训练 notebook 必须先完成 5-20 step smoke test。
- 训练 notebook 必须在加载模型前验证 smoke manifest 的音频路径，缺失时输出缺失样例。
- 训练 notebook 的依赖 cell 必须处理 Colab 预装旧版 `torchao`，避免 PEFT LoRA 注入阶段失败。
- 评测 notebook 能读取 prediction JSONL 并输出 WER/CER。
- 失败信息要写入输出文件或日志，不能只显示在 notebook cell 中。

## 验收标准

- `00_clone_github_colab.ipynb` 完成 clone/update，并打印当前 commit 与版本验收结果。
- `01_baseline_colab.ipynb` 产出 baseline predictions 和 metrics。
- `02_make_dataset_colab.ipynb` 产出 train/val/test JSONL。
- `03_train_lora_colab.ipynb` 产出可加载 LoRA adapter。
- `04_eval_colab.ipynb` 产出 overall 和 scenario-level 指标。
- `05_router_colab.ipynb` 在 router 阶段产出阈值和分类指标。
- 所有 notebook 的输入、输出和依赖都能追溯到配置文件。
