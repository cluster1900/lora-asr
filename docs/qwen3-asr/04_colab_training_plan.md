# Colab 训练方案

最后更新：2026-06-19

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

当前 notebook 的第一版只覆盖训练前探测，不直接启动训练。原因是 Qwen3-ASR
官方 `qwen-asr` wrapper 的内部模块结构需要以当前环境真实加载结果为准，不能
复用 Mega-ASR 或其他工程的 target 规则。

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
```

训练阶段：

0. Probe：加载模型并导出 `named_modules()`、LoRA target 候选和摘要。
1. Smoke test：20 条样本，5-20 steps。
2. MVP train：10k-20k 样本，1 epoch。
3. Scenario-balanced train：按声学场景做加权采样。

输出：

- `outputs/lora_probe/qwen3_asr_1_7b/module_snapshot.json`
- `outputs/lora_probe/qwen3_asr_1_7b/module_summary.csv`
- `outputs/lora_probe/qwen3_asr_1_7b/lora_target_candidates.json`
- `outputs/lora_probe/qwen3_asr_1_7b/lora_target_candidates.md`
- LoRA adapter checkpoint。
- trainer state。
- training logs。

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

baseline 推理不使用聊天 prompt，而是使用 `Qwen3ASRModel.transcribe(audio=..., language=...)`。训练阶段的 target 格式需要在 LoRA MVP 前通过官方示例、模块结构和 smoke training 决定。

候选 target 格式：

```text
language English<asr_text>THE TRANSCRIPT TEXT
```

也要测试纯转写目标：

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
- notebook 使用 Google Drive 根目录变量，不写死个人本地路径。
- baseline notebook 至少能对 1 条 clean 和 1 条 degraded 音频推理。
- 训练 notebook 必须先完成 5-20 step smoke test。
- 评测 notebook 能读取 prediction JSONL 并输出 WER/CER。
- 失败信息要写入输出文件或日志，不能只显示在 notebook cell 中。

## 验收标准

- `01_baseline_colab.ipynb` 产出 baseline predictions 和 metrics。
- `02_make_dataset_colab.ipynb` 产出 train/val/test JSONL。
- `03_train_lora_colab.ipynb` 产出可加载 LoRA adapter。
- `04_eval_colab.ipynb` 产出 overall 和 scenario-level 指标。
- `05_router_colab.ipynb` 在 router 阶段产出阈值和分类指标。
- 所有 notebook 的输入、输出和依赖都能追溯到配置文件。
