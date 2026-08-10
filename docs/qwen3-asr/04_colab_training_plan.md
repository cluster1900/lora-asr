# Colab 训练方案

## 背景与范围

Colab 是第一优先训练环境。第一轮只训练一个 BF16 LoRA adapter，不做量化训练、direct SFT
对照、target/LR sweep、teacher 或 RL。

唯一入口是 `notebooks/12_fast_finetune_colab.ipynb`。Notebook 只挂载 Drive、安装 pinned 依赖并
依次调用仓库 CLI；不内嵌第二份 Python 实现。

## 环境与配置

依赖由 `requirements-colab.txt` 固定，保留 Colab 自带 CUDA torch。模型、Qwen 源码和数据集均
使用 40 字符 revision。新产物统一写入 Drive 的 `qwen3-asr-public-a2s/` 独立命名空间，训练时
音频放 `/content/qwen3-asr-runtime/` 本地 SSD，避免读取旧 Mega-ASR 实验目录。

训练配置：LoRA r=8、alpha=16、dropout=0.05，effective batch 128。Phase I/II 学习率
`1e-6`；Phase III audio/projection `5e-7`、decoder `1e-6`。

## 执行顺序

1. 挂载 Drive、拉取仓库、安装 `requirements-colab.txt` 和固定 flash-attn。
2. `probe -> stage --mode smoke -> smoke`，然后 BF16 base 跑 smoke clean/degraded。
3. 训练配置校验后跑 10 step，再由新进程 `--smoke-steps 12 --resume auto` 续 2 step。
4. smoke 门禁通过后才运行 `stage --mode full -> build`。
5. BF16 base 对随机固定 train 顺序先评分 60k；若 `<0.70` 的可用行不足 30k，逐级扩到
   100k、160k、200k，并用同一 prediction 文件 `--resume`。
6. 生成 curriculum 与 512 base canary，再启动正式三阶段训练。
7. release adapter 对固定 10k validation 和 5k Bench 各评测一次。

训练命令：

```bash
python train/train_qwen3_asr_a2s.py \
  --config configs/train/qwen3_asr_public_200k_a2s.yaml \
  --smoke-steps 10

python train/train_qwen3_asr_a2s.py \
  --config configs/train/qwen3_asr_public_200k_a2s.yaml \
  --smoke-steps 12 \
  --resume auto

python train/train_qwen3_asr_a2s.py \
  --config configs/train/qwen3_asr_public_200k_a2s.yaml
```

正式训练按 Phase I -> II -> III 顺序执行。阶段边界保存 adapter 和完整 Trainer checkpoint；
canary 不通过时不得继续下一阶段。新 run 不接受旧 adapter，也不能通过 CLI 跳过阶段。

## Teacher 决策

首轮不需要 Teacher。四个公开数据源已经提供 gold transcript，A2S curriculum 需要的是固定 BF16
base 相对 gold 的 WER/CER，而不是新的伪标签。GPT-5.5 当前也不是音频输入模型；即使通过
API key/base URL 接入，也只能做文本后处理，既不能替代 ASR Teacher，又会改变 gold label 和评测
口径。该输入模态以 [OpenAI 官方 GPT-5.5 模型页](https://developers.openai.com/api/docs/models/gpt-5.5)
为准。只有未来加入无标注音频时，才单独评审音频转写模型；不把该分支塞进当前 notebook。

## 测试与验收

- 10 step 后保存，重新加载并继续 2 step；global step 必须连续。
- resolved config、model revision、target map hash 和 manifest hash 随 run 保存。
- 每阶段 loss、gradient、learning rate 均为有限值。
- 每阶段 canary 分别报告 robust language macro 与 clean language macro；robust 相对 base 回退不超过
  15%，clean 绝对错误率增加不超过 0.02，同时报告空输出、重复、过长、幻觉式输出和
  inference error。
- 最终 adapter/processor 可从新进程加载并推理 clean/degraded 各一条。

## 影响与停止条件

OOM 时先降低 per-device batch 并等比增加 gradient accumulation，保持 effective batch 128。
target 合同漂移、resume 不连续、非有限训练状态或 canary 超阈值时立即停止，不通过跳过门禁来继续。
