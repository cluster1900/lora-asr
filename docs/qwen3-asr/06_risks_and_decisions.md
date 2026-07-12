# 风险与决策

最后更新：2026-07-12

## 当前决策

### D001: 基础模型固定为 Qwen3-ASR-1.7B

- 使用 `Qwen/Qwen3-ASR-1.7B`。
- 正式训练和评测固定 BF16；4bit 只保留历史复现价值。
- 模型、qwen-asr 和官方 finetuning 源码必须 pin revision。

### D002: 第一轮直接使用公开 200k

- Train：160k robust + 20k English clean + 20k Chinese clean。
- Validation：8k robust + 1k English clean + 1k Chinese clean。
- Test：Voices-in-the-Wild-Bench 5k + LibriSpeech test-clean + AISHELL-1 test。
- 不开发自建增强工厂，不训练 50k/100k 过渡模型。

原因：历史 TTS source 太少；公开数据已覆盖 7 atomic 和 54 个 compound 场景。Broad
decoder LoRA 且首轮无 router，因此 clean retention 固定 20%。

### D003: 正式 Trainer 迁向 Qwen 官方结构

参考 Qwen 官方 `finetuning/qwen3_asr_sft.py` 的 prompt、collator、label mask、Trainer、
scheduler、validation 和 resume。当前逐样本 trainer 冻结为历史，不继续扩建。

PEFT target、schema、配置、评测和 release 由本项目独立维护，不复用 Mega-ASR wrapper、
训练入口或 target 规则。

### D004: 第一轮使用 343-target joint broad LoRA

- 96 audio attention、48 audio MLP、3 speech projection、112 decoder attention、84
  decoder MLP。
- `r=8`，预计 12,365,824 个 LoRA 参数。
- 排除 `lm_head`、embedding、norm 和三个 Conv2d frontend。
- 名称为 `conv_out` 的 Linear 层按实际类型处理，不用名称猜模块类型。

### D005: 学习率固定为 1e-6

全 audio+decoder scope 使用保守全局 `1e-6`，不沿用草案中的 `5e-6`。Mega-ASR 公开
all-scope 命令和论文附录也采用 1e-6 量级。首轮不做 LR sweep。

### D006: 100-step canary 属于正式 run

10+2 smoke 只验证通路与 resume。正式 run step 100 使用固定 512 条 validation 检查生成
和指标崩溃；失败立即停止，通过后继续。Canary 不是过渡模型。

### D007: GPT-5.5 teacher 后置

当前 gold transcript 不调用 teacher。未来接入使用：

```text
TEACHER_API_KEY
TEACHER_BASE_URL
TEACHER_MODEL=gpt-5.5
```

官方 OpenAI Python SDK 当前示例支持 `gpt-5.5`、Responses API、`api_key` 和 `base_url`/
`OPENAI_BASE_URL`。但第三方兼容 endpoint 是否支持音频输入必须实测。Teacher 不得覆盖
gold transcript，密钥不得写入文件或 checkpoint。

### D008: 结果分支固定

- >=10% 且 clean 通过：交付 200k adapter，先测 Mega-ASR gap。
- 5%-10%：只做一次压缩 A2S，不扩全量、不 sweep。
- <5%：停止并排查数据、prompt、labels、Trainer 和 evaluator。
- robust 好但 clean 失败：提高 clean retention 后最多重跑一次，再考虑 router。

### D009: Router 不是第一轮组件

未来 router 只有在 clean 冲突真实存在时才有价值，且标签应基于逐样本 base-vs-LoRA
收益，不是简单 clean/degraded 分类。

### D010: Drive 只保存大 artifact

Google Drive 不承担 200k 小文件逐样本 I/O。训练数据在 `/content` 本地 SSD staging；
Drive 保存大 parquet/tar shard、manifest、checkpoint、prediction 和结果。

### D011: Mega-ASR 只做外部 baseline

发布模型可以在 `references/` 环境中运行并输出 prediction，但其代码不得成为主工程依赖。
只有同一 Bench manifest、normalization 和 evaluator 下的本项目复测才用于“接近 Mega-ASR”
判断。

## 主要风险

### R001: 一次 direct SFT 达不到 Mega-ASR

公开消融显示直接 SFT 约 7% 相对改善，A2S 约 14%-15%，完整系统约 18%-19%。

缓解：第一轮目标是快速验证与交付，不预先承诺完整效果；5%-10% 只触发压缩 A2S。

### R002: 数据下载与 Drive I/O 比训练更慢

公开 robust 数据约 197.5 GB，160k 子集仍可能接近 50 GB 压缩数据。

缓解：按 shard 选择、断点续传、大文件持久化、本地 SSD staging；不复制 200k Drive
小文件。

### R003: Source leakage

同一 clean utterance 可能有多个退化版本，audio hash 不同但内容相同。

缓解：先派生 `source_utterance_id` 再 group split；硬检查 source/name/path/hash/benchmark
id。Transcript overlap 报告并抽查，不单独当作通用硬失败。

### R004: 语言字段缺失

Voices-in-the-Wild-2M 当前没有显式 language 字段。

缓解：使用固定 transcript 规则推断 en/zh；混合或不确定样本 reject；保存分布与抽听记录。

### R005: Broad LoRA 损害 clean 或增加 hallucination

缓解：20% clean train、20% clean validation、1e-6、100-step canary、clean/failure 硬门槛。

回滚：只允许提高 clean retention 重跑一次；仍失败才讨论 router。

### R006: Trainer 看似 resume，实际只重载 adapter

缓解：10+2 集成测试必须验证 optimizer/scheduler/RNG/Trainer state/global step。缺一项不得
启动 200k。

### R007: 依赖或模型结构漂移

缓解：pin Qwen model、官方 repo、qwen-asr 和训练依赖；343 target 按 5 个分组校验；
保存 resolved config 和版本。

### R008: 评测静默美化错误

现有 evaluator 对空 reference 的行为不安全，且缺少双语/32-cell 聚合。

缓解：空 reference 硬失败；English WER、Chinese CER 分开；保存逐样本 edits、原始与归一化
prediction。

### R009: Colab 中断导致长任务丢失

缓解：prediction 增量写入、checkpoint 原子复制、正式 50%/100% state 可恢复、manifest
选样固定。Canary 只用于早停，不替代正式恢复点。

### R010: 历史 adapter 不可复载

当前 checkout 的 v3-v5 adapter 权重缺失。

影响：历史结果保留为审计证据，不能作为 release 或正式 baseline；新主线必须把 adapter
权重、hash 和 release manifest 作为验收硬条件。

## 当前参考

- Qwen3-ASR 官方 finetuning：https://github.com/QwenLM/Qwen3-ASR/tree/main/finetuning
- Voices-in-the-Wild-2M：https://huggingface.co/datasets/zhifeixie/Voices-in-the-Wild-2M
- Voices-in-the-Wild-Bench：https://huggingface.co/datasets/zhifeixie/Voices-in-the-Wild-Bench
- Mega-ASR 论文：https://arxiv.org/abs/2605.19833
- OpenAI Python SDK：https://github.com/openai/openai-python
