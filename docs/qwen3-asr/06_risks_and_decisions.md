# 风险与决策

最后更新：2026-06-19

## 已定决策

### D001: 第一版基础模型使用 Qwen3-ASR-1.7B

决策：

- 从 `Qwen/Qwen3-ASR-1.7B` 开始。

原因：

- 是专用 ASR 模型，官方提供 `qwen-asr` 推理工具。
- 1.7B 体量更适合 Colab Free/Pro 做 baseline 和小规模 LoRA 验证。
- 官方模型卡说明其支持多语言 ASR、离线/流式推理和长音频转写。

影响：

- baseline 推理应优先使用 `Qwen3ASRModel.transcribe`，不再使用通用聊天模板。
- 训练阶段仍需独立探测真实模块名和 LoRA target，不能直接复制参考工程规则。

### D002: 先做独立 Colab MVP

决策：

- 第一目标是由我们自己实现的小型端到端闭环。

原因：

- Mega-ASR 的规模不适合 Colab。
- 最终项目不应成为 Mega-ASR fork。
- 在扩大规模前，需要先证明 Qwen3-ASR LoRA 能改善 degraded ASR。

影响：

- 初始结果只作为可行性信号，不代表最终模型质量。
- MVP 代码从一开始就按独立项目组织。

### D003: 先使用带 scenario 标签的合成退化数据

决策：

- 先生成可控 degraded audio，再收集大规模真实音频。

原因：

- 可复现，且更容易按 scenario 做评估。

影响：

- 后续必须加入真实录音 holdout，避免只适配合成伪影。

### D004: Router 放在第二阶段

决策：

- 先训练 LoRA，再训练 router。

原因：

- 只有确认 LoRA 改善 degraded 样本，并可能损害 clean 样本后，router 才有明确价值。

影响：

- 早期评测先比较 base 与 LoRA always-on。

### D005: Mega-ASR 仅作为参考

决策：

- 不把 Mega-ASR 作为运行时或训练代码库。

原因：

- Mega-ASR 的工程结构、训练入口和 wrapper 与其发布目标深度绑定。
- 即使基础模型同属 Qwen3-ASR，本项目仍需要自己的 manifest、评测、Colab 和训练闭环。
- 我们需要完全掌控数据、训练、推理、测试和部署。

影响：

- 可以观察 Mega-ASR 的行为和结果，但具体实现必须面向 Qwen3-ASR API 编写。

## 风险

### R001: Qwen3-ASR 在强退化音频上仍可能失败

影响：

- 模型可能空输出、漏词、重复、过度归一化，或在严重退化时补全错误内容。

缓解：

- baseline 阶段固定 `language` 参数并保存原始输出。
- 增加输出清理和归一化，但保留原始 prediction 便于分析。
- 在评测中统计空输出、重复输出、长度异常和幻觉式输出。

### R002: Colab 显存可能不足以训练 Qwen3-ASR LoRA

影响：

- 免费或小 GPU runtime 上训练可能失败。

缓解：

- 先做 baseline 和 smoke test。
- 使用 4bit quantization。
- batch size 设为 1，使用梯度累积。
- 限制最大音频长度。
- 正式训练使用 A100 或外部 runtime。

### R003: LoRA target 可能没有覆盖关键音频路径

影响：

- 训练可能只改善语言风格，而没有改善声学鲁棒性。

缓解：

- 检查 Qwen3-ASR 模块名。
- 做 ablation：仅后层 LLM、仅音频投影、联合 target。
- 按 scenario 追踪结果。

### R003A: 官方 qwen-asr wrapper 可能不直接暴露训练所需模块

影响：

- `Qwen3ASRModel` 可能主要面向推理，内部真实 `torch.nn.Module` 根节点不一定稳定暴露。
- 如果无法可靠定位模块，PEFT/LoRA 训练入口需要改为更底层的 Transformers 加载路径。

缓解：

- 在训练前执行 `train/inspect_qwen3_asr_modules.py`，保存 root、module snapshot 和候选 target。
- 不在探测完成前实现正式训练循环。
- 如果 wrapper 不适合训练，保留 qwen-asr 作为推理入口，训练阶段切换到官方模型结构可支持的低层加载方式。

### R004: 合成退化可能无法迁移到真实音频

影响：

- 模型在合成测试集上提升，但真实录音不提升。

缓解：

- 保留真实世界 holdout。
- 后续加入真实 noisy recordings。
- 使用多样噪声和 RIR 素材。

### R005: Clean speech 表现可能退化

影响：

- always-on LoRA 可能损害普通转写。

缓解：

- 训练中混入 clean 数据。
- 使用 router mode。
- 把 clean regression 阈值作为发布门槛。

### R006: 归一化不一致会误导评测

影响：

- WER/CER 可能高估或低估收益。

缓解：

- 使用固定 normalizer。
- 同时保存原始预测和归一化预测。
- 报告 scenario-level 指标。

## 待定决策

- D006: MVP 使用英文、中文，还是双语。
- D007: 首个源数据集。
- D008: 是否发布中间 adapter。
- D009: 使用 Unsloth、Transformers Trainer、TRL，还是自定义训练循环。

## 待定决策处理点

| 决策 | 必须解决的时间点 | 负责文档 | 如果未解决的处理 |
| --- | --- | --- | --- |
| D006: MVP 语言选择 | 开始 `03 数据 MVP` 前 | `03_data_plan.md`、`03_data_mvp.md` | 默认先做英文 MVP |
| D007: 首个源数据集 | 开始 `03 数据 MVP` 前 | `03_data_plan.md`、`03_data_mvp.md` | 默认 LibriSpeech smoke set |
| D008: 是否发布中间 adapter | `05 LoRA 训练 MVP` 验收后 | `08_scale_up_and_release.md` | 默认不发布，只本地/Drive 保存 |
| D009: 训练框架选择 | 开始 `05 LoRA 训练 MVP` 前 | `05_lora_training_mvp.md`、`04_colab_training_plan.md` | 默认 Transformers + PEFT，除非 Colab 资源证明不可行 |
