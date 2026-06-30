# 风险与决策

最后更新：2026-06-30

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

### D009: 训练框架回退到 Transformers + PEFT

决策：

- 第一版 LoRA 训练尝试过 Unsloth，但兼容性检查失败。
- 当前训练 backend 回退到 Transformers + PEFT。

原因：

- Colab Free 显存有限，因此曾优先尝试 Unsloth。
- 官方 Unsloth/Qwen 文档覆盖普通 Qwen3/Qwen3 MoE 高效微调。
- Qwen3-ASR 是音频 ASR 模型，`model_type=qwen3_asr`，当前 Unsloth `FastModel.from_pretrained` 走标准 Transformers AutoConfig 路径时无法识别该架构。

影响：

- `configs/train/qwen3_asr_lora_mvp.yaml` 使用 `backend: transformers_peft`。
- `05C` 进入 Transformers + PEFT smoke training。
- qwen-asr 仍作为 baseline 和评测推理入口保留。

### D010: MVP 150 固定为 held-out test，不作为正式 LoRA 训练集

决策：

- `baseline_mvp_150` 继续作为固定 held-out test。
- 正式 LoRA MVP 从独立 bootstrap train/val manifest 开始训练。

原因：

- 20 step smoke training 曾使用 MVP 150 样本验证训练链路，但这不应用于判断模型效果。
- 如果正式训练继续使用同一批测试样本，base-vs-LoRA 对比会高估收益。
- 第一版 LoRA MVP 的核心问题是“能否在未见 held-out 场景上改善 noise/reverb，同时控制 clean regression”。

影响：

- 新增 `scripts/create_lora_mvp_dataset.py` 生成 train/val。
- 新增 `configs/train/qwen3_asr_lora_mvp_train.yaml` 作为正式 MVP 训练配置。
- 评测使用固定 MVP 150 held-out test；当前 LoRA 对比应优先使用
  `outputs/base_recheck_mvp_150/metrics.qwen3_asr_base_recheck.mvp_150.json`
  作为 4bit base 对照。历史 `outputs/baseline_mvp_150/` 只作为旧环境参考。

### D011: v1 LoRA 不进入 router，先做 v2 ablation

决策：

- 按历史 base 口径，v1 LoRA always-on 没有通过 MVP 验收，不进入 router。
- 下一步新增 v2 ablation：attention-only、noise/reverb-only、更低学习率、更少 step。
- 该判断已被 D013 修正：历史 base 口径与当前 4bit LoRA 评测口径不一致，v1
  需要以 base recheck 重算。

原因：

- v1 clean WER 小幅改善，但目标场景 noise 和 reverb 分别相对 base 变差 24.84% 和 25.63%。
- degraded-only WER 从 0.602296 变为 0.676931，相对变差 12.39%。
- router 的价值建立在 LoRA 对 degraded 样本有收益的前提上；当前 always-LoRA 不优于 base。

影响：

- `05E` 先做训练策略和 target ablation。
- 暂不实现 router 阈值、router 训练或 router 推理。
- v1 checkpoint 和输出保留为失败对照，不覆盖。

### D012: v2 attention-only 短跑仍不进入 router

决策：

- 按历史 base 口径，v2 LoRA attention-only 短跑不通过 MVP 验收，不进入 router。
- 下一轮继续在 LoRA always-on 层面做快速 ablation。
- 该判断已被 D013 修正：以 4bit base recheck 为对照时，v2 在 noise/reverb
  有弱收益，但仍未达到 MVP 改善门槛。

原因：

- v2 采用更保守设置：96 个 audio tower attention target、noise/reverb-only 训练、学习率 1e-5、150 step、`scenario + text_length_bucket` 均衡采样。
- v2 相比 v1 在 dropout 和 far_field 上略有改善，clean 持平，但目标场景 noise/reverb 没有改善。
- fixed MVP 150 上，noise WER 为 0.438413，reverb WER 为 0.532359，均差于 base 和 v1。
- 这说明 v1 的主要问题不能只归因于 speech projection、clean 样本混入、训练步数过多或短跑长度覆盖不足。

影响：

- router 继续暂停，因为 LoRA 对 degraded 目标场景尚无明确收益。
- 后续优先验证更早停止点、更小学习率、后层 audio attention target、训练目标格式和数据难度。
- v2 输出保留为 ablation 对照，不覆盖 v1 或 base。

### D013: 当前 LoRA 对比改用 4bit base recheck 口径

决策：

- `outputs/base_recheck_mvp_150/` 中的 4bit base recheck 作为当前 LoRA v1/v2
  的主对照。
- 历史 `outputs/baseline_mvp_150/` base 指标保留为旧环境参考，不再作为当前
  LoRA 是否改善的主判断依据。
- LoRA v1/v2 仍不进入 router，因为收益幅度不足且 dropout/far_field 未改善。

原因：

- LoRA v1/v2 推理使用 4bit base 后挂载 adapter；历史 base 来自早期 baseline
  notebook，输出中没有记录量化方式。
- 4bit base recheck 使用同一 MVP 150 manifest、同一音频、`dtype=float16`、
  `device_map=cuda:0`、`quantization=4bit`。
- base recheck overall WER 为 0.550313，明显差于历史 base 0.483925。
- 差异主要集中在 noise、reverb 和 far_field，说明历史 base 与当前 LoRA
  评测口径不一致。
- 以 base recheck 为准，LoRA v1 对 noise+reverb 有约 5.45% 相对改善，LoRA
  v2 有约 2.52% 相对改善，均未达到第一版 10% 改善门槛。

影响：

- 后续实验表格必须同时标注 `historical_base` 和 `base_recheck`，避免混淆。
- MVP 通过标准改为相对 4bit base recheck 计算。
- 当前结论从“LoRA 完全失败”修正为“目标场景有弱收益，但不足以进入 router”。

### D014: v3 使用 v1 target 组合并聚焦 noise/reverb

决策：

- v3 使用 v1 的 99 个 target：audio tower attention + speech projection。
- v3 不训练 clean 样本，只训练 noise/reverb。
- v3 使用 `scenario + text_length_bucket` 均衡采样，默认 `max_steps=450`、
  `learning_rate=2e-5`。

原因：

- 以 4bit base recheck 为对照，v1 对 noise+reverb 的相对改善约 5.45%，强于 v2 的约 2.52%。
- v2 移除 speech projection 后收益变弱，说明 speech projection 可能对当前合成退化任务有帮助。
- v1 混入 clean 后 clean 没退化，但目标收益不足；v3 先聚焦 noise/reverb，尝试放大目标收益。
- 长短样本必须均衡，避免 v1 manifest 顺序下前段主要覆盖 short 的问题影响短跑判断。

影响：

- 新增 `configs/train/qwen3_asr_lora_mvp_v3_target_focus.yaml`。
- 新增 `notebooks/07_train_lora_mvp_v3_colab.ipynb`。
- router 继续暂停；只有 noise 或 reverb 相对 4bit base recheck 接近或达到 10% 改善，才进入 router 前检查。

结果：

- v3 已完成 450 step 训练和 MVP 150 held-out 评测。
- 相对 4bit base recheck，overall WER 从 0.550313 降到 0.540292，相对改善约 1.82%。
- noise WER 从 0.450939 降到 0.413361，相对改善约 8.33%，是当前最接近 10% 门槛的单场景结果。
- reverb WER 从 0.544885 降到 0.515658，相对改善约 5.36%。
- noise+reverb 合并 WER 从 0.497912 降到 0.464509，相对改善约 6.71%。
- clean WER 持平为 0.008351，无 clean regression；dropout/far_field 仍轻微退化。

后续决策：

- v3 证明 target-focus 方向有效，但未满足 10% 相对 WER 改善门槛。
- 当前不进入 router，继续做 v3b/v4 ablation。
- 下一轮优先保留 99 target 和长短均衡采样，再测试更合适的训练步数、学习率或中间 checkpoint 选择。

### D015: v4 先做 checkpoint sweep，而不是直接换 target 或进入 router

决策：

- v4 保留 v3 的 99 个 target、noise/reverb-only 训练、`scenario + text_length_bucket` 均衡采样和 `learning_rate=2e-5`。
- v4 将 `max_steps` 扩到 600，并保存 160/320/480 step 中间 adapter，加上 final 600 step 一起评测。
- `save_steps=160` 用于对齐 `gradient_accumulation_steps=16` 的完整 optimizer update。

原因：

- v3 已经是当前最优方向，noise 相对 4bit base recheck 改善约 8.33%，距离 10% 门槛很近。
- 直接换 target 或扩大到 dropout/far_field 会引入多个变量，难以判断收益来源。
- 只看 final checkpoint 可能错过最佳停止点，也可能把过训误判为训练方向无效。

影响：

- 新增 `configs/train/qwen3_asr_lora_mvp_v4_checkpoint_sweep.yaml`。
- 新增 `notebooks/08_train_lora_mvp_v4_checkpoint_sweep_colab.ipynb`。
- 训练脚本开始实际支持 `output.save_steps` 和 `output.keep_last_checkpoints`，会写出中间 adapter 与 `saved_checkpoints` 元数据。
- router 继续暂停，直到 checkpoint sweep 中至少一个候选满足 target 场景 10% 改善和 clean 无明显退化。

结果：

- v4 checkpoint sweep 已完成，160/320/480/final 600 step 均完成 held-out MVP 150 评测。
- 没有 checkpoint 达到 10% 相对 WER 改善门槛。
- v4 600 的 reverb WER 为 0.507307，相对 4bit base recheck 改善约 6.90%，是当前 reverb 最好结果。
- v4 600 的 noise+reverb 合并 WER 为 0.463466，略好于 v3 的 0.464509。
- v4 600 的 far_field WER 为 1.135699，明显差于 base recheck 的 0.985386 和 v3 的 0.995825。
- v3 仍是 overall、noise 和 degraded-only 的当前最优 LoRA。

后续决策：

- 单纯增加 target-focus step 暂停。
- 当前不进入 router。
- 下一轮应优先测试数据/目标/正则约束或 target 结构，而不是继续增加 step。

### D016: v5 优先测试 late audio MLP，不训练 text decoder

决策：

- v5 在当前 99 个 audio attention + speech projection target 基础上，只加入 `audio_tower.layers.12-23.fc1/fc2`。
- v5 不训练 text decoder attention、text decoder MLP、`lm_head` 或 speech conv。
- v5 使用较低学习率和 checkpoint sweep，默认 `learning_rate=1.5e-5`、`max_steps=480`、`save_steps=160`。

原因：

- v3 证明 audio attention + speech projection 有效，但收益不足 10%。
- v4 证明单纯增加 step 会牺牲 far_field，说明需要改变 capacity 或约束，而不是继续拉长训练。
- audio tower MLP 负责音频表征的非线性变换，可能比继续训练 attention 更适合学习噪声/混响补偿。
- 只选后半层 audio MLP 是为了降低破坏低层声学特征的风险。
- text decoder 和 `lm_head` 更可能放大语言补全和幻觉式输出，小数据阶段暂不训练。

影响：

- 新增 `configs/train/qwen3_asr_lora_mvp_v5_late_audio_mlp.yaml`。
- 新增 `notebooks/09_train_lora_mvp_v5_late_audio_mlp_colab.ipynb`。
- router notebook 顺延为 `notebooks/10_router_colab.ipynb`。
- v5 验收必须同时看 target 场景收益和 far_field regression；若 far_field 明显回退，不进入 router。

结果：

- v5 checkpoint sweep 已完成，160/320/final 480 step 均完成 held-out MVP 150 评测。
- v5 step 0480 是 v5 最优，overall WER 0.541127，noise WER 0.419624，reverb WER 0.517745，noise+reverb WER 0.468685。
- 相对 4bit base recheck，v5_0480 的 noise+reverb 改善约 5.87%，弱于 v3 的 6.71%。
- v5 没有像 v4_0600 那样造成 far_field 大幅崩溃，但 dropout/far_field 仍没有实质改善。

后续决策：

- v5 证明 late audio MLP 扩容不是当前主解。
- 暂停继续扩大 audio-side target。
- 下一轮转向 hard-profile 数据对齐和 WER difficulty bucket。

### D017: Mega-ASR 差距排查后的优化重心

决策：

- 后续不再把“继续延长当前 99 target noise/reverb-only 训练”作为主路径。
- 下一轮优先验证 hard-profile 数据对齐、dropout/far_field 约束和 A2S-style 小闭环。
- router 继续暂停，直到 LoRA 在目标场景达到 10% 相对改善且观察场景没有明显恶化。

原因：

- Mega-ASR 公开材料显示，其提升来自 2.4M 样本、7 类原子声学条件、54 类复合场景、A2S-SFT、DG-WGPO 和 router 的系统组合。
- 本项目当前 train 只有 360 条，且训练 noise/reverb 为 medium profile，而 held-out MVP 150 为 hard profile，存在明显退化强度错配。
- 当前最强小闭环 v3 只训练约 1.68M 参数，约占当前加载模型参数 0.14%，主要覆盖 audio attention + speech projection；这能带来局部 noise/reverb 改善，但不足以解决严重 far_field/dropout 下的重复和幻觉式输出。
- v4 checkpoint sweep 已证明单纯延长 step 不能稳定突破 10%，并会引发 far_field 明显回退。
- v5 已证明增加后半层 audio MLP 不能超过 v3。

影响：

- 新实验的首要验收不只看 noise/reverb，也必须看 far_field/dropout 和重复/过长输出风险。
- 文档和实验表格必须继续使用 4bit base recheck 作为主对照。
- 若后续训练 text decoder 或扩大到 `all` 类 target，必须同步提高 clean regression 和 hallucination 风险监控。

### D018: v6 起采用阶段化训练微调方案

决策：

- 新增 `docs/qwen3-asr/09_training_finetune_strategy.md` 作为 v6 大阶段的主训练方案。
- v6A 回到 v3 的 99 target，优先验证 hard-profile data alignment 和 base WER difficulty bucket。
- v6B 再加入 dropout/far_field mixed constraint。
- v6C 做 A2S-style encoder/aligner curriculum。
- v6D 才做 text decoder pilot，且默认单独 adapter、低学习率、可回滚。
- Notebook 文件编号继续递增，不用 notebook 编号表示实验版本。

原因：

- v5 的 123 target 和约 2.67M 可训练参数没有超过 v3，说明 target 容量不是当前第一瓶颈。
- 当前训练/测试 profile 错配明显，且训练数据没有 base WER 分桶，无法模拟 Mega-ASR 的 A2S-SFT 难度递进。
- dropout/far_field 长期只作为观察集，导致目标场景提升时容易出现未训练场景回退。

影响：

- 后续 notebook 和配置应按数据版本和 v6 子阶段命名，不再只按 target 命名。
- 每轮训练必须保存 difficulty manifest、scenario/difficulty 分布、失败标签和统一 comparison 表。
- 只有 v6A/v6B 达到 10% target 场景改善并控制 clean/far_field/dropout 后，才继续 text decoder 和 router。

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

### R003D: 当前微调与 Mega-ASR 系统工程存在量级差距

影响：

- 使用数百条 TTS 合成 bootstrap 和小范围 audio-side LoRA，很难复现 Mega-ASR 公开报告的接近 30% 鲁棒 ASR 提升。
- 如果忽略数据规模、场景覆盖、A2S-SFT、DG-WGPO 和 router 的差异，可能会把合理的小幅收益误判为训练代码失败。

缓解：

- 把 Mega-ASR 作为目标能力形态而不是短期数值 baseline。
- 每轮实验明确标注训练数据量、退化 profile、target 范围、可训练参数量和 base 对照口径。
- 先追求固定 MVP 150 上 target 场景 10% 相对改善，再逐步扩展到复合场景和 router。

### R003E: 训练/测试退化强度错配会压低收益

影响：

- 当前训练 noise/reverb 是 medium profile，测试是 hard profile；模型可能只学到温和退化修正，无法迁移到更重退化。
- 继续调 learning rate 或 step 可能收益有限，并可能加重 far_field/dropout 回退。

缓解：

- 新增 hard-profile train/val，并记录各场景平均 SNR、RMS ratio 和 near-silence ratio。
- 每轮评测同时保留 clean、target degraded 和观察 degraded 场景。
- 对 hard negative 加入少量 dropout/far_field 约束，避免目标场景提升换来未训练场景幻觉增加。

### R003C: LoRA 可能改善 clean 但损害 degraded

影响：

- v1 已出现该现象：clean WER 小幅改善，但 noise/reverb/degraded-only 变差。
- v2 继续验证了该风险：clean 未退化，dropout/far_field 略稳，但目标 noise/reverb 仍相对 base 变差。
- base recheck 后，该风险需要重新表述：相对 4bit base，v1/v2 对 noise/reverb
  有弱收益，但观察场景 dropout/far_field 没有改善，仍不能只看 target 场景收益。
- 如果只看 loss 或 clean 指标，可能误判训练成功。

缓解：

- LoRA MVP 验收必须以 fixed held-out MVP 150 的 scenario-level WER 为准。
- 在 LoRA 没有赢 base 之前，不进入 router。
- 做 target/data/step/lr ablation，并记录每一轮 base-vs-LoRA 对比。

### R003A: 官方 qwen-asr wrapper 可能不直接暴露训练所需模块

影响：

- `Qwen3ASRModel` 可能主要面向推理，内部真实 `torch.nn.Module` 根节点不一定稳定暴露。
- 如果无法可靠定位模块，PEFT/LoRA 训练入口需要改为更底层的 Transformers 加载路径。

缓解：

- 在训练前执行 `train/inspect_qwen3_asr_modules.py`，保存 root、module snapshot 和候选 target。
- 不在探测完成前实现正式训练循环。
- 如果 wrapper 不适合训练，保留 qwen-asr 作为推理入口，训练阶段切换到官方模型结构可支持的低层加载方式。

### R003B: Unsloth 可能不支持 Qwen3-ASR 的音频 ASR 架构

影响：

- Unsloth 可能能加载普通 Qwen3 LLM，但不能加载 `Qwen3ASRForConditionalGeneration`。
- 即使能加载，也可能只能按 `q_proj`、`v_proj` 这类叶子名挂 LoRA，导致同时命中文本 decoder，无法只训练 audio tower。

缓解：

- 新增 `train/check_unsloth_qwen3_asr.py` 做兼容性检查。
- 兼容性检查已保存 JSON 输出，用于记录失败原因。
- 当前已回退到 Transformers + PEFT 的正则 target 方案。

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

### R007: Bootstrap 合成训练集过拟合合成 TTS 和规则退化

影响：

- LoRA 可能学习到 macOS `say` 的声音特征或 `medium` profile 的固定伪影，而不是真实鲁棒 ASR 能力。

缓解：

- 第一版只把 bootstrap 数据作为训练闭环启动集。
- 固定 MVP 150 hard profile 作为 held-out test，并保留 dropout/far_field 观察项。
- 达到第一版指标后，再引入 LibriSpeech/Common Voice 和真实噪声/RIR 数据。

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
- D009: 使用 Unsloth、Transformers Trainer、TRL，还是自定义训练循环。当前决策：Unsloth 兼容失败，回退 Transformers + PEFT。

## 待定决策处理点

| 决策 | 必须解决的时间点 | 负责文档 | 如果未解决的处理 |
| --- | --- | --- | --- |
| D006: MVP 语言选择 | 开始 `03 数据 MVP` 前 | `03_data_plan.md`、`03_data_mvp.md` | 默认先做英文 MVP |
| D007: 首个源数据集 | 开始 `03 数据 MVP` 前 | `03_data_plan.md`、`03_data_mvp.md` | 默认 LibriSpeech smoke set |
| D008: 是否发布中间 adapter | `05 LoRA 训练 MVP` 验收后 | `08_scale_up_and_release.md` | 默认不发布，只本地/Drive 保存 |
| D009: 训练框架选择 | 开始 `05 LoRA 训练 MVP` 前 | `05_lora_training_mvp.md`、`04_colab_training_plan.md` | Unsloth 已验证不兼容，回退 Transformers + PEFT |
