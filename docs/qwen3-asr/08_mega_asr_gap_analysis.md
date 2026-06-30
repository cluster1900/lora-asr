# Mega-ASR 差距排查

最后更新：2026-06-30

## 背景

本次排查源于一个核心问题：为什么 `xzf-thu/Mega-ASR` 报告的鲁棒 ASR 提升很高，而本项目当前 Qwen3-ASR LoRA 微调只取得小幅收益。2026-06-30 已补充 v5 late audio MLP 结果，确认单纯扩大 audio-side target 仍不足以达到目标。

需要先明确：本项目仍是独立 Qwen3-ASR Robust ASR 项目，Mega-ASR 只作为外部方法参考和 baseline，不作为代码底座。本次排查只分析公开材料和本项目已有实验结果，不复制 Mega-ASR 的私有 wrapper、训练入口或 target 规则。

## 范围

本次做：

- 核对 Mega-ASR 公开 README 和 arXiv 摘要中的方法、数据规模、训练策略和 router 说明。
- 汇总本项目 base recheck、LoRA v1/v2/v3/v4 的固定 MVP 150 held-out 指标。
- 对比两者在数据、训练目标、LoRA target、优化阶段、router 和评测口径上的差异。
- 给出下一轮验证方案和验收标准。

本次不做：

- 不运行 Mega-ASR 上游代码。
- 不把 `references/mega-asr-upstream/` 作为实现来源。
- 不声称本项目已经达到 Mega-ASR 效果。
- 不改训练代码或 notebook。

## 外部事实

公开材料显示，Mega-ASR 的提升不是单一 LoRA 设置带来的。

- Mega-ASR README 称其训练覆盖 7 类原子声学条件、54 类复合声学场景，并使用 2.4M 训练样本，场景包括 noise、far-field、obstruction、echo/reverberation、recording artifacts、electronic distortion 和 transmission dropout。
- arXiv 摘要称 Voices-in-the-Wild-2M 覆盖 7 类经典声学现象和 54 类物理合理复合场景，并使用 A2S-SFT 与 DG-WGPO。
- README 的 finetune 示例使用 `lora_scope all`，公开 target 范围覆盖 audio encoder、aligner 和 text decoder。公开训练说明还提到先按 WER 难度做 encoder/aligner 课程训练，再训练 LLM 语义恢复，最后联合 fine-tune。
- README 明确说明 Mega-ASR 在大量高 WER 数据上训练会轻微损害基础识别能力，因此默认 Transformers 后端使用 router 动态决定是否挂 LoRA；vLLM 后端则物化 LoRA，不做逐样本 router。
- README 也写明 DG-WGPO RL 模块在未来更新中发布，因此当前公开代码不等价于完整论文训练闭环。

参考来源：

- https://github.com/xzf-thu/Mega-ASR
- https://arxiv.org/abs/2605.19833

## 本项目证据

当前公平对照必须使用 `outputs/base_recheck_mvp_150/` 的 4bit base recheck，而不是历史 baseline。历史 base 与 LoRA 评测加载口径不同，已被 base recheck 证实会污染判断。

| model | overall | clean | noise | reverb | noise+reverb | dropout | far_field | degraded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| historical base | 0.483925 | 0.010438 | 0.336117 | 0.415449 | 0.375783 | 0.759916 | 0.897704 | 0.602296 |
| 4bit base recheck | 0.550313 | 0.008351 | 0.450939 | 0.544885 | 0.497912 | 0.762004 | 0.985386 | 0.685804 |
| LoRA v1 | 0.543215 | 0.008351 | 0.419624 | 0.521921 | 0.470772 | 0.770355 | 0.995825 | 0.676931 |
| LoRA v2 | 0.545720 | 0.008351 | 0.438413 | 0.532359 | 0.485386 | 0.764092 | 0.985386 | 0.680063 |
| LoRA v3 | 0.540292 | 0.008351 | 0.413361 | 0.515658 | 0.464509 | 0.768267 | 0.995825 | 0.673278 |
| v4 step 0600 | 0.567850 | 0.008351 | 0.419624 | 0.507307 | 0.463466 | 0.768267 | 1.135699 | 0.707724 |
| v5 step 0320 | 0.544468 | 0.008351 | 0.446764 | 0.515658 | 0.481211 | 0.768267 | 0.983299 | 0.678497 |
| v5 step 0480 | 0.541127 | 0.008351 | 0.419624 | 0.517745 | 0.468685 | 0.766180 | 0.993737 | 0.674322 |

相对 4bit base recheck 的主要结论：

- v3 是当前最优 always-on LoRA：overall 相对改善约 1.82%，noise 改善约 8.33%，reverb 改善约 5.36%，noise+reverb 合并改善约 6.71%。
- v4 step 0600 的 reverb 最好，改善约 6.90%，noise+reverb 合并约 6.92%，但 far_field 明显回退约 15.25%，导致 overall 和 degraded 变差。
- v5 step 0480 是 v5 最优 checkpoint，noise+reverb 合并改善约 5.87%，没有超过 v3；v5 step 0320 对 far_field 略好于 base，但 target 场景收益不足。
- clean 全部持平，没有 clean regression。
- dropout/far_field 基本没有被训练改善，其中 far_field 在长训后会触发重复/幻觉式输出。

训练设置上的关键证据：

- 当前正式训练集只有 360 train + 90 val，且只覆盖 clean/noise/reverb；v3/v4 实际只训练 noise/reverb。
- MVP 150 held-out 测试是 hard profile，noise 平均近似 SNR 约 -2.32 dB，reverb 约 -0.52 dB；LoRA 训练集是 medium profile，train noise 平均近似 SNR 约 +3.66 dB，reverb 约 +2.71 dB。训练退化强度明显弱于测试。
- 当前最有效 LoRA target 主要是 audio tower attention + speech projection，共 99 个 target；真实可训练参数约 1.68M，占当前加载模型参数约 0.14%。v5 加入 24 个后半层 audio MLP 后，可训练参数增至约 2.67M，但收益没有超过 v3。没有训练 text decoder、lm_head，也没有分阶段课程训练。
- 当前训练是普通 answer-token SFT；没有按 base WER 过滤/分桶，没有 A2S 渐进式训练，没有 DG-WGPO 类似的 WER-gated RL，也没有反重复奖励。

## 原因判断

### 1. 数据量和场景覆盖不在同一量级

Mega-ASR 是百万级样本、7 类原子声学条件、54 类复合场景；本项目当前是数百条合成 TTS bootstrap，只训练 noise/reverb。dropout、far_field、obstruction、artifact、distortion、compound 场景没有进入训练，因此这些场景不改善是预期内结果。

### 2. 训练/测试退化强度错配

本项目训练使用 medium profile，测试使用 hard profile。noise 训练平均 SNR 约 +3.66 dB，而测试约 -2.32 dB；reverb 训练约 +2.71 dB，测试约 -0.52 dB。模型在较温和退化上学到的修正，很难迁移到更重的 held-out 退化。

### 3. 当前 LoRA 容量主要限制在声学侧小范围

当前可训练参数约 0.14%，主要改 audio attention 和少量 projection。它能改善部分局部声学替换，但对严重退化下的语义重建、重复输出和 off-audio 幻觉控制不足。Mega-ASR 公开 `lora_scope all` 覆盖 audio encoder、aligner 和 text decoder，并使用分模块学习率。

### 4. 缺少 A2S 课程训练和 WER-gated 优化

Mega-ASR 的公开说明强调按 WER 难度递进训练：先 encoder/aligner，再 LLM，再联合；DG-WGPO 又对低 WER 和高 WER 样本使用不同粒度奖励，并加入反重复门控。本项目目前只有普通 SFT，训练目标不能直接惩罚 WER、漏词、重复和幻觉。

### 5. router 不是锦上添花，而是 Mega-ASR 系统组成

Mega-ASR README 明确说大量高 WER 数据训练会带来基础识别能力轻微退化，所以用 router 决定是否激活 LoRA。我们的 always-on LoRA 对 clean 没退化，但 target 场景还没过 10% 门槛，router 暂停是合理的。后续如果扩大 target 到 LLM 或更难数据，clean regression 风险会上升，router 会变成必须项。

### 6. 历史 base 口径曾放大了“提升低/训练失败”的误判

按历史 base 看，v1/v2 像是明显变差；按 4bit base recheck 看，v1/v2 对 noise/reverb 有弱收益。当前提升低是真问题，但不是“完全没学到”，而是“目标场景只学到小幅局部修正，且没有覆盖未训练场景”。

## 测试与验收

已完成的验证：

- 固定 MVP 150 held-out test 上完成 historical base、4bit base recheck、LoRA v1/v2/v3/v4 指标对齐。
- 检查 train/val/test 场景分布和退化统计，确认训练 profile 与测试 profile 存在强度错配。
- 检查 LoRA target 和 trainable 参数，确认当前训练容量远小于 Mega-ASR 公开 `lora_scope all` 路径。
- 抽查逐样本 delta，确认 v3/v4 的收益集中在少量 noise/reverb 样本，far_field 仍存在重复和幻觉式输出。

下一轮验收标准：

- 新训练集必须同时包含 train/val/test 的退化统计，至少给出各场景平均 SNR、RMS ratio、near-silence ratio。
- 新实验必须同时报告 overall、clean、noise、reverb、dropout、far_field、noise+reverb、degraded WER。
- 每轮必须报告 relative improvement vs 4bit base recheck，不再用 historical base 作为主对照。
- 若扩大 target 到 audio MLP 或 text decoder，必须额外报告重复输出率、过长输出率和 clean regression。
- 只有目标场景相对改善达到 10%，且 clean 无退化、far_field/dropout 没有明显恶化，才恢复 router MVP。

## 下一步

优先级从高到低：

1. 做数据 profile 对齐：新增 hard noise/reverb 训练集，先不要再只用 medium profile 打 hard test。
2. 为训练样本补 base prediction、base WER、difficulty bucket 和 failure tags，训练采样从 scenario-only 升级为 scenario + WER bucket。
3. 增加少量 dropout/far_field hard negative 或 mixed constraint，防止 v4 式 far_field 幻觉回退。
4. 设计一个独立 A2S-style 小闭环：按 base WER 分桶，先训练 audio encoder/aligner，再测试是否需要训练 text decoder。
5. 在 LoRA 达到 target 场景门槛后再恢复 router，router 必须以 clean/degraded 分类指标和最终 WER 双重验收。

## 影响

- 当前结论不支持声称本项目接近 Mega-ASR；只能说 v3/v4 已验证 noise/reverb 有弱收益。
- 后续优化重点应从“单纯延长 step”转向“数据难度对齐 + target 容量扩展 + 课程训练”。
- 评测报告必须继续标注 historical base 与 4bit base recheck 的差异，避免把环境变化误读为训练收益。
