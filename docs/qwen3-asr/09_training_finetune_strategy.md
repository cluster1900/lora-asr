# 训练微调设计方案

最后更新：2026-06-30

## 背景

v5 late audio MLP 训练已经完成。结果显示：在 v3/v4 的 audio attention + speech projection 基础上加入后半层 audio MLP，虽然把可训练参数从约 1.68M 提高到约 2.67M，但没有稳定超过 v3，也没有达到 LoRA MVP 10% 相对改善门槛。

这说明当前瓶颈不是“再多训练一点 target”可以解决的。要向 Mega-ASR 的能力形态靠近，需要把训练微调重新设计成系统工程：数据构建、base WER 分桶、A2S-style 阶段训练、target 范围、反幻觉约束、router 和评测闭环必须一起设计。

Mega-ASR 公开信息显示，其高收益来自 2.4M 训练样本、7 类原子声学条件、54 类复合声学场景、A2S-SFT、DG-WGPO 和 router 的组合。本项目仍保持独立实现，Mega-ASR 只作为方法参考和外部 baseline。

参考：

- https://github.com/xzf-thu/Mega-ASR
- https://arxiv.org/abs/2605.19833

## v5 结果结论

当前公平 base 仍为 `outputs/base_recheck_mvp_150/` 的 4bit base recheck。

| model | overall | clean | noise | reverb | noise+reverb | dropout | far_field | degraded |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4bit base recheck | 0.550313 | 0.008351 | 0.450939 | 0.544885 | 0.497912 | 0.762004 | 0.985386 | 0.685804 |
| v3 target-focus | 0.540292 | 0.008351 | 0.413361 | 0.515658 | 0.464509 | 0.768267 | 0.995825 | 0.673278 |
| v4 step 0600 | 0.567850 | 0.008351 | 0.419624 | 0.507307 | 0.463466 | 0.768267 | 1.135699 | 0.707724 |
| v5 step 0160 | 0.552401 | 0.008351 | 0.442589 | 0.553236 | 0.497912 | 0.768267 | 0.989562 | 0.688413 |
| v5 step 0320 | 0.544468 | 0.008351 | 0.446764 | 0.515658 | 0.481211 | 0.768267 | 0.983299 | 0.678497 |
| v5 step 0480 | 0.541127 | 0.008351 | 0.419624 | 0.517745 | 0.468685 | 0.766180 | 0.993737 | 0.674322 |

主要判断：

- v5 最好 checkpoint 是 step 0480：overall 相对 base 改善约 1.67%，noise+reverb 改善约 5.87%。
- v5 没有超过 v3：v3 noise+reverb 改善约 6.71%，且 overall/degraded 略优。
- v5 没有像 v4 step 0600 那样造成 far_field 大幅崩溃，但 far_field/dropout 仍没有实质改善。
- v5 的更多 audio-side 容量没有解决 hard profile 泛化、dropout/far_field 幻觉和高 WER 语义恢复。

结论：停止以“继续扩大当前 audio-side target 或延长 step”为主线，转入数据与训练目标重构。

## 总体目标

阶段性目标不是一次性复刻 Mega-ASR，而是沿着类似系统形态逐步逼近：

1. 在固定 MVP 150 hard profile 上，让 noise 或 reverb 单场景相对 4bit base 改善超过 10%，clean 无退化，far_field/dropout 不明显恶化。
2. 在扩展 MVP 1k-2k 上，让 degraded-only 相对 base 改善 10%-15%，并覆盖 clean、noise、reverb、far_field、dropout、noise_reverb、far_field_noise。
3. 在真实/半真实 holdout 上，让 degraded-only 相对 base 改善 15%-25%，router 模式优于 always-base 和 always-LoRA。
4. 只有达到多数据源、多场景、可复现指标后，才允许与 Mega-ASR 发布模型做正式外部 baseline 对比。

## 设计原则

- 数据优先：每次 target 或训练方法变化前，先确认训练/验证/测试的退化强度和场景覆盖是否匹配。
- 分阶段训练：先让音频 encoder/aligner 适应声学退化，再考虑 text decoder 语义恢复，最后做联合微调。
- WER 驱动：所有训练样本要有 base prediction、base WER、错误标签和难度桶，不再只按 scenario 采样。
- 保留 clean 能力：每个阶段都需要 clean retention 样本和 clean regression 门槛。
- 反幻觉优先：far_field/dropout 的重复、过长、off-audio 输出是硬风险，不允许用 target 场景收益掩盖。
- router 后置：LoRA always-on 没有明确收益前不做 router；一旦扩大到 text decoder 或 hard high-WER 数据，router 必须进入验收。

## 数据体系

### 数据分层

Tier 0：当前 bootstrap

- 规模：360 train + 90 val。
- 场景：clean/noise/reverb。
- 用途：保留为历史小闭环，不再作为主要优化数据。

Tier 1：Hard-profile MVP train/val

- 规模：建议 2k-5k train，300-500 val。
- 场景：clean、noise、reverb、dropout、far_field、noise_reverb、far_field_noise。
- 目标：解决当前 medium train vs hard test 的强度错配。
- 要求：每条样本记录增强参数、近似 SNR、RMS ratio、near-silence ratio、text_length_bucket。

Tier 2：多源真实语音 + 合成退化

- 规模：建议 20k-50k train，2k-5k val/test。
- clean 来源：LibriSpeech、Common Voice；中文扩展再引入 AISHELL/WenetSpeech。
- 噪声/RIR 来源：MUSAN、DNS Challenge、ESC-50、UrbanSound8K、公开 RIR。
- 目标：降低 macOS TTS 和固定合成伪影过拟合。

Tier 3：Voices-in-the-Wild-style 复合场景

- 规模：建议 100k+。
- 场景：从 7 类原子条件扩展到 20-50 个复合场景。
- 目标：向 Mega-ASR 的 full-scenario robust ASR 形态靠近。

### 难度标注

每个训练样本在进入训练前都要跑 base inference 并生成 difficulty manifest：

```json
{
  "audio": "data/robust_v1/audio/train/noise/000001.wav",
  "answer": "reference text",
  "base_prediction": "base transcript",
  "base_wer": 0.42,
  "scenario": "noise",
  "difficulty_bucket": "wer_30_50",
  "failure_tags": ["substitution_heavy"],
  "text_length_bucket": "long",
  "approx_snr_db": -2.1,
  "seed": 20260710
}
```

推荐难度桶：

- `wer_0_10`：clean retention 和轻微退化。
- `wer_10_30`：低难度声学修正。
- `wer_30_50`：主要训练桶，适合 encoder/aligner。
- `wer_50_70`：高难度，适合后续 A2S-style 语义恢复。
- `wer_70_plus`：默认只做观察或小比例 hard negative，避免小模型早期学到幻觉补全。

## 训练阶段

### Phase A: 数据对齐 SFT

目的：

- 先证明 hard-profile 数据对齐能超过 v3，而不是继续堆 target。

数据：

- Tier 1 hard-profile train/val。
- 采样比例建议：clean 15%，noise 25%，reverb 25%，noise_reverb 15%，far_field 10%，dropout 10%。
- `wer_30_50` 和 `wer_10_30` 为主，`wer_50_70` 少量加入，`wer_70_plus` 暂不训练。

target：

- audio tower attention + speech projection，即 v3 的 99 target。
- 不加 text decoder。

训练：

- 4bit PEFT LoRA。
- `r=8`，`alpha=16`，`dropout=0.05`。
- `learning_rate=1e-5` 到 `2e-5`。
- 每 160 或 320 step 保存 checkpoint。
- 使用 validation WER 选 checkpoint，不只看训练 loss。

通过标准：

- 在 MVP 150 上 noise 或 reverb 至少一个场景相对 base 改善超过 10%。
- clean WER 不高于 base。
- far_field/dropout 相对 base 恶化不超过 2%。

### Phase B: Encoder/Aligner 扩容

目的：

- 在数据对齐有效后，再扩大 audio-side 容量。

target：

- audio tower attention + audio tower MLP + speech projection。
- 优先先做后 12 层 MLP，再评估全层 MLP。

数据：

- 继续 Tier 1/Tier 2。
- 加入更多 `wer_30_50` 和少量 `wer_50_70`。

训练：

- 分模块学习率：encoder MLP 和 attention 可低一些，speech projection 可略高。
- 默认不训练 text decoder。

通过标准：

- 相比 Phase A，noise+reverb 和 degraded-only 至少再提升 2%-3% 相对 WER。
- far_field repeat_like_rate 不升高。

### Phase C: LLM 轻量语义恢复

目的：

- 只在 audio-side 已经稳定后，测试 text decoder 是否能减少高 WER 样本的漏词和语义错配。

target：

- 只选 text decoder 后 4-8 层 attention/MLP，禁止 `lm_head`。
- 或单独训练 `llm` adapter，与 audio adapter 分开保存，便于回滚。

数据：

- 只使用 `wer_30_70` 的样本。
- 必须混入 clean retention 和 anti-hallucination negative。
- 高 WER 样本不能超过 batch 的 30%。

训练：

- `learning_rate=3e-6` 到 `8e-6`。
- checkpoint 更密集。
- 每个 checkpoint 必须评估 repeat_like、too_long、hallucination_like。

通过标准：

- reverb/far_field 高 WER 样本有改善。
- clean 不退化。
- repeat_like_rate、too_long_rate 不高于 audio-only 最优 checkpoint。

回滚条件：

- clean WER 相对 base 退化超过 3%。
- far_field repeat_like_rate 或 too_long_rate 明显上升。
- dropout hallucination_like_rate 上升超过 5 个百分点。

### Phase D: Joint SFT

目的：

- 在 encoder/aligner 和 LLM 子实验都有效后，做低学习率联合微调。

target：

- encoder + aligner + selected LLM。
- 保持 adapter 可拆分或至少保留上一步 checkpoint。

训练：

- 小学习率，短 checkpoint sweep。
- 使用 validation WER 和失败标签选择 checkpoint。

通过标准：

- 扩展 MVP 上 degraded-only 相对 base 改善 10%-15%。
- router 候选集上 always-LoRA 对 degraded 明显优于 base，但 clean 可能轻微退化。

### Phase E: WER-gated Preference/RL 小闭环

目的：

- 独立实现一个可控的 WER-gated 优化，学习 Mega-ASR 的思想，但不依赖其 DG-WGPO 代码。

第一版不直接做复杂 RL，先做 preference data：

- 对每条样本保存 base、SFT adapter、候选 checkpoint 的 prediction。
- 用 WER、重复率、长度比构造 chosen/rejected。
- 低 WER 样本偏向 token-level 精准；高 WER 样本偏向完整性和反幻觉。

可选训练方式：

- DPO/ORPO 类偏好优化，如果 qwen-asr 训练路径可支持。
- 或先做 sample-weighted SFT，把高价值修正样本加权。

通过标准：

- repeat_like、too_long、hallucination_like 降低。
- WER 不因反幻觉约束而整体恶化。

### Phase F: Router

目的：

- 解决 robust adapter 和 clean/base 能力之间的冲突。

输入特征：

- base confidence proxy，如输出长度比、重复特征、ASR logprob 如果可取。
- 音频质量特征，如 SNR proxy、RMS、静音比例、谱平坦度、reverb proxy。
- 场景分类器或轻量 degraded detector。

输出：

- `use_base` / `use_lora` / `uncertain_fallback`。

通过标准：

- clean/degraded accuracy、precision、recall 全部报告。
- router 模式在混合测试集上优于 always-base 和 always-LoRA。
- clean WER 相对 base 退化小于 3%-5%。

## v6 执行计划

后续所有训练计划统一定义为 v6 大阶段。v6 内部用子阶段区分训练目标，
notebook 文件名继续递增编号，避免实验版本号和 notebook 序号互相干扰。

### v6A: hard-profile data alignment

目标：

- 验证“数据难度对齐”是否比 v5 target 扩容更有效。

范围：

- 新增 hard-profile train/val 数据。
- Notebook 10 先从已提交的 `lora_mvp` clean 音频派生 7 类 hard-profile 场景，默认 1680 train / 420 val。
- Notebook 11 再对 v6A train/val 跑 4bit base inference，生成 `base_wer` 和 `difficulty_bucket`。
- 训练 target 回到 v3 的 99 target。
- 不训练 audio MLP，不训练 text decoder。

最小执行步骤：

1. `10_make_hard_profile_dataset_colab.ipynb` 生成 v6A hard-profile manifest 和 stats。
2. `11_score_base_difficulty_colab.ipynb` 给 v6A manifest 打 base WER 分桶，过滤明显过难或过易样本。
3. `12_train_lora_v6a_hard_profile_colab.ipynb` 使用 v3 99 target 训练，并在固定 MVP 150 上评测。

调整规则：

- 如果 Notebook 10 stats 显示 clipping 或 active silence 过高，先把 `variants_per_utterance` 保持 1-2，并降低 hard profile 强度，不直接训练。
- 如果 Notebook 11 中 `wer_70_plus` 占比过高，v6A 训练先排除或小比例采样该 bucket。
- 如果 v6A 没有超过 v3，不继续扩大 target，优先扩展真实 clean/noise/RIR 数据源。

验收：

- noise 或 reverb 至少一个场景相对 base 改善超过 10%。
- noise+reverb 合并超过 v3 的 6.71%。
- far_field/dropout 不比 base 恶化超过 2%。

### v6B: hard-profile + mixed constraint

目标：

- 在 v6A 有效后，加入 dropout/far_field 约束，减少观察场景回退。

范围：

- 训练中加入 10%-20% dropout/far_field。
- 仍不训练 text decoder。

验收：

- target 场景不明显丢失 v6A 收益。
- far_field repeat_like_rate 不超过 base。
- dropout too_short_rate 不高于 base。

### v6C: A2S-style encoder/aligner curriculum

目标：

- 按 WER 难度分桶，模拟 Mega-ASR 的 acoustic-to-semantic progressive SFT 第一阶段。

范围：

- 先训练 `wer_10_30` 和 `wer_30_50`，再加入少量 `wer_50_70`。
- target 使用 encoder/aligner。

验收：

- 扩展 MVP degraded-only 相对 base 改善 10%-15%。
- 高 WER 样本的 too_short 和 hallucination_like 有下降。

### v6D: text decoder pilot

目标：

- 小比例测试 LLM adapter 是否能改善高 WER 语义恢复。

范围：

- text decoder 后 4-8 层。
- 低学习率。
- 单独 adapter，便于回滚。

验收：

- 高 WER reverb/far_field 样本改善。
- clean 和 repeat_like 不恶化。

## 测试矩阵

每轮训练必须输出：

- checkpoint summary。
- target_modules 和 trainable 参数量。
- train/val/test manifest hash 或路径。
- base WER 分桶统计。
- overall 和 scenario WER。
- clean regression。
- repeat_like、too_long、too_short、hallucination_like。
- top improvements / top regressions。

固定评测集：

- MVP 150 hard profile：用于延续历史对比。
- MVP 1k hard profile：用于 v6A/v6B 后新增。
- Real/noisy holdout：用于 v6C 后新增。

## 验收与停止条件

阶段通过：

- 达到该阶段 WER 门槛。
- clean regression 在阈值内。
- 失败标签没有明显恶化。
- 输出、配置、随机种子和指标均已保存。

停止或回滚：

- 连续两轮 target 扩容未超过 v3，则停止 target 扩容。
- text decoder 导致 clean 或 repeat_like 恶化，则回滚到 audio-only。
- 新数据提升只出现在合成测试，不出现在真实 holdout，则暂停模型训练，回到数据构建。

## 影响

- 本方案会改变后续工作的重心：从 notebook 级单轮 ablation，转向数据版本、训练阶段和评测矩阵共同管理。
- v5 作为“audio MLP target 扩容不足以解决问题”的证据保留，不再作为主路线。
- 后续文档和配置命名应从 v6 开始体现数据版本和训练阶段；notebook 文件名继续递增编号。
