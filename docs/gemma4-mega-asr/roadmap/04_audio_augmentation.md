# 04 音频增强

最后更新：2026-06-07

## 背景

真实退化音频很难一次性收集齐。MVP 阶段需要通过可控增强生成噪声、远场、混响、失真、dropout 等场景，让模型先接触典型失败模式。

## 目标

实现可复现的音频退化增强脚本，并为每条增强样本记录参数。

## 范围

本步骤只做离线增强，不做实时增强训练。

## 输入

- clean audio manifest。
- 噪声素材目录。
- RIR 素材目录，可选。
- 增强配置文件。
- 随机种子。

## 输出

- 增强后的音频文件。
- 增强 manifest。
- 每条样本的 augmentation metadata。
- 增强质量检查报告。

## 首批场景

1. clean
2. noise
3. reverb
4. far_field
5. clipping
6. dropout
7. noise_reverb
8. far_field_noise

## 需要实现的文件

- `scripts/augment_audio.py`
- `scripts/check_audio_quality.py`
- `configs/data/augmentation_mvp.yaml`

## 执行步骤

1. 定义增强配置 schema。
2. 实现 noise mixing，支持 SNR 参数。
3. 实现 reverb/RIR 卷积。
4. 实现 far-field 模拟。
5. 实现 clipping 和 saturation。
6. 实现 dropout。
7. 生成增强音频并写 metadata。
8. 随机抽样听音检查。
9. 更新数据方案和进度文档。

## 参数记录

每条增强样本至少记录：

- `scenario`
- `source_audio`
- `output_audio`
- `noise_file`
- `snr_db`
- `rir_file`
- `clip_threshold`
- `dropout_rate`
- `seed`

## 测试标准

- 每类场景至少生成 5 条样本。
- 输出音频可被 soundfile/torchaudio 读取。
- 增强后音频非空，峰值不全为 0。
- metadata 与输出文件一一对应。
- 同一 seed 下结果可复现。

## 验收标准

- MVP 所需至少 5 类退化可生成。
- 增强 manifest 可直接进入训练/评测数据构建。
- 抽样听音确认退化存在但 transcript 仍合理。
- 质量检查报告记录异常样本。

## 风险

- 增强过强导致语音不可辨。缓解：设置 SNR、dropout、clipping 上下限。
- 合成退化不贴近真实世界。缓解：后续加入真实录音 holdout。
- 音量爆炸或溢出。缓解：统一 peak normalize 和 clipping 检查。

