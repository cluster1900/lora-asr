# 快速 A2S 测试方案

最后更新：2026-07-22

## 背景与目标

测试只回答三个问题：数据与训练合同是否真实可复现、单 adapter 三阶段 A2S 是否按预期切换、
最终 Phase III 相对同环境 BF16 base 是否获得可发布的净收益。历史 4bit/MVP 结果只作为历史
证据，不作为本轮 baseline 或 checkpoint 选择依据。

## 范围与原则

- 测试覆盖 metadata/data、golden batch、target scope、10+2 resume、统一推理、evaluator、
  三次阶段 canary、一次 full validation 和一次固定 test。
- 所有指标使用固定 manifest、model/data revision、decode、normalization 和随机种子。
- 阶段 canary 只负责尽早终止坏 run，不挑 checkpoint。
- 固定 test 只在唯一正式候选上运行一次，不反向影响训练选择。

## 测试层级

### 1. 静态与 CLI

- 所有 Python 文件可解析，新 CLI `--help` 返回 0。
- YAML、JSON、JSONL 可解析，配置 schema 拒绝未知字段和非法 phase。
- 项目依赖、git commit、模型/数据 revision 和 seed 固定并写入运行记录。

### 2. 数据与 curriculum

- Metadata probe 验证固定 revision、54 个 split、必需字段、行数、en/zh 配额和磁盘预算。
- 128-row smoke 覆盖 en/zh、clean、7 atomic、compound 和 Bench real/synthetic。
- Full 行数精确为 200k train、10k validation、5k Bench。
- 每条 resolved `audio` 存在且可解码，`answer` 非空，`language=en|zh`，时长为 0.5-30 秒。
- Source group、配额、duration 和 rejects 检查通过；source id、audio hash、benchmark id 和
  硬 path overlap 为 0。
- Transcript overlap 只报告，不作为唯一泄漏证据。
- 同 seed/revision/config 生成完全相同的 row selection、manifest hash 和 30k curriculum。
- Curriculum 每行保存 base prediction、English WER 或 Chinese CER、统一
  `base_error_rate`；三个累计视图分别满足 `<0.30`、`<0.50`、`<0.70`，合计固定 30k。

### 3. Golden batch

- Qwen 官方 prompt 模板与 pinned revision 一致。
- Label 只覆盖 answer token，prompt、audio placeholder 和 padding 为 `-100`。
- en/zh 的有效 label 数大于 0，target 文本可无损解码回 reference。
- Audio feature mask、dtype、device 和 batch padding 正确。

### 4. 单 adapter target 与阶段切换

- 343 个 Linear target 按 audio attention 96、audio MLP 48、projection 3、decoder
  attention 112、decoder MLP 84 精确匹配。
- `lm_head`、embedding、norm 和三个 Conv2d frontend 命中数为 0；`conv_out` 只在运行时
  类型为 Linear 时进入 projection。
- 所有 target 一次注入同一 adapter，不创建或合并多个 adapter。
- Phase I 仅 27 个 upper-4 audio/projection target 有梯度。
- Phase II 仅 196 个 decoder target 有梯度。
- Phase III 全部 343 个 target 有梯度，两个学习率 parameter group 均符合 resolved config。
- 每阶段 target 数量、module type、model revision 或 target-map hash 不匹配时立即失败。

### 5. Smoke、保存与 resume

- 128 条平衡 fixture 训练 10 个 optimizer step，loss 与梯度范数有限。
- Checkpoint 保存 adapter、optimizer、scheduler、RNG、Trainer state、pipeline state、完整配置、
  target map/hash 和 manifest hash。
- 释放原进程后，新进程 resume 2 个 optimizer step，global step 从 10 连续到 12。
- 新进程加载 adapter，en/zh clean/degraded 各 1 条推理成功。
- Resume 前后固定 batch 的数据顺序、phase、学习率和随机状态符合保存记录。

### 6. 三个阶段 canary

Phase I、II、III 各自完成后，都在同一固定 512 条 validation canary 上运行一次：

- 输出有效率 >=95%。
- robust macro error 相对 BF16 base 恶化不超过 15%。
- empty、repeat-like、too-long 任一相对 BF16 base 增幅 <=5 个百分点。
- loss、梯度范数和学习率均有限。

任一条件失败时 runner 必须停止并保存该阶段 prediction、失败样本、训练状态和 resolved
config。三个 canary 使用相同 sample id 和同一 BF16 base prediction，不得按阶段重采样；
它们不用于选择 checkpoint。

### 7. 统一推理

统一推理入口同时支持 base 与可选 adapter：

- 使用真实 batch 或受控 micro-batch，不维护两份业务逻辑。
- 每条 prediction 增量写入 JSONL 并 flush。
- `--resume` 按 sample id 跳过已完成行，重复启动不产生重复记录。
- 单条失败写入 `error` 并继续整批。
- 每行保存 model id/revision、adapter id、dtype、device、decoding 和耗时。

### 8. Evaluator

Evaluator 必须拒绝空 reference，不得把空 reference 计为 0 error。输出必须包含：

- English WER、Chinese CER。
- `language x real/synthetic x condition` 32-cell 指标与 macro。
- real/synthetic、atomic/mixed、language 分组。
- clean regression。
- empty、repeat-like、too-long、hallucination-like。
- 原始 prediction、归一化 prediction、逐样本 edits 和 error。

English word edits 与 Chinese character edits 禁止合并成一个 overall WER/CER。

## 正式评测顺序

```text
Phase I -> 512 canary
Phase II -> 512 canary
Phase III -> 512 canary
          -> 10k full validation 一次
          -> Bench 5k 一次
          -> LibriSpeech test-clean 一次
          -> AISHELL-1 test 一次
          -> release reload
```

正式路径只对 Phase III 运行一次 10k full validation。若 Phase III validation 预检未通过但
Phase II canary 正常，可按需对 Phase II 运行一次 full validation，用于判断退化发生在哪一阶段；该
诊断不构成多候选 sweep，也不自动追加训练。Bench 和 clean test 仍只用于最终固定候选，不能
据此反向挑选 checkpoint。

## 产品验收标准

- English robust WER 相对 BF16 base 改善 >=10%。
- Chinese robust CER 相对 BF16 base 改善 >=10%。
- Bench 32-cell macro error 相对改善 >=10%，至少 24/32 cell 改善。
- real 与 synthetic macro 都改善。
- LibriSpeech test-clean WER 增幅 <= `max(0.3 个百分点, base WER 的 5%)`。
- AISHELL-1 test CER 增幅 <= `max(0.5 个百分点, base CER 的 5%)`。
- empty、repeat-like、too-long、hallucination-like 增幅均 <=1 个百分点。
- 单一 adapter、processor、release manifest、新进程加载和可恢复批量推理齐全。

## 互斥结果分支

按以下顺序判定，每次 run 只能进入一个分支：

| 判定 | 唯一动作 |
| --- | --- |
| 三项 robust 均 >=10%，且 clean/failure 全部通过 | 发布 adapter，运行同 evaluator 的 Mega-ASR gap 比较，然后停止训练 |
| robust 三项均 >=10%，但 clean 或 failure 未通过 | 停止并按失败类型记录 clean retention 需求，不自动重跑 |
| 三项 robust 均 >=5%，且至少一项 <10% | 保存实验 adapter 和诊断后停止；先归因，不自动加训练 |
| 任一 robust <5% | 停止，检查数据、base-error 桶、target、prompt、labels 和 evaluator |

以上条件本身互斥且覆盖全部结果；任何分支都不会在本轮自动扩展实验树。

## Mega-ASR 接近标准

使用 Mega-ASR 发布模型在同一 Bench manifest、decode、normalization 和 evaluator 下生成
独立 prediction。只有本项目 Bench macro error <= Mega-ASR macro error 的 1.10 倍，且
clean 门槛通过，才允许写“接近 Mega-ASR 微调效果”。在此之前只报告相对 BF16 base 的实测
结果。

## 自动化最小集

只覆盖高风险合同：

- metadata/schema/manifest/quota/dedup 单元测试。
- Qwen golden batch 测试。
- target 分组、禁止模块和三阶段梯度切换测试。
- checkpoint 10+2 resume 集成测试。
- evaluator 空 reference、en WER、zh CER 和 32-cell 聚合测试。
- inference 增量写入、错误续跑和 resume 测试。
- 固定 512 canary sample-id/hash 一致性测试。

## 验收、影响与当前缺口

测试完成的定义是：上述自动化与 10+2 smoke 通过，三次 canary 有固定输入和结果，Phase III
只做一次 full validation 与固定 test，release 在新进程中可加载，并将 prediction、metrics、
resolved config 和失败样本全部保存。

这套测试删除重复 full evaluation，缩短训练期间停顿，但也意味着不能用多个中间 checkpoint
事后挑最好结果。当前历史数据脚本、WER/CER、错误分析和 CLI smoke 可运行；新 200k 数据、
A2S runner、统一推理、32-cell evaluator 和对应自动测试仍需按实际完成情况更新
`00_progress.md`，未跑出结果前不得标记主线完成。
