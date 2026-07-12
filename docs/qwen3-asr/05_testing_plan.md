# 测试方案

最后更新：2026-07-12

## 目标

证明新主线的数据、Trainer、推理、评测和发布产物真实可用，并用同环境 BF16 base 判断
LoRA 是否获得净收益。历史 4bit/MVP 结果只做回归参考。

## 测试层级

### 1. 静态与 CLI

- 所有 Python 文件可解析。
- 新 CLI `--help` 返回 0。
- YAML/JSON/JSONL 可解析。
- 项目依赖版本固定并写入运行记录。

### 2. 数据

- Smoke 覆盖 en/zh、clean、7 atomic、compound、Bench real/synthetic。
- Full 行数精确为 200k train、10k validation、5k Bench。
- 每条 resolved `audio` 存在、可解码，`answer` 非空，`language=en|zh`。
- Source group、防泄漏、配额、duration 和 rejects 检查通过。
- Transcript overlap 只报告；source id、audio hash 和 benchmark id 硬 overlap 为 0。
- 同 seed/revision/config 可复现相同 manifest hash。

### 3. Golden batch

- Qwen 官方 prompt 模板与 pinned revision 一致。
- Label 只覆盖 answer token，prompt、audio placeholder 和 padding 为 `-100`。
- en/zh 样本的有效 label 数大于 0，target 文本可无损解码回 reference。
- Audio feature mask、dtype 和 device 正确。

### 4. LoRA target

- 343 个 Linear target 按 5 个分组精确匹配。
- `lm_head`、embedding、norm 和三个 Conv2d frontend 命中数为 0。
- `conv_out` 按实际类型校验；它是 Linear 时允许进入 speech projection 分组。
- PEFT 实际可训练参数约 12,365,824，以运行时统计为准。

### 5. Smoke 与 resume

- 128 条数据训练 10 optimizer step，loss 有限。
- checkpoint 保存 adapter、optimizer、scheduler、RNG、Trainer state 和完整配置。
- 新进程 resume 2 optimizer step，global step 从 10 到 12。
- 新进程加载 adapter，en/zh clean/degraded 各 1 条推理成功。

### 6. Canary

正式 run step 100 在固定 512 条 validation 上执行：

- 输出有效率 >=95%。
- empty、repeat-like、too-long 任一相对 BF16 base 增幅 <=5 个百分点。
- robust macro error 相对 BF16 base 恶化不超过 15%。
- loss、梯度范数和学习率均有限。

失败时训练进程必须停止并保存诊断，不继续到 50%。

### 7. 推理

统一推理入口同时支持 base 与可选 adapter：

- 至少真实 batch 或受控 micro-batch，不再维护两份近乎相同的业务逻辑。
- 每条 prediction 增量写入 JSONL 并 flush。
- `--resume` 按 sample id 跳过已完成行。
- 单条失败写入 `error`，不终止整批。
- 每行保存 model id/revision、adapter id、dtype、device、decoding 和耗时。

### 8. 评测

Evaluator 必须拒绝空 reference；不得把空 reference 计为 0 error。

输出：

- English WER。
- Chinese CER。
- `language x real/synthetic x condition` 32-cell 指标与 macro。
- real/synthetic、atomic/mixed、language 分组。
- clean regression。
- empty、repeat-like、too-long、hallucination-like。
- 原始 prediction、归一化 prediction、逐样本 edits 和 error。

English word edits 与 Chinese character edits 禁止合并成一个 overall WER/CER。

## Checkpoint 选择

50% 与 100% checkpoint 只使用 10k validation 比较。唯一候选确定后，Bench 5k 与双语
clean test 只运行一次。不得根据 fixed test 反向选择 checkpoint。

## 验收标准

### 产品 MVP

- English robust WER 相对 BF16 base 改善 >=10%。
- Chinese robust CER 相对 BF16 base 改善 >=10%。
- Bench 32-cell macro error 相对改善 >=10%，至少 24/32 cell 改善。
- real 与 synthetic macro 都改善。
- LibriSpeech test-clean 增幅 <= `max(0.3 个百分点, base WER 的 5%)`。
- AISHELL-1 test 增幅 <= `max(0.5 个百分点, base CER 的 5%)`。
- 四类失败率增幅均 <=1 个百分点。

### Mega-ASR 接近标准

使用 Mega-ASR 发布模型在同一 Bench manifest、normalization 和 evaluator 下生成本项目
自己的 prediction。只有本项目 macro error <= Mega-ASR macro error 的 1.10 倍，且 clean
门槛通过，才允许写“接近 Mega-ASR 微调效果”。

## 自动化最小集

新增测试只覆盖高风险合同，不建立庞大测试框架：

- schema/manifest/dedup 单元测试。
- Qwen golden batch 测试。
- target 分组与禁止模块测试。
- checkpoint 10+2 resume 集成测试。
- evaluator 空 reference、en WER、zh CER 和 32-cell 聚合测试。
- inference 增量写入与 resume 测试。

## 当前已验证与缺口

现有数据脚本、WER/CER、错误分析和 CLI smoke 可运行；本机缺少 GPU 训练依赖，本次审计
未重跑 Qwen 推理或训练。新 200k 数据、官方 Trainer、统一推理、32-cell evaluator 和自动
测试均尚未实现，因此快速主线仍不能标记完成。
