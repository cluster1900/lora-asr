# 07 Router MVP

最后更新：2026-06-30

## 背景

鲁棒 LoRA 可能提升 degraded audio，但也可能损害 clean speech。Router 的作用是判断音频是否退化，并决定是否启用 LoRA。

## 目标

训练一个轻量 clean/degraded 分类器，并验证 router 模式是否优于 always-base。

## 范围

本步骤只做二分类 router MVP，不做复杂多场景分类。

## 输入

- 带 `is_degraded` 标签的音频 manifest。
- router 训练配置。
- base 和 LoRA 推理入口。

## 输出

- router checkpoint。
- threshold 配置。
- router eval report。
- router mode prediction JSONL。

## 需要实现的文件

- `router/train_router.py`
- `router/infer_router.py`
- `router/model.py`
- `configs/train/router_mvp.yaml`
- `notebooks/10_router_colab.ipynb`

## 执行步骤

1. 提取 log-mel spectrogram。
2. 训练 clean/degraded 二分类器。
3. 在验证集上选择 threshold。
4. 在测试集上报告 accuracy、precision、recall、F1。
5. 接入推理：clean 走 base，degraded 走 LoRA。
6. 对比 always-base、always-LoRA、router。
7. 更新架构、测试、进度文档。

## 测试标准

- router 能对单条音频输出 `degraded_prob`。
- threshold 来源记录在配置或报告中。
- 批量推理能保存 route decision。
- router 失败不应中断 ASR 推理，应有 fallback 策略。

## 验收标准

- router 模式在混合测试集上优于 always-base。
- clean regression 小于预设阈值，MVP 默认小于 5% 相对退化。
- degraded 场景表现接近 always-LoRA。
- router 指标和 ASR 指标都已报告。

## 风险

- router 错把 degraded 判成 clean。缓解：优先提高 degraded recall。
- router 对合成退化过拟合。缓解：加入真实 holdout。
- 双模型路径维护复杂。缓解：统一推理接口，记录 route_source。
