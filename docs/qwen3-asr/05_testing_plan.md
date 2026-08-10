# 测试方案

## 范围

测试覆盖数据、训练合同、推理恢复和双语评测。单元测试不下载模型或公开数据；GPU smoke 在 Colab
单独执行。

## 本地测试

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile scripts/prepare_public_robust_manifests.py \
  train/train_qwen3_asr_a2s.py inference/qwen3_asr_infer.py evaluation/eval_wer.py
python3 scripts/prepare_public_robust_manifests.py --help
python3 train/train_qwen3_asr_a2s.py --help
python3 inference/qwen3_asr_infer.py --help
python3 evaluation/eval_wer.py --help
```

通过标准：测试全部通过、CLI 可解析、无语法错误、`git diff --check` 无错误。

## 数据测试

128-row smoke 必须覆盖全部 robust split、English/Chinese clean，且路径、时长、hash、配额和泄漏
检查通过。正式 manifest 的行数必须精确匹配 200k/10k/512/30k/5k 合同，CLI 不允许覆盖这些
固定数量或跳过计数检查。

## 训练与推理测试

- 10+2 step resume 后 checkpoint、配置、global step 和 adapter 可恢复。
- 缺少或重复 `sample_id`、缺少 `audio` 时必须明确报错或记录单条推理错误。
- 推理帮助中不得重新出现模型、精度、device、batch 或语言覆盖参数。
- clean 与 degraded 各至少一条成功；单条失败写 `error` 并继续。
- prediction 每完成一条即 flush+fsync；重跑 `--resume` 不重复样本。

## 评测

- English 计算 WER，Chinese 计算 CER，按 scenario 聚合。
- Voices-in-the-Wild-Bench 输出 language x origin x scenario 的 32-cell macro。
- 保存 raw/normalized reference、prediction、edit count 和所有指标。
- normalization 固定为 lowercase + 去标点，不提供运行时覆盖。
- 空 reference 硬失败；inference error 按全删除计分并进入失败率。
- 报告 clean regression、空输出、重复输出、过长和幻觉式输出。

## 阶段验收

adapter 必须在同一 manifest、同一 BF16 base、同一 evaluator 下改善至少一个 degraded 场景，
同时量化 clean regression。Router 不在当前范围；没有正式结果时不得标记模型阶段完成。

## 影响

删除历史 fixture 后，所有本地测试只使用临时合成 JSON/对象，不依赖仓库内 checkpoint、prediction
或音频文件。
