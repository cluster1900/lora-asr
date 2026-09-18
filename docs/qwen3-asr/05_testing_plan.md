# 测试方案

## 范围

测试覆盖数据、训练合同、推理恢复和双语评测。单元测试不下载模型或公开数据；GPU smoke 在 V100
服务器单独执行。

## 本地测试

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile evaluation/eval_wer.py
python3 evaluation/eval_wer.py --help
```

通过标准：测试全部通过、CLI 可解析、无语法错误、`git diff --check` 无错误。

## 数据测试

数据下载/物化是第一个可执行阶段。模型下载、base inference、SFT、DPO 和 RL 都必须等待
`DATASET_COMPLETE.json`。

128-row smoke 必须覆盖全部 robust split、English/Chinese clean，且路径、时长、hash、配额和泄漏
检查通过。SFT、DPO、RL pilot manifest 必须分别满足 08 号合同配额；只有三个 pilot 全部通过后才允许
构建 full role pools。Bench/test 数据不能进入任何训练阶段。

data builder 的 fixture 必须验证：

- 同一命令第二次执行不重复下载或追加；中断后能补齐剩余配额。
- candidate 指向的音频缺失或 hash 改变时，该行不计入 resume 进度并被重新物化。
- 原始音频不可覆盖；处理后音频必须可读、mono、16 kHz、PCM/WAV、0.5–30 秒，并同时记录原始/处理后 hash 和处理版本。
- Robust 同一个 source index 在不同 scenario 得到同一 source utterance ID，且只进入一个
  train/validation 分区。
- Clean train/validation 分别来自配置指定 split；Bench smoke 覆盖 en/zh x real/synthetic。
- stage report 的 requested/materialized/resumed/rejected/shortage 数量与文件一致。

目录 README 合同测试继续检查维护文件清单；它不能作为 V100 训练验收。V100 还必须记录 GPU 型号、
CUDA/PyTorch 版本、dtype、attention 实现、world size 和 manifest hash。

## V100 训练与推理测试

- FP16 单 batch 前向和反向成功，loss、gradient、learning rate 均为有限值。
- 4 卡 DDP 的 global batch 等于配置值；10+2 step resume 后 checkpoint、配置、world size、global
  step 和 adapter 可恢复。
- 缺少或重复 `sample_id`、缺少 `audio` 时必须明确报错或记录单条推理错误。
- clean 与 degraded 各至少一条成功；单条失败写 `error` 并继续。
- prediction 每完成一条即 flush+fsync；重跑 `--resume` 不重复样本。

## 评测

- English 计算 WER，Chinese 计算 CER，按 scenario 聚合。
- Voices-in-the-Wild-Bench 输出 language x origin x scenario 的 32-cell macro。
- 保存 raw/normalized reference、prediction、edit count 和所有指标。
- 空 reference 硬失败；inference error 按全删除计分并进入失败率。
- 评测支持官方 `text`（兼容历史 `answer`），条件分组支持 `clean` 与 `degraded`（兼容 `atomic`/`compound`）。
- 报告 clean regression、空输出、重复输出、过长和幻觉式输出。
- Canary gate 不得用 clean+robust 混合宏平均冒充 robust 指标。

## SFT/DPO/RL 阶段验收

SFT pilot 必须在同一 manifest、同一 FP16 base、同一 evaluator 下改善至少一个 degraded 场景，同时
clean 错误率绝对增加不超过 0.02，有效输出率不少于 0.95，失败率增加不超过 0.05，且 adapter 能在新进程加载并成功 `merge_and_unload` 作为下阶段底座。

DPO 测试必须验证：

- chosen/rejected 非空、不同、可追溯，ties 和坏音频进入 rejects，Clean 数据通过扩大候选源保障足额产出；
- DPO trainer 不读取 gold/error-rate 审计字段，支持预计算参考模型 logps；
- held-out preference accuracy ≥0.55；
- 相对 SFT 至少一个 degraded scenario 改善，clean 相对 SFT 回退不超过 0.02，且相对 Base 全局累积回退不超过 0.025；
- DPO adapter 能在新进程加载并成功 `merge_and_unload` 作为 RL 底座。

RL 测试必须验证：

- GRPO 组大小 $G=4$，采样 `temperature=0.7`, `top_p=0.9`，避免同质方差归零；
- reward 组件（asr 及各项惩罚）和最终 reward 对空输出、重复、过长、hallucination 的单测；
- rollout 每行包含 policy checkpoint、seed、prediction、reward、KL 和 error；
- reference policy 冻结，KL 正则与 gradient 有限；
- held-out mean reward 相对 DPO reference 提升 ≥0.05；
- 相对 DPO 至少一个 degraded scenario 改善，clean 相对 DPO 回退不超过 0.02，且相对 Base 全局累积回退不超过 0.025；
- RL adapter 能在新进程加载并完成 clean/degraded 推理。

没有 SFT、DPO、RL 三个阶段的 gate.json、四组 prediction、合并 release 模型和固定 test 结果，不得标记完整后训练完成。
Router 不在当前范围。

## 影响

删除历史 fixture 后，所有本地测试只使用临时合成 JSON/对象，不依赖仓库内 checkpoint、prediction
或音频文件。
