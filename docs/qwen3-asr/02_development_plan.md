# 开发计划

## 背景与范围

仓库只维护一个可运行闭环，避免历史实验入口与正式入口并存。当前范围是补齐 Colab 数据 staging
并执行既有 A2S 流程；不新增模型分支、sweep、router 或独立评测器。

## 唯一流程

1. `scripts/prepare_public_robust_manifests.py`：probe、stage、smoke、build、validate、curriculum。
2. `inference/qwen3_asr_infer.py`：生成 BF16 base curriculum 分数和 base/test prediction。
3. `train/train_qwen3_asr_a2s.py`：运行 smoke 或 Phase I/II/III，并保存完整状态。
4. `evaluation/eval_wer.py`：接收 prediction 和 output directory，固定生成 scored JSONL、metrics
   JSON 及 scenario/cell/language CSV。

核心配置只有：

- `configs/data/public_robust_200k.yaml`
- `configs/train/qwen3_asr_public_200k_a2s.yaml`

当前接口只接受新 manifest 合同：每行必须有唯一 `sample_id` 和 `audio`。依赖版本已固定，不保留
旧 prediction ID、旧 `audio_path` 字段或旧 `qwen-asr` 参数兼容层。推理 CLI 不允许覆盖模型、
revision、精度、device、batch 或语言，防止正式比较漂移。

## 开发步骤

1. 数据脚本从 pinned Hub revision 流式 staging，仅物化配额需要的音频，输出四份 candidate
   JSONL、rejects 和报告；重复执行从已落盘且 hash 有效的行继续。
2. 唯一 notebook `notebooks/12_fast_finetune_colab.ipynb` 只编排现有 CLI，不复制数据、训练或
   评测逻辑。
3. 运行 metadata probe 和 128-row smoke；不满足 schema、配额、音频或泄漏检查时停止。
4. BF16 base curriculum 评分先跑 60k，数量不足再扩到 100k/160k/200k，不预先推理全部 200k。
5. 生成 30k curriculum，并固定生成 512-row base canary 指标。
6. 运行 10+2 step resume smoke，确认 checkpoint、optimizer、scheduler、RNG 和配置可恢复。
7. 训练器只按 Phase I -> II -> III 推进；每阶段后运行 512 canary，失败则停止。
8. 对 base 与 adapter 使用同一 validation/test manifest 和 evaluator。

## 测试

每次代码变更运行：

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile scripts/prepare_public_robust_manifests.py \
  train/train_qwen3_asr_a2s.py inference/qwen3_asr_infer.py evaluation/eval_wer.py
```

数据或训练合同变化还要运行 `--help`、config validation 和 128-row smoke。正式执行必须记录命令、
revision、manifest hash、随机种子和输出路径。

## 完成条件

- 干净 Colab 可从配置生成固定 manifest。
- clean/degraded 推理均能逐条写 prediction。
- 10+2 resume 成功且 resolved config 随 checkpoint 保存。
- base/adapter 均产出 English WER、Chinese CER、scenario 与 32-cell 指标。
- release adapter 和 processor 可重新加载。

## 影响

新功能必须直接延伸上述四个入口。训练器不提供跳阶段或命令行注入旧 adapter 的入口；恢复只读取
当前 output directory 的 pipeline state 和 checkpoint。需要第二套入口时，先证明现有接口无法
表达需求并更新本文件。
