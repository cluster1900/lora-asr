# 开发进度

最后更新：2026-09-18

## 当前状态

项目已将正式目标改为 4 × Tesla V100-SXM2-32GB 服务器训练。服务器只读盘点已完成：GPU
空闲、`/data` 有约 5.5TB 可用空间，系统 Python 没有训练依赖；现有 `/data/mini-k3/venv`
虽然有 PyTorch 2.5.1+cu121，但没有 qwen-asr、Transformers 或 PEFT，且不能作为本项目的共享
运行环境。

已完成：

- 完成 V100 服务器资源、CUDA、磁盘、网络镜像和现有进程的只读盘点。
- 确认 V100 主线必须采用 FP16、eager attention、独立环境和单机多卡 DDP。
- 完成 SFT → DPO → RL 完整链路、4 卡 DDP、数据角色隔离和阶段 gate 规划。
- 完善执行合同关键机制：固化 LoRA targets（LLM Decoder + 音频 Projection）、各阶段 `merge_and_unload()` 权重接力、DPO 离线预计算 logps（显存减半）、RL GRPO $G=4$ 采样约束、Clean 候选池 5x 扩充应对平局过滤、以及全局 Base 锚定的 clean 累积退化上限（≤0.025）。
- 评测模块与测试用例同步支持 `text` 字段与 `clean|degraded` 条件分组。
- 本地合同测试通过；这些测试只覆盖静态逻辑，不能替代 V100 GPU 验证。
- 已同步架构、开发、数据、训练、测试、风险和根 README 的规划说明。

尚未完成：

- V100 独立环境、requirements 和运行时配置迁移。
- FP16/eager attention 的模型加载、单 batch 前向反向和 4 卡 DDP smoke。
- 服务器上的 128-row 数据 smoke、base baseline 和 checkpoint resume。
- 数据 builder、SFT/DPO/RL pilot、固定 validation/test、最终 RL release adapter 和外部 baseline。

因此当前状态是“规划已同步，代码仍未迁移，尚无 V100 训练结果”。

## 既有实现的基线记录

- `python3 -m unittest discover -s tests -v`：当前本地合同测试通过。
- 旧训练、推理、Colab notebook、Colab requirements、BF16/A2S 配置和旧数据 builder 已删除。
- V100 数据 builder、训练 runner 和推理 runner 尚未重建；当前仓库不包含可启动训练入口。

## 下一步

1. 先实现完整数据下载/物化和角色池 builder，直到 `DATASET_COMPLETE.json` 通过。
2. 数据门禁通过后再创建 V100 FP16/eager 环境，运行 base smoke 和 10+2 checkpoint resume。
3. 依次执行 SFT、DPO、RL 三个 pilot，并检查 merge、preference、reward、KL 和 clean 累积退化。
4. 三个 pilot 全部通过后，按 08 号合同依次运行 full 和最终 test。

## 验收

只有固定 manifest、配置、随机种子、V100 base/SFT/DPO/RL 对比、WER/CER、preference、reward、原始预测、三阶段 gate 和 release
artifacts 全部存在时，才可标记阶段完成。当前不得声称达到或超过 Mega-ASR。
