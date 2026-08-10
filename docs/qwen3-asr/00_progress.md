# 开发进度

最后更新：2026-08-10

## 当前状态

仓库已重置为单一 A2S 主线。历史 v1-v6A 训练、合成数据、checkpoint、预测结果、router
占位和旧 Colab notebook 已退出当前设计，不再保留其实现或复现材料。

已实现：

- 固定公开数据配额、JSONL 校验、泄漏检查、512-row canary 和 30k curriculum。
- 单 adapter 三阶段 A2S runner，包含 target 合同、checkpoint/resume 和阶段 canary。
- BF16 base/adapter 统一推理，逐条持久化并支持 resume。
- English WER、Chinese CER、scenario/32-cell 聚合和失败输出统计。
- 上述核心逻辑的本地自动测试。

尚未完成：

- 从干净 Colab 环境物化公开音频和正式 manifest。
- 10+2 step GPU smoke、checkpoint 加载和继续训练验证。
- Phase I/II/III 正式训练、canary、10k validation 和固定 test。
- release adapter、processor、结果摘要和 Mega-ASR 同 evaluator 外部 baseline。

因此当前状态是“代码合同可本地验证，正式数据与 GPU 结果未验证”。

## 本次重置验证

- `python3 -m unittest discover -s tests -v`：38 项通过。
- 四个正式入口的 `--help` 与 `py_compile`：通过。
- 训练配置 `--validate-only --print-plan`：通过，target 分组为 96/48/3/112/84，合计 343。
- Hub metadata probe：本机缺少 `datasets`，未做真实远端验证；该依赖已固定在 Colab requirements。
- 新训练入口已去除旧 adapter 注入、跳阶段、旧 manifest 字段和旧依赖参数兼容路径。
- 推理入口已固定 BF16 运行合同，删除会造成 base/adapter 对比漂移的覆盖参数。
- YAML 已删除代码不读取的重复/占位字段；数据集 split 作为后续 staging 合同保留。

## 下一步

1. 新建唯一 Colab notebook，完成依赖安装、Drive 挂载和 candidate staging。
2. 生成 128-row smoke manifest，跑 BF16 base clean/degraded 推理。
3. 跑 10 step、保存 checkpoint，再继续 2 step；核对 target、状态和配置。
4. 门禁通过后才启动完整数据和三阶段训练。

## 验收

只有固定 manifest、配置、随机种子、base 对比、WER/CER、原始预测和 release artifacts
全部存在时，才可标记训练阶段完成。当前不得声称达到或超过 Mega-ASR。
