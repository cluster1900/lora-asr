# 开发进度

最后更新：2026-08-10

## 当前状态

仓库已重置为单一 A2S 主线。历史 v1-v6A 训练、合成数据、checkpoint、预测结果、router
占位和旧 Colab notebook 已退出当前设计，不再保留其实现或复现材料。

已实现：

- 固定公开数据配额、JSONL 校验、泄漏检查、512-row canary 和 30k curriculum。
- pinned Hub streaming staging：smoke/full 配额、音频物化、100-row durable checkpoint、hash
  resume 修复、Robust 90/10 base-source 分区和 clean 官方 split 隔离。
- 单 adapter 三阶段 A2S runner，包含 target 合同、checkpoint/resume 和阶段 canary。
- BF16 base/adapter 统一推理，逐条持久化并支持 resume。
- English WER、Chinese CER、scenario/32-cell 聚合和失败输出统计。
- 唯一 Colab 入口 `notebooks/12_fast_finetune_colab.ipynb`，覆盖 probe、128-row smoke、10+2
  resume、full build、按需 curriculum 评分、base canary、正式训练和 release 评测。
- 上述核心逻辑的本地自动测试。

尚未完成：

- 在干净 Colab 实际执行公开音频 staging 和正式 manifest。
- 10+2 step GPU smoke、checkpoint 加载和继续训练验证。
- Phase I/II/III 正式训练、canary、10k validation 和固定 test。
- release adapter、processor、结果摘要和 Mega-ASR 同 evaluator 外部 baseline。

因此当前状态是“代码合同可本地验证，正式数据与 GPU 结果未验证”。

## 本次重置验证

- `python3 -m unittest discover -s tests -v`：45 项通过。
- 四个正式入口的 `--help` 与 `py_compile`：通过。
- 训练配置 `--validate-only --print-plan`：通过，target 分组为 96/48/3/112/84，合计 343。
- pinned `datasets==5.0.0` 真实 Hub metadata probe：四源通过；确认 54 robust split、16 Bench
  split、LibriSpeech 4 split 和 AISHELL-1 120098/14326/7176 三切分。
- 四源各读取 1 条 decode-free streaming row：`audio` 均提供 `bytes/path`，gold text 与语言推断通过；
  未执行完整 128-row 音频解码。
- 新训练入口已去除旧 adapter 注入、跳阶段、旧 manifest 字段和旧依赖参数兼容路径。
- 推理入口已固定 BF16 运行合同，删除会造成 base/adapter 对比漂移的覆盖参数。
- YAML 已删除代码不读取的重复/占位字段；数据集 split 作为后续 staging 合同保留。

## 下一步

1. 在干净 Colab 依次执行 Notebook 的 metadata、smoke staging 和 128-row build。
2. 跑 BF16 base clean/degraded 推理，再跑 10 step、保存 checkpoint并继续 2 step。
3. 核对 target、global step、配置和 canary 路径后，才执行 full staging/build。
4. curriculum 满 30k、base canary 就绪后启动三阶段训练和固定评测。

## 验收

只有固定 manifest、配置、随机种子、base 对比、WER/CER、原始预测和 release artifacts
全部存在时，才可标记训练阶段完成。当前不得声称达到或超过 Mega-ASR。
