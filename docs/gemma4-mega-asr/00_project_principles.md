# 项目原则

最后更新：2026-06-07

## 项目定位

这是一个基于 Gemma 4 12B 的独立鲁棒 ASR 项目。

Mega-ASR 是参考项目，不是实现底座。我们学习它的方法，但最终代码、架构、训练流程和部署流程都应由我们自己实现。

原 Mega-ASR 工程只能保存在本地忽略目录 `references/mega-asr-upstream/`。新工程根目录以 Gemma 4 Robust ASR 为主，不再把 Mega-ASR 的源码、脚本和资源散放在主路径中。`references/` 不进入 git。

## 可以借鉴 Mega-ASR 的部分

- 鲁棒 ASR 的问题定义。
- 声学退化场景分类。
- 渐进式监督微调思路。
- clean/degraded 音频路由思路。
- 以 WER/CER 为核心的评测方式。
- 空输出、幻觉、漏识别等失败模式分析。

## 不直接复用的部分

- Qwen3-ASR 专用 wrapper。
- Qwen3-ASR 模块名。
- Qwen3-ASR LoRA target 正则。
- Mega-ASR 推理适配层。
- Mega-ASR 的最终项目结构。
- 任何把 Gemma 支持做成 Qwen3-ASR 补丁的代码路径。

## 参考工程隔离规则

- `references/mega-asr-upstream/` 只读参考。
- 不在该目录下新增 Gemma 4 功能代码。
- 不从新工程导入该目录中的 Python 模块。
- 不提交 `references/` 下的任何文件。
- 如果参考其中的思路，必须先记录到本项目文档，再独立实现。

## 工程原则

1. 基于 Gemma 4 的真实 API 和模型结构开发。
2. 数据、训练、推理、router、评测模块彼此解耦。
3. Colab 是第一优先训练环境。
4. 每次实验都应能从 JSONL manifest 和配置文件复现。
5. 重要决策和结果必须同步到文档。
6. 先做小而完整的闭环，再做大规模训练。

## 命名

当前工作名：

- Gemma 4 Robust ASR

在项目拥有自己的结果和身份前，避免使用容易让人误解为 Mega-ASR fork 或替代品的名称。
