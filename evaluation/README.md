# evaluation

存放新工程的评测脚本。

目标：

- 统一计算 WER/CER。
- 输出 overall 和 scenario-level 指标。
- 保存失败样本，支持错误分析。

禁止：

- 不依赖 `references/mega-asr-upstream/` 中的评测代码作为运行时模块。

