# notebooks 目录说明

## 目录职责

本目录保存面向 Google Colab 的可执行编排入口。Notebook 只负责挂载 Drive、准备环境并按顺序调用
仓库 CLI，不在单元格中复制第二套数据、训练、推理或评测实现。

当前只保留一条快速微调路径，避免多个 Notebook 参数漂移或使用不同产物目录。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `12_fast_finetune_colab.ipynb` | 唯一 Colab 入口。覆盖 metadata probe、128-row smoke、10+2 resume、完整 staging、30k curriculum、base baseline、三阶段训练和 release 评测。 |
| `README.md` | 说明 Notebook 的定位、执行阶段、产物目录和维护要求。 |

## Notebook 执行阶段

1. `Metadata 与 128-row smoke`：验证数据源、音频解码、manifest 和 base clean/degraded 推理。
2. `10+2 step checkpoint/resume 门禁`：验证 343 targets、训练保存与新进程恢复。
3. `按配额 staging 完整数据`：生成 200k train、10k validation、512 canary 和 5k Bench。
4. `最少量 BF16 base 评分`：从 60k 开始按需扩展，生成固定 30k curriculum。
5. `固定 base baseline`：保存 canary、validation 和 Bench 的基线指标。
6. `单 adapter 三阶段训练`：依次运行 Phase I/II/III、阶段 canary 和 release 评测。

前一阶段未通过时不得继续下一阶段。staging、推理和训练均使用持久化/恢复机制，运行时中断后重跑
对应单元即可。

## 路径与产物

- 仓库：`/content/mega-asr`。
- 临时音频：`/content/qwen3-asr-runtime/data`，使用 Colab 本地 SSD，运行时释放后会消失。
- Drive 根目录：`MyDrive/qwen3-asr-public-a2s/`。
- Drive 子目录：`candidates/`、`manifests/`、`base/`、`runs/`、`results/`。

Notebook 必须保留固定仓库 commit 输出和独立 Drive 命名空间，不能读取旧 Mega-ASR 实验产物。

## 打开方式

[在 Google Colab 中打开](https://colab.research.google.com/github/cluster1900/lora-asr/blob/main/notebooks/12_fast_finetune_colab.ipynb)

对应静态测试：`tests/test_fast_finetune_notebook.py`。

## 维护要求

新增、重命名或删除 Notebook 时，必须同步更新“文件清单”和仓库根 README。修改执行阶段、路径、
CLI 参数、产物或门禁时，必须在同一提交更新本 README、Notebook 内说明、专项文档和静态测试。
除非架构文档明确变更，始终只保留一个正式 Colab 入口。
