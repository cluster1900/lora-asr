# notebooks 目录说明

## 目录职责

本目录保留 Notebook 目录合同，但当前没有正式 Notebook。训练入口已经切换到 V100 服务器上的
CLI；后续如需新增 Notebook，必须先证明它不会复制第二套数据、训练或评测逻辑。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `README.md` | 说明本目录当前为空、入口约束和维护要求。 |

## 当前状态

旧的 Colab notebook、Colab 依赖和 BF16/A2S 编排已经删除。当前方案见
`docs/qwen3-asr/07_v100_server_training_plan.md`，正式实现完成前不得新增临时 Notebook 作为训练入口。

## 维护要求

新增、重命名或删除 Notebook 时，必须同步更新“文件清单”和仓库根 README。修改执行阶段、路径、
CLI 参数、产物或门禁时，必须同步更新本 README、专项文档和静态测试。Notebook 不能成为服务器
CLI 之外的第二套业务逻辑。
