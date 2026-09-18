# scripts 目录说明

## 目录职责

本目录预留 V100 服务器的数据 builder。旧的 200k/A2S manifest builder、配置和测试已经删除；新
实现必须从固定 source、音频 staging、manifest、泄漏检查和恢复语义重新开始。

## 文件清单

| 文件 | 作用 |
| --- | --- |
| `README.md` | 说明当前目录为空、未来 data builder 合同和维护要求。 |

## 当前状态

当前没有可执行脚本。未来 data builder 必须支持 smoke、pilot 和 full 三个阶段，统一写入
`/data/mega-asr/`，并保存 source revision、manifest hash、音频 hash、随机种子和恢复日志。
不得读取 `/data/mini-k3`，不得把 test 数据写入训练 manifest。

## 维护要求

新增、重命名或删除脚本时，必须同步更新“文件清单”和仓库根 README。修改数据源、配额、schema、
输出文件、恢复策略或泄漏规则时，必须同步更新本 README、`docs/qwen3-asr/03_data_plan.md`、
开发/测试/进度文档和对应测试。目录没有可执行脚本时，不得在 README 中保留失效命令。
