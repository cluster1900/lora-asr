# notebooks

当前唯一正式 Colab 入口是待实现的 `12_fast_finetune_colab.ipynb`，执行合同见
`docs/qwen3-asr/04_colab_training_plan.md`。

Notebook 只负责环境、Drive、配置选择和调用脚本，不复制数据、训练或评测逻辑。固定顺序为：

1. 挂载 Drive、安装 pinned 依赖并记录 commit/revision。
2. 运行 metadata probe；数据未完成时可先跑 128-row trainer fixture。
3. 构建 200k/10k/5k manifest 和 30k curriculum。
4. 执行 10 step smoke、新进程 resume 到 12 step。
5. 启动或恢复单 adapter A2S 三阶段训练。
6. 每阶段只跑 512 canary；Phase III 通过后跑一次完整 validation、Bench 与 clean test。

`00`-`11` notebook 均为历史实验或仓库辅助工具，不再扩建；`16_router_colab.ipynb` 不是
当前计划。提交 notebook 前清理输出、登录状态和个人 token，所有密钥只通过 Colab Secret。
