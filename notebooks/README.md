# notebooks

存放 Colab 优先的 notebook。正式 notebook 放在本目录，`colab/` 不作为主工程目录使用。

规划：

- `01_baseline_colab.ipynb`：Gemma 4 12B baseline smoke 推理与 WER/CER 评测。
- `02_make_dataset_colab.ipynb`
- `03_train_lora_colab.ipynb`
- `04_eval_colab.ipynb`
- `05_router_colab.ipynb`

规则：

- notebook 必须能从 Google Drive 读取输入和写出结果。
- 关键参数应来自 `configs/`。
- 提交到仓库前应清理执行输出、登录 widget 状态和个人 token。
- 每个关键单元应有中文注释，说明输入、输出和失败时应检查什么。
