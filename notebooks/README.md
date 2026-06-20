# notebooks

存放 Colab 优先的 notebook。正式 notebook 放在本目录，`colab/` 不作为主工程目录使用。

规划：

- `00_clone_github_colab.ipynb`：挂载 Google Drive，clone/update GitHub 工程，并打印 commit 与关键修复标记。
- `00clonegithub.ipynb`：兼容旧命名，内容与 `00_clone_github_colab.ipynb` 保持一致。
- `01_baseline_colab.ipynb`：Qwen3-ASR-1.7B baseline smoke 推理与 WER/CER 评测。
- `02_mvp_150_eval_colab.ipynb`：读取本地生成并上传到 Drive 的 150 条 MVP 评测集，运行 Qwen3-ASR baseline 推理和场景级 WER/CER。
- `03_train_lora_colab.ipynb`：执行 Qwen3-ASR 模块探测、LoRA target 候选导出和 Unsloth 兼容性检查；Unsloth 已验证不兼容，后续 smoke training 走 Transformers + PEFT。
- `02_make_dataset_colab.ipynb`
- `04_eval_colab.ipynb`
- `05_router_colab.ipynb`

规则：

- notebook 必须能从 Google Drive 读取输入和写出结果。
- 关键参数应来自 `configs/`。
- 提交到仓库前应清理执行输出、登录 widget 状态和个人 token。
- 每个关键单元应有中文注释，说明输入、输出和失败时应检查什么。
