# Gemma 4 Robust ASR

这是一个基于 Gemma 4 12B 的独立鲁棒 ASR 项目。

项目目标是完成一个类似 Mega-ASR 能力形态的产品雏形：具备鲁棒 ASR LoRA、音频质量 router、统一推理入口、数据增强管线、评测体系和发布文档。但本项目不以 Mega-ASR 代码作为实现底座，所有新功能都应基于 Gemma 4 的真实 API 和我们自己的工程结构开发。

## 文档入口

- [项目方案](docs/gemma4-mega-asr/README.md)
- [路线图总览](docs/gemma4-mega-asr/roadmap/OVERVIEW.md)
- [执行路线图](docs/gemma4-mega-asr/roadmap/README.md)
- [文档验收与追踪矩阵](docs/gemma4-mega-asr/07_document_acceptance.md)
- [协作规范](AGENTS.md)

## 参考工程

原 Mega-ASR 上游工程应放在本地忽略目录：

- `references/mega-asr-upstream/`

该目录只用于查阅和对照，不作为新工程运行时依赖，不在其中继续开发 Gemma 4 功能。`references/` 已被 `.gitignore` 排除，不进入 git。

## 下一步

按路线图执行：

1. [01 独立项目骨架](docs/gemma4-mega-asr/roadmap/01_project_scaffold.md)
2. [02 Baseline 评估](docs/gemma4-mega-asr/roadmap/02_baseline_eval.md)
3. [03 数据 MVP](docs/gemma4-mega-asr/roadmap/03_data_mvp.md)
