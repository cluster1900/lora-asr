# Colab 训练方案

最后更新：2026-06-29

## 目标

在 Colab 上跑通 Qwen3-ASR-1.7B 的 baseline 到 LoRA 训练完整闭环。

## Colab 资源现实约束

Qwen3-ASR-1.7B 比超大模型更适合 Colab，但全参训练仍不现实。现实路径是先跑 baseline，再做 QLoRA 或 LoRA，小 batch、梯度累积、短音频。

推荐硬件：

- Colab Free：baseline 脚本、数据准备、极小 smoke test。
- Colab Pro T4/L4：小规模 QLoRA 实验。
- Colab Pro+ A100：第一版更稳定的 LoRA/QLoRA 训练和较大 batch 评测。
- 外部 GPU runtime：适合数小时以上训练。

## Google Drive 目录

```text
/content/drive/MyDrive/qwen3-asr/
  data/
    raw/
    augmented/
    jsonl/
  checkpoints/
    qwen3-asr-1.7b-lora/
  outputs/
    baseline/
    eval/
  logs/
```

## Notebook 00: 拉取或更新工程

名称：

- `notebooks/00_clone_github_colab.ipynb`

目的：

- 在新的 Colab Free runtime 中挂载 Google Drive。
- 如果 `/content/drive/MyDrive/qwen3-asr` 不存在，则从 GitHub clone 项目。
- 如果目录已存在且是 git 仓库，则执行 `fetch + pull --ff-only`。
- 打印当前本地 commit、远端 commit 和关键修复标记，避免只看到 `Already up to date` 却无法判断代码版本。

为什么需要单独 notebook：

- Colab 的 Drive 目录会跨 runtime 保留，代码可能来自旧 clone。
- `git pull` 输出 `Already up to date` 只说明当前本地分支相对远端没有新提交，不说明当前 notebook 是否拿到了我们期望的修复。
- 训练 notebook 失败时，必须先确认 Drive 中的工程代码版本，再继续排查模型或依赖问题。

输入：

- GitHub 仓库：`https://github.com/cluster1900/lora-asr.git`
- 分支：`main`
- Drive 项目目录：`/content/drive/MyDrive/qwen3-asr`

输出：

- Drive 中的项目目录。
- 当前 `HEAD` 短 hash。
- 最新一条 commit message。
- 关键文件存在性和关键字符串检查结果。

测试标准：

- 空 Drive 项目目录时，可以 clone 到 `/content/drive/MyDrive/qwen3-asr`。
- 已存在 git 仓库时，可以 fast-forward 更新。
- notebook 必须打印 `git rev-parse --short HEAD` 和 `git log -1 --oneline`。
- notebook 必须检查 `resolve_training_model`、`target_root_prefix`、`gradient_checkpointing: false` 等关键修复标记。

验收标准：

- 执行完成后显示“版本验收通过”。
- 若 Drive 中代码不是 git 仓库，必须明确报错并提示用户先确认目录处理方式。
- 默认不得丢弃用户在 Drive 仓库里的本地修改；只有手动设置 `FORCE_RESET=True` 才允许强制对齐远端。

## Notebook 00B: GitHub 输出提交与推送

名称：

- `notebooks/00_github_commit_push_colab.ipynb`

目的：

- 单独提交并推送 Colab 产生的受控实验输出，例如 `outputs/`、`data/`、`checkpoints/` 中需要保留的结果。
- 避免把 clone/update 和 commit/push 混在同一个 notebook 中。
- 解决 Colab 中 HTTPS push 报 `could not read Username` 的授权问题。

背景：

- Colab 训练会在 Google Drive 项目目录中产生 prediction、metrics、error analysis、checkpoint metadata 等输出。
- 这些输出需要有选择地记录到 GitHub，方便在本地和后续 notebook 中复盘。
- Colab 不能交互式输入 GitHub 用户名和密码，因此直接 `git push origin main` 可能报 `could not read Username`。

授权方式：

1. 在 GitHub 创建 fine-grained personal access token，至少给目标仓库 Contents read/write 权限。
2. 在 Colab 左侧 Secrets 中新增 `GITHUB_TOKEN`，值为该 token。
3. Notebook 中通过 `google.colab.userdata.get("GITHUB_TOKEN")` 读取 token。
4. push 时临时使用带 token 的 HTTPS URL；不要执行 `git remote set-url` 写入 token。

安全要求：

- `COMMIT_AND_PUSH_OUTPUTS` 默认为 `False`。
- 提交前必须打印 `git status --short`。
- 默认提交路径只包含受控输出和文档相关目录，不提交 `.env`、`references/` 或私有文件。
- 默认不强制添加 `.gitignore` 忽略的模型权重；如确实需要提交 ignored 文件，必须手动设置 `FORCE_ADD_IGNORED=True`。
- 没有 staged changes 时跳过 commit/push。

验收标准：

- 未设置 token 或开关未打开时，不会产生 commit。
- 设置 `COMMIT_AND_PUSH_OUTPUTS=True` 且有 staged changes 时，能创建 commit 并 push 到 `origin/main`。
- notebook 输出不打印完整 token。

## Notebook 01: Baseline

名称：

- `notebooks/01_baseline_colab.ipynb`

目的：

- 加载 `Qwen/Qwen3-ASR-1.7B`。
- 在小评测集上运行 Qwen3-ASR `transcribe`。
- 计算 WER/CER。
- 保存预测结果。

主要步骤：

1. 挂载 Google Drive。
2. 安装依赖。
3. 登录 Hugging Face，如果需要。
4. 加载 Qwen3-ASR model。
5. 对每条音频生成转写。
6. 归一化预测。
7. 计算 WER/CER。

输出：

- `outputs/baseline/predictions.jsonl`
- `outputs/baseline/metrics.json`

## Notebook 02: 数据构建

名称：

- `notebooks/02_make_dataset_colab.ipynb`

目的：

- 下载或挂载 clean speech。
- 生成 degraded audio。
- 构建 train/val/test JSONL。

输出：

- `data/jsonl/train.jsonl`
- `data/jsonl/val.jsonl`
- `data/jsonl/test.jsonl`

## Notebook 03: LoRA 训练

名称：

- `notebooks/03_train_lora_colab.ipynb`

目的：

- 先完成 Qwen3-ASR 模块探测和 LoRA target 候选导出。
- 再跑 20 step Transformers + PEFT smoke training，确认训练链路可用。

当前 notebook 覆盖训练前探测和 Transformers + PEFT smoke training。Qwen3-ASR
官方 `qwen-asr` wrapper 的内部模块结构需要以当前环境真实加载结果为准，不能
复用 Mega-ASR 或其他工程的 target 规则。

`05C` smoke training 已完成 20 step 验收。正式 LoRA MVP 从 `05D` 开始，
由 `notebooks/04_train_lora_mvp_colab.ipynb` 执行。

历史 Unsloth 兼容性检查结果：

- `compatible=false`。
- 失败原因是 Transformers AutoConfig 不识别 `model_type=qwen3_asr`。
- 后续训练入口回退到 Transformers + PEFT，不继续在当前 MVP 中调试 Unsloth 依赖。
- 兼容性结论保留在 `outputs/lora_probe/qwen3_asr_1_7b/unsloth_compatibility.json`；当前 notebook 不再重复执行已知失败的 Unsloth 安装和检查。

依赖注意事项：

- 当前 notebook 固定 `qwen-asr==0.0.6`、`transformers==4.57.6`、`accelerate==1.12.0`，避免破坏 Qwen3-ASR 官方 wrapper 已验证过的依赖组合。
- Colab 预装包中 `pandas`、`requests` 容易被升级到冲突版本，因此 notebook 固定 `pandas==2.2.2`、`requests==2.32.4`。
- PEFT 注入 LoRA 时会探测 `torchao`。部分 Colab runtime 预装 `torchao==0.10.0`，而当前 PEFT 只支持 `torchao>0.16.0`，会在 `get_peft_model()` 阶段报错。当前 notebook 默认卸载 `torchao`，因为本 smoke training 不依赖 torchao。
- 训练 manifest 不包含音频本体；执行 LoRA smoke training 前，`data/mvp_eval/audio/` 必须已经同步到 Google Drive 项目目录。执行正式 LoRA MVP 前，应先运行 `scripts/create_lora_mvp_dataset.py` 生成 `data/lora_mvp/audio/` 和 train/val manifest。
- 如果安装后出现 pip resolver warning，先确认目标脚本是否能运行，再决定是否重启 runtime；warning 不一定等于 cell 失败。

初始配置：

```yaml
model_id: Qwen/Qwen3-ASR-1.7B
quantization: 4bit
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
batch_size: 1
gradient_accumulation_steps: 16
learning_rate: 2e-5
epochs: 1
max_audio_seconds: 20
max_new_tokens: 256
gradient_checkpointing: false
```

smoke 阶段默认关闭 gradient checkpointing。当前 LoRA target 位于 audio tower，
而 Qwen3-ASR 不是普通文本 LLM；在 k-bit PEFT 准备阶段启用 checkpointing 可能
引入自定义架构兼容问题。先用 4bit + batch size 1 跑通训练闭环，再决定是否单独
测试 checkpointing。

训练阶段：

0. Probe：加载模型并导出 `named_modules()`、LoRA target 候选和摘要。
1. PEFT compatibility：按正则精确匹配 99 个 audio tower target。
2. Smoke test：20 条样本，5-20 steps，当前入口为 `train/train_qwen3_asr_lora.py`。
3. MVP bootstrap train：交给 `04_train_lora_mvp_colab.ipynb`。
4. Scenario-balanced train：按声学场景做加权采样，后续再扩到真实数据。

输出：

- `outputs/lora_probe/qwen3_asr_1_7b/module_snapshot.json`
- `outputs/lora_probe/qwen3_asr_1_7b/module_summary.csv`
- `outputs/lora_probe/qwen3_asr_1_7b/lora_target_candidates.json`
- `outputs/lora_probe/qwen3_asr_1_7b/lora_target_candidates.md`
- `outputs/lora_probe/qwen3_asr_1_7b/unsloth_compatibility.json`
- `checkpoints/qwen3-asr-1.7b-lora/target_modules.json`
- `checkpoints/qwen3-asr-1.7b-lora/training_config.json`
- `checkpoints/qwen3-asr-1.7b-lora/loss_log.jsonl`
- `checkpoints/qwen3-asr-1.7b-lora/summary.json`
- `checkpoints/qwen3-asr-1.7b-lora/adapter/`

正式 LoRA MVP bootstrap 输出：

- `data/jsonl/lora_mvp_train.local.jsonl`
- `data/jsonl/lora_mvp_val.local.jsonl`
- `data/jsonl/lora_mvp_stats.local.json`
- `checkpoints/qwen3-asr-1.7b-lora-mvp/target_modules.json`
- `checkpoints/qwen3-asr-1.7b-lora-mvp/training_config.json`
- `checkpoints/qwen3-asr-1.7b-lora-mvp/loss_log.jsonl`
- `checkpoints/qwen3-asr-1.7b-lora-mvp/summary.json`
- `checkpoints/qwen3-asr-1.7b-lora-mvp/adapter/`

## Notebook 04: LoRA MVP Bootstrap 训练

名称：

- `notebooks/04_train_lora_mvp_colab.ipynb`

目的：

- 使用独立 `data/jsonl/lora_mvp_train.local.jsonl` 启动正式 LoRA MVP bootstrap 训练。
- 先执行 preflight，确认模型加载、音频路径、99 个 LoRA target 和输出目录都正常。
- 再跑默认 600 step 训练，产出 `checkpoints/qwen3-asr-1.7b-lora-mvp/`。

输入：

- `configs/train/qwen3_asr_lora_mvp_train.yaml`
- `data/jsonl/lora_mvp_train.local.jsonl`
- `data/jsonl/lora_mvp_val.local.jsonl`
- `data/lora_mvp/audio/`

输出：

- `checkpoints/qwen3-asr-1.7b-lora-mvp/target_modules.json`
- `checkpoints/qwen3-asr-1.7b-lora-mvp/training_config.json`
- `checkpoints/qwen3-asr-1.7b-lora-mvp/loss_log.jsonl`
- `checkpoints/qwen3-asr-1.7b-lora-mvp/summary.json`
- `checkpoints/qwen3-asr-1.7b-lora-mvp/adapter/`
- `checkpoints/qwen3-asr-1.7b-lora-mvp/processor/`

通过标准：

- preflight `summary.json` 中 `status=preflight_ok`。
- 正式训练 `summary.json` 中 `status=trained`。
- `loss_log.jsonl` 行数等于默认 `MAX_STEPS=600`。
- target count 等于 99。
- adapter 和 processor 均已保存。

## Notebook 05: LoRA MVP 评测

名称：

- `notebooks/05_eval_lora_mvp_colab.ipynb`

目的：

- 加载 `checkpoints/qwen3-asr-1.7b-lora-mvp/adapter/`。
- 在固定 MVP 150 held-out test 上运行 LoRA always-on 推理。
- 复用 baseline 同一套 WER/CER 与错误分析脚本，产出 base-vs-LoRA 对比。
- 量化 clean regression，并记录 noise/reverb、dropout/far_field 场景结果。

背景：

- `04_train_lora_mvp_colab.ipynb` 只证明训练完成并保存 adapter。
- LoRA 是否有效必须看 held-out test WER/CER，而不能只看 loss。
- 第一版先评估 always-on LoRA；只有确认 LoRA 相对当前公平 base 有足够收益且
  clean regression 可接受后，才进入 router。当前公平 base 口径为 4bit
  base recheck。

输入：

- `data/jsonl/baseline_mvp_150.local.jsonl`
- `checkpoints/qwen3-asr-1.7b-lora-mvp/adapter/`
- `outputs/base_recheck_mvp_150/metrics.qwen3_asr_base_recheck.mvp_150.json`
- `outputs/baseline_mvp_150/metrics.qwen3_asr_base.mvp_150.json`，仅作为历史 base 参考

指标：

- overall WER/CER。
- scenario-level WER/CER。
- clean regression。
- empty output rate。
- long hallucination rate。
- latency per audio second。

输出：

- `outputs/lora_mvp_eval/predictions.qwen3_asr_lora_mvp.mvp_150.jsonl`
- `outputs/lora_mvp_eval/predictions.qwen3_asr_lora_mvp.mvp_150.scored.jsonl`
- `outputs/lora_mvp_eval/metrics.qwen3_asr_lora_mvp.mvp_150.json`
- `outputs/lora_mvp_eval/metrics_by_scenario.qwen3_asr_lora_mvp.mvp_150.csv`
- `outputs/lora_mvp_eval/error_analysis/`

通过标准：

- mini LoRA inference 至少跑通 2 条样本。
- full LoRA inference 产出 150 行 prediction JSONL。
- WER/CER 评测与错误分析脚本执行完成。
- notebook 打印 base 与 LoRA 的 scenario-level 对比表。

验收标准：

- noise 或 reverb 至少一个场景相对 base 改善。
- clean WER 相对 base 的退化被量化，第一版警戒线为相对退化不超过 5%。
- dropout 与 far_field 作为观察场景完整记录。

## Notebook 06: LoRA MVP v2 快速 ablation

名称：

- `notebooks/06_train_lora_mvp_v2_colab.ipynb`

目的：

- 在 v1 LoRA 未通过 held-out 验收后，快速定位训练策略问题。
- 默认运行 attention-only、noise/reverb-only、低学习率、150 step 的 v2 短跑。
- 训练采样按 `scenario + text_length_bucket` 均衡轮转，确保 short/long 音频都被覆盖。
- 训练完成后立即在 MVP 150 held-out test 上评测，并对比 base、v1 和 v2。

通过标准：

- preflight target count 等于 96。
- loss 有限，adapter 可保存。
- loss log 同时覆盖 noise/reverb 和 short/long。
- held-out eval 生成 prediction、scored、metrics、scenario CSV 和错误分析。

验收标准：

- noise 或 reverb 至少一个场景相对 base 改善。
- clean regression 被量化。
- 如果 v2 仍未达到相对 4bit base recheck 的明确改善门槛，继续做 target/data/lr
  ablation，不进入 router。

当前执行结果：

- v2 已完成 150 step 训练，target count 为 96，adapter 和 processor 均已保存。
- loss log 覆盖 noise/reverb 和 short/long：noise long 38、noise short 38、reverb long 37、reverb short 37。
- held-out MVP 150 推理和评测已完成，empty output rate 为 0。
- v2 相比 v1 在 dropout 和 far_field 上略有改善，clean 与 v1 持平；按历史
  base 口径看 noise/reverb 仍更差，但 base recheck 证明历史 base 与当前
  4bit LoRA 评测口径不一致。
- 按 4bit base recheck 口径，v2 对 noise+reverb 有约 2.52% 相对改善，仍未
  达到第一版 10% 改善门槛，因此继续保留 router 暂停状态。

v2 结论对下一轮训练的影响：

- 单纯减少 step、降低学习率、移除 clean 样本和移除 speech projection 让 v2
  比 v1 更保守，但并没有带来更强 target 场景收益。
- 后续快速迭代应继续保留固定 held-out MVP 150，并优先测试 v1 风格 target、
  更早停止点、audio tower 后层 attention-only、训练目标格式和数据难度，而不是直接扩大训练规模。
- 每一轮仍必须输出 historical base、base recheck、v1 和当前版本的 scenario-level
  对比，避免只看 loss 或旧 base 判断训练成功。

## Base Recheck: MVP 150 复核基线

名称：

- `scripts/run_qwen3_asr_base_recheck.py`

目的：

- 在继续 LoRA ablation 前，重新跑一次 Qwen3-ASR base。
- 使用与 LoRA v1/v2 评测一致的 manifest、音频、dtype、device_map、max_new_tokens
  和量化参数。
- 验证历史 base 指标是否可信，避免把 base 环境差异误判为 LoRA 效果。

推荐 Colab 命令：

```bash
python scripts/run_qwen3_asr_base_recheck.py \
  --manifest data/jsonl/baseline_mvp_150.local.jsonl \
  --audio-root /content/drive/MyDrive/qwen3-asr \
  --output-dir outputs/base_recheck_mvp_150 \
  --model-id Qwen/Qwen3-ASR-1.7B \
  --dtype float16 \
  --device-map cuda:0 \
  --quantization 4bit \
  --max-inference-batch-size 1 \
  --max-new-tokens 128 \
  --language English \
  --compare-metrics outputs/baseline_mvp_150/metrics.qwen3_asr_base.mvp_150.json \
  --compare-metrics outputs/lora_mvp_eval/metrics.qwen3_asr_lora_mvp.mvp_150.json \
  --compare-metrics outputs/lora_mvp_v2_eval/metrics.qwen3_asr_lora_mvp_v2.mvp_150.json
```

输出：

- `outputs/base_recheck_mvp_150/metrics.qwen3_asr_base_recheck.mvp_150.json`
- `outputs/base_recheck_mvp_150/error_analysis/analysis_summary.json`
- `outputs/base_recheck_mvp_150/comparison.json`
- `outputs/base_recheck_mvp_150/comparison_by_scenario.csv`

验收标准：

- 新 base recheck 完整跑完 MVP 150。
- 与历史 base 的差异必须记录到 `00_progress.md`，再决定是否以 recheck base
  作为后续 LoRA 对比口径。

当前执行结果：

- base recheck 已完成，overall WER 为 0.550313，历史 base overall WER 为 0.483925。
- noise/reverb/far_field 上 base recheck 明显差于历史 base，说明旧 base 不再适合作为
  当前 4bit LoRA 的公平对照。
- 后续 LoRA 对比以 base recheck 为主，historical base 只作为旧环境参考。

## Notebook 07: LoRA MVP v3 Target-Focus

名称：

- `notebooks/07_train_lora_mvp_v3_colab.ipynb`

目的：

- 在确认 LoRA 相对 4bit base recheck 有弱收益后，继续把 noise/reverb 改善推到
  10% 以上。
- 复用 v1 更有效的 target 组合：audio tower attention + speech projection，共 99 个 target。
- 去掉 clean 训练样本，聚焦 noise/reverb，同时使用 `scenario + text_length_bucket`
  均衡轮转，避免长短样本失衡。

默认配置：

- `configs/train/qwen3_asr_lora_mvp_v3_target_focus.yaml`
- output：`checkpoints/qwen3-asr-1.7b-lora-mvp-v3-target-focus`
- eval output：`outputs/lora_mvp_v3_eval`
- target count：99
- scenario filter：noise,reverb
- learning rate：2e-5
- max steps：450
- sampling：`scenario_bucket_round_robin`
- base 对照：`outputs/base_recheck_mvp_150/metrics.qwen3_asr_base_recheck.mvp_150.json`

验收标准：

- preflight target count 等于 99。
- loss log 覆盖 noise/reverb 的 short 和 long。
- held-out MVP 150 推理、WER/CER 和错误分析完成。
- noise 或 reverb 至少一个场景相对 4bit base recheck 接近或达到 10% 相对 WER 改善。
- clean WER 不相对 base recheck 退化超过 5%。
- dropout/far_field 必须记录；若仍不改善，继续暂停 router。

## Notebook 07: Router

名称：

- `notebooks/08_router_colab.ipynb`

目的：

- 训练并校准 clean/degraded 音频分类器。

输出：

- router checkpoint。
- threshold config。
- routing evaluation report。

## 训练目标格式

baseline 推理不使用聊天 prompt，而是使用 `Qwen3ASRModel.transcribe(audio=..., language=...)`。训练阶段使用底层 forward + labels，因此需要显式构造 prompt 和 answer mask。

当前 smoke training target 格式：

```text
language English<asr_text>THE TRANSCRIPT TEXT
```

`labels` 中 prompt 和 padding 全部置为 `-100`，只有 `THE TRANSCRIPT TEXT` 和结束 token 参与 loss。后续仍可对比纯转写目标：

```text
THE TRANSCRIPT TEXT
```

最终格式由验证集 WER、空输出率、重复输出率和 clean regression 决定。

## Checkpoint 策略

- 每 200-500 optimizer steps 保存一次。
- Colab 中保留最近 3 个 checkpoint。
- 最终 adapter 同步到 Google Drive。
- 每个 adapter 旁边写入 `training_config.json`。

## 提前停止条件

出现以下情况应提前停止：

- loss 变成 NaN。
- 输出大多为空。
- 输出出现大量重复模板文本。
- validation WER 连续两次变差。
- Colab runtime 接近超时且近期没有保存 checkpoint。

## 测试标准

- 每个 notebook 能从空 Colab runtime 按顺序执行关键单元。
- `00_clone_github_colab.ipynb` 必须能确认本地代码版本和关键修复标记。
- notebook 使用 Google Drive 根目录变量，不写死个人本地路径。
- baseline notebook 至少能对 1 条 clean 和 1 条 degraded 音频推理。
- 训练 notebook 必须先完成 5-20 step smoke test。
- 训练 notebook 必须在加载模型前验证 smoke manifest 的音频路径，缺失时输出缺失样例。
- 训练 notebook 的依赖 cell 必须处理 Colab 预装旧版 `torchao`，避免 PEFT LoRA 注入阶段失败。
- 评测 notebook 能读取 prediction JSONL 并输出 WER/CER。
- 失败信息要写入输出文件或日志，不能只显示在 notebook cell 中。

## 验收标准

- `00_clone_github_colab.ipynb` 完成 clone/update，并打印当前 commit 与版本验收结果。
- `01_baseline_colab.ipynb` 产出 baseline predictions 和 metrics。
- `02_make_dataset_colab.ipynb` 产出 train/val/test JSONL。
- `03_train_lora_colab.ipynb` 产出可加载 LoRA adapter。
- `04_train_lora_mvp_colab.ipynb` 产出正式 LoRA MVP adapter。
- `05_eval_lora_mvp_colab.ipynb` 产出 LoRA always-on overall 和 scenario-level 指标。
- `06_train_lora_mvp_v2_colab.ipynb` 产出 v2 ablation 指标。
- `07_router_colab.ipynb` 在 router 阶段产出阈值和分类指标。
- 所有 notebook 的输入、输出和依赖都能追溯到配置文件。
