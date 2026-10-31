# tests

## 目录职责

维护项目的合同测试与单元测试，覆盖 manifest 校验、数据 schema、推理输出、评测指标、训练 smoke、checkpoint 恢复、DPO 配对审计、RL 梯度与优势计算以及目录 README 同步。

## 当前文件清单

- `README.md`：本说明文档。
- `test_directory_readmes.py`：目录 README 结构与清单合同测试，验证所有受管理目录下的文件与文档要求一致。
- `test_zh_cer.py`：`zh-cer-v1` 归一化、编辑距离、对齐回溯和健康度标志。`score_pair` 仍只返回编辑总数。
- `test_eval_cer.py`：CER 聚合、WER 缺省向后兼容、锁定 `jieba==0.42.1` + 内置词典 SHA-256 中文 WER（micro/macro/scenario 与推理错误行）、分词器名称/版本/词典哈希不符拒绝、`configs/inference/base_eval_wer.yaml` CLI 运行和重复预测失败。
- `test_run_inference.py`：缺失音频写入错误并继续处理后面的行。
- `test_phase0_support.py`：读回哈希和固定种子噪声。
- `test_phase0_lock.py`：可安装环境锁的 verify/install 拆分，以及错误 pin 被拒绝。
- `test_manifest.py`：Phase 1 的可复现构建、五类场景、拒收、正式配置 `insufficient_data`、夹具上的退出码 0、同源泄漏、官方切分保留、说话人跨 split 拒收、混用切分模式和 seed 不匹配。合成夹具不是训练集。
- `test_sample_aishell_pilot.py`：pilot 抽样的轮转、全角拉丁、削波、重复文本、配额不足和整包时长表。夹具不是训练集。
- `test_train_sft.py`：SFT 配置、解码器 LoRA 名称、token 加权 loss、EOS mask、短窗口、checkpoint 指针和 smoke 子集。不下载模型。
- `test_sft_release.py`：SFT release 只接收选出的 passed step，文本不一致时保持 `staged`。staged release 不能导出 hypotheses，续写必须是 manifest 前缀。不下载模型。
- `test_build_dpo_pairs.py`：DPO pair 的 chosen/rejected、tie、空输出、超 token、上限内长幻觉、跨 split 泄漏、坏音频和未发布 release。数量不足时退出码 3。不下载模型，不算 logprob。
- `test_strip_dpo_gap_pairs.py`：dropout 静音段剥离后的新 pair 目录。纯 gap 删除进入 `gap_ties.jsonl`，其他场景保持原文和 token 数。源哈希不符或输出目录名是冻结目录时不写文件。不下载模型，不算 logprob。
- `test_reference_logprob.py`：reference logprob 是序列和而不是 token 均值。误差线、缓存前缀和复核失败都不发布 pair。不下载模型。
- `test_dpo_contract.py`：DPO pair loss、preference 门禁、长度分桶、听音豁免、dropout 静音段剥离、无标点原位替换与单字编辑/参考 margin 过滤、直通式差异 token 梯度掩码、确定性多样本 dropout 锚切片 `dropout_anchor_step_slice`、`optim.scheduler_max_steps` 余弦周期解耦、结束时 Drive 同步、`configs/train/dpo_pilot.yaml`、`configs/train/dpo_dropout_probe.yaml`、`configs/train/dpo_gap_strip_probe.yaml`、`configs/train/dpo_pilot_nondropout.yaml`、`configs/train/dpo_pilot_aligned.yaml`、`configs/train/dpo_pilot_edit1_seq.yaml`、`configs/train/dpo_pilot_edit1_anchor.yaml`、`configs/train/dpo_pilot_edit1_anchor_stop40.yaml`、`configs/train/dpo_pilot_edit1_anchor_step40_h100.yaml` 和 `configs/train/dpo_pilot_edit1_anchor_step40_h100_a4.yaml`。dropout 锚只收 train dropout 的 `rejected_text`，权重超过 0.1 或和整句反传一起用时拒绝。停步配置只把 `max_steps` 改成 40；其他 pilot 写成 40 会被拒绝。不下载模型，不把处置改成已听。trainer 源码不调用 `strip_dropout_gap`。
- `test_dpo_release.py`：DPO release 只接收选出的 passed pilot step（如 step 40），核对 `preference.json` 与 `scored.jsonl`，拒绝 `risk_stop`、偏好未过线或重复写出；读回文本不一致时保持 `staged`，全部一致时转为 `published` 且不改动 payload hash，同时保留 `audit_disposition: unlistened`。不下载模型。
- `test_rl_contract.py`：Phase 4 RL（`group_pg_mixed_v1`）确定性候选种子、单候选 reward 与 `[-1, 1]` 截断、G=4 组有效性与截断/错误整组剔除、组均值中心化 advantage、全组同分零损失（含零 KL）、`k3` 逐 token 代理 KL 与固定 step 0 前缀词表条件 `raw_kl`、零信号窗口不更新 optimizer/scheduler、reference 权重 checksum 冻结校验、组内无标点等长替换差异 token 掩码 `aligned_group_mismatch_mask`、仅非零信号 group 累积窗口、`dropout` 父版本假设锚、`configs/train/rl_smoke.yaml`/`rl_pilot.yaml`/`rl_pilot_aligned.yaml` 配置解析以及 `train/rl_release.py` 打包读回校验。不下载模型。
- `test_phase5_benchmark.py`：Phase 5 封存 `test.jsonl` 四阶段（`base`/`sft`/`dpo`/`rl`）终局评测合同、未发布 release 拒绝、500 条 clean 测试源音频确定性重建 2500 条测试波形并与 `sha256.json` 验签（含跨架构 `noise` 缩放因子 `±8 ULP` 对齐与损坏文件拒绝）、按 `sample_id` 前缀断点续跑、`benchmark_summary.json` / `run_report.md` 汇总与 Drive 读回同步，以及 `notebooks/01_qwen3_asr_post_training_pipeline.ipynb` 主 notebook 结构合同。不下载模型。
- `test_alimeeting_ood_benchmark.py`：Phase 6 Track 2 AliMeeting Eval 只读 OOD 数据物化与四阶段评测合同测试，覆盖 Praat TextGrid 解析、跨说话人重叠剔除、副语言/拉丁词过滤、近场单声道与远场 8 通道 `ch0` 1:1 配对切片、会议分层确定性抽样、Drive 缓存验签恢复、`trainable=true`/未发布 release/`decode_eval_sha256` 失配拒绝、四阶段 CER + 锁定 `jieba==0.42.1` WER + 配对源簇 Bootstrap 95% CI 汇总及前缀续跑。不下载模型或公开语料。

## 主要输入输出

- **输入**：项目源代码、文档、mock 数据与测试夹具（fixtures）。
- **输出**：单元测试与契约测试通过状态，失败时输出明确断言原因。

## 使用入口

```bash
python3 -m unittest discover -s tests -v
python3 -m unittest tests.test_manifest tests.test_directory_readmes -v
```

## 维护要求

新增模块、修改接口或目录文件变更时，必须编写对应测试并在同一提交同步更新对应目录的 `README.md`。提交前必须确保测试全部通过。
