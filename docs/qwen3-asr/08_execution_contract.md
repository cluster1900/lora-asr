# 执行合同：SFT、DPO、RL、数据与验收

本文档是开始编码和启动实验前的强制合同。完整后训练链路为：

```text
Qwen3-ASR base -> SFT LoRA -> DPO preference optimization -> RL reward optimization -> release
```

三种方法都属于正式目标。SFT、DPO、RL 必须分别有独立 manifest、配置、checkpoint、预测和 gate；
任何阶段缺失只能标记为 `blocked`，不能把前一阶段结果当作完整后训练结果。

## 1. 固定模型与 4 卡执行合同

- 基础模型：`Qwen/Qwen3-ASR-1.7B`。
- 固定 revision：`7278e1e70fe206f11671096ffdd38061171dd6e5`。
- 服务器：4 × Tesla V100-SXM2-32GB，单机 4 卡 DDP，四张卡必须同时参与当前阶段。
- 启动方式：`torchrun --standalone --nproc_per_node=4`；SFT、DPO、RL 依赖前一阶段产物，因此
  三阶段顺序执行，不能同时启动三个阶段。
- dtype：FP16；`bf16=false`、`tf32=false`；第一版 attention 为 `eager`。
- 训练初始 batch：每卡 micro batch 1，gradient accumulation 16，global batch `1 × 16 × 4 = 64`。
  任何调整都必须写入 resolved config 并重新计算 global batch。
- LoRA 目标 allowlist 固定为：`thinker.audio_tower.conv_out`、`thinker.audio_tower.proj1`、`thinker.audio_tower.proj2`，以及
  `thinker.model.layers.{0..27}.self_attn.{q_proj,k_proj,v_proj,o_proj}` 和
  `thinker.model.layers.{0..27}.mlp.{gate_proj,up_proj,down_proj}`；只接受运行时 `Linear`，禁止 norm/embed/lm_head/Conv2d，
  总计 199 个 target（projection 3 + decoder 196）；必须保存 canonical target map、数量和 `target_map_hash`。初始 LoRA 为 rank $r=16$、alpha $\alpha=32$、dropout $0.05$。
- 阶段权重生命周期：先 `adapter.save_pretrained()` 保存独立 adapter，再在模型副本上执行 `merge_and_unload()` 保存
  merged 权重；对固定样本验证 merge 前后输出一致。下一阶段从 merged 权重重新注入新 adapter，不能复用已 merge 的对象。
- 推理模式：支持单卡 batch 1 推理，或四进程按 manifest shard 并行（每进程单卡、batch 1），合并后必须按 `sample_id` 去重并恢复原顺序。
- 随机种子：`20260722`；改 seed 必须生成新的 run_id。
- 项目根目录：`/data/mega-asr`；禁止读取 `/data/mini-k3` 的环境、数据或 checkpoint。
- 不使用 FlashAttention-2；不使用 Mega-ASR 私有 wrapper、训练入口或 target 规则。

每次运行必须写 `environment.json`，至少包含 Python、PyTorch、CUDA、GPU、qwen-asr、Transformers、
PEFT、datasets、Git commit、model revision、dtype、attention、world size 和 seed。

## 2. 方法范围与阶段关系

| 阶段 | 起始模型 | 训练输入 | 输出 | 是否必做 |
|---|---|---|---|---|
| SFT | base | audio + gold text | `sft_release` adapter 及 merge 后权重 | 必做 |
| DPO | `sft_release` (merged) | audio + chosen/rejected preference pair | `dpo_release` adapter 及 merge 后权重 | 必做 |
| RL | `dpo_release` (merged) | audio + reference text + sampled rollouts/reward | `rl_release` adapter 及最终发布权重 | 必做 |

A2S、Router、Teacher 和量化训练不进入这条正式链路。若未来要加入，必须新增 RFC 和独立对照组。

## 3. 固定数据集与分区

### 3.1 原始公开来源

| source_dataset | 固定 revision | 使用 split | 用途 | 许可记录 |
|---|---|---|---|---|
| `zhifeixie/Voices-in-the-Wild-2M` | `a8a35d3319737190d6fd3d39157b258eaab35980` | pinned robust train/validation splits | degraded SFT、DPO、RL | 记录 source license 和原始字段 |
| `openslr/librispeech_asr` | `71cacbfb7e2354c4226d01e70d77d5fca3d04ba1` | `train.100`、`validation` | English clean SFT、DPO、RL | CC-BY-4.0 |
| `knoveleng/aishell1-mandarin` | `c6dde006238091dc7c81cf5888208140bba33cba` | `train`、`validation` | Chinese clean SFT、DPO、RL | Apache-2.0，保留 OpenSLR SLR33 来源 |
| `zhifeixie/Voices-in-the-Wild-Bench` | `788f5d72c6b0e9091b5c2e432370923b6f9f0660` | pinned test splits | 只做最终 evaluation | 不得进入任何训练阶段 |

Robust 数据按原始话语 identity 做固定 90/10 train/validation 分区；同一 source identity 的不同
scenario 不能跨分区。Clean 数据使用官方 train/validation split 严格隔离，不得从混合池切分：
官方 train split（LibriSpeech `train.100` 28,539 条、AISHELL-1 `train` 120,098 条）专供训练角色（`sft_train` 16k + `dpo_train_pool` 10k + `rl_train_pool` 1k = 27,000 条，LibriSpeech 留 1,539 条清洗缓冲）；
官方 validation split（LibriSpeech `validation` 约 2,703 条、AISHELL-1 `validation` 14,326 条）专供验证角色（`validation` 1,000 + `dpo_val_pool` 1,000 + `rl_val_pool` 200 = 2,200 条），两者天然隔离、绝对无泄漏。

### 3.2 完整后训练数据角色与数量

以下是 full run 的目标分区，所有角色由固定 seed 和 source identity hash 生成，角色之间不得重叠：

| 角色 | Robust degraded | English clean | Chinese clean | 进入阶段 |
|---|---:|---:|---:|---|
| `sft_train` | 120,000 | 16,000 | 16,000 | SFT 训练 |
| `dpo_train_pool` | 32,000 source → 16,000 pairs | 10,000 source → 2,000 pairs | 10,000 source → 2,000 pairs | 生成 DPO 训练 pairs，源音频与其他角色不重叠 |
| `dpo_val_pool` | 3,200 source → 1,600 pairs | 1,000 source → 200 pairs | 1,000 source → 200 pairs | 独立 DPO 验证 pairs，源音频与训练池不重叠 |
| `rl_train_pool` | 16,000 | 1,000 | 1,000 | RL rollout 训练 |
| `rl_val_pool` | 1,600 | 200 | 200 | 独立 RL rollout 验证 |
| `validation` | 8,000 | 1,000 | 1,000 | 所有阶段固定 ASR 评测 |
| `bench_test` | 5,000 | — | — | 只做最终评测 |

注：针对 Clean 语音（LibriSpeech/AISHELL-1），模型预测与 gold 文本重合率高导致平局率（tie rate）高，
因此 Clean 音频 source pool 放大至目标 pair 数的 5 倍，Robust source pool 至少放大至 2 倍；每条源音频可产生多个受控候选。
实际有效 pair 数必须达到表中目标，否则数据阶段失败。

SFT pilot 从 `sft_train` 取 5,000 robust + 1,000 English clean + 1,000 Chinese clean。DPO pilot
从 `dpo_train_pool` 取 2,000 robust + 500 English clean + 500 Chinese clean pairs，并从 `dpo_val_pool` 取 500 验证 pairs。RL pilot 从 `rl_train_pool` 取
2,000 robust + 500 English clean + 500 Chinese clean，并从 `rl_val_pool` 取 500 验证音频。pilot 通过后才生成 full 规模。

### 3.3 SFT manifest 输入

每行必须是 JSON object，至少包含：

```json
{
  "sample_id": "stable-id",
  "audio": "relative/or/absolute/path.wav",
  "text": "gold transcript",
  "language": "en|zh",
  "scenario": "clean|noise|reverb|far_field|dropout|...",
  "condition_group": "clean|degraded",
  "audio_origin": "real|synthetic",
  "source_dataset": "dataset-id",
  "source_revision": "40-char-revision",
  "source_split": "split-name",
  "source_index": 123,
  "source_utterance_id": "source-stable-id",
  "duration_s": 3.2,
  "license": "license-id",
  "seed": 20260722,
  "audio_sha256": "sha256"
}
```

`sample_id` 唯一；`audio` 存在且可解码；`text` 非空；时长 0.5–30 秒；source、split、identity 和
hash 可追溯。旧 `answer` 字段不属于新合同，只允许在一次适配步骤中转换成 `text`。

## 4. DPO preference 合同

DPO pairs 从 `dpo_train_pool` 与 `dpo_val_pool` 生成；禁止使用最终 `validation` role 或 `bench_test` 音频。`dpo_val_pool` 只生成 preference validation，不能进入 DPO train。对每条 audio 生成至少两个候选：
base prediction、SFT prediction，针对高准确率的 Clean 样本引入受控的 deletion/repetition/normalization negative。用
gold transcript 计算语言对应的 WER/CER，只保留 chosen 与 rejected 不相同且分数有严格差异的 pair：

- `chosen`：错误率较低、非空、非明显重复的候选；
- `rejected`：错误率较高或触发失败标签的候选；
- gold transcript 只用于离线构造和审计，不作为 DPO prompt 的额外输入；
- ties、两边都为空、音频错误和 source leakage 行进入 rejects，不进入 DPO train。Clean 语音通过放大 5 倍候选源确保过滤 ties 后足额达成目标。
- 参考模型与预计算优化（V100 显存减半）：参考策略 $\pi_{\text{ref}}$ 为冻结的 SFT 合并模型。支持并推荐在生成 DPO manifest 阶段离线预计算 `ref_chosen_logp` 与 `ref_rejected_logp`；同时记录 `ref_model_revision`、`ref_model_sha256`、`tokenizer_hash`、
  `chat_template_hash`、`logp_config_hash` 和 max token 配置。在包含预计算 logps 时，DPO 训练无需在 GPU 常驻参考模型，
  显存占用直降约 50%；任何 provenance hash 不匹配都必须拒绝该 manifest。

DPO manifest 每行至少包含：

```json
{
  "sample_id": "stable-id",
  "audio": "audio.wav",
  "language": "en|zh",
  "prompt": "ASR prompt",
  "chosen": "preferred transcript",
  "rejected": "dispreferred transcript",
  "preference_source": "base_vs_sft_gold_oracle_v1",
  "judge": "wer_cer_rule_v1",
  "chosen_error_rate": 0.10,
  "rejected_error_rate": 0.80,
  "ref_chosen_logp": -4.215,
  "ref_rejected_logp": -12.873,
  "ref_model_revision": "revision",
  "ref_model_sha256": "sha256",
  "tokenizer_hash": "sha256",
  "chat_template_hash": "sha256",
  "logp_config_hash": "sha256",
  "source_dataset": "dataset-id",
  "source_revision": "revision",
  "source_utterance_id": "source-id",
  "audio_sha256": "sha256",
  "seed": 20260722
}
```

DPO full 目标为至少 20,000 训练 pairs（16,000 robust、2,000 English clean、2,000 Chinese clean），以及独立
不重叠的 2,000 pair preference validation（来自 `dpo_val_pool`）。DPO 训练不得读取 `chosen_error_rate`、`rejected_error_rate`
或 gold transcript 字段；这些字段只用于构造审计和 gate。

## 5. RL reward 与 rollout 合同

RL 从 `dpo_release`（合并模型）开始，使用 `rl_train_pool` 与 `rl_val_pool` 的音频和 gold reference 计算序列级 reward。RL 不使用
Bench/test。第一版采用带 reference policy KL 正则项的 Group Relative Policy Optimization (GRPO)：

- 组大小与采样：同一 `sample_id` 必须生成完整 $G=4$ 个候选，使用 `group_id`、`group_size=4`、`rollout_rank` 标识；
  可在单卡生成 4 个，或四卡各生成 1 个后 all-gather，必须在计算 advantage 前完成组内聚合。采样参数推荐 `temperature=0.85`、
  `top_k=50`、`top_p=0.92`，并为每张卡和每个 Candidate 注入独立 RNG seed，确保声学与语言多样性，避免归一化后文本坍缩；组内候选奖励标准差 ≤ epsilon 的 group（如全对或全错同质组）其标准化优势置 0（不贡献策略梯度，保护 clean 稳定性）。鉴于 ASR 训练集包含约 45% 的清晰语音（Clean 错误率极低，4 次采样全部正确天然产生零奖励方差），单批次零奖励方差率监控上限设定为 80%（在动态种子下均值实测约 63.9%，单批次由于 N=64 二项抽样波动可达 ~76%）；若连续 2 步批次零方差率超过 80%（排除单批次抽样偶然性，确认采样分布严重坍缩或温度失效），必须在 `pipeline_state.json` 写入 `FAILED_ZERO_VARIANCE` 并抛出异常硬中止 RL 训练。门禁核验以全流程平均批次零方差率 ≤ 75% 为准。
- 四卡 Rollout 全量收集：所有 rank 必须各自记录 `rollouts_rank_{rank}.jsonl`，在保存 checkpoint 或训练结束时汇聚为完整的 `rollouts.jsonl`（预期 $60 \times 16 \times 4 \times 4 = 15,360$ 行），严禁仅落盘 Rank 0；每条记录必须包含 `sample_id`、`condition_group`、`group_id`、`group_size`、`rollout_rank`、`reward`、`advantage`、`kl_to_reference` 等全字段，并通过全量审计。
- 独立验证集统一口径：周期性验证必须统一评测全量 573 条 `rl_val_pool.jsonl`（禁止 Sub-25 抽样与全量混算）；在训练启动前（Step 0）必须强制评测初始 Policy / 冻结 Reference 底座的基准指标并写入 `loss_log.jsonl`（标注 `val_eval_scope: "Full Held-out"`），后续各 Step 的奖励提升严格以 $R_{\text{step}} - R_0$ 计算。
- GRPO 损失函数：遵循业内标准，对组内候选标准化优势 $A_i = \frac{r_i - \text{mean}(\{r\})}{\text{std}(\{r\}) + \epsilon}$，并在策略损失中加入针对冻结参考策略（`dpo_release`）的 KL 正则约束 $\beta D_{KL}(\pi_\theta || \pi_{\text{ref}})$。

reward 配置必须冻结并写入 `reward_config.yaml`：

```text
asr reward = 1 - min(language_error_rate, 1.0)
empty penalty = -0.25
repeat penalty = -0.25
too_long penalty = -0.15
hallucination penalty = -0.25
final reward = clip(sum, -1.0, 1.0)
```

其中 `asr reward` 按语言选择 WER 或 CER；`hallucination`、`repeat`、`too_long` 的检测规则必须
在代码和测试中固定。每个 rollout 必须记录：

```json
{
  "sample_id": "stable-id",
  "group_id": "stable-id:rollout-step",
  "group_size": 4,
  "rollout_rank": 0,
  "policy_checkpoint": "dpo_release",
  "rollout_seed": 20260722,
  "prediction": "sampled transcript",
  "language": "en|zh",
  "reference_error_rate": 0.40,
  "reward_components": {"asr": 0.60, "empty": 0.0, "repeat": 0.0, "too_long": 0.0, "hallucination": 0.0},
  "reward": 0.60,
  "kl_to_reference": 0.02,
  "error": ""
}
```

RL full 目标为至少 18,000 训练 audio（16,000 robust、1,000 English clean、1,000 Chinese clean），以及独立
不重叠的 2,000 rollout validation（来自 `rl_val_pool`）。RL 训练输出必须同时保存 rollout JSONL、reward summary、KL summary 和
policy checkpoint；缺少 rollout 或 reward 明细时，RL 阶段不算完成。

## 6. 统一输出合同

每个 run 位于 `/data/mega-asr/runs/<run_id>/`，至少包含：

- `resolved_config.yaml`：所有默认值解析后的配置；
- `environment.json`：环境、版本和 4 卡信息；
- `manifest_sha256.json`：输入 manifest 的 hash、行数、角色和 source revision；
- `pipeline_state.json`：阶段、global step、world size 和最后有效 checkpoint；
- `checkpoints/`：可重新加载的训练 checkpoint；
- `adapter/`：当前阶段和最终 PEFT adapter、processor；
- `predictions/`：base/SFT/DPO/RL 的逐条 JSONL；
- `rollouts/`：DPO pair 审计和 RL rollout/reward JSONL；
- `metrics/`：`metrics.json`、`by_language.csv`、`by_scenario.csv`、必要时 `by_cell.csv`；
- `failures.jsonl`：推理错误、空输出、重复、过长和 hallucination 样本；
- `gate.json`：每项门禁的 status、阈值、实际值和失败原因。

prediction 至少包含 `sample_id`、`text`、`prediction`、`language`、`scenario`、`model_id`、
`model_revision`、`dtype`、`attention`、`method`、`adapter_dir`、`error` 和 `infer_seconds`。
评测必须按 English WER、Chinese CER、language、scenario、condition_group 和 audio_origin 聚合。

## 7. 执行顺序与验收门禁

### E0：合同冻结（文档阶段）

输入：本文档、架构/数据/测试文档、固定 model revision 和 seed。

输出：SFT/DPO/RL 配置草案、schema fixtures、run_id 规则、目录结构。

通过标准：所有字段、数据源、方法范围、reward、阈值和停止条件已经写入配置或测试；没有隐式默认值。
E0 只冻结合同，不下载模型、不启动训练。

### E1：完整数据下载/物化与角色池（首个可执行步骤）

输入：四个 pinned source 的 metadata、下载镜像配置和许可证记录。

执行要求：先完整下载或物化本项目需要的原始音频、metadata 和 source index；原始文件不可覆盖。然后依次完成
完整性/许可证检查、音频标准化（可读、mono、16 kHz、PCM/WAV、0.5–30 秒）、文本/语言/场景规范化、source identity
与泄漏检查，再按固定 seed/source identity 创建 `sft_train`、`dpo_train_pool`、`dpo_val_pool`、`rl_train_pool`、
`rl_val_pool`、`validation` 和 `bench_test` 的 processed manifest。此阶段不得下载模型、运行 base inference、
训练 SFT/DPO/RL 或生成 model-dependent preference/rollout。

输出：raw source cache、processed 音频、角色 manifest、source/license report、rejects、原始/处理后 audio hash、
manifest hash、处理版本、`RAW_COMPLETE.json`、`PROCESSED_COMPLETE.json`、各角色 `COMPLETE.json` 和恢复日志。

通过标准：四个 source 均完整可追溯；所有角色配额和 source identity 隔离通过；处理后音频可读且满足采样率/声道/时长合同；
schema 完整、无重复 sample_id、无 train/validation/test leakage；每个角色的 `COMPLETE.json` 和总 `DATASET_COMPLETE.json`
存在。只有 `DATASET_COMPLETE.json` 存在后，才能进入 E2。

DPO pairs 和 RL rollouts 依赖模型输出，不能在 E1 伪造；E1 必须先把它们所需的 source audio pool
完整准备好，派生数据分别在 E5/E6 生成。

### E2：环境、模型加载与 base smoke

输入：E1 的完整角色池、固定 model revision、V100 环境合同。

输出：`environment.json`、FP16/eager 单 batch probe、128-row smoke manifest、base predictions、WER/CER、
source report 和 rejects。

通过标准：4 卡可见且 world size 正确；模型加载成功；clean/degraded 至少各一条成功推理；所有失败有
`error`；无 OOM/NaN；base 输出可恢复。

### E3：SFT checkpoint smoke

输入：SFT smoke manifest、base model、SFT config。

输出：10 step checkpoint、恢复后的 12 step checkpoint、loss/gradient 日志、`sft_smoke` adapter 和
merge 试产物。

通过标准：global step 10→12 连续；optimizer、scheduler、RNG、config、world size 可恢复；新进程能加载
adapter；`adapter.save_pretrained()` 与副本 `merge_and_unload()` 均成功。

### E4：SFT pilot

输入：5,000 robust + 1,000 English clean + 1,000 Chinese clean。

输出：`sft_pilot` adapter、合并后基础权重、base/SFT predictions、metrics、failures、gate report。

通过标准：至少一个 degraded scenario 改善；clean 错误率绝对增加 ≤0.02；有效输出率 ≥0.95；失败率
增加 ≤0.05；无数据泄漏；adapter 与 merged base 均可重新加载。

### E5：DPO pair 与 DPO pilot

输入：来自 `dpo_train_pool` 的 2,000 robust + 500 English clean + 500 Chinese clean DPO pairs，以及
来自 `dpo_val_pool` 的 500 对不重叠 preference validation；参考 logprob provenance 必须通过 hash 检查。

输出：`dpo_pilot` adapter、合并后基础权重、chosen/rejected 审计、preference accuracy、base/SFT/DPO
predictions 和 gate。

通过标准：preference validation accuracy ≥0.55；至少一个 degraded scenario 相对 SFT 改善；clean 相对
SFT 回退 ≤0.02，且相对 Base 累积回退 ≤0.025；有效输出率 ≥0.95；DPO 未读取 gold/error-rate 审计字段；
adapter 与 merged base 均可重新加载。

### E6：RL rollout 与 RL pilot

输入：来自 `rl_train_pool` 的 2,000 robust + 500 English clean + 500 Chinese clean、来自 `rl_val_pool`
的 500 条验证音频、`dpo_pilot` 合并底座、冻结 reference policy、reward config 和 rollout config（G=4）。

输出：`rl_pilot` adapter、最终合并权重、rollout/reward/KL JSONL、reward summary、base/SFT/DPO/RL predictions
和 gate。

通过标准：held-out mean reward 相对冻结 DPO reference 在全量验证集上提升 ≥0.05（严格同口径 Full Held-out 相比，缺失 Step 0 基线或口径不一致视为 FAILED）；全程平均零方差 group 比例 ≤75%；四卡 rollout 完整收集（15,360 行）且 audit 通过；至少一个 degraded scenario 相对 DPO 改善；clean 相对 DPO 回退 ≤0.02，且相对 Base 累积回退 ≤0.025；robust macro 相对 DPO 零恶化（≤ 0.0）；有效输出率 ≥0.95；KL、reward、gradient 均有限；最终 adapter 与 merged model 均可加载。

### E7：full SFT → full DPO → full RL

只有 E4、E5、E6 全部通过才允许执行 full。三个 full 阶段仍必须按顺序运行，每阶段保留上一个有效
adapter、合并权重和完整 gate。任何阶段失败都停止扩大规模并回退，不得跳过 DPO/RL 直接宣称完整后训练完成。

### E8：最终 release

输入：最终 RL adapter 与发布模型权重、固定 validation、5,000 Bench test。

输出：base/SFT/DPO/RL 四组 predictions、metrics、release adapter、发布模型、processor、配置、manifest/source
hash、失败样本和最终 gate。

最终通过标准：四组结果可重算；RL 满足 E6 指标；最终 clean/degraded gate 通过（含相对 Base 全局 clean
错误率累积绝对增加 ≤0.025 红线）；新进程可加载最终 adapter 与 merged 权重；没有 test leakage；文档、
配置、日志和结果路径完整。

## 8. 停止、回滚和报告规则

以下任一情况立即停止当前阶段：FP16 非有限 loss/gradient、OOM、global batch 漂移、resume 不连续、
manifest/source leakage、chosen/rejected 无法审计、reward/梯度/KL 非有限、clean regression 超阈值、
有效输出率低于 0.95，或出现非官方 wrapper。

每次停止都保留最后有效 checkpoint、输入 hash、环境、命令、失败样本和 gate.json。报告必须明确写出
`passed`、`failed`、`blocked` 或 `not_applicable`，不得用“训练完成”覆盖未通过的 DPO/RL 阶段。
