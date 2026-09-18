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
- 推理可用四进程按 manifest shard 并行，合并后必须按 `sample_id` 去重并恢复原顺序。
- 随机种子：`20260722`；改 seed 必须生成新的 run_id。
- 项目根目录：`/data/mega-asr`；禁止读取 `/data/mini-k3` 的环境、数据或 checkpoint。
- 不使用 FlashAttention-2；不使用 Mega-ASR 私有 wrapper、训练入口或 target 规则。

每次运行必须写 `environment.json`，至少包含 Python、PyTorch、CUDA、GPU、qwen-asr、Transformers、
PEFT、datasets、Git commit、model revision、dtype、attention、world size 和 seed。

## 2. 方法范围与阶段关系

| 阶段 | 起始模型 | 训练输入 | 输出 | 是否必做 |
|---|---|---|---|---|
| SFT | base | audio + gold text | `sft_release` adapter | 必做 |
| DPO | `sft_release` | audio + chosen/rejected preference pair | `dpo_release` adapter | 必做 |
| RL | `dpo_release` | audio + reference text + sampled rollouts/reward | `rl_release` adapter | 必做 |

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
scenario 不能跨分区。clean 数据使用官方 train/validation split；不得从混合池重新切分。

### 3.2 完整后训练数据角色与数量

以下是 full run 的目标分区，所有角色由固定 seed 和 source identity hash 生成，角色之间不得重叠：

| 角色 | Robust degraded | English clean | Chinese clean | 进入阶段 |
|---|---:|---:|---:|---|
| `sft_train` | 120,000 | 16,000 | 16,000 | SFT |
| `dpo_pool` | 16,000 | 2,000 | 2,000 | 生成 DPO pairs |
| `rl_pool` | 16,000 | 1,000 | 1,000 | RL rollout/reward |
| `validation` | 8,000 | 1,000 | 1,000 | 所有阶段固定评测 |
| `bench_test` | 5,000 | — | — | 只做最终评测 |

SFT pilot 从 `sft_train` 取 5,000 robust + 1,000 English clean + 1,000 Chinese clean。DPO pilot
从 `dpo_pool` 取 2,000 robust + 500 English clean + 500 Chinese clean。RL pilot 从 `rl_pool` 取
2,000 robust + 500 English clean + 500 Chinese clean。pilot 通过后才生成 full 规模。

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

DPO pairs 从 `dpo_pool` 生成，不能使用 validation 或 Bench 音频。对每条 audio 生成至少两个候选：
base prediction、SFT prediction，必要时加入受控的 deletion/repetition/normalization negative。用
gold transcript 计算语言对应的 WER/CER，只保留 chosen 与 rejected 不相同且分数有严格差异的 pair：

- `chosen`：错误率较低、非空、非明显重复的候选；
- `rejected`：错误率较高或触发失败标签的候选；
- gold transcript 只用于离线构造和审计，不作为 DPO prompt 的额外输入；
- ties、两边都为空、音频错误和 source leakage 行进入 rejects，不进入 DPO train。

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
  "source_dataset": "dataset-id",
  "source_revision": "revision",
  "source_utterance_id": "source-id",
  "audio_sha256": "sha256",
  "seed": 20260722
}
```

DPO full 目标为至少 20,000 pairs（16,000 robust、2,000 English clean、2,000 Chinese clean），另有
不重叠的 2,000 pair preference validation。DPO 训练不得读取 `chosen_error_rate`、`rejected_error_rate`
或 gold transcript 字段；这些字段只用于构造审计和 gate。

## 5. RL reward 与 rollout 合同

RL 从 `dpo_release` 开始，使用 `rl_pool` 的音频和 gold reference 计算序列级 reward。RL 不使用
Bench/test。第一版采用带 reference policy KL 约束的 group policy optimization，初始每条音频每卡
生成 2 个候选，reference policy 固定为 `dpo_release` 的冻结副本。

reward 配置必须冻结并写入 `reward_config.yaml`：

```text
wer/cer reward = 1 - min(language_error_rate, 1.0)
empty penalty = -0.25
repeat penalty = -0.25
too_long penalty = -0.15
hallucination penalty = -0.25
final reward = clip(sum, -1.0, 1.0)
```

其中 `wer/cer reward` 按语言选择 WER 或 CER；`hallucination`、`repeat`、`too_long` 的检测规则必须
在代码和测试中固定。每个 rollout 必须记录：

```json
{
  "sample_id": "stable-id",
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

RL full 目标为至少 18,000 audio（16,000 robust、1,000 English clean、1,000 Chinese clean），另有
不重叠的 rollout validation。RL 训练输出必须同时保存 rollout JSONL、reward summary、KL summary 和
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

### E0：合同冻结

输入：本文档、架构/数据/测试文档、固定 model revision 和 seed。

输出：SFT/DPO/RL 配置、schema fixtures、run_id 规则、目录结构。

通过标准：所有字段、数据源、方法范围、reward、阈值和停止条件已经写入配置或测试；没有隐式默认值。

### E1：数据 smoke 与 base baseline

输入：四个 pinned source 的 metadata、smoke 音频和固定 base。

输出：SFT smoke manifest、DPO/RL schema fixture、base predictions、WER/CER、source report、rejects。

通过标准：音频可读、schema 完整、无重复 sample_id、无 source leakage；clean/degraded 至少各一条
成功推理；所有失败有 `error`。

### E2：SFT checkpoint smoke

输入：SFT smoke manifest、base model、SFT config。

输出：10 step checkpoint、恢复后的 12 step checkpoint、loss/gradient 日志和 `sft_smoke` adapter。

通过标准：无 OOM/NaN；global step 10→12 连续；optimizer、scheduler、RNG、config、world size 可恢复；
新进程能加载 adapter。

### E3：SFT pilot

输入：5,000 robust + 1,000 English clean + 1,000 Chinese clean。

输出：`sft_pilot` adapter、base/SFT predictions、metrics、failures、gate report。

通过标准：至少一个 degraded scenario 改善；clean 错误率绝对增加 ≤0.02；有效输出率 ≥0.95；失败率
增加 ≤0.05；无数据泄漏；adapter 可重新加载。

### E4：DPO pilot

输入：2,000 robust + 500 English clean + 500 Chinese clean 的 DPO pairs，以及不重叠 preference validation。

输出：`dpo_pilot` adapter、chosen/rejected 审计、preference accuracy、base/SFT/DPO predictions 和 gate。

通过标准：preference validation accuracy ≥0.55；至少一个 degraded scenario 相对 SFT 改善；clean
错误率相对 SFT 绝对增加 ≤0.02；有效输出率 ≥0.95；DPO train 没有读入 gold/error-rate 审计字段。

### E5：RL pilot

输入：2,000 robust + 500 English clean + 500 Chinese clean 的 RL pool、`dpo_pilot`、冻结 reference policy、
reward config 和 rollout config。

输出：`rl_pilot` adapter、rollout/reward/ KL JSONL、reward summary、base/SFT/DPO/RL predictions 和 gate。

通过标准：held-out mean reward 相对冻结 DPO reference 提升 ≥0.05；至少一个 degraded scenario 相对 DPO
改善；clean 错误率相对 DPO 绝对增加 ≤0.02；有效输出率 ≥0.95；KL、reward、gradient 均有限；所有
rollout 可追溯。

### E6：full SFT → full DPO → full RL

只有 E3、E4、E5 全部通过才允许执行 full。三个 full 阶段仍必须按顺序运行，每阶段保留上一个有效
adapter 和完整 gate。任何阶段失败都停止扩大规模并回退，不得跳过 DPO/RL 直接宣称完成完整后训练。

### E7：最终 release

输入：最终 RL adapter、固定 validation、5,000 Bench test。

输出：base/SFT/DPO/RL 四组 predictions、metrics、release adapter、processor、配置、manifest/source
hash、失败样本和最终 gate。

最终通过标准：四组结果可重算；RL 满足 E5 指标；最终 clean/degraded gate 通过；新进程可加载最终
adapter；没有 test leakage；文档、配置、日志和结果路径完整。

## 8. 停止、回滚和报告规则

以下任一情况立即停止当前阶段：FP16 非有限 loss/gradient、OOM、global batch 漂移、resume 不连续、
manifest/source leakage、chosen/rejected 无法审计、reward/梯度/KL 非有限、clean regression 超阈值、
有效输出率低于 0.95，或出现非官方 wrapper。

每次停止都保留最后有效 checkpoint、输入 hash、环境、命令、失败样本和 gate.json。报告必须明确写出
`passed`、`failed`、`blocked` 或 `not_applicable`，不得用“训练完成”覆盖未通过的 DPO/RL 阶段。
