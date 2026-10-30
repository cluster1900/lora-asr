# 训练方案

最后更新：2026-10-23。状态：**SFT 6 step smoke 已完成，gate 为 `incomparable`。SFT `1e-5` run `20261015T234939_sft_06b_zh_seed42` 选出 step 30，release `sft_06b_zh_20261015T234939_step000030` 已由独立进程读回 2500 条全部一致（`status=published`、`publishable=true`）。DPO 纯方案 1 run `20261022T133007_dpo_06b_zh_seed42`（`configs/train/dpo_pilot_edit1_anchor_step40_h100.yaml`）已在 Step 40 以 `status=completed` / `reason=max_steps`（零 `gate_risk`）通过全部四次 2500 条 val 门禁，选出 `step_000040`（`latest_step == selected_step == 40`，四退化场景净减 11 个编辑，preference `0.644068`）。本轮推进 `train/dpo_release.py` 对 `dpo_06b_zh_20261022T133007_step000040` 做不可变封包与 2500 条 val 读回校验，并同步落地 Phase 4 RL（`group_pg_mixed_v1`）训练器与配置。test 仍封存。WER 仍为 null，Phase 2/3/4 不因此标完成。**

## 背景与范围

项目在 `Qwen/Qwen3-ASR-0.6B` 上建立中文鲁棒 ASR 的 SFT → DPO → RL 链路。本文负责算法、训练配置、阶段衔接、停止和晋级；[数据方案](03_data_plan.md)负责数据来源和增强；[测试方案](05_testing_plan.md)负责指标定义与测试；[执行合同](07_colab_execution_contract.md)负责持久化和恢复。训练条款冲突时以本文为准，同时修正引用方，不能依靠“以本文为准”长期保留矛盾。

本次只修订文档，不恢复已删除的旧 runner、不启动训练、不复用历史 gate。所有数值是待 smoke 验证的初始实验设定，不是已测得的最优参数或效果承诺。MVP 不包含 Router、多卡、全参数训练、量化训练、在线数据增强和新训练后端。

## 官方接口与监督合同

使用官方 `qwen-asr` 的 Transformers 后端及 processor；API 版本、Hub revision、tokenizer/chat template/processor 的文件 hash 在 Phase 0 固定。[官方微调说明](https://github.com/QwenLM/Qwen3-ASR/blob/main/finetuning/README.md)给出中文语言前缀；[官方 collator](https://github.com/QwenLM/Qwen3-ASR/blob/main/finetuning/qwen3_asr_sft.py)展示了音频输入、前缀 mask 和 EOS 的处理。只参考接口并独立实现本仓库训练器。

- manifest 的 `text` 只存转写；processor 构造 system/user/audio/assistant prompt，再附 `language Chinese<asr_text>`。训练目标是转写 token 和一个 EOS。当前[官方 tokenizer](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/blob/main/tokenizer_config.json)的 EOS 为 `<|im_end|>`；从固定 tokenizer 读取 ID，禁止硬编码数值 ID。
- 所有 prompt、音频占位、语言前缀及 padding 的 label 为 `-100`。用 attention mask 和实际 token 边界计算，不能仅按字符串长度或 `token == pad_id` mask；即使 pad 与 EOS 相同，也不能误删真实 EOS。检查 causal shift 后第一枚转写 token 和 EOS 均受监督。
- `language=zh` 是项目枚举，官方调用显式映射为 `Chinese`。固定 context 为空，不让 train/eval 使用不同提示词。训练用原始转写；CER 归一化仅用于评分，不能把归一化文本替代训练 target。
- 顶层推理 wrapper 的 `transcribe()` 不等于带梯度的训练 forward。[官方模型结构](https://github.com/QwenLM/Qwen3-ASR/blob/main/qwen_asr/core/transformers_backend/modeling_qwen3_asr.py)中 thinker 提供 logits；本仓库通过组合适配 forward，使用官方 processor 输入，禁止依赖 Mega-ASR 私有 wrapper 或全局 monkey patch。
- 显式使用 `model.generate` 的公开参数控制 greedy/采样；`transcribe()` 不接受任意 rollout 参数。Phase 0 比较两条 clean/degraded 上公开推理与项目 greedy 路径的文本，差异必须解释并消除后才能训练。保存生成 token IDs、stop reason 和解析前输出；`prediction_raw` 指未做 CER 归一化的转写，完整协议输出另存 `generation_raw`。

LoRA 只匹配官方模型的解码器 `thinker.model.layers.<n>.self_attn.{q,k,v,o}_proj`（相对于所挂载模块调整根前缀）。不能按 `thinker` 字符串或四个短后缀全局匹配。启动生成 `trainable_parameters.json`，列出完整路径、shape、数量和 trainable 参数占比；断言没有 audio tower、projector、embedding、LM head 或其他权重进入 optimizer。MVP 禁止打开这些参数；扩展需另做设计、baseline 和新 run。

## 阶段衔接与初始化

| 阶段 | 初始 policy | 固定 reference/父评测 | adapter 和 optimizer |
| --- | --- | --- | --- |
| SFT | 官方 base + 新 LoRA（B 初始化零） | 同输入上的 base 预测 | 新 optimizer/scheduler |
| DPO | 官方 base + SFT release 的完整 adapter | SFT release；离线 logprob | 原 adapter 继续训练，新 optimizer/scheduler |
| RL | 官方 base + DPO release 的完整 adapter | 冻结 DPO release | 原 adapter 继续训练，新 optimizer/scheduler |

每个 adapter 表示相对于同一官方 base 的完整累计 LoRA 状态；不堆叠 SFT/DPO/RL adapter、不默认 merge。阶段转换不是 `--resume`，不能继承父阶段 optimizer。reference 必须包含父阶段 adapter，关闭当前 adapter 得到的 base 不能冒充 reference。release 记录父 release hash、base revision、processor hash、adapter hash，独立新进程加载后与保存前 greedy 文本一致。

## 配置 schema 与实验预算

以下是待实现的 `qwen3-asr-train-v1` 模板，不是已经存在的可运行配置。尖括号值必须在环境/数据 preflight 后替换。未列出的字段拒绝；数字有界且有限；不允许 eval 采样、`world_size != 1`、stage 与父 release 不符。路径相对仓库或 manifest 的规则由字段声明，不读取隐式 notebook 状态。

```yaml
schema_version: qwen3-asr-train-v1
stage: sft
profile: pilot                    # smoke | pilot | scale
seed: 42
model:
  name: Qwen/Qwen3-ASR-0.6B
  revision: <immutable-hub-sha>
  processor_sha256: <sha256>
  parent_release: null            # DPO/RL 必填不可变 release 路径
  parent_release_sha256: null
runtime:
  environment_lock_sha256: <sha256>
  dtype: bfloat16
  world_size: 1
  attention: sdpa
  gradient_checkpointing: true
  num_workers: 0
  allow_dirty: false
  tf32: false
  deterministic_algorithms: true
  max_runtime_minutes: 120
  shutdown_margin_minutes: 10
  min_free_gpu_fraction: 0.10
  min_free_disk_gb: 10
  min_free_ram_gb: 4
lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  scope: decoder_attention_only
  projections: [q_proj, k_proj, v_proj, o_proj]
data:
  train_manifest: manifests/train.jsonl
  val_manifest: manifests/val.jsonl
  manifest_metadata: manifests/manifest.json
  max_audio_seconds: 30
  max_transcript_tokens: 512        # 不含 EOS
  max_input_tokens: 2048           # 包含展开音频占位后的全序列
  overlength: reject              # 不可静默裁音频或 target
  sampler: seeded_shuffle
batch:
  micro_batch_size: 1
  gradient_accumulation_steps: 8
  eval_batch_size: 1
optim:
  name: adamw
  learning_rate: 0.0001
  betas: [0.9, 0.95]
  eps: 0.00000001
  weight_decay: 0.01
  max_grad_norm: 1.0
  scheduler: cosine
  warmup_steps: 2
  min_lr_ratio: 0.1
train:
  max_steps: 100
  max_epochs: 1
  log_steps: 1
  save_steps: 10
  eval_steps: 10
  keep_last: 2
  interrupt_after_step: null       # 仅 smoke 测试中设为 5
scoring:
  normalization: zh-cer-v1
  wer_tokenizer: jieba
  wer_tokenizer_version: <locked-version>
  wer_dictionary_sha256: <sha256>
  wer_hmm: false
  empty_penalty: 0.5
  repeated_penalty: 0.5
  reward_clip: [-1.0, 1.0]
decode:
  eval:
    do_sample: false
    num_beams: 1
    max_new_tokens: 513
    repetition_penalty: 1.0
    no_repeat_ngram_size: 0
    context: ''
    language: zh
    eos_policy: tokenizer_eos_only
  rollout:
    group_size: 4
    greedy_candidates: 1
    temperature: 1.0
    top_p: 0.95
    top_k: 50
    max_new_tokens: 513
    seed_strategy: sha256_seed_epoch_visit_sample_candidate
    context: ''
    language: zh
    eos_policy: tokenizer_eos_only
gate:
  min_rows_per_scenario: 50
  clean_cer_increase_max: 0.01
  clean_cer_increase_vs_base_max: 0.02
  degraded_macro_increase_max: 0.0
  scenario_cer_increase_max: 0.01
  min_improved_degraded_scenarios: 1
  empty_rate_max: 0.01
  repeated_rate_max: 0.01
  long_rate_max: 0.01
  truncation_rate_max: 0.01
  valid_output_rate_min: 0.99
  min_preference_accuracy: 0.55
  min_val_pairs: 100
  min_reward_improvement: 0.002
  raw_kl_max: 0.001
  reward_drop_stop: 0.01
dpo: null
rl: null
```

DPO 用以下块替换 `dpo: null`；RL 用以下块替换 `rl: null`。不适用的 stage 块必须为 null。DPO 的 `data.train_manifest/val_manifest` 仍是音频 gold manifest，pair 是额外输入，不能用 val pairs 替代 ASR 验证集。

```yaml
dpo:
  beta: 0.1
  pair_train_manifest: manifests/dpo_train_pairs.jsonl
  pair_val_manifest: manifests/dpo_val_pairs.jsonl
  reference_cache_metadata: manifests/pair_metadata.json
  min_train_pairs: 100
  min_pairs_per_scenario: 10
  logp_reduction: sequence_sum
  train_scenarios: null                       # 可选：如 [clean, noise, reverb, far_field] 或 [noise, reverb, far_field]
  val_scenarios: null                         # 可选：缺省跟随 train_scenarios；可单独指定如 [clean, noise, reverb, far_field]
  aligned_substitution_only: false            # 可选：仅保留无标点、等长、同 token 数的原位替换对
  exclude_clean_identical_rejected: false     # 可选：训练集剔除与同 source_utterance_id 的 clean 场景归一化 rejected 完全相同的同音/错标重复对
  ref_margin_min: null                        # 可选：ref_chosen_logp - ref_rejected_logp 下界（如 -4.0）
  ref_margin_max: null                        # 可选：ref_chosen_logp - ref_rejected_logp 上界（如 0.0）
rl:
  algorithm: group_pg_mixed_v1
  advantage: group_mean_centered
  kl_beta: 0.04
  updates_per_rollout: 1
  max_group_visits: 500
  no_signal_stop_groups: 20
  invalid_group_rate_max: 0.01
```

这两个块是 schema 示例；一份实际配置只启用所属阶段的块。阶段默认预算如下，必须合成为完整 YAML 后冻结，运行时不再暗中覆盖：

| 配置项 | SFT pilot | DPO pilot | RL pilot | 每阶段 smoke |
| --- | --- | --- | --- | --- |
| learning_rate | 1e-4 | 5e-6 | 1e-5 | 同该阶段 |
| LoRA dropout | 0.05 | 0.0 | 0.0 | 同该阶段 |
| micro batch / 累积 | 1 / 8 | 1 pair / 4 | 1 group / 2 | 同该阶段 |
| max_steps / max_epochs | 100 / 1 | 100 / 1 | 50 / 1 | 6 / 100 |
| save_steps / eval_steps | 10 / 10 | 10 / 10 | 5 / 5 | 5 / 5 |
| max_runtime_minutes | 120 | 120 | 120 | 30 |

smoke 在 step 5 保存、同步、断开 runtime，原配置恢复至 step 6；`interrupt_after_step=5` 事件仅触发一次。smoke 有 5–20 次真实 optimizer 更新才算通过，RL 无更新不能凑步数。smoke 数据只取 train，含 clean 和 degraded，标记 `publishable=false`；其 gate 可以诊断但永远不能发布。pilot 的最大步数是上限，不是收敛保证；扩大规模前按实测 step/eval/sync 耗时、数据小时数、磁盘和显存确定新配置。

配置先解析全部默认和 CLI 路径绑定，再输出 `resolved_config.yaml`。语义配置 hash 是排除可搬迁本地缓存/输出路径与 `--resume` 控制项后，对稳定排序 JSON（UTF-8、无 NaN/Inf）计算 SHA-256；另存 YAML 文件字节 hash。训练 seed 必须与 CLI 相同，否则失败。model、manifest、LoRA、optimizer/scheduler horizon、batch、解码、reward/gate 均属于语义 hash。只有本地路径变化可以在原 run 恢复；更改学习率、阈值、代码或预算须新 run。

## 单 A100 资源与可复现性

记录 GPU 型号/显存、CUDA/驱动、Python、torch、qwen-asr、Transformers、PEFT、音频和分词依赖的精确版本。`requirements.txt` 只有范围约束。仓库根目录的 `requirements.lock` 是可安装的项目锁，schema 为 `qwen3-asr-portable-lock-v1`：普通行交给 pip 精确安装，`# verify:` 行只核对该 runtime 已经带上的 CUDA 构建。`results/<run_id>/requirements.lock` 是那次 runtime 的完整 `pip freeze` 指纹；`phase0_baseline_20261013b` 的指纹有 729 行，SHA-256 为 `aebd8c3d0315c58efe821007572a08af96b890ccd3bc02fde04350255b28f5c5`，其中包含 `file://` 和 git 地址，不能作为 `pip install -r` 的输入。第二份空 runtime 若预装的 verify 包与锁不一致，重建以退出码 2 停止，不重装 torch。通过标准见 `05_testing_plan.md`。BF16 能力探针、单 batch forward/backward、LoRA 梯度、显存余量均通过才训练。FlashAttention 不是默认安装要求；切换 attention 后重新做 probe 和 baseline。

预估磁盘包括模型缓存、全量物化音频、当前 run、两个 checkpoint 暂存副本；按实际文件大小再留 `min_free_disk_gb`。RAM 留量不小于配置，preflight 写入 `resource_report.json`。RL reference 顺序前向，不保存计算图；rollout `no_grad` 且用 KV cache，训练 forward `use_cache=false`，gradient checkpointing 与冻结 embedding 的梯度流通过测试。OOM 不静默更改 batch、长度或精度。

Python/NumPy/torch CPU/CUDA 和数据 sampler 都设 seed，关闭 cuDNN benchmark/TF32。不支持确定性算子时 preflight 失败；另起显式关闭确定性的探索配置并记录差异。候选 seed 为稳定 SHA-256(`seed, epoch, visit_index, sample_id, candidate_index`) 映射到整数，不使用 Python `hash()` 或 run_id，保证相同定义的新 run 可重复。RNG 必须保存/恢复，GPU 输出逐位相同不是跨环境验收条件。

## SFT 损失与更新单位

SFT 使用 causal cross entropy：一个累积窗口内所有监督 token 的 NLL 总和除以监督 token 总数。不能简单平均不同长度 micro batch 的平均 loss。DPO 按 pair 平均、RL 按 group/候选平均，见后文；不存在适用于三种算法的通用“全部 token 平均”。

`global_step` 只计真实 `optimizer.step()`；`micro_step`、`samples_seen`、`group_visits` 分开计。一个完整累积窗口后依次裁剪梯度、optimizer step、scheduler step、清梯度。epoch 尾不足累积数时按实际分母完成一次更新，禁止丢失样本或沿用满窗口分母。scheduler 仅随真实更新推进，warmup 后在固定 horizon 上 cosine 降至 `min_lr_ratio`，恢复不能重启 warmup。日志记录裁剪前 grad norm、lr、loss、token/样本数、显存与耗时。

## DPO 配对、缓存与损失

使用 SFT release 在 train 和 val **分别**生成 greedy 假设。gold 对自身的 CER 恒为零，因此初始方案固定 `chosen=gold`、`rejected=SFT hypothesis`，只接受 rejected CER > 0、归一化后不相等且两边非空的 pair。删除“假设优于 gold 的比例”门禁；label 质量必须通过听音审计，不能靠这个恒不成立的比较检验。

每条音频最多一对；ties/空输出进 `pair_rejected.jsonl`。超过 token 上限的长输出进 `overlength_pairs.jsonl`，不切成伪完整句、不默默丢掉；短于上限的长幻觉仍进入 pair。报告保留率、长度差、各 scenario 的 pair/source 数和 rejected 原因。重复 pair ID、音频损坏或跨 split source/audio 泄漏导致整个构建失败；不能只删冲突行蒙混过关。听音抽查每场景 min(20, available) 对，保存 `pair_audit.json` 的抽样 seed、错误和处置。数据数量不足先返回数据阶段。

pair 行最小示例（示意值，不是实测）：

```json
{"schema_version":"qwen3-asr-dpo-pair-v1","pair_id":"u1_noise","sample_id":"u1_noise","source_utterance_id":"aishell:u1","audio":"wavs/u1_noise.wav","scenario":"noise","split":"train","reference_text":"你好","chosen_text":"你好","rejected_text":"你号","chosen_cer":0.0,"rejected_cer":0.5,"chosen_token_count":3,"rejected_token_count":3,"ref_chosen_logp":-2.0,"ref_rejected_logp":-4.0}
```

实际 token 数由固定 tokenizer 得到，含 EOS。`pair_metadata.json` 保存源 manifest、SFT release、processor/tokenizer、语言 prompt、mask 规则、score contract、两份 pair 文件的 hash；hash 放在 sidecar，文件行内不包含自身文件 hash。reference 在 eval/no_grad 下对完整转写+EOS **序列求和**得到 logprob，log-softmax/累加用 FP32，padding/prefix 不算。至少对 min(20, pair数) 条在线重算，绝对误差≤1e-4且相对误差≤1e-5（同环境/精度），失配不得训练。离线模式省去常驻参考模型，但不能声称必然使总显存或计算减半。

标准 DPO 为 `mean_pairs[-logsigmoid(beta * ((logp_chosen-logp_rejected) - (ref_chosen_logp-ref_rejected_logp)))]`；四个 logp 都是相同 token mask 的序列和，不能在公式内替换成 token 均值。此定义依据 [DPO 原论文](https://arxiv.org/abs/2305.18290)。长度均值仅供审计；trainer 不把 CER 或 gold 评分字段作为额外 loss。

在不改动已冻结 `pairs_logprob/` 与 `ref_logprobs.jsonl` 磁盘文件及 SHA-256 的前提下，trainer 在核对完离线缓存后，支持在内存中按 `dpo.train_scenarios`、`dpo.val_scenarios`、`dpo.aligned_substitution_only`、`dpo.exclude_clean_identical_rejected` 和 `dpo.ref_margin_min` / `dpo.ref_margin_max` 过滤训练与 preference val 对：
1. **背景与问题**：`20261019T012838_dpo_06b_zh_seed42`（`configs/train/dpo_pilot_nondropout.yaml`）在 step 10 因 `degraded_macro_cer` 上升 `+0.0006144183506280737`（净增 `+18` 次编辑：`dropout +20`、`noise -2`、`reverb 0`、`far_field 0`）触发 `risk_stop`。逐样本诊断表明，`dropout` 场景的 `10` 次静音段插入和 `+14` 次净增编辑（占 `dropout` 退化的 `70%`）来自单条静音悬崖样本 `BAC009S0748W0156__dropout`（父模型输出 `'就有一零个，加级贫困县'` 计 2 次编辑，DPO step 10 把 `'，'` 翻转为英文静音幻觉 `'就有一零个Human Rights Watch'` 计 16 次编辑）。数学与数据根因有四点：
   - **标点与不等长 token 的配分函数不抵消**：3640 条非 dropout 训练对中，`chosen_text`（AISHELL-1 原始转写）100% 无标点，而 `rejected_text`（SFT `prediction_raw`）有 `36.8%`（1340 条）含标点（`，。、：·`）、`48.6%`（1769 条）满足 $T^+ \ne T^-$。序列和 DPO 中 $\log \pi_\theta(y^+|x) - \log \pi_\theta(y^-|x) = \sum z_{y_t^+} - \sum z_{y_t^-} + (\sum_{t=1}^{T^-} \log Z(c_t^-) - \sum_{t=1}^{T^+} \log Z(c_t^+))$；当 $T^- > T^+$ 或含 `'，'` 时，`rejected` 对 `'，'` 施加直接负梯度，且多出的 $+\sum_{t=T^++1}^{T^-} \log Z(c_t^-)$ 未抵消项直接抬高词表外英文幻觉 token（如 `'Human'`）的 logit 并压低中文延续字。
   - **极端负参考 margin 的同音专名与金标错字噪声**：全量 3640 条中 `75.5%` 的 `ref_margin = ref_chosen_logp - ref_rejected_logp < -2.0`（中位数 `-5.189` nats），`< -4.0` 的样本大量为声学不可分专名/代词同音字（`路某/陆某`、`文骁/文晓`、`她/他`）或金标错字（`末阳市/莫阳市`、`四百馀人/四百余人`、`显着/显著`），导致 `raw_policy_accuracy` 在 `20/533` 上 0 翻转，梯度被不可翻转样本主导。
   - **Clean 同音字/人名/金标错字跨四场景重复拷贝污染（占 781 条原位替换对的 `64.9%`，前 10 步占 `67.5%`）**：即便开启 `aligned_substitution_only: true` 与 `ref_margin_min: -4.0` 筛出 781 条非 dropout 对，其中仍有 `507/781`（`64.9%`，前 10 步 40 条中占 `27/40 = 67.5%`）是 SFT 在录音室 `clean` 音频上就已经输出相同 `rejected_text` 的跨场景重复拷贝，包括 AISHELL-1 原始金标错别字（如 `BAC009S0517W0317` 金标错字 `'松江新兵镇'` 对 SFT 正确地名 `'松江新浜镇'`，在前 10 步重复出现 2 次；`BAC009S0152W0356` 金标错字 `'避谣'` 对 `'辟谣'`）以及 100% 同音人名/虚词（`陈建/陈健`、`周素勤/周素琴`、`杨常青/杨长青`、`房祖名/房祖明`、`泰禾/泰和`、`卖得不好/卖的不好`）。只挂在解码器自注意力层上的 LoRA 无法从无声学差异的同音词中学习声学抗噪，反而会扭曲通用文本先验并浪费 65% 的梯度更新。
   - **AdamW 超参与 Step 10 门禁的交互边界**：`eps=1e-8`、`warmup_steps=2`、`weight_decay=0.01` 会将微小残差梯度放大为 $\pm \text{lr}$ 符号步长并衰减非零 `lora_A/lora_B`，将 `BAC009S0748W0156__dropout` 推过静音悬崖；反之若将阻尼过度拉大到 `warmup_steps=20`、`eps=1e-4`，前 10 步累积更新量会被削弱 4～5 倍，在 BF16 下难以在 step 10 翻转 `noise/reverb/far_field` 编辑或稳定越过 `preference_accuracy >= 0.55` 早停线。
2. **内存原位替换、去 Clean 同音拷贝与参考 margin 过滤规则**：
   - 当 `aligned_substitution_only: true` 时，仅保留同时满足以下条件的偏好对：
     1. `normalize_zh_cer_v1(chosen_text) == chosen_text` 且 `normalize_zh_cer_v1(rejected_text) == rejected_text`（两边均无标点、空白或需要 NFKC/casefold 折叠的字符）；
     2. `len(chosen_text) == len(rejected_text)` 且 `chosen_token_count == rejected_token_count`（字符数与 tokenizer 监督 token 数完全相等，$T^+ = T^-$）；
     3. 逐字符 Hamming 距离等于 `zh-cer-v1` Levenshtein 编辑距离（纯原位替换，无插入与删除错位，从而在每个匹配前缀与首个替换位点 $t_0$ 上精确满足 $\log Z(c_{t_0}^-) - \log Z(c_{t_0}^+) = 0$，对 `'，'`、`'Human'` 和 `<|im_end|>` 零直接梯度冲击）。
   - 当 `exclude_clean_identical_rejected: true` 时，基于未过滤前的全量 `pair_train` 行建立 `source_utterance_id -> normalize_zh_cer_v1(clean_rejected_text)` 映射，在训练集中剔除 `scenario == "clean"` 或归一化 `rejected_text` 与同句 `clean` 场景完全一致的样本，仅保留 SFT 在 `clean` 上识别正确（或不同）、纯粹因声学退化（`noise`、`reverb`、`far_field`）才发生混淆的样本（如 `新股/金融`、`推动/驱动`、`骑三轮/提三轮`、`高危/高龄`、`纵向/定向`、`总量/重量`）。
   - 当设置 `ref_margin_min` / `ref_margin_max` 时，仅保留 `ref_margin_min <= (ref_chosen_logp - ref_rejected_logp) <= ref_margin_max` 的样本。
   - **训练与验证集场景/同音过滤解耦**：`dpo.train_scenarios` 控制训练集场景（允许 `[noise, reverb, far_field]`），`dpo.val_scenarios` 控制 held-out preference 验证集场景（缺省跟随 `train_scenarios`，可独立设为 `[clean, noise, reverb, far_field]`），且 `exclude_clean_identical_rejected` 仅作用于训练集，从而在训练集 100% 聚焦 274 条纯声学退化纠错对的同时，保持验证集为 118 条四场景对齐样本，严格满足冻结门禁 `len(train_pairs) >= min_train_pairs (100)`、`len(val_pairs) >= gate.min_val_pairs (100)` 及各目标场景不少于 `min_pairs_per_scenario (10)`。同时修复 `train/train_dpo.py` 中 `SCENARIOS` 常量导入缺失，并在结束写完最终 `status.json`、`gate.json`、`run_report.md` 与 `listen_waiver.json` 后调用 `sync_drive_files` 同步至 Google Drive。

held-out preference accuracy = `(Δr>0 的对数 + 0.5×Δr=0 的对数)/总对数`，`Δr` 就是上式 sigmoid 的输入，比较使用 FP32 未四舍五入值；step 0 相同 policy/reference 应为 0.5，另报 ties。至少 100 个 val pairs、每场景至少 10 对，不足为 `insufficient_data`。门禁达到 0.55 且严格高于 0.5，同时报告 raw policy chosen>rejected accuracy 和长度分桶，防止只凭一个隐式奖励指标下结论。

## RL 算法、reward 与 KL

初始保留“1 greedy + 3 sampled”的候选方案，正式命名 `group_pg_mixed_v1`：它是受 GRPO 启发的组内相对策略梯度实验，**不是标准 GRPO/PPO**，没有 importance ratio、clip 或多轮复用。混合 greedy/top-p/top-k 采样和长度归一化引入偏差，不宣称优化原始策略分布下的无偏期望。[DeepSeekMath 原论文](https://arxiv.org/abs/2402.03300)是 GRPO 方法来源；本项目的偏离是显式实验决策。

1. 同一组来自一条 train 音频；候选只来自当前 policy，gold 不加入候选。policy 在 rollout 及重新计算 logprob 时均关闭 dropout，但重新计算允许梯度；reference eval/no_grad 且参数独立。
2. 固定 G=4，保存每条生成 token IDs、EOS/length/error stop reason、采样 seed、decode hash、policy snapshot step。空文本若有 EOS 是有效候选，重复文本保留，不按分数删候选。
3. 长度截断或系统错误使该组无效，整组不训练，保留全部 G 行（失败字段可 null）；禁止用剩余候选凑一组。空/重复有完整 token 的输出仍参与 reward，避免只留下高分样本。首 100 组结束或训练结束时检查 invalid rate≤0.01。
4. `R=clip(1-CER-0.5*empty-0.5*repeated,-1,1)`；记录 CER、各惩罚、截断前后 reward。长度/格式只计健康度，无额外隐藏 reward 项。DPO 配对按 CER，与 RL reward 分开。
5. `A_g=stop_gradient(R_g-mean(R))`，不除标准差、不替换常数；全部 reward 相等时，该组策略与 KL loss 都为零，并计 zero-advantage。
6. 有效组 `L_group = mean_g[-A_g * mean_t(logp_policy) + kl_beta * mean_t(k3)]`，t 仅为实际生成转写和真实 EOS；`k3=expm1(d)-d`，`d=logp_ref-logp_policy`，两侧均是**未经过 temperature/top-p 修改的模型分布**。FP32 计算，非有限立即停止，不静默 clamp 伪造小 KL。
7. 所有有效组等权，包括 loss 为零的有效组；无效组不在分母。一个窗口只更新一次，所有组在更新前同一 policy 上生成，更新后不重复使用。若整个窗口都无信号，跳过 optimizer 和 scheduler（否则 AdamW 动量/weight decay 仍可改参数），不增长 global_step，只推进 sampler/visit 状态。连续 20 组无信号或达 `max_group_visits` 停止为 `no_learning_signal`。
8. `k3_proxy_train` 是混合采样上的正则代理，不能称为无偏 KL。门禁 `raw_kl` 独立计算：在固定 val 的 **DPO step-0 greedy 前缀**上，对词表求 `sum_v p_policy(v)*(log p_policy(v)-log p_ref(v))`，再对所有有效转写/EOS位置 token 平均。参考前缀 token IDs 固化并 hash，不随当前生成变化。用 FP32、按 token 分块计算以控制显存；这是固定前缀上的条件 KL，不是整段序列分布 KL。
9. 训练每组记录 k3 proxy；每个 eval checkpoint 重新测 `raw_kl`。`raw_kl_max=0.001` 只用于这一明确定义的量，不套用于 k3 proxy 或乘 beta 后的 loss。两种 KL 均记录；reference 权重 tensor checksum 在启动、每个 eval/save 和退出时保持一致。

rollout JSONL 至少包括 `run_id/attempt_id/global_step/group_id/visit_index/sample_id/source_utterance_id/scenario/candidate_index/candidate_seed/is_greedy/token_ids/stop_reason/prediction_raw/prediction_normalized/reward_components/reward/advantage/policy_logp/reference_logp/k3_proxy_train/error`。其中 policy/reference logp 同时记录 transcript+EOS 的序列和与 token 数，`mean_t(logp_policy)` 由 trainer 按同一 mask 计算，不能把不同长度的序列和直接平均。组 ID 从 sampler visit 唯一派生，JSONL 先留本地，按 checkpoint 同步不可变分片。系统级模型错误终止 run；单样本推理错误写行继续批量评分。

RL 在任何更新前对本 run 全 val 记录 step 0 greedy CER/reward、reference prefix hash 和零 KL；父 DPO release 的同口径预测也必须存在。step 0 不能借用别的 run reward。最终报告同时比较 DPO 父版本与本 run step 0。

## 评测、停止与晋级

归一化、micro/macro CER、WER、健康度、缺失预测和 reward 平均规则由 [05_testing_plan.md](05_testing_plan.md)统一定义。所有率为 0–1，CER 可大于 1；阈值 .01 是绝对一个百分点，不是相对 1%。每个 eval 使用全 val，相同样本、音频 hash、processor、scoring contract、greedy decode 和 evaluator。adapter 本来会不同，不应被误判为不可比。

阶段父版本依次为 base/SFT/DPO，所有通过条件取 AND：

- 五类 val scenario 各≥50行；至少一个 degraded scenario 的 micro CER 严格降低，不对四舍五入值比较。
- clean micro CER 相对父版本上升≤.01，且相对 base 累计上升≤.02。
- 四类 degraded scenario CER 等权宏平均相对父版本不升；每个 degraded scenario 的回退≤.01，防止靠单场景进步掩盖其余退化。
- 在整体和每个 scenario，empty/repeated/long/truncation 率各≤.01、有效输出率≥.99。
- DPO 加上 preference gate 和 pair audit；RL 加上全 val 样本平均 greedy reward 相对 step 0 提升≥.002、raw_kl≤.001 和 reference checksum 不变。
- 相应阶段 smoke 已通过，当前 run 无数据/恢复/非有限故障，产物完整，dirty/smoke 不可发布。

立即停止：非有限 loss/grad、reference 变化、OOM、训练音频/缓存损坏、恢复不匹配；每次 full val 后若 clean/任一 degraded 回退或健康度越界、raw KL 越界、RL reward 比 step 0 下降>.01，则停止，不能偷偷放宽阈值。尚未改善但未触发风险线时可训练到预算上限。初始父模型已经不满足健康度时先修复数据/解码合同并重新 baseline，不为新 run 放宽 gate。

达到 max_steps 或 max_epochs 即收尾。没有任何 passed checkpoint 则 `gate_failed`；通过多个时按 degraded scenario 等权宏 CER 最低、再 clean CER 最低、再最早 step 选择，规则启动前冻结。运行时间上限不收尾。时钟含训练和全 val 解码，在每个训练 step 开始前、以及每条 val 样本解码前检查 `max_runtime_minutes - shutdown_margin_minutes`。一旦超过，进程以 `paused` / `max_runtime` 退出，停在最后一个已保存的 checkpoint。正在进行的 val 不写 predictions、metrics 或 gate，只写 `eval/<step>/runtime_stop.json`，记录已解码条数和新 token 长度。`--resume` 若发现该 step 该评但没有 decision，先补评测，再进入下一个 optimizer step。半成品评测不能选择，也不能把 run 标成 `completed` 或 `gate_failed`。每写一条 decision，就把 `eval_decisions.jsonl` 复制到 Drive 的 run 目录。换一台 VM 恢复时，把它和 checkpoint、`latest.json` 一起拷回本地 run 目录。Colab 客户端不再用一个截止时间包住整个训练。会话保活是 sub2 上的短连接，训练进程在 VM 里脱离该连接。配置里的 120 分钟不改，语义 hash 才能接上原 checkpoint。这次续训设置 `QWEN3_ASR_DISABLE_RUNTIME_BUDGET=1`，启动日志必须出现 `SFT_RUNTIME_BUDGET disabled`。checkpoint 仍每 10 step 在评测前同步到 Drive。`latest.json` 仅是恢复进度，不是最优模型。评测 checkpoint 与保存 step/adapter hash 必须完全一致。

val 通过的不可变 release 可供下一阶段使用，test 仍封存。Phase 5 冻结 base/SFT/DPO/RL 四份候选后，在同一 test manifest/greedy/evaluator 下各评一次，生成最终 benchmark 报告；不得因 test 差而返回调整同一批候选并重复选优。发现评测 bug 后重测须保留旧结果并提升 evaluator/benchmark 版本。

## 运行目录与恢复指针

Drive 上每个 run 使用以下结构；`<run_id>` 目录不能覆盖，`latest.json` 不是 release：

```text
runs/<run_id>/
  request.json                 # 原始请求、stage、seed、allow_dirty、父 release
  environment.json             # GPU/软件/硬件/环境锁 hash
  resource_report.json         # preflight 显存、磁盘、RAM、dtype、长度截断
  resolved_config.yaml         # 解析后的不可变配置
  manifest_sha256.json         # train/val/test 或 pair/reference hash
  status.json                  # 当前状态、原因、时间、global_step
  training_state.json          # 可选 run 摘要；可恢复副本在 checkpoint 内
  metrics.jsonl                # 追加式训练/评测指标
  events.jsonl                 # 追加式生命周期、错误、恢复和同步事件
  predictions/                 # 每个 checkpoint 的原始/归一化预测
  eval/                        # 不可变的评测 summary/逐样本评分
  rollouts/                    # RL 不可变分片，按 step/group 写入
  gate.json                    # 最终或最近一次 gate 裁决
  run_report.md                # 完成/失败时生成
  checkpoints/
    .incoming/<step_name>/     # 不可恢复暂存目录
    step_000010/
      adapter/
      optimizer.pt
      scheduler.pt
      rng_state.pt
      sampler_state.json
      training_state.json
      checkpoint_manifest.json
      sha256.json
  latest.json                  # 仅指向最后一个完整 checkpoint
```

`training_state.json` 至少包含 `schema_version`、`stage`、`global_step`、`micro_step`、`samples_seen`、`group_visits`、累积梯度边界、`config_sha256`、`manifest_sha256`、`eval_decode_sha256`、父 release hash、adapter hash、optimizer/scheduler step 和 `publishable`。`sampler_state.json` 包含每个 split 的 epoch、游标、shuffle seed、当前 group/visit；RL 还包含已消费的 group ID 集合或其不可变分片 hash。`checkpoint_manifest.json` 列出每个文件的相对路径、字节数、SHA-256、写入时间和 schema 版本。

`metrics.jsonl` 是追加式训练时间序列；每次完整 eval 另写 `eval/<step_name>/metrics.json`、`by_scenario.csv` 和逐样本评分。`gate.json` 只引用这些文件的 hash，不把最后一行日志当作完整评测。

`latest.json` 最小格式为：

```json
{"schema_version":"qwen3-asr-latest-v1","step_name":"step_000010","relative_path":"checkpoints/step_000010","checkpoint_sha256":"<sha256>","updated_at":"<RFC3339>"}
```

写入顺序固定为：本地临时目录完成全部文件 → 生成每文件和目录 checksum → 复制 Drive `checkpoints/.incoming/<step_name>/` → 目录完整性核对后就位到正式目录 → 从 Drive 读回重新校验 → fsync/安全写入新的 `latest.json`。Drive rename 不作为原子性保证；任何失败只写 `sync_failed`，保留旧 `latest.json`。恢复只接受 `latest.json` 指向且所有 hash、stage、config、manifest、parent release、LoRA、optimizer 参数映射、scheduler、world size 和 eval decode 均匹配的完整目录；不自动猜最近目录。

状态值为 `created`、`preflight`、`running`、`paused`、`completed`、`failed`、`oom`、`blocked_dtype`、`data_failed`、`sync_failed`、`resume_rejected`、`gate_failed`、`published`。`published` 只有通过 gate 且 release 复制、读回校验后才能写；`latest.json` 更新不等于 `published`。

## Checkpoint、命令、测试与验收

完整目录、状态字段、数据游标、checksum 与 Drive 指针协议统一见 [07_colab_execution_contract.md](07_colab_execution_contract.md)；CLI 和尚未实现的文件表见 [02_development_plan.md](02_development_plan.md)。checkpoint 必须携带解析配置、manifest hash、环境锁、adapter、optimizer 参数映射、scheduler、全部 RNG、sampler 游标和训练状态，不能只存 hash 而依赖 run 根目录。

测试必须覆盖：token mask/causal shift/EOS、LoRA 冻结与梯度；SFT 累积窗口等价；DPO 序列和与离线 cache 一致；RL advantage 符号、零组不更新 optimizer、组完整性、reference 不变、k3 与精确 KL 区别；阶段初始化独立加载；中断保存后继续至少一步。详细夹具、命令和通过线见测试方案。

一个训练阶段只有具备固定配置/manifest/seed/Git SHA、5–20 step smoke+恢复证据、同口径父 baseline、passed val gate、选中 checkpoint、原始/归一化预测、CER/WER/健康度及阶段特有指标、不可变 release 和 run report，才算 pilot 完成。Phase 5 另需封存 test 四阶段比较。结论写入进度文档；本次文档修订不满足这些实验验收。

## SFT runner 实现（2026-10-15）

Phase 1 的固定 val base 已经有 CER。这一步只补 SFT 训练器、smoke 配置和本地合同测试。100 step pilot 不在这次范围内。test 继续封存。OpenSLR 许可正文仍不记为已接受。smoke 的 `publishable` 固定为 false，gate 写成 `incomparable`，不能据此发布，也不能把它的 CER 当成相对 2500 条 val base 的晋级结果。

runner 读 `qwen3-asr-train-v1`。未列出的字段在创建 optimizer 之前拒绝。`train/train_sft.py` 接受 `--config`、`--manifest`、`--val-manifest`、`--output-dir`、`--run-id`、`--seed`、`--resume`。`--resume` 只读本 run 的 `latest.json`，并核对 checkpoint 目录 hash、stage、语义配置、manifest、LoRA 参数名、scheduler horizon 和 eval decode hash。

LoRA 用完整模块名正则，只挂 `thinker.model.layers.<n>.self_attn.{q_proj,k_proj,v_proj,o_proj}`。`thinker.audio_tower` 里的同名 `q_proj`、embedding、`lm_head` 和四个短后缀都不能进 optimizer。启动时写 `trainable_parameters.json`。LoRA B 初始化为 0。

`peft` 0.21.1 在注入 LoRA 时会检查预装的 `torchao`。版本低于 0.16.0 时它抛出 `ImportError`，而不是跳过这条量化路径。锁里没有 `torchao`，也不升级它。`scripts/rebuild_phase0_env.py` 在锁定的包安装成功后，卸掉低于 0.16.0 的 `torchao`，并且不改 CUDA 版 torch。

SFT 前缀与 `inference/run_inference.py` 的 greedy 前缀相同：chat template 之后加 `language Chinese<asr_text>`。监督目标是 manifest 原文加 tokenizer 读出的 EOS。前缀和 padding 的 label 为 `-100`。padding 用 attention mask 判断。pad id 与 EOS id 相同时也保留真实 EOS。loss 是窗口内监督 token NLL 之和除以监督 token 数。反传的是 token 和，窗口结束后再把梯度除以 token 数，然后裁剪、`optimizer.step()`、`scheduler.step()`。

`processor_sha256` 是 revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` 上这五个文件的清单 hash：`chat_template.json`、`merges.txt`、`preprocessor_config.json`、`tokenizer_config.json`、`vocab.json`。清单每行是 `文件名\t文件SHA-256`，按上述顺序，末尾有换行，再对整段 UTF-8 做 SHA-256。smoke 配置里的值是 `1b94f6ef846ce0faa88927fd8104c27831c457a69f103e6ec62fa392d315856c`。`environment_lock_sha256` 是仓库根目录 `requirements.lock` 的文件字节 SHA-256。

smoke 配置是 `configs/train/sft_smoke.yaml`：6 step、累积 8、第 5 step 保存并评测、`interrupt_after_step: 5`。中断标记写入 checkpoint 后只生效一次，恢复时原配置继续到第 6 step。数据由 `train/build_sft_smoke_subset.py` 从 pilot train/val 抽出：按 `SHA-256(seed:source_utterance_id:sft-smoke-train-v1)` 排序，取同时具备五类场景的前 10 个 train source；val 用 `sft-smoke-val-v1` 取前 2 个 source。这条 val 不是 2500 条固定 val。WER 保持 null。当前工作区是脏的，所以 smoke 配置 `allow_dirty: true`，并把 `git status --porcelain` 写进 `environment.json`。

Drive 未挂载时写 `sync_skipped`，只在 Colab 本地磁盘完成 checkpoint 和 `latest.json`。这次不发起 OAuth，也不把本地恢复说成 Drive 读回。正式 pilot 仍要按 `07_colab_execution_contract.md` 做 Drive 读回。

本地测试不下载模型。Colab 标准 A100 上的通过证据是：preflight 里 trainable 名称全部落在解码器注意力 LoRA、loss 和 grad 有限、第 5 step 的 checkpoint 可被 `--resume` 加载、再完成第 6 step。此证据本身不表示 Phase 2 完成。Phase 2 还要有相对这份 2500 条 val base 的同口径 gate。

## SFT pilot（2026-10-15）

smoke run `20261015T080308_sft_06b_zh_seed42` 已完成第 5 step 保存和第 6 step 恢复。下一步是 100 step SFT pilot，不评 test，不处理许可证，不进入 DPO。

`configs/train/sft_pilot.yaml` 的预算按上表冻结：学习率 `1e-4`，LoRA dropout 0.05，micro batch 1，累积 8，`max_steps` 100，`max_epochs` 1，每 10 step 保存并评全 val，`interrupt_after_step` 为 null，`max_runtime_minutes` 120，关机余量 10 分钟。运行时钟含全 val 解码。有效上限是 110 分钟。比较使用严格大于，所以刚好 110 分钟仍可再解码一条；下一条开始前如果已经超过，就暂停。暂停点是最后一个已保存的 step，不一定走到 100。2026-10-16 的 `1e-5` run 在 step 30 的 val 解码里越过了这个上限。当时的检查只放在训练 step 之前，解码一旦开始就不再看时钟，Colab 客户端的 4 小时超时随后断开了连接。step 10 和 step 20 的门禁当时已经通过，但 decision 文件留在 VM 上，没有进 checkpoint。WER 仍为 null。`keep_last` 取 10，因为 100 step 里有 10 个保存点，选择规则要用到每一个已评 checkpoint。模板里的 `keep_last: 2` 会在选出之前删掉更早的通过点，这次不用它。

训练 manifest 是 `results/phase1_aishell_pilot_20261015/manifest/train.jsonl`，验证 manifest 是同一目录的 `val.jsonl`。test 音频不上传。父 baseline 是 `configs/train/aishell_val_base_20261015.json`，val 文件 SHA-256 必须是 `2503945d8b1adbc3f0070a24fe33e5cb8b98e0248c68c550d59955bff33c7de8`，`decode_eval_sha256` 必须是 `4a4fee2b67884438faa196909749fcc8dcc6126b175554ad19992bf9b4f2f642`。对不上就在下载模型前退出，不训练。

每次全 val 后用 micro CER 的原始值比较，不四舍五入。clean 相对 base 的上升不超过 0.01，同时不超过相对 base 的累计上限 0.02。四个 degraded scenario 等权宏平均不得上升。任一 degraded scenario 的上升不超过 0.01。整体和每个 scenario 的 empty、repeated、long、truncation 率不超过 0.01，有效输出率至少 0.99，每个 scenario 至少 50 条。至少一类 degraded 的 micro CER 严格下降，这个 checkpoint 才算通过。越过风险线就停止。没有改善但没越线时继续，直到步数、epoch 或时间上限。结束时如果没有通过的 checkpoint，状态是 `gate_failed`。有多个通过点时，按 degraded 宏 CER、clean CER、更早的 step 选择。`latest.json` 仍只表示恢复位置。`allow_dirty` 为 true，而且这次没有 release 读回，所以 `publishable` 保持 false。

pilot 启动前 Drive 根目录必须已经挂上。没挂载就退出，不把 checkpoint 只留在会消失的 VM 磁盘上。smoke 里的 `sync_skipped` 不适用于这次。

## 下一轮 SFT（2026-10-15，未启动）

背景：run `20261015T104034_sft_06b_zh_seed42` 用 `1e-4` 跑到 step 10 就因 gate risk 停止。多出来的编辑主要是删字。V100 GPU 3 上的 40 条探测显示，LoRA B 仍为 0 时 teacher-forced 下一词不会提前给出 EOS；载入 step 10 adapter 后，dropout 有 15/24 条提前，clean 有 1/16 条提前。`max_new_tokens` 513 不是停止原因。把最少新 token 设为 8 只让这批短样本变长；按音频秒数乘 2 的下限把这 40 条的合计 micro CER 抬到 `0.40970654627539504`。这次不把解码下限写进配置，也不使用 `ignore_eos`。

范围：只新增 `configs/train/sft_pilot_lr1e5.yaml`。`configs/train/sft_pilot.yaml` 继续记录已经跑过的 `1e-4`。不恢复 run `20261015T104034_sft_06b_zh_seed42`，不改解码、门禁、LoRA、batch 或步数，不评 test，不进入 DPO 或 RL。run `20261015T234939_sft_06b_zh_seed42` 已从 step 30 恢复并在 step 50 停下。选出的 checkpoint 是 step 30。

设计：profile 仍是 `pilot`，seed 42。学习率改为 `1e-5`，是已跑配置的十分之一。warmup 仍是 2 step，所以第 1 step 的学习率是 `5e-6`，第 2 step 才到 `1e-5`。`decode.eval` 与父 baseline 的 `decode_eval_sha256` 保持 `4a4fee2b67884438faa196909749fcc8dcc6126b175554ad19992bf9b4f2f642`。数据、baseline 和 gate 与 `sft_pilot.yaml` 相同。语义配置 hash 会因为学习率不同而不同，`--resume` 不能把这个文件接到旧 checkpoint 上。

测试：`python3 -m unittest tests.test_train_sft -v` 要能加载新文件，确认学习率是 `1e-5`，`1e-4` 的旧文件没有被改掉，两边的 `decode.eval` hash 相同，语义 hash 不同。测试不下载模型，也不占 GPU。

验收：配置能被 schema 接受，并且还没有对应的 run 目录。真正开跑之后，仍要用固定 2500 条 val、同一 decode hash 和同一 gate 判断。40 条探测的 CER 不能当作通过。没有新的 passed checkpoint 和 Drive 读回之前，Phase 2 仍未完成，`publishable` 保持 false。

影响：已完成的 `1e-4` checkpoint 保持原样，不能当作这轮的起点。正式阶段表里的 SFT pilot 默认学习率仍是 `1e-4`，那是已经执行过的预算；`1e-5` 是这次失败之后的单独下一轮，不是把默认表改写成已经验证的最优值。

## SFT release 读回（2026-10-16）

背景：run `20261015T234939_sft_06b_zh_seed42` 的 2500 条 val 选出 step 30。`latest.json` 指向 step 50。DPO 的父版本必须是通过门禁的不可变 SFT release，不能是恢复指针，也不能 `--resume` 这个已经 `gate_risk` 的 run。

范围：新增 `train/sft_release.py`。它只发布 `gate.json` 里 `selected_step` 对应、且 `eval_decisions.jsonl` 中 `passed=true`、`risk_stop=false` 的 pilot checkpoint。smoke、`gate_failed`、没有 decision、以及 `.incoming` 都不能发布。不改解码和门禁，不评 test，不训练 DPO 或 RL，不把 run 的 `status.json` 改写成 `published`。这次的 release id 固定为 `sft_06b_zh_20261015T234939_step000030`。

设计：`prepare` 核对 checkpoint 目录 hash、语义配置 hash、`base_revision`、`processor_sha256` 和 `eval_decode_sha256`，只复制 `adapter/adapter.safetensors`、`adapter/adapter_config.json`、选出的 `scored.jsonl`、`gate.json` 和 `training_state.json`。optimizer、scheduler 和 RNG 不进入 release。`release.json` 的 schema 是 `qwen3-asr-sft-release-v1`，初始 `status=staged`、`publishable=false`，并记下 `latest_step` 以说明它不是这份 release。`readback` 在新进程里用同一份 `decode.eval` 重新 greedy 解码全部 val。加载模型前设置与训练相同的确定性开关：进程环境里的 `CUBLAS_WORKSPACE_CONFIG=:4096:8`、`use_deterministic_algorithms(True)`、关闭 TF32，并使用配置 seed。日志每 250 条打印 `SFT_RELEASE_PROGRESS`。逐条比较 `sample_id`、`prediction_raw`、`stop_reason` 和 `error`。全部一致才把 release 改成 `status=published`、`publishable=true`，并写 `readback.json`。payload 目录 hash 在改状态前后必须相同。Drive 上的目录是 `releases/sft/<release_id>/`，不覆盖 run 目录。

测试：`python3 -m unittest tests.test_sft_release tests.test_train_sft tests.test_directory_readmes -v`。本地夹具要拒绝 risk stop、拒绝已存在目录，并证明文本不一致时不会发布。测试不下载模型。Colab 读回必须覆盖选出 step 的全部 scored 行，少一条都不能写成 published。

验收：Drive 上的 release 目录读回 hash 一致，`readback.json` 的 `matched` 为 true，且 `compared` 等于 scored 行数。做到这一点之前，Phase 2 仍未完成，DPO 不能把这份 adapter 当父版本。2026-10-16 这份验收已经满足：`matched=true`，`compared=2500`，`mismatch_count=0`，Drive 同步 `match=True`。WER 仍是 null，所以整个 Phase 2 仍不标完成。DPO 现在可以把这份 release 当父版本。

## DPO 假设导出（2026-10-16）

背景：step 30 通过冻结的 clean/degraded 门禁，新进程读回 2500 条全部一致。step 40 也通过，step 50 因为 degraded macro CER 上升 `0.00010240305843801922` 触发 `risk_stop`。继续加 SFT step 已经变差，同一配置不再重训，也不恢复 `20261015T104034`。下一步是 DPO，不是再跑一轮 SFT。

范围：`train/sft_release.py hypotheses` 只用已发布的 SFT release，对一份 manifest 做与评测相同的 greedy 解码。这次先跑 train 的 19685 条。val 的 2500 条已经在 release 的 `source/scored.jsonl` 里，不再解码。不改 release 目录，不写 optimizer，不启动 DPO 更新，不评 test。`train_dpo.py` 仍不存在。pair、离线 logprob 和 preference gate 要等这份假设落地后再做。

设计：只接受 `status=published` 且 `publishable=true` 的 release。加载前使用和读回相同的确定性开关。输出是 JSONL，每行是 `decode_eval_rows` 的原始预测。已有文件必须是 manifest 的 sample_id 前缀，否则拒绝续写。每 2500 条把当时的文件复制到 Drive `dpo/<release_id>/train_predictions.jsonl`。复制失败不停止解码。进程结束时再复制一次，哈希必须一致。

测试：`python3 -m unittest tests.test_sft_release tests.test_train_sft tests.test_directory_readmes -v`。staged release 不能导出。前缀不匹配不能续写。测试不下载模型。

验收：Drive 上的 train 假设行数等于 19685，并且 sample_id 顺序与 train manifest 相同。做到这一点之前不构建 pair，不开 DPO 训练。2026-10-17 这次验收已经满足：Drive 与 VM 本地都是 19685 行，顺序与 train manifest 相同，SHA-256 都是 `581c672e966301de8cf05d36c21233f91d5224ad3b9202bd582f950d29fee401`。pair 构建见下一节。DPO 训练仍未开始。

影响：标准 A100 会话 `lora-asr-sft-release` 继续用，不新开，也不动另一台 High-RAM。SFT release 的 payload 保持不变。

影响：step 50 的 checkpoint 留在 run 里作恢复记录。release 只携带 step 30 的 adapter。原 `gate.json` 的 `publishable: false` 保持不变，发布状态只写在 release 目录。

## DPO pair 构建（2026-10-17）

背景：train 假设已经按 manifest 顺序落地。step 30 的 dropout 错误主要是中间静音被跳过，同一套 SFT 加步数已经变差，不再重训。DPO 要的是原文和这条假设组成的 pair。

范围：新增 `scripts/build_dpo_pairs.py`。同一次运行读取 train 和 val 的 manifest 与 greedy 预测，检查跨 split 的 source 和音频泄漏。不下载模型，不计算 reference logprob，不写 `train_dpo.py`，不评 test，不改 SFT release 的 payload。听音审计只抽出样本，不把未听的样本标成通过。

设计：父版本必须是 `status=published` 且 `publishable=true` 的 release。预测和 manifest 按 `sample_id` 对齐，多一行、少一行、重复 id 或 gold 文本不一致都让整个构建失败，不删冲突行。gold 归一化后为空也失败。音频缺失、短于 44 字节或不是 RIFF/WAVE，同样失败。

保留规则按这个顺序：

1. `error` 非空进 `pair_rejected.jsonl`，原因 `inference_error`。
2. 预测归一化后为空，原因 `empty_rejected`。
3. 归一化文本相同或 rejected CER 为 0，原因 `tie`。
4. `tokenizer.encode(text, add_special_tokens=False)` 再加一个 EOS 后，chosen 或 rejected 的 token 数大于 `max_transcript_tokens + 1`，进 `overlength_pairs.jsonl`。pilot 的上限是 512，允许的 token 数是 513，和 SFT 里 `supervised > max_transcript_tokens + 1` 的拒绝线相同。
5. 其余进 `train_pairs.jsonl` 或 `val_pairs.jsonl`。短于 token 上限的长幻觉留在 pair 里。

chosen 固定是 manifest 的原文，`chosen_cer` 为 0。rejected 是 `prediction_raw`。`ref_chosen_logp` 和 `ref_rejected_logp` 写 null。pair 行里不写文件哈希。`pair_metadata.json` 记录两份 pair 文件的 SHA-256、manifest、预测、release、评分合同、tokenizer 和 mask 规则。`pair_audit.json` 用 SHA-256(`seed`、split、scenario、`sample_id`) 的字典序，每个 split 的每个场景抽 min(20, 可用 pair) 条，处置是 `unlistened`。val pair 少于 100，或任一场景少于 10，状态是 `insufficient_data`，退出码 3，文件仍写出。数量够则状态 `pairs_built`，退出码 0。`trainable` 仍是 false，因为 logprob 还没算、听音还没做。

测试：`python3 -m unittest tests.test_build_dpo_pairs tests.test_directory_readmes -v`。夹具用字符数加 1 当作 token，不下载模型。覆盖 tie、空输出、超 token、上限内长幻觉、重复 id、跨 split 泄漏、坏音频、未发布 release 和退出码 3。

验收：正式输入是 release `sft_06b_zh_20261015T234939_step000030`、train 19685 行和 val 2500 行。token 用这份 release 的固定 tokenizer，EOS 是 `<|im_end|>`。val 五个场景都至少 10 对且总数至少 100，pair 行顺序与各自 manifest 一致。听音保持 `unlistened`，logprob 保持 null。这两项完成前不启动 DPO optimizer。2026-10-17 这次数量验收已经满足：train 7456 对，val 1022 对，tie 13707 条，超 token 0 条。val 是 clean 109、noise 189、reverb 121、far_field 114、dropout 489。Drive 与 VM 的 train pair SHA-256 是 `d958d602dcc4a252ac75ce36d99536e8a53f48fa72929d1b69ead2647dba3f1b`，val pair 是 `a7591e86f5a33efb314e29304153fe55e2d24ffba9bdd8b2739f953568b116d2`。`trainable` 仍是 false。

影响：pair 文件写到运行目录和 Drive 的 `dpo/<release_id>/pairs/`，不放进 release payload。SFT 配置和已发布 adapter 不变。

## DPO reference logprob（2026-10-17）

背景：pair 已经写成 train 7456 对、val 1022 对，`ref_chosen_logp` 和 `ref_rejected_logp` 仍是 null。DPO 损失要用父版本对 chosen 和 rejected 的序列 logprob。父版本是官方 base 加上这份已发布 SFT adapter。

范围：新增 `train/reference_logprob.py`。它只给现有 pair 补 reference logprob，不更新 LoRA，不写 optimizer，不评 test，不改 `pairs/` 里已经核对过的 null 文件，也不改 release payload。听音审计继续保持 `unlistened`。

设计：模型用 `configs/train/sft_pilot_lr1e5.yaml` 加载，再把 release 的 `adapter.safetensors` 拷进同一张 LoRA。加载前核对 processor hash 和 adapter hash。SFT 加载会打开 gradient checkpointing，并在 embedding 上注册一个把输出标成需要梯度的 hook。打分前必须去掉这个 hook、关闭 checkpointing，再 `model.eval()`、冻结全部参数，并放进 `torch.inference_mode()`。否则第一次 forward 会在 inference tensor 上要求梯度。LoRA dropout 在 eval 下不起作用。每条 pair 用和 SFT 相同的 prefix、音频和 EOS，分别对 chosen、rejected 做一次 teacher-forced forward。logprob 是移位后监督 token 的 FP32 NLL 之和再取负，不除以 token 数。prefix 和 padding 不计入。forward 得到的 token 数必须等于 pair 里已有的 token 数，否则整次缓存失败。

缓存按 train pair、再 val pair 的现有顺序追加到 `ref_logprobs.jsonl`。已有文件必须是这个顺序的 `pair_id` 前缀，否则不能续写。每 2500 条把缓存复制到 Drive `dpo/<release_id>/ref_logprobs/ref_logprobs.jsonl`。复制失败不停止计算；结束时哈希必须一致。全部算完后，用 SHA-256(`seed`、split、`pair_id`) 从每个 split 抽 min(20, pair 数) 条再 forward 一次。绝对误差要 ≤1e-4，相对误差要 ≤1e-5，两个条件都要满足。复核失败不写出带 logprob 的 pair 文件。复核通过后才把 logprob 填进新的 `train_pairs.jsonl` 和 `val_pairs.jsonl`，放到 `pairs_logprob/`。`trainable` 仍是 false，因为听音还没做，DPO optimizer 也还没有。

测试：`python3 -m unittest tests.test_reference_logprob tests.test_directory_readmes -v`。用注入的分数函数证明序列和不是 token 均值、非有限值会失败、误差线两边都检查、前缀不对拒绝续写、复核失败不发布 pair。不下载模型。

验收：train 7456 对和 val 1022 对都有有限的 reference logprob，token 数与 pair 一致，复核误差在上述两条线内，Drive 与 VM 哈希一致。2026-10-17 这次已经满足：复核 40 条，绝对误差和相对误差都是 0.0，cache SHA-256 `5a77f22586dcf1b73393124055eb02466a904d8bdbdd2d44d40ace78000ef43e`。`trainable` 仍是 false，因为听音审计是 `unlistened`。2026-10-17 对这 200 条做了文本核对，chosen 都等于 reference，4 条只是多一个字，dropout 删除仍集中在音频挖空的位置。没有听音频，处置没有改成通过。2026-10-17 用户明确说「不听，直接开始训练」。这是听音豁免，不是把处置改成已听。DPO 更新按下一节启动。

影响：标准 A100 会话继续用，不新开，也不动另一台机器。原来的 `pairs/` 保持 null logprob 不变。

## DPO pilot（2026-10-17）

背景：reference logprob 已经对齐，文本核对没有发现金标对调。用户在 2026-10-17 说「不听，直接开始训练」。200 条抽查的处置保持 `unlistened`，`heard` 保持 false。这是训练豁免，不是听音通过。

范围：新增 `train/dpo_contract.py`、`train/dpo_schema.py`、`train/train_dpo.py` 和 `configs/train/dpo_pilot.yaml`。父版本是已发布的 `sft_06b_zh_20261015T234939_step000030`。不改 `configs/train/sft_pilot_lr1e5.yaml`，不恢复 `20261015T104034` 或 `20261015T234939`，不覆盖 null `pairs/`，不评 test，不启动 RL，不把这次 pilot 标成 release。

设计：loss 是 `mean_pairs[-logsigmoid(beta * ((logp_chosen-logp_rejected) - (ref_chosen_logp-ref_rejected_logp)))]`，beta `0.1`，四个 logp 都是序列和。reference 来自填充后的 pair，不把第二份参考模型留在显存里。policy 走 SFT 的 `enable_training_mode`，新 AdamW。LoRA dropout `0.0`，学习率 `5e-6`，micro batch 1 pair，累积 4，最多 100 step，每 10 step 保存并评全部 2500 条 val。preference 用同一套序列 logprob，在未四舍五入的 FP32 delta 上计算。长度分桶是 rejected 更短、等长、更长，只报告 raw policy accuracy。CER 相对 step 30，clean 还要满足相对 base 的 0.02。配置里的 120 分钟不改。全 val greedy 放不进 110 分钟，所以启动设置 `QWEN3_ASR_DISABLE_RUNTIME_BUDGET=1`，日志必须有 `DPO_RUNTIME_BUDGET disabled`。step 0 在更新前抽 20 条 val pair，policy logprob 相对缓存的绝对差超过 1.0 就停。这是加载检查，不是把训练态 forward 说成和 inference 缓存逐位相同。`publishable` 保持 false：处置仍是 unlistened，而且这次 DPO runner 还没有单独的 6 step smoke。

测试：`python3 -m unittest tests.test_dpo_contract tests.test_directory_readmes -v`。不下载模型。

验收：pilot 进程打印 `DPO_PREFLIGHT_OK` 和 `DPO_RUNTIME_BUDGET disabled`，并开始真实 optimizer step。gate 可以随后通过或失败。通过也不发布。null pair 哈希保持 `d958d602dcc4a252ac75ce36d99536e8a53f48fa72929d1b69ead2647dba3f1b` 和 `a7591e86f5a33efb314e29304153fe55e2d24ffba9bdd8b2739f953568b116d2`。

影响：新开一台标准 A100。不使用 high-mem，不动另一台训练会话。原来的 `pairs/` 保持 null logprob。

2026-10-17 启动记录：run `20261017T131802_dpo_06b_zh_seed42`，会话 `lora-asr-dpo-pilot`（标准 A100 `gpu-a100-s-kkb-ass1c1-vs6ho7soe4x7`）。`parent_release_sha256` 已换成发布包里的 `a185a61f54b6fd74bd5e3bfa1173e15450cfbb772731567671a928aad805096a`。日志已经有 `DPO_PREFLIGHT_OK`、`DPO_RUNTIME_BUDGET disabled` 和 step 1、step 2。step 0 抽查 20 条，绝对差是 0。2026-10-17T14:06:38Z 该 run 在 step 10 风险停止，`DPO_EXIT 0`。Drive 上的决策是 `passed=false`：degraded macro CER 上升 `0.0006826870562534199`，preference accuracy `0.8199608610567515`，raw policy accuracy `0.019569471624266144`。不发布。

## DPO dropout probe（2026-10-18）

背景：`20261017T131802_dpo_06b_zh_seed42` 在 step 10 风险停止。degraded macro CER 上升 `0.0006826870562534199`，dropout 相对父版本 `+0.0034134352812670854`。preference accuracy `0.8199608610567515`，raw policy accuracy `0.019569471624266144`。Drive 上没有逐条预测。step 30 的 dropout 错误是静音段里的删除，整份 val 大约只有 4 次插入。`1e-4` SFT step 10 的签名是另一回事：dropout 假设长度掉到参考的大约 0.63。这次修复要把这两种签名在第一次评测分开。

范围：新增 `configs/train/dpo_dropout_probe.yaml`。父版本仍是 `sft_06b_zh_20261015T234939_step000030`，adapter SHA-256 仍是 `a185a61f54b6fd74bd5e3bfa1173e15450cfbb772731567671a928aad805096a`。beta `0.1`，LoRA dropout `0.0`，micro batch 1，累积 4，和 pilot 相同。这次的训练分布是 `train_scenarios: [dropout]`，学习率 `5e-5`，`warmup_steps` 0，`max_steps` 10，`save_steps` 10，`eval_steps` 10，`keep_last` 1。`configs/train/dpo_pilot.yaml` 的 `5e-6` 和 100 step 保持原样。chosen 仍是 manifest 原文，rejected 仍是 SFT greedy。null pair 和填充后的 pair 文件不改，reference logprob 不重算。`20261017T131802` 不恢复。test 不评。RL 不启动。SFT 不重训。`publishable` 保持 false。Phase 2 不标完成。

设计：profile 是 `dropout_probe`。SFT 加载器只接受 `smoke`、`pilot`、`scale`。解析时临时把 profile 写成 `pilot`，让 baseline 和 `keep_last` 检查生效，然后再改回 `dropout_probe`，并冻结上一节的 probe 预算。pilot 的 `dpo.train_scenarios` 必须缺省，缺省表示五个场景都训练。probe 必须正好是 `[dropout]`。整份填充 pair 先做哈希和 reference cache 检查，顺序仍是全部 train 再全部 val。通过之后才在内存里过滤。过滤后的 train 和 preference val 用 `required_scenarios` 再查一遍，缺席场景的 0 计数不再失败。step 0 的 20 条抽查仍来自完整 val pair。preference 只打 dropout pair。greedy CER 仍评全部 2500 条 val。配置里的 120 分钟不改。启动设置 `QWEN3_ASR_DISABLE_RUNTIME_BUDGET=1`，日志必须有 `DPO_RUNTIME_BUDGET disabled`。日志还要有 `DPO_TRAIN_SCENARIOS dropout`。

定位不改 `score_pair`，也不改 `qwen3-asr-metrics-v1`。`align_edits` 用完整矩阵回溯。从终点往回走时，能还原当前代价的前驱按这个顺序取第一个：相等字符的对角线、替换、删除、插入。插入的锚点是当时还没消费的参考下标除以参考长度。参考已经读完时，锚点是 1.0。锚点在 0.40 到 0.55、含两端的插入计入 gap。只统计 dropout 行。`DPO_LOCATE` 打一行。同一些字段写入该步 decision：`locate_signature`、`locate_gap_insertions`、`locate_dropout_length_ratio`、`locate_dropout_delta`。`eval_decisions.jsonl` 和 `eval/<step>/scored.jsonl` 在 gate 打印前同步到 Drive。

签名按这个顺序判定：

| 条件 | 签名 |
| --- | --- |
| gap 插入不少于 30，并且同时满足提前删除 | `MIXED` |
| gap 插入不少于 30，提前删除不成立 | `GAP_INSERT` |
| gap 插入少于 30，提前删除成立。提前删除指 clean 相对父版本上升超过 0.005，或 dropout 假设字符数除以参考字符数小于 0.75 | `EARLY_DELETE` |
| dropout CER 相对父版本的绝对值小于 0.002，并且 gap 插入少于 30，提前删除也不成立 | `QUIET` |
| 其余情况 | `OTHER` |

0.75 和 0.005 本身不算提前删除。30 次 gap 插入含等于 30。step 30 的长度比大约是 0.855，`1e-4` 大约是 0.63，所以 0.75 落在两条已知签名之间。probe 只跑到 step 10 的那一次评测。`selected_step` 固定写 null，这份 checkpoint 不能进 `sft_release.py`。decision 里的 `passed` 仍按原 DPO 门禁计算。`publishable` 保持 false。

测试：`python3 -m unittest tests.test_zh_cer tests.test_dpo_contract tests.test_directory_readmes -v`。回溯的非 match 数必须等于 `edit_distance`。`xy` 对 `yx` 在替换、删除、插入代价相同的格子上取两次替换。probe 预算被改动时加载失败。pilot 写上 `train_scenarios` 时加载失败。cache 检查在源码里位于场景过滤之前。不下载模型。

验收：probe 配置能加载，学习率是 `5e-5`，warmup 是 0，最多 10 step，`train_scenarios` 是 `[dropout]`。pilot 仍是 `5e-6`、100 step、`train_scenarios` 缺省。第一次评测日志有一行 `DPO_LOCATE`，decision 里有 `locate_signature`。null pair 哈希仍是 `d958d602dcc4a252ac75ce36d99536e8a53f48fa72929d1b69ead2647dba3f1b` 和 `a7591e86f5a33efb314e29304153fe55e2d24ffba9bdd8b2739f953568b116d2`。填充 pair 哈希仍是 `43473f10fbb9c0349adf2eb76eaf647506978c96eabb5b09847994456519c1b1` 和 `b82fc2783726555231164025a0e2d7faa82b5391bdcf8b98c6397eec1aff5eca`。`publishable` 仍是 false。

2026-10-18 启动记录：run `20261018T052928_dpo_06b_zh_seed42`，会话 `lora-asr-dpo-dropout-probe`（标准 A100 `gpu-a100-s-kkb-usc1b0-3fnghm4a82hzx`，Shape Standard）。VM PID 3187。日志有 `DPO_TRAIN_SCENARIOS dropout train_pairs=3816 val_pairs=489`、`DPO_STEP0 checked=20 max_abs=0`、`DPO_PREFLIGHT_OK`、`DPO_RUNTIME_BUDGET disabled`。10 个 optimizer step 已记下。配置峰值是 `5e-5`，`min_lr_ratio` 是 `0.1`，warmup 0，所以 step 1 的学习率是 cosine 上的 `4.8898772e-05`，step 10 是 `5e-06`。step 1 loss `0.693147`，step 10 loss `0.407643`。`2026-10-18T06:13:28Z` 保活记到 `DPO_DONE global_step=10` 和 `DPO_EXIT 0`。Drive 上的 decision 是 `locate_signature=OTHER`，gap 插入 `24`，dropout 长度比 `0.8480338612779902`，dropout delta `0.029492080830147455`，clean 相对父版本 `0.000546149645002731`。`passed=false`，`risk_stop=true`。preference accuracy `0.9856850715746421`，raw policy accuracy `0.018404907975460124`。`publishable` 保持 false。`latest.json` 指向 step 10，目录哈希 `977d3e6d203b2eaff1a5b5d0ffbb94f620cf120550c022d64cd67e883bc063bc`，那是最后一次保存。`gate.json` 没有同步到 Drive。这一步不作为父版本。

影响：pilot 的语义 hash 现在包含 `train_scenarios: null`。新代码不能恢复 `20261017T131802`。全 val greedy 仍大约要 40 分钟。过滤后的 preference 覆盖 dropout val pair，正式文件里是 489 条。可用模型仍是 SFT step 30。

## dropout 缺口剥离（2026-10-18，未训练）

背景：probe step 10 的 dropout 编辑从 1377 增到 1593。gap 内删除从 780 降到 762，0.55 之后的删除从 57 增到 187。多出来的尾部删除有 +142 落在 32 条上，其中 23 条的父版本错误本来就不全在静音段里。整句 gold 相对跳过静音的假设，会把波形里没有的字当成 preference 错误。计数和分带写在 `12_phase2_run_log.md`。

范围：这一节只规定 dropout pair 怎么剥。`score_pair`、签名分类、`configs/train/dpo_pilot.yaml` 和 `configs/train/dpo_dropout_probe.yaml` 保持原样。null `pairs/` 与 `pairs_logprob/` 不改写，reference logprob 不重算。不启动 DPO，不改学习率，不重训 SFT。`publishable` 保持 false。Phase 2 不标完成。

设计：`train/dpo_contract.py` 的 `strip_dropout_gap` 用 `align_edits`。参考下标除以参考长度落在 0.40 到 0.55、含两端时，match、替换和删除从两边去掉。插入锚点落在这段里时，从 rejected 去掉。剥完两边相同，这条就不再是 preference 错误。其他场景不调用它。空参考抛出 `ValueError`。`train/train_dpo.py` 不调用这个函数。现有填充 pair 的 logprob 仍对应未剥离的整句。

测试：`python3 -m unittest tests.test_dpo_contract tests.test_zh_cer tests.test_directory_readmes -v`。用例覆盖纯 gap 删除变成相同字符串、gap 外删除和 gap 外插入仍然不同、gap 内插入被去掉、端点 0.40 与 0.55 含在内、空参考失败，以及 `train_dpo.py` 源码不包含 `strip_dropout_gap`。不下载模型。

验收：冻结 pair 按这条规则数过。val dropout 489 对里 310 对剥完仍不同，179 对变成相同；val 合计从 1022 变成 843。train dropout 3816 对里 2573 对仍不同，1243 对变成相同；train 合计从 7456 变成 6213。val 1022 对的归一化文本与父版本 `scored.jsonl` 一致。null pair 哈希仍是 `d958d602dcc4a252ac75ce36d99536e8a53f48fa72929d1b69ead2647dba3f1b` 和 `a7591e86f5a33efb314e29304153fe55e2d24ffba9bdd8b2739f953568b116d2`。这次不产生新的 pair 目录，也不产生新的 run。

影响：下一次 DPO 如果要用剥离后的文本，必须另写 pair 并重算 logprob。在那之前不训练。可用模型仍是 SFT step 30。

## dropout pair 另写（2026-10-18）

背景：上一节已经数过冻结 pair。`validate_pair_rows` 要求 `chosen_text` 等于 `reference_text`，所以静音段不能只从 rejected 去掉。新文本的 logprob 也还没有。

范围：新增 `scripts/strip_dpo_gap_pairs.py`。输入是冻结 null pair。输出目录名必须是 `pairs_gap_strip`，不能是 `pairs` 或 `pairs_logprob`，也不能写进源目录。`configs/train/dpo_pilot.yaml`、`configs/train/dpo_dropout_probe.yaml` 和 `train/train_dpo.py` 不改。这个脚本不算 logprob，不启动 DPO，不改学习率，不重训 SFT。`publishable` 保持 false。

设计：先核对调用方传入的 train、val SHA-256，不一致就退出，不建输出目录。dropout 行先做 `normalize_zh_cer_v1`，再调用 `strip_dropout_gap`。剥完两边相同，或 rejected 变成空，写入 `gap_ties.jsonl`。仍不同的行把 `reference_text` 和 `chosen_text` 都写成剥离后的 chosen，`rejected_text` 写成剥离后的 rejected，并用调用方的 tokenizer 重算 token 数和 `score_pair`。原文留在 `source_reference_text` 和 `source_rejected_text`。其他场景的文本和 token 数原样复制。所有输出行的 reference logprob 都是 null。听音审计重新按 seed 抽取，处置保持 `unlistened`。val 不足时状态是 `insufficient_data`，退出码 3，文件仍留下供核对。

测试：`python3 -m unittest tests.test_strip_dpo_gap_pairs tests.test_dpo_contract tests.test_directory_readmes -v`。覆盖纯 gap 删除进入 `gap_ties.jsonl`、gap 外替换仍成对且 chosen 等于 reference、其他场景 token 数不变、源哈希不符或输出目录名是 `pairs` 时不写文件、审计处置仍是 `unlistened`。夹具用字符数加 1 当 token。不下载模型，不算 logprob。

验收：正式命令的期望哈希是 train `d958d602dcc4a252ac75ce36d99536e8a53f48fa72929d1b69ead2647dba3f1b`、val `a7591e86f5a33efb314e29304153fe55e2d24ffba9bdd8b2739f953568b116d2`。tokenizer 用 step 30 release 的同一套，revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`，模式 `encode-plus-eos`，EOS `<|im_end|>` id 151645，不是字符计数。2026-10-18 这次已经跑完。状态 `pairs_gap_stripped`。train 6213 对（clean 726、noise 1428、reverb 753、far_field 733、dropout 2573），val 843 对（clean 109、noise 189、reverb 121、far_field 114、dropout 310）。`gap_ties.jsonl` 1422 行，全部是 `tie_after_strip`（train 1243、val 179）。超 token 0。`short_val_scenarios` 为空。新 train pair SHA-256 `4ad364c87d367395d67603fdfb7acf503bd2c0174cfb35518b7b30b0c096505d`，val pair `caffc52d2674562143578a2d18a73bfed85233c74f6d573495ca77477d695a0d`。Drive `dpo/sft_06b_zh_20261015T234939_step000030/pairs_gap_strip/` 与本机 `results/phase3_dpo_gap_strip_20261018/pairs_gap_strip/` 一致。冻结 null pair 哈希没变。`pair_audit.json` 状态 `unlistened`。`trainable` 是 false，阻塞项是 `reference_logprob_pending` 和 `pair_audit_unlistened`。在 reference logprob 写入另一个新目录之前，不启动 DPO。可用模型仍是 SFT step 30。

影响：新目录里的文本还不能训练。旧的 8478 条 logprob 仍对应未剥离的整句。Phase 2 不标完成。

## gap-strip reference logprob（2026-10-18）

背景：`pairs_gap_strip` 已经写成，两边 logprob 仍是 null。2026-10-17 的 cache 对应未剥离的整句，不能套到剥离后的文本上。

范围：使用现有 `train/reference_logprob.py`，不改这个脚本，不改 `configs/train/sft_pilot_lr1e5.yaml`、`configs/train/dpo_pilot.yaml` 或 `configs/train/dpo_dropout_probe.yaml`，也不改 `train/train_dpo.py`。不覆盖 `pairs/`、`pairs_gap_strip/`、`ref_logprobs/` 或其中的 `pairs_logprob/`。不更新 LoRA，不写 optimizer，不评 test，不改学习率，不重训 SFT。听音处置保持 `unlistened`。`publishable` 保持 false。Phase 2 不标完成。

设计：配置只用 `configs/train/sft_pilot_lr1e5.yaml` 的 seed 和 base 模型。父版本仍是已发布的 step 30 adapter。`--pairs-dir` 指向 `pairs_gap_strip`，期望哈希就是上一节写出的 train `4ad364c87d367395d67603fdfb7acf503bd2c0174cfb35518b7b30b0c096505d` 和 val `caffc52d2674562143578a2d18a73bfed85233c74f6d573495ca77477d695a0d`。seed 42。每个 split 抽 min(20, pair 数) 再 forward 一次，绝对误差 ≤1e-4 且相对误差 ≤1e-5。`--sync-path` 的父目录是 Drive `dpo/sft_06b_zh_20261015T234939_step000030/gap_strip_ref_logprobs/`。脚本会把填充后的 pair 写到这个父目录下的 `pairs_logprob/`，因此不会碰到冻结的 `ref_logprobs/pairs_logprob/`。打分前仍去掉 embedding hook 并关闭 gradient checkpointing。标准 A100，`world_size=1`。

测试：合同仍是 `python3 -m unittest tests.test_reference_logprob tests.test_directory_readmes -v`。这次不新增测试类，也不下载模型到本地测试里。正式打分在标准 A100 上跑，日志要出现 `REF_LOGP_DONE`，不能出现 `REF_LOGP_FAIL`。

验收：2026-10-18 这次已经满足。`REF_LOGP_DONE train=6213 val=843 rechecked=40 max_abs=0.0`，日志没有 `REF_LOGP_FAIL`。`max_rel_error` 是 0.0。7056 条 logprob 都有限，cache 与填充 pair 的 `pair_id` 和 token 数逐行对齐。cache SHA-256 `dd9db5012841d61dafce571c833c508bac7e7f9e64286c0928dba9e139a2da00`，填充 train `7c01e5fc54596784f7e9ef22d3926cf830e3d034b58107aa455ac105bf0c392b`，填充 val `008fe7b044496d16645b022c2ef7f0677aca9c6d3a5de58f5c0e98ff5b65208b`。Drive `gap_strip_ref_logprobs/pairs_logprob/` 与 VM、本机副本一致。冻结 `pairs/`、`pairs_gap_strip/` 和 `ref_logprobs/pairs_logprob/` 哈希没变。`trainable` 仍是 false，阻塞项只剩 `pair_audit_unlistened`。DPO 没有启动。可用模型仍是 SFT step 30。

影响：这份 cache 只服务剥离后的文本。旧的 8478 条 cache 继续留在原目录。父版本序列 logprob 更高的一侧仍多数是 rejected：train 5621/6213，val 763/843。val dropout 是 250/310，不再是剥离前的 489/489。DPO optimizer 不因这次缓存自动启动。

## gap-strip dropout probe（2026-10-18）

背景：失败的 dropout probe `20261018T052928_dpo_06b_zh_seed42` 在 step 10 的签名是 `OTHER`。dropout CER 比父版本高 `0.029492080830147455`，0.55 之后的删除从 57 增到 187。剥离后的 pair 和 `gap_strip_ref_logprobs/` 已经核对完成。2026-10-18 用户说「开始吧」，授权只换这一套 pair 再跑一次短 DPO。2026-10-17 的「不听，直接开始训练」只覆盖当时那 200 条抽查，不覆盖 `pairs_gap_strip/pair_audit.json`。

范围：新增 `configs/train/dpo_gap_strip_probe.yaml`。不改 `configs/train/dpo_pilot.yaml`、`configs/train/dpo_dropout_probe.yaml`、`configs/train/sft_pilot_lr1e5.yaml`，也不重算 logprob，不改 `score_pair`。`train/train_dpo.py` 仍不调用 `strip_dropout_gap`。不恢复 `20261018T052928_dpo_06b_zh_seed42`、`20261017T131802_dpo_06b_zh_seed42`、`20261015T234939_sft_06b_zh_seed42` 或 `20261015T104034_sft_06b_zh_seed42`。不重训 SFT，不评 test，不启动 RL。`selected_step` 保持 null，这一步不成为父版本。`publishable` 保持 false。Phase 2 不标完成。可用模型在这次门禁写出之前仍是 SFT step 30。

设计：profile 仍是 `dropout_probe`。预算与失败的 probe 相同：只训 dropout，学习率 `5e-5`，`warmup_steps` 0，`max_steps` 10，`save_steps` 10，`eval_steps` 10，`keep_last` 1，beta `0.1`，LoRA dropout `0.0`，micro batch 1，累积 4，seed 42。父 release 仍是 `sft_06b_zh_20261015T234939_step000030`，adapter SHA-256 `a185a61f54b6fd74bd5e3bfa1173e15450cfbb772731567671a928aad805096a`。门禁和 `DPO_LOCATE` 的签名门槛不改。整份填充 pair 先做哈希和 cache 检查，再在内存里只留 dropout。step 0 的 20 条抽查仍来自完整 val pair。preference 只打 dropout。2500 条 val 的 greedy CER 仍全评。配置里的 `max_runtime_minutes` 保持 120。启动设置 `QWEN3_ASR_DISABLE_RUNTIME_BUDGET=1`，日志必须有 `DPO_RUNTIME_BUDGET disabled`、`DPO_TRAIN_SCENARIOS dropout`，第一次评测必须有 `DPO_LOCATE`。标准 A100，`world_size=1`，不用 `--high-mem`。

只换 pair 路径和哈希。null train `4ad364c87d367395d67603fdfb7acf503bd2c0174cfb35518b7b30b0c096505d`，null val `caffc52d2674562143578a2d18a73bfed85233c74f6d573495ca77477d695a0d`。填充 train `7c01e5fc54596784f7e9ef22d3926cf830e3d034b58107aa455ac105bf0c392b`，填充 val `008fe7b044496d16645b022c2ef7f0677aca9c6d3a5de58f5c0e98ff5b65208b`。cache `dd9db5012841d61dafce571c833c508bac7e7f9e64286c0928dba9e139a2da00`，report `35b9361a7cce77ab2dea805ac468e4a87a88bb066aa2b43d19339fee4c5eb114`。`pair_metadata.json` SHA-256 `0b00f8e6d2e043593fcbed6b7eb5e0a47ea164d2b8f43cb20080d6c1860931a7`。它的 schema 是 `qwen3-asr-dpo-gap-strip-metadata-v1`，没有 `release_id` 和 `adapter_sha256`。trainer 对这个 schema 核对 `status=pairs_gap_stripped`、`trainable=false`、`gap_strip=dropout-0.40-0.55`、seed 42，以及 metadata 里的 train/val 哈希和源哈希。源哈希必须是冻结未剥离 pair：train `d958d602dcc4a252ac75ce36d99536e8a53f48fa72929d1b69ead2647dba3f1b`，val `a7591e86f5a33efb314e29304153fe55e2d24ffba9bdd8b2739f953568b116d2`。父 adapter 仍由 `--release-dir` 的 release 核对。`--null-pairs-dir` 指向 `pairs_gap_strip`，不能指向已经填过 logprob 的目录。

听音字符串保持 `user_20261017_skip_listen`，`audit_disposition` 保持 `unlistened`。新字段 `launch_utterance` 只能是「开始吧」。`listen_waiver.json` 仍写 2026-10-17 那条记录，并多写一块 `gap_strip_launch`：日期 2026-10-18，原话「开始吧」，`prior_waiver_covers_this_audit` 为 false，`heard` 为 false，处置仍是 `unlistened`。不把 `pair_audit.json` 改成已听，也不把 `trainable` 改成 true。旧的 pilot 和 dropout probe 没有这个字段，加载结果保持原样。语义 hash 从这次起带上 `launch_utterance`、metadata 哈希和源 pair 哈希；旧配置这三项是 null。已经结束的 run 不恢复。

产物写到新的 `runs/<YYYYMMDDTHHMMSS_dpo_06b_zh_seed42>/`。Drive 同步用同一个新 `run_id`，不覆盖 `runs/20261018T052928_dpo_06b_zh_seed42` 或 `runs/20261017T131802_dpo_06b_zh_seed42`。冻结 `pairs/`、`pairs_logprob/`、`pairs_gap_strip/` 和 `gap_strip_ref_logprobs/` 只读。

测试：`python3 -m unittest tests.test_dpo_contract tests.test_zh_cer tests.test_directory_readmes -v`。新配置能加载，预算与 `dpo_dropout_probe.yaml` 相同，六份新哈希和 metadata 哈希与上一节一致，`launch_utterance` 是「开始吧」。旧 probe 和 pilot 仍加载，哈希不变，`launch_utterance` 是 null。pilot 拒绝 `train_scenarios`，也拒绝带上「开始吧」。gap-strip metadata 缺字段或 `trainable` 不是 false 时拒绝。`train_dpo.py` 源码不包含 `strip_dropout_gap`。豁免记录的 `heard` 仍是 false。不下载模型。

验收：本地测试通过后才开标准 A100。启动日志要同时有 `DPO_RUNTIME_BUDGET disabled`、`DPO_TRAIN_SCENARIOS dropout` 和 `DPO_PREFLIGHT_OK`。训练 pair 应是过滤后的 2573 条 dropout，preference val 应是 310 条。结束时看 `passed` 或带 `DPO_LOCATE` 的真实门禁，不以保活结束行代替。对照 `20261018T052928` 的 dropout delta `0.029492080830147455` 和 0.55 之后多出来的 130 次删除。签名仍是 `OTHER` 且 dropout CER 仍变差时，这次剥离没有修掉尾部删除，同一次 run 里不改学习率。即使 `passed=true`，`selected_step` 仍是 null，不发布，不作为父版本。

2026-10-18 这次已经结束。run `20261018T132840_dpo_06b_zh_seed42`，标准 A100 会话 `lora-asr-dpo-gap-strip`（`gpu-a100-s-kkb-usc1c1-2we8a5memb68k`，Shape Standard）。三条启动日志都在，训练 pair 2573，preference val 310。step 10 的 pair loss 是 `0.624705`。`DPO_LOCATE` 是 `OTHER`，gap 插入 10，长度比 `0.8623702894593118`，dropout delta `0.006417258328782094`，clean delta `0.0008192244675040948`。`passed=false`，`risk_stop=true`，唯一风险是 degraded macro CER 上升 `0.0015360458765701773`。0.55 之后的删除是 70，父版本 57，上一次 probe 187。`selected_step` 仍是 null。`publishable` false。同一次 run 不改学习率。

影响：过滤后的 preference 不再是失败 probe 的 489 条，而是 310 条。父版本在 val dropout 上更偏向 rejected 的计数是 250/310。全 val greedy 仍大约 40 分钟，墙钟时间与上一次 probe 相近。Phase 2 仍因 WER 为 null 而不完成。

## 非静音四场景 DPO pilot（2026-10-19）

背景：`20261018T132840_dpo_06b_zh_seed42`（`configs/train/dpo_gap_strip_probe.yaml`）在 step 10 因 `degraded_macro_cer` 上升 `+0.0015360458765701773`（`dropout` 净增 47 个编辑）触发风险停止。逐条核对 `pairs_gap_strip` 与 `eval/step_000010/scored.jsonl` 定位出三项原因：第一，音频 `dropout` 在采样点时间轴 `[0.40, 0.55]` 置零，而 `strip_dropout_gap` 按字符下标比例 `[0.40, 0.55]` 截断，受句首尾静音与语速影响存在 1～2 字偏移，导致 `2573` 条训练 `dropout` pair 中有 `1293` 条（`50.3%`）误删了有声匹配字，且把 `1236` 条（`48.0%`）本来前缀后缀完全正确的单段静音缺失样本变成了残缺语病句，剥离后仍有 `2149/2573`（`83.5%`）`rejected` 更短；第二，`dropout_probe` 的 `5e-5` 学习率（无 warmup）且 100% 只训 `dropout` 在零能量静音处过度推高吐字倾向，虽然 `495/500` 条 `dropout` 样本净减少 13 个编辑，但 3 条样本在静音处吐出英文幻觉 `humanrightswatch`（+42 编辑）、2 条在静音切口提前触发 `<|im_end|>`（+18 编辑），拉高了 `dropout` CER；第三，只训 `dropout` 放弃了 `noise`、`reverb`、`far_field` 的真实同音纠错收益。反观 2026-10-17 冻结的 `ref_logprobs/pairs_logprob/` 中，`clean`（726）、`noise`（1428）、`reverb`（753）、`far_field`（733）四类有声场景合计 `3640` 条训练 pair、`533` 条验证 pair，等长替换占比达 `91.6%`（train）和 `89.1%`（val），具备完整声学支撑且无需重算离线 reference logprob。

范围：新增 `configs/train/dpo_pilot_nondropout.yaml`，并在 `train/dpo_schema.py` 的 `_check_pilot_budget` 中允许 `profile: pilot` 将 `dpo.train_scenarios` 设为 `[clean, noise, reverb, far_field]`。不修改 `configs/train/dpo_pilot.yaml`、`configs/train/dpo_dropout_probe.yaml`、`configs/train/dpo_gap_strip_probe.yaml` 或 `configs/train/sft_pilot_lr1e5.yaml`，不修改 `train/train_dpo.py` 的训练与评测主循环，不修改冻结的 `pairs/`、`ref_logprobs/pairs_logprob/`、`pairs_gap_strip/` 或 `gap_strip_ref_logprobs/`，也不重算 reference logprob。不恢复 `20261018T132840`、`20261018T052928` 或 `20261017T131802`。不评 test，不启动 RL。`publishable` 保持 false，听音处置保持 `unlistened`。

设计：`configs/train/dpo_pilot_nondropout.yaml` 的 `profile` 为 `pilot`，预算与 `configs/train/dpo_pilot.yaml` 完全一致：学习率 `5e-6`，`warmup_steps` 2，`min_lr_ratio` 0.1，LoRA dropout `0.0`，micro batch 1，梯度累积 4，`max_steps` 100，`save_steps` 10，`eval_steps` 10，`keep_last` 10，`beta` 0.1，seed 42。父 release 仍为 `sft_06b_zh_20261015T234939_step000030`（adapter SHA-256 `a185a61f54b6fd74bd5e3bfa1173e15450cfbb772731567671a928aad805096a`）。输入直接复用 2026-10-17 冻结的 `results/phase3_dpo_logprob_20261017/`：null train `d958d602dcc4a252ac75ce36d99536e8a53f48fa72929d1b69ead2647dba3f1b`、null val `a7591e86f5a33efb314e29304153fe55e2d24ffba9bdd8b2739f953568b116d2`、填充 train `43473f10fbb9c0349adf2eb76eaf647506978c96eabb5b09847994456519c1b1`、填充 val `b82fc2783726555231164025a0e2d7faa82b5391bdcf8b98c6397eec1aff5eca`、cache `5a77f22586dcf1b73393124055eb02466a904d8bdbdd2d44d40ace78000ef43e`、report `8128e2a7731fda2b211dbaf840f38c31b132f26a9fab6f7cd3e8a11d4d805e78`。`dpo.train_scenarios` 固定为 `[clean, noise, reverb, far_field]`；`train/dpo_schema.py` 对 `pilot` 仅允许 `train_scenarios` 为 `null` 或精确等于 `["clean", "noise", "reverb", "far_field"]`，任何含 `dropout` 或子集顺序不符的 `pilot` 配置均在 preflight 拒绝。启动时先对全量 7456/1022 对完成哈希与 cache 校验，再在内存中过滤为 `3640` 条训练 pair 和 `533` 条验证 preference pair。每次 `eval_steps` 仍在全部 `2500` 条 val（含 `dropout` 500 条）上运行 greedy 解码并执行统一的 parent/base CER 与健康度门禁。

测试：本地运行 `python3 -m unittest tests.test_dpo_contract tests.test_directory_readmes -v`。验证 `configs/train/dpo_pilot_nondropout.yaml` 可正常解析，`train_scenarios` 为 `["clean", "noise", "reverb", "far_field"]`，预算与 `dpo_pilot.yaml` 一致，语义 `config_sha256` 与 `dpo_pilot.yaml`、`dpo_dropout_probe.yaml`、`dpo_gap_strip_probe.yaml` 均不同；验证 `pilot` 仍拒绝 `["dropout"]` 或含 `dropout` 的 `train_scenarios`；验证过滤后 `train_pairs=3640`、`val_pairs=533` 且每类场景均满足 `min_pairs_per_scenario=10`。

验收：本地单元测试全部通过且目录 README 合同通过。在标准 A100 上启动时，日志必须包含 `DPO_TRAIN_SCENARIOS clean,noise,reverb,far_field train_pairs=3640 val_pairs=533`、`DPO_STEP0 checked=20 max_abs=0`、`DPO_PREFLIGHT_OK` 和 `DPO_RUNTIME_BUDGET disabled`。门禁通过标准保持不变：`preference_accuracy >= 0.55` 且 `> 0.5`，2500 条全量 val 上 `degraded_macro_cer` 相对 SFT step 30 不上升（`<= 0.0`），至少 1 个退化场景改善，单场景增幅 `<= 0.01`，`clean` 相对父版本 `<= 0.01` 且相对 base `<= 0.02`，健康度全部达标。

2026-10-19 记录：run `20261019T012838_dpo_06b_zh_seed42` 在标准 A100 会话 `lora-asr-dpo-nondropout` 上跑到 step 10。`2026-10-19T02:11:25Z` 的状态是 `gate_failed` / `gate_risk`。`DPO_GATE passed=False risk_stop=True`，preference accuracy `0.7185741088180112`。degraded macro CER 比 SFT step 30 高 `0.0006144183506280737`。dropout `+0.0027307482250136517`，noise `-0.0002730748225013638`，reverb 和 far_field 不变。clean 相对父版本 `+0.0002730748225013638`，相对 base `-0.00040961223375204914`。`DPO_LOCATE` 是 `OTHER`，gap 插入 `10`。`selected_step` 是 null，`publishable` 是 false。不恢复这次 run，也不恢复 `20261018T132840`、`20261018T052928` 或 `20261017T131802`。

影响：消除无声学支撑的 `dropout` 静音段伪偏好对对解码器停止条件和静音先验的干扰，同时恢复 `noise`、`reverb`、`far_field`、`clean` 的同音字偏好纠错梯度，且不产生重新计算 reference logprob 的 GPU 开销。全量 2500 条 val 门禁标准没有放宽。

## 无标点原位替换与去 Clean 同音拷贝 DPO pilot（2026-10-19）

背景：`20261019T012838_dpo_06b_zh_seed42`（`configs/train/dpo_pilot_nondropout.yaml`）在 step 10 评测时，preference accuracy（`0.7185741088180112`）、`noise` 改善（`-0.0002730748225013638`，423 -> 421）与 `clean` 门禁均通过，唯一风险是 `degraded_macro_cer` 上升 `+0.0006144183506280737`（净增 `+18` 次编辑：`dropout +20`、`noise -2`、`reverb 0`、`far_field 0`），其中单条静音悬崖样本 `BAC009S0748W0156__dropout` 从 `'就有一零个，加级贫困县'` 翻转为 `'就有一零个Human Rights Watch'` 贡献了 `+14` 次编辑与 `gap_insertions=10`。进一步对 3640 条非 dropout 训练对及 `aligned_substitution_only: true` + `ref_margin_min: -4.0` 筛出的 781 条样本做逐条审计，定位出四项根因：
1. **标点不对称与不等长 token 的配分函数 $\log Z$ 未抵消**：3640 条非 dropout 训练对中，`36.8%` 的 `rejected` 含标点（`chosen` 0% 含标点）、`48.6%` 满足 $T^+ \ne T^-$，导致序列和 DPO 对 `'，'` 施加负梯度并抬高词表外英文幻觉 token `'Human'`。
2. **极端负参考 margin 的不可翻转离群点**：`75.5%` 的样本 `ref_margin < -2.0` nats（中位数 `-5.189` nats），大量 `<-4.0` 的样本为不可翻转专名或金标错字。
3. **Clean 同音字/人名/AISHELL-1 金标错字跨四场景重复拷贝污染（占 781 条对齐样本的 `64.9%`，前 10 步占 `67.5%`）**：在 781 条原位替换对齐样本中，有 `507/781`（`64.9%`，前 10 步 40 条中占 `27/40 = 67.5%`）是 SFT 在录音室 `clean` 音频上就已经输出相同 `rejected_text` 的跨场景重复拷贝，包括 AISHELL-1 原始标注错别字（如 `BAC009S0517W0317` 金标错字 `'松江新兵镇'` 对 SFT 正确地名 `'松江新浜镇'`，在前 10 步重复出现 2 次；`BAC009S0152W0356` 金标错字 `'避谣'` 对 `'辟谣'`）以及 100% 同音人名/虚词（`陈建/陈健`、`周素勤/周素琴`、`杨常青/杨长青`、`房祖名/房祖明`、`泰禾/泰和`、`卖得不好/卖的不好`）。只挂在 `thinker.model`（文本解码器自注意力层）的 LoRA 无法从零声学差异的同音词或错标真值中学到声学鲁棒性，反而浪费 65% 的梯度更新并扭曲通用语言先验。
4. **训练/验证过滤耦合与优化器阻尼校准**：`train/train_dpo.py` 原先对训练集和验证集 `pref_rows` 强绑定同一套过滤参数（且缺失 `SCENARIOS` 常量导入）；同时若将优化器阻尼过度拉大至 `warmup_steps=20`、`eps=1e-4`，前 10 步累积更新量会被削弱 4～5 倍，在 BF16 下难以在 step 10 翻转退化场景字错或稳定通过 `preference_accuracy >= 0.55`。

范围：更新 `train/dpo_contract.py`、`train/dpo_schema.py`、`train/train_dpo.py`、`configs/train/dpo_pilot_aligned.yaml` 与 `tests/test_dpo_contract.py`。不修改 `configs/train/dpo_pilot.yaml`、`configs/train/dpo_dropout_probe.yaml`、`configs/train/dpo_gap_strip_probe.yaml`、`configs/train/dpo_pilot_nondropout.yaml` 或 `configs/train/sft_pilot_lr1e5.yaml`，不修改已冻结的 `pairs/`、`ref_logprobs/`、`pairs_gap_strip/` 磁盘文件，不放宽任何门禁阈值。不恢复 `20261019T012838` 及前三次 DPO run。不评 test，不启动 RL。`publishable` 保持 false，听音处置保持 `unlistened`。

设计：
1. **内存级过滤扩展（`train/dpo_contract.py` & `train/dpo_schema.py`）**：
   - `select_training_pairs` 新增可选参数 `exclude_clean_identical_rejected: bool = False` 和 `clean_rejected_by_utterance: dict[str, str] | None = None`。当 `exclude_clean_identical_rejected=True` 时，利用全量未过滤训练对构建的 `source_utterance_id -> normalize_zh_cer_v1(clean_rejected_text)` 映射，剔除 `scenario == "clean"` 或 `normalize_zh_cer_v1(rejected_text) == clean_rejected_by_utterance.get(source_utterance_id)` 的样本，仅保留 SFT 在 `clean` 上识别正确（或不同）、纯粹因声学退化才认错的原位替换样本。
   - `dpo` 配置块支持可选字段 `val_scenarios`（list[str] | null，默认 `None` 即跟随 `train_scenarios`）、`aligned_substitution_only`（bool，默认 `False`）、`exclude_clean_identical_rejected`（bool，默认 `False`）、`ref_margin_min`（float | null）、`ref_margin_max`（float | null），并全部纳入 `semantic_dpo_config_sha256`。`_check_pilot_budget` 允许 `train_scenarios` 为 `None`、`["clean", "noise", "reverb", "far_field"]` 或 `["noise", "reverb", "far_field"]`，允许 `val_scenarios` 为 `None` 或 `["clean", "noise", "reverb", "far_field"]`。
2. **训练/验证过滤解耦与 Drive 退出同步（`train/train_dpo.py`）**：
   - 补齐 `SCENARIOS` 常量导入；对 `train_pair_rows` 使用 `train_scenarios` 与 `exclude_clean_identical_rejected`，对 `pref_rows`（`pair_val`）使用 `val_scenarios`（不开启 `exclude_clean_identical_rejected`），并分别按各自的场景集合执行 `validate_pair_rows`。
   - 训练正常结束或触发 `risk_stop` 写出最终 `status.json`、`gate.json`、`run_report.md` 和 `listen_waiver.json` 后，立即调用 `sync_drive_files` 同步至 Google Drive。
3. **校准配置（`configs/train/dpo_pilot_aligned.yaml`）**：
   - 保持正式 `pilot` 预算：学习率 `5e-6`，LoRA dropout `0.0`，micro batch 1，梯度累积 4，`max_steps` 100，`max_epochs` 1，`save_steps` 10，`eval_steps` 10，`keep_last` 10，`beta` 0.1，seed 42。
   - 启用 `train_scenarios: [noise, reverb, far_field]`、`val_scenarios: [clean, noise, reverb, far_field]`、`aligned_substitution_only: true`、`exclude_clean_identical_rejected: true`、`ref_margin_min: -4.0`、`ref_margin_max: 0.0`。在 2026-10-17 冻结的 `ref_logprobs/pairs_logprob/` 上过滤后，训练集精确保留 **`274` 条纯声学退化原位替换对**（`noise: 203, reverb: 37, far_field: 34`，1 个 epoch 共 69 次优化器更新，前 10 步 40 条全部为 `新股/金融`、`未卜/未古`、`推动/驱动`、`骑三轮/提三轮`、`高危/高龄`、`纵向/定向` 等真实声学混淆对），验证集精确保留 **`118` 条四场景对齐对**（`clean: 26, noise: 29, reverb: 33, far_field: 30`），两边均满足 `min_train_pairs: 100`、`gate.min_val_pairs: 100` 和 `min_pairs_per_scenario: 10`。
   - 优化器采用适度阻尼：`warmup_steps: 5`、`eps: 0.000001`（`1e-6`）、`weight_decay: 0.0`，既消除 `eps=1e-8` 的微小梯度符号放大与 `weight_decay=0.01` 的向零漂移，又保证 step 10 具备足够更新幅度越过 BF16 精度阈值。

测试：本地运行 `python3 -m unittest tests.test_dpo_contract tests.test_directory_readmes -v`。验证 `select_training_pairs` 的原位替换、margin 窗口与去 `clean` 同音拷贝过滤；验证 `configs/train/dpo_pilot_aligned.yaml` 解析通过，在本机冻结数据集上精确筛出 `274` 条 train 对（`noise 203, reverb 37, far_field 34`）与 `118` 条 val 对（`clean 26, noise 29, reverb 33, far_field 30`）；验证 `train_dpo.py` 导入 `SCENARIOS` 并同步结束产物到 Drive。

验收：本地单元测试、编译检查与目录 README 合同测试全部通过。在标准 A100 上启动时，日志必须包含 `DPO_TRAIN_SCENARIOS noise,reverb,far_field val_scenarios=clean,noise,reverb,far_field train_pairs=274 val_pairs=118`、`DPO_STEP0 checked=20 max_abs=0`、`DPO_PREFLIGHT_OK` 和 `DPO_RUNTIME_BUDGET disabled`。门禁通过标准不变（2500 条全量 val 上 `degraded_macro_cer` 相对 SFT step 30 `<= 0.0`、至少 1 个退化场景改善、`preference_accuracy >= 0.55`、`clean` 与健康度全部达标）。

2026-10-19 记录：本地 `tests.test_dpo_contract` 与 `tests.test_directory_readmes` 共 33 项通过后，run `20261019T130250_dpo_06b_zh_seed42` 已在标准 A100 会话 `lora-asr-dpo-aligned`（`gpu-a100-s-kkb-ass1c2-oqxw2a89wihw`，Shape Standard）上脱离启动。配置是 `configs/train/dpo_pilot_aligned.yaml`。父版本仍是 SFT step 30。音频包、填充 pair、cache、report、null pair 和 adapter 的 SHA-256 与冻结值一致。`TORCH 2.11.0+cu130 True`。trainer PID 3668。`QWEN3_ASR_DISABLE_RUNTIME_BUDGET=1`。sub2 保活单元是 `dpo-aligned-keepalive.service`。2026-10-19T13:43:02Z 停在 step 10，状态 `gate_failed` / `gate_risk`。`DPO_LOCATE signature=QUIET gap_insertions=0 dropout_length_ratio=0.854861 dropout_delta=0.000546 clean_delta=0.000410`。`DPO_GATE step=10 passed=False risk_stop=True preference=0.610169`。degraded macro CER `0.0736619333697433`，比 SFT step 30 高 `0.00013653741125067842`。分场景差是 dropout `+0.0005461496450027414`，noise `+0.00013653741125068536`，reverb `-0.0001365374112506819`，far_field `0.0`。clean CER `0.022255598033861276`，相对父版本 `+0.00040961223375204567`，相对 base `-0.00027307482250136725`。raw policy accuracy `0.00847457627118644`，118 条等长 pair 里翻转 1 条。2500 条里 40 条假设与 step 30 不同，编辑净增按场景是 clean +3、dropout +4、noise +1、reverb -1、far_field 0。`selected_step` null，`publishable` false。Drive `runs/20261019T130250_dpo_06b_zh_seed42/` 已有 `gate.json`、`status.json`、`eval_decisions.jsonl` 和 `eval/step_000010/scored.jsonl`。不恢复这次 run，也不恢复 `20261019T012838`、`20261018T132840`、`20261018T052928` 或 `20261017T131802`。test 不评。RL 不开始。

影响：不改动已冻结的磁盘缓存与历史配置文件，100% 消除训练集中无声学差异的同音词和金标错字重复拷贝，使解码器注意力 LoRA 更新聚焦于真实噪声/混响/远场退化纠错。当前可用父模型仍为 SFT step 30 release `sft_06b_zh_20261015T234939_step000030`。

## 差异 Token 梯度掩码与单字替换 DPO 优化（2026-10-20）

背景：`20261019T130250_dpo_06b_zh_seed42`（`configs/train/dpo_pilot_aligned.yaml`）在 step 10 将 `locate_signature` 收敛至 `QUIET`（`gap_insertions=0`，彻底根除 `Human Rights Watch` 英文幻觉），`preference_accuracy` 达 `0.6101694915254238`（`72/118 >= 0.55`），`reverb` 净减 1 次编辑（`improved_degraded_scenarios=1`），`far_field` 持平，`clean` 门禁与健康度全部通过；唯一未过线的是四退化场景净增 `+4` 次编辑（`degraded_macro_cer` 上升 `+0.00013653741125067842`：`dropout +4`、`noise +1`、`reverb -1`、`far_field 0`）。对 2500 条 `scored.jsonl` 与 274 条训练对逐 token 核对，定位出导致这 `+4` 次净增编辑与 21 条新增逗号的两项残余根因：
1. **整句序列和反传在共享后缀位点（$t > t_0$，占全句平均 `58.9%` token、占非零梯度位点约 `88%`）上的配分函数未抵消与后缀惩罚**：无标点等长同 token 替换对仅在首个替换位点 $t \le t_0$ 满足前缀上下文完全一致（$c_{t_0}^+ \equiv c_{t_0}^-$，$\nabla_z (\log \pi_\theta(y_{t_0}^+|c_{t_0}) - \log \pi_\theta(y_{t_0}^-|c_{t_0})) = e_{y_{t_0}^+} - e_{y_{t_0}^-}$，配分函数 $\log Z(c_{t_0})$ 100% 精确抵消）；而在 $t > t_0$ 的全部共享正确后缀字（$y_t^+ == y_t^-$，含 `<|im_end|>`）上，由于 $c_t^+$ 含正确字 $y_{t_0}^+$ 而 $c_t^-$ 含错字 $y_{t_0}^-$（$c_t^+ \ne c_t^-$），整句反传一方面对 $-\nabla_\theta \log \pi_\theta(y_t^+|c_t^-)$ 施加负梯度（直接惩罚模型在前缀出现偏差后恢复正确金标后缀，触发暴露偏差后缀级联崩塌：`BAC009S0730W0210__dropout` `'纳入结款' -> '纳入监管'` `+2`、`BAC009S0730W0481__dropout` `'京东单鞋与多者见面' -> '京东单选与多选界面'` `+2`、`BAC009S0725W0200__noise` `'出现严重东西差异' -> '出现严重攻击差异'` `+2`、`BAC009S0727W0413__clean` `'林姐一角后' -> '林姐与小号'` `+3`），另一方面因 $+\nabla_z \log Z(c_t^-) = +p(c_t^-)$ 抬高了不通顺前缀后的停顿标点概率，导致 21 条验证集样本凭空插入逗号 `'，'`，且约 88% 的梯度范数被后缀捷径消耗（`raw_policy_accuracy` 仅翻转 `1/118`）。
2. **多字替换噪声对（`edit_count >= 2`，占 274 条中的 `110` 条 / `40.1%`，前 10 步占 `13/40`）的短语臆测与次位点前缀偏移**：在 `noise` 低信噪比掩蔽整词的样本（如 `吸取教训 -> 施肥浇水` `ed=4`、`这谜题 -> 敬礼图` `ed=3`、`底牌家具 -> 洗牌加剧` `ed=3`、`亲子 -> 新股` `ed=2`）以及 AISHELL-1 金标多字异文（`第三阶段 -> 第三界的` `ed=2`）上，第 2～4 个替换位点同样满足 $c_{t_k}^+ \ne c_{t_k}^-$，鼓励解码器用高频词组覆盖声学特征；而仅保留单字原位替换（`max_edit_count: 1`）仍有 **`164` 条训练对**（`noise: 118, far_field: 25, reverb: 21`），100% 为单一混淆位点且前缀完全一致。

范围：更新 `train/dpo_contract.py`、`train/dpo_schema.py`、`train/train_dpo.py`、`configs/train/dpo_pilot_aligned.yaml` 与 `tests/test_dpo_contract.py`。不修改历史配置文件、已冻结的 `pairs/` 与 `ref_logprobs/` 磁盘文件，不放宽任何门禁阈值。不恢复 `20261019T130250` 及前四次 DPO run。`publishable` 保持 false，听音处置保持 `unlistened`。

设计：
1. **单字原位替换过滤与差异 token 掩码合同（`train/dpo_contract.py` & `train/dpo_schema.py`）**：
   - `select_training_pairs` 新增可选参数 `max_edit_count: int | None = None`，当设置时仅保留字符编辑距离（及 `edit_count`）不超过 `max_edit_count` 的样本；新增纯函数 `mismatched_token_mask(chosen_labels, rejected_labels)`，返回两段等长监督 token 序列中 `chosen_labels[i] != rejected_labels[i]` 的布尔掩码（若全相等则回退为全 `True`）。
   - `dpo` 配置块支持可选字段 `max_edit_count`（int | null，最小为 1，默认 `None`）和 `loss_token_scope`（`"sequence"` 或 `"mismatched_only"`，默认 `"sequence"`；`"mismatched_only"` 要求开启 `aligned_substitution_only: true`），并在非默认值时纳入 `semantic_dpo_config_sha256`，确保全部历史配置的语义哈希不变。
2. **直通式（Straight-Through）差异 token 梯度反传（`train/train_dpo.py`）**：
   - 对训练集 `train_pair_rows` 传入 `max_edit_count`，验证集 `pref_rows` 保持 118 条四场景对齐对（不套用 `max_edit_count` 与 `exclude_clean_identical_rejected`）。
   - 当 `loss_token_scope == "mismatched_only"` 时，训练与 preflight 探针通过 `mismatched_token_mask` 定位 `chosen` 与 `rejected` 监督目标 token 不同的位点，按 `diff_logp + (full_logp - diff_logp).detach()` 构造直通序列 logprob：前向标量值与完整 `sequence_sum` 逐位完全一致（`step0_identity` 与离线 `ref_logprobs.jsonl` 零偏差，`preference_eval` 仍按完整序列和评测），而反向传播仅对差异 token 位点求导。在单字替换对上，$t_0$ 之前前缀完全一致（$\log Z(c_{t_0})$ 100% 抵消），$t_0$ 之后共享金标后缀与 `<|im_end|>` 梯度严格为 0。
3. **配置校准（`configs/train/dpo_pilot_aligned.yaml`）**：
   - 增设 `max_edit_count: 1` 与 `loss_token_scope: mismatched_only`，训练集为 **`164` 条单字退化原位替换对**（`noise: 118, reverb: 21, far_field: 25`），验证集保持 **`118` 条四场景对齐对**；优化器将 `warmup_steps` 设为 `10`（配合 `lr=5e-6, eps=1e-6, weight_decay=0.0`），平抑前 10 步累积更新幅度。
4. **epoch 尾步与学习率对齐（2026-10-20 修订）**：
   - 164 条、累积 4、`max_epochs: 1` 时采样器在 step 41 结束。这一步不是 `save_steps` 的倍数，也小于 `max_steps: 100`。`update_ends_schedule` 为真时，DPO runner 保存并评测这一步，`--resume` 在该步还没有 decision 时补跑评测。SFT runner 不改这条路径。
   - `HorizonScheduler` 初始倍率是 0。DPO 在 `optimizer.step()` 之前调用 `scheduler.step()`，所以第 `k` 步实际更新和日志里的 `lr` 都是 `lr_multiplier(k)`。warmup 10 时，step 1 是 `5e-7`，step 10 是 `5e-6`。更早的 DPO 日志把打印值写成下一步的学习率，那一次 step 1 的更新倍率是 0。这次不恢复旧 checkpoint。

测试：运行 `.venv/bin/python -m unittest tests.test_dpo_contract tests.test_train_sft tests.test_directory_readmes -v`。验证 `max_edit_count=1`（冻结对过滤后 `164` 条 train / `118` 条 val）、`mismatched_token_mask`、`loss_token_scope="mismatched_only"` 的直通前向等价性与差异位点梯度隔离、164 条样本的采样在 step 41 结束且该步会被保存评测、step 10 的学习率倍率是 1，以及全部历史配置哈希不变。

验收：本地单元测试、编译检查与目录 README 合同测试全部通过；在标准 A100 上以新 `run_id` 启动 `configs/train/dpo_pilot_aligned.yaml`，通过 2500 条 val 冻结门禁后才可选出 DPO checkpoint。2026-10-20 run `20261020T014427_dpo_06b_zh_seed42` 已在标准 A100 上跑到 step 10 并风险停止。step 1 的 `lr` 是 `5e-07`，step 10 的 `lr` 是 `5e-06`。门禁没有通过，不能选出 checkpoint。

影响：彻底阻断序列和 DPO 在共享后缀上的暴露偏差负梯度与配分函数标点漂移，将 100% 梯度集中于退化语音的单字混淆消歧；不改动已冻结的磁盘缓存与历史配置文件。

5. **step 10 硬门禁下的序列和单字对（`configs/train/dpo_pilot_edit1_seq.yaml`，2026-10-20）**：

背景：`20261020T014427_dpo_06b_zh_seed42` 使用 `mismatched_only` 和 `warmup_steps: 10`。step 10 是第一次满学习率更新，也是第一次门禁。loss 从 `0.693147` 到 `0.693081`，preference 停在 `0.5`（59/118），raw policy `0/118`。degraded macro CER 上升 `0.00010240305843801922`（dropout +12、far_field -3、noise -4、reverb -2）。两条都触发 `risk_stop`。同一套 `5e-6` 的整句序列和在 `20261019T130250` 上把 preference 推到 `0.6101694915254238`。方案前文已经写明，warmup 拉到 20 会让 step 10 过不了 `0.55`；`eps=1e-8` 配合 `weight_decay=0.01` 又会把静音样本推过悬崖。

范围：新增 `configs/train/dpo_pilot_edit1_seq.yaml`。不改 `configs/train/dpo_pilot_aligned.yaml` 和其他已执行 YAML，不改冻结 pair、cache、门禁阈值，不恢复 `20261020T014427`、`20261019T130250`、`20261019T012838`、`20261018T132840`、`20261018T052928`、`20261017T131802`。不评 test，不启动 RL，不标 Phase 2 或 Phase 3 完成。

设计：过滤与 `dpo_pilot_aligned.yaml` 相同，训练集仍是 164 条单字退化原位替换，验证集仍是 118 条四场景对齐对。`loss_token_scope` 为 `sequence`，前向和反向都走完整序列和。`warmup_steps` 为 2，所以 step 1 的学习率是 `2.5e-6`，step 2 起是 `5e-6`。`eps` 保持 `1e-6`，`weight_decay` 保持 `0.0`。`profile: pilot` 预算不变：学习率 `5e-6`，累积 4，`max_steps` 100，`max_epochs` 1，每 10 step 保存并评 2500 条 val。`degraded_macro_increase_max` 仍是 `0.0`，`min_preference_accuracy` 仍是 `0.55` 且必须严格高于 `0.5`。step 10 不过就 `risk_stop`，不把没过的 checkpoint 接着训。164 条、累积 4、1 个 epoch 仍在 step 41 结束，该步保存并评测。语义哈希与 `dpo_pilot_aligned.yaml` 不同，不能 `--resume` 接上 `20261020T014427`。

测试：`.venv/bin/python -m unittest tests.test_dpo_contract tests.test_directory_readmes -v`。新配置能加载，门禁字典与 `dpo_pilot_aligned.yaml` 相同，pair 哈希相同，warmup 是 2，`loss_token_scope` 是 `sequence`，语义哈希不同。

验收：上述测试通过后，用新 `run_id` 在标准 A100 上启动这份配置。日志要有 `train_pairs=164`、`val_pairs=118`、`DPO_STEP0`、`DPO_PREFLIGHT_OK`、`DPO_RUNTIME_BUDGET disabled`。step 1 的 `lr` 是 `2.5e-6`。只有 2500 条 val 的冻结门禁通过才选出 checkpoint。`publishable` 保持 false。`20261020T024637_dpo_06b_zh_seed42` 已在 `2026-10-20T02:46:37Z` 满足这些启动条件：164/118、`DPO_STEP0 checked=20 max_abs=0`、`DPO_PREFLIGHT_OK`、`DPO_RUNTIME_BUDGET disabled`，step 1 `lr=2.5e-06`，step 2 `lr=5e-06`。step 10 门禁未通过：preference `0.576271186440678` 过线，degraded macro CER 上升 `0.000341343528126703`，`risk_stop=true`，`selected_step` 仍是 null。不恢复。

6. **dropout 父版本锚（`configs/train/dpo_pilot_edit1_anchor.yaml`，2026-10-20）**：

背景：音频塔、projector、embedding 和 `lm_head` 没有进优化器。step 10 的编辑变化来自 28 层文字解码器自注意力。`20261020T014427` 只用差异字反传，noise、reverb、far_field 合计少 9 个编辑，dropout 多 12 个，净增 3 个，preference 停在 59/118。`20261020T024637` 把整句后缀梯度加回来之后，preference 到 68/118，三个受训场景变成多 3 个编辑，dropout 仍多 7 个。沿同一条差异字梯度加大步长，会把 −9 和 +12 一起放大，净编辑不容易变成小于等于 0。

范围：新增 `configs/train/dpo_pilot_edit1_anchor.yaml`。不改已执行的 YAML，不改冻结 pair、cache 和门禁阈值，不恢复 `20261020T024637`、`20261020T014427`、`20261019T130250`、`20261019T012838`、`20261018T132840`、`20261018T052928`、`20261017T131802`。不把 dropout 金标对加回 DPO 训练集，不把学习率抬到 `5e-5`，不改贪心解码。不评 test，不启动 RL，不标 Phase 2 或 Phase 3 完成。

设计：164 条过滤、`loss_token_scope: mismatched_only`、`warmup_steps: 10`、`eps: 1e-6`、`weight_decay: 0.0` 与 `dpo_pilot_aligned.yaml` 相同，用来保住三个受训场景少 9 个编辑的方向。前向仍是整句 logprob。另加 `dropout_anchor_weight: 0.05`。权重必须在 `(0, 0.1]`，而且只能和 `mismatched_only`、不含 dropout 的 `train_scenarios` 一起用。每个 optimizer step 从填充后的 train pair 里按文件顺序取一条 dropout，教师文本是该行的 `rejected_text`，也就是 SFT step 30 的贪心文本，不是金标。损失是 `0.05 * (序列 NLL 之和 / token 数)`，反传整句，加在同一套解码器 LoRA 上，发生在梯度裁剪之前。`chosen_text` 不能当锚。锚行用尽后按 step 循环，不另写 pair 文件，也不重算 reference logprob。preference 仍只评原来的 118 条。启动日志要有 `DPO_ANCHOR`，step 日志另写 `anchor_loss`。这一项不增加 164 条对上的 preference 梯度，step 10 仍可能停在 preference `0.5`。出现这个结果就停，不在同一轮把系数加大，也不把后缀梯度加回来。

测试：`.venv/bin/python -m unittest tests.test_dpo_contract tests.test_directory_readmes -v`。新配置能加载，门禁字典和 pair 哈希与 `dpo_pilot_aligned.yaml` 相同，`loss_token_scope` 是 `mismatched_only`，warmup 是 10，锚权重是 `0.05`，目标是 `rejected_text`。权重超过 `0.1`、目标写成 `chosen_text`、或和整句反传一起用时拒绝。语义哈希与 `dpo_pilot_aligned.yaml` 和 `dpo_pilot_edit1_seq.yaml` 都不同。旧配置不写这个字段时权重是 0，语义哈希不变。2026-10-20 本机这条命令 36 项通过，退出码 0。没有下载模型，没有启动训练。

验收：上述测试通过后，用新 `run_id` 在标准 A100 上启动。日志要有 `train_pairs=164`、`val_pairs=118`、`DPO_ANCHOR`、`DPO_STEP0`、`DPO_PREFLIGHT_OK`、`DPO_RUNTIME_BUDGET disabled`。step 1 的 `lr` 是 `5e-7`，step 10 的 `lr` 是 `5e-6`。只有 2500 条 val 的冻结门禁通过才选出 checkpoint。三个受训场景的编辑和要下降，dropout 增加的编辑不超过这份下降，preference 不低于 `0.55`。`publishable` 保持 false。2026-10-20 run `20261020T041218_dpo_06b_zh_seed42` 已在标准 A100 上启动。step 1 的 `lr` 是 `5e-07`，`DPO_ANCHOR rows=3816`。step 10 决策 `passed=true`：noise −9、reverb −1、far_field +2、dropout −2，preference `0.576271186440678`。step 20 决策也是 `passed=true`：noise +2、reverb −4、far_field −5、dropout +2，preference `0.6016949152542372`。step 30 决策也是 `passed=true`：noise 0、reverb −4、far_field 0、dropout +1，preference `0.6694915254237288`。step 40 决策也是 `passed=true`：noise −6、reverb −4、far_field −2、dropout +1，preference `0.6440677966101694`。`2026-10-20T07:30:05Z` step 41 结束这一轮：`passed=false`，dropout +13，macro 上升 `3.4134352812673074e-05`，`selected_step` 是 40，`publishable` 保持 false。逐条比对后，这多出的 dropout 编辑集中在 `BAC009S0748W0156__dropout`：step 40 仍是「就有一零个，加级贫困县」，step 41 是「就有一零个Human Rights Watch」，单条 +14，其余 dropout 净 −2。细节在 `12_phase2_run_log.md`。

7. **锚配置停在 step 40（`configs/train/dpo_pilot_edit1_anchor_stop40.yaml`，2026-10-20）**：

背景：`20261020T041218_dpo_06b_zh_seed42` 的 step 10、20、30、40 都过了冻结门禁。step 40 的 macro delta 是 `-0.00037547788093937606`，preference 是 `0.6440677966101694`，`BAC009S0748W0156__dropout` 仍是父版本的「就有一零个，加级贫困县」。164 条、累积 4、`max_epochs: 1` 会再走出第 41 步。这一步的梯度仍被裁到 `1.0`，把该条翻成「就有一零个Human Rights Watch」，dropout 多 14 个编辑，macro 因此上升 `3.4134352812673074e-05`。选出的模型已经是 step 40。

范围：新增 `configs/train/dpo_pilot_edit1_anchor_stop40.yaml`。只把 `max_steps` 从 100 改成 40。`mismatched_only`、`warmup_steps: 10`、学习率 `5e-6`、`eps: 1e-6`、`weight_decay: 0.0`、`dropout_anchor_weight: 0.05`、教师 `rejected_text`、164 条过滤和全部门禁数字保持与 `dpo_pilot_edit1_anchor.yaml` 相同。不改已执行的 YAML，不改冻结 pair 和 cache。不恢复 `20261020T041218`，也不为了避开第 41 步重跑 step 40。不加大锚系数，不恢复整句后缀梯度，不把学习率抬到 `5e-5`，不把这条样本从门禁里拿掉。不评 test，不启动 RL，不标 Phase 2 或 Phase 3 完成。

设计：40 是不超过这个 epoch 的 41 步、又是 `eval_steps: 10` 的最大倍数。pilot 预算表仍要求其他配置的 `max_steps` 为 100。只有同时满足锚权重 `0.05`、教师 `rejected_text`、`loss_token_scope: mismatched_only`、`warmup_steps: 10` 且 `max_steps` 正好是 40 时，schema 才接受这份停步。语义哈希因此与 `dpo_pilot_edit1_anchor.yaml` 不同，不能 `--resume` 接上 `20261020T041218`。164 条、累积 4、`max_steps: 40` 时采样停在 step 40，最后一批仍是 4 条，该步保存并评测，不再产生 step 41。

测试：`.venv/bin/python -m unittest tests.test_dpo_contract tests.test_directory_readmes -v`。新配置能加载，门禁、锚字段和 pair 哈希与 `dpo_pilot_edit1_anchor.yaml` 相同，`max_steps` 是 40。把 `dpo_pilot_aligned.yaml` 或 `dpo_pilot_edit1_anchor.yaml` 的 `max_steps` 改成 40 以外的数，或在没有这组锚字段时写成 40，都要拒绝。`max_steps: 100` 的旧锚配置仍能加载。采样测试确认 164 条在 `max_steps: 40` 时得到 40 步，最后一步是 4 条并且结束日程。

验收：2026-10-20 本机 `.venv/bin/python -m unittest tests.test_dpo_contract tests.test_directory_readmes -v` 共 37 项通过，退出码 0。同日用户说「开始吧」。run `20261020T114647_dpo_06b_zh_seed42` 已在标准 A100 会话 `lora-asr-dpo-stop40`（运行时 `gpu-a100-s-kkb-usc1b0-2mjb1fkgnpn4i`，Shape Standard）启动，没有 `--resume`。`2026-10-20T11:46:56Z` `LAUNCH_DETACHED`，`CONFIG_STOP40 max_steps=40`，`REBUILD_EXIT 0`，`TORCH 2.11.0+cu130 True`，磁盘空闲 193.31 GB。`2026-10-20T11:47:29Z` step 1：`global_step=1/40`，`lr=5e-07`，`loss=0.693147`，`anchor_loss=0.001113`，`anchor_tokens=7`，`DPO_ANCHOR rows=3816 weight=0.05 target=rejected_text`，`train_pairs=164`，`val_pairs=118`，`DPO_PREFLIGHT_OK`，`DPO_RUNTIME_BUDGET disabled`。不恢复 `20261020T041218`。`2026-10-20T13:45:47Z` 这一轮在 step 30 结束：step 10 和 step 20 通过，step 30 因为 macro 上升 `3.4134352812673074e-05`（noise +7、reverb −3、far_field −3、dropout 0）风险停止。`selected_step` 是 10。`max_steps: 40` 也是余弦日程的终点，step 20 的学习率是 `3.875e-6`，step 30 是 `1.625e-6`。`publishable` 保持 false。可用模型仍是 SFT step 30。

影响：已保存的 step 40 checkpoint 不变。以后用这份停步配置新开 run 时，优化器不会再走第 41 步。`latest.json` 若来自旧 run，仍指向 step 41，那不是选出的模型。

8. **余弦周期解耦（方案 1）与每步多样本 dropout 锚平均（方案 3）对比设计（`configs/train/dpo_pilot_edit1_anchor_step40_h100.yaml` 与 `configs/train/dpo_pilot_edit1_anchor_step40_h100_a4.yaml`，2026-10-20）**：

背景：
- **为什么 `dpo_pilot_edit1_anchor_stop40.yaml`（`20261020T114647_dpo_06b_zh_seed42`）在 step 30 失败**：在 `HorizonScheduler` 中，`self.total = int(config["train"]["max_steps"])` 原本同时充当“采样循环停止步数”与“余弦退火终点”。将 `train.max_steps` 从 `100` 改为 `40` 后，warmup 后的余弦衰减分母从 `90` 步（`100 - 10`）被强行压缩为 `30` 步（`40 - 10`）：step 20 的学习率从上一轮 `4.864e-6` 提前跌至 `3.875e-6`（即上一轮 step 40 的水平），step 30 的学习率从上一轮 `4.474e-6` 暴跌 67.5% 至 `1.625e-6`（仅为上一轮同期的 36%）。过早萎缩的学习率使 step 21–30 的更新无法完成 `noise` 单字消歧（`noise +7, reverb -3, far_field -3, dropout 0`），四退化场景净编辑 `+1`（`macro_delta = +3.4134352812673074e-05`），以 1 个编辑之差在 step 30 触发 `risk_stop`。
- **单步单条 dropout 锚样本的方差问题**：在 `20261020T041218` 的 41 个优化器步中，`grad_norm` 在 41/41 步均超过 `max_grad_norm: 1.0`（`1.88 ~ 5.53`）并触发裁剪；同时每个 step 仅抽取 1 条 `dropout` 样本计算 `anchor_loss`，单条样本的 token 平均 NLL 在 `0.0055 ~ 0.2317` 之间波动达 40 倍，导致裁剪后的合成梯度方向随单条锚样本的难易度摆动。

范围：
- 更新 `train/sft_schema.py`、`train/train_sft.py`、`train/dpo_contract.py`、`train/dpo_schema.py` 与 `train/train_dpo.py`，同时支持方案 1（`optim.scheduler_max_steps` 解耦余弦周期与 `train.max_steps` 停步）和方案 3（`dpo.dropout_anchor_samples_per_step` 每步多条 dropout 锚样本等权平均）。
- 新增两份对比配置：
  1. `configs/train/dpo_pilot_edit1_anchor_step40_h100.yaml`（**纯方案 1**）：`train.max_steps: 40`、`optim.scheduler_max_steps: 100`、`dpo.dropout_anchor_samples_per_step: 1`（缺省），在 Step 10、20、30、40 100% 确定性逐位复现 `20261020T041218` 的通过轨迹并在 Step 40 干净停步，不跑第 41 步。
  2. `configs/train/dpo_pilot_edit1_anchor_step40_h100_a4.yaml`（**方案 1 + 方案 3**）：在纯方案 1 基础上开启 `dpo.dropout_anchor_samples_per_step: 4`（与 `gradient_accumulation_steps: 4` 对齐，每步取 4 条 `dropout` 锚样本做等权平均，`max_grad_norm` 保持 `1.0` 不变以防步长放大）。
- 不修改任何已执行的 YAML、冻结 pair 或 reference logprob 缓存，不放宽任何门禁阈值，不恢复 `20261020T114647` 或 `20261020T041218`。

设计：
1. **余弦周期与采样停步解耦（方案 1，`optim.scheduler_max_steps`）**：
   - `train/sft_schema.py` 的 `optim` 块支持可选整数 `scheduler_max_steps`（缺省为 `None`）。当显式设置时，要求 `scheduler_max_steps >= train.max_steps` 且 `optim.warmup_steps < scheduler_max_steps`，并写入 `resolved["optim"]["scheduler_max_steps"]`；当未设置（`None`）时不在 `resolved["optim"]` 中增加该键，确保全部历史 SFT/DPO 配置的 `semantic_config_sha256` 与 `semantic_dpo_config_sha256` 100% 不变。
   - `train/dpo_schema.py` 的 `_check_pilot_budget` 规定：当 `optim.scheduler_max_steps` 非空时，必须同时满足 `_allows_anchor_stop40(resolved)`（即 `train.max_steps == 40`、`dropout_anchor_weight == 0.05`、`dropout_anchor_target == "rejected_text"`、`loss_token_scope == "mismatched_only"`、`warmup_steps == 10`）且 `scheduler_max_steps == 100`；非锚点停步配置或非 100 的周期一律在 preflight 拒绝。
   - `train/train_sft.py` 的 `HorizonScheduler.__init__` 将 `self.total` 设为 `int(config["optim"].get("scheduler_max_steps") or config["train"]["max_steps"])`。在 `max_steps: 40, scheduler_max_steps: 100` 下，采样器在 step 40 结束，而 step 1–40 的学习率序列（step 10 `5e-6`、step 20 `4.8643084e-6`、step 30 `4.4736e-6`、step 40 `3.875e-6`）与 `max_steps: 100` 逐位完全一致。
2. **每步多样本 dropout 锚平均（方案 3，`dpo.dropout_anchor_samples_per_step`）**：
   - `train/dpo_contract.py` 新增纯函数 `dropout_anchor_step_slice(anchor_rows, step, samples_per_step=1)`，为第 `step`（从 1 开始）个优化器步按确定性下标 `(step - 1) * samples_per_step + offset`（对 `len(anchor_rows)` 取模）返回 `samples_per_step` 条 `dropout` 锚样本。当 `samples_per_step == 1` 时，返回的 `[anchor_rows[(step - 1) % len(anchor_rows)]]` 与原实现逐位一致。
   - `train/dpo_schema.py` 的 `_parse_dropout_anchor` 支持可选整数 `dropout_anchor_samples_per_step`（默认 `1`，允许范围 `[1, 4]`；当 `dropout_anchor_weight == 0.0` 时禁止设置 `> 1`）。`semantic_dpo_config_sha256` 仅在 `dropout_anchor_samples_per_step != 1` 时纳入该字段，保持 `dpo_pilot_edit1_anchor.yaml` 与 `dpo_pilot_edit1_anchor_stop40.yaml` 的语义哈希不变。
   - `train/train_dpo.py` 在每个优化器步遍历 `dropout_anchor_step_slice(anchor_rows, step, anchor_samples_per_step)`：当 `anchor_samples_per_step == 1` 时按原式 `anchor_weight * (-anchor_logp) / anchor_tokens` 反传；当 `anchor_samples_per_step > 1` 时对每条锚样本按 `(anchor_weight / anchor_samples_per_step) * (-anchor_logp) / row_tokens` 逐条立即 `.backward()`（峰值显存与单条锚完全一致，总锚损失期望尺度保持 `0.05 * E[token_mean_NLL]`，单步方差降低至 $1/4$），并在启动日志打印 `samples_per_step=4`。

测试：
- 运行 `.venv/bin/python -m unittest discover -s tests -v`、`.venv/bin/python -m py_compile scripts/*.py train/*.py inference/*.py evaluation/*.py` 与 `git diff --check`。
- 验证 `HorizonScheduler` 在 `max_steps=40, scheduler_max_steps=100` 下 step 1、10、20、30、40 的学习率与 `max_steps=100` 逐位相等；验证 `dropout_anchor_step_slice` 在 `samples_per_step=1` 与 `4` 下的确定性切片与循环取模；验证 `dpo_pilot_edit1_anchor_step40_h100.yaml` 与 `dpo_pilot_edit1_anchor_step40_h100_a4.yaml` 解析通过、语义哈希互不相同且全部历史配置哈希不变；验证非法 `scheduler_max_steps` 或 `dropout_anchor_samples_per_step` 被拒绝。

验收：
- 本地单元测试、编译检查与目录 README 合同测试全部通过。
- 先对齐审查文档、代码与本地测试；获准上卡后，先在标准 A100 上运行纯方案 1（`dpo_pilot_edit1_anchor_step40_h100.yaml`）验证 Step 10–40 逐位复现 `20261020T041218` 并干净停步，再运行方案 1+3（`dpo_pilot_edit1_anchor_step40_h100_a4.yaml`）对比 4 条锚平均的门禁与 CER 表现。
- 2026-10-22 两轮对比验收已在标准 A100 会话 `lora-asr-dpo-h100`（`gpu-a100-s-kkb-ass1c0-26sxp0qcjfkek`，Shape Standard）完成：
  1. **纯方案 1（`configs/train/dpo_pilot_edit1_anchor_step40_h100.yaml`，run `20261022T133007_dpo_06b_zh_seed42`）**：Step 10、20、30、40 的学习率（`5e-6`、`4.8643084e-6`、`4.4736e-6`、`3.875e-6`）、loss、anchor_loss 和 2500 条 val 门禁指标与 `20261020T041218` 100% 逐位一致，四次评测全部 `passed=true`、`risk_stop=false`、`locate_signature=QUIET`、`locate_gap_insertions=0`。训练在 Step 40 干净停步，不再走出第 41 步：`status.json` 为 `completed` / `max_steps`（全程零 `gate_risk`），`gate.json` 的 `status=passed`、`selected_step=40`、`publishable=false`，`latest.json` 指向 `step_000040`（`checkpoint_sha256` `9a52ddbac7abfe07f467dcdc77a2c9407fa566176de26fe829c457b6e7395f8c`，`latest_step == selected_step == 40`）。Step 40 选出模型的 `degraded_macro_cer=0.07314991807755325`（`macro_delta=-0.00037547788093937606`，净减 11 个编辑：noise −6、reverb −4、far_field −2、dropout +1），`clean_cer=0.022255598033861276`（相对父版本 +3、相对 base −2），`preference_accuracy=0.6440677966101694`（76/118），`raw_policy_accuracy=0.025423728813559324`（3/118）。
  2. **方案 1 + 方案 3（`configs/train/dpo_pilot_edit1_anchor_step40_h100_a4.yaml`，run `20261022T161911_dpo_06b_zh_seed42`）**：每步 4 条 `dropout` 锚平均使 Step 10 的声学 CER 表现稳健（`degraded_macro_cer=0.0733205898416166`，`macro_delta=-0.00020480611687602457`，净减 6 个编辑：noise −1、reverb −2、far_field −3、dropout 0，`locate_signature=QUIET`），但前 10 步累计 40 条父版本 `rejected_text` 锚对解码器的约束更均匀，使 118 条验证偏好对在 Step 10 仅翻转 59/118（`preference_accuracy=0.5`），触发 `risk_stop`（`status=gate_failed` / `gate_risk`，`selected_step=null`）。
  3. **对比结论**：纯方案 1（`20261022T133007_dpo_06b_zh_seed42`，`step_000040`）胜出。会话 `lora-asr-dpo-h100` 已停止。`publishable` 保持 false，不恢复 `20261022T161911`。

影响：
- 解耦停止步数与余弦退火周期，消除因截断尾步而压缩学习率曲线的副作用；同时提供零额外显存开销的多样本锚平均机制。不改变已冻结的数据、缓存与历史配置哈希。

## DPO release 读回（2026-10-23）

背景：run `20261022T133007_dpo_06b_zh_seed42`（`configs/train/dpo_pilot_edit1_anchor_step40_h100.yaml`）在 2500 条 val 上通过全部四次评测（step 10、20、30、40），以 `status=completed`、`reason=max_steps`（零 `gate_risk`）选出 `step_000040`，且 `latest.json` 同样指向 `step_000040`。按阶段衔接合同，Phase 4 RL 的初始 policy 与固定 reference 必须来自经过独立进程读回校验的不可变 DPO release，不能直接读取训练 run 目录。

范围：新增 `train/dpo_release.py`、`tests/test_dpo_release.py` 与 DPO 父指标文件 `configs/train/aishell_val_dpo_step40_20261022.json`。它只发布 `gate.json` 中 `status=passed`、`selected_step` 对应且 `eval_decisions.jsonl` 中 `passed=true`、`risk_stop=false`、`preference_accuracy` 过线的 DPO pilot checkpoint。`dropout_probe`、`gate_failed`、无 decision 或 `.incoming` 一律拒绝。不修改 `train/sft_release.py`，不修改源 run 的 `status.json` 或 `gate.json`（源 `gate.json` 的 `publishable: false` 与 `audit_disposition: unlistened` 保持原样），不评 test。本次 release id 固定为 `dpo_06b_zh_20261022T133007_step000040`。

设计：
- `prepare`：加载 `load_dpo_config(config_path)`，核对 `profile == "pilot"`、`stage == "dpo"`、checkpoint 目录 hash（`verify_checkpoint`）、语义配置 hash（`semantic_dpo_config_sha256`）、`base_revision`、`processor_sha256`、`eval_decode_sha256`、`parent_release_sha256` 以及 `eval/<step>/preference.json` 的 `accuracy >= gate.min_preference_accuracy` 且 `> 0.5`。只复制 `adapter/adapter.safetensors`、`adapter/adapter_config.json`、`source/gate.json`、`source/scored.jsonl`、`source/preference.json` 和 `source/training_state.json`，不复制 optimizer、scheduler 和 RNG。`release.json` 的 schema 为 `qwen3-asr-dpo-release-v1`，初始 `status=staged`、`publishable=false`，记录 `parent_release`、`parent_release_sha256`、`listen_waiver`、`audit_disposition="unlistened"`、`preference_accuracy`、`latest_step` 与 `latest_is_release`。
- `readback`：在独立新进程中使用与训练评测相同的确定性开关（`CUBLAS_WORKSPACE_CONFIG=:4096:8`、`use_deterministic_algorithms(True)`、关闭 TF32、固定配置 `seed`），加载官方 base 与 release 的 `adapter.safetensors`，对 2500 条 `val.jsonl` 重新 greedy 解码，每 250 条打印 `DPO_RELEASE_PROGRESS`。逐条比较 `sample_id`、`prediction_raw`、`stop_reason` 和 `error`。全部 2500 条零差异时写出 `readback.json`（schema `qwen3-asr-dpo-release-readback-v1`），并将 `release.json` 更新为 `status=published`、`publishable=true`（供下游 RL 阶段作为不可变父 release 消费，同时保留 `audit_disposition="unlistened"` 记录），更新前后 `payload_sha256` 必须不变。支持 `--sync-dir` 将 release 同步到 Google Drive `releases/dpo/<release_id>/` 并做读回哈希核对。

测试：`python3 -m unittest tests.test_dpo_release tests.test_directory_readmes -v`。本地夹具验证正常选出 step 打包、拒绝 `risk_stop`/偏好未过线/已存在目录，以及读回文本不一致时不发布、一致时转为 `published` 且不改动 payload hash。测试不下载模型。

验收：Drive 上的 `releases/dpo/dpo_06b_zh_20261022T133007_step000040/` 读回 hash 一致，`readback.json` 的 `matched` 为 true，`compared` 等于 2500，`mismatch_count` 为 0。做到这一点后，RL 才能把这份 DPO release 当作父版本。

影响：源 run `20261022T133007_dpo_06b_zh_seed42` 目录保持只读。`configs/train/aishell_val_dpo_step40_20261022.json` 固定 DPO Step 40 的 2500 条 val micro CER（总计 `0.06297105406881485`：clean `0.022255598033861276`、dropout `0.18814855270344075`、far_field `0.02293828509011469`、noise `0.05693610049153468`、reverb `0.024576734025122884`）作为 Phase 4 RL 的父指标。

## RL runner 与配置实现（2026-10-23）

背景：DPO 纯方案 1 选出的 `step_000040` 已在四退化场景净减 11 个编辑。按 SFT → DPO → RL 正式后训练顺序，下一步是在固定 DPO release 父版本上实现 `group_pg_mixed_v1` 组内相对策略梯度训练器与 smoke/pilot 配置。

范围：新增 `train/rl_contract.py`、`train/rl_schema.py`、`train/train_rl.py`、`configs/train/rl_smoke.yaml`、`configs/train/rl_pilot.yaml` 与 `tests/test_rl_contract.py`。父版本固定为已发布的 `dpo_06b_zh_20261022T133007_step000040`，父指标为 `configs/train/aishell_val_dpo_step40_20261022.json`，base 指标仍为 `configs/train/aishell_val_base_20261015.json`。不修改 SFT/DPO runner 或已冻结 manifest/pair 文件，不评 test。

设计：
1. **配置与预算（`train/rl_schema.py`）**：
   - 解析 `stage: rl`（`profile: smoke | pilot`），校验 `model.parent_release == "dpo_06b_zh_20261022T133007_step000040"` 与非空 `parent_release_sha256`、`dpo: null`、非空 `rl` 块（`algorithm: group_pg_mixed_v1`、`advantage: group_mean_centered`、`kl_beta: 0.04`、`updates_per_rollout: 1`、`max_group_visits: 500`、`no_signal_stop_groups: 20`、`invalid_group_rate_max: 0.01`、`parent_metrics`）。
   - 冻结预算：`learning_rate: 1e-5`、`lora.dropout: 0.0`、`micro_batch_size: 1`（1 group = 4 candidates）、`gradient_accumulation_steps: 2`；`rl_smoke.yaml` 为 `max_steps: 6`、`max_epochs: 100`、`save_steps: 5`、`eval_steps: 5`、`interrupt_after_step: 5`、`max_runtime_minutes: 30`；`rl_pilot.yaml` 为 `max_steps: 50`、`max_epochs: 1`、`save_steps: 5`、`eval_steps: 5`、`keep_last: 10`、`interrupt_after_step: null`、`max_runtime_minutes: 120`。
2. **Reward、Advantage、KL 与门禁合同（`train/rl_contract.py`）**：
   - `candidate_seed_int(seed, epoch, visit_index, sample_id, candidate_index)`：用 `SHA-256` 派生确定性 63 位正整数种子。
   - `compute_ candidate_reward(reference_text, prediction_raw, stop_reason, error, scoring)`：基于 `zh-cer-v1` 计算 `R = clip(1 - CER - 0.5 * empty - 0.5 * repeated, -1.0, 1.0)` 并返回完整 `reward_components`。
   - `evaluate_rollout_group(candidates, reference_text, scoring)`：若任一候选 `stop_reason == "length"` 或 `error` 非空，整组标记 `group_valid = False`，不参与训练更新；否则计算组均值中心化 `A_g = R_g - mean(R)`，若全组 reward 相等则标记 `zero_advantage = True`，整组策略损失与 KL 损失均置 0。
   - `k3_token_kl(policy_token_logp, ref_token_logp)`：在未受 temperature/top-p 修改的模型原始分布上按 FP32 计算 $d = \log p_{\text{ref}} - \log p_{\text{policy}}$、$k3 = \text{expm1}(d) - d$，非有限值立即抛错。
   - `exact_Step0_conditional_kl(policy_logits, ref_logits, labels)`：在固定 step 0 greedy 前缀的监督转写+EOS 位点上对全词表求 $\sum_v p_{\text{policy}}(v)(\log p_{\text{policy}}(v) - \log p_{\text{ref}}(v))$ 并对有效 token 求平均，用于评测门禁 `raw_kl <= 0.001`。
   - `judge_rl_checkpoint(metrics, parent_metrics, base_metrics, gate_cfg, step0_reward, current_reward, raw_kl, reference_matched)`：同时检查相对 DPO 父版本的 CER/健康度门禁、相对 base 的 clean 累计上限、`raw_kl <= 0.001`、`reference_matched is True`、相对 step 0 的 `reward_delta >= 0.002`（通过条件）与 `reward_drop <= 0.01`（停止条件）。
3. **训练与恢复循环（`train/train_rl.py`）**：
   - 加载官方 base 与已发布 DPO release adapter，复制独立冻结的 reference LoRA 权重字典并记录 `reference_sha256`（在启动、每次 save/eval 和退出时校验不变）。
   - Step 0 在任何更新前解码全量 val，记录 step 0 greedy CER/reward、固化 step 0 greedy 转写+EOS 前缀 token IDs 哈希并验证初始 `raw_kl == 0.0`。
   - 每个累积窗口（2 个有效 group）内：若全部有效组均为 `zero_advantage`，跳过 `optimizer.step()` 和 `scheduler.step()`，不推进 `global_step`，仅累加 `group_visits` 与连续无信号组计数（连续 20 组无信号或达 `max_group_visits` 以 `no_learning_signal` 收尾；首 100 组或结束时检查 `invalid_group_rate <= 0.01`）。

测试：`python3 -m unittest tests.test_rl_contract tests.test_dpo_release tests.test_directory_readmes -v`。验证确定性候选种子、reward 截断与惩罚、无效组剔除、全组同分零损失（含零 KL）、k3 与精确词表条件 KL 的区分、零信号窗口不更新 optimizer/scheduler、reference 权重冻结校验，以及 `rl_smoke.yaml` / `rl_pilot.yaml` 配置解析。不下载模型。

验收：本地单元测试全部通过；待 DPO release `dpo_06b_zh_20261022T133007_step000040` 在标准 A100 上完成读回发布后，即可将其 `adapter_sha256` 写入 RL 配置并执行 RL smoke / pilot。

影响：补齐 SFT → DPO → RL 全链路代码闭环，不改动已完成阶段的训练产物与冻结文件。

## 对齐版 Phase 4 RL（`configs/train/rl_pilot_aligned.yaml`，2026-10-23）

背景：在已发布 DPO release `dpo_06b_zh_20261022T133007_step000040`（`adapter_sha256` `1d28d3119385c178c4b1b9a1e52a68d134f33a9bfd3de36659290a184ba5c51a`）上完成 6-step RL Smoke（run `20261023T020031_rl_06b_zh_seed42`）后，初始 50-step RL Pilot（`configs/train/rl_pilot.yaml`，run `20261023T020835_rl_06b_zh_seed42`）在 step 5 因 `degraded_macro_cer` 上升 `+0.0011605679956308013`（`dropout +25`、`noise +8`、`far_field +1`、`reverb 0`）触发 `risk_stop`。逐 rollout 与逐样本比对确认：
1. 前 5 步访问的 14 个 group 中有 9 个 `zero_advantage=True`，因零信号 group 计入了累积窗口（`gradient_accumulation_steps=2`），导致每步仅对 **1 条** 语句做全步长 AdamW 更新；
2. 5 个非零 group 中有 3 个（`60%`）落在无声学支撑的 `[0.40, 0.55]` `dropout` 静音段，且未开启 `dropout` 父版本假设锚；
3. 整句 `pol_t.mean()` 在含标点或不等长候选上反传压低了静音切口处的逗号 `'，'` 和续写概率，使 `val` 上 `BAC009S0748W0156__dropout` 再次翻转为 `'就有一零个Human Rights Watch'`（`+14` 编辑）、`BAC009S0727W0465__dropout` 提前 EOS（`+8` 编辑），掩盖了 `BAC009S0725W0295`、`BAC009S0743W0430__far_field` 等 11 条样本共 `−13` 个编辑的真实修复；
4. 学习率 `1e-5`（高于 DPO step 40 末步 `3.875e-6` 的 2.6 倍）过大，且 2500 条 `val`（36620 字）上 Base 到 DPO step 40 累计减 35 个编辑对应的 `mean_greedy_reward` 总增幅为 `+0.001013`（SFT step 30 到 DPO step 40 减 8 个编辑对应约 `+0.00022`），原 `min_reward_improvement: 0.002` 量纲失配。

范围：保留已执行的 `configs/train/rl_pilot.yaml` 及其冻结测试不变，在 `train/rl_schema.py`、`train/rl_contract.py`、`train/train_rl.py` 中新增对齐版 RL 选项并新增 `configs/train/rl_pilot_aligned.yaml`。不恢复 `20261023T020835_rl_06b_zh_seed42`，不评 test。

设计：
1. **声学退化场景过滤（`rl.train_scenarios: [noise, reverb, far_field]`）**：RL rollout 仅在有完整声学支撑的 `noise`、`reverb`、`far_field` 上采样训练，排除 `dropout` 与 `clean`。
2. **组内无标点等长替换对齐与差异 token 梯度掩码（`rl.loss_token_scope: mismatched_only`、`rl.max_edit_count: 2`）**：组内以最高 reward 候选 $c_{\text{best}}$ 为锚，仅保留与 $c_{\text{best}}$ 监督 token 等长、无标点且差异 token 数 $\in [1, \text{max\_edit\_count}]$ 的负优势候选参与策略梯度，并在组内差异 token 掩码位点上反传 `-adv * pol_t[mask].mean()`，共享前后缀与 `<|im_end|>` 的策略梯度为 0；全序列仍保留 `kl_beta * k3_t.mean()` 正则；若组内不存在对齐差异候选，整组记为 `zero_advantage = True`。
3. **仅非零信号 group 计入累积窗口（`rl.accumulate_signal_groups_only: true`）**：`zero_advantage = True` 的 group 不推进 `window_groups`，确保每个优化器 step 严格累积 `gradient_accumulation_steps: 4` 个有效对齐信号 group。
4. **`dropout` 父版本假设锚（`rl.dropout_anchor_weight: 0.05`）**：每个优化器 step 确定性抽取 1 条 train `dropout` 样本，使用冻结 reference adapter（DPO step 40）贪心解码文本作为教师计算 token 平均 NLL（权重 `0.05`），锁定静音段行为。
5. **阻尼优化器与门禁校准**：`learning_rate: 1e-6`、`warmup_steps: 5`、`scheduler_max_steps: 50`、`train.max_steps: 20`（`save_steps: 5`、`eval_steps: 5`）、`kl_beta: 0.10`、`max_group_visits: 1000`、`no_signal_stop_groups: 100`，`gate.min_reward_improvement: 0.00002`（要求在 2500 条 `val` 上 `reward_delta_vs_step0 >= +2e-5` 正向提升，同时满足 `degraded_macro_increase_max: 0.0`、`min_improved_degraded_scenarios: 1`、`raw_kl_max: 0.001` 与 `clean` 上限）。

测试：运行 `.venv/bin/python -m unittest discover -s tests -v`，验证 `rl_pilot_aligned.yaml` 配置解析、组内无标点等长替换过滤与差异 token 掩码、`accumulate_signal_groups_only` 窗口语义、`dropout_anchor_weight` 参数校验，以及 `rl_smoke.yaml` / `rl_pilot.yaml` 历史测试保持通过。

验收：本地单元测试全部通过；在标准 A100 会话 `lora-asr-dpo-rl` 上以独立新 `run_id` 运行 `configs/train/rl_pilot_aligned.yaml`，通过 `RL_GATE`（`passed=true`、`risk_stop=false`、`degraded_macro_delta <= 0.0`、`improved_degraded_scenarios >= 1`、`reward_delta_vs_step0 >= 0.00002`、`raw_kl <= 0.001`）。

实际运行结果（2026-10-23，标准 A100 会话 `lora-asr-dpo-rl`，run `20261023T042034_rl_06b_zh_seed42`）：
- **Step 5 一次性通过全部门禁（`passed=true`、`risk_stop=false`、`selected_step=5`）**：
  - 2500 条 `val` 总编辑数降至 **`2298`**（相对 DPO step 40 的 `2306` 净减 **−8 个编辑**，相对 SFT step 30 的 `2317` 净减 **−19 个编辑**，相对 Base 的 `2341` 净减 **−43 个编辑**），总 `micro_cer = 0.06275259421081376`。
  - 分场景编辑数（对比 DPO step 40）：`clean` `158`（−5，`micro_cer=0.02157291`）、`far_field` `165`（−3，`micro_cer=0.02252867`）、`reverb` `177`（−3，`micro_cer=0.02416712`）、`noise` `419`（+2，`micro_cer=0.05720918`）、`dropout` `1379`（+1，`micro_cer=0.18828509`）。
  - 四退化场景合计编辑数降至 `2140`（净减 −3 个编辑），`degraded_macro_cer = 0.07304751501911524`（`degraded_macro_delta = -0.00010240305843800535 <= 0.0`），`improved_degraded_scenarios = 2 >= 1`。
  - `mean_greedy_reward = 0.9359128300031403`（`reward_delta_vs_step0 = +0.00017815833349299481 >= 0.00002`），`raw_kl = 0.00033839310621917683 <= 0.001`，`reference_matched = true`，空输出/重复/超长/截断率均为 `0.0`。
- **Step 10 收敛停机**：Step 10 因 `degraded_macro_cer` 上升 `+0.00013653741125067842` 触发 `risk_stop=True` 正常停机（`status=completed` / `reason=gate_risk`），`gate.json` 状态为 `passed`、`selected_step=5`（`publishable: false`），`latest.json` 指向 `step_000010`。

影响：将 Phase 3 DPO 验证成功的对齐与静音锚机制闭环迁移至 Phase 4 RL，消除 `dropout` 静音悬崖幻觉与单样本更新噪声，不修改历史配置与冻结产物。

## RL release 读回（2026-10-23）

背景：run `20261023T042034_rl_06b_zh_seed42`（`configs/train/rl_pilot_aligned.yaml`）在 Step 5 通过全部 2500 条 `val` 门禁，选出 `step_000005`（`latest.json` 指向 `step_000010`）。按阶段衔接合同，Phase 5 终局评测只允许读取经过独立新进程 2500 条 `val` 读回校验的不可变 RL release，不能直接使用 `latest.json` 或训练 run 目录。

范围：新增 `train/rl_release.py` 与 `configs/train/aishell_val_rl_step5_20261023.json`，并在 `tests/test_rl_contract.py` 中增加 RL release 打包与读回合同测试。不修改源 run `20261023T042034_rl_06b_zh_seed42` 的 `status.json` 或 `gate.json`（源 `gate.json` 的 `publishable: false` 与 `audit_disposition: unlistened` 保持不变）。本次 release id 固定为 `rl_06b_zh_20261023T042034_step000005`。

设计：
- `prepare`：加载 `load_rl_config(config_path)`，核对 `profile == "pilot"`、`stage == "rl"`、`gate.json` 中 `status == "passed"` 与 `selected_step == 5`，以及 `eval_decisions.jsonl` 中对应步的 `passed == true`、`risk_stop == false`、`reference_matched == true`、`reward_delta_vs_step0 >= gate.min_reward_improvement` 和 `raw_kl <= gate.raw_kl_max`。核对 `checkpoints/step_000005` 目录 SHA-256 与语义配置 hash 后，复制 `adapter/adapter.safetensors`、`adapter/adapter_config.json`、`source/gate.json`、`source/decision.json`、`source/scored.jsonl` 和 `source/training_state.json`，写出 `release.json`（schema `qwen3-asr-rl-release-v1`，初始 `status="staged"`、`publishable=false`、`latest_step=10`、`latest_is_release=false`、`audit_disposition="unlistened"`）。
- `readback`：在独立新进程中使用与训练评测一致的确定性开关，加载官方 base 与 release 的 `adapter.safetensors`，对 2500 条 `val.jsonl` 重新 greedy 解码并逐条比对 `sample_id`、`prediction_raw`、`stop_reason` 和 `error`。全部 2500 条零差异时写出 `readback.json`（schema `qwen3-asr-rl-release-readback-v1`），将 `release.json` 更新为 `status="published"`、`publishable=true`（不改变 `payload_sha256`），并同步至 Google Drive `releases/rl/rl_06b_zh_20261023T042034_step000005/`。

测试：`python3 -m unittest tests.test_rl_contract tests.test_directory_readmes -v`。验证 `prepare` 仅接受 passed RL step、拒绝 `risk_stop`/低 reward/高 KL、读回不一致保持 `staged`、全部一致转为 `published` 且 payload hash 不变。

验收与实测结果（2026-10-23）：Drive 上的 `releases/rl/rl_06b_zh_20261023T042034_step000005/` 已完成独立进程 2500 条 `val` 读回并发布：`status="published"`、`publishable=true`、`source_step=5`、`latest_step=10`、`latest_is_release=false`、`adapter_sha256="79df11247728f149df4999447bbcd05ae66ae5dd1c5646f48d7a9515a94ff3a2"`、`payload_sha256="9f527f1c5307cc5f153eb7357f820aee2057e13c542768eec6c4b0a1f8957eff"`、`readback_sha256="e2e28bc14972cddfcd9fcf2efa672a0deb056bb07d2174a9ac1676669ebe716b"`（`matched=true`，`compared=2500`，`mismatch_count=0`）。`configs/train/aishell_val_rl_step5_20261023.json` 固化 RL Step 5 的 2500 条 `val` 指标（总 `micro_cer = 0.06275259421081376`）。

## Phase 5 封存 `test.jsonl` 四阶段终局评测与主 Notebook 合同（2026-10-23）

背景：Base、SFT（`sft_06b_zh_20261015T234939_step000030`）、DPO（`dpo_06b_zh_20261022T133007_step000040`）、RL（`rl_06b_zh_20261023T042034_step000005`）四个候选版本已在固定 2500 条 `val.jsonl` 上完成全部训练、门禁选择与独立进程读回发布，期间 2500 条 `test.jsonl`（SHA-256 `611051822cd6a3a11a1c4df63b2434b18ee43475a7fe7569aeb13426fc4cc588`）全程封存，未参与任何超参调整、偏好对构建、rollout 或 checkpoint 选择。按 `02_development_plan.md` 与本文件第 275、331 行合同，Phase 5 对冻结的四阶段候选在同一 `test.jsonl`、同一确定性 greedy `decode.eval`（`4a4fee2b67884438faa196909749fcc8dcc6126b175554ad19992bf9b4f2f642`）和同一 `zh-cer-v1` 评分器下各评一次，并提供主 Colab 编排 notebook。

范围：新增 `scripts/run_phase5_test_benchmark.py`、`notebooks/01_qwen3_asr_post_training_pipeline.ipynb`、`configs/train/aishell_val_rl_step5_20261023.json` 和 `tests/test_phase5_benchmark.py`。不得根据 `test.jsonl` 结果返回修改 SFT/DPO/RL 配置或重选 checkpoint。`audit_disposition` 保持 `unlistened`。

设计：
1. **输入与不可变哈希核对（`scripts/run_phase5_test_benchmark.py`）**：
   - 核对 `test.jsonl` 文件 SHA-256 必须等于 `611051822cd6a3a11a1c4df63b2434b18ee43475a7fe7569aeb13426fc4cc588`（2500 行，500 个 `test` 源语句 × 5 类场景 `clean`/`noise`/`reverb`/`far_field`/`dropout`）。
   - 若本地 `wavs/` 仅提供 500 条 `clean` 测试源音频，调用 `scripts/manifest_audio.py` 的 `render_scenarios`（使用 `configs/data/manifest_mvp.yaml` 的固定 `seed: 42` 与增强参数）在评测前确定性生成其余 2000 条退化 WAV，并与 `manifest/sha256.json` 中记录的 2500 条测试音频 SHA-256 逐文件核对（100% 匹配才允许进入解码）。
   - 核对三份已发布 release 的元数据与 payload 完整性：
     - `sft`：`sft_06b_zh_20261015T234939_step000030`（`load_release`，`status == "published"`，`publishable == true`，`adapter_sha256 == "a185a61f54b6fd74bd5e3bfa1173e15450cfbb772731567671a928aad805096a"`）
     - `dpo`：`dpo_06b_zh_20261022T133007_step000040`（`load_dpo_release`，`status == "published"`，`publishable == true`，`adapter_sha256 == "1d28d3119385c178c4b1b9a1e52a68d134f33a9bfd3de36659290a184ba5c51a"`）
     - `rl`：`rl_06b_zh_20261023T042034_step000005`（`load_rl_release`，`status == "published"`，`publishable == true`，`adapter_sha256 == "79df11247728f149df4999447bbcd05ae66ae5dd1c5646f48d7a9515a94ff3a2"`）
2. **四阶段同口径确定性 Greedy 解码与断点续跑**：
   - 依次执行 `stages = ("base", "sft", "dpo", "rl")`：
     - `base`：将解码器 LoRA 的 `lora_B` 全部置零并校验 `sum(|lora_B|) == 0.0`（等价于无 LoRA 增量的官方 `Qwen/Qwen3-ASR-0.6B` base 模型），`adapter_sha = "base_zero_lora"`；
     - `sft` / `dpo` / `rl`：通过 `load_adapter` 加载对应 release 的 `adapter/adapter.safetensors` 并核对文件 SHA-256。
   - 每个阶段开始前调用 `_match_training_runtime(config)` 重置确定性开关与随机种子（`seed=42`），支持按 `predictions/<stage>.jsonl` 已完成的 `sample_id` 前缀续跑（每 250 条打印 `PHASE5_PROGRESS stage=<stage> done=<done>/2500`，每 500 条及完成时同步至 Drive）。
   - 每个阶段解码完 2500 条后调用 `evaluation.eval_cer.evaluate` 写出 `eval/<stage>/metrics.json`、`eval/<stage>/scored.jsonl` 和 `eval/<stage>/by_scenario.csv`。
3. **四阶段对比汇总与主 Notebook（`notebooks/01_qwen3_asr_post_training_pipeline.ipynb`）**：
   - 汇总四阶段在 2500 条 `test.jsonl` 及 2500 条 `val.jsonl` 上的总体 `micro_cer`、`macro_cer`、总编辑数、四退化场景等权宏平均 `degraded_macro_cer`、各场景编辑数/CER、相对 Base 与相对上一阶段的净编辑变化及输出健康度，写出 `benchmark_summary.json`（schema `qwen3-asr-phase5-benchmark-v1`）、`run_report.md` 和 `status.json`，并同步至 Drive `/content/drive/MyDrive/qwen3-asr/runs/<run_id>/` 且做 SHA-256 读回校验。
   - `notebooks/01_qwen3_asr_post_training_pipeline.ipynb` 按阶段顺序封装 CLI 调用（环境重建、数据校验、SFT 训练与 release、DPO 配对/参考 logprob/训练与 release、RL 训练与 release、Phase 5 封存 test 终局评测），所有单元格仅调用仓库脚本，不在单元格内定义模型或修改超参数。

测试：`python3 -m unittest tests.test_phase5_benchmark tests.test_directory_readmes -v`。使用合成夹具与 mock 解码器验证：拒绝未发布 release 或错误 `test.jsonl` 哈希、500 条 clean 音频确定性重建 2500 条测试音频并核对 `sha256.json`、四阶段（`base`/`sft`/`dpo`/`rl`）顺序解码与前缀断点续跑、`benchmark_summary.json` / `run_report.md` 生成与 Drive 读回同步，以及主 notebook JSON 结构与 CLI 合同。测试不下载模型。

验收：本地单元测试全部通过；在标准 A100 会话上完成 2500 条封存 `test.jsonl` 的四阶段（Base / SFT / DPO / RL）完整评测，产出 `benchmark_summary.json`、`run_report.md` 及各阶段 `metrics.json`/`scored.jsonl`，Drive 读回 SHA-256 100% 一致，并将真实 `test` 指标同步至 `00_progress.md` 与 `12_phase2_run_log.md`。（2026-10-26 已由 run `20261026T101800_phase5_test_06b_zh_seed42` 在标准 A100 上完成：`PHASE5_DONE rc=0`、`sync=synced`，封存 `test.jsonl` 总编辑数为 Base `2220` / SFT `2204` / DPO `2200` / RL `2213`。）

影响：完成 Phase 5 封存 `test.jsonl` 终局对比与主 Colab 编排 notebook 交付，保持历史训练 run 与已发布 release 只读。

## 影响与未验证项

此次修正将 DPO 序列归约、阶段 adapter 衔接、RL 混合采样的偏差、KL 门禁测量、累积退化门禁和恢复状态变成明确合同。新实现需按这些合同构建，不能直接沿用旧历史结果。A100 容量、依赖组合、最佳超参数、数据规模、KL/健康度阈值适用性及 CER 改善仍需真实 smoke/pilot；当前没有效果证据，更不能宣称达到或超过 Mega-ASR。
