# 训练方案

最后更新：2026-10-12。状态：**设计合同已补齐；runner、环境锁、GPU smoke 和效果验收尚未完成**。

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

记录 GPU 型号/显存、CUDA/驱动、Python、torch、qwen-asr、Transformers、PEFT、音频和分词依赖的精确版本。当前 `requirements.txt` 只有范围约束，不能充当环境锁；Phase 0 须提交验证过的锁文件并从第二个空 runtime 重建。BF16 能力探针、单 batch forward/backward、LoRA 梯度、显存余量均通过才训练。FlashAttention 不是默认安装要求；切换 attention 后重新做 probe 和 baseline。

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

达到 max_steps/max_epochs/运行时间任一上限即收尾。没有任何 passed checkpoint 则 `gate_failed`；通过多个时按 degraded scenario 等权宏 CER 最低、再 clean CER 最低、再最早 step 选择，规则启动前冻结。`latest.json` 仅是恢复进度，不是最优模型。评测 checkpoint 与保存 step/adapter hash 必须完全一致。

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

## 影响与未验证项

此次修正将 DPO 序列归约、阶段 adapter 衔接、RL 混合采样的偏差、KL 门禁测量、累积退化门禁和恢复状态变成明确合同。新实现需按这些合同构建，不能直接沿用旧历史结果。A100 容量、依赖组合、最佳超参数、数据规模、KL/健康度阈值适用性及 CER 改善仍需真实 smoke/pilot；当前没有效果证据，更不能宣称达到或超过 Mega-ASR。
