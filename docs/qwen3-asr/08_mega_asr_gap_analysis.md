# Mega-ASR 差距分析

最后更新：2026-07-22

## 结论

当前项目已有历史 LoRA、推理和评测闭环，但尚未在新的固定公开数据与 BF16 公平口径下
证明净收益。最快且最接近公开方法证据的首轮方案，不是先做一次 direct SFT，再决定是否
补做 A2S，而是直接执行一次压缩 A2S：30k base-error curriculum 的 acoustic warm start，
再依次完成 200k decoder 和 200k joint 训练。

这条路径的合理目标是接近 Mega-ASR-Base 的 A2S-SFT 收益等级。由于本项目只使用 200k
训练集，且不包含 2.4M 数据与 RL，不能预先承诺复现 Mega-ASR 完整系统效果。

## 范围

本文件只回答三件事：公开方法提供了什么证据、当前实现还缺什么、什么结果才允许描述为
“接近 Mega-ASR”。具体文件、参数和执行顺序以 `02_development_plan.md` 为唯一合同。

本轮不把 Mega-ASR 作为代码底座，不复用其私有 Qwen wrapper、训练入口、LoRA target
规则或 adapter merge 逻辑。主实现只面向 pinned Qwen3-ASR 官方 `qwen-asr`/Transformers
API。

## 公开方法证据

Mega-ASR 论文 Table 5 在 Voices/Noizeus 上的消融如下：

| 方案 | Voices | Noizeus | 相对 Qwen3-ASR |
| --- | ---: | ---: | --- |
| Qwen3-ASR | 8.94 | 9.45 | baseline |
| Direct SFT without A2S | 8.31 | 8.79 | 约 7% 改善 |
| Mega-ASR-Base A2S-SFT | 7.59 | 8.12 | 约 14%-15% 改善 |
| Mega-ASR full | 7.35 | 7.64 | 约 18%-19% 改善 |

这组结果说明 direct SFT 的信息价值低于 A2S-SFT。先跑 200k direct，再补 30k x2、
200k、200k A2S，会把首轮样本 exposure 从约 460k 增加到约 660k，却没有解决方法上的
已知差距，因此 direct SFT 已从默认路径删除。

论文 Table 6 还显示 rule-based reward 与 LLM judge 的 WER 收益接近，而单 step 时间更短。
但本轮连 RL 都不进入训练合同，更不需要额外 judge 或 Teacher。公开数据已有 gold
transcript，base prediction 与 gold 的 WER/CER 已足以构造 A2S 难度分层。

## 当前项目与目标方法

| 维度 | Mega-ASR 公开方案 | 本项目首轮方案 | 当前状态 |
| --- | --- | --- | --- |
| 基础模型 | Qwen3-ASR-1.7B | `Qwen/Qwen3-ASR-1.7B` | 已固定 |
| 代码边界 | Mega-ASR 工程 | Qwen 官方 Trainer 结构 + 本项目 PEFT/A2S 编排 | 待实现 |
| 训练规模 | 论文约 2.4M | 160k robust + 40k clean | manifest 待生成 |
| 数据场景 | 多类真实与合成退化 | 当前 Hub revision 的 7 atomic + 47 compound | metadata probe 待完成 |
| 难度课程 | base WER 的 `<30/<50/<70` | 30k，英文 WER/中文 CER 统一为 `base_error_rate` | 待生成 |
| SFT | acoustic -> decoder -> joint | 同一 adapter 的三阶段压缩 A2S | runner 待实现 |
| LoRA target | 上游规则 | 从 pinned Qwen 运行时模块快照独立生成 343 target | target map 待固化 |
| RL | DG-WGPO | 不进入本轮 | 明确删除 |
| Router | 有 | 不进入本轮 | 明确删除 |
| 公平评测 | base、SFT、full | BF16 base、A2S LoRA、外部 Mega-ASR 同口径 | 待运行 |
| 发布物 | 完整系统 | 单一 adapter + processor + release manifest | 待生成 |

Hub 当前 revision 实际是 54 个 split，总计 7 个 atomic 和 47 个 compound；不能沿用
“7 atomic + 54 compound”的旧描述。正式下载前必须用 metadata-only probe 再验证 split、
字段、语言配额和磁盘预算，避免先下载大体量音频后才发现配置不成立。

## 必须补齐的最小闭环

首轮只补以下四项：

1. 固定 revision、seed 和 hash 的 200k train、10k validation、5k Bench manifest。
2. 从 robust train 派生 30k base-error curriculum，只为候选样本执行 BF16 base scoring。
3. 基于 Qwen 官方 Trainer 结构的单-adapter A2S runner，支持阶段切换、完整 checkpoint 和
   新进程 resume。
4. 同一 decode、normalization 与 evaluator 下的 BF16 base、A2S LoRA 和 Mega-ASR 外部
   baseline 对比。

第一轮明确不需要 direct SFT 对照训练、Teacher、Router、RL、自建增强、全量 645,925
样本 difficulty scoring、rank/target/LR sweep，或预先评测 50%/100% checkpoint。它们都
不能成为首轮交付依赖。

## 单 Adapter 的方法边界

本项目不是复制 Mega-ASR 的多阶段实现。343 个 target 从固定 Qwen revision 的运行时模块
树独立识别并一次性注入同一个 adapter：

- Phase I 只让 upper-4 audio attention、upper-4 audio MLP 与 projection 共 27 个 target
  接收梯度。
- Phase II 只让 decoder 196 个 target 接收梯度。
- Phase III 让全部 343 个 target 接收梯度。

阶段之间只切换 `requires_grad` 和 optimizer parameter groups，不 merge 或叠加多个
adapter。这样保持 A2S 的 acoustic、semantic、joint 优化顺序，同时最终仍由 Qwen 官方
base + 一个标准 PEFT adapter 加载。

## 测试与验收

实现正确性先由 golden batch、target-switch gradient、10+2 optimizer-step 新进程 resume
和 en/zh clean/degraded 四条推理验证。正式训练每个阶段只运行固定 512 条 validation
canary；Phase III 完成后才运行一次 10k validation、Bench 5k 和两套 clean test。

产品门要求 English robust WER、Chinese robust CER 和 Bench 32-cell macro error 均相对
BF16 base 改善至少 10%，至少 24/32 cell 改善，real 与 synthetic macro 均改善，并同时
通过 clean regression、空输出、重复输出、过长输出和幻觉式输出限制。

若三项 robust 指标均达到 10% 且 clean/failure 门通过，才发布 adapter 并进行外部差距
比较；任一指标低于 5% 时先排查数据、prompt、labels、target 与 evaluator，不自动扩大
数据或增加训练分支。

## “接近 Mega-ASR”的口径

只有运行 Mega-ASR 发布模型，并对固定 Bench manifest 使用相同 decode、normalization、
WER/CER 和 32-cell evaluator 后，才允许比较。本项目 Bench macro error 不高于 Mega-ASR
的 1.10 倍，并同时通过 clean 与失败率门槛，才允许写“接近 Mega-ASR 微调效果”。

通过自身产品门但未满足上述外部口径时，只能描述为“达到本项目 200k A2S 发布标准”，
不能声称达到或超过 Mega-ASR。

## 影响

新主线将历史 direct joint、v1-v6 trainer/config/notebook 和旧输出降为历史证据，不进入
新 runner 的 import、配置或模型选择路径。后续只有实际结果证明问题集中在高错误率语义
失败，或 robust 与 clean 存在无法由 retention 解决的冲突时，才分别评估是否独立立项
RL 或 Router；这不是本轮的自动下一步。

## 参考

- Mega-ASR：https://github.com/xzf-thu/Mega-ASR
- 论文：https://arxiv.org/abs/2605.19833
- 数据：https://huggingface.co/datasets/zhifeixie/Voices-in-the-Wild-2M
- Benchmark：https://huggingface.co/datasets/zhifeixie/Voices-in-the-Wild-Bench
