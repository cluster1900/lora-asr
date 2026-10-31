# 数据方案

最后更新：2026-10-30

## 背景与范围

MVP 以中文语音为主，覆盖 `clean`、`noise`、`reverb`、`far_field`、`dropout`，验证模型是否改善退化语音而不损害 clean speech。本阶段建立可审计、可复现的 JSONL 数据合同，不把未审计的外部数据直接送入训练，也不在 manifest 内保存训练 seed。

## Manifest schema

每行至少包含以下字段：

```json
{"schema_version":"qwen3-asr-manifest-v2","sample_id":"utt0001__noise","source_utterance_id":"utt0001","audio":"wavs/utt0001__noise.wav","text":"纯中文转写","language":"zh","scenario":"noise","split":"train"}
```

带官方身份的行另有 `speaker_id`、`official_split`，可选 `source_package`。`official_split` 为 `dev` 时，`split` 仍是 `val`。Phase 0 的 smoke 行继续使用 `qwen3-asr-manifest-v1`。

字段约束：

- `sample_id` 在 train/val/test 全局唯一；`source_utterance_id` 标识同一原始语句的所有增强版本。
- `audio` 是相对 manifest 所在目录的 POSIX 路径，不能使用 notebook 临时路径、Drive 绝对路径或 `..` 穿越。
- `text` 是纯中文转写，不含 `language Chinese<asr_text>` 前缀、特殊 token 或 reward；前缀和 `<|im_end|>` 由 trainer 按固定 `qwen-asr` 版本添加。
- `language` 当前固定为 `zh`；`scenario` 只能是五类 MVP 值；`split` 必须与文件名一致。
- 可选 `augmentation` 对象记录噪声文件、SNR、混响、远场距离、dropout 位置和处理版本；不能覆盖原始 audio 或 text。

manifest 目录必须同时保存 `train.jsonl`、`val.jsonl`、`test.jsonl`、`manifest.json` 和 `sha256.json`。`manifest.json` 记录原始数据版本、生成器 Git SHA、音频处理版本、行数、每个 split/scenario 计数、固定 seed 和质量检查摘要；`sha256.json` 记录每个文件的 SHA-256、字节数和生成时间。

## 来源、物化与音频规范

原始音频、标注和增强器版本必须能追溯到来源 ID；下载或增强失败写入审计，不用空音频占位。训练前把音频物化为 16 kHz、单声道、PCM/WAV，建议时长范围为 0.5–30 秒；超过上限的样本在 manifest 审计中统计，是否截断或拒绝必须由配置决定并记录。训练 step 只读取 Colab 本地缓存中的物化文件，Drive 保存原始 manifest、hash 和结果。

## 切分、配额与泄漏检查

先按 `source_utterance_id` 分组，再切 train/val/test；同一 source 的 clean 和所有增强版本只能进入一个 split。test 不参与 SFT、DPO pair 构造、RL rollout、checkpoint 选择和任何阈值调参，阶段门禁只读 val。

每个正式 pilot 的 train 至少包含五类 scenario；val 必须包含 clean 与四类 degraded，且用来宣称场景改善的 cell 至少 50 条。脚本输出每个 split × scenario 的行数、唯一 source 数、空文本数、坏音频数、重复 ID 数、时长分位数和音频 hash 冲突。缺少配额时状态为 `insufficient_data`，不能用复制行冒充新样本。

## 固定生成与质量检查

数据脚本从固定输入版本、配置和 seed 生成；同一命令第二次执行必须得到相同的 ID 顺序、计数和摘要。检查至少包括：

1. JSON 可解析，必填字段存在，类型和枚举正确。
2. 音频路径存在、可解码、采样率/声道/编码符合规范；文本非空且 UTF-8 有效。
3. `sample_id` 不重复，source 不跨 split，train/val/test 的 audio/text/source hash 交集为空。
4. 五类 scenario 覆盖和每个 gate cell 的最小行数满足配置。
5. manifest、物化音频和生成器版本写入 hash；改动任一输入都会改变 manifest SHA-256。

坏样本不应让批处理崩溃：写入 `rejected.jsonl`（含 `sample_id`、原因、路径和处理版本），汇总中的 rejected/shortage 数必须与文件一致。只有通过校验的 manifest 才能交给推理或训练。

## 训练接口影响

SFT 读取 `text` 并在运行时构造监督前缀；DPO pair builder 可读取 train/val manifest 和 SFT greedy predictions，但必须保留 source ID 和 pair 审计；RL 读取 train manifest，不把参考转写作为候选。任何新字段都必须更新 schema 版本、`configs/README.md`、训练方案和测试夹具。

## 验收

固定输入、配置和 seed 能重建三个 JSONL、`manifest.json` 和 `sha256.json`；所有音频路径存在且可读；split 无明显同源泄漏；五类 scenario 覆盖；坏样本有明确审计；train/val/test 行数、source 数和摘要可被第二次运行复核。未通过任一项时，Phase 1 保持未完成，后续训练不得启动。

## Phase 1 构建接口

正式配置是 `configs/data/manifest_mvp.yaml`，schema 为 `qwen3-asr-data-v1`。`audited` 为 false，`quota.min_val_rows_per_scenario` 为 50。源 JSONL 每行只有 `source_utterance_id`、`audio`、`text`，可选 `language`。`audio` 相对源文件所在目录，拒绝绝对路径和 `..`。

```bash
python3 scripts/build_manifest.py \
  --config configs/data/manifest_mvp.yaml \
  --sources <source.jsonl> \
  --output-dir <manifest-dir> \
  --seed 42
python3 scripts/manifest_validate.py \
  --manifest <manifest-dir> \
  --config configs/data/manifest_mvp.yaml \
  --check-audio
```

`--seed` 必须与配置里的 seed 相同。构建器对每个通过检查的 source 写出五条 16 kHz 单声道 PCM16 WAV：`clean`、5 dB 高斯 `noise`、固定梳状 `reverb`、衰减加滑动平均的 `far_field`，以及按比例抹掉一段采样的 `dropout`。Phase 0 那条缺失文件只用于推理失败演练，不进入这五个可训练场景。处理版本是 `phase1-aug-v1`。噪声种子是 SHA-256(`seed`, source id, scenario) 的前 4 字节，不用 Python `hash()`。

切分先按 source 分组。SHA-256(`seed`, source id, `split`) 落到 `[0,1)` 后，小于 `splits.test` 进 test，小于 `splits.test + splits.val` 进 val，其余进 train。同一 source 的五条场景使用同一个 split。`sample_id` 是 `{source_utterance_id}__{scenario}`。四类退化行必须带 `augmentation`，clean 行不带。跨 split 的相同文本，或任意 source 之间相同的音频字节，会拒绝后出现的 source，并写入 `rejected.jsonl`。

输出目录固定包含 `train.jsonl`、`val.jsonl`、`test.jsonl`、`rejected.jsonl`、`manifest.json`、`sha256.json` 和 `wavs/`。`manifest.json` 不写墙钟时间。`sha256.json` 的 `generated_at` 不参与“同一输入是否得到同一 manifest SHA-256”的比较。`audited: false` 时，即使行数达到配额，状态仍是 `insufficient_data`，不能交给 SFT。

文本必须含至少一个 CJK 统一汉字，不能含拉丁字母或 `language Chinese`、`<asr_text>`、`<|`。建议时长 0.5–30 秒，超出则拒绝。退出码见 `05_testing_plan.md`。

2026-10-15 正式配置命名为 `aishell1-openslr33-pilot-20261015`，`audited` 改为 true。依据是 pilot 源没有未关闭的音文、切边、静音、削波、切分或配额问题，用户指示把许可证以外的问题关闭。OpenSLR 许可证栏 Apache-2.0 和正文 “free for academic use” 仍并列，这次不把它写成已接受。行数不够时状态仍是 `insufficient_data`。`audited: false` 时，即使行数够也要留下 `source_not_audited`。

## 官方切分

没有 `speaker_id` 和 `official_split` 的源文件继续使用哈希切分。行 schema 是 `qwen3-asr-manifest-v2`，集合摘要是 `qwen3-asr-manifest-set-v2`。

整份源文件只能有一种切分方式。一部分行带官方身份、另一部分不带时，构建器在写出 manifest 之前以退出码 2 停止，错误是 `mixed_split_mode`。

官方模式每一行还要有：

- `speaker_id`：字符规则与 `source_utterance_id` 相同。
- `official_split`：只能是 `train`、`dev` 或 `test`。写成 `val` 的行拒绝，原因 `bad_official_split`。
- 可选 `source_package`：源包名称，字符规则与 source id 相同。

`dev` 写入 `val.jsonl`。行内 `split` 是 `val`，`official_split` 仍保留 `dev`。构建器不把样本改到另一个官方 split，也不改写 utterance id。同一 `speaker_id` 出现在两个 split 时，按 `source_utterance_id` 排序，保留先出现的那个 split，其余行进入 `rejected.jsonl`，原因 `speaker_cross_split`。跨 split 的相同文本、相同 clean 音频字节，仍拒绝后出现的 source。

`manifest.json` 的 `split_audit` 记录模式、各 split 说话人、train/val、train/test、val/test 的说话人交集、源包名称和拒收原因计数。校验器重算这些字段。交集非空时问题名是 `speaker_leakage:`，状态为 `failed`。

5 小时、至少 50 名说话人、dev 与 test 各 500 条的抽样不在构建器里。`scripts/sample_aishell_pilot.py` 先写成带官方身份的源 JSONL，再调用现有构建命令。正式配置仍是 `source_version: unset`、`audited: false`。抽样不读取模型错误率，也不改 `text_problem`。

抽样合同：

- 背景：50 条抽听已经有结论，pilot 仍需要一份可复现的官方切分源。
- 范围：只选择 AISHELL-1 的 pilot 源。抽样器本身不改正式配置。2026-10-15 的审阅结论另写在配置里。不启动 SFT，不改官方转写，不改 `text_has_latin`。
- 输入：`aishell_transcript_v0.8.txt`，以及由整包扫出的时长表。时长表每行有 `utterance_id`、`speaker_id`、`official_split`、`duration_seconds`，可以带 `format_ok` 和 `clipped_samples`。
- 过滤：空文本、提示标记、半角拉丁字母、没有汉字，沿用 `text_problem`。全角拉丁字母另以 `fullwidth_latin` 拒绝。时长不在 0.5 到 30 秒、格式不合格、有满幅削波样本、缺少时长、同一句话的另一条，都进入抽样拒收表。相同 `text` 只保留 `SHA-256(seed:utterance_id:dedupe)` 较小的一条。
- 选择：seed 42。排序键是 `SHA-256(seed:utterance_id:pilot-v1)`。说话人按编号轮转。train 目标 18000 秒，单人上限是目标毫秒数整除 50；轮转直到时长达到目标，允许最后一条略超。用到的 train 说话人必须至少 50 名。dev 和 test 各取 500 条，同样按说话人轮转。配额不够时退出码 3，不写 `source.jsonl`。
- 输出：`source.jsonl` 只有 `source_utterance_id`、`audio`、`text`、`language`、`speaker_id`、`official_split`、`source_package`。`audio` 写成 `audio/{official_split}/{utterance_id}.wav`，`language` 是 `zh`，`source_package` 是 `aishell1-openslr33`。另外写 `rejected.jsonl`、`selected.jsonl` 和 `selection.json`。
- 测试：`tests/test_sample_aishell_pilot.py` 用小配额夹具检查轮转、全角拉丁、削波、重复文本、配额不足和两次运行一致。
- 验收：夹具测试退出码 0。真实运行的 `selection.json` 里 train 时长不少于 18000 秒、train 说话人不少于 50、dev 和 test 各 500 条。这仍不是审阅通过的训练集。
- 影响：构建器、正式配置和已有 manifest 不变。全角拉丁字母不会因为这次抽样就改由 `text_problem` 拒绝。

AISHELL-1 在标成已审阅前要记下 2026-10-14 的核对结果，命令在 `10_phase1_run_log.md` 的 P9：

- OpenSLR 许可证栏是 Apache-2.0，论文也写 Apache 2.0，同一页正文有 “free for academic use”。Hugging Face 卡片在未指定 revision 时写着 `apache-2.0`。许可核对还没有结束。
- 本机转写 `aishell_transcript_v0.8.txt` 去掉空白后，ASCII 拉丁字母和数字都是 0 行。7 行含全角拉丁字母，当前 `text_problem` 不拒绝。抽样器用单独的 `fullwidth_latin` 拒绝这 7 行，不放进 pilot 源。`text_has_latin` 保持不变。5 小时和 500 条仍是过滤前的目标，不够时按事先固定的轮转顺序补。
- Hugging Face commit `8515aae54a8ded02c55fe270e32bd146b4180158` 只有 `S0002.tar.gz` 到 `S0101.tar.gz`。这 100 个包的首成员都在 `train/` 下，官方 dev/test 路径返回 404。OpenSLR 上游仍是约 15G 的 `data_aishell.tgz`。按说话人拆开的镜像必须单独核对版本和 hash。
- OpenSLR 整包的外层是 `data_aishell/wav/Sxxxx.tar.gz`，没有 `wav/train`、`wav/dev`、`wav/test`。官方切分是每个说话人包内部的第一级目录。解出全部说话人包之前，必须先抽查 `S0002`、`S0724` 和目录中的最后一个说话人包，并写成 `layout_preflight.json`。`S0002` 的包内路径必须以 `train/` 开头，`S0724` 必须以 `dev/` 开头。2026-10-14 的抽查结果是 `S0002=train`、`S0724=dev`、目录末尾的 `S0723=train`。说话人编号的顺序不是切分顺序。Drive 未挂载、说话人包不是 400 个、或抽查路径没有切分时，不得开始整包解压。
- `resource_aishell.tgz` 的 SHA-256 是 `1a6749854456e9402bc7295767937367afed1327799a5e1df0ed64baa5f77409`。`speaker.info` 是 400 行性别，没有官方切分清单。

P9 的 50 条只覆盖本机已解压的 train `S0002`、`S0003`、`S0004`，而且没有人听。P11 的 50 条是另一份，train 25 加 dev 25，结论写在 `10_phase1_run_log.md`。官方文本和这 50 个音频文件都没有改。许可文字先不处理。先前三个说话人子集不作为这套官方切分的输入。

## 影响与 OpenSLR 33 许可证合规边界

2026-10-14 的候选比较见 [候选比较与首批建议](11_dataset_candidates.md)。AISHELL-1 仍是优先审阅候选。官方切分接口已经接到构建器和校验器上。P14 把正式配置命名为 `aishell1-openslr33-pilot-20261015`，`audited` 改为 true。

### OpenSLR 33（AISHELL-1）许可证合规收口（2026-10-30）

- **背景**：OpenSLR 33 发布页元数据栏与 AISHELL-1 原始论文标注许可证为 `Apache-2.0`，但同一发布页正文同时写有 `"free for academic use"`。此前文档将两处并列标注留作待收口合规事项。
- **范围**：针对本项目基于 `aishell1-openslr33-pilot-20261015` 生成的五场景 pilot 数据集及在其上训练产出的 SFT/DPO/RL pilot LoRA adapter 明确合规使用边界；不修改已冻结的 `train.jsonl`、`val.jsonl`、`test.jsonl` 与音频哈希。
- **设计（交集最严原则）**：取两处声明的交集（最严约束）执行数据与模型合规边界：
  1. 本项目当前所有基于 AISHELL-1（OpenSLR 33）的实验、评测与 pilot LoRA adapter 权重**严格限定于非商业学术研究与方法验证用途（academic / non-commercial research & benchmark evaluation only）**。
  2. 代码仓库不提交、不托管、不公开再分发 AISHELL-1 原始语音或合成退化 WAV 音频文件（Git 仓库仅保存可复现抽样脚本、JSONL 标注文本清单与 SHA-256 校验索引）。
  3. 若后续需要将训练权重用于商业生产部署，必须切换至无学术用途附加限制的纯商业授权语料重新构建 manifest 并从官方 `Qwen/Qwen3-ASR-0.6B` 重新训练，不得直接将基于 AISHELL-1 训练的 pilot adapter 用于商业闭源服务。
- **测试与核查**：通过 `git ls-files` 与 `.gitignore` 规则确保仓库不含任何 AISHELL-1 `.wav` / `.tgz` / `.tar` 音频载荷（`results/` 下的大型音频包与 `references/` 均不被追踪或分发），并在 `11_dataset_candidates.md`、`06_risks_and_decisions.md`、`configs/README.md` 与 `00_progress.md` 同步记录该合规边界。
- **验收**：全部数据、风险、配置和进度文档口径一致，无未决许可证状态冲突，且 Git 追踪树中零原始音频文件泄漏。
- **影响**：在不改动任何已有 manifest、哈希与 Phase 1–5 实验产物的条件下完成 OpenSLR 33 合规闭环；后续接入 AliMeeting（CC BY-SA 4.0）或 SLR28（Apache-2.0）时同样按各数据源原始许可独立声明边界。

### Phase 6 Track 2：AliMeeting Eval（OpenSLR 119）只读真实远场 OOD 数据合同（2026-10-30）

- **背景**：Phase 1–5 的 `far_field`、`reverb`、`noise` 均由 AISHELL-1 近讲干净语音经合成滤波器得到；需要引入真实物理会议室采集的同步“近场耳麦 vs 远场麦克风阵列”公开基准，检验四阶段（`base`、`sft`、`dpo`、`rl`）向真实房间混响与环境底噪的零样本域外（OOD）泛化能力。
- **范围**：仅使用 **AliMeeting Eval**（OpenSLR 119，`Eval_Ali.tar.gz`，字节数 `3673718355`，8 场真实多人会议共约 4 小时，许可证 `CC BY-SA 4.0`）构建**只读域外评测集**（`split: "ood_eval"`、`usage_policy: "read_only_ood_evaluation"`、`trainable: false`）。严禁将任何 AliMeeting 样本混入 `train.jsonl`、DPO pair 或 RL rollout，也不回写或改动任何 Phase 1 既有 manifest。
- **设计**：
  1. 配置文件为 `configs/data/alimeeting_ood_eval.yaml`（`schema_version: qwen3-asr-ood-data-v1`），固定 `seed: 42`、`far_field_channel: 0`（从 `Eval_Ali_far/audio_dir/{room}_{meeting}_MS*.wav` 8 通道阵列录音确定性抽取第 0 通道）、时长窗口 `[1.0, 20.0]` 秒、非静音阈值（`min_peak_abs: 0.001`、`min_rms: 0.0001`）与配对抽样上限 `max_paired_sources: 500`（最低门禁 `min_paired_sources: 50`）。
  2. 物化脚本 `scripts/prepare_alimeeting_ood_eval.py` 使用纯标准库解析 `Eval_Ali_near/textgrid_dir/*.TextGrid` 与 `Eval_Ali_far/textgrid_dir/*.TextGrid`：
     - **严格单说话人零重叠过滤**：在同一场会议的全部说话人轨（含近场各说话人 TextGrid 并集与远场多轨 TextGrid）上，任一活跃语音区间与另一区间重叠超过 `overlap_eps_sec: 0.001` 秒，或标注含显式重叠标记 `(/OVERLAP)`，一律拒收并写入 `rejected.jsonl`（原因 `speaker_overlap` 或 `overlap_tag`）。
     - **纯中文文本清洗与对齐**：剥离官方副语言/静音标记（`<sil>`、`<%>`、`<->`、`<$>`、`<#>`、`<_>`、`<space>`、`(~)`、`(SIL)`、反引号与 `&`），拒绝含半角/全角拉丁字母的语码转换片段（`contains_latin_code_switch`）及 `zh-cer-v1` 归一化后不足 `2` 个字符的极短语气词（`empty_or_too_short_text`）；要求近场与远场同说话人同文区间的起止同步偏差不超过 `max_near_far_sync_drift_sec: 0.25` 秒且时长差不超过 `0.08` 秒。
     - **会议分层确定性抽样与配对切片**：按 `SHA-256(seed:source_utterance_id:alimeeting-ood-v1)` 在每场会议内排序后按 `meeting_id` 轮转抽取最多 `500` 个配对源语句，为每个 `source_utterance_id` 同步切出 `near_field`（耳麦单声道）与 `far_field`（阵列 `ch0` 单声道）两条 16 kHz PCM16 WAV（共最多 `1000` 条），写出 `ood_eval.jsonl`（`qwen3-asr-ood-manifest-v1`）、`rejected.jsonl`、`manifest.json`（`qwen3-asr-ood-manifest-set-v1`）与 `sha256.json`，并支持将切好的只读 OOD 包持久化缓存至 Google Drive `data/alimeeting_eval/v1_seed42/`。
- **测试**：`python3 -m unittest tests.test_alimeeting_ood_benchmark -v` 使用合成双说话人 Praat TextGrid 与近场单声道/远场 8 通道 WAV 夹具验证重叠语音剔除、拉丁字母与标注符号过滤、近远场同步配对、确定性会议轮转抽样、`trainable=false` 只读标记及二次运行 SHA-256 一致性。
- **验收**：夹具单测退出码 0；Colab 真实 `Eval_Ali.tar.gz` 物化后 `selected_sources >= 50`、`near_field` 与 `far_field` 行数严格 1:1 配对、全部 WAV 通过 `sha256.json` 验签且零训练泄漏。
- **影响**：新增独立只读 OOD 数据入口，不影响 Phase 1 `manifest_mvp.yaml`、`build_manifest.py`、`manifest_validate.py` 或任何训练数据集。

采用相对路径、文件级 split、纯文本 `text` 和显式 source ID 后，旧的绝对路径 manifest、混入特殊 token 的标注和未带版本 hash 的结果不能直接复用；必须重新物化或提供迁移报告。该合同同时支持本地单测和 Colab Drive 恢复，不改变外部数据原始文件。
