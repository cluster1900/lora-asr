# 中文语音候选比较与首批数据建议

核查日期：2026-10-15（2026-10-30 更新合规与 WER 收口状态）。状态：候选顺序保持不变。P14 将正式配置改为 `source_version: aishell1-openslr33-pilot-20261015`、`audited: true`；2026-10-30 已按交集最严原则完成 OpenSLR 33 非商业学术研究用途合规边界收口，并锁定 `jieba==0.42.1` 词典哈希补齐中文 WER 辅助指标。

## 背景与范围

Phase 0 已验证基础推理与 Drive 读回，Phase 1 缺少经过审阅的真实语音源。本次比较服务于单张 A100 上的中文鲁棒 ASR pilot：既要有可控的 clean/退化对照，也要能进一步检查真实环境泛化。

本次核查发布方页面、原始论文、数据许可标注、切分说明和下载包规格，形成候选顺序与接入建议。随后的 P9 下载了 1.2M 的 `resource_aishell.tgz`，并核对了 Hugging Face 上 100 个 train 说话人包；没有下载约 15G 的 `data_aishell.tgz`，没有人工抽听，未取得本项目 CER，也未启动训练。下列排序是工程判断；页面给出的标注质量不等于本项目审计通过。

## 比较结果

包大小沿用发布页面的标注，表示下载压缩包，不是解压后空间或本项目抽样后的体积。所有数据均需固定下载版本和文件 hash。

| 候选 | 已核实的特点与发布许可 | 接入成本与局限 | 建议用途 |
| --- | --- | --- | --- |
| AISHELL-1 | 400 人、安静环境普通话；16 kHz；发布页许可证栏为 Apache-2.0，正文另有 “free for academic use”；官方 train/dev/test 为 340/40/20 人且说话人不重叠。[发布页](https://www.openslr.org/33/)、[论文](https://arxiv.org/html/1709.05522) | 官方主包约 15G。2026-10-14 核对的 Hugging Face commit `8515aae` 只有 `S0002` 到 `S0101` 的 train 包，没有 dev/test 音频。朗读风格和人群覆盖有限，需要另加声学退化。论文总量描述为 170 余小时、分表时长为近似值，实际可用时长以导入统计为准 | **首选：clean 母集与五场景配对 pilot** |
| WenetSpeech S | S 子集 100 小时，来源覆盖网络视频和播客，标签经 OCR/ASR 与过滤产生；官网标注 CC BY 4.0，同时写明非商业用途和原作者保留音频版权。[子集与标签说明](https://github.com/wenet-e2e/WenetSpeech)、[官网](https://wenet-e2e.github.io/WenetSpeech/) | 需要申请下载密码；100 小时是逻辑子集，不能据此假定独立小包可下载。标签需复核，长录音要按原始录音分组；说话人隔离能力须另查 | **后续扩充：口语、内容和声学多样性** |
| AliMeeting | 118.75 小时会议，近场耳麦与 8 通道远场；Train/Eval/Test 为 104.75/4/10 小时；CC BY-SA 4.0。[发布页](https://www.openslr.org/119/) | Eval 包约 3.42G，远场 Train 包约 73.24G。需要固定通道、按标注切片，并处理重叠说话；会议/说话人交集须实际审计 | **优先真实场景评测候选**；首版限定非重叠短片段 |
| RealMAN | 约 83.7 小时语音录音，32 通道、多种真实声学场景；扬声器回放中文源语音，提供转写与直达声参考；CC BY 4.0。[官方仓库](https://github.com/Audio-WestlakeU/RealMAN) | val 包约 27.5 GB，train 包约 531.4 GB；资源成本高。验证集含混合的语音/噪声，不应描述为全部自然现场对话；需固定通道并追踪原始语句 | **真实声学专项备选**；适合后续检查房间、距离、移动条件 |
| AISHELL-4 | 120 小时、211 场会议，8 通道，含自然会话、重叠和噪声；CC BY-SA 4.0。[发布页](https://www.openslr.org/111/) | train 三包标注约 7G/25G/14G，test 5.2G；同样需要会议切片和重叠处理，新增会话级验证切分需设计 | 会议场景备选，第一轮无需与 AliMeeting 同时接入 |
| MAGICDATA Read / Primewords | 前者 755 小时、1080 人，后者 100 小时、296 人；两者发布页均标注 CC BY-NC-ND 4.0。[MAGICDATA](https://www.openslr.org/68/)、[Primewords](https://www.openslr.org/47/) | MAGICDATA train 52G，Primewords 主包 9G；NC/ND 条件使增强产物共享与未来用途需要额外核实，当前无必要优先增加这一接入问题 | 暂不作为首批来源 |

### 选择依据

优先审阅 AISHELL-1，是因为可把同一条标注可靠的语句变成 clean 与四种退化，直接比较声学条件变化下的 CER，并保留官方说话人切分。它的作用是建立可解释的后训练实验；朗读合成退化上的改善不能直接证明会议或真实远场效果。

AliMeeting 的 Eval 包更适合先做小规模真实会议验证；若主要关心房间声学、距离和移动源，再接 RealMAN。这个优先级来自本项目单通道 ASR 与存储预算的适配判断，不是数据集质量排名。AliMeeting 和 RealMAN 都不在第一批训练中混入。

WenetSpeech S 留给下一轮扩大训练分布。其官方 README 在 2026-01-09 提醒 `TEST_NET` 有标签错误，指向[问题记录 #63](https://github.com/wenet-e2e/WenetSpeech/issues/63)。接入时须冻结原始标签版本和经审阅的修订差异；不能按本模型预测直接改 gold，也不能把不同标签版本的指标并列为同一 benchmark。官网的用途文字与许可证名称需一并记录，不能只根据 GitHub 代码仓库的 Apache-2.0 判断音频许可。

## 第一批 AISHELL-1 设计建议

以下是待审阅、待实现的抽样方案，不是已生成的数据，也不替代现行 v1 合同：

| 部分 | 建议规模 | 抽样与用途 |
| --- | --- | --- |
| 原始样本审阅 | 首轮 50 条 | 从 train/dev 按说话人、时长和文本类型抽样；记录音文一致、切边、静音、读错、数字和中英混合问题。单独的 50 条不能证明整库。2026-10-15 这批 pilot 的审阅结论是这 50 条加上全量自动拒收。dev 25 条有人听，train 25 条是转写核对，`heard` 仍是 false。许可证正文不在这条结论里 |
| pilot train | 原始语音先取 5 小时，覆盖至少 50 名 train 说话人 | 在官方 train 内按说话人分层、限制单人占比；固定 seed 42；样本选择不使用模型错误率 |
| pilot val | dev 中 500 条独立原始语句，尽量覆盖全部 40 人 | 每条生成五类场景，得到 2500 行、每场景 500 行；仍只有 500 个独立 source |
| pilot test | test 中 500 条独立原始语句，尽量覆盖全部 20 人 | 在评测前冻结名单；不用于难度选择、增强调参、checkpoint 选择或 DPO/RL 构造，只在发布时评测 |

采用子集并应用本项目去重与文本过滤后，结果必须命名为本项目 AISHELL-1 子集，不能称为官方完整 AISHELL-1 benchmark。未达配额时按预先固定的抽样顺序补齐，记录拒收原因，不按识别错误选择 val/test。

五类等时长物化后，5 小时原始 train 约成为 25 小时音频。按 16 kHz、单声道 PCM16 估算，纯采样数据约 2.88 GB（计算式为 `25 * 3600 * 16000 * 2` 字节，不含 WAV 头和文件系统开销）；实际还需原始下载包、解压目录、val/test 和临时空间。抽小子集不会自动降低上游整包下载需求。

## 当前实现与建议之间的差距

`scripts/build_manifest.py` 现在有两种切分。源行没有 `speaker_id` 和 `official_split` 时，仍按 source ID 哈希切分。整份源都带这两个字段时，保留 `train`、`dev`、`test`，把 `dev` 写入 `val.jsonl`，并在行内留下原来的 `official_split`、`speaker_id` 和可选 `source_package`。同一说话人跨 split、跨 split 的相同文本、相同音频字节都拒绝后出现的 source，不改 ID，也不把样本挪到另一个 split。`manifest.json` 的 `split_audit` 报告说话人交集。混用两种切分时退出码 2。

5 小时、50 名说话人和 dev/test 各 500 条由 `scripts/sample_aishell_pilot.py` 写成带官方身份的源 JSONL，不放进构建器，也不改官方切分。直接把官方整包交到哈希模式仍会打散 340/40/20。合同细节在 `03_data_plan.md`。

此前撤回的三说话人子集只有非正式历史价值。本次把 AISHELL-1 列为候选，不表示认可旧抽样方式，也不使旧产物成为正式训练输入。

现有增强是高斯噪声、梳状混响、衰减加滑动平均和局部 dropout。它们可测试流程，但覆盖真实环境有限。后续增强候选为 [MUSAN noise](https://www.openslr.org/17/)（发布页 CC BY 4.0，整包 11G）与 [RIR/Noise SLR28](https://www.openslr.org/28/)（Apache-2.0，约 1.3G）。两者是增强素材，不是中文转写母集。可优先审阅较小的 SLR28，按原始噪声录音和房间/RIR 家族隔离训练与评测，记录 hash、SNR、卷积与幅度处理，版本变化后重跑 base。

## 测试、验收与进入下一阶段的条件

候选研究的验收是来源链接、许可标注、用途、下载量与接入缺口清楚；本轮已满足。正式数据验收仍未满足，后续依次检查：

1. 获取固定版本的元数据与音频，记录归档 hash、许可和官方 split/speaker 清单；镜像要核对完整性，不能仅凭同名数据卡认可。P10 已核对 OpenSLR 整包和 340/40/20。许可正文仍未处理。
2. 官方切分的夹具合同已经通过，见 `10_phase1_run_log.md` 的 P8。正式语料接入时还要再查：官方 split 得到保留，speaker/source 跨 split 时拒绝，同源增强始终同 split，两次生成 hash 一致，混合文本和坏音频有拒收记录。
3. 留下抽听审阅记录、全量自动审计、拒收统计和处理结论；仍未解决的标签错误不可作为 gold 进入 DPO 或 reward。P11 留下了 50 条抽听结论。全量时长、格式和削波审计随 pilot 抽样的时长表一起做，拒收统计写在 `selection.json`。
4. 正式 manifest 满足 schema、音频、hash、泄漏与五场景配额；审阅完成后才更新配置中的来源与审计状态，独立 validator 带 `--check-audio` 退出码为 0。P13 在未审阅配置下退出码是 3。P14 把来源命名为 `aishell1-openslr33-pilot-20261015` 并把 `audited` 改为 true，再重新校验。许可证正文仍未处理。
5. 固定 val 上先跑 base，报告分场景 CER、WER 和空/重复/超长/失败率。2026-10-30 已通过 `configs/inference/base_eval_wer.yaml` 锁定 `jieba==0.42.1` 及内置 `dict.txt` SHA-256（`7197c3211ddd98962b036cdf40324d1ea2bfaa12bd028e68faa70111a88e12a8`），补齐 `val.jsonl` 与 `test.jsonl` 四阶段中文 WER。用 val 判断难度是否提供可测信号；如需修改增强，创建新版本、记录原因并重新建立全部对比基线，test 保持冻结。
6. 进入 SFT 时先做 5–20 step smoke 和恢复测试，再验证相对同一 base 的 clean/degraded gate。不能用训练 loss 下降或合成 fixture 通过代替识别指标。

真实会议评测需单独固定通道和非重叠片段规则，按会议隔离开发与保留集；只评非重叠片段时明确其覆盖范围，不能宣称完整多人会议性能。公开语料是否曾进入基础模型预训练尚未核实，不能称作模型绝对未见数据。

## 对现有工作的影响

`configs/data/manifest_mvp.yaml` 是 `source_version: aishell1-openslr33-pilot-20261015`、`audited: true`。Phase 1–5 已在固定 `val.jsonl` 与封存 `test.jsonl` 上完成闭环；OpenSLR 33 的 `Apache-2.0` 与 `"free for academic use"` 并列声明已按交集最严原则收口为“仅限非商业学术研究与评测、不公开再分发原始或派生音频”（详见 `03_data_plan.md`）。下一阶段优先接入 **AliMeeting Eval**（CC BY-SA 4.0，非重叠短片段，近场耳麦 vs 8 通道远场）进行真实世界 OOD 泛化评测。
