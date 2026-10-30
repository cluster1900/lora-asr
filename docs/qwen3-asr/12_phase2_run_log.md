# Phase 2 运行记录

最后更新：2026-10-19。

SFT runner 和 `configs/train/sft_smoke.yaml` 已经写进仓库。本地合同测试不加载模型。6 step 保存和恢复已经完成，结果见下文「2026-10-15 smoke 结果」。

100 step pilot 已在 step 10 因 gate risk 停止，结果见下文「2026-10-15 pilot 结果」。test 不评。smoke 的小 val 不能当成 2500 条 val base 的晋级结果。

## 2026-10-15 smoke 挂起

第一次 `colab exec` 在内核连上之前报了 `RuntimeError: Connection was lost`，那次没有进入训练。同一台标准 A100 重启内核后，探针确认包还在，训练脚本已经开始安装环境锁。6 step 的保存和恢复还没有结果。

- 会话：`lora-asr-sft-smoke`
- 分配名：`gpu-a100-s-kkb-ass1c0-2u8tcya4h1aw4`
- 硬件：A100，形状 Standard，没有用 high-mem
- 包 SHA-256：`ca76b3444c4c6a062c38c2938b7989bfa3ef8dcdfff3513bbd56fbc3961f5b2c`
- 包内容：训练入口、评测和推理依赖、`configs/train/sft_smoke.yaml`、`requirements.lock`、`scripts/rebuild_phase0_env.py`，以及 60 条 smoke 音频
- sub2 启动单元：`sft-smoke-retry.service`，主进程 3772987，父进程 1
- 日志：`/root/colab-keepalive/sft_smoke_20261015.log`
- 配置：`configs/train/sft_smoke.yaml`，seed 42，`max_steps` 6，第 5 step 保存并暂停，第二个进程 `--resume` 到第 6 step
- 已看到的标记：`SESSION_CREATED`、`UPLOAD_DONE`、`PROBE_PACK True`、`PROBE_OK`、`UNPACK_OK`，环境锁安装退出码 0
- Drive：这次不发起挂载。脚本只在 `/content/drive/MyDrive/qwen3-asr` 已经存在时复制 checkpoint

第一次训练进程的 run_id 是 `20261015T075552_sft_06b_zh_seed42`。模型已经加载。`peft` 0.21.1 注入 LoRA 时因为预装 `torchao` 0.10.0 退出，`status.json` 为 `failed`，`global_step` 为 0，没有 checkpoint。包装进程随后写出 `SFT_WRAPPER_FAIL`。`colab exec` 自己的退出码仍是 0，因为 IPython 把 `SystemExit: 1` 当成单元格输出。torch 仍是 `2.11.0+cu130`，CUDA 仍可用。

这次在同一台 VM 上执行了 `pip uninstall -y torchao`。卸载后新进程里 `peft` 的 `is_torchao_available()` 返回 false。没有升级 torch，也没有把 `torchao` 写进锁。

## 2026-10-15 smoke 结果

第二次进程跑完了保存和恢复。会话随后已停止。Drive 没有挂载，`sync` 为 `sync_skipped`。本机副本在 `results/phase2_sft_smoke_20261015/runs/20261015T080308_sft_06b_zh_seed42/`。运行包 SHA-256 是 `d01d376d311c6d374128edee7e32381600e68056c6f7c5989e0afa414a77ce61`。

- run_id：`20261015T080308_sft_06b_zh_seed42`
- 硬件：`NVIDIA A100-SXM4-40GB`，42405855232 字节，形状 Standard
- 环境：Python 3.13.15，torch `2.11.0+cu130`，CUDA 13.0，transformers 4.57.6，peft 0.21.1。环境锁安装报告状态 `passed`
- 第一个进程在 `2026-10-15T08:03:58Z` 于 step 5 写出 checkpoint 并暂停。`interrupt_consumed` 为 true，已看样本 40
- 第二个进程在 `2026-10-15T08:04:03Z` 启动，`resume_ok` 的 `global_step` 是 5，随后只完成 step 6。已看样本 48
- `status.json`：`completed`，原因 `max_steps`，`global_step` 6
- `latest.json` 指向 `checkpoints/step_000006`，目录 SHA-256 `9a730e6a58ed65d2ef1583c598ba1989288ba254e8f4b01a599532b83459f02d`。step 5 的目录 SHA-256 是 `582f5478ad15b257d2b84d06257b1a9b30c90f4d735dc4039df8d2044018cae4`。两份都在本机按文件重新计算通过
- 可训练参数 224 个、4587520 个数，全部是 `thinker.model.layers.<n>.self_attn.{q,k,v,o}_proj.lora_{A,B}.default.weight`，n 从 0 到 27。没有 audio tower、embedding 或 `lm_head`
- 六步 loss 都有限：2.6998863220214844、2.478891611099243、2.9767048358917236、1.1649221181869507、0.23732353746891022、0.3785109221935272。这是 50 行上的 6 step，不能当成效果
- step 5 的 10 条小 val 没有空输出、重复输出或推理错误。`wer` 为 null。micro CER 不拿来和 2500 条 base 比较
- `gate.json`：`incomparable`，`publishable` false。原因是这条 val 不是固定的 2500 条 val
- VM 上没有 `.git`，`environment.json` 的 `git_sha` 为空。配置允许 dirty

100 step pilot 的启动记录在下一节。test 没有评。Phase 2 还没有完成。

旧的 val 客户端进程 3725331 当时还在，它对应的会话已经不在。这次没有停掉它。`colab new` 没有被它挡住。

## 2026-10-15 pilot 挂起

目的是跑 100 step SFT pilot。配置是 `configs/train/sft_pilot.yaml`，seed 42，`max_steps` 100，`max_epochs` 1，`save_steps` 和 `eval_steps` 都是 10。验证集是固定 2500 条 val。父指标在 `configs/train/aishell_val_base_20261015.json`，文件里的 manifest、predictions、decode hash 和五个场景 micro CER 与 `results/phase1_aishell_pilot_20261015/val_base/metrics.json` 一致。test 不进包，也不评。

- 会话：`lora-asr-sft-pilot`
- 分配名：`gpu-a100-s-kkb-ass1c2-ws8pebbidltk`
- 硬件：A100，形状 Standard
- Drive：`/content/drive/MyDrive` 和 `/content/drive/MyDrive/qwen3-asr` 探针都存在。授权地址没有写入本记录
- sub2 单元：`sft-pilot-20261015.service`，主进程 3795076，`LAUNCH_START` 为 `2026-10-15T10:08:15Z`
- 日志：`/root/colab-keepalive/sft_pilot_20261015.log`
- 包：train 19685 行、val 2500 行、对应 22185 个 wav，以及训练代码、`requirements.lock`、`scripts/rebuild_phase0_env.py`。wav 合计 3250441450 字节。sub2 上的 tar 是 3.1G，SHA-256 `99d15056e560eb27537f75b065cd7307621d1c4e57eddd65cdadfcb5d8d6c66c`，与传输流一致
- 第一次 `colab upload` 在 `2026-10-15T10:11:32Z` 开始。`10:13:01` 内核 OOM 杀掉了 `colab` 进程 3796240，anon-rss 约 3.6GB，机器内存 3.8GB。单元 `sft-pilot-20261015.service` 结果是 `oom-kill`。VM 上没有 tar，最后执行仍是 Drive 探针。训练没有开始
- 重试单元 `sft-pilot-20261015b.service`，主进程 3802992，`2026-10-15T10:28:46Z` 用 SSH 流式上传同一个 tar。日志是 `/root/colab-keepalive/sft_pilot_20261015b.log`
- `UPLOAD_DONE` 后 VM 上的 tar SHA-256 仍是 `99d15056e560eb27537f75b065cd7307621d1c4e57eddd65cdadfcb5d8d6c66c`。`EXEC_START` 为 `2026-10-15T10:39:38Z`
- 环境锁状态 `passed`。torch 保持 `2.11.0+cu130`，CUDA 可用。预装 `torchao` 0.10.0 已卸载，新进程 `TORCHAO_AVAILABLE False`
- run_id：`20261015T104034_sft_06b_zh_seed42`。`LAUNCH_OK` 已出现。可训练参数 4587520，`SFT_PREFLIGHT_OK`
- 已看到 step 1 到 step 5，loss 有限。这还不是 2500 条 val 的 gate
- `2026-10-15T10:41:14Z` 开始把同一个 tar 复制到 `/content/drive/MyDrive/qwen3-asr/data/aishell1/20261015/sft_pilot_train_val_20261015.tar`。复制完成，Drive 文件 SHA-256 仍是 `99d15056e560eb27537f75b065cd7307621d1c4e57eddd65cdadfcb5d8d6c66c`
- 本地合同测试：`tests.test_train_sft`、`tests.test_phase0_lock`、`tests.test_directory_readmes` 共 22 个，退出码 0

结果见下一节。Phase 2 没有完成。publishable 保持 false。

## 2026-10-15 pilot 结果

run `20261015T104034_sft_06b_zh_seed42` 在 step 10 触发 gate risk 并停止。进程退出码 0。`status.json` 是 `gate_failed` / `gate_risk`，时间 `2026-10-15T11:17:12Z`，`global_step` 10。`SFT_GATE_STATUS failed`。没有选中 checkpoint。`publishable` false。test 没有评。

- 本机副本：`results/phase2_sft_pilot_20261015/runs/20261015T104034_sft_06b_zh_seed42/`
- 运行包 SHA-256：`c941c5001344311c74fa76e0780a5da23cbd967d6bf87584f28c49a72281af1d`
- `latest.json` 指向 `checkpoints/step_000010`，目录 SHA-256 `0f87a5687f7b1bc0ba552c3594b6450bc389dbdd4605f063bca84012317d8c4d`。本机重新计算通过
- Drive `runs/20261015T104034_sft_06b_zh_seed42/checkpoints/step_000010/sha256.json` 的文件 SHA-256 是 `20d471a315d822015663a76b3e79dbc60679c66e4d4e7420042eec62088e3932`，与本机副本相同。`sync` 为 `synced`
- 2500 条 val 全部评完。合计 micro CER `0.11862370289459312`。空输出率 `0.0004`，重复、过长、截断和推理错误都是 0，`valid_output_rate` `0.9996`。WER 为 null
- 场景 micro CER：clean `0.038367012561441836`，dropout `0.4029219006007646`，far_field `0.04027853631895139`，noise `0.07359366466411797`，reverb `0.03795740032768979`。相对 base 的增量分别是 `0.015838339705079193`、`0.21477334789732386`、`0.016521026761332602`、`0.013653741125068272`、`0.012697979246313489`
- 五个场景都变差，退化场景等权 macro 上升 `0.06441152375750957`。没有场景改善。clean 增量超过 `0.01`，四个退化场景也都超过 `0.01`
- 会话在产物下载和 Drive 读回之后停止

这不是鲁棒性结论，也没有和外部系统比较。Phase 2 没有完成。失败的 gate 不自动续训。

同一份 2500 条上，step 10 相对 base 多出来的编辑主要是删字，不是换字。clean 的替换数是 153 对 152，删除数从 6 增到 122。dropout 的删除数从 1070 增到 2716，假设文本长度从参考的 0.855 降到 0.630。500 条 dropout 里有 287 条比 base 更差，257 条假设短于参考的 60%。输出仍以 EOS 结束，没有触发长度截断。评测前只看过 80 条训练样本。

## 2026-10-15 EOS 定位

在 V100 的 GPU 3 上做了一次 40 条探测。GPU 0–2 当时被别的训练占用，没有动。探测结束后 GPU 3 的计算模式已设回 Prohibited。日志在 `/home/ai/lora-asr-eos-probe/probe.log`。这不是新的 gate，也没有改 `configs/train/sft_pilot.yaml`。没有使用 `ignore_eos`。`max_new_tokens` 仍是 513。学习率没有重训。

样本是从 step 10 的 val 里挑的：16 条明显变短的 dropout，加上这 16 条的 clean 对照，再加 8 条长度接近 base 的 dropout。权重是 step 10 的 `adapter.safetensors`。先在 LoRA B 仍为 0 时做 teacher-forced 下一词，再载入 adapter 做同一组前向。训练步本身不调用 `generate`。

Teacher-forced 结果：base 的 16 条 clean 和 24 条 dropout 都没有在参考 EOS 之前把 argmax 打到 EOS。step 10 的 clean 有 1/16 提前，dropout 有 15/24 提前。参考文本的监督标签都以 EOS 结束。非末位监督位置上 EOS 的平均概率，base 接近 0，step 10 的 clean 是 `0.00716849170595558`，dropout 是 `0.05818861112222457`。所以提前结束已经在 teacher-forced 的下一词里出现，不只是自由解码走偏。EOS id 是 151645。

同一份 step 10 权重上的自由解码，40 条合计 micro CER：`min_new_tokens` 不设是 `0.31715575620767494`，设 4 是 `0.3047404063205418`，设 8 是 `0.1343115124153499`，按音频秒数乘 2 取整是 `0.40970654627539504`。dropout 的平均假设字数从 11.54 升到设 8 时的 18.25，clean 在不设和设 4 时都是 21.625。秒数下限把 clean 的 micro CER 抬到 `0.1792717086834734`。这 40 条是按短输出挑选的，不能拿来和 2500 条 base 比较，也不能当成 gate 通过。

下一轮配置是 `configs/train/sft_pilot_lr1e5.yaml`，学习率 `1e-5`。`sft_pilot.yaml` 的 `1e-4` 不改。解码、数据和 gate 保持与已跑 pilot 相同，因此不设 `min_new_tokens`，也不使用 `ignore_eos`。

## 2026-10-16 1e-5 启动

run `20261015T234939_sft_06b_zh_seed42` 已在标准 A100 会话 `lora-asr-sft-lr1e5`（`gpu-a100-s-kkb-usw4b0-2hdiz51phy8n`）启动。保活单元是 sub2 上的 `sft-lr1e5-20261016.service`，日志 `/root/colab-keepalive/sft_lr1e5_20261016.log`。数据包从 Drive 上的 `sft_pilot_train_val_20261015.tar` 拷到 VM 本地后校验，SHA-256 仍是 `99d15056e560eb27537f75b065cd7307621d1c4e57eddd65cdadfcb5d8d6c66c`。没有使用 `colab upload` 传这个 3.1GB 包，也没有 `--resume`。

环境复核通过：torch `2.11.0+cu130`，CUDA True，`TORCHAO_AVAILABLE False`。`SFT_LORA trainable=4587520 total=787013632`。`SFT_PREFLIGHT_OK`。第 1 step 学习率是 `5e-06`，第 2 step 是 `1e-05`。前两步 loss 与 `1e-4` 那次的同一窗口相同，第 3 step 起变为 `2.349385`。随后的评测和中断见下一节。test 不评。

## 2026-10-16 1e-5 在 step 30 评测中断开

保活日志 `/root/colab-keepalive/sft_lr1e5_20261016.log`。训练循环从 `2026-10-15T23:50:08Z` 的 step 1 开始。step 10 在 `2026-10-15T23:50:42Z` 保存，`2026-10-16T00:33:14Z` 写出 `SFT_GATE step=10 passed=True risk_stop=False`。step 20 在 `2026-10-16T00:33:53Z` 保存，`2026-10-16T01:15:45Z` 写出 `SFT_GATE step=20 passed=True risk_stop=False`。两次 2500 条评测各约 42 分钟。step 30 在 `2026-10-16T01:16:24Z` 保存，loss `0.489452`。评测随即开始。`2026-10-16T03:47:20Z` 客户端报 `Connection was lost`，`EXEC_CLIENT_EXIT 1`，`SFT_WRAPPER_FAIL`。这是 `colab exec --timeout 14400` 到点。sub2 上随后没有活跃会话。

配置里的有效上限是 110 分钟，大约到 `2026-10-16T01:40Z`。检查只在训练 step 开始前执行，step 30 的评测已经开始，所以没有停下。checkpoint 在评测前同步，step 30 应当已经在 Drive 上；这次没有重新挂载 Drive 核对。各场景 CER 写在 VM 的 `eval_decisions.jsonl`，不在 checkpoint 里，会话消失后本机读不到。日志只有门禁布尔值。run 没有 `SFT_DONE`，不能当成收尾。

同一天把 runner 改成每条 val 样本解码前检查这个上限。超过后写 `paused` / `max_runtime` 和 `eval/<step>/runtime_stop.json`，不写半成品 gate。恢复时先补该 step 的评测。这次断开的进程用的是改动前的代码，没有被这次修改续上。

## 2026-10-16 1e-5 从 step 30 恢复

Drive 已重新挂上，并核对过 run `20261015T234939_sft_06b_zh_seed42`。`latest.json` 指向 `checkpoints/step_000030`，`checkpoint_sha256` 是 `f720429814f6eed80a2fc72fbb09db72aff1f732377a0efcbe2b51b1e3797ff2`，`updated_at` 是 `2026-10-16T01:16:24Z`。`adapter.safetensors` 为 9204704 字节。Drive 上没有 `eval_decisions.jsonl`。

新会话是标准 A100 `lora-asr-sft-lr1e5`（`gpu-a100-s-kkb-usw4b0-3j8v332rku05a`）。数据包从 Drive tar 拷到 VM 本地后校验，SHA-256 仍是 `99d15056e560eb27537f75b065cd7307621d1c4e57eddd65cdadfcb5d8d6c66c`。没有用 `colab upload` 传这个 3.1GB 包。overlay 是未改的 `configs/train/sft_pilot_lr1e5.yaml`，以及带运行时间开关的 `train/train_sft.py`、`train/sft_contract.py`。环境重建退出码 0，`torchao` 0.10.0 已卸掉。

训练由 VM 内 PID 12766 启动，`start_new_session` 脱离了 Colab 客户端。客户端这次只负责拷贝、重建和拉起，超时只包住这段准备，不包住训练。日志已有 `SFT_LORA trainable=4587520 total=787013632`、`SFT_RESUME global_step=30`、`SFT_RUNTIME_BUDGET disabled`，随后进入 step 30 的 val 生成。`nvidia-smi` 看到该进程占用 A100。环境变量是 `QWEN3_ASR_DISABLE_RUNTIME_BUDGET=1`。配置里的 120 分钟没有改，语义 hash 仍能接上这个 checkpoint。

sub2 保活单元是 `sft-lr1e5-resume-20261016.service`，日志 `/root/colab-keepalive/sft_lr1e5_resume_keepalive.log`。它每 180 秒做一次短连接，没有总截止时间，也不再启动第二个训练进程。checkpoint 仍每 10 step 在评测前同步到 Drive。step 30 的 CER 还没有出来。Phase 2 未完成。test 不评。DPO 和 RL 不开始。

## 2026-10-16 1e-5 在 step 50 停下

PID 12766 写出 `SFT_DONE global_step=50` 和 `SFT_EXIT 0`。`status.json` 是 `completed` / `gate_risk`，时间 `2026-10-16T10:56:35Z`。进程退出后变成僵尸，父进程是还活着的 Colab kernel。

三次 2500 条 val 的门禁：

| step | passed | risk_stop | overall micro CER | clean | dropout | degraded macro delta |
| --- | --- | --- | --- | --- | --- | --- |
| 30 | true | false | 0.06318951392681595 | 0.02184598580010923 | 0.18801201529219005 | -0.0007509557618787521 |
| 40 | true | false | 0.06357181867831786 | 0.02170944838885855 | 0.1907427635172037 | -0.00023894046968869764 |
| 50 | false | true | 0.06398143091206991 | 0.02239213544511196 | 0.1923812124522119 | 0.00010240305843801922 |

固定 base 的合计 micro CER 是 `0.06392681594756963`。step 30 四个 degraded 场景的 micro CER 都低于 base：noise `0.057755324959038776`，reverb `0.025122883670125613`，far_field `0.023211359912616056`，dropout `0.18801201529219005`。clean 也低于 base `0.022528672856362644`。该 step 的 empty、repeated、long、truncation 都是 0，`valid_output_rate` 是 1，WER 是 null。

step 40 仍通过。dropout micro CER 升到 `0.1907427635172037`，增加 `0.0025942108137629594`，低于场景上限 `0.01`。合计 `long_rate` 是 `0.0004`，dropout 的 `long_rate` 是 `0.002`。

step 50 的唯一 risk 是 degraded macro CER 增加 `0.00010240305843801922`。冻结上限 `degraded_macro_increase_max` 是 `0.0`。dropout micro CER 是 `0.1923812124522119`，增加 `0.004232659748771156`。合计 `long_rate` 是 `0.0008`，dropout 的 `long_rate` 是 `0.004`，都低于 `0.01`。clean 仍低于 base。

`gate.json` 状态是 `passed`，`selected_step` 是 30，因为 step 30 的 degraded macro CER `0.07352539595849263` 低于 step 40 的 `0.07403741125068268`。`publishable` 是 false。`latest.json` 指向 `checkpoints/step_000050`，sha256 `dd769ba712c2388bd2a6b3831c5686869373b79ed08f6b82b1f5ac5a4a285808`，`updated_at` `2026-10-16T10:17:54Z`。这是最后保存的进度，不是选出的模型。

`gate.json`、`status.json`、`run_report.md`、`eval_decisions.jsonl`，以及 step 30/40/50 的 `metrics.json`、`by_scenario.csv`、`scored.jsonl` 已拷到 Drive 的同一 run 目录，读回 SHA-256 与 VM 本地一致。Drive 上 `checkpoints/.incoming` 是空目录。Phase 2 不标完成。test 不评。DPO 和 RL 不开始。会话 `lora-asr-sft-lr1e5` 随后被清理，不再存在。

## 2026-10-16 step 30 release 读回

本地 `tests.test_sft_release`、`tests.test_train_sft`、`tests.test_directory_readmes` 共 23 项通过，没有下载模型。新的标准 A100 会话是 `lora-asr-sft-release`（`gpu-a100-s-kkb-ass1c0-3nbqye7skpkr7`，Shape Standard）。另一台 High-RAM A100 还在，没有动它。

这一步只冻结 step 30，release id 是 `sft_06b_zh_20261015T234939_step000030`。不读 `latest.json` 的 step 50，不恢复训练，不评 test，不开 DPO 或 RL。第一次 `drivemount` 退出码是 0，但单元格里是 `ValueError: mount failed`。随后 `drive.mount(..., force_remount=True)` 打出 `Mounted at /content/drive`，`/content/drive/MyDrive/qwen3-asr` 存在，tar、gate、decisions、latest、step 30 adapter 和 `eval/step_000030/scored.jsonl` 都在，release 目录还不存在。

读回由 VM 内 PID 4667 启动，`start_new_session` 脱离了 Colab 客户端。客户端只负责拷贝和拉起。sub2 保活单元是 `sft-release-20261016.service`，日志 `/root/colab-keepalive/sft_release_keepalive.log`。它每 180 秒短连接一次，没有总截止时间，也不会再启动第二个进程。

`2026-10-16T14:03:30Z` 的保活看到 `SFT_RELEASE_PROGRESS 1000/2500`，PID 仍存活。`14:06:33Z` 的短连接报 `Connection was lost`，探测进程卡住，保活循环没有往下走。会话当时仍在。卡住的客户端已被清掉，没有另开读回。下一次探测读到 VM 日志：`SFT_RELEASE_PROGRESS 2500/2500`，`SFT_RELEASE_PUBLISHED compared=2500`，`READBACK_EXIT 0`，`DRIVE_RELEASE_SYNC match=True`。PID 4667 已经退出。

随后直接读 Drive 上的 JSON。release 目录 `releases/sft/sft_06b_zh_20261015T234939_step000030/` 里 `release.json` 是 `published` / `publishable=true`，`source_step=30`，`latest_step=50`，`latest_is_release=false`。`readback.json` 是 `matched=true`，`compared=2500`，`mismatch_count=0`。run 的 `status.json` 仍是 `completed` / `gate_risk`。`gate.json` 仍是 `passed`，`publishable=false`，`selected_step=30`。这次核对打印 `CONFIRM_OK`。

WER 仍是 null。Phase 2 不标完成。test 不评。DPO optimizer 和 RL 不开始。标准 A100 会话在核对时还在，没有主动停掉。另一台 High-RAM A100 没有动。

step 30 通过了冻结门禁，读回也全部一致，所以不把同一配置再训一次。step 50 已经因为 degraded macro CER 上升而停下。下一阶段是 DPO。`train_dpo.py` 还不存在，先用已发布的 release 解码 train。

`2026-10-16T14:37:26Z` 的保活看到 PID 17873 存活，`SFT_HYP_RESUME 0/19685`。磁盘剩余 181.6 GB。val 的 `edit_count>0` 行数是 clean 109/500、noise 189/500、reverb 121/500、far_field 114/500、dropout 489/500。保活单元是 `sft-hyp-20261016.service`。输出写到 Drive 的 `dpo/sft_06b_zh_20261015T234939_step000030/train_predictions.jsonl`。

`14:49Z` 的短连接报 `Connection was lost` 后客户端卡住，保活循环停在这一次探测上。解码进程没有被这次卡住带停。`2026-10-17T00:41Z` 只清掉卡住的 `colab exec`，保活 bash 还在。下一次探测在 `2026-10-17T00:44:33Z` 读到 PID 17873 已退出，日志里有 `SFT_HYP_PROGRESS 19685/19685`、两条 `SFT_HYP_SYNC 19685/19685` 和 `SFT_HYP_DONE rows=19685`。保活随后记 `KEEPALIVE_DONE` 并退出。Jupyter 执行通道之后又断了一次，核对改走 Colab SSH。Drive 文件和 VM 本地文件都是 8741520 字节、19685 行，sample_id 顺序与 train manifest 相同，SHA-256 都是 `581c672e966301de8cf05d36c21233f91d5224ad3b9202bd582f950d29fee401`。这次核对打印 `CONFIRM_OK`。没有重新挂载 Drive，也没有再开第二个解码。pair 和 DPO 更新都还没开始。标准 A100 会话还在，没有主动停掉。High-RAM 没有动。

## 2026-10-17 step 30 和 base 的错误对齐

用 release 里的 `source/scored.jsonl` 和本地 `results/phase1_aishell_pilot_20261015/val_base/scored.jsonl` 按 `sample_id` 做了 Levenshtein 回溯。每个场景 500 行、7324 个参考字，平均一句 14.648 字。对齐在代价相同时优先记成替换。

dropout 的 step 30 假设有 458/500 条和 base 逐字相同。14 条编辑变少，14 条变多，净编辑少 1。替换 302→307，删除 1070→1066，插入 6→4。假设长度是参考的 0.855。481/500 条比同一句话的 clean 假设短，486/500 条仍以参考的最后两个字结尾。有删除的 481 条里，最长删除段的中心中位数是 0.478，408 条落在参考的 40%–55%。这些最长段合计 1055 字，平均 2.19 字，和 `configs/data/manifest_mvp.yaml` 里 dropout 挖掉的 15% 音频（`start_ratio: 0.4`，`width_ratio: 0.15`）一样长。模型把这段静音跳过，前后文还在。

clean 有 493/500 条和 base 逐字相同，编辑净减少 5。删除保持 6，替换从 153 降到 148。剩下的是同音或专名，例如「起拍」听成「奇葩」、「钟溥敏」听成「钟馥敏」。noise 是唯一有可见改动的场景：472/500 条文本相同，替换 405→392，净少 16 个编辑。reverb 和 far_field 的文本几乎没变。

## 2026-10-17 DPO pair

`scripts/build_dpo_pairs.py` 在标准 A100 会话上读取已发布 release、train 19685 条假设和 val 的 `source/scored.jsonl`。没有重新解码，也没有重挂 Drive。tokenizer 是 revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`，EOS id 151645。加载时出现 Mistral regex 警告；对全部 22185 条 chosen 和 rejected 文本，打开 `fix_mistral_regex` 后 token 序列没有变化。

结果是 `pairs_built`。train 7456 对，val 1022 对，tie 13707 条。没有空输出、推理错误或超 token。val 分场景 pair 数是 clean 109、noise 189、reverb 121、far_field 114、dropout 489。dropout 平均字长差 train `-2.2371593291404612`、val `-2.1717791411042944`。产物在 VM `results/phase3_dpo_pairs_20261017/`，并复制到 Drive `dpo/sft_06b_zh_20261015T234939_step000030/pairs/`。train pair SHA-256 `d958d602dcc4a252ac75ce36d99536e8a53f48fa72929d1b69ead2647dba3f1b`，val pair SHA-256 `a7591e86f5a33efb314e29304153fe55e2d24ffba9bdd8b2739f953568b116d2`，七个文件都读回一致。`pair_audit.json` 是 `unlistened`。reference logprob 是 null。`trainable` 是 false。DPO optimizer 没有启动。

## 2026-10-17 reference logprob 已启动

同一台标准 A100 会话 `lora-asr-sft-release` 上启动 `train/reference_logprob.py`。父版本是已发布的 step 30 adapter。输入是上面这份 null logprob pair，train 7456 对、val 1022 对。本地合同测试 `tests.test_reference_logprob` 和目录 README 测试共 8 项通过，没有下载模型。打分前会去掉 SFT 的 embedding hook 并关闭 gradient checkpointing。

进程 PID 193168，`start_new_session` 脱离 SSH。日志在 VM `/content/dpo_logprob.log`。输出目录是 VM `results/phase3_dpo_logprob_20261017/`。Drive 缓存路径是 `dpo/sft_06b_zh_20261015T234939_step000030/ref_logprobs/ref_logprobs.jsonl`。复核通过后的填充文件会写到该目录下的 `pairs_logprob/`，不覆盖原来的 `pairs/`。sub2 保活单元 `dpo-logprob-20261017.service` 用短 SSH 查看，不包住整个任务。

`2026-10-17T02:32:51Z` 的第一次查看：PID 193168 状态 `S`，显存 2302 MiB，利用率 0%，模型仍在加载。`2026-10-17T02:33:54Z` 进程状态 `R`，显存 2362 MiB，利用率 17%，日志是 `REF_LOGP_RESUME 0/8478` 和 `REF_LOGP_PROGRESS 1/8478`。第一条 chosen 和 rejected 的 token 数与 pair 文件一致。

## 2026-10-17 reference logprob 完成

保活在 `2026-10-17T03:07:21Z` 记到 `KEEPALIVE_DONE`。VM 日志只有 `REF_LOGP_DONE train=7456 val=1022 rechecked=40 max_abs=0.0`，没有 `REF_LOGP_FAIL`。PID 193168 已退出。`ref_report.json` 的 `status` 是 `reference_cached`，`max_rel_error` 是 0.0，`seed` 是 42，`release_id` 是 `sft_06b_zh_20261015T234939_step000030`，`blocked_by` 是 `pair_audit_unlistened`，`trainable` 是 false。

8478 行 cache 与填充后的 pair 逐行对齐，logprob 都有限。VM 与 Drive `ref_logprobs/pairs_logprob/` 四个文件一致，父目录里的 `ref_logprobs.jsonl` 也一致：

- `ref_logprobs.jsonl`：2058907 字节，SHA-256 `5a77f22586dcf1b73393124055eb02466a904d8bdbdd2d44d40ace78000ef43e`
- `train_pairs.jsonl`：5906700 字节，SHA-256 `43473f10fbb9c0349adf2eb76eaf647506978c96eabb5b09847994456519c1b1`
- `val_pairs.jsonl`：811283 字节，SHA-256 `b82fc2783726555231164025a0e2d7faa82b5391bdcf8b98c6397eec1aff5eca`
- `ref_report.json`：625 字节，SHA-256 `8128e2a7731fda2b211dbaf840f38c31b132f26a9fab6f7cd3e8a11d4d805e78`

原来的 null pair 没被覆盖。train 仍是 `d958d602dcc4a252ac75ce36d99536e8a53f48fa72929d1b69ead2647dba3f1b`，val 仍是 `a7591e86f5a33efb314e29304153fe55e2d24ffba9bdd8b2739f953568b116d2`，VM 和 Drive 都对得上。

序列 logprob 更高的一侧是 rejected：train 7326/7456，val 1002/1022。按 token 平均后方向几乎一样。分场景里，val dropout 是 489/489，train dropout 是 3808/3816。这是父版本更相信自己的 greedy 文本，不是 DPO 已经更新过。听音审计仍是 `unlistened`。DPO optimizer 和 RL 没有启动。

## 2026-10-17 审计文本核对

标准 A100 会话 `lora-asr-sft-release` 这时已经不在。新开的是 CPU 会话 `lora-asr-dpo-audit`，形状 Standard，只用来读已经授权的 Drive。没有开 GPU，也没有动 High-RAM。

`pair_audit.json` 仍是 `unlistened`。按 seed 42 的 SHA-256 排序，train 和 val 各五个场景各 20 条，一共 200 条。文本核对没有把处置改成听过。原来的 null pair 哈希没变：train `d958d602dcc4a252ac75ce36d99536e8a53f48fa72929d1b69ead2647dba3f1b`，val `a7591e86f5a33efb314e29304153fe55e2d24ffba9bdd8b2739f953568b116d2`。

200 条的 chosen 都等于 reference。没有空 rejected、重复输出或超长幻觉。4 条被标成多插入一个字，仍然是模型错误，不是金标和假设对调：

- train clean `BAC009S0026W0258__clean`：`软件` 听成 `软件儿`
- train noise `BAC009S0349W0434__noise`：`上` 听成 `上面`
- train reverb `BAC009S0190W0124__reverb`：句首多了 `水，`
- val far_field `BAC009S0740W0129__far_field`：`房` 后多了 `产`，并多了一个逗号

dropout 抽样 40 条里 39 条有删除，33 条最长删除段的中心在 0.40–0.55。146 条是等长替换。200 条里 186 条的 rejected 序列 logprob 更高。这些条没有听过。产物在 Drive `dpo/sft_06b_zh_20261015T234939_step000030/audit_text_review/`。`audit_rows.jsonl` SHA-256 `eefddea447e64823f1af6bb0d7105b88435219729b4819923e677b69c998568a`。状态是 `text_checked_unlistened`。这次核对结束时 DPO optimizer 还没有启动。

## 2026-10-17 听音豁免

用户说「不听，直接开始训练」。200 条抽查的 `disposition` 保持 `unlistened`，`heard` 保持 false，`pair_audit.json` 不改。null pair 哈希仍是 train `d958d602dcc4a252ac75ce36d99536e8a53f48fa72929d1b69ead2647dba3f1b`、val `a7591e86f5a33efb314e29304153fe55e2d24ffba9bdd8b2739f953568b116d2`。DPO pilot 配置是 `configs/train/dpo_pilot.yaml`。学习率 `5e-6`，LoRA dropout `0.0`，累积 4，beta `0.1`。父版本是 release `sft_06b_zh_20261015T234939_step000030`。`publishable` 保持 false。test 不评。RL 不开始。

## 2026-10-17 DPO pilot 启动

标准 A100 会话 `lora-asr-dpo-pilot`，runtime `gpu-a100-s-kkb-ass1c1-vs6ho7soe4x7`，Shape Standard。Drive 目录 `/content/drive/MyDrive/qwen3-asr` 已用 `release.json` 读到，不是只看挂载退出码。父 adapter SHA-256 `a185a61f54b6fd74bd5e3bfa1173e15450cfbb772731567671a928aad805096a`，已写入 `configs/train/dpo_pilot.yaml`。音频包从 Drive 拷到 VM 本地后哈希是 `99d15056e560eb27537f75b065cd7307621d1c4e57eddd65cdadfcb5d8d6c66c`。填充 pair、cache、report 和 null pair 的哈希与配置一致。环境重建退出码 0，torch `2.11.0+cu130`，CUDA 可用。

run `20261017T131802_dpo_06b_zh_seed42`。PID 9519。输出目录 `results/phase3_dpo_pilot_20261017/runs/20261017T131802_dpo_06b_zh_seed42`。VM 训练日志 `/content/dpo_pilot.log`，包装日志 `/content/dpo_pilot_wrapper.log`。sub2 保活单元 `dpo-pilot-keepalive.service`，日志 `/root/colab-keepalive/dpo_pilot_keepalive.log`。`QWEN3_ASR_DISABLE_RUNTIME_BUDGET=1`。

启动时日志：

- `DPO_LORA trainable=4587520 total=787013632`
- `DPO_STEP0 checked=20 max_abs=0`
- `DPO_PREFLIGHT_OK`
- `DPO_RUNTIME_BUDGET disabled`
- `DPO_STEP global_step=1/100 loss=0.693147 grad_norm=10.250000 lr=2.5e-06 pairs=4`
- `DPO_STEP global_step=2/100 loss=0.693147 grad_norm=7.562500 lr=5e-06 pairs=4`

step 1–2 的 loss 等于 `log(2)`，和刚载入父 adapter、policy 还贴着 reference 时的平局一致。学习率符合 warmup：第 1 step `2.5e-6`，第 2 step `5e-6`。step 0 的绝对差是 0，没有放宽 1.0 的加载线。随后 step 3 到 step 10 的 pair loss 从大约 `0.694` 降到 `0.658`。`publishable` 仍是 false。test 不评。RL 不开始。null `pairs/` 没有改。

## 2026-10-17 DPO step 10 风险停止

保活在 `2026-10-17T14:06:38Z` 记到 `DPO_DONE global_step=10` 和 `DPO_EXIT 0`。PID 9519 已退出。A100 会话 `lora-asr-dpo-pilot` 随后不在。100 step 预算没有跑完。退出码 0 是风险停止后的正常收尾。

Drive 上有 `checkpoints/step_000010`、`latest.json` 和 `eval_decisions.jsonl`。`latest.json` 更新于 `2026-10-17T13:19:24Z`，指向 step 10，目录哈希 `98f50bf2ddef7016cc7185c39427ec0ec22988f3a61f93bb5f2c4e60395e43e4`。这是最后一次保存，不是选出的模型。`gate.json`、`status.json` 和逐条预测没有出现在 Drive。

step 10 决策：

- `passed=false`，`risk_stop=true`
- 风险只有一条：degraded macro CER 上升 `0.0006826870562534199`
- degraded macro `0.07420808301474605`，父版本 `0.07352539595849263`
- 相对父版本：dropout `+0.0034134352812670854`，far_field `-0.0002730748225013638`，noise `-0.00040961223375204914`，reverb `0.0`
- 变好的 degraded 场景是 2 个
- clean CER `0.022119060622610594`，比父版本高 `0.0002730748225013638`，比 base 低 `0.00040961223375204914`
- preference accuracy `0.8199608610567515`，ties `0`
- raw policy accuracy `0.019569471624266144`。shorter_rejected 519 对里 raw accuracy `0.0038535645472061657`，equal_length 483 对 `0.037267080745341616`，longer_rejected 20 对 `0.0`
- `publishable=false`

preference accuracy 量的是 policy 相对冻结 reference 的边际有没有转向 chosen。raw policy accuracy 量的是 policy 自己的 chosen logprob 是否高于 rejected。step 10 前者过了 0.55，后者仍约 0.02。dropout 上升触发了 `degraded_macro_increase_max=0.0`。没有逐条文本，不能判断静音段是不是被填成了乱字。

## 2026-10-18 dropout probe 合同

下一步配置是 `configs/train/dpo_dropout_probe.yaml`，profile `dropout_probe`。父版本仍是 step 30 release。学习率 `5e-5`，warmup 0，累积 4，beta `0.1`，最多 10 step，保存和评测都在 step 10。训练 pair 在整份哈希和 reference cache 检查通过后，只留 dropout。preference 也只打 dropout。2500 条 val 的 greedy CER 仍全评。

`DPO_LOCATE` 在第一次评测打印，并写入 decision。签名按合同顺序是 `MIXED`、`GAP_INSERT`、`EARLY_DELETE`、`QUIET`、`OTHER`。gap 是 dropout 行里锚点落在参考 0.40–0.55 的插入，门槛 30。提前删除是 clean 相对父版本上升超过 0.005，或 dropout 假设/参考长度比小于 0.75。`eval/<step>/scored.jsonl` 和 `eval_decisions.jsonl` 一起同步。`selected_step` 保持 null。`publishable` 保持 false。

## 2026-10-18 dropout probe 启动

run `20261018T052928_dpo_06b_zh_seed42` 已在标准 A100 会话 `lora-asr-dpo-dropout-probe` 启动。runtime `gpu-a100-s-kkb-usc1b0-3fnghm4a82hzx`，Shape Standard。训练进程是 VM PID 3187，日志 `/content/dpo_probe.log`。sub2 保活单元 `dpo-probe-keepalive.service` 从 `2026-10-18T05:34:11Z` 开始，日志 `/root/colab-keepalive/dpo_probe_keepalive.log`。旧单元 `dpo-pilot-keepalive.service` 仍是 inactive。

准备阶段读到的哈希与合同一致。音频包 `99d15056e560eb27537f75b065cd7307621d1c4e57eddd65cdadfcb5d8d6c66c`。环境锁 `cb7f7d03116c4b18100df8a165085eac4a014a765018a8cfb0d31b63c26d56a0`。填充 train `43473f10fbb9c0349adf2eb76eaf647506978c96eabb5b09847994456519c1b1`，填充 val `b82fc2783726555231164025a0e2d7faa82b5391bdcf8b98c6397eec1aff5eca`。cache `5a77f22586dcf1b73393124055eb02466a904d8bdbdd2d44d40ace78000ef43e`。report `8128e2a7731fda2b211dbaf840f38c31b132f26a9fab6f7cd3e8a11d4d805e78`。null train `d958d602dcc4a252ac75ce36d99536e8a53f48fa72929d1b69ead2647dba3f1b`，null val `a7591e86f5a33efb314e29304153fe55e2d24ffba9bdd8b2739f953568b116d2`。父 adapter `a185a61f54b6fd74bd5e3bfa1173e15450cfbb772731567671a928aad805096a`。环境重建退出码 0。torch `2.11.0+cu130`，CUDA 可用。当时 `/content` 剩余 193.31 GB。

日志有 `DPO_TRAIN_SCENARIOS dropout train_pairs=3816 val_pairs=489`、`DPO_STEP0 checked=20 max_abs=0`、`DPO_PREFLIGHT_OK`、`DPO_RUNTIME_BUDGET disabled`。配置峰值学习率 `5e-5`，`warmup_steps` 0，`min_lr_ratio` 0.1。下表是当时日志里的 10 个 optimizer step，每步 `pairs=4`。学习率符合 `lr_multiplier(step, 0, 10, 0.1)`。

| step | loss | grad_norm | lr |
| --- | --- | --- | --- |
| 1 | 0.693147 | 12.125000 | 4.8898772e-05 |
| 2 | 0.693147 | 7.937500 | 4.5702882e-05 |
| 3 | 0.576454 | 8.062500 | 4.0725168e-05 |
| 4 | 0.476869 | 5.437500 | 3.4452882e-05 |
| 5 | 0.458921 | 5.281250 | 2.75e-05 |
| 6 | 0.490969 | 3.546875 | 2.0547118e-05 |
| 7 | 0.444251 | 2.859375 | 1.4274832e-05 |
| 8 | 0.392833 | 3.218750 | 9.2971176e-06 |
| 9 | 0.432408 | 3.031250 | 6.1012284e-06 |
| 10 | 0.407643 | 3.312500 | 5e-06 |

step 1 和 step 2 的 loss 是 `log(2)`。探测时 preference 进度是 `DPO_PREF_PROGRESS step=10 1/489`。评测结果在下一节。`selected_step` 保持 null。`publishable` 保持 false。`20261017T131802_dpo_06b_zh_seed42` 不恢复。`dpo_pilot.yaml` 不重跑。test 不评。RL 不开始。Phase 2 不标完成。

## 2026-10-18 dropout probe 结果

保活在 `2026-10-18T06:13:28Z` 记到 `DPO_DONE global_step=10` 和 `DPO_EXIT 0`。PID 3187 已退出。训练会话随后不在。读回用的是新的 CPU 会话 `lora-asr-dpo-probe-read`。第一次 `drivemount` 退出码是 0，日志里仍有 `ValueError: mount failed`。`drive.mount(..., force_remount=True)` 之后读到 `release.json`，`adapter_sha256` 是 `a185a61f54b6fd74bd5e3bfa1173e15450cfbb772731567671a928aad805096a`。读完后这个 CPU 会话已停止。当前没有活跃 Colab 会话。

Drive `runs/20261018T052928_dpo_06b_zh_seed42/` 有 `checkpoints/step_000010/`、`latest.json`、`eval_decisions.jsonl` 和 `eval/step_000010/scored.jsonl`。`latest.json` 指向 step 10，目录哈希 `977d3e6d203b2eaff1a5b5d0ffbb94f620cf120550c022d64cd67e883bc063bc`。这是最后一次保存，不是选出的模型。`gate.json`、`status.json` 和 `run_report.md` 没有出现在 Drive。`eval_decisions.jsonl` 是 1 行、1110 字节。`scored.jsonl` 是 2500 行、1902866 字节。

step 10 决策：

- `passed=false`，`risk_stop=true`
- 风险有两条：`dropout micro CER increased by 0.029492080830147455`，`degraded macro CER increased by 0.007577826324412881`
- `locate_signature=OTHER`
- gap 插入 `24`
- dropout 假设/参考长度比 `0.8480338612779902`
- dropout delta `0.029492080830147455`
- clean 相对父版本 `0.000546149645002731`，相对 base `-0.0001365374112506819`
- clean CER `0.02239213544511196`
- degraded macro `0.08110322228290551`，父版本 `0.07352539595849263`，macro delta `0.007577826324412881`
- 相对父版本：dropout `0.029492080830147455`，far_field `0.0`，noise `0.0005461496450027345`，reverb `0.00027307482250136725`
- 变好的 degraded 场景是 0 个
- preference accuracy `0.9856850715746421`，ties `0`。这 489 对只含 dropout
- raw policy accuracy `0.018404907975460124`。shorter_rejected 481 对里 raw accuracy `0.016632016632016633`，equal_length 8 对 `0.125`，longer_rejected 0 对
- `publishable=false`

`scored.jsonl` 按场景合计，每场景 500 行、7324 个参考字：

| 场景 | 编辑 | micro CER | 假设字数 | 长度比 | empty | repeated | long | truncated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 164 | 0.02239213544511196 | 7326 | 1.0002730748225013 | 0 | 0 | 0 | 0 |
| noise | 427 | 0.05830147460404151 | 7313 | 0.9984980884762424 | 0 | 0 | 0 | 0 |
| reverb | 186 | 0.02539595849262698 | 7324 | 1.0 | 0 | 0 | 0 | 0 |
| far_field | 170 | 0.023211359912616056 | 7325 | 1.0001365374112507 | 0 | 0 | 0 | 0 |
| dropout | 1593 | 0.2175040961223375 | 6211 | 0.8480338612779902 | 0 | 0 | 3 | 0 |

`OTHER` 的原因是四条预注册规则都没打中。gap 插入 24，少于 30，所以不是 `GAP_INSERT`。长度比 `0.8480338612779902` 不低于 0.75，clean delta `0.000546149645002731` 不超过 0.005，所以不是 `EARLY_DELETE`。dropout delta 的绝对值大于 0.002，所以不是 `QUIET`。dropout 这次是 1593 次编辑。决策里的 delta `0.029492080830147455` 等于 216/7324，也就是比 step 30 多 216 次编辑。假设总长度停在 6211，没有掉到 `1e-4` SFT 那种 0.63。3 条 dropout 被标成超长，没有空输出、重复或截断。

preference accuracy 高，只说明相对冻结 reference 的边际转向了 chosen。raw policy accuracy 仍约 0.018，policy 自己的 chosen logprob 仍然大多低于 rejected。这个 checkpoint 不作为父版本。可用模型仍是 SFT step 30。test 不评。RL 不开始。Phase 2 不标完成。

## 2026-10-18 dropout 编辑分解

背景：`OTHER` 只说明四条预注册签名都没打中。这一节把多出来的 216 次 dropout 编辑拆开，再按拟定的缺口剥离规则数现有 pair。不改学习率，不启动新的 DPO。

范围：对齐用的是 Drive 上两份 `scored.jsonl` 的本地副本。父版本是 release `sft_06b_zh_20261015T234939_step000030/source/scored.jsonl`，SHA-256 `e434e476c176753548a006d50bf739cef450aa823713784dfbb37aa56f9977f2`，1903568 字节。probe 是 `runs/20261018T052928_dpo_06b_zh_seed42/eval/step_000010/scored.jsonl`，SHA-256 `db93ad534b8b950f17bab82cd8e6cf444ac3b504d301bdb01a13cf8747387591`，1902866 字节。pair 是冻结的 null 文件，train SHA-256 `d958d602dcc4a252ac75ce36d99536e8a53f48fa72929d1b69ead2647dba3f1b`，5686840 字节；val SHA-256 `a7591e86f5a33efb314e29304153fe55e2d24ffba9bdd8b2739f953568b116d2`，781081 字节。读文件的 CPU 会话 `lora-asr-pair-count` 在复制完成后已停止。null `pairs/` 和 `pairs_logprob/` 没有改写。

设计：对齐用仓库里的 `align_edits`。每行非 match 数等于该行 `edit_count`。带区按参考下标除以参考长度，0.40 到 0.55 含两端算 gap。插入锚点是当时还没消费的参考下标；参考已经读完时锚点是 1.0，记在 gap 之后。删除用被删字符自己的下标。pair 文件里是原文，计数前用 `normalize_zh_cer_v1`。val 的 1022 对和父版本 `scored.jsonl` 逐条对过：缺行 0，场景不一致 0，归一化文本不一致 0。

dropout 从 1377 次编辑变成 1593 次。操作数从删除 1066、插入 4、替换 307，变成删除 1166、插入 53、替换 374。增量是删除 +100、插入 +49、替换 +67。假设字数从 6262 降到 6211。500 条里 400 条假设相同。变差的 42 条多了 249 次编辑，变好的 25 条少了 33 次，其余 433 条编辑数相同。

| 操作 | gap 前 | gap 内 | gap 后 | 合计增量 |
| --- | --- | --- | --- | --- |
| 插入 | 2 → 27（+25） | 0 → 24（+24） | 2 → 2（0） | +49 |
| 删除 | 229 → 217（-12） | 780 → 762（-18） | 57 → 187（+130） | +100 |
| 替换 | 70 → 76（+6） | 108 → 133（+25） | 129 → 165（+36） | +67 |

gap 后多出来的 130 次删除，有 +142 落在最长删除段终点后移的 32 条上，其余行合计 -12。这 32 条里，父版本是纯 gap 删除的有 9 条，另外 23 条本来就带 gap 外的错误。最长删除段的中心仍在 gap 内的行，从 391 降到 364。有删除的行从 481 降到 474。终点后移 32，前移 39，不变 410，父版本没有删除段的 19 条。`gap_insertion_count` 从 0 到 24，和 `DPO_LOCATE` 的 24 一致。长度比仍是 `0.8480338612779902`。签名保持 `OTHER`。

另外四个场景的编辑增量很小：clean 160 → 164（删除 +1、插入 +3），相同假设 479，gap 插入 0 → 1；far_field 170 → 170（删除 -2、插入 +1、替换 +1），相同假设 483，gap 插入 0；noise 423 → 427（插入 +2、替换 +2），相同假设 462，gap 插入 0 → 1；reverb 184 → 186（删除 -1、插入 +1、替换 +2），相同假设 483，gap 插入 0。

剥离规则只用于 dropout。对齐后，gap 内的 match、替换和删除从两边去掉，锚点在 gap 内的插入也去掉。剩下的两边字符串相同，就不再算 preference 错误。其他场景不剥。

| 划分 | 原 pair | 剥完仍不同 | 剥完相同 |
| --- | --- | --- | --- |
| train dropout | 3816 | 2573 | 1243 |
| val dropout | 489 | 310 | 179 |
| train 合计 | 7456 | 6213 | 1243 |
| val 合计 | 1022 | 843 | 179 |

train 的 1243 对里，1117 对原来只有 gap 内删除，126 对是 gap 内的替换或插入。val 的 179 对里，157 对原来只有 gap 内删除，22 对是 gap 内的其他差异。其余场景的 pair 数不变：train clean 726、noise 1428、reverb 753、far_field 733；val clean 109、noise 189、reverb 121、far_field 114。dropout 仍高于每场景最少 10 对，val 合计仍高于 100。

验收：这一节的数字来自本机 `.venv/bin/python` 对上述哈希文件的对齐，没有改仓库里的 pair 文件。`score_pair` 没有改。签名分类没有加新类别。

影响：只丢掉纯 gap 删除，仍会训到那 32 条里的 23 条。两边一起剥掉静音段，train 还留 2573 对 dropout，val 还留 310 对。现有 logprob 对应的是未剥离的整句，不能直接套到剥离后的文本上。下一次 DPO 要先另写 pair 并重算 logprob。这次不训练。可用模型仍是 SFT step 30。`publishable` 保持 false。Phase 2 不标完成。

## 2026-10-18 dropout pair 另写

`scripts/strip_dpo_gap_pairs.py` 在 CPU 会话 `lora-asr-gap-strip` 上读取冻结 null pair。Drive 已挂上，`releases/sft/sft_06b_zh_20261015T234939_step000030/release.json` 读到 1416 字节。tokenizer 是 revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`，模式 `encode-plus-eos`，EOS `<|im_end|>` id 151645。四个 tokenizer 文件的 SHA-256 是 `tokenizer_config.json` `4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c`、`vocab.json` `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910`、`merges.txt` `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5`、`chat_template.json` `75a8cfca24f00de72d796fbfed6858fc9614ef3dabd8696684cc3bc03a9c58ff`。

日志是 `GAP_STRIP_STATUS pairs_gap_stripped train_pairs=6213 val_pairs=843 gap_ties=1422`。train 分场景是 clean 726、noise 1428、reverb 753、far_field 733、dropout 2573。val 分场景是 clean 109、noise 189、reverb 121、far_field 114、dropout 310。`gap_ties.jsonl` 1422 行全部是 `tie_after_strip`，train 1243、val 179。`overlength_pairs.jsonl` 是空文件，SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`。`short_val_scenarios` 为空。

Drive 目录是 `dpo/sft_06b_zh_20261015T234939_step000030/pairs_gap_strip/`。本机副本是 `results/phase3_dpo_gap_strip_20261018/pairs_gap_strip/`。两边哈希一致：

- `train_pairs.jsonl`：5250224 字节，SHA-256 `4ad364c87d367395d67603fdfb7acf503bd2c0174cfb35518b7b30b0c096505d`
- `val_pairs.jsonl`：709125 字节，SHA-256 `caffc52d2674562143578a2d18a73bfed85233c74f6d573495ca77477d695a0d`
- `gap_ties.jsonl`：1368228 字节，SHA-256 `cdf1353b395dc05da28a7c2a725b5d009fac2aa48a796a101be74ce6e84d3c06`
- `pair_report.json`：1591 字节，SHA-256 `2b015b31aa0182a6626d323c99b6e2cad2497bc664834d98e3537bae32f453dc`
- `pair_audit.json`：9004 字节，SHA-256 `c34f89e5cf4d8a4ce35301c613b4404daba373ba63616d95ec410d4d3e414f61`
- `pair_metadata.json`：1155 字节，SHA-256 `0b00f8e6d2e043593fcbed6b7eb5e0a47ea164d2b8f43cb20080d6c1860931a7`

复制之后重新核对冻结源：train 仍是 `d958d602dcc4a252ac75ce36d99536e8a53f48fa72929d1b69ead2647dba3f1b`，val 仍是 `a7591e86f5a33efb314e29304153fe55e2d24ffba9bdd8b2739f953568b116d2`。`pair_audit.json` 状态是 `unlistened`。`trainable` 是 false，阻塞项是 `reference_logprob_pending` 和 `pair_audit_unlistened`。没有重算 logprob，没有启动 DPO。可用模型仍是 SFT step 30。`publishable` 保持 false。Phase 2 不标完成。

下一步用现有 `train/reference_logprob.py` 给这 7056 对重算 reference logprob。配置是 `configs/train/sft_pilot_lr1e5.yaml`，父版本是 step 30 release，seed 42。输出父目录是 Drive `dpo/sft_06b_zh_20261015T234939_step000030/gap_strip_ref_logprobs/`，填充文件落到它下面的 `pairs_logprob/`。不覆盖 `ref_logprobs/pairs_logprob/`。算完之前不训练。

## 2026-10-18 gap-strip reference logprob 已启动

标准 A100 会话 `lora-asr-gap-logprob`（`gpu-a100-s-kkb-ass1c2-hh4ip1r0z29t`，Shape Standard，`NVIDIA A100-SXM4-40GB`）。没有用 `--high-mem`。CPU 读会话 `lora-asr-gap-strip` 在打分进程起来之后已停止。

启动前只读核对了冻结文件，哈希没变：null train `d958d602dcc4a252ac75ce36d99536e8a53f48fa72929d1b69ead2647dba3f1b`，null val `a7591e86f5a33efb314e29304153fe55e2d24ffba9bdd8b2739f953568b116d2`，旧 cache `5a77f22586dcf1b73393124055eb02466a904d8bdbdd2d44d40ace78000ef43e`，填充 train `43473f10fbb9c0349adf2eb76eaf647506978c96eabb5b09847994456519c1b1`，填充 val `b82fc2783726555231164025a0e2d7faa82b5391bdcf8b98c6397eec1aff5eca`。音频包 SHA-256 仍是 `99d15056e560eb27537f75b065cd7307621d1c4e57eddd65cdadfcb5d8d6c66c`。adapter SHA-256 `a185a61f54b6fd74bd5e3bfa1173e15450cfbb772731567671a928aad805096a`。环境重建退出码 0，torch `2.11.0+cu130`，CUDA true，`torchao` 0.10.0 已卸掉。

进程 PID 4484，`start_new_session` 脱离 SSH。显存 2462 MiB，利用率 17%。日志有 `REF_LOGP_RESUME 0/7056` 和 `REF_LOGP_PROGRESS 1/7056`。sub2 保活单元是 `gap-logprob-keepalive.service`，只做短连接查看，不包住打分进程。

## 2026-10-18 gap-strip reference logprob 完成

保活在 `2026-10-18T11:33:39Z` 记到 `KEEPALIVE_DONE`。VM 日志是 `REF_LOGP_DONE train=6213 val=843 rechecked=40 max_abs=0.0`，没有 `REF_LOGP_FAIL`。PID 4484 已退出。`ref_report.json` 的 `status` 是 `reference_cached`，`max_rel_error` 是 0.0，`seed` 是 42，`release_id` 是 `sft_06b_zh_20261015T234939_step000030`，`blocked_by` 只剩 `pair_audit_unlistened`，`trainable` 是 false。

7056 行 cache 与填充 pair 的 `pair_id`、token 数和 logprob 逐行对齐，logprob 都有限。VM、Drive `gap_strip_ref_logprobs/pairs_logprob/` 和本机 `results/phase3_dpo_gap_strip_logprob_20261018/` 一致。父目录里的 `ref_logprobs.jsonl` 也一致：

- `ref_logprobs.jsonl`：1711035 字节，SHA-256 `dd9db5012841d61dafce571c833c508bac7e7f9e64286c0928dba9e139a2da00`
- `train_pairs.jsonl`：5431942 字节，SHA-256 `7c01e5fc54596784f7e9ef22d3926cf830e3d034b58107aa455ac105bf0c392b`
- `val_pairs.jsonl`：733839 字节，SHA-256 `008fe7b044496d16645b022c2ef7f0677aca9c6d3a5de58f5c0e98ff5b65208b`
- `ref_report.json`：624 字节，SHA-256 `35b9361a7cce77ab2dea805ac468e4a87a88bb066aa2b43d19339fee4c5eb114`

算完后又核对了冻结文件，哈希没变。null train `d958d602dcc4a252ac75ce36d99536e8a53f48fa72929d1b69ead2647dba3f1b`，null val `a7591e86f5a33efb314e29304153fe55e2d24ffba9bdd8b2739f953568b116d2`，旧 cache `5a77f22586dcf1b73393124055eb02466a904d8bdbdd2d44d40ace78000ef43e`，旧填充 train `43473f10fbb9c0349adf2eb76eaf647506978c96eabb5b09847994456519c1b1`，旧填充 val `b82fc2783726555231164025a0e2d7faa82b5391bdcf8b98c6397eec1aff5eca`。`pairs_gap_strip` 的 null 文件也没变。

序列 logprob 更高的一侧仍多数是 rejected：train 5621/6213，val 763/843。分场景是 train clean 710/726、dropout 2103/2573、far_field 713/733、noise 1364/1428、reverb 731/753；val clean 107/109、dropout 250/310、far_field 111/114、noise 178/189、reverb 117/121。val dropout 不再是剥离前的 489/489。这仍是父版本更相信自己的 greedy 文本，不是 DPO 已经更新过。听音审计仍是 `unlistened`。写到这里时 DPO optimizer 还没有启动。`publishable` 保持 false。Phase 2 不标完成。

## 2026-10-18 gap-strip dropout probe 已启动

用户说「开始吧」之后，用 `configs/train/dpo_gap_strip_probe.yaml` 开了新 run `20261018T132840_dpo_06b_zh_seed42`。标准 A100 会话 `lora-asr-dpo-gap-strip`（`gpu-a100-s-kkb-usc1c1-2we8a5memb68k`，Shape Standard），没有用 `--high-mem`。第一次 `drivemount` 的 CLI 退出码是 0，内核仍是 `ValueError: mount failed`。随后 `force_remount` 打印 `Mounted at /content/drive`，`release.json` 是 1416 字节。

启动日志有 `DPO_TRAIN_SCENARIOS dropout train_pairs=2573 val_pairs=310`、`DPO_PREFLIGHT_OK` 和 `DPO_RUNTIME_BUDGET disabled`。trainer PID 9533。10 个 optimizer step 已结束。step 10 的 pair loss 是 `0.624705`，grad norm `2.953125`，这一步的 pair 数是 4，也就是累积步长。打印的学习率 `5e-06` 是 cosine 走到 `min_lr_ratio` 0.1 的下限，配置学习率仍是 `5e-5`。2500 条 val 还没有 `DPO_LOCATE`。sub2 保活单元是 `dpo-gap-strip-keepalive.service`，日志 `/root/colab-keepalive/dpo_gap_keepalive.log`。

`selected_step` 保持 null。`publishable` false。听音处置仍是 `unlistened`。这一步不作为父版本。可用模型仍是 SFT step 30。不恢复 `20261018T052928` 或 `20261017T131802`。Phase 2 不标完成。test 不评。RL 不开始。

## 2026-10-18 gap-strip dropout probe 结束

保活在 `2026-10-18T14:12:33Z` 记到 `DPO_LOCATE`、`DPO_GATE`、`DPO_DONE global_step=10` 和 `DPO_EXIT 0`。trainer PID 9533 已退出。`status.json` 是 `gate_failed` / `gate_risk`。`gate.json` 状态 `failed`，`selected_step` null，`publishable` false，听音处置 `unlistened`。这四份文件和 `run_report.md`、`listen_waiver.json` 已写到 Drive `runs/20261018T132840_dpo_06b_zh_seed42/`。`eval_decisions.jsonl` 与 VM 相同。

决策：`passed=false`，`risk_stop=true`。唯一风险是 degraded macro CER 上升 `0.0015360458765701773`，到 `0.0750614418350628`，父版本 `0.07352539595849263`。`locate_signature=OTHER`。gap 插入 10。dropout 长度比 `0.8623702894593118`。dropout delta `0.006417258328782094`。clean CER `0.022665210267613325`，比父版本高 `0.0008192244675040948`，比 base 高 `0.0001365374112506819`。分场景相对父版本：dropout `+0.006417258328782094`，noise `+0.0005461496450027345`，far_field `0.0`，reverb `-0.0008192244675040948`。preference accuracy `0.8419354838709677`，raw policy accuracy `0.24516129032258063`，tie 0。preference 只打 310 条剥离后的 dropout，不和上一次 489 条未剥离 pair 的 `0.9856850715746421` 直接比。

dropout 500 条用同一套对齐重数。这次删除是 gap 内 756、0.40 之前 219、0.55 之后 70；插入是 gap 内 10、0.40 之前 25、0.55 之后 2；替换 342。假设 6316 字，参考 7324 字，编辑 1424，micro CER `0.19442927362097215`。父版本是 gap 内删除 780、0.55 之后 57、替换 307、假设 6262 字、编辑 1377。上一次 probe 是 gap 内删除 762、0.55 之后 187、gap 插入 24、替换 374、假设 6211 字、编辑 1593。0.55 之后的删除从上次多出来的 130 次收到 13 次。dropout CER 仍高于父版本。签名仍是 `OTHER`。同一次 run 不改学习率。

这一步不作为父版本。可用模型仍是 SFT step 30。Phase 2 不标完成。test 不评。RL 不开始。不恢复这个 run。

## 2026-10-19 本地重算 gap-strip probe

用 Drive 上的 `eval/step_000010/scored.jsonl`（1904310 字节）和 step 30、`20261018T052928` 的 scored 在本机重跑 `judge_dpo_checkpoint`。2500 行的 `sample_id` 顺序一致。重算结果与当时的 decision 相同：`passed=false`，唯一风险是 degraded macro CER 上升 `0.0015360458765701773`。五个场景的空输出、重复、截断都是 0，有效输出都是 1。dropout 的超长率是 `0.006`，低于 `0.01`。preference `0.8419354838709677` 高于 `0.55`。clean、单场景增幅都在上限内。

dropout 500 条里 424 条假设与 step 30 相同。编辑数变差 21 条，变好 23 条，净增 47 个编辑。其中 3 条假设写成了 `humanrightswatch`：`BAC009S0738W0443__dropout` 多 16 个编辑，`BAC009S0748W0156__dropout` 多 14 个，`BAC009S0755W0182__dropout` 多 12 个，合计 42。这个字符串不在本地 manifest、pair 或文档里。下一条大头是 `BAC009S0733W0324__dropout`，假设停在「公司去年实现营业收」，多 10 个编辑。

本地反事实：只把这 3 条换回 step 30 的假设，macro 仍高 `0.00010240305843800535`，dropout delta 剩 `0.000682687056253406`，仍然失败。只删掉 0.40 之前的插入，macro 仍高 `0.0006826870562534199`。0.40 之前和 gap 内的插入一起删，macro 仍高 `0.000341343528126703`。把全部 dropout 假设换回 step 30 之后，macro delta 是 `-0.00006826870562534615`，其余风险为空，这组数字才能过。因此这次输出不能靠改分数通过。门禁不放宽，这个 run 不恢复，学习率不在这次结果上改。

## 2026-10-19 gap-strip 失败根因与非静音四场景 DPO 方案

背景：连续三次 DPO 运行（`20261017T131802` 全场景 pilot、`20261018T052928` 未剥离 dropout probe、`20261018T132840` gap-strip dropout probe）都在 step 10 因 `dropout` 上升推高 `degraded_macro_cer` 而触发风险停止。需要将 `pairs_gap_strip` 的文本构造与 `scored.jsonl` 的逐条错例对齐，明确为什么剥离 `[0.40, 0.55]` 字符区间后 `dropout` 仍然退化，并制定下一步可复现方案。

范围：核对本机 `results/phase3_dpo_gap_strip_logprob_20261018/` 的 `2573` 条 train dropout pair、`310` 条 val dropout pair，以及 `results/phase3_dpo_gap_strip_probe_20261018/runs/20261018T132840_dpo_06b_zh_seed42/eval/step_000010/scored.jsonl` 的 `2500` 条预测。新增 `configs/train/dpo_pilot_nondropout.yaml`，不改动已有历史配置文件与冻结 pair 目录。

设计（根因诊断与新配置约定）：

1. **字符比例 `[0.40, 0.55]` 与波形静音时间轴错位**：`apply_dropout` 抹掉的是音频采样点 `[0.40, 0.55]`，但因句首尾静音与语速波动，被静音抹掉的字常落在字符下标 `[0.33, 0.50]` 或 `[0.42, 0.60]`。`strip_dropout_gap` 按固定字符比例 `[0.40, 0.55]` 截断后，`2573` 条 train dropout pair 中有 `1293` 条（`50.3%`）误删了本来在音频中有声且被 SFT 正确识别的匹配字（`op == "match"`），且有 `1236` 条（`48.0%`）原始 SFT 仅跳过一段连续静音、前缀后缀完全正确的样本被切成了残缺词组（如 Step 1 的 `BAC009S0598W0352__dropout` 被切成 `chosen=在带伤坚持完成上午百米自由泳预赛之后` 对 `reject=在带伤坚持完成上百米自由泳预赛之后`，`BAC009S0522W0194__dropout` 被切成 `上述人士没认这份资料` 对 `上述人士认这份资料`）。剥离后仍有 `2149/2573`（`83.5%`）`char_length_delta < 0`（其中 `1626` 条占 `63.2%` 为纯删字差异），持续迫使模型在零能量静音段吐字。
2. **探针高学习率激发少数静音切口幻觉与提前停机**：在 `20261018T132840` 的 500 条 val `dropout` 中，23 条变好（减少 32 个编辑，如恢复出 `保时捷`、`行政处罚`），21 条变差（增加 79 个编辑）。变差集中在前 5 条：3 条在静音处吐出预训练英文先验 `Human Rights Watch`（`BAC009S0738W0443__dropout` +16、`BAC009S0748W0156__dropout` +14、`BAC009S0755W0182__dropout` +12，合计 +42 编辑），2 条在静音切口提前输出 EOS（`BAC009S0733W0324__dropout` +10、`BAC009S0727W0465__dropout` +8，合计 +18 编辑）。其余 `495/500` 条样本实际净减少 13 个编辑。
3. **非静音四场景过滤方案**：在 2026-10-17 冻结的 `ref_logprobs/pairs_logprob/` 中，排除无声学支撑的 `dropout` 后，`clean`（726/109）、`noise`（1428/189）、`reverb`（753/121）、`far_field`（733/114）合计 `3640` 条 train pair、`533` 条 val pair，其中等长同音替换分别占 `91.6%`（`3333/3640`）和 `89.1%`（`475/533`）。新增 `configs/train/dpo_pilot_nondropout.yaml`，保持正式 pilot 预算（`lr=5e-6`、`warmup_steps=2`、`max_steps=100`、`beta=0.1`），设置 `train_scenarios: [clean, noise, reverb, far_field]`，直接复用 2026-10-17 冻结的 `ref_logprobs.jsonl` 与填充 pair，在全量 2500 条 val（含 `dropout`）上评测门禁。

测试：运行 `.venv/bin/python -m unittest discover -s tests -v`，验证 `dpo_pilot_nondropout.yaml` 加载、哈希钉死、场景过滤与 `pilot` 预算检查全部通过。

验收：本地全部单元测试与目录 README 合同通过后，等待在标准 A100 上启动新 run。

影响：不需要重新生成 pair 或重跑离线 reference logprob。可用父模型仍是 SFT step 30 release `sft_06b_zh_20261015T234939_step000030`。`publishable` 保持 false。Phase 2 和 Phase 3 仍未标完成。

## 2026-10-19 非静音四场景 DPO pilot 已启动

标准 A100 会话 `lora-asr-dpo-nondropout`（`gpu-a100-s-kkb-usc1c2-11unk4js01xc1`，Shape Standard，`NVIDIA A100-SXM4-40GB`）。没有用 `--high-mem`。Drive 上 `release.json` 是 1416 字节。配置是 `configs/train/dpo_pilot_nondropout.yaml`。run `20261019T012838_dpo_06b_zh_seed42`。第一次包装脚本在环境重建通过、`TORCH 2.11.0+cu130 True` 已经打印之后，因为没有接住 torch 探针的标准输出而退出，当时没有 optimizer step，也没有 pid 文件。修正后的同一次 VM 重新拉起，才是这个 run。

启动日志有 `DPO_TRAIN_SCENARIOS clean,noise,reverb,far_field train_pairs=3640 val_pairs=533`、`DPO_STEP0 checked=20 max_abs=0`、`DPO_PREFLIGHT_OK` 和 `DPO_RUNTIME_BUDGET disabled`。step 1 学习率 `2.5e-06`，step 2 学习率 `5e-06`。step 10 的 pair loss 是 `0.692519`，grad norm `3.484375`，学习率 `4.9264134e-06`。trainer PID 3121。sub2 保活单元是 `dpo-nondropout-keepalive.service`。不恢复 `20261018T132840`、`20261018T052928` 或 `20261017T131802`。test 不评。RL 不开始。

## 2026-10-19 非静音四场景 DPO pilot 在 step 10 风险停止

`2026-10-19T02:11:25Z`，`status.json` 是 `gate_failed` / `gate_risk`，`global_step` 10。日志是 `DPO_LOCATE signature=OTHER gap_insertions=10 dropout_length_ratio=0.855543 dropout_delta=0.002731 clean_delta=0.000273`，然后 `DPO_GATE step=10 passed=False risk_stop=True preference=0.718574`，`DPO_GATE_STATUS failed`，`DPO_DONE global_step=10`，`DPO_EXIT 0`。

2500 条 val 相对 SFT step 30：`degraded_macro_cer` 从 `0.07352539595849263` 升到 `0.0741398143091207`，差 `0.0006144183506280737`。分场景差是 dropout `+0.0027307482250136517`（`0.18801201529219005` 到 `0.1907427635172037`），noise `-0.0002730748225013638`（`0.057755324959038776` 到 `0.05748225013653741`），reverb 和 far_field 都是 `0.0`。clean 相对父版本 `+0.0002730748225013638`（`0.022119060622610594`），相对 base `-0.00040961223375204914`。preference accuracy `0.7185741088180112`，raw policy accuracy `0.0375234521575985`，ties `0`。长度桶 raw accuracy：更短 rejected `0.07894736842105263`（38 对），等长 `0.035789473684210524`（475 对），更长 rejected `0.0`（20 对）。风险只有 degraded macro CER 上升。`selected_step` 是 null，`publishable` 是 false。

Drive `runs/20261019T012838_dpo_06b_zh_seed42/` 上已有 step 10 checkpoint、`eval_decisions.jsonl` 和 `eval/step_000010/scored.jsonl`。训练器在退出前没有把最终的 `status.json`、`gate.json` 和 `run_report.md` 同步上去。退出后按源文件哈希补了这三份，读回一致：`status.json` 110 字节，`gate.json` 2011 字节，`run_report.md` 421 字节，`eval_decisions.jsonl` 1057 字节。

这一步不作为父版本。不恢复 `20261019T012838`。可用模型仍是 SFT step 30。`publishable` 保持 false。Phase 2 和 Phase 3 仍未标完成。不评 test，不启动 RL。

## 2026-10-19 非静音 DPO step 10 失败根因与无标点原位替换 + 去 Clean 同音拷贝 DPO 修复方案

背景：`20261019T012838_dpo_06b_zh_seed42`（`configs/train/dpo_pilot_nondropout.yaml`）在 step 10 评测时，preference accuracy（`0.7185741088180112`）、`noise` 改善（`-0.0002730748225013638`）、`dropout` 单场景升幅（`+0.0027307482250136517 <= 0.01`）与 `clean` 门禁全部通过，唯一触发 `passed=false` / `risk_stop=true` 的是 `degraded_macro_cer` 相对父版本 SFT step 30 上升 `+0.0006144183506280737`（上限 `0.0`）。需要将编辑变化精确分解到每条样本与训练批次，并对初步筛选出的 781 条原位替换对做二次深度审计，落地完整代码与配置修复。

范围：核对 2500 条 val 编辑变化、3640 条 train / 533 条 val 非 dropout 偏好对分布及 781 条原位替换对的跨场景同音重复污染；更新 `train/dpo_contract.py`、`train/dpo_schema.py`、`train/train_dpo.py` 与 `tests/test_dpo_contract.py`，更新 `configs/train/dpo_pilot_aligned.yaml`。不放宽任何门禁阈值，不修改历史配置或已冻结的 `pairs/`、`ref_logprobs/`、`pairs_gap_strip/` 磁盘文件。

设计（定量根因与修复机制）：

1. **逐场景与逐样本编辑分解（`+18` 次净增编辑）**：
   - 每个 val 场景含 500 条语音、总参考字数 `7324`（1 次编辑对应单场景 micro CER `1/7324 = 0.0001365374112506827`，对应四退化场景等权宏平均 `degraded_macro_cer` `1/29296 = 0.00003413435281267067`）。
   - SFT step 30 父版本在四退化场景的编辑数为 `noise=423`、`reverb=184`、`far_field=170`、`dropout=1377`（合计 `2154/29296 = 0.07352539595849263`），`clean=160`（base 为 `165`）。
   - `20261019T012838` step 10 的编辑数为 `noise=421`（`-2`）、`reverb=184`（`0`）、`far_field=170`（`0`）、`dropout=1397`（`+20`，与 SFT step 40 的 `0.1907427635172037` 完全同值），四退化场景净增 `+20 - 2 + 0 + 0 = +18` 次编辑（`+18/29296 = +0.0006144183506280737`）。
   - `DPO_LOCATE` 报告的 `gap_insertions=10` 100% 来自单条静音悬崖样本 `BAC009S0748W0156__dropout`（金标 `'就有一零个国家级贫困县'`）：父模型 SFT step 30 输出 `'就有一零个，加级贫困县'`（用逗号 `'，'` 跨过 `[0.40, 0.55]` 零能量静音段后继续输出中文，计 2 次编辑），而 DPO step 10 在静音切口把 `'，'` 翻转为英文静音先验 `'就有一零个Human Rights Watch'`（计 16 次编辑，单条净增 **`+14` 次编辑**，占 `dropout` 净增 `+20` 的 `70%`、占总宏平均超标 `+18` 的 `77.8%`）。
2. **四项耦合数学与数据根因**：
   - **根因 A：`rejected` 标点不对称与不等长 token 的配分函数 $\log Z$ 未抵消（36.8% 含标点、48.6% token 不等长，前 10 步占 50%）**：在 3640 条非 dropout 训练对中，`chosen_text`（AISHELL-1 原始金标）0% 含标点，而 `rejected_text`（SFT `prediction_raw`）有 `1340/3640`（`36.8%`）含标点（`，。、：·`）、`1769/3640`（`48.6%`）满足 `chosen_token_count != rejected_token_count`（其中 `1158` 条 $T^- > T^+$；前 10 步的 40 条中有 9 条含标点、20 条 $T^+ \ne T^-$）。序列和 DPO 中：
     $$\log \pi_\theta(y^+|x) - \log \pi_\theta(y^-|x) = \sum_{t=1}^{T^+} z_{y_t^+}(c_t^+) - \sum_{t=1}^{T^-} z_{y_t^-}(c_t^-) + \left(\sum_{t=1}^{T^-} \log Z(c_t^-) - \sum_{t=1}^{T^+} \log Z(c_t^+)\right)$$
     当 $T^- > T^+$ 或 `rejected` 含 `'，'` 时，不仅 `'，'` 受到直接 $-1$ 梯度压制，多出的 $+\sum_{t=T^++1}^{T^-} \log \sum_{v \in \mathcal{V}} e^{z_v(c_t^-)}$ 项还直接抬高词表外英文 token（`'Human Rights Watch'`）的 logit 并压低句中中文延续字，导致 `BAC009S0748W0156__dropout` 在静音段从 `'，'` 跌落至 `'Human'`。反之，在无标点、等字长、同 token 数的纯原位替换对（$T^+ = T^-$）上，匹配前缀与首个替换位点 $t_0$ 的上下文完全相同（$c_{t_0}^+ = c_{t_0}^-$），$\log Z(c_{t_0}^-) - \log Z(c_{t_0}^+) \equiv 0$，对 `'，'`、`'Human'` 与 `<|im_end|>` 零直接梯度冲击。
   - **根因 B：无界极端负参考 margin 中的同音专名与金标错字（75.5% 的 `ref_margin < -2.0` nats，`raw_policy_accuracy` 0 翻转）**：3640 条训练对的 `ref_margin = ref_chosen_logp - ref_rejected_logp` 中位数为 `-5.189` nats（val 为 `-6.307` nats），`< -4.0` nats 的样本大量为声学完全同音的人名/代词（`路某/陆某` `-7.37`、`文骁/文晓` `-8.25`、`李朱濠/李朱豪` `-7.02`、`她/他` `-6.99`）或 AISHELL-1 金标错字（`末阳市/莫阳市` `-6.11`、`四百馀人/四百余人` `-4.51`、`显着/显著` `-6.98`），`> 0.0` 的样本（55 条）则是父模型已偏向 chosen 或金标错字（`获熟/获悉`）。这导致 step 10 的 `raw_policy_accuracy` 停留在 `20/533 -> 20/533`（0 条翻转），`noise/reverb/far_field` 仅净减 2 次编辑。
   - **根因 C：781 条原位替换对中仍有 `64.9%`（`507/781`）为 Clean 同音字/人名/AISHELL-1 金标错字在四场景下的重复拷贝（前 10 步占 `27/40 = 67.5%`）**：即便开启 `aligned_substitution_only: true` 与 `ref_margin_min: -4.0` 筛出 781 条非 dropout 对，由于同一句 `source_utterance_id` 在 `clean`、`noise`、`reverb`、`far_field` 下重复出现，只要 SFT step 30 在录音室 `clean` 语音上输出了同音人名/虚词（如 `陈建/陈健`、`周素勤/周素琴`、`杨常青/杨长青`、`房祖名/房祖明`、`泰禾/泰和`、`克而瑞/克尔瑞`、`卖得不好/卖的不好`）或遇到 AISHELL-1 原始金标错别字（如 `BAC009S0517W0317` 金标错字 `'松江新兵镇'` 对 SFT 正确地名 `'松江新浜镇'`，在前 10 步重复出现 2 次；`BAC009S0152W0356` 金标错字 `'避谣'` 对 `'辟谣'`），它在 `noise/reverb/far_field` 下也输出了完全相同的归一化 `rejected_text`（占 781 条中的 `507` 条）。仅挂载在文本解码器自注意力层（`thinker.model`）的 LoRA 在零声学差异的同音词和错标真值上无法学到声学鲁棒性，反而浪费 65% 的梯度更新并扭曲解码器语言先验。而一旦剔除 `clean` 场景自身以及归一化 `rejected_text` 与同句 `clean` 完全一致的重复拷贝，剩余 **`274` 条训练对**（`noise: 203, reverb: 37, far_field: 34`）100% 是 SFT 在 `clean` 上识别正确（或不同）、纯粹因声学退化才发生发音混淆的高价值样本（前 10 步 40 条全部为 `新股/金融`、`未卜/未古`、`推动/驱动`、`骑三轮/提三轮`、`旺销/望乡`、`高危/高龄`、`纵向/定向`、`总量/重量`、`抄底/超低`、`继续/急剧`、`堵漏/阻挠` 等真声学退化对）。
   - **根因 D：AdamW 超参与训练/验证过滤耦合的工程边界**：一方面，SFT step 30 的 `lora_A` 和 `lora_B` 均非零，`eps=1e-8` 使微小残差梯度仍按 $\pm \text{lr}$ 满幅移动全部 458 万个参数，`weight_decay=0.01` 则把 `lora_A/lora_B` 向 0 衰减；但若将阻尼过度拉大到 `warmup_steps=20`、`eps=1e-4`，前 10 步平均学习率仅 `1.375e-6`，累积更新量被削弱 4～5 倍，在 BF16 下难以在 step 10 翻转退化场景字错或稳定越过 `preference_accuracy >= 0.55` 早停线。另一方面，`train/train_dpo.py` 原先对训练集和验证集 `pref_rows` 共用同一套过滤参数（且未导入 `SCENARIOS` 常量），若直接将三退化场景去同音过滤套用到 `pair_val`，会使验证集对数降至 37 条而触发 `gate.min_val_pairs: 100` 门禁失败。
3. **代码与配置修复**：
   - `train/dpo_contract.py`：`select_training_pairs(rows, scenarios, *, aligned_substitution_only=False, ref_margin_min=None, ref_margin_max=None, exclude_clean_identical_rejected=False, clean_rejected_by_utterance=None)` 增加内存级无标点原位等长替换过滤、参考 margin 区间过滤以及基于全量 `clean` 映射的去同音重复拷贝过滤（并提供辅助函数 `build_clean_rejected_map(rows)`）。
   - `train/dpo_schema.py`：`dpo` 块支持可选字段 `val_scenarios`（list[str] | null，默认 `None`）、`aligned_substitution_only`（bool，默认 `False`）、`exclude_clean_identical_rejected`（bool，默认 `False`）、`ref_margin_min`（float | null，默认 `None`）、`ref_margin_max`（float | null，默认 `None`），全部纳入 `semantic_dpo_config_sha256`；`_check_pilot_budget` 允许 `train_scenarios` 为 `None`、`["clean", "noise", "reverb", "far_field"]` 或 `["noise", "reverb", "far_field"]`，允许 `val_scenarios` 为 `None` 或 `["clean", "noise", "reverb", "far_field"]`。
   - `train/train_dpo.py`：补齐 `SCENARIOS` 常量导入；在核对离线缓存 `_check_cache` 后，对 `train_pair_rows` 按 `train_scenarios` 与 `exclude_clean_identical_rejected` 过滤，对 `pref_rows`（`pair_val`）按 `val_scenarios` 过滤（不开启 `exclude_clean_identical_rejected`），并在退出前调用 `sync_drive_files` 将最终 `status.json`、`gate.json`、`run_report.md` 和 `listen_waiver.json` 同步到 Drive。
   - `configs/train/dpo_pilot_aligned.yaml`：保持正式 pilot 预算（`lr=5e-6`、`beta=0.1`、`max_steps=100`、`eval_steps=10`、`save_steps=10`），启用 `train_scenarios: [noise, reverb, far_field]`、`val_scenarios: [clean, noise, reverb, far_field]`、`aligned_substitution_only: true`、`exclude_clean_identical_rejected: true`、`ref_margin_min: -4.0`、`ref_margin_max: 0.0`（本机冻结对过滤后为 `274` 条 train：`noise 203, reverb 37, far_field 34`；`118` 条 val：`clean 26, noise 29, reverb 33, far_field 30`），并将优化器阻尼校准为 `warmup_steps: 5`、`eps: 0.000001`（`1e-6`）、`weight_decay: 0.0`。

测试：运行 `.venv/bin/python -m unittest discover -s tests -v`，验证 `select_training_pairs` 原位替换、margin 过滤与去 `clean` 同音拷贝过滤，`dpo_pilot_aligned.yaml` 配置解析与 274/118 对数核对，以及 `train_dpo.py` 的 `SCENARIOS` 导入和结束时 Drive 同步合同全部通过。

验收：本地单元测试、编译检查与目录 README 合同测试全部退出码 0；后续在标准 A100 上以新 `run_id` 启动 `configs/train/dpo_pilot_aligned.yaml` 并通过 2500 条 val 冻结门禁后，才允许选出 DPO checkpoint。

影响：不改动已冻结的 `pairs/`、`ref_logprobs/`、`pairs_gap_strip/` 与任何历史配置文件。当前可用模型仍为 SFT step 30 release `sft_06b_zh_20261015T234939_step000030`。

## 2026-10-19 对齐替换 DPO 已启动

本地 `python3 -m unittest tests.test_dpo_contract tests.test_directory_readmes -v` 是 33 项通过。标准 A100 会话 `lora-asr-dpo-aligned`（`gpu-a100-s-kkb-ass1c2-oqxw2a89wihw`，Shape Standard，没有 `--high-mem`）。第一次 `drivemount` 在凭证传播后仍是 `ValueError: mount failed`，CLI 退出码 0。随后 `force_remount` 打印 `Mounted at /content/drive`，`release.json` 是 1416 字节。

配置是 `configs/train/dpo_pilot_aligned.yaml`。run `20261019T130250_dpo_06b_zh_seed42`。音频包 SHA-256 `99d15056e560eb27537f75b065cd7307621d1c4e57eddd65cdadfcb5d8d6c66c`。填充 train `43473f10fbb9c0349adf2eb76eaf647506978c96eabb5b09847994456519c1b1`，填充 val `b82fc2783726555231164025a0e2d7faa82b5391bdcf8b98c6397eec1aff5eca`，cache `5a77f22586dcf1b73393124055eb02466a904d8bdbdd2d44d40ace78000ef43e`，report `8128e2a7731fda2b211dbaf840f38c31b132f26a9fab6f7cd3e8a11d4d805e78`，null train `d958d602dcc4a252ac75ce36d99536e8a53f48fa72929d1b69ead2647dba3f1b`，null val `a7591e86f5a33efb314e29304153fe55e2d24ffba9bdd8b2739f953568b116d2`，adapter `a185a61f54b6fd74bd5e3bfa1173e15450cfbb772731567671a928aad805096a`。环境重建退出码 0，`torchao` 0.10.0 已卸掉。`TORCH 2.11.0+cu130 True`。trainer PID 3668。包装日志有 `LAUNCH_DETACHED`。sub2 保活单元是 `dpo-aligned-keepalive.service`，日志 `/root/colab-keepalive/dpo_aligned_keepalive.log`。

启动时期望的场景行是 `DPO_TRAIN_SCENARIOS noise,reverb,far_field val_scenarios=clean,noise,reverb,far_field train_pairs=274 val_pairs=118`。这条已经打出。step 0 抽查是 `DPO_STEP0 checked=20 max_abs=0`。step 10 的 pair loss 是 `0.692625`，grad norm `3.343750`，学习率 `4.9693129e-06`。

## 2026-10-19 对齐替换 DPO 在 step 10 风险停止

`2026-10-19T13:43:02Z`，`status.json` 是 `gate_failed` / `gate_risk`，`global_step` 10。日志是 `DPO_LOCATE signature=QUIET gap_insertions=0 dropout_length_ratio=0.854861 dropout_delta=0.000546 clean_delta=0.000410`，然后 `DPO_GATE step=10 passed=False risk_stop=True preference=0.610169`，`DPO_GATE_STATUS failed`，`DPO_DONE global_step=10`，`DPO_EXIT 0`。保活在 `2026-10-19T13:43:35Z` 记到 `KEEPALIVE_DONE`。trainer PID 3668 已退出。

2500 条 val 相对 SFT step 30：degraded macro CER 从 `0.07352539595849263` 升到 `0.0736619333697433`，差 `0.00013653741125067842`，也就是 4 个编辑。分场景差是 dropout `+0.0005461496450027414`（+4）、noise `+0.00013653741125068536`（+1）、reverb `-0.0001365374112506819`（-1）、far_field `0.0`。clean CER `0.022255598033861276`，相对父版本 `+0.00040961223375204567`（+3），相对 base `-0.00027307482250136725`（-2）。preference accuracy `0.6101694915254238`，raw policy accuracy `0.00847457627118644`，tie 0。118 条验证 pair 全是等长。改善的退化场景数是 1。`selected_step` 是 null，`publishable` 是 false。听音处置仍是 `unlistened`。

和 step 30 逐条对齐后，2500 条里 40 条假设不同。没有 `Human Rights Watch`，gap 插入是 0，签名是 `QUIET`。最大的单条变差是 clean `BAC009S0727W0413`，从「林姐一角后」变成「林姐与小号」，多 3 个编辑。noise `BAC009S0750W0197` 把「二次抗议」改回「二次探底」，少 2 个编辑。

Drive `runs/20261019T130250_dpo_06b_zh_seed42/` 上已有 step 10 checkpoint、`gate.json`、`status.json`、`run_report.md`、`eval_decisions.jsonl` 和 `eval/step_000010/scored.jsonl`。`run_report.md` 的 sync 是 `synced`。标准 A100 会话 `lora-asr-dpo-aligned` 已停止。High-RAM 会话 `gpu-a100-hm-kkb-ass1c2-2ywpbxh5nicjr` 未动。

这一步不作为父版本。不恢复 `20261019T130250`，也不恢复 `20261019T012838`、`20261018T132840`、`20261018T052928` 或 `20261017T131802`。可用模型仍是 SFT step 30。`publishable` 保持 false。Phase 2 和 Phase 3 仍未标完成。不评 test，不启动 RL。

## 2026-10-20 对齐替换 DPO step 10 残余 +4 编辑根因与差异 Token 梯度掩码 + 单字替换优化方案

背景：`20261019T130250_dpo_06b_zh_seed42`（`configs/train/dpo_pilot_aligned.yaml`）在 step 10 将 `locate_signature` 收敛至 `QUIET`（`gap_insertions=0`，彻底根除 `Human Rights Watch` 英文幻觉），`preference_accuracy` 达 `0.6101694915254238`（`72/118 >= 0.55`），`reverb` 净减 1 次编辑（`improved_degraded_scenarios=1`），`far_field` 持平，`clean` 门禁与健康度全部通过；唯一触发 `passed=false` / `risk_stop=true` 的是四退化场景净增 `+4` 次编辑（`degraded_macro_cer` 上升 `+0.00013653741125067842`：`dropout +4`、`noise +1`、`reverb -1`、`far_field 0`）。需要精确拆解这 `+4` 次净增编辑与 21 条新增逗号的逐 token 机制并完成优化。

范围：核对 `20261019T130250` 的 2500 条 `scored.jsonl` 与前 10 步 40 条训练对；更新 `train/dpo_contract.py`、`train/dpo_schema.py`、`train/train_dpo.py`、`configs/train/dpo_pilot_aligned.yaml` 与 `tests/test_dpo_contract.py`。不放宽任何门禁阈值，不修改历史配置文件或已冻结的磁盘文件。

设计（定量根因与修复机制）：

1. **逐场景与逐样本编辑分解（单字消歧净减 `-12` 次编辑 vs 后缀级联净增 `+16` 次编辑）**：
   - 在四退化场景中，10 条语音取得单字混淆纠错收益，合计净减 **`-12` 次编辑**（如 `reverb` `BAC009S0730W0481` `'金一丹写新书' -> '金一丹携新书'` `-1`、`BAC009S0757W0180` `'人民币对美元' -> '人民币兑美元'` `-1`；`far_field` `BAC009S0726W0171` `'单价记录' -> '单价纪录'` `-1`、`BAC009S0747W0372` `'张嘉伟' -> '张家伟'` `-1`；`noise` `BAC009S0750W0197` `'二次抗议' -> '二次探底'` `-2`、`BAC009S0726W0471` `-2`、`BAC009S0727W0413` `'云锦一角' -> '云姐一角'` `-1`；`dropout` `BAC009S0736W0347` `'朱健' -> '朱建'` `-1`、`BAC009S0761W0280` `-1`、`BAC009S0738W0350` `-1`）。
   - 但 13 条退化语音与 2 条 `clean` 语音在前缀出现偏差后发生后缀级联替换，抵消了上述收益：包括 `BAC009S0730W0210__dropout` `'纳入结款' -> '纳入监管'`（`+2`）、`BAC009S0730W0481__dropout` `'京东单鞋与多者见面' -> '京东单选与多选界面'`（`+2`）、`BAC009S0725W0200__noise` `'出现严重东西差异' -> '出现严重攻击差异'`（`+2`）、`BAC009S0727W0413__clean` `'林姐一角后' -> '林姐与小号'`（`+3`）、`BAC009S0759W0477__clean` `'最长寿驯养' -> '最长寿炫耀'`（`+1`）；同时有 **21 条验证样本在 `prediction_raw` 中凭空插入了中文逗号 `'，'`**（尽管 274 条训练对 0% 含标点）。
2. **两项耦合根因**：
   - **根因 A：整句序列和反传在共享后缀位点（$t > t_0$，占全句平均 `58.9%` token、占非零梯度位点约 `88%`）上的暴露偏差负梯度与 $\log Z$ 标点漂移**：在无标点等长同 token 替换对中，前缀上下文仅在首个替换位点 $t \le t_0$ 满足 $c_{t_0}^+ \equiv c_{t_0}^-$（此时 $\nabla_z (\log \pi_\theta(y_{t_0}^+|c_{t_0}) - \log \pi_\theta(y_{t_0}^-|c_{t_0})) = e_{y_{t_0}^+} - e_{y_{t_0}^-}$，配分函数 $\log Z(c_{t_0})$ 100% 精确抵消）。然而在 $t > t_0$ 的全部共享正确后缀字（$y_t^+ == y_t^-$，含 `<|im_end|>`）上，$c_t^+$ 含正确字 $y_{t_0}^+$ 而 $c_t^-$ 含错字 $y_{t_0}^-$（$c_t^+ \ne c_t^-$）：一方面，对 $-\log \pi_\theta(y_t^+|c_t^-)$ 反传负梯度直接惩罚模型在前缀出现偏差后继续输出正确金标后缀字 $y_t^+$，导致验证集一旦在句中遇到未翻转错字或 `dropout` 静音缺口，后续文本便偏离金标发生级联替换；另一方面，$+\nabla_z \log Z(c_t^-) = +p(c_t^-)$ 推高了错字前缀后的停顿标点 `'，'`，并且约 88% 的梯度范数被后缀捷径消耗（`raw_policy_accuracy` 仅 `1/118`）。
   - **根因 B：多字替换噪声对（`edit_count >= 2`，占 274 条的 `40.1%`，前 10 步占 `13/40`）的整词臆测与次位点前缀偏移**：前 10 步包含 `吸取教训 -> 施肥浇水`（`ed=4`）、`这谜题 -> 敬礼图`（`ed=3`）、`底牌家具 -> 洗牌加剧`（`ed=3`）、`亲子 -> 新股`（`ed=2`，后缀含 `'协同监管'`）以及 AISHELL-1 金标异文 `第三阶段 -> 第三界的`（`ed=2`），在多字掩蔽处推动解码器以高频词组覆盖声学特征；而仅保留 `max_edit_count: 1` 的单字原位替换对仍有 **`164` 条训练对**（`noise: 118, far_field: 25, reverb: 21`），100% 为单字声学混淆且替换位点前缀完全一致。
3. **代码与配置修复**：
   - `train/dpo_contract.py`：`select_training_pairs` 新增 `max_edit_count: int | None = None`；新增纯函数 `mismatched_token_mask(chosen_labels, rejected_labels)` 计算监督 token 序列中 `chosen != rejected` 的布尔掩码。
   - `train/dpo_schema.py`：`dpo` 配置块支持可选字段 `max_edit_count`（int | null，默认 `None`）和 `loss_token_scope`（`"sequence"` 或 `"mismatched_only"`，默认 `"sequence"`），非默认值时纳入 `semantic_dpo_config_sha256`。
   - `train/train_dpo.py`：训练集过滤传入 `max_edit_count`（验证集保持 118 条）；当 `loss_token_scope == "mismatched_only"` 时，通过 `diff_logp + (full_logp - diff_logp).detach()` 构造直通序列 logprob，保持前向序列和与离线缓存 100% 一致，反向仅对差异 token 位点求导。
   - `configs/train/dpo_pilot_aligned.yaml`：设置 `max_edit_count: 1`（`164` 条 train / `118` 条 val）、`loss_token_scope: mismatched_only` 与 `warmup_steps: 10`。

测试：运行 `.venv/bin/python -m unittest discover -s tests -v`，验证 `max_edit_count=1`（冻结对过滤后 `164` train / `118` val）、`mismatched_token_mask`、直通式 `mismatched_only` 前向等价与梯度掩码，以及历史配置语义哈希不变。

验收：本地单元测试、编译检查与目录 README 合同测试全部通过；后续在标准 A100 上以新 `run_id` 启动 `configs/train/dpo_pilot_aligned.yaml` 并通过 2500 条 val 门禁。

影响：不改动已冻结的磁盘缓存与历史配置文件。当前可用模型仍为 SFT step 30 release `sft_06b_zh_20261015T234939_step000030`。

## 2026-10-20 epoch 尾步保存与学习率对齐

背景：164 条训练对、梯度累积 4、`max_epochs: 1` 时，优化器在 step 41 结束。原 runner 只在 10 的倍数和 `max_steps` 保存。step 41 不保存，循环结束后 `last_saved != global_step`，抛出 `run ended without a checkpoint at the last completed step`，退出码 1，不写 `gate.json`。另外 `HorizonScheduler` 从倍率 0 开始，原先在 `optimizer.step()` 之后才推进，step 1 的实际更新是 0，日志却打印下一步的学习率。

范围：`train/sft_contract.py` 增加 `update_ends_schedule`，`pending_checkpoint_eval` 接受可选的 `schedule_end_step`。`train/train_dpo.py` 在采样结束的那一步保存并评测，并在 Adam 之前推进 scheduler。不改 `configs/train/dpo_pilot_aligned.yaml` 的预算，不改 SFT runner 的保存条件，不恢复已失败的 DPO run。

设计：164 条样本产生 41 个更新，最后一批仍是 4 条。step 10、20、30、40 照旧保存评测。step 41 也保存并评测。warmup 10、学习率 `5e-6` 时，step 1 的日志和学习率都是 `5e-7`，step 10 都是 `5e-6`。

测试：`tests.test_train_sft` 核对 41 步尾步和 step 10 倍率。`tests.test_dpo_contract` 核对 DPO 源码先 `scheduler.step()` 再 `optimizer.step()`，并在 `ends_schedule` 时保存评测。

验收：上述测试通过后，用新 `run_id` 在标准 A100 上启动 `configs/train/dpo_pilot_aligned.yaml`。日志需有 `train_pairs=164`、`val_pairs=118`、`DPO_STEP0`、`DPO_PREFLIGHT_OK`、`DPO_RUNTIME_BUDGET disabled`。step 1 的 `lr` 是 `5e-7`。若训练走到 step 41，该步必须有 checkpoint 和 decision，不能以 `failed` 收尾。

## 2026-10-20 对齐 DPO 新 run 已启动

`20261020T014427_dpo_06b_zh_seed42` 已在标准 A100 会话 `lora-asr-dpo-edit1`（运行时 `gpu-a100-s-kkb-ass1c0-1n0b237js6cy3`，Shape Standard）启动。配置是 `configs/train/dpo_pilot_aligned.yaml`，没有 `--resume`，也没有恢复 `20261019T130250_dpo_06b_zh_seed42`。父版本仍是已发布的 SFT step 30 release `sft_06b_zh_20261015T234939_step000030`。

启动日志：

- `DPO_TRAIN_SCENARIOS noise,reverb,far_field val_scenarios=clean,noise,reverb,far_field train_pairs=164 val_pairs=118`
- `DPO_STEP0 checked=20 max_abs=0`
- `DPO_PREFLIGHT_OK`
- `DPO_RUNTIME_BUDGET disabled`
- step 1：`lr=5e-07`，loss `0.693147`，grad_norm `2.765625`
- step 10 优化器步：`lr=5e-06`，loss `0.693081`，grad_norm `2.875000`

step 10 的 2500 条 val 门禁还没有返回。`publishable` 保持 false。不选 checkpoint。Phase 2 和 Phase 3 不标完成。test 不评。RL 不开始。sub2 保活单元是 `dpo-edit1-keepalive.service`，日志是 `/root/colab-keepalive/dpo_edit1_keepalive.log`。

## 2026-10-20 step 10 门禁未通过

`20261020T014427_dpo_06b_zh_seed42` 在 step 10 风险停止。`status.json` 是 `gate_failed` / `gate_risk`，`gate.json` 状态是 `failed`，`selected_step` 是 null，`publishable` 是 false。进程 `DPO_EXIT 0`。epoch 尾步没有走到，step 41 的保存路径这次没有被执行。

两条风险：

- degraded macro CER 从父版本 `0.07352539595849263` 升到 `0.07362779901693065`，增加 `0.00010240305843801922`。按每场景 7324 个参考字，四个退化场景是 dropout +12、far_field -3、noise -4、reverb -2，净增 3 个编辑。三个退化场景变好，dropout 变差。`improved_degraded_scenarios` 是 3。
- preference accuracy 是 `0.5`，118 对里 59 正 59 负，tie 是 0。raw policy accuracy 是 `0.0`，118 条等长对里没有一条的 policy 序列 logprob 更偏向 chosen。门槛是至少 `0.55` 且严格高于 `0.5`。

clean CER 是 `0.02239213544511196`，相对父版本 +4 个编辑，相对 base 少 1 个编辑。`DPO_LOCATE` 是 `QUIET`，gap 插入 0，dropout 长度比 `0.8533588203167668`。step 10 loss 是 `0.693081`，step 1 是 `0.693147`。

Drive `runs/20261020T014427_dpo_06b_zh_seed42/` 有 `gate.json`、`status.json`、`latest.json`、`eval_decisions.jsonl`、`eval/step_000010/scored.jsonl`、`metrics.json`、`by_scenario.csv`、`preference.json` 和 `checkpoints/step_000010`。`latest.json` 指向 step 10，那是最后保存的进度，不是选出的模型。不恢复这一轮。可用模型仍是 SFT step 30。Phase 2 和 Phase 3 不标完成。test 不评。RL 不开始。

## 2026-10-20 序列和单字对

`20261020T014427` 的 preference 停在 `0.5`，因为 `warmup_steps: 10` 让 step 10 才第一次用满 `5e-6`，而且 `mismatched_only` 只对一个 token 反传。用户说「开始吧」，门禁保持硬性条件。新配置是 `configs/train/dpo_pilot_edit1_seq.yaml`。164 条单字过滤、118 条验证对、冻结 pair 哈希和全部门禁阈值与 `dpo_pilot_aligned.yaml` 相同，包括 `degraded_macro_increase_max: 0.0` 和 `min_preference_accuracy: 0.55`。`loss_token_scope` 改为 `sequence`。`warmup_steps` 改为 2，step 1 的学习率是 `2.5e-6`，step 2 起是 `5e-6`。`eps` 仍是 `1e-6`，`weight_decay` 仍是 `0.0`。不恢复 `20261020T014427`。step 10 不过就停止。

`20261020T024637_dpo_06b_zh_seed42` 已在 `2026-10-20T02:46:37Z` 启动。会话是标准 A100 `lora-asr-dpo-seq`（运行时 `gpu-a100-s-kkb-usc1f1-3a0z87pyzpc69`，Shape Standard）。配置是 `configs/train/dpo_pilot_edit1_seq.yaml`，没有 `--resume`。父版本仍是已发布的 SFT step 30 release `sft_06b_zh_20261015T234939_step000030`。Drive 上的 `release.json` 是 1416 字节。输出目录是 `results/phase3_dpo_edit1_seq_20261020/runs/20261020T024637_dpo_06b_zh_seed42`。训练进程 PID 是 5482。

启动日志：

- `DPO_TRAIN_SCENARIOS noise,reverb,far_field val_scenarios=clean,noise,reverb,far_field train_pairs=164 val_pairs=118`
- `DPO_STEP0 checked=20 max_abs=0`
- `DPO_PREFLIGHT_OK`
- `DPO_RUNTIME_BUDGET disabled`
- step 1：`lr=2.5e-06`，loss `0.693147`，grad_norm `2.765625`
- step 2：`lr=5e-06`，loss `0.693201`，grad_norm `2.437500`

torch 是 `2.11.0+cu130`，CUDA 可用。sub2 保活单元是 `dpo-seq-keepalive.service`，日志是 `/root/colab-keepalive/dpo_seq_keepalive.log`。

## 2026-10-20 序列单字对 step 10 门禁未通过

`20261020T024637_dpo_06b_zh_seed42` 在 step 10 风险停止。`status.json` 是 `gate_failed` / `gate_risk`，时间 `2026-10-20T03:26:52Z`。`gate.json` 状态是 `failed`，`selected_step` 是 null，`publishable` 是 false。进程 `DPO_EXIT 0`。保活在 `2026-10-20T03:27:56Z` 记下 `KEEPALIVE_DONE`。

唯一写入 `risks` 的是 degraded macro CER 从父版本 `0.07352539595849263` 升到 `0.07386673948661933`，增加 `0.000341343528126703`。按每场景 7324 个参考字，四个退化场景是 dropout +7、far_field +1、noise +2、reverb 0，净增 10 个编辑。`improved_degraded_scenarios` 是 0，所以「至少一个退化场景变好」也没有达到。`risk_stop` 由 macro 上升触发。

preference accuracy 是 `0.576271186440678`，118 对里 68 正 50 负，tie 是 0，过了 `0.55` 且高于 `0.5`。raw policy accuracy 是 `0.00847457627118644`，118 条等长对里只有 1 条的 policy 序列 logprob 更偏向 chosen。preference 的 `mean_loss` 是 `0.6913896347981224`。

clean CER 是 `0.02239213544511196`，相对父版本 +4 个编辑，相对 base 少 1 个编辑，两条 clean 上限都没破。空输出、重复、超长和截断都是 0，`valid_output_rate` 是 1。`DPO_LOCATE` 是 `QUIET`，gap 插入 0，dropout 长度比 `0.854997269251775`，dropout delta `0.0009557618787547906`。step 10 优化器 loss 是 `0.696607`，学习率 `4.9264134e-06`。

Drive `runs/20261020T024637_dpo_06b_zh_seed42/` 有 `gate.json`、`status.json`、`latest.json`、`eval_decisions.jsonl`、`eval/step_000010/scored.jsonl`、`metrics.json`、`by_scenario.csv`、`preference.json` 和 `checkpoints/step_000010`。`latest.json` 指向 step 10，sha256 `5b1df6b16ec948863befcff28aafb51b4a82d04ddd84cb59f5f63af79a94c519`，更新时间 `2026-10-20T02:47:55Z`。那是最后保存的进度，不是选出的模型。不恢复这一轮。可用模型仍是 SFT step 30。Phase 2 和 Phase 3 不标完成。test 不评。RL 不开始。

## 2026-10-20 dropout 父版本锚

`20261020T014427` 的差异字反传让 noise、reverb、far_field 合计少 9 个编辑，dropout 多 12 个。`20261020T024637` 的整句反传让 preference 过线，三个受训场景不再变好。音频塔没有更新，编辑变化在文字解码器自注意力上。新配置是 `configs/train/dpo_pilot_edit1_anchor.yaml`。164 条过滤、门禁、`mismatched_only`、`warmup_steps: 10`、`eps: 1e-6`、`weight_decay: 0.0` 与 `dpo_pilot_aligned.yaml` 相同。每个 optimizer step 额外对一条 train dropout 做 `dropout_anchor_weight: 0.05` 的 token 平均 NLL，教师是该行 `rejected_text`，也就是 SFT step 30 的贪心文本。金标不进入这项损失。不恢复 `20261020T014427` 或 `20261020T024637`。这项不增加 preference 梯度，step 10 仍可能停在 `0.5`。若停在 preference，不在同一轮加大系数。2026-10-20 本机 `tests.test_dpo_contract` 与 `tests.test_directory_readmes` 共 36 项通过。随后开了标准 A100 会话 `lora-asr-dpo-anchor`（运行时 `gpu-a100-s-kkb-ass1c2-i8hz6186ww7b`，Shape Standard）。Drive `release.json` 是 1416 字节。没有 `--resume`，也没有用 `colab upload` 传音频包。`2026-10-20T04:10:12Z` 包装进程已启动。`2026-10-20T04:12:40Z` `LAUNCH_DETACHED`。run `20261020T041218_dpo_06b_zh_seed42`，trainer PID 3581。磁盘空闲 193.31 GB，`REBUILD_EXIT 0`，`TORCH 2.11.0+cu130 True`。`2026-10-20T04:13:16Z` `STEP1_OK`。保活 `dpo-anchor-keepalive.service` 已接上。不恢复 `20261020T014427` 或 `20261020T024637`。

## 2026-10-20 dropout 父版本锚已启动

`20261020T041218_dpo_06b_zh_seed42` 已在标准 A100 会话 `lora-asr-dpo-anchor`（运行时 `gpu-a100-s-kkb-ass1c2-i8hz6186ww7b`，Shape Standard）启动。配置是 `configs/train/dpo_pilot_edit1_anchor.yaml`，没有 `--resume`。父版本仍是已发布的 SFT step 30 release `sft_06b_zh_20261015T234939_step000030`。

`2026-10-20T04:13:18Z` 的保活行：

- `PING`：`preflight=1` `budget=1` `scenarios=1` `anchor=1` `step1=1`，`alive=1`，pid 3581
- `DPO_TRAIN_SCENARIOS noise,reverb,far_field val_scenarios=clean,noise,reverb,far_field train_pairs=164 val_pairs=118`
- `DPO_ANCHOR rows=3816 weight=0.05 target=rejected_text`
- step 1：`loss=0.693147` `grad_norm=2.781250` `lr=5e-07` `pairs=4` `anchor_loss=0.001113` `anchor_tokens=7`
- step 4：`loss=0.694414` `grad_norm=2.265625` `lr=2e-06` `pairs=4` `anchor_loss=0.002105` `anchor_tokens=9`

step 10 的 2500 条 val 门禁还没有返回。`publishable` 保持 false。不选 checkpoint。Phase 2 和 Phase 3 不标完成。test 不评。RL 不开始。sub2 保活单元是 `dpo-anchor-keepalive.service`，日志是 `/root/colab-keepalive/dpo_anchor_keepalive.log`。

## 2026-10-20 dropout 父版本锚 step 10 通过

`20261020T041218_dpo_06b_zh_seed42` 的 step 10 决策是 `passed=true`，`risk_stop=false`，`risks` 为空。`improved_degraded_scenarios` 是 3。preference accuracy `0.576271186440678`（118 对里 68 正，tie 0）。raw policy accuracy `0.00847457627118644`（1/118）。`publishable` 仍是 false。

四个退化场景相对父版本的编辑，每场景 7324 个参考字：noise −9、reverb −1、far_field +2、dropout −2。三个受训场景合计 −8。dropout 没有增加编辑。macro delta 是 `-0.000341343528126703`。clean CER `0.022119060622610594`，相对父版本 +2，相对 base −3。`DPO_LOCATE` 是 `QUIET`，gap 插入 0，dropout 长度比 `0.855407`，dropout delta `-0.00027307482250135684`。step 10 的学习率是 `5e-06`，loss `0.696090`，`anchor_loss=0.000370`。

训练没有停。step 20 的优化器步已经写进日志：`lr=4.8643084e-06`，loss `0.687737`，`anchor_loss=0.007134`。

## 2026-10-20 dropout 父版本锚 step 20 通过

`20261020T041218_dpo_06b_zh_seed42` 的 step 20 决策是 `passed=true`，`risk_stop=false`，`risks` 为空。`improved_degraded_scenarios` 是 2。preference accuracy `0.6016949152542372`（118 对里 71 正，tie 0）。raw policy accuracy `0.01694915254237288`（2/118）。`publishable` 仍是 false。

四个退化场景相对父版本的编辑，每场景 7324 个参考字：noise +2、reverb −4、far_field −5、dropout +2。三个受训场景合计 −7。macro delta 是 `-0.0001706717640633515`。clean CER `0.02184598580010923`，相对父版本持平，相对 base −5。`DPO_LOCATE` 是 `QUIET`，gap 插入 0，dropout 长度比 `0.8551338066630256`，dropout delta `0.0002730748225013846`。

和 step 10 比，macro 的改善从 −2.5 个编辑收到 −1.25 个，dropout 从 −2 变成 +2。门禁仍通过。step 30 的优化器步已写出：`lr=4.4736e-06`，loss `0.688653`，`anchor_loss=0.011583`。

## 2026-10-20 dropout 父版本锚 step 30 通过

`20261020T041218_dpo_06b_zh_seed42` 的 step 30 决策是 `passed=true`，`risk_stop=false`，`risks` 为空。`improved_degraded_scenarios` 是 1。preference accuracy `0.6694915254237288`（118 对里 79 正，tie 0）。raw policy accuracy `0.01694915254237288`（2/118）。`publishable` 仍是 false。

四个退化场景相对父版本的编辑，每场景 7324 个参考字：noise 0、reverb −4、far_field 0、dropout +1。三个受训场景合计 −4。macro delta 是 `-0.00010240305843800535`。clean CER `0.02239213544511196`，相对父版本 +4，相对 base −1。`DPO_LOCATE` 是 `QUIET`，gap 插入 0，dropout 长度比 `0.8551338066630256`，dropout delta `0.0001365374112506923`。

macro 的改善继续收窄：step 10 是 −2.5 个编辑，step 20 是 −1.25 个，step 30 是 −0.75 个。净编辑仍是负的。step 40 的优化器步已写出：`lr=3.875e-06`，loss `0.694773`，`anchor_loss=0.001709`。

## 2026-10-20 dropout 父版本锚 step 40 通过

`20261020T041218_dpo_06b_zh_seed42` 的 step 40 决策是 `passed=true`，`risk_stop=false`，`risks` 为空。`improved_degraded_scenarios` 是 3。preference accuracy `0.6440677966101694`（118 对里 76 正，tie 0）。raw policy accuracy `0.025423728813559324`（3/118）。`publishable` 仍是 false。

四个退化场景相对父版本的编辑，每场景 7324 个参考字：noise −6、reverb −4、far_field −2、dropout +1。三个受训场景合计 −12。macro delta 是 `-0.00037547788093937606`。clean CER `0.022255598033861276`，相对父版本 +3，相对 base −2。`DPO_LOCATE` 是 `QUIET`，gap 插入 0，dropout 长度比 `0.8551338066630256`，dropout delta `0.0001365374112506923`。

这一步的 macro 改善是 −2.75 个编辑，比 step 10 的 −2.5 和 step 30 的 −0.75 都大。step 41 的优化器步已写出：`lr=3.806311e-06`，loss `0.681038`，`anchor_loss=0.003779`。164 条对、累积 4、`max_epochs: 1` 时，step 41 是这一轮的最后一次更新。

## 2026-10-20 dropout 父版本锚在 step 41 结束

`20261020T041218_dpo_06b_zh_seed42` 在 `2026-10-20T07:30:05Z` 结束。`DPO_EXIT 0`，`DPO_DONE global_step=41`。从 `2026-10-20T04:12:40Z` 的 `LAUNCH_DETACHED` 算起，约 3 小时 17 分。没有更多更新。

step 41 决策是 `passed=false`，`risk_stop=true`。唯一风险是 degraded macro CER 上升 `3.4134352812673074e-05`。preference accuracy `0.6779661016949152`（80/118，tie 0）。raw policy 仍是 3/118。四个退化场景相对父版本的编辑：noise −4、reverb −5、far_field −3、dropout +13。三个受训场景合计 −12，dropout 多出的 13 个把净编辑变成 +1。clean CER `0.022665210267613325`，相对父版本 +6，相对 base +1，两条 clean 上限都没破。`DPO_LOCATE` 是 `QUIET`，gap 插入 10。

`status.json` 是 `completed` / `gate_risk`，`global_step` 41。`gate.json` 的 `status` 是 `passed`，`selected_step` 是 40。选择规则是已通过的 checkpoint 里 degraded macro 最低，其次 clean，其次 step。step 40 的 macro delta 是 `-0.00037547788093937606`。`publishable` 保持 false。不恢复这一轮。不发布。Phase 2 和 Phase 3 不标完成。test 不评。RL 不开始。可用模型仍是 SFT step 30。`latest.json` 指向 step 41，那是最后保存的进度，不是选出的模型。

停住的读数是 2500 条 val 的贪心 CER，不是训练 loss。保活日志里抽到的 DPO loss 是 step 1 `0.693147`、step 10 `0.696090`、step 20 `0.687737`、step 30 `0.688653`、step 40 `0.694773`、step 41 `0.681038`。step 41 的 preference 是这一轮最高的 `0.6779661016949152`。`anchor_loss` 每个 optimizer step 换一条 dropout，数值在 `0.000370` 到 `0.011583` 之间跳动，step 41 是 `0.003779`。这一轮没有 RL。从 step 40 到 step 41，三个受训场景的编辑和保持 −12，dropout 从 +1 变成 +13，其中 gap 插入从 0 变成 10，四场景净编辑从 −11 变成 +1。

`anchor_loss` 的跳动和 step 41 的 dropout 跳变对不上。它等于 `0.05 * 该条父版本文本的 token 平均 NLL`，step 1 用第 1 条 train dropout，step 41 用第 41 条。保活只留下当时最新的一步，能看到的是 step 1、4、10、20、30、40、41。对应的 token 平均 NLL 是 `0.0223`、`0.0421`、`0.0074`、`0.1427`、`0.2317`、`0.0342`、`0.0756`。最大的一档在 step 30，随后的门禁里 dropout 仍是 +1。step 41 的锚损失处在中间，梯度范数是抽到的最大一档 `3.109375`。反传系数是 `0.05 / token 数`，抽到的步上大约是 `0.0036` 到 `0.010`，同一条 DPO 更新里单个差异字的梯度上限是 `beta = 0.1`。完整 41 步的序列在 Drive 的 `metrics.jsonl`，这次会话停掉后没有再读。

2026-10-20 用只读 CPU 会话核对了 Drive 上的 `eval/step_000040/scored.jsonl`、`eval/step_000041/scored.jsonl` 和 41 行 `metrics.jsonl`。2500 条里有 37 条归一化文本变了：clean 4、dropout 16、far_field 2、noise 11、reverb 4。编辑差是 clean +3、dropout +12、far_field −1、noise +2、reverb −1。dropout 的 +12 里面，`BAC009S0748W0156__dropout` 单条从 2 个编辑变成 16 个。step 40 的原文是「就有一零个，加级贫困县」，和父版本相同；step 41 的原文是「就有一零个Human Rights Watch」。静音带 `[0.40, 0.55]` 上的 10 次插入都在这一条，插的是 `humanright`，后面的 `国家级贫困县` 被换成 `swatch`。其余 dropout 净少 2 个编辑。去掉这一条，step 41 的 dropout 相对父版本是 −1，四场景净编辑是 −13，macro 仍下降。`DPO_LOCATE` 仍写成 `QUIET`，因为 10 次插入低于 30 的门槛，dropout delta `0.001775` 也低于 `0.002`。

这条在 `20261019T012838` 的 step 10 已经翻成过同一个英文串。本轮 step 40 还保持着父版本文本。夹在两次评测之间的更新是 epoch 尾部的 4 条单字对，加上文件序第 41 条 dropout 锚。4 条是 `BAC009S0065W0166__noise`（再/在，margin `-2.12458`）、`BAC009S0185W0478__far_field`（鼓/股，margin `-0.757392`）、`BAC009S0143W0444__noise`（信/庆，margin `-2.46036`）、`BAC009S0016W0478__reverb`（间/家，margin `-3.25094`）。锚是 `BAC009S0005W0273__dropout`，教师文本「在上有多处展馆」，6 个 token。这些 source 都不在发生变化的 val 样本里，文本里也没有 `Human Rights Watch`。

41 步的 DPO loss 里，step 41 的 `0.681038` 最低。`anchor_loss` 最大是 step 30 的 `0.011583`，step 41 是 `0.003779`，对应 token 平均 NLL `0.075589`。`max_grad_norm` 是 `1.0`，41 步的 `grad_norm` 都大于 1，更新幅度都被裁到同一上限。step 41 裁剪前是 `3.109375`，排第 4；最大是 step 37 的 `5.531250`，而 step 40 的门禁里 dropout 仍是 +1。没有 RL。不恢复 `20261020T041218`，不加大锚系数。`selected_step` 仍是 40。可用模型仍是 SFT step 30。

## 2026-10-20 锚配置改为停在 step 40

背景：step 40 已过冻结门禁。第 41 步是 164 条、累积 4 的 epoch 尾步，把 `BAC009S0748W0156__dropout` 从父版本文本翻成「就有一零个Human Rights Watch」。

范围：新增 `configs/train/dpo_pilot_edit1_anchor_stop40.yaml`。只改 `max_steps: 40`。锚权重 `0.05`、教师 `rejected_text`、`mismatched_only`、warmup 10、学习率 `5e-6`、`eps` `1e-6`、`weight_decay` 0 和门禁都保持与 `dpo_pilot_edit1_anchor.yaml` 相同。不改已执行的 YAML 和冻结哈希。不恢复 `20261020T041218`。这次不新开 A100。

设计：schema 只在这组锚字段同时成立时接受 `max_steps: 40`。其他 pilot 仍必须是 100。语义哈希与旧锚配置不同，不能接上 `20261020T041218`。40 步是评测间隔的倍数，最后一批仍是 4 条。

测试：`tests.test_dpo_contract` 与目录 README 合同。新配置能加载；aligned 或缺少锚字段时写成 40 会拒绝；164 条样本在 `max_steps: 40` 时停在 step 40。2026-10-20 本机 37 项通过，退出码 0。

验收：测试通过，并且没有新的 run。这一档的结果仍是 `20261020T041218` 的 step 40。`publishable` 保持 false。

影响：已保存的 checkpoint 不变。以后用这份配置开跑时不再产生 step 41。可用模型仍是 SFT step 30。

## 2026-10-20 停步配置启动

2026-10-20 用户说「开始吧」。run `20261020T114647_dpo_06b_zh_seed42` 在标准 A100 会话 `lora-asr-dpo-stop40` 上启动。运行时 `gpu-a100-s-kkb-usc1b0-2mjb1fkgnpn4i`，Shape Standard。没有 `--resume`，也没有用 `colab upload` 传音频包。父版本仍是 `sft_06b_zh_20261015T234939_step000030`。

`2026-10-20T11:46:56Z` `LAUNCH_DETACHED`。`CONFIG_STOP40 max_steps=40`，`DISK_FREE_GB 193.31`，`REBUILD_EXIT 0`，`TORCH 2.11.0+cu130 True`。trainer PID 2315。`2026-10-20T11:47:29Z` step 1：`global_step=1/40`，`loss=0.693147`，`grad_norm=2.781250`，`lr=5e-07`，`pairs=4`，`anchor_loss=0.001113`，`anchor_tokens=7`。`DPO_TRAIN_SCENARIOS` 是 noise、reverb、far_field，`train_pairs=164`，`val_pairs=118`。`DPO_ANCHOR rows=3816 weight=0.05 target=rejected_text`。`DPO_PREFLIGHT_OK`，`DPO_RUNTIME_BUDGET disabled`。

不恢复 `20261020T041218`。评测在 step 10、20、30、40。step 40 是这一轮的最后一步。门禁还没出。`publishable` 保持 false。可用模型仍是 SFT step 30。

## 2026-10-20 停步配置在 step 30 结束

`20261020T114647_dpo_06b_zh_seed42` 在 `2026-10-20T13:45:47Z` 结束。`status.json` 是 `completed` / `gate_risk`，`global_step` 30。`DPO_EXIT 0` 出现在 `2026-10-20T13:48:20Z`。从 `11:46:56Z` 的 `LAUNCH_DETACHED` 算起，约 1 小时 59 分。没有跑到 step 40。

step 10 与 `20261020T041218` 的 step 10 相同：`passed=true`，preference `0.576271186440678`（68/118），noise −9、reverb −1、far_field +2、dropout −2，macro delta `-0.000341343528126703`。clean 相对父版本 +2，相对 base −3。`DPO_LOCATE` 是 `QUIET`，gap 插入 0。学习率仍是 `5e-6`，因为前 10 步只看 warmup。

step 20 也是 `passed=true`。preference `0.6271186440677966`（74/118），noise −2、reverb −1、far_field −3、dropout −1，四个退化场景都有改善，合计 −7 个编辑，macro delta `-0.00023894046968868377`。clean 相对父版本持平，相对 base −5。gap 插入 0。这一步的学习率是 `3.875e-6`。`max_steps: 40` 同时是余弦日程的终点，所以 step 20 已经落到上一轮 step 40 的学习率。

step 30 是 `passed=false`，`risk_stop=true`。唯一风险是 degraded macro CER 上升 `3.4134352812673074e-05`。noise +7、reverb −3、far_field −3、dropout 0，净编辑 +1。preference `0.6694915254237288`（79/118）。clean 相对父版本仍是 0。`DPO_LOCATE` 是 `QUIET`，gap 插入 0，dropout delta 是 0。三份 `scored.jsonl` 里都没有 `Human Rights Watch`。这一步的学习率是 `1.625e-6`，loss `0.686811`，`anchor_loss` `0.012737`。raw policy 三步都是 1/118。

`gate.json` 的 `status` 是 `passed`，`selected_step` 是 10。选择规则在已通过的 checkpoint 里取 degraded macro 最低的一步；step 10 的 macro 低于 step 20。`latest.json` 指向 step 30，那是最后保存的进度。`publishable` 保持 false。不恢复 `20261020T114647`，也不恢复 `20261020T041218`。不加大锚系数。Phase 2 和 Phase 3 不标完成。可用模型仍是 SFT step 30。Drive 上有 step 10、20、30 的 checkpoint、`scored.jsonl` 和 `eval_decisions.jsonl`。标准 A100 会话在读完这些文件后停止。

## 2026-10-20 余弦日程视界解耦（方案 1）与每步多样本 dropout 锚平均（方案 3）实现

- **背景**：`20261020T114647_dpo_06b_zh_seed42`（`dpo_pilot_edit1_anchor_stop40.yaml`）将 `train.max_steps` 设为 `40` 时，`HorizonScheduler.total` 同步缩到 `40`，余弦衰减窗口从 `90` 步（`100 - 10`）压缩到 `30` 步（`40 - 10`），使 Step 20 学习率从 `4.864e-6` 掉到 `3.875e-6`、Step 30 学习率从 `4.474e-6` 骤降 67.5% 到 `1.625e-6`，并在 Step 30 触发 `noise +7` 编辑回退（净编辑 `+1`，`macro_delta = +3.413e-5`）。同时，单步单条 dropout 锚的 token 平均 NLL 在 `0.0055 ~ 0.2317` 间跳变 40 倍。
- **范围**：在 `train/sft_schema.py`、`train/train_sft.py`、`train/dpo_schema.py`、`train/dpo_contract.py`、`train/train_dpo.py` 中实现可选 `optim.scheduler_max_steps`（解耦余弦日程总步数与 `train.max_steps`）和 `dpo.dropout_anchor_samples_per_step`（每步平均 `k` 条 dropout 锚样本，`k ∈ {1..4}`，总锚权重仍为 `0.05`）。新增两份对比配置：
  1. `configs/train/dpo_pilot_edit1_anchor_step40_h100.yaml`（纯方案 1：`train.max_steps: 40`，`optim.scheduler_max_steps: 100`，`dpo.dropout_anchor_samples_per_step: 1`，100% 确定性复现 `20261020T041218` 的 Step 10~40 并在 Step 40 干净停步）。
  2. `configs/train/dpo_pilot_edit1_anchor_step40_h100_a4.yaml`（方案 1 + 方案 3：额外启用 `dpo.dropout_anchor_samples_per_step: 4`，每步切片 4 条非空 `rejected_text` 的 train dropout 样本并按 `0.05 / 4 = 0.0125` 平均反传）。
  已执行的 YAML 配置与冻结哈希保持不变；本次按用户确认暂不启动 A100 会话。
- **测试与验收**：本机运行 `.venv/bin/python -m unittest discover -s tests -v`，全量 **112 项单元测试全部通过**（含新增 `test_step40_horizon100_and_multi_anchor_configs_decouple_cosine_and_average_anchors` 及目录 README 合同测试），退出码 0。

## 2026-10-22 纯方案 1（100 步余弦周期 + Step 40 停步）DPO 已启动

2026-10-22 用户确认先启动纯方案 1 配置 `configs/train/dpo_pilot_edit1_anchor_step40_h100.yaml`。标准 A100 会话 `lora-asr-dpo-h100`（运行时 `gpu-a100-s-kkb-ass1c0-26sxp0qcjfkek`，Shape Standard，无 `--high-mem`）。Drive `release.json` 为 1416 字节。没有 `--resume`，也没有用 `colab upload` 传音频包。父版本仍是已发布的 SFT step 30 release `sft_06b_zh_20261015T234939_step000030`。

`2026-10-22T13:30:10Z` `LAUNCH_DETACHED`。`DISK_FREE_GB 187.04`，`CONFIG_STEP40_H100 max_steps=40 scheduler_max_steps=100`，`REBUILD_EXIT 0`，`TORCH 2.11.0+cu130 True`。run `20261022T133007_dpo_06b_zh_seed42`，trainer PID 9716。

`2026-10-22T13:31:08Z` `STEP1_OK`，`dpo-h100-keepalive.service` 已接管保活（日志 `/root/colab-keepalive/dpo_h100_keepalive.log`）：

- `PING alive=1 pid=9716 run_id=20261022T133007_dpo_06b_zh_seed42 preflight=1 budget=1 scenarios=1 anchor=1 config=1 step1=1`
- `DPO_TRAIN_SCENARIOS noise,reverb,far_field val_scenarios=clean,noise,reverb,far_field train_pairs=164 val_pairs=118`
- `DPO_ANCHOR rows=3816 weight=0.05 target=rejected_text`
- step 1：`global_step=1/40 loss=0.693147 grad_norm=2.781250 lr=5e-07 pairs=4 anchor_loss=0.001113 anchor_tokens=7`
- step 4：`global_step=4/40 loss=0.694414 grad_norm=2.265625 lr=2e-06 pairs=4 anchor_loss=0.002105 anchor_tokens=9`
- step 5：`global_step=5/40 loss=0.696213 grad_norm=2.421875 lr=2.5e-06 pairs=4 anchor_loss=0.002353 anchor_tokens=5`

step 10 的 2500 条 val 门禁尚未返回。`publishable` 保持 false。不恢复历史 DPO run。可用模型仍是 SFT step 30。

## 2026-10-22 纯方案 1 DPO step 10 通过并进入 step 20 评测

`2026-10-22T14:11:08Z` 保活记到 `20261022T133007_dpo_06b_zh_seed42` 的 step 10 门禁通过：`DPO_GATE step=10 passed=True risk_stop=False preference=0.576271`（68/118），与 `20261020T041218` 的 step 10 完全一致（step 10 优化器步为 `loss=0.696090 grad_norm=2.906250 lr=5e-06 pairs=4 anchor_loss=0.000370 anchor_tokens=6`）。

随后训练推进到 step 20：`global_step=20/40 loss=0.687737 grad_norm=2.515625 lr=4.8643084e-06 pairs=4 anchor_loss=0.007134 anchor_tokens=14`。学习率 `4.8643084e-06`、DPO loss `0.687737` 与 `anchor_loss=0.007134` 均与 `20261020T041218`（100 步余弦周期）逐位一致，验证 `optim.scheduler_max_steps: 100` 解耦生效。当前正在进行 step 20 的 2500 条 val 门禁评测。`publishable` 保持 false。

## 2026-10-22 纯方案 1 DPO step 20 通过并进入 step 30 评测

`2026-10-22T14:51:04Z` 保活记到 `20261022T133007_dpo_06b_zh_seed42` 的 step 20 门禁通过：`DPO_GATE step=20 passed=True risk_stop=False preference=0.601695`（71/118），与 `20261020T041218` 的 step 20 完全一致。

随后训练推进到 step 30：`global_step=30/40 loss=0.688653 grad_norm=2.375000 lr=4.4736e-06 pairs=4 anchor_loss=0.011583 anchor_tokens=5`。学习率 `4.4736e-06`、DPO loss `0.688653` 与 `anchor_loss=0.011583` 均与 `20261020T041218` 的 step 30 逐位一致。当前正在进行 step 30 的 2500 条 val 门禁评测。`publishable` 保持 false。

## 2026-10-22 纯方案 1 DPO step 30 通过并进入最终 step 40 评测

`2026-10-22T15:37:52Z` 保活记到 `20261022T133007_dpo_06b_zh_seed42` 的 step 30 门禁通过：`DPO_GATE step=30 passed=True risk_stop=False preference=0.669492`（79/118），与 `20261020T041218` 的 step 30 完全一致（解决了此前 `20261020T114647` 因 40 步余弦周期压缩导致 step 30 失败的问题）。

随后训练推进到本轮最后一步 step 40：`global_step=40/40 loss=0.694773 grad_norm=2.484375 lr=3.875e-06 pairs=4 anchor_loss=0.001709 anchor_tokens=10`。学习率 `3.875e-06`、DPO loss `0.694773` 与 `anchor_loss=0.001709` 均与 `20261020T041218` 的 step 40 逐位一致。当前正在进行最终 step 40 的 2500 条 val 门禁评测（由于 `max_steps: 40`，评测完成后将直接结束，不再执行第 41 步尾步）。`publishable` 保持 false。

## 2026-10-22 纯方案 1 DPO 在 step 40 无风险完赛

`20261022T133007_dpo_06b_zh_seed42` 在 `2026-10-22T16:08:40Z` 完成全部 40 步与 4 次 2500 条 val 门禁评测，保活在 `2026-10-22T16:11:21Z` 记到 `DPO_GATE step=40 passed=True risk_stop=False preference=0.644068`、`DPO_DONE global_step=40`、`DPO_EXIT 0`、`KEEPALIVE_DONE`。从 `13:30:10Z` 的 `LAUNCH_DETACHED` 算起，总耗时约 2 小时 38 分。

- `status.json`：`{"status": "completed", "reason": "max_steps", "time": "2026-10-22T16:08:40Z", "global_step": 40}`（SHA-256 `dce37a6861ca184fa6f05a2e962961943cc7faed5db13994f2141ca472a6f2ad`），全程无 `gate_risk`。
- `gate.json`：`status` 为 `passed`，`selected_step` 为 `40`，`publishable` 为 `false`（SHA-256 `74c0488d643d6c93c2448164879c4d57729fa04a580fc953b8a5d91813082b00`）。
- `latest.json`：指向 `step_000040`，`checkpoint_sha256` 为 `9a52ddbac7abfe07f467dcdc77a2c9407fa566176de26fe829c457b6e7395f8c`（SHA-256 `c04594ad2408c645f0c142f0becdf5015d36b0b3769f9335604fe9ea248892dd`），`latest_step` 与 `selected_step` 一致。
- 四次评测（step 10、20、30、40）全部 `passed=true`、`risk_stop=false`、`risks=[]`、`locate_signature=QUIET`、`locate_gap_insertions=0`，各项指标与 `20261020T041218` 对应步 100% 逐位一致：
  - step 10：`degraded_macro_cer=0.07318405243036592`（`macro_delta=-0.000341343528126703`：noise −9、reverb −1、far_field +2、dropout −2），`clean_cer=0.022119060622610594`（相对父版本 +2、相对 base −3），`preference_accuracy=0.576271186440678`（68/118）。
  - step 20：`degraded_macro_cer=0.07335472419442927`（`macro_delta=-0.0001706717640633515`：noise +2、reverb −4、far_field −5、dropout +2），`clean_cer=0.02184598580010923`（相对父版本 0、相对 base −5），`preference_accuracy=0.6016949152542372`（71/118）。
  - step 30：`degraded_macro_cer=0.07342299290005462`（`macro_delta=-0.00010240305843800535`：noise 0、reverb −4、far_field 0、dropout +1），`clean_cer=0.02239213544511196`（相对父版本 +4、相对 base −1），`preference_accuracy=0.6694915254237288`（79/118）。
  - step 40（选出模型）：`degraded_macro_cer=0.07314991807755325`（`macro_delta=-0.00037547788093937606`，净减 11 个编辑：noise −6、reverb −4、far_field −2、dropout +1），`clean_cer=0.022255598033861276`（相对父版本 +3、相对 base −2），`preference_accuracy=0.6440677966101694`（76/118），`raw_policy_accuracy=0.025423728813559324`（3/118）。
- Drive `runs/20261022T133007_dpo_06b_zh_seed42/` 上的 `status.json`、`gate.json`、`latest.json`、`run_report.md`（SHA-256 `193a9e5f1cf8174e60da9e1ec32b66975b2330950c56daf515da35aade7d4a61`）和 `eval_decisions.jsonl`（SHA-256 `ba88d71b183eacf8dedf341d73b7c22e4594741088b8c230b5236cd2aa9d0744`）与 VM 本地文件哈希完全一致。

接着复用当前已挂载 Drive 的标准 A100 会话 `lora-asr-dpo-h100`，以独立新 `run_id` 启动方案 1 + 方案 3 对比配置 `configs/train/dpo_pilot_edit1_anchor_step40_h100_a4.yaml`（`dropout_anchor_samples_per_step: 4`）。

## 2026-10-22 方案 1 + 方案 3（4 样本 dropout 锚等权平均）DPO 已启动

`2026-10-22T16:19:28Z` 在同一标准 A100 会话 `lora-asr-dpo-h100`（运行时 `gpu-a100-s-kkb-ass1c0-26sxp0qcjfkek`，Shape Standard）脱离启动 `configs/train/dpo_pilot_edit1_anchor_step40_h100_a4.yaml`，没有 `--resume`。`DISK_FREE_GB 184.45`，`CONFIG_STEP40_H100_A4 max_steps=40 scheduler_max_steps=100 samples_per_step=4`，`REBUILD_EXIT 0`，`TORCH 2.11.0+cu130 True`。新 run 为 `20261022T161911_dpo_06b_zh_seed42`，trainer PID 51447。

`2026-10-22T16:20:04Z` `STEP1_OK`，`dpo-h100-a4-keepalive.service` 已接管保活（日志 `/root/colab-keepalive/dpo_h100_a4_keepalive.log`）：

- `PING alive=1 pid=51447 run_id=20261022T161911_dpo_06b_zh_seed42 preflight=1 budget=1 scenarios=1 anchor=1 config=1 step1=1`
- `DPO_TRAIN_SCENARIOS noise,reverb,far_field val_scenarios=clean,noise,reverb,far_field train_pairs=164 val_pairs=118`
- `DPO_ANCHOR rows=3816 weight=0.05 target=rejected_text samples_per_step=4`
- step 1：`global_step=1/40 loss=0.693147 grad_norm=2.781250 lr=5e-07 pairs=4 anchor_loss=0.004074 anchor_tokens=30`
- step 4：`global_step=4/40 loss=0.690930 grad_norm=2.265625 lr=2e-06 pairs=4 anchor_loss=0.002730 anchor_tokens=36`

step 10 的 2500 条 val 门禁尚未返回。`publishable` 保持 false。不恢复历史 DPO run。

## 2026-10-22 方案 1 + 方案 3（4 样本 dropout 锚等权平均）DPO 在 step 10 因 preference=0.5 风险停止

`20261022T161911_dpo_06b_zh_seed42` 在 `2026-10-22T16:59:10Z` 结束 step 10 的 2500 条 val 门禁评测，保活在 `2026-10-22T17:00:16Z` 记到 `DPO_LOCATE step=10 signature=QUIET gap_insertions=0 dropout_len_ratio=0.854861 dropout_delta=0.000000`、`DPO_GATE step=10 passed=False risk_stop=True preference=0.500000`、`DPO_DONE global_step=10`、`DPO_EXIT 0`、`KEEPALIVE_DONE`。

- step 10 优化器步：`global_step=10/40 loss=0.691479 grad_norm=2.890625 lr=5e-06 pairs=4 anchor_loss=0.003964 anchor_tokens=48`。
- `status.json`：`{"status": "gate_failed", "reason": "gate_risk", "time": "2026-10-22T16:59:10Z", "global_step": 10}`（SHA-256 `595eb09c3670f7f29aedad92db1c277f4b684945e9b0c02d083c31b45b057f73`）。
- `gate.json`：`status` 为 `failed`，`selected_step` 为 `null`，`publishable` 为 `false`，`risks` 为 `["preference accuracy 0.5 is below the DPO gate"]`（SHA-256 `eb32f980c62879dab76562183b3cb3e178a85274a2816880e9204dad99a8e627`）。
- `latest.json`：指向 `step_000010`，`checkpoint_sha256` 为 `e95835298e4a65086f333b2d27177fc005934f3c9e215b8b4205507b2a374b14`（SHA-256 `74dc344dc6ca1ba183e12092d699349d4d50f724b65ea4803bc6e667dffbe875`）。
- step 10 详细指标：
  - 声学 CER 门禁项全部满足：`degraded_macro_cer=0.0733205898416166`（`macro_delta=-0.00020480611687602457`，四退化场景净减 6 个编辑：noise `-0.00013653741125067842` [−1]、reverb `-0.0002730748225013638` [−2]、far_field `-0.00040961223375204914` [−3]、dropout `0.0` [0]，`improved_degraded_scenarios=3`），`clean_cer=0.022119060622610594`（相对父版本 `+0.0002730748225013638` [+2]、相对 base `-0.00040961223375204914` [−3]），`locate_signature=QUIET`，`locate_gap_insertions=0`，`locate_dropout_length_ratio=0.8548607318405244`。
  - 唯一未过项：`preference_accuracy=0.5`（59/118，`ties=0`），`raw_policy_accuracy=0.00847457627118644`（1/118）。前 10 步每步等权平均 4 条 `dropout` 锚（累计 40 条锚样本对父版本 `rejected_text` 施加全句 NLL 锚定）虽然把 `dropout` 编辑波动压成 `0.0`，但在 10 步 warmup 期内对解码器参数靠近父版本的约束更均匀，使 118 条验证偏好对在 step 10 比纯方案 1（68/118，`0.576271`）少翻转了 9 对，恰好落在 `0.5`。
- Drive `runs/20261022T161911_dpo_06b_zh_seed42/` 上的 `status.json`、`gate.json`、`latest.json`、`run_report.md`（SHA-256 `76046faa0a6faba01c446eba4a4bd69271afefc06ac03e2bceef980bdf9937e1`）和 `eval_decisions.jsonl`（SHA-256 `517279ef5e9c91e91c370c9d7cf3d6f6cc629167a1aa786bc4dceb72b68fd7a5`）与 VM 本地文件哈希完全一致。
- 对比结论：**纯方案 1（`configs/train/dpo_pilot_edit1_anchor_step40_h100.yaml`，run `20261022T133007_dpo_06b_zh_seed42`）在 step 10、20、30、40 四次门禁全部通过，以 `status=completed` / `reason=max_steps`（零 `gate_risk`）选出 `step_000040`（`latest_step == selected_step == 40`，四退化场景净减 11 个编辑，preference `0.644068`），优于方案 1+3**。
- 标准 A100 会话 `lora-asr-dpo-h100` 已停止，`colab sessions` 确认 `No active sessions found on server.`。`publishable` 保持 false。不恢复 `20261022T161911_dpo_06b_zh_seed42`。可用已发布模型仍是 SFT step 30。

## 2026-10-23 DPO Release（`dpo_06b_zh_20261022T133007_step000040`）打包、独立进程回读与 Phase 4 RL 启动

按确认结论采用纯方案 1（`20261022T133007_dpo_06b_zh_seed42`，`selected_step=40`）作为 Phase 3 DPO 胜出模型，并在标准 A100 会话 `lora-asr-dpo-rl`（运行时 `gpu-a100-s-ass1c0-2anaiy6uyjmsc`，Shape Standard）启动 DPO Release 发布与 Phase 4 RL 串联流水线：

- 本地实现与单测驗證：`train/dpo_release.py`、`train/rl_contract.py`、`train/rl_schema.py`、`train/train_rl.py`、`configs/train/aishell_val_dpo_step40_20261022.json`、`configs/train/rl_smoke.yaml`、`configs/train/rl_pilot.yaml` 及 `tests/test_dpo_release.py`、`tests/test_rl_contract.py` 共 124 项单元测试全部通过（退出码 0）。
- `train/dpo_release.py prepare` 已在 `2026-10-23T01:07:09Z` 完成 `dpo_06b_zh_20261022T133007_step000040` 的 staging 并同步到 Drive `/content/drive/MyDrive/qwen3-asr/releases/dpo/dpo_06b_zh_20261022T133007_step000040/`：
  - `source_run_id`: `20261022T133007_dpo_06b_zh_seed42`
  - `source_step`: `40`（`latest_step=40`，`latest_is_release=true`）
  - `source_checkpoint_sha256`: `9a52ddbac7abfe07f467dcdc77a2c9407fa566176de26fe829c457b6e7395f8c`
  - `adapter_sha256`: `1d28d3119385c178c4b1b9a1e52a68d134f33a9bfd3de36659290a184ba5c51a`
  - `payload_sha256`: `723c986034ffddb59da67018cd5842183b70ec58022f03e97dcc73360fd92fcd`
  - `audit_disposition`: `unlistened`
- 本地 `configs/train/rl_smoke.yaml` 与 `configs/train/rl_pilot.yaml` 已将 `model.parent_release_sha256` 固定为 `1d28d3119385c178c4b1b9a1e52a68d134f33a9bfd3de36659290a184ba5c51a`。
- `train/dpo_release.py readback` 已在 `2026-10-23T01:44:54Z` 完成 2500 条固定 `val` 的独立进程 greedy 解码与逐条核对并同步回 Drive `/content/drive/MyDrive/qwen3-asr/releases/dpo/dpo_06b_zh_20261022T133007_step000040/`：
  - `DPO_RELEASE_READBACK matched=True compared=2500 mismatches=0 sha256=34d5843565faf6d06b971086b03110ddc6402ca93c5fcd319ca6cca79b9c647c`
  - `release.json` 已翻转为 `status: "published"`、`publishable: true`（`readback_sha256: "34d5843565faf6d06b971086b03110ddc6402ca93c5fcd319ca6cca79b9c647c"`，`audit_disposition: "unlistened"`）。
- `RL_SMOKE_START` 首次探针（`20261023T014455_rl_06b_zh_seed42`）在 `preflight` 的 `probe_loss.backward()` 捕获到 PyTorch autograd 版本校验错误（`[CUDABFloat16Type [16, 1024]], which is output 0 of AsStridedBackward0, is at version 4; expected version 2 instead`）：原因是 `score_candidate_tokens` 先执行了带梯度的 policy 前向（保存 `param.t()` 视图版本 2），随后在 `.backward()` 之前调用 `use_reference_adapter` 对同一 LoRA 参数执行两次原位 `param.copy_()`（版本升至 4）。
- 修复方案：在 `train/train_rl.py` 的 `score_candidate_tokens` 中先于 `no_grad` 下完成 `use_reference_adapter` 的 reference 前向并恢复 policy 参数，再执行带梯度的 policy 前向；同时在训练循环中对每个候选的 `pol_t` 立即计算 `cand_loss_tensor / (len(candidates) * accum_target)` 并调用 `.backward()` 释放计算图，随后再进入下一候选的 reference 切换。本机 125 项单元测试全部通过。
- 6-step RL Smoke（run `20261023T020031_rl_06b_zh_seed42`，配置 `configs/train/rl_smoke.yaml` SHA-256 `9aca5f54b1825864b25032cab95e2bbd087c032a8dbea9059c0b5f41b7bab577`）已在 `lora-asr-dpo-rl` 上完赛：
  - Part 1（steps 1..5）：通过 `RL_PREFLIGHT_OK` 与 `RL_STEP0`，在 `2026-10-23T02:05:58Z` 按 `interrupt_after_step: 5` 干净中断并写出 `status=paused` / `global_step=5`（`RL_SMOKE_PART1_PAUSED_AT_5`）。
  - Part 2（`--resume` step 6）：从 `step_000005` 恢复 adapter、optimizer、scheduler 和 RNG 状态，在 `2026-10-23T02:07:21Z` 完成第 6 步（`RL_STEP global_step=6/6 loss=-1.295582 grad_norm=9.750000 lr=1e-06 visits=22`，验证 2 步 warmup 后学习率保持 `1e-06` 未重置），并在 `2026-10-23T02:08:35Z` 通过 step 6 评测与 `gate.json` 校验（`RL_SMOKE_OK run_id=20261023T020031_rl_06b_zh_seed42 global_step=6`）。
- 50-step RL Pilot（run `20261023T020835_rl_06b_zh_seed42`，配置 `configs/train/rl_pilot.yaml` SHA-256 `bc35cce3c6511bfe9d2a35e95818b8ac1c923aec7fcc9b8323d667682879de1e`）已在 `2026-10-23T02:08:35Z` 脱离启动（trainer PID `46451`）：
  - `2026-10-23T02:49:12Z` 完成 step 0 的 2500 条固定 `val` 评测与前缀固化：`RL_STEP0 micro_cer=0.06297105 mean_reward=0.93573467 prefix_sha256=d055f08a1b6824ce5dd986c2af754cd87dbadf7e966d6af154e254af99f3485a raw_kl=0.0`（`micro_cer` 与父版本 DPO step 40 的 `0.06297105406881486` 一致）。
  - 同刻记到第 1 步优化器步：`RL_STEP global_step=1/50 loss=-0.003096 grad_norm=0.189453 lr=5e-06 visits=4`（2 步 warmup 的第 1 步学习率 `5e-06`）。
  - `dpo-rl-stage.service` 已打印 `STAGE_RL_PILOT_RUNNING` 并切换至 `dpo-rl-keepalive.service`（日志 `/root/colab-keepalive/dpo_rl_keepalive.log`）持续保活推进。`publishable` 保持 `false`，`audit_disposition` 保持 `unlistened`，`test` 不评。

## 2026-10-23 初始 RL Pilot（`20261023T020835_rl_06b_zh_seed42`）在 step 5 风险停止与根因分析

`20261023T020835_rl_06b_zh_seed42` 在 `2026-10-23T03:42:38Z`（保活 `03:44:56Z`）完成 step 5 的 2500 条固定 `val` 评测与 `raw_kl` 计算后因 `gate_risk` 停止：`RL_GATE step=5 passed=False risk_stop=True reward=0.934643 delta=-0.001092 raw_kl=0.00041455`、`RL_DONE global_step=5 visits=14`、`RL_EXIT 0`。

- **核心状态与产物哈希（VM 与 Drive 完全一致）**：
  - `status.json`：`{"status": "gate_failed", "reason": "gate_risk", "time": "2026-10-23T03:42:38Z", "global_step": 5}`（SHA-256 `49fcff4928f54ca45d2196f4dd656bb940d1da2bb4d89c2979768b2adf6458be`）
  - `gate.json`：`status="failed"`，`selected_step=null`，`publishable=false`（SHA-256 `fba94257c2ccf914018550ad4ca43cc264390fc1d315fe36b029af35b4a5a257`）
  - `latest.json`：指向 `step_000005`，`checkpoint_sha256="d5b28ec344b2ed850ed9be2ad68a5aa23de8602c328d6141e9293f2aed8e5c1f"`（SHA-256 `34a6949b7a98353682d80aaa53c748ed676ce873a849908bad663a47b3399f5c`）
  - `eval_decisions.jsonl`（SHA-256 `ba6ee981aa7c4a7304787d6c1bff6b2499c11506eb3d92a84b4b399e92fe8182`）：
    - `micro_cer=0.06395412342981976`（2342/36620，相对 step 0 的 2306 净增 +36 编辑）
    - `clean_cer=0.022528672856362644`（165/7324，相对父版本 +2、相对 base 0）
    - `degraded_macro_cer=0.07431048607318405`（`degraded_macro_delta=+0.0011605679956308013`：dropout +25 [1378→1403]、noise +8 [417→425]、far_field +1 [168→169]、reverb 0 [180→180]）
    - `mean_greedy_reward=0.9346425445640163`（`step0_reward=0.9357346716696473`，`reward_delta_vs_step0=-0.0010921271056310022`）
    - `raw_kl=0.0004145508838134531`（满足 `<= 0.001`），`reference_matched=true`
- **逐 rollout 与逐样本根因定位**：
  1. **零信号 group 挤占梯度累积窗口导致单样本 AdamW 步**：14 个访问 group 中有 9 个 `zero_advantage=True`，由于零信号 group 计入了 `gradient_accumulation_steps=2` 窗口，Step 1～5 每一步实际上都只包含 **1 个** 非零信号 group，被 AdamW 归一化为全步长更新。
  2. **60% 的非零更新落在无声学支撑的 `dropout` 静音段**：5 个非零 group 中有 3 个（Step 2 `BAC009S0598W0352__dropout`、Step 4 `BAC009S0336W0407__dropout`、Step 5 `BAC009S0522W0194__dropout`）为 `[0.40, 0.55]` 静音抹零样本，且未开启 `dropout` 父版本锚。
  3. **整句 `pol_t.mean()` 反传与标点不等长污染后缀条件概率**：在含标点或不等长候选上对整句求导压低了静音切口处的逗号 `'，'` 和续写概率，直接导致 `val` 上 `BAC009S0748W0156__dropout` 从 `'就有一零个，加级贫困县'`（2 编辑）再次翻转为 `'就有一零个Human Rights Watch'`（16 编辑，**+14 编辑**），`BAC009S0727W0465__dropout` 从 `'海都记者联系了仙医院办公室负责人'`（2 编辑）在静音切口提前 EOS 为 `'海都记者联系了仙'`（10 编辑，**+8 编辑**），仅此两条就造成 `+22` 个 `dropout` 编辑，掩盖了 `BAC009S0725W0295`（clean/noise/reverb 各 −1）、`BAC009S0743W0430__far_field`（−2）等 11 条样本共 **−13 个编辑**的真实修正。
  4. **后 DPO 学习率过高（`1e-5`）与 `min_reward_improvement` 量纲失配**：DPO step 40 末步学习率已衰减至 `3.875e-6`，而 `rl_pilot.yaml` 在 2 步内升至 `1e-5`（2.6 倍）；同时 2500 条 `val`（36620 字）上 Base 到 DPO step 40 累计减 35 个编辑对应的 `mean_greedy_reward` 总增幅为 `+0.001013`（SFT step 30 到 DPO step 40 减 8 个编辑对应约 `+0.00022`），原 `min_reward_improvement: 0.002` 相当于要求再减 ~73 个编辑。

## 2026-10-23 对齐版 Phase 4 RL（`configs/train/rl_pilot_aligned.yaml`）设计与实现

- **背景与范围**：将 Phase 3 DPO 纯方案 1 验证成功的四项对齐与锚定机制迁移至 Phase 4 RL，在 `train/rl_schema.py`、`train/rl_contract.py`、`train/train_rl.py` 中支持对齐版 RL 配置 `configs/train/rl_pilot_aligned.yaml`（保留 `configs/train/rl_pilot.yaml` 历史配置不变，不恢复 `20261023T020835_rl_06b_zh_seed42`）：
  1. **声学退化场景过滤（`rl.train_scenarios: [noise, reverb, far_field]`）**：RL rollout 仅在具备完整声学波形支撑的 `noise`、`reverb`、`far_field` 三个退化场景上采样与更新，排除 `dropout` 与 `clean`。
  2. **无标点等长替换对齐与差异 token 梯度隔离（`rl.loss_token_scope: mismatched_only`、`rl.max_edit_count: 2`）**：组内以最高 reward 候选 $c_{\text{best}}$ 为锚，仅保留与 $c_{\text{best}}$ 监督 token 等长、无标点且差异 token 数 $\in [1, \text{max\_edit\_count}]$ 的候选参与策略梯度，并在差异 token 掩码位点上反传 `-adv * pol_t[mask].mean()`，共享前后缀与 `<|im_end|>` 的策略梯度严格为 0；全序列仍保留 `kl_beta * k3_t.mean()` 正则；无对齐差异候选的 group 记为 `zero_advantage=True`。
  3. **仅非零信号 group 计入累积窗口（`rl.accumulate_signal_groups_only: true`）**：`zero_advantage=True` 的 group 不推进 `window_groups`，确保每个优化器 step 严格累积 `gradient_accumulation_steps: 4` 个有效信号 group。
  4. **`dropout` 父版本假设锚（`rl.dropout_anchor_weight: 0.05`）**：每个优化器 step 确定性抽取 1 条 train `dropout` 样本，以冻结 reference adapter（DPO step 40）的贪心预测文本为教师计算 token 平均 NLL（权重 `0.05`），锁定静音段行为。
  5. **阻尼优化器与奖励门禁校准**：`learning_rate: 1e-6`、`warmup_steps: 5`、`scheduler_max_steps: 50`、`train.max_steps: 20`（`save_steps: 5`、`eval_steps: 5`）、`kl_beta: 0.10`、`max_group_visits: 1000`、`no_signal_stop_groups: 100`，并将 `gate.min_reward_improvement` 校准为 `0.00002`（要求在 2500 条 `val` 上 `reward_delta_vs_step0 >= +2e-5` 即严格正向提升，同时满足 `degraded_macro_increase_max: 0.0`、`min_improved_degraded_scenarios: 1`、`raw_kl_max: 0.001` 与 `clean` 上限）。

## 2026-10-23 对齐版 RL Pilot（`20261023T042034_rl_06b_zh_seed42`）在 step 5 一次性通过全部门禁并在 step 10 收敛选出 `step_000005`

- **启动与 Step 0 缓存验签**：`2026-10-23T04:20:34Z` 在同一标准 A100 会话 `lora-asr-dpo-rl`（运行时 `gpu-a100-s-ass1c0-2anaiy6uyjmsc`，Shape Standard）脱离启动 `configs/train/rl_pilot_aligned.yaml`（SHA-256 `d36009acec2f2210403f281af9577b5e33b687bd2f6f4bd758bb6c38773c5da8`），运行 ID `20261023T042034_rl_06b_zh_seed42`，trainer PID `79405`。127 项本地与 VM 单元测试全部通过（`UNITTEST_EXIT 0`）。`eval/step_000000` 经 `reference_sha256`（`128b15b0040fe88064a7345678843c2734904a74f356e5a4aada914e559394df`）验签复用：`RL_STEP0 micro_cer=0.06297105 mean_reward=0.93573467 prefix_sha256=d055f08a1b6824ce5dd986c2af754cd87dbadf7e966d6af154e254af99f3485a raw_kl=0.0`。
- **Step 1～10 训练轨迹（`metrics.jsonl`，SHA-256 `342f1ab7cc8c317e3c0714f9f074d75d8959debf4cf1cdf78ee3990659c5ecbd`）**：
  - Step 1：`samples_seen=4, group_visits=35, invalid_groups=0, loss=0.000452, grad_norm=1.953125, lr=2e-07, anchor_loss=0.001137, anchor_tokens=7`
  - Step 2：`samples_seen=8, group_visits=85, invalid_groups=0, loss=-0.006053, grad_norm=1.796875, lr=4e-07, anchor_loss=0.005904, anchor_tokens=10`
  - Step 3：`samples_seen=12, group_visits=135, invalid_groups=0, loss=-0.001998, grad_norm=0.820312, lr=6e-07, anchor_loss=0.003554, anchor_tokens=5`
  - Step 4：`samples_seen=16, group_visits=190, invalid_groups=0, loss=-0.001327, grad_norm=0.824219, lr=8e-07, anchor_loss=0.001991, anchor_tokens=9`
  - Step 5：`samples_seen=20, group_visits=238, invalid_groups=0, loss=-0.002100, grad_norm=0.851562, lr=1e-06, anchor_loss=0.002016, anchor_tokens=5`
  - Step 6：`samples_seen=24, group_visits=273, invalid_groups=0, loss=-0.005312, grad_norm=3.546875, lr=9.989038e-07, anchor_loss=0.001005, anchor_tokens=5`
  - Step 7：`samples_seen=28, group_visits=311, invalid_groups=0, loss=-0.000269, grad_norm=0.898438, lr=9.956206e-07, anchor_loss=0.000866, anchor_tokens=7`
  - Step 8：`samples_seen=32, group_visits=329, invalid_groups=0, loss=-0.001312, grad_norm=0.824219, lr=9.901664e-07, anchor_loss=0.002039, anchor_tokens=15`
  - Step 9：`samples_seen=36, group_visits=366, invalid_groups=0, loss=-0.003917, grad_norm=1.312500, lr=9.825678e-07, anchor_loss=0.005561, anchor_tokens=7`
  - Step 10：`samples_seen=40, group_visits=400, invalid_groups=0, loss=0.000154, grad_norm=0.695312, lr=9.728617e-07, anchor_loss=0.000390, anchor_tokens=6`
- **Step 5 全量 2500 条 `val` 门禁结果（`passed=True`，选出模型 `selected_step=5`）**：
  - `RL_GATE step=5 passed=True risk_stop=False reward=0.935913 delta=+0.000178 raw_kl=0.00033839`
  - 总 `micro_cer`: **`0.06275259421081376`**（`2298 / 36620`，相对 DPO step 40 的 `2306` 净减 **−8 个编辑**，相对 SFT step 30 的 `2317` 净减 **−19 个编辑**，相对 Base 的 `2341` 净减 **−43 个编辑**），总 `macro_cer`: `0.06408716999686175`。
  - 分场景编辑数与 `micro_cer`（对比 DPO step 40）：
    - `clean`: `158 / 7324`（`micro_cer = 0.021572910977607863`，相对 DPO 父版本 **−5 编辑** / `-0.00068269`，相对 Base **−7 编辑** / `-0.00095576`）
    - `far_field`: `165 / 7324`（`micro_cer = 0.022528672856362644`，相对 DPO 父版本 **−3 编辑** / `-0.00040961`）
    - `reverb`: `177 / 7324`（`micro_cer = 0.024167121791370837`，相对 DPO 父版本 **−3 编辑** / `-0.00040961`）
    - `noise`: `419 / 7324`（`micro_cer = 0.05720917531403605`，相对 DPO 父版本 `+2 编辑` / `+0.00027307`）
    - `dropout`: `1379 / 7324`（`micro_cer = 0.18828509011469144`，相对 DPO 父版本 `+1 编辑` / `+0.00013654`）
  - 四退化场景合计编辑数：`2140`（相对 DPO step 40 的 `2143` 净减 **−3 个编辑**），`degraded_macro_cer = 0.07304751501911524`（`degraded_macro_delta = -0.00010240305843800535 <= 0.0`），`improved_degraded_scenarios = 2 >= 1`。
  - `mean_greedy_reward = 0.9359128300031403`（`reward_delta_vs_step0 = +0.00017815833349299481 >= 0.00002`），`raw_kl = 0.00033839310621917683 <= 0.001`，`reference_matched = true`，`empty_rate = repeated_rate = long_rate = truncation_rate = 0.0`，`valid_output_rate = 1.0`。
- **Step 10 评测与正常停机**：
  - `2026-10-23T06:34:44Z` 完成 Step 10 评测：`RL_GATE step=10 passed=False risk_stop=True reward=0.935669 delta=-0.000065 raw_kl=0.00035391`（`edit_count = 2308`：`clean 161`、`dropout 1378`、`far_field 169`、`noise 421`、`reverb 179`，唯一风险是 `degraded_macro_cer` 上升 `+0.00013653741125067842`），触发 `risk_stop=True` 正常停机（`RL_DONE global_step=10 visits=400`、`RL_EXIT 0`）。
- **核心产物与 Drive 哈希读回（100% 一致）**：
  - `status.json`: `{"status": "completed", "reason": "gate_risk", "time": "2026-10-23T06:34:44Z", "global_step": 10}`（SHA-256 `f1f5a5f86ed7462b460d489efbe4ef2b2ef5bf9cdbfac274ed4b7824f027be98`）
  - `gate.json`: `status="passed"`, `selected_step=5`, `publishable=false`（SHA-256 `fb0599292716064456b63f023a6b9b073234c1ea328ac43bbfd7e2590c7f662b`）
  - `latest.json`: `step_name="step_000010"`, `checkpoint_sha256="cf0f3ec1422aa7213c9a81d5f5e9af321f253775c71943e61496f25659940168"`（SHA-256 `e301c04ddcf690b97a4aaaa1b65b55e50db9f8d3e047810a685b02dc778c841e`）
  - `run_report.md`: SHA-256 `58b75664fd851be351a7df91863deeec6af218df1cc23ca60c0abf7b88957ee7`
  - `eval_decisions.jsonl`: SHA-256 `761ebc4e5f4279c2c67585188aad623455afe73cff3375e52c467c0ba46d84b6`
  - `eval/step_000005/metrics.json`: SHA-256 `7b44366ae8f1c782c1f5c6c69f45449647581c3ea8a0b5a19959dc575ac52413`
  - `eval/step_000010/metrics.json`: SHA-256 `4cf069418d935f26aaa739a3f6ac9772a29f826bc7c08c88085d6000f0cba16e`
- **RL Release（`rl_06b_zh_20261023T042034_step000005`）封包与 2500 条 `val` 独立进程读回发布**：
  - `2026-10-23T06:40:26Z` 完成 `train/rl_release.py prepare` staging，并在 `2026-10-23T07:18:54Z`（保活 `07:20:33Z`）完成 2500 条固定 `val` 的独立新进程 greedy `readback` 校验并发布至 Drive `/content/drive/MyDrive/qwen3-asr/releases/rl/rl_06b_zh_20261023T042034_step000005/`：
    - `status`: `"published"`，`publishable`: `true`
    - `source_run_id`: `"20261023T042034_rl_06b_zh_seed42"`，`source_step`: `5`（`latest_step`: `10`，`latest_is_release`: `false`）
    - `source_checkpoint_sha256`: `"6856a87d48667ef3d755ecde5d0e6eab4ca7d0a50188f00ffe7a549f8ed5804e"`
    - `parent_release`: `"dpo_06b_zh_20261022T133007_step000040"`（`parent_release_sha256`: `"1d28d3119385c178c4b1b9a1e52a68d134f33a9bfd3de36659290a184ba5c51a"`）
    - `adapter_sha256`: `"79df11247728f149df4999447bbcd05ae66ae5dd1c5646f48d7a9515a94ff3a2"`
    - `payload_sha256`: `"9f527f1c5307cc5f153eb7357f820aee2057e13c542768eec6c4b0a1f8957eff"`
    - `readback_sha256`: `"e2e28bc14972cddfcd9fcf2efa672a0deb056bb07d2174a9ac1676669ebe716b"`（`matched=true`，`compared=2500`，`mismatch_count=0`）
    - `release.json` SHA-256: `"66d7dc36bd1c5c8f5c4f13ea96c2ecb41b42c3d5c360e0bf2ed1d39e3e86654b"`，`sha256.json` SHA-256: `"fbba242fd5d03751ac2899801f66110af33884aaad925cb9a55f2f0d2b2d1b3b"`（VM 与 Drive 100% 一致）
    - `audit_disposition`: `"unlistened"`（源 run `gate.json` 的 `publishable` 保持 `false`）。
  - 标准 A100 会话 `lora-asr-dpo-rl` 已在 `2026-10-23T07:24:00Z` 正常停止（`colab sessions` 确认 `No active sessions found on server.`）。`test` 仍封存不评。

## 2026-10-26 Phase 5 `test.jsonl` 跨架构（arm64 vs x86_64）`noise` 浮点缩放 1-4 ULP 对齐修复

- **背景**：`2026-10-23T12:34:13Z` 首次启动 `20261023T123000_phase5_test_06b_zh_seed42` 时，在解码开始前（`stage=0`）以 `PHASE5_EXIT 2` 退出。在标准 A100 VM 上复现定位到根因：`ensure_test_audio_verified` 从 `pilot_sample_audio.tar` 解出 500 条 `clean` 测试源音频并在 Linux `x86_64`（AVX-512）上重建 2000 条退化波形后，`clean`、`dropout`、`far_field`、`reverb` 四个场景共 2000 条 WAV 与 macOS `arm64`（NEON）生成的 `manifest/sha256.json` 100% 逐字节一致，但 `noise` 场景有 206/500 条出现 SHA-256 差异（首条为 `wavs/BAC009S0764W0144__noise.wav`）。
- **根因证明**：`add_gaussian_noise` 中 `np.mean(np.square(audio))` 与 `np.mean(np.square(noise))` 对 `float32` 数组求和时，`arm64` NEON 与 `x86_64` AVX-512 的 SIMD 归约顺序存在约 $10^{-8}$ 的尾数舍入差，使 `np.float32(scale)` 在 206 条样本上偏移 `±1 ~ ±4 ULP`（例如 `BAC009S0764W0144` 在 macOS 上为 `0xdf0eb43b`，在 Linux 上为 `0xe00eb43b`，仅差 1 ULP；其中 203 条在 `±2 ULP` 内，2 条为 `-3 ULP`，1 条为 `+4 ULP`），当单个采样点恰处于 `PCM_16` 的 `0.5 LSB` 量化边界时就会翻转 1 LSB。
- **设计与修复**：在 `scripts/run_phase5_test_benchmark.py` 的 `ensure_test_audio_verified` 中增加 `_reconcile_noise_ulp`：当 `noise` 场景 WAV 的 SHA-256 与 `sha256.json` 不一致时，围绕 `np.float32(scale)` 用 `np.nextafter` 双向搜索 `±8 ULP` 候选缩放因子并重新写出 `PCM_16` WAV；只有当某个 ULP 候选的整文件 SHA-256 与 `sha256.json` 完全相等时才予通过，否则仍抛出 `ValueError("test audio sha256 mismatch ...")`。
- **测试与验证**：在 `tests/test_phase5_benchmark.py` 新增 ULP 偏移对齐与防篡改单测；在标准 A100 VM 上实测 2500 条 `test.jsonl` 音频全部 100% 通过 `sha256.json` 校验（`verified_files=2500`，零不匹配）。

## 2026-10-26 Phase 5 封存 `test.jsonl` 四阶段终局评测完赛（`20261026T101800_phase5_test_06b_zh_seed42`）

- **运行环境与保活时间线**：
  - 会话：标准 A100 Colab 会话 `lora-asr-phase5-test`（运行时 `gpu-a100-s-usc1f0-20oh5uujaz2g8`，Shape Standard），由 `sub2` 的 `phase5-keepalive.service` 脱离保活。
  - 启动与完赛时间：`2026-10-26T10:19:40Z` 启动 `scripts/run_phase5_test_benchmark.py`（PID `3937`，`seed=42`），依次完成 `base`（`11:03:38Z` 记到完成）、`sft`（`11:39:57Z` 记到完成）、`dpo`（`12:16:12Z` 记到完成）、`rl`（`12:55:28Z` 记到完成）共 `4 × 2500 = 10000` 条确定性 greedy 解码与 `zh-cer-v1` 评分，在 `2026-10-26T12:55:28Z` 正常退出（`PHASE5_DONE rc=0`、`PHASE5_EXIT 0`）。
  - Drive 同步状态：`PHASE5_BENCHMARK_DONE run_id=20261026T101800_phase5_test_06b_zh_seed42 sync=synced base_edits=2220 sft_edits=2204 dpo_edits=2200 rl_edits=2213`。`sync_phase5_files` 对 `predictions/{base,sft,dpo,rl}.jsonl`、`eval/{base,sft,dpo,rl}/{scored.jsonl,metrics.json,by_scenario.csv}`、`benchmark_summary.json`、`run_report.md`、`status.json` 全部完成 Drive `/content/drive/MyDrive/qwen3-asr/runs/20261026T101800_phase5_test_06b_zh_seed42/` 写入与 SHA-256 逐文件读回校验（`sync=synced`）。随后 Colab 会话已释放（`colab sessions` 为空）。
- **封存 `test.jsonl`（2500 条，总金标字数 `36710`，5 个场景各 `500` 条 / `7342` 字，四退化场景合计 `29368` 字）四阶段指标**：
  - **`base`（`Qwen/Qwen3-ASR-0.6B`，`base_zero_lora`）**：
    - `PHASE5_STAGE_DONE stage=base edits=2220 micro_cer=0.06047399 degraded_macro_cer=0.07106374`
    - 总体编辑数：`2220 / 36710`，`micro_cer = 0.060473985290111685`（`6.0474%`）
    - 四退化场景合计编辑数：`2087 / 29368`，`degraded_macro_cer = 0.07106374284935985`（`7.1064%`）
    - `clean` 编辑数：`133 / 7342`，`clean_cer = 0.01811495505311904`（`1.8115%`）
  - **`sft`（`sft_06b_zh_20261015T234939_step000030`，`adapter_sha256=a185a61f...`）**：
    - `PHASE5_STAGE_DONE stage=sft edits=2204 micro_cer=0.06003814 degraded_macro_cer=0.07048488`
    - 总体编辑数：`2204 / 36710`（**相对 Base 净减 −16 个编辑**），`micro_cer = 0.06003813674748025`（`6.0038%`，绝对下降 `-0.0436%`）
    - 四退化场景合计编辑数：`2070 / 29368`（**相对 Base 净减 −17 个退化编辑**），`degraded_macro_cer = 0.07048488150367748`（`7.0485%`，绝对下降 `-0.0579%`）
    - `clean` 编辑数：`134 / 7342`（相对 Base `+1` 个编辑），`clean_cer = 0.018251157722691364`（`1.8251%`）
  - **`dpo`（`dpo_06b_zh_20261022T133007_step000040`，纯方案 1，`adapter_sha256=1d28d311...`）**：
    - `PHASE5_STAGE_DONE stage=dpo edits=2200 micro_cer=0.05992917 degraded_macro_cer=0.07051893`
    - 总体编辑数：**`2200 / 36710`**（**相对 Base 净减 −20 个编辑，相对 SFT 净减 −4 个编辑，为封存 `test.jsonl` 总体最优阶段**），`micro_cer = 0.059929174611822395`（`5.9929%`，相对 Base 绝对下降 `-0.0545%`）
    - 四退化场景合计编辑数：`2071 / 29368`（**相对 Base 净减 −16 个退化编辑**，相对 SFT `+1` 个编辑），`degraded_macro_cer = 0.07051893217107056`（`7.0519%`）
    - `clean` 编辑数：**`129 / 7342`**（**相对 Base 净减 −4 个编辑，相对 SFT 净减 −5 个编辑**），`clean_cer = 0.017570144374829747`（`1.7570%`）
  - **`rl`（`rl_06b_zh_20261023T042034_step000005`，对齐版 RL Pilot，`adapter_sha256=79df1124...`）**：
    - 总体编辑数：`2213 / 36710`（**相对 Base 净减 −7 个编辑**，相对 SFT `+9` 个编辑，相对 DPO `+13` 个编辑），`micro_cer = 0.060283301552710435`（`6.0283%`，相对 Base 绝对下降 `-0.0191%`）。
    - 注：因保活轮询间隔为 180 秒，`rl` 阶段末尾与总控收尾在同一轮询窗口内完成，保活日志保留了最终行 `PHASE5_BENCHMARK_DONE ... rl_edits=2213`；`rl` 的分场景 `metrics.json`/`by_scenario.csv` 及四阶段逐条 `scored.jsonl` 已持久化在 Drive `runs/20261026T101800_phase5_test_06b_zh_seed42/`，待下次挂载 Drive 时可按需拉回本地 `results/` 做逐场景样本级差异拆解。
- **`val.jsonl`（36620 字）与封存 `test.jsonl`（36710 字）跨集泛化结论**：
  1. **后训练三阶段在封存 `test.jsonl` 上全部优于 Base（`2220` 编辑）**：SFT（`2204`，`-16`）、DPO（`2200`，`-20`）、RL（`2213`，`-7`）均在零超参泄漏的一次性封存测试集上保持相对官方 `Qwen/Qwen3-ASR-0.6B` 的正向 CER 净收益。
  2. **`Base → SFT → DPO` 在 `val` 与 `test` 上均呈现严格单调下降**：`val` 上 `2341 → 2314 (-27) → 2306 (-35)`；封存 `test` 上 `2220 → 2204 (-16) → 2200 (-20)`，且 DPO 同时把 `test` 的 `clean` 错误从 `134`（SFT）压低到 `129`（相对 Base `-4`、相对 SFT `-5`），验证了纯方案 1（单字原位替换 + `mismatched_only` + `dropout` 父假设锚 + 100 步余弦周期在 step 40 收束）具备稳健的跨说话人泛化能力。
  3. **RL（Step 5）存在轻度 `val → test` 泛化落差**：在 `val` 上 RL 相对 DPO 进一步减少 `8` 个编辑（`2306 → 2298`），但在封存 `test` 上比 DPO 多 `13` 个编辑（`2200 → 2213`，仍比 Base 少 `7` 个编辑）。这表明小样本 pilot（`lr=1e-6`、5 个优化器步、20 个对齐信号 group）上的组内在线采样策略梯度对开发集分布更敏感；在当前 pilot 规模下，**若以封存 `test.jsonl` 终局泛化为准，`dpo_06b_zh_20261022T133007_step000040` 是综合表现最稳健的模型版本**。



