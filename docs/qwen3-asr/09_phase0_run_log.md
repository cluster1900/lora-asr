# Phase 0 运行记录

最后更新：2026-10-13

这份记录按执行顺序列出脚本、完整命令、退出码和结果，用来回看哪一步失败。摘要仍写在 `00_progress.md`。原始产物在 `results/<run_id>/`，其中 `command_log.md` 由 `scripts/run_phase0_baseline.py` 追加，内容是该进程内部每一次子命令的 stdout 和 stderr。

判定只用三种：`passed`、`failed`、`blocked`。`blocked` 表示这一步没有做完，后面的步骤不能当成 Phase 0 验收。

## 记录格式

每一节固定包含：

- 目的
- 脚本或入口
- 命令
- 退出码
- 结果
- 判定

## run `phase0_baseline_20261013a`

这次会话在 Drive 挂载前就结束了。下面是事后根据 `results/phase0_baseline_20261013a/` 补记的事实，不是当时逐条落盘的命令日志。

### A1 创建普通内存 A100

- 目的：取得一张标准形状的 A100，不使用 high-mem。
- 脚本：Colab CLI。
- 命令：`colab new -s lora-asr-phase0 --gpu A100`
- 退出码：0
- 结果：会话 READY。`colab status` 显示 Hardware A100、Shape Standard。VM 内 GPU 为 `NVIDIA A100-SXM4-40GB`，显存 42405855232 字节。
- 判定：passed

### A2 安装依赖

- 目的：安装 `qwen-asr`，并确认没有把已有的 CUDA torch 换掉。
- 脚本：`/tmp/phase0_install.py`，由 `colab exec -f` 送到 VM。
- 命令：`colab exec -s lora-asr-phase0 -f /tmp/phase0_install.py --timeout 900`
- 退出码：0
- 结果：torch 保持 `2.11.0+cu130`，`cuda True`。安装后 transformers `4.57.6`。环境探针里的 qwen-asr 版本是 `0.0.6`，Python `3.13.15`。
- 判定：passed

### A3 跑 baseline

- 目的：环境探针、一条 clean、一条 5 dB noise、一条缺失文件、CER、本地读回。
- 脚本：`scripts/run_phase0_baseline.py`
- 命令：

```bash
python3 scripts/run_phase0_baseline.py \
  --config configs/inference/base_eval.yaml \
  --output-dir /content/run/phase0_baseline_20261013a \
  --run-id phase0_baseline_20261013a \
  --seed 42 \
  --git-sha 183ff50b7ebddfd1227aeb2e969d9c804b1d90bf \
  --drive-root /content/drive/MyDrive/qwen3-asr
```

包装命令是 `colab exec -s lora-asr-phase0 -f /tmp/phase0_run.py --timeout 1500`。当时配置里的 `model.revision` 还是 null。上传包 SHA-256 为 `bf69a630b32f41d654cac97dfaa19404de46a799fb0617f68a4c1e9c7e5eb334`。

- 退出码：3
- 结果：IPython 把 `SystemExit: 3` 打印出来。`status.json` 为 `completed_local_readback`，`drive_status=missing`。clean 和 noise 的 greedy 文本都是「甚至出现交易几乎停滞的情况。」，与 `transcribe()` 一致，`stop_reason=eos`，两条 micro CER 为 0。缺失行是 `FileNotFoundError`。三条合计 micro CER `0.3333333333333333`，13 个编辑都来自缺失行。本地 `readback.json` 的 `matched` 为 true。加载到的模型 revision 是 `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`。
- 判定：blocked。转写完成，Drive 读回没有发生。

### A4 收回会话

- 目的：停止计费。
- 命令：`colab stop -s lora-asr-phase0`
- 退出码：0
- 结果：Session terminated。产物已下载到 `results/phase0_baseline_20261013a/`。
- 判定：passed

## run `phase0_baseline_20261013b`

目标是在新的普通内存 A100 上挂载 Drive，让同一次运行把 predictions、metrics 和 environment 写进 `/content/drive/MyDrive/qwen3-asr/runs/phase0_baseline_20261013b/` 再读回。配置里的 revision 已固定为 `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`。

### B1 创建普通内存 A100

- 目的：新开一张标准 A100，不复用已经停止的 `lora-asr-phase0`。
- 脚本：Colab CLI。
- 命令：`colab new -s lora-asr-phase0b --gpu A100`
- 退出码：0
- 结果：Session READY。随后 `colab status -s lora-asr-phase0b` 显示 Hardware A100、Shape Standard、Status IDLE。分配名 `gpu-a100-s-kkb-usc1f1-2io684rtrx88z`。本次上传包 SHA-256 为 `3971656e836ddb77a557b98a7b1af3a6f94bc05a2fc79783d6f6ffa3a858d180`，父提交仍是 `183ff50b7ebddfd1227aeb2e969d9c804b1d90bf`，包内代码相对该提交有未提交修改。
- 判定：passed

### B2 挂载 Google Drive

- 目的：把 `/content/drive` 挂到当前普通内存 A100，供同一次 baseline 写入并读回。
- 脚本：Colab CLI `drivemount`。本地钩子在 `colab_cli/commands/automation.py` 的 `drivefs_hook`。授权未完成时它打印一次性 OAuth 地址，然后用 `open("/dev/tty").readline()` 等待回车。远程 `drive.mount` 约 120 秒超时。
- 命令：`colab drivemount -s lora-asr-phase0b /content/drive`
- 退出码：本地进程 0。这个 0 只表示 CLI 包装结束，不表示挂载成功。
- 结果：约 124 秒后，内核抛出 `ValueError: mount failed`（`google/colab/drive.py` 超时分支）。日志里出现 `REQUIRED: Google Drive Authorization needed` 和 `Press Enter after you have granted access...`。浏览器打开了授权页并到达 “Please close this tab / window.”，但回车没有在 120 秒内送到 `/dev/tty`，所以 `dryrun=false` 的凭据下发没有执行。OAuth 地址含一次性 state，不写入本文件；原始输出在会话终端日志 `call-a13a9b96-9559-447c-bf60-f4aada4b0228-140.log`。事后 `colab ls` 只有 `.config/` 和 `sample_data/`，没有 `drive/`。会话仍是 IDLE，分配名未变，最后执行时间 `2026-10-13 18:40:23`。
- 判定：failed。旧授权链接已失效，不能复用。

### B3 用可控终端重试挂载

- 目的：在远程 `drive.mount` 的约 120 秒超时内完成浏览器同意，并把回车送到本地 CLI 的 `/dev/tty`。
- 脚本：`/tmp/drive_mount_pty.py`。它用 `pty.fork` 启动 `colab drivemount`，把输出写到 `/tmp/drive_mount_b3.log`。只有 `/tmp/drive_auth_ok` 出现后才向伪终端主端写入换行。浏览器同意用 Chrome DevTools 打开当次授权页。OAuth 地址只留在本机临时文件，不写入本记录。
- 命令：

```bash
python3 /tmp/drive_mount_pty.py
# 授权页出现 “Please close this tab / window.” 之后：
touch /tmp/drive_auth_ok
colab exec -s lora-asr-phase0b -f /tmp/phase0_drive_probe.py --timeout 60
```

- 退出码：`drive_mount_pty.py` 为 0。探针命令为 0。
- 结果：伪终端 2026-10-13T18:46:32+0800 启动。第一次自动截到的 `https://` 是 Colab 域名短链接，不是授权地址；真正的授权地址在 `Please visit:` 后面，已改写入 `/tmp/drive_auth_url.txt` 再打开。授权页标题是 “Please close this tab / window.”，state 对应当次会话。回车在 +48.2 秒送出。日志接着出现 `Authorizing VM...`、`Credentials propagated. Resuming mount...`、`Mounted at /content/drive`。进程在 58.9 秒以 status 0 结束。探针输出：`drive_exists True`，`mydrive_exists True`，MyDrive 下 5 个已有目录，还没有 `qwen3-asr`。
- 判定：passed

### B4 上传代码并安装依赖

- 目的：把已记录哈希的源码包送到新会话，安装 `qwen-asr`，并确认不替换已有 CUDA torch。
- 脚本：`/tmp/phase0_install.py`。包是 B1 记录的 `/tmp/lora-asr-phase0b.tgz`，里面含 `CommandLog` 和固定 revision。打包之后才写入的 B2、B3 记录不在这个包里，不影响 VM 上的运行合同。
- 命令：

```bash
colab upload -s lora-asr-phase0b /tmp/lora-asr-phase0b.tgz /content/lora-asr-phase0b.tgz
colab exec -s lora-asr-phase0b -f /tmp/phase0_install.py --timeout 900
```

- 退出码：上传 0，安装 0。
- 结果：上传确认路径 `/content/lora-asr-phase0b.tgz`。安装前 torch `2.11.0+cu130`，`cuda True`，GPU `NVIDIA A100-SXM4-40GB`。安装后 torch 版本和 CUDA 未变。`qwen_asr.__version__` 不存在，打印为 `unknown`。transformers `4.57.6`。整段耗时约 49 秒。
- 判定：passed

### B5 跑 baseline 并写入 Drive

- 目的：在已挂载的 Drive 上完成环境探针、clean、5 dB noise、缺失文件、CER，以及 7 个读回文件的 SHA-256 核对。
- 脚本：`/tmp/phase0_run_b.py` 解包到 `/content/lora-asr`，再调用 `scripts/run_phase0_baseline.py`。解包使用 `tarfile` 的 `filter=data`。
- 命令：

```bash
colab exec -s lora-asr-phase0b -f /tmp/phase0_run_b.py --timeout 1500
```

VM 内实际命令是：

```bash
python3 scripts/run_phase0_baseline.py \
  --config configs/inference/base_eval.yaml \
  --output-dir /content/run/phase0_baseline_20261013b \
  --run-id phase0_baseline_20261013b \
  --seed 42 \
  --git-sha 183ff50b7ebddfd1227aeb2e969d9c804b1d90bf \
  --drive-root /content/drive/MyDrive/qwen3-asr
```

- 退出码：包装进程 0。`baseline_exit 0`。IPython 额外打印了 `SystemExit: 0`。子进程 `run_inference.py` 的退出码是 1，因为缺失音频被记成行错误后进程按设计返回 1；总控看到预测文件存在且退出码不是 2，继续做了 CER 和读回。
- 结果：`status.json` 为 `completed`，`drive_status=mounted`，`transcribe_matches_greedy=true`，时间 `2026-10-13T10:51:12+00:00`。加载 revision `5eb144179a02acc5e5ba31e748d22b0cf3e303b0`，与配置一致。`decode_eval_sha256` 为 `4a4fee2b67884438faa196909749fcc8dcc6126b175554ad19992bf9b4f2f642`。smoke manifest SHA-256 为 `ce8dabca5d7ae19a205f552ed8f58dadd97188a6f5e0c35b12cd7f533f8c1927`。clean 与 noise 的 greedy 和 `transcribe()` 都是「甚至出现交易几乎停滞的情况。」，`stop_reason=eos`，各 8 个 token，micro CER 都是 0。dropout 行是 `FileNotFoundError: audio file does not exist: wavs/does_not_exist.wav`，13 个编辑。三条合计 micro CER 和 macro CER 都是 `0.3333333333333333`。WER 为 null，原因 `wer_tokenizer_not_configured`。Drive 目录 `/content/drive/MyDrive/qwen3-asr/runs/phase0_baseline_20261013b` 有 7 个读回文件，`readback.json` 的 `matched` 为 true。`command_log.md`、`run_report.md` 和 `requirements.lock` 留在 Colab 本地运行目录，不在这 7 个读回文件里。推理 stderr 有 oneDNN 的 `Could not find cuda drivers... GPU will not be used`；同一运行的资源报告仍是 A100-SXM4-40GB、BF16、`world_size=1`。`git_dirty` 为 null，因为 VM 没有 `.git`。父提交记录为 `183ff50b7ebddfd1227aeb2e969d9c804b1d90bf`。
- 判定：passed。5 dB 噪声这条仍然全对，这里只记录探针，不作为鲁棒性结论。

### B6 下载产物

- 目的：把 Colab 本地运行目录带回本机，方便对照 `command_log.md`。
- 脚本：`/tmp/phase0_summarize_b.py` 负责打印摘要并打包；下载用 Colab CLI。
- 命令：

```bash
colab exec -s lora-asr-phase0b -f /tmp/phase0_summarize_b.py --timeout 180
colab download -s lora-asr-phase0b /content/phase0_baseline_20261013b.tgz /tmp/phase0-results/phase0_baseline_20261013b.tgz
tar -xzf /tmp/phase0-results/phase0_baseline_20261013b.tgz -C results
```

- 退出码：摘要 0，下载 0，解包 0。
- 结果：远端包 311655 字节，SHA-256 `fb6bf378de546496ce7b839d44d500d4309c29adb30c81ddbb89058d6ba0dbf8`。本机目录是 `results/phase0_baseline_20261013b/`。该目录被 gitignore，不进入提交。
- 判定：passed

### B7 停止会话

- 目的：产物已经下载，释放这张普通内存 A100。
- 脚本：Colab CLI。
- 命令：`colab stop -s lora-asr-phase0b`
- 退出码：0
- 结果：`Session terminated`。随后 `colab status -s lora-asr-phase0b` 报告会话不存在，`colab sessions` 没有活跃会话。
- 判定：passed

## run `phase0_lock_rebuild_20261013`

目标是在第二份空的普通内存 A100 上安装仓库根目录 `requirements.lock`，并核对 `# verify:` 里的 CUDA 包没有被换掉。完整 729 行冻结指纹不作为 pip 输入。

### C1 创建普通内存 A100

- 目的：取得一张与 `20261013b` 不同的标准 A100，形状保持 Standard。
- 脚本：Colab CLI。
- 命令：`colab new -s lora-asr-lock --gpu A100`
- 退出码：0
- 结果：Session READY。`colab status -s lora-asr-lock` 显示 Hardware A100、Shape Standard、Status IDLE。分配名 `gpu-a100-s-kkb-usc1f1-13utn1udg7cd0`。
- 判定：passed

### C2 安装可安装锁

- 目的：在预装 CUDA torch 与锁一致时，把 `requirements.lock` 的安装行装进这张空 runtime。
- 脚本：`/tmp/phase0_lock_run.py` 调用已上传的 `/content/rebuild_phase0_env.py`。锁文件是仓库根目录 `requirements.lock`。
- 命令：

```bash
colab upload -s lora-asr-lock requirements.lock /content/requirements.lock
colab upload -s lora-asr-lock scripts/rebuild_phase0_env.py /content/rebuild_phase0_env.py
colab upload -s lora-asr-lock /tmp/phase0_lock_run.py /content/phase0_lock_run.py
colab exec -s lora-asr-lock -f /tmp/phase0_lock_run.py --timeout 900
```

VM 内命令是 `python3 /content/rebuild_phase0_env.py --lock /content/requirements.lock --output-dir /content/run/phase0_lock_rebuild_20261013`。

- 退出码：上传 0，包装进程 0。IPython 打印 `SystemExit: 0`。这段约 39 秒。包装进程的 stdout 没有带上 JSON，报告是随后用 `colab exec -f /tmp/phase0_lock_show.py` 读出的。
- 结果：`rebuild_report.json` 的 `status` 为 `passed`，`pip_exit_code` 为 0。安装前和安装后 torch、torchaudio、torchvision、torchcodec 都是锁中的版本，`cuda_before` 和 `cuda_after` 都是 true。13 个安装包安装后全部 `match: true`，其中包括 `qwen-asr==0.0.6` 和 `transformers==4.57.6`。pip 实际把预装的 transformers 5.18.0、huggingface_hub 1.33.0、tokenizers 0.23.2 换成了锁里的 4.57.6、0.36.2、0.22.2，并装上 `qwen-asr`。`pip_stderr.txt` 有一条 resolver 冲突：`diffusers 0.40.0` 要求 `huggingface-hub<2.0,>=1.23.0`，当前是 0.36.2。diffusers 不在这份锁里。
- 判定：passed。完整 729 行冻结指纹没有重放。

### C3 导入检查

- 目的：确认版本号一致之后，`qwen_asr` 和 CUDA torch 能导入。
- 脚本：`/tmp/phase0_lock_import.py`。
- 命令：`colab exec -s lora-asr-lock -f /tmp/phase0_lock_import.py --timeout 180`
- 退出码：0
- 结果：`torch 2.11.0+cu130 cuda True`，`qwen_asr` 文件在 `/usr/local/lib/python3.13/dist-packages/qwen_asr/__init__.py`，transformers `4.57.6`。
- 判定：passed

### C4 下载报告并停止会话

- 目的：留下报告，并释放 A100。
- 命令：

```bash
colab download -s lora-asr-lock /content/run/phase0_lock_rebuild_20261013/rebuild_report.json results/phase0_lock_rebuild_20261013/rebuild_report.json
colab download -s lora-asr-lock /content/run/phase0_lock_rebuild_20261013/pip_stdout.txt results/phase0_lock_rebuild_20261013/pip_stdout.txt
colab download -s lora-asr-lock /content/run/phase0_lock_rebuild_20261013/pip_stderr.txt results/phase0_lock_rebuild_20261013/pip_stderr.txt
colab stop -s lora-asr-lock
```

- 退出码：三次下载都是 0，停止是 0。
- 结果：本机目录 `results/phase0_lock_rebuild_20261013/`。`Session terminated`。`colab sessions` 没有活跃会话。
- 判定：passed

## 会话 `lora-asr-drive` Drive 预挂载

这次只申请 Colab 上的 Google Drive 挂载，供稍后使用。没有打开或操作 GCP 控制台。OAuth 地址和 state 不写入本文件。挂载凭据跟这台 VM 走，停止会话后需要重新同意。

### D1 创建普通内存 A100

- 目的：准备一张稍后可挂 Drive 的标准 A100，不使用 high-mem。
- 脚本：Colab CLI。
- 命令：`colab new -s lora-asr-drive --gpu A100`
- 退出码：0
- 结果：会话 `lora-asr-drive`，分配名 `gpu-a100-s-kkb-usc1b1-3uz0u2t7qnxwz`，Hardware A100，Shape Standard，Variant GPU。
- 判定：passed

### D2 第一次挂载

- 目的：在该会话挂上 `/content/drive`。
- 脚本：`/tmp/drive_mount_pty.py` 的折行版本。它在看到第一条 `https://` 时就把那一行写入 `/tmp/drive_auth_url.txt`。
- 命令：`COLAB_SESSION=lora-asr-drive python3 /tmp/drive_mount_pty.py`
- 退出码：本地进程被终止，没有正常退出码。远程挂载没有成功。
- 结果：默认伪终端宽度把授权地址折成 951 字符和后续 467 字符。浏览器只打开了第一段。同意页到达 “Please close this tab / window.”，但 state 里的 scopes 在 `"email","h` 处截断。回车在 +73.3 秒送出后，CLI 打印 `[colab] Error propagating: 400`。两个本地进程已结束。会话没有停止，内核仍标记 `automation(drivemount)`。
- 判定：failed。截断后的授权地址不能复用。

### D3 重启内核并重试挂载

- 目的：清掉卡住的 `drive.mount`，用足够宽的终端拿到完整授权地址，并在远程超时前送出回车。
- 脚本：改写后的 `/tmp/drive_mount_pty.py`。子进程把终端设为 80 行、4096 列，并设置 `COLUMNS=4096`。只有同时看到 `Please visit:` 和 `Press Enter` 之后，才把中间文字去掉空白拼成一条 URL 写入 `/tmp/drive_auth_url.txt`。`/tmp/drive_auth_ok` 出现后才写入换行。探针是 `/tmp/drive_probe.py`，写入 `qwen3-asr/.mount_probe` 后立刻删除。
- 命令：

```bash
colab restart-kernel -s lora-asr-drive
COLAB_SESSION=lora-asr-drive python3 /tmp/drive_mount_pty.py
# 授权页出现 “Please close this tab / window.” 且 state 含完整 drive scope 之后：
touch /tmp/drive_auth_ok
colab exec -s lora-asr-drive -f /tmp/drive_probe.py --timeout 60
```

- 退出码：`restart-kernel` 为 0。`drive_mount_pty.py` 的子进程 status 0。探针为 0。
- 结果：伪终端 2026-10-13T20:14:22+0800 启动。拼出的授权地址是一行，1418 字符，主机是 `accounts.google.com`。浏览器同意的是 Google Drive for desktop，账户是 hawk.wu4@gmail.com，关闭页 state 的 endpoint 是 `gpu-a100-s-kkb-usc1b1-3uz0u2t7qnxwz`，scopes 含完整的 `https://www.googleapis.com/auth/drive`。回车在 +52.7 秒送出。日志接着出现 `Credentials propagated. Resuming mount...` 和 `Mounted at /content/drive`。进程在 61.9 秒结束。探针：`drive_exists True`，`mydrive_exists True`，MyDrive 6 个条目，`has_qwen3_asr True`，探针读回 `ok\n` 并已删除。随后 `colab status` 为 IDLE。会话保留，没有执行 `colab stop`。
- 判定：passed。这次挂载只在本机会话存活期间有效。
