# Phase 1 运行记录

最后更新：2026-10-14

这份记录按执行顺序列出 Phase 1 manifest 的脚本、完整命令、退出码和结果。摘要写在 `00_progress.md`。合成音只用来证明门禁，不是审计过的语音语料，也不能交给 SFT。

判定只用三种：`passed`、`failed`、`blocked`。这里的 `passed` 只表示这一步的命令符合预期。正式配置得到 `insufficient_data` 时，Phase 1 仍然未完成。

## 记录格式

每一节固定包含：

- 目的
- 脚本或入口
- 命令
- 退出码
- 结果
- 判定

## 本地合同测试

### P1 按文档命令直接构建

- 目的：用正式配置对一条合成音执行文档里的构建命令。
- 脚本：`scripts/build_manifest.py`。
- 命令：

```bash
python3 scripts/build_manifest.py \
  --config configs/data/manifest_mvp.yaml \
  --sources results/phase1_synthetic_20261013/sources/source.jsonl \
  --output-dir results/phase1_synthetic_20261013/manifest \
  --seed 42
```

实际解释器是 `.venv/bin/python`。源是 0.6 秒、280 Hz 的正弦，文本是「样本一」，`source_utterance_id` 是 `tiny0001`。

- 退出码：1
- 结果：`ModuleNotFoundError: No module named 'evaluation'`。直接运行 `scripts/build_manifest.py` 时，Python 把 `scripts/` 放在导入路径最前面，仓库根目录不在路径里。`manifest_validate.py` 同样报 `No module named 'scripts'`。没有写出 `manifest.json`。
- 判定：failed

### P2 补上仓库根目录并重跑合同测试

- 目的：让文档中的文件入口可以导入 `evaluation` 和 `scripts`，并锁住退出码。
- 脚本：`scripts/build_manifest.py`、`scripts/manifest_validate.py` 在导入前把仓库根目录插入 `sys.path`，与 `scripts/run_phase0_baseline.py` 相同。回归测试在 `tests/test_manifest.py`。
- 命令：

```bash
.venv/bin/python -m unittest tests.test_manifest tests.test_directory_readmes -v
```

- 退出码：0
- 结果：9 个测试，0.527 秒，`OK`。覆盖两次构建的 JSONL 和 WAV 哈希一致、同一 source 的五条场景共用 split 且五个 WAV 哈希不同、缺失文件、0.1 秒音频、拉丁文本和 `../` 路径进入 `rejected.jsonl`、正式配置小样本退出码 3、夹具在 `audited: true` 且 val 配额为 1 时退出码 0、手改同一 source 到 train 和 val 后退出码 1、seed 不一致时退出码 2 且不写 `train.jsonl`、改一条文本会改变 JSONL 哈希，以及直接运行两个脚本文件时的导入。夹具退出码 0 不是训练集。
- 判定：passed

### P3 正式配置的合成试跑

- 目的：在合同测试通过后，用未改动的 `configs/data/manifest_mvp.yaml` 留下一次可回看的构建和校验。
- 脚本：`scripts/build_manifest.py`、`scripts/manifest_validate.py`。输入是 P1 的同一条合成源。
- 命令：

```bash
.venv/bin/python scripts/build_manifest.py \
  --config configs/data/manifest_mvp.yaml \
  --sources results/phase1_synthetic_20261013/sources/source.jsonl \
  --output-dir results/phase1_synthetic_20261013/manifest \
  --seed 42
.venv/bin/python scripts/manifest_validate.py \
  --manifest results/phase1_synthetic_20261013/manifest \
  --config configs/data/manifest_mvp.yaml \
  --check-audio
```

- 退出码：构建 3，校验 3。
- 结果：两次都打印 `insufficient_data`。`manifest.json` 的 `problems` 为空，`audited` 为 false。5 行都在 train，`tiny0001` 的五个场景各 1 条；val 和 test 都是 0。五条 val 短缺的 `minimum` 都是 50、`rows` 都是 0。另有一条短缺 `source_not_audited`，`source_version` 是 `unset`。`sha256.json` 的 schema 是 `qwen3-asr-sha256-v1`，列出 10 个文件，不含它自己。`generator_git_sha` 是 `183ff50b7ebddfd1227aeb2e969d9c804b1d90bf`，这是当时的 `HEAD`。工作区还有未提交修改，所以这个 SHA 不能单独复现这次代码。产物在 `results/phase1_synthetic_20261013/`，该目录被 git 忽略。
- 判定：passed。预期就是退出码 3。Phase 1 没有完成，SFT 不能开始。

## AISHELL-1 子集

`audited` 仍是 false。下面的 `passed` 只表示命令和预期退出码一致，不表示 Phase 1 可以交给 SFT。

### P4 导入合同测试

- 目的：锁住转写去重、配额前缀和缺失音频，不访问网络。
- 脚本：`tests/test_import_aishell.py`，一起重跑 manifest 和目录 README 合同。
- 命令：

```bash
.venv/bin/python -m unittest tests.test_import_aishell tests.test_manifest tests.test_directory_readmes -v
```

- 退出码：0
- 结果：13 个测试，0.688 秒，`OK`。
- 判定：passed

### P5 下载三个说话人并写出源 JSONL

- 目的：按转写前缀准备足够的真实语句，使 seed 42 的模拟切分里 val 至少 70 条。
- 脚本：`scripts/import_aishell_source.py`。转写和音频来自 Hugging Face `AISHELL/AISHELL-1` commit `8515aae54a8ded02c55fe270e32bd146b4180158`，许可证 Apache-2.0。
- 命令：

```bash
curl -L --fail --retry 3 -o results/phase1_aishell_subset_20261013/raw/aishell_transcript_v0.8.txt \
  https://huggingface.co/datasets/AISHELL/AISHELL-1/resolve/8515aae54a8ded02c55fe270e32bd146b4180158/data_aishell/transcript/aishell_transcript_v0.8.txt
curl --http1.1 -L --fail --retry 5 --retry-all-errors -C - -o results/phase1_aishell_subset_20261013/raw/tars/S0002.tar.gz \
  https://huggingface.co/datasets/AISHELL/AISHELL-1/resolve/8515aae54a8ded02c55fe270e32bd146b4180158/data_aishell/wav/S0002.tar.gz
curl --http1.1 -L --fail --retry 5 --retry-all-errors -C - -o results/phase1_aishell_subset_20261013/raw/tars/S0003.tar.gz \
  https://huggingface.co/datasets/AISHELL/AISHELL-1/resolve/8515aae54a8ded02c55fe270e32bd146b4180158/data_aishell/wav/S0003.tar.gz
curl --http1.1 -L --fail --retry 5 --retry-all-errors -C - -o results/phase1_aishell_subset_20261013/raw/tars/S0004.tar.gz \
  https://huggingface.co/datasets/AISHELL/AISHELL-1/resolve/8515aae54a8ded02c55fe270e32bd146b4180158/data_aishell/wav/S0004.tar.gz
tar -xzf results/phase1_aishell_subset_20261013/raw/tars/S0002.tar.gz -C results/phase1_aishell_subset_20261013/raw/wavs
tar -xzf results/phase1_aishell_subset_20261013/raw/tars/S0003.tar.gz -C results/phase1_aishell_subset_20261013/raw/wavs
tar -xzf results/phase1_aishell_subset_20261013/raw/tars/S0004.tar.gz -C results/phase1_aishell_subset_20261013/raw/wavs
.venv/bin/python scripts/import_aishell_source.py \
  --transcript results/phase1_aishell_subset_20261013/raw/aishell_transcript_v0.8.txt \
  --wav-root results/phase1_aishell_subset_20261013/raw/wavs \
  --output-dir results/phase1_aishell_subset_20261013/sources \
  --seed 42 \
  --min-val 70 \
  --min-train 1
```

- 退出码：转写下载 0。`S0002` 和 `S0004` 第一次下载 0。`S0003` 第一次在 HTTP/2 取消，退出码 92，文件停在 15736924 字节；用 `--http1.1` 从该位置续传后退出码 0。三个包的字节数分别是 36455331、42385634、25559425，与数据集列出的大小一致。解包后导入退出码 0。
- 结果：转写 SHA-256 是 `b5f33b9e0b47548e20a5ea4e504297f80df41d559133924c4e5b7b544c15b5c4`。导入写出 748 行，模拟切分是 train 588、val 70、test 90，`missing_wavs` 为 0。说话人是 `S0002`、`S0003`、`S0004`，三人都同时出现在 train、val、test。`speaker_independent` 为 false，`audited` 为 false。整份转写里有 7729 条重复文本被跳过。音频和源 JSONL 在 git 忽略的 `results/phase1_aishell_subset_20261013/`。
- 判定：passed。这只说明源文件准备好了，不是说话人独立的评测集。

### P6 用正式配置构建并校验

- 目的：看行数配额是否已经满足，以及 `audited: false` 是否仍挡住退出码 0。
- 脚本：`scripts/build_manifest.py`、`scripts/manifest_validate.py`。配置是 `configs/data/manifest_mvp.yaml`。
- 命令：

```bash
.venv/bin/python scripts/build_manifest.py \
  --config configs/data/manifest_mvp.yaml \
  --sources results/phase1_aishell_subset_20261013/sources/source.jsonl \
  --output-dir results/phase1_aishell_subset_20261013/manifest \
  --seed 42
.venv/bin/python scripts/manifest_validate.py \
  --manifest results/phase1_aishell_subset_20261013/manifest \
  --config configs/data/manifest_mvp.yaml \
  --check-audio
```

- 退出码：构建 3，校验 3。
- 结果：两次都是 `insufficient_data`。`problems` 为空，`rejected` 为 0。3740 行，train/val/test 的 source 数是 588/70/90，五个场景在每个 split 里条数相同。val 每个场景 70 条，高于配额 50。时长最短 1.344 秒，中位 4.646 秒，最长 10.44625 秒。唯一短缺是 `source_not_audited`，`source_version` 是 `aishell1-transcript-v0.8`。`generator_git_sha` 是 `183ff50b7ebddfd1227aeb2e969d9c804b1d90bf`，工作区仍有未提交修改。
- 判定：passed。预期就是退出码 3。行数配额已经达到，Phase 1 仍未完成，SFT 不能开始。

## 2026-10-14 撤回三说话人子集

P4–P6 保留作历史记录。那次子集不是原定数据方案里的语料。正式配置改回 `source_version: unset`、`audited: false`。`scripts/import_aishell_source.py` 和 `tests/test_import_aishell.py` 已删除。`results/phase1_aishell_subset_20261013/` 仍留在本机并被 git 忽略，不作为正式语料。

### P7 恢复正式配置后的合同测试和合成试跑

- 目的：确认撤回导入器之后，正式配置仍然按原定门禁得到 `insufficient_data`。
- 脚本：`tests/test_manifest.py`、`tests/test_directory_readmes.py`、`scripts/build_manifest.py`、`scripts/manifest_validate.py`。
- 命令：

```bash
.venv/bin/python -m unittest tests.test_manifest tests.test_directory_readmes -v
.venv/bin/python scripts/build_manifest.py \
  --config configs/data/manifest_mvp.yaml \
  --sources results/phase1_synthetic_20261013/sources/source.jsonl \
  --output-dir results/phase1_synthetic_20261013/manifest \
  --seed 42
.venv/bin/python scripts/manifest_validate.py \
  --manifest results/phase1_synthetic_20261013/manifest \
  --config configs/data/manifest_mvp.yaml \
  --check-audio
```

- 退出码：测试 0。构建 3。校验 3。
- 结果：9 个测试，0.535 秒，`OK`。合成源仍是 `tiny0001`。构建打印 `{"status": "insufficient_data", "rows": 5, "rejected": 0}`。`manifest.json` 的 `source_version` 是 `unset`，`audited` 是 false。val 五个场景各 0 条，配额 50。短缺还包括 `source_not_audited`。产物仍在 `results/phase1_synthetic_20261013/`。
- 判定：passed。预期就是退出码 3。Phase 1 没有完成，SFT 不能开始。没有另选语料。

### P8 官方切分合同

- 目的：在下载 AISHELL-1 之前，让构建器能保留官方 train/dev/test 和说话人，并拒绝跨 split 的说话人。
- 脚本：`scripts/manifest_contract.py`、`scripts/build_manifest.py`、`scripts/manifest_validate.py`、`tests/test_manifest.py`。
- 命令：

```bash
.venv/bin/python -m unittest tests.test_manifest tests.test_directory_readmes -v
```

- 退出码：0。
- 结果：11 个测试，0.868 秒，`OK`。新增两项：`official_split: dev` 的行进入 `val.jsonl`，且该 source 的哈希切分不是 val；同一 `speaker_id` 的后一条被写入 `rejected.jsonl`，原因 `speaker_cross_split`，没有被挪进先出现的 split。`official_split: val` 的原因是 `bad_official_split`。把 val 行的说话人改成 train 里已有的说话人后，校验退出码 1，问题名 `speaker_leakage:`。哈希行和官方行放在同一份源文件时，构建退出码 2，stderr 是 `mixed_split_mode`，不写 `train.jsonl`。行 schema 是 `qwen3-asr-manifest-v2`。正式配置没有改。
- 判定：passed。这只证明夹具合同。Phase 1 没有完成，SFT 不能开始。没有下载语料。

### P9 固定版本核对与 50 条自动检查

- 目的：固定 AISHELL-1 的一份版本、归档 hash 和官方切分信息，并对现有音频做 50 条自动检查。检查项是音文能否对上、时长、切边能量、静音、削波、数字和中英混合。这一步不生成正式 manifest，也不把语料标成已审阅。
- 脚本：`results/phase1_aishell_review_20261014/qc/review50.py`。它在 git 忽略的 `results/` 里，不是 `scripts/` 的维护入口。正式配置没有调用它。
- 命令：

```bash
mkdir -p results/phase1_aishell_review_20261014/upstream results/phase1_aishell_review_20261014/qc
curl -fsSL --retry 5 --retry-all-errors \
  -o results/phase1_aishell_review_20261014/upstream/resource_aishell.tgz \
  "https://openslr.trmal.net/resources/33/resource_aishell.tgz"
shasum -a 256 results/phase1_aishell_review_20261014/upstream/resource_aishell.tgz
tar -tzf results/phase1_aishell_review_20261014/upstream/resource_aishell.tgz
shasum -a 256 \
  results/phase1_aishell_review_20261014/upstream/resource_aishell.tgz \
  results/phase1_aishell_subset_20261013/raw/aishell_transcript_v0.8.txt
.venv/bin/python results/phase1_aishell_review_20261014/qc/review50.py
```

随后用 Hugging Face revision API 核对 commit，并另读默认分支的 wav 目录数量。相对峰值 5% 的静音重读写进 `review50_summary.json` 的 `silence_reread`。

- 退出码：资源包下载和两次 SHA-256 为 0。`review50.py` 为 0。revision API 返回的 `sha` 与请求的 commit 相同。
- 结果：
  - 固定查看的 Hugging Face 数据集是 `AISHELL/AISHELL-1`，commit `8515aae54a8ded02c55fe270e32bd146b4180158`。wav 目录列出 100 个文件，`data_aishell/wav/S0002.tar.gz` 到 `S0101.tar.gz`。每个包只取 gzip 前 8192 字节，100 个包的首个 tar 成员都是 `train/Sxxxx/`。`S0102.tar.gz`、`S0200.tar.gz`、`S0340.tar.gz`、`S0724.tar.gz`、`S0764.tar.gz`、`S0900.tar.gz`，以及 `data_aishell/wav/dev/S0724/BAC009S0724W0121.wav` 和 `data_aishell/wav/train/S0002/BAC009S0002W0122.wav`，在这个 commit 上都是 404。同一天不带 revision 的数据集 API 返回 `sha` `bbe295d530192a4cd41644b711c9aecd087df653`，卡片许可是 `apache-2.0`；那个 `sha` 的 wav 目录也是 100 个文件。文件钉扎用的是前一个 commit。
  - 转写 `aishell_transcript_v0.8.txt` 为 141600 行、10091431 字节。SHA-256 是 `b5f33b9e0b47548e20a5ea4e504297f80df41d559133924c4e5b7b544c15b5c4`。Hugging Face 上该文件的 git oid 是 `05217b79e3df3f9ea07f667bf028e7b1a4e39d2a`，大小与本机一致。去掉空白后，ASCII 拉丁字母 0 行，数字 0 行，`text_problem` 命中 0 行。全角拉丁字母有 7 行：`BAC009S0737W0408`（ｔ）、`BAC009S0739W0467`（ａ）、`BAC009S0748W0439`（ｋ）、`BAC009S0761W0385`（ｃ）、`BAC009S0768W0458`（ｂ）、`BAC009S0902W0421`（ｋ）、`BAC009S0915W0180`（ａ）。当前规则不拒绝全角字母。这 7 个说话人都不在上述 100 个包里。空白合并后的重复文本有 5723 组、13452 行；其中 480 组同时出现在这 100 个说话人和镜像之外的说话人。镜像之外仍可能是官方 train，这组计数不能当成跨 split 泄漏。
  - `resource_aishell.tgz` 为 1246920 字节，SHA-256 是 `1a6749854456e9402bc7295767937367afed1327799a5e1df0ed64baa5f77409`。成员是 `lexicon.txt` 和 `speaker.info`。`speaker.info` 有 400 行，格式如 `0002 M`，其中 F 214 行、M 186 行，没有 train/dev/test。
  - 本机三个包的 SHA-256 与该 commit 的 LFS oid 一致，大小也一致：`S0002.tar.gz` 36455331 字节，`5700ffa081f42c5a1be701147e680490fa494db959ca3c10ae205c3658af159f`；`S0003.tar.gz` 42385634 字节，`c15938c88bcdd0f5b2ad4b64862d2bdd0a577fdb966777aceb84572aea105517`；`S0004.tar.gz` 25559425 字节，`c9169b501da24ad2b91ffd802a06bf4ed145860b96dd088188be30465d7ef058`。首成员分别是 `train/S0002/`、`train/S0003/`、`train/S0004/`。
  - 50 条从已解压的 train `S0002`、`S0003`、`S0004` 抽出。可配对转写的 wav 是 364、354、355。另有 3 条 wav 不在转写中：`BAC009S0002W0413`、`BAC009S0004W0186`、`BAC009S0004W0419`。抽样先按说话人配额 17、17、16，再按时长把该说话人分成三段，段内保留 `SHA-256(42:{utterance_id}:review50)` 较小的语句。结果是短 18、中 17、长 15。50 条都是 16 kHz、单声道、PCM16。时长 1.909938 到 8.303 秒。`text_problem` 为 0，削波样本为 0，峰值绝对值 1085 到 10018。绝对阈值 328 下，首端静音中位 0.533 秒、最长 1.035 秒，尾端中位 0.486 秒、最长 0.776 秒。这些文件峰值远低于满幅，绝对阈值会把较轻的语音尾算进静音。按峰值 5% 重读后，首端中位 0.327 秒、最长 0.605 秒，尾端中位 0.309 秒、最长 0.683 秒。没有人听过这 50 条。明细在 `results/phase1_aishell_review_20261014/qc/review50.jsonl`。
  - 本机剩余约 17 Gi。没有下载约 15G 的 `data_aishell.tgz`。`configs/data/manifest_mvp.yaml` 仍是 `source_version: unset`、`audited: false`。没有启动 SFT，没有新建 Colab 会话。
- 判定：blocked。100 个包的 train 前缀、转写 hash 和资源包 hash 已经记下，但官方 dev/test 音频不在这份镜像里，50 条也只是三位 train 说话人的自动检查。Phase 1 没有完成，SFT 不能开始。

### P10 在 Colab 磁盘下载完整包

- 目的：按选定方案把 OpenSLR 的 `data_aishell.tgz` 下到 Colab 本机磁盘。子集和 hash 随后写入 Google Drive。这一步不用 GPU。正式配置不改，`audited` 保持 false。
- 脚本：Colab CLI。远端启动脚本只留在本机 `/tmp/colab_aishell_download.py`，不进入 `scripts/`。
- 命令：

```bash
colab usage
colab new -s lora-asr-aishell
colab exec -s lora-asr-aishell -f /tmp/colab_disk_probe.py --timeout 60
colab status -s lora-asr-aishell
COLAB_SESSION=lora-asr-aishell python3 /tmp/drive_mount_pty.py
colab exec -s lora-asr-aishell -f /tmp/colab_aishell_download.py --timeout 90
```

- 退出码：会话创建、磁盘探针、Drive 挂载和下载启动都是 0。
- 结果：会话 `lora-asr-aishell` 是 CPU、Standard，没有 GPU。`/content` 可用约 220 Gi。Drive 挂在 `/content/drive`，`/content/drive/MyDrive/qwen3-asr` 存在。下载地址是 `https://openslr.trmal.net/resources/33/data_aishell.tgz`，`Content-Length` 是 15582913665 字节。进程 1478 已开始写入 `/content/aishell/data_aishell.tgz`。启动后不久文件约 38 MiB，进程仍在。账户当时余额 815.44 compute units，费率 0.08/小时。授权地址没有写入本记录。
- 保活：本机监视已停。SSH 主机 `sub2` 上的 `/root/colab-keepalive/aishell_download_keepalive.py` 在会话里每 60 秒打印一次大小，直到下载结束或 curl 退出。启动命令是 `colab exec -s lora-asr-aishell -f /root/colab-keepalive/aishell_download_keepalive.py --timeout 43200`，进程号 3535493。`sub2` 根分区只剩约 17G，放不下这个压缩包，所以文件仍在 Colab 磁盘。启动后日志为 `size=678502400 alive=1`。
- 自动续跑：2026-10-14 12:12 UTC，`sub2` 上的 `/root/colab-keepalive/aishell_p10_watch.sh`（进程 3571627，父进程 1）等下载保活退出后，再执行 `colab exec -s lora-asr-aishell -f /root/colab-keepalive/aishell_p10_continue.py --timeout 43200`。脚本会核对 15582913665 字节和 SHA-256，从包内路径整理说话人切分，把 dev/test 说话人包打成一个 tar 写入 `/content/drive/MyDrive/qwen3-asr/data/aishell1/20261014/`，并从 train/dev 各抽 25 条做自动检查。不改 `audited`，不构建正式 manifest，不启动 SFT。脚本副本在 `results/phase1_aishell_review_20261014/qc/p10_continue.py`。
- 2026-10-14 13:36 UTC 文件已经是 15582913665 字节，但 curl 变成了内核的僵尸进程，保活一直看到 `alive=1`，续跑没有启动。13:38 UTC 重启了这个会话的内核，没有停止 VM。新的 `colab exec` 进程是 3587840。日志已经出现 `P10_DOWNLOAD_DONE` 和 `P10_SHA256_START`。
- 第一次续跑在列目录后失败。外层 404 个成员是 `data_aishell/wav/Sxxxx.tar.gz`，没有 train/dev/test 目录。SHA-256 已算出：`a4a0313cde0a933e0e01a451f77de0a23d6c942f4694af5bb7f40b9dc38143fe`，长度 15582913665。抽查 `S0724.tar.gz` 的内部路径是 `dev/S0724/`。
- 13:52 UTC 前后的第一次重跑在抽查包内路径之前就开始解 400 个包，已停掉。
- 解包前核对现在写在脚本里：外层不得出现 `wav/train`、`wav/dev`、`wav/test`；说话人包必须是 400 个；Drive 必须已挂载；抽查 `S0002`、`S0724` 和目录最后一个包。`S0002` 必须是 `train/`，`S0724` 必须是 `dev/`。不过这三项，不整包解压。
- 2026-10-14 的核对结果是 `P10_PREFLIGHT_OK S0002=train S0724=dev S0723=train`。`S0723` 在目录末尾，但是 train。说话人编号的顺序不是切分顺序。核对通过后才出现 `P10_EXTRACT_START 401`。进程 3590945。
- 整包解压完成后，400 个说话人包的内部分类是 train 340、dev 40、test 20，未知 0。50 条自动检查是 train 25、dev 25，来自 24 个 train 说话人和 20 个 dev 说话人。全部是合格的 16 kHz 单声道 PCM16，时长 2.644 到 7.758 秒，无削波，`text_problem` 为 0。没有人听过。
- Drive 目录 `/content/drive/MyDrive/qwen3-asr/data/aishell1/20261014/` 有 `data_aishell.tgz.sha256`、`split_index.json`（27425 字节）、`aishell1_dev_test_speaker_tars.tar`（2445813760 字节）、转写、`review50.jsonl`、`review50_summary.json` 和 `p10_status.json`。日志以 `P10_PIPELINE_DONE` 结束，进程已退出。
- 判定：passed。这里只表示压缩包哈希、官方切分清单、dev/test 子集和 50 条自动检查已经落盘。样本没有人工抽听，`audited` 仍是 false，`source_version` 仍是 `unset`。Phase 1 没有完成，SFT 没有启动。

### P11 准备 50 条人工抽听

- 目的：把 P10 的 train 25 条和 dev 25 条做成可以逐条听的审阅材料。人要听音文是否一致、有没有切边、静音是否异常、有没有读错、有没有数字或中英混合。自动检查不能代替人听。
- 范围：从 Drive 取出审阅清单和 dev 音频；train 说话人包不在 Drive 上，需要在新的 Colab 磁盘上重新下载完整包后再抽这 25 条。空白审阅表先留下，听完之前 `heard` 保持 false。不改 `configs/data/manifest_mvp.yaml`，不把 `audited` 改成 true，不改 `source_version`，不构建正式 manifest，不实现 5 小时抽样器，不启动 SFT。
- 背景：2026-10-15 早上旧 VM 已经不在。15G 压缩包和 train 说话人包只在那张磁盘上。Drive 的 `qwen3-asr/data/aishell1/20261014/` 仍有 dev/test 说话人包、转写、切分清单和 `review50.jsonl`。本机 `results/phase1_aishell_review_20261014/qc/review50.jsonl` 是 P9 的三位 train 说话人，不能当成这 50 条。
- 会话：新建 CPU Standard 会话 `lora-asr-aishell`，不用 GPU，不用 GCP。Drive 用 Colab 的一次性授权重新挂载。授权地址不写入本记录。
- 脚本：一次性脚本放在 `results/phase1_aishell_listen_20261015/`，并复制到 `sub2` 的 `/root/colab-keepalive/`。不写入 `scripts/`。
- 命令：

```bash
colab sessions
colab new -s lora-asr-aishell
colab drivemount -s lora-asr-aishell /content/drive
colab exec -s lora-asr-aishell --timeout 40 -f /root/colab-keepalive/start_download.py
colab exec -s lora-asr-aishell --timeout 1800 -f /root/colab-keepalive/extract_dev_wavs.py
colab download -s lora-asr-aishell /content/aishell/listen50/dev_wavs.tar /tmp/p11_dev_wavs.tar
```

- 2026-10-15：`colab sessions` 当时没有活跃会话。新建的 `lora-asr-aishell` 是 CPU、Standard，没有 GPU。第一次 `drivemount` 的标准输入是 `/dev/null`，同意页已经出现 “Please close this tab / window.”，回车没有送到进程，内核在超时后报 `ValueError: mount failed`。随后用 pty 重新挂载，同意完成后送出回车，日志是 `Mounted at /content/drive`，退出码 0。授权地址没有写入本记录。
- Drive 目录 `qwen3-asr/data/aishell1/20261014/` 仍在。`review50.jsonl` 29586 字节，50 行，train 25、dev 25，`heard` 全是 false。train 说话人 24 个：`S0004`、`S0025`、`S0055`、`S0086`、`S0098`、`S0101`、`S0115`、`S0124`、`S0139`、`S0169`、`S0185`、`S0197`、`S0204`、`S0230`、`S0236`、`S0239`、`S0249`、`S0252`、`S0345`、`S0351`、`S0423`、`S0512`、`S0719`、`S0723`。其中 `S0723` 在外层包的末尾，所以要抽齐 train 这 25 条，需要重新读完整份 `data_aishell.tgz`。本机旧子集里只有 `BAC009S0004W0312` 这一条对得上。
- dev 抽取退出码 0。20 个 dev 说话人包里写出 25 条 wav，小包 `dev_wavs.tar` 为 3409920 字节，SHA-256 `c1f95ea50a5b4ac6840affb4370ae1265811dec83f2ce58fabbd6fb487976b8a`。同一份已复制到 Drive `qwen3-asr/data/aishell1/20261015/dev_wavs.tar`。本机副本在 `results/phase1_aishell_listen_20261015/wavs/dev/`，25 个文件都在，没有缺号，格式都是 16 kHz、单声道、PCM16。
- 空白审阅表是 `results/phase1_aishell_listen_20261015/listen_sheet.jsonl` 和 `listen_sheet.md`。字段是音文一致、切边、静音、读错、数字或中英混合。`heard` 保持 false。没有跑识别。
- train 重下：`start_download.py` 退出码 0，curl 进程 11311，独立会话，目标 `/content/aishell/data_aishell.tgz`。2026-10-15 08:05 +0800 文件达到 15582913665 字节，curl 日志是 `100 14.5G`。进程随后变成僵尸 `Zs`，父进程是内核，没有被回收。监视脚本还把每次检查用的 shell 命令行也当成 curl，所以一直记 `alive=1`，抽取没有启动。2026-10-15 09:18 +0800 改成只认 `comm` 为 `curl` 且状态不是 `Z` 的进程，重启监视（进程 3692388）。下一次检查是 `alive=0`，日志出现 `START_TRAIN_EXTRACT` 和 `P11_TRAIN_EXTRACT_START`。脚本会先核对 SHA-256 `a4a0313cde0a933e0e01a451f77de0a23d6c942f4694af5bb7f40b9dc38143fe`，再把 25 条 train wav 写入 Drive `qwen3-asr/data/aishell1/20261015/train_wavs.tar`。哈希和抽文件还在进行。
- 2026-10-15 用户听过当时本机已有的 dev 25 条，原话是「音频没问题」。`listen_sheet.jsonl` 里这 25 条 `heard` 改为 true，音文一致、切边、静音、读错、数字或中英混合记为通过，备注写明只覆盖这 25 条。train 25 条当时 `wavs/train/` 是空的，保持 `heard: false`。这次没有改 `configs/data/manifest_mvp.yaml`。
- train 抽取退出码 0。SHA-256 与 P10 一致：`a4a0313cde0a933e0e01a451f77de0a23d6c942f4694af5bb7f40b9dc38143fe`。24 个说话人包写出 25 条 wav。`train_wavs.tar` 为 3491840 字节，SHA-256 `e4eeb2db10d22b7098d65914cd720e29d0efe03352368e5b979d637e7b4e6f17`。Drive 上同一路径 `qwen3-asr/data/aishell1/20261015/train_wavs.tar` 也是 3491840 字节。本机 `wavs/train/` 有 25 个文件，和审阅表的 train 语句一一对应，格式都是 16 kHz、单声道、PCM16。这 25 条 `heard` 仍是 false。
- 2026-10-15 按用户要求，用多模态转写核对这 50 条是否和参考文本一致，避免人耳抽听堵住后面的步骤。接口是 `POST https://api.x.ai/v1/stt`，模型 `grok-voice-transcribe-2.0`，`language=zh`，`format=false`。没有把参考文本当作 `keyterm` 送进去。50 次请求都是 HTTP 200。结果在 `results/phase1_aishell_listen_20261015/stt_grok.jsonl`。
- 去掉标点和空白后，34 条与 `text_joined` 相同。`BAC009S0004W0312` 的「二十四」写成了「24」，`BAC009S0169W0270` 的「二零零零」写成了「2000」，其余字相同。另外 14 条有字差：train 6 条、dev 8 条。train 是 `BAC009S0098W0374`（比如在/比卢泰）、`BAC009S0719W0464`（首例/手力）、`BAC009S0230W0388`（连扳/连班，比分写成 21-17 和 21-16）、`BAC009S0124W0331`（弱/若）、`BAC009S0345W0352`（里/已）、`BAC009S0239W0439`（市卫计委/是卫济委）。dev 字差落在用户已经听过并通过的那 25 条里，所以字差不能单独当成音频不合格。没有空转写，没有拉丁字母，检测语言都是中文。train 25 条的 `heard` 仍是 false。
- 2026-10-15 复核这 14 条字差。`results/phase1_aishell_subset_20261013/raw/aishell_transcript_v0.8.txt` 里的 16 条相关原文和审阅表的 `text_joined` 一致，包括两条数字写法。字差不是审阅表抄错。第二次请求仍用 `grok-voice-transcribe-2.0`、`language=zh`、`format=false`，只把官方分歧词放进 `keyterm`。14 条都返回了这些词：比如、首例、连扳、技术弱、圈里、卫计委、庾澄庆、张嘉欣、倒右手、救生背心、这件、郭健、坝上、专电、信贷纪录、发债、互赔。结果在 `stt_keyterm.jsonl`。另外单独用 `市卫计委` 提示 `BAC009S0239W0439` 时，模型写成「刘阳市卫计委」，把「刘洋」带偏了，所以不采用这一次，官方文本仍是「刘洋市卫计委」。参考文本不改。
- 2026-10-15 再量了 50 条波形的首尾。边界样本相对峰值最高是 0.115，没有一条边界超过峰值的 12%。8 条在 50 毫秒内就有超过峰值 5% 的样本：dev 的 `BAC009S0735W0348`、`BAC009S0730W0290`、`BAC009S0731W0433`、`BAC009S0731W0191`，train 的 `BAC009S0086W0307`、`BAC009S0098W0374`、`BAC009S0139W0300`、`BAC009S0169W0270`。这 8 条的转写都覆盖了官方整句。音频文件不改。train 里这 4 条的切边记为贴边，其余 train 记为通过，静音记为通过。dev 保持用户已经给出的通过。
- 判定：这 50 条没有还要改的音频或文本。许可按当次要求先不处理。`audited` 仍是 false，`source_version` 仍是 `unset`。Phase 1 没有完成，SFT 没有启动。CPU 会话 `lora-asr-aishell` 仍在，15G 完整包只在这张磁盘上。

### P12 抽出 pilot 源

- 背景：50 条抽听已经结案，pilot 还没有可复现的官方切分源。
- 范围：扫描 OpenSLR 整包里的 wav，写出时长、格式和削波，再按 `03_data_plan.md` 的抽样合同选 train 18000 秒、至少 50 名说话人，以及 dev/test 各 500 条。不改官方转写，不改音频，不改 `text_problem`，不改正式配置，不把 `audited` 改成 true，不启动 SFT。许可文字仍不处理。
- 脚本：`scripts/sample_aishell_pilot.py`。整包扫描在 CPU 会话 `lora-asr-aishell` 上运行，输出目录是 `/content/aishell/pilot_catalog`。
- 2026-10-15：本机夹具测试 `tests.test_sample_aishell_pilot`、`tests.test_manifest` 和 `tests.test_directory_readmes` 已通过。
- 整包扫描在 CPU 会话上结束，日志有 `CATALOG_DONE speakers=400 wavs=141925`。`catalog.json` 记录 train/dev/test 波形 120418/14331/7176，说话人包没有切分冲突，满幅削波 22 条。时长表 SHA-256 是 `a8ba1f29047edfd3c30ccc107c7a3151a8d90fd7101118347d3f6c37f91e64fe`。
- 本机 `select` 退出码 0。`selection.json` 的状态是 `selected`。train 3937 条、18000.466 秒、340 名说话人，单人最长 78.879 秒，短于 360 秒上限。dev 500 条、40 名说话人。test 500 条、20 名说话人。拒收是重复文本 7728、无转写波形 325、满幅削波 22、全角拉丁 7。这 7 条是 `BAC009S0737W0408`、`BAC009S0739W0467`、`BAC009S0748W0439`、`BAC009S0761W0385`、`BAC009S0768W0458`、`BAC009S0902W0421`、`BAC009S0915W0180`。
- `source.jsonl` 有 4937 行，字段只有 `source_utterance_id`、`audio`、`text`、`language`、`speaker_id`、`official_split`、`source_package`。SHA-256 是 `7ee673dbfd1f43c8761d80125b2ac1c35d30d16148dfe33b8c229555c7640e49`。`selection.json` 的 SHA-256 是 `21415568cbdd70dc66c26d0baff974653f4441028d42ee37d13bcdc9c5174764`。产物在 `results/phase1_aishell_pilot_20261015/`。
- 选出的 4937 条 wav 已从整包解出。本机 `results/phase1_aishell_pilot_20261015/sample/audio/` 与 `source.jsonl` 的路径一一对应，没有缺号，也没有多出的文件。抽查格式是 16 kHz、单声道、PCM16。音频包 `pilot_sample_audio.tar` 为 735211520 字节，SHA-256 是 `7ee998a09a24c7515c84762f713e99e2b7ecfb78bf8c7643cbb1dacbdc6f6d99`。Drive `qwen3-asr/data/aishell1/20261015/` 上有这份音频包，以及 `source.jsonl`、`selection.json` 和 `rejected.jsonl`。
- 判定：pilot 源和对应音频已经按合同落盘。正式配置没有改，`audited` 仍是 false，`source_version` 仍是 `unset`。Phase 1 没有完成，SFT 没有启动。许可仍不处理。

### P13 构建五场景 manifest

- 背景：pilot 源已经落盘，还没有交给现有构建器。
- 命令：`scripts/build_manifest.py --config configs/data/manifest_mvp.yaml --sources results/phase1_aishell_pilot_20261015/sample/source.jsonl --output-dir results/phase1_aishell_pilot_20261015/manifest --seed 42`。随后独立运行 `scripts/manifest_validate.py --manifest results/phase1_aishell_pilot_20261015/manifest --config configs/data/manifest_mvp.yaml --check-audio`。
- 构建输出最后一行是 `{"status": "insufficient_data", "rows": 24685, "rejected": 0}`。校验退出码是 3，输出是 `{"status": "insufficient_data", "exit_code": 3}`。
- `manifest.json` 的唯一短缺是 `source_not_audited`，`source_version` 为 `unset`，`problems` 为空。官方切分模式。train/val/test 源是 3937/500/500。五个场景在每个 split 上都是这个数量。说话人交集 `train_val`、`train_test`、`val_test` 都是空。拒收原因是空。源包是 `aishell1-openslr33`。
- `manifest.json` SHA-256 是 `d62b1817433ad3974afe743001360d58757c233ea356cf7ebc9e0fac461f4111`。`sha256.json` 收录 24690 个文件，自身 SHA-256 是 `78fe801b4f17a015219ecccdbae01fb8784f634421e63371ac642f188bea933b`。
- 判定：构建和带音频的校验都符合正式配置的预期。`configs/data/manifest_mvp.yaml` 没有改。Phase 1 没有完成，SFT 没有启动。许可仍不处理。

### P14 把 pilot 标成已审阅

- 背景：P13 的唯一短缺是 `source_not_audited`。用户要求把许可证以外的问题解决；数据侧没有未关闭的音文、切边、静音、削波、切分或配额问题，于是把 `audited` 改为 true 并执行下一步。
- 范围：正式配置改为 `source_version: aishell1-openslr33-pilot-20261015`、`audited: true`。OpenSLR 许可证栏 Apache-2.0 和正文 “free for academic use” 仍并列，不写成已接受。train 25 条抽听的 `heard` 仍是 false。不改官方转写，不改音频，不启动 SFT。
- 测试：`python3 -m unittest tests.test_manifest tests.test_sample_aishell_pilot tests.test_directory_readmes -v` 为 14 个测试，退出码 0。正式配置对一条合成音仍是退出码 3，原因是 val 配额，不再包含 `source_not_audited`。
- 重建命令与 P13 相同，seed 42。构建退出码 0，输出 `{"status": "passed", "rows": 24685, "rejected": 0}`。独立 `manifest_validate.py --check-audio` 退出码 0，输出 `{"status": "passed", "exit_code": 0}`。`shortages` 和 `problems` 都是空。
- `manifest.json` SHA-256 是 `811de19e9021eb79c2ecfdbf725665013bac2a8418401a115dcae9acbc90bd1a`。`sha256.json` SHA-256 是 `7e7af4ae7b1900eaba3bae6a807a486cb4d48fc4d37ce68993f87951780c51ee`。
- 下一步：在这份 val manifest 上跑 `Qwen/Qwen3-ASR-0.6B` 的 greedy base，报告分场景 CER 和健康度。WER 仍按 `configs/inference/base_eval.yaml` 不计算。test 不评。SFT 不启动。
- 2026-10-15：标准 A100 会话 `lora-asr-base-val` 上，`inference/run_inference.py --config configs/inference/base_eval.yaml --manifest manifest/val.jsonl --seed 42` 写出 2500 行。`predictions.jsonl` SHA-256 是 `e7589fff432ea111d308012d4faecb76dc8b0abf907d21c231435456a1e0ebba`。`evaluation/eval_cer.py` 退出码 0。val 合计 micro CER `0.06392681594756963`，macro CER `0.06527801794260739`，编辑数 2341，参考字符数 36620。分场景 micro CER：clean `0.022528672856362644`，far_field `0.023757509557618788`，reverb `0.0252594210813763`，noise `0.0599399235390497`，dropout `0.18814855270344075`。每个场景 500 条。空输出、重复、超长、截断和推理错误率都是 0，`valid_output_rate` 是 1。WER 为 null，原因 `wer_tokenizer_not_configured`。产物在 `results/phase1_aishell_pilot_20261015/val_base/`。test 没有评。这是 base 在合成退化 val 上的数字，不是鲁棒性结论，也不与 Mega-ASR 比较。SFT 没有启动。
