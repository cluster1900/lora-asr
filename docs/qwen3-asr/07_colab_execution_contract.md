# Colab 执行合同

## 背景、前置条件与边界

Colab 是正式训练环境，单张 A100、单进程、`world_size=1`；Drive 负责输入版本、checkpoint、日志和结果持久化，不能承担每 step 的读写。启动前必须有固定 Git SHA、已登记的依赖版本、manifest 目录和阶段配置；没有 Phase 0 baseline 和 manifest gate 时不得启动 SFT。

每次运行需要一个新的 `run_id`，恢复才复用原 `run_id`。正式 run 要求 checkout 无未提交修改；`allow_dirty=true` 仅用于不能发布的 smoke。运行者必须确认 Drive 根目录、剩余空间、A100 dtype 能力和模型访问权限，并把这些信息写入 `request.json`。

## 目录布局

```text
/content/lora-asr/              当前代码
/content/cache/                 本地模型、16 kHz 音频和本地 manifest
/content/run/                   当前运行临时目录
<DRIVE_ROOT>/qwen3-asr/         manifests、runs、releases、environment
```

Drive 上的 `runs/<run_id>/` 产物以 `04_training_plan.md` 的运行目录为准。`07` 不再另列一套文件名。

## 启动顺序

1. 挂载 Drive 并确认根目录。
2. 拉取固定 Git revision，安装锁定依赖。
3. 记录 Python、GPU、CUDA、包版本和 Git revision。
4. 把中文 manifest、音频、模型缓存和 `latest.json` 指向的完整 checkpoint 同步到本地。音频在本地转为 16 kHz，并写出本地路径 manifest。
5. 校验 hash、配置和恢复状态，再启动 smoke 或正式阶段。正式 run 拒绝未提交修改。

实现后 notebook 只编排下面的 CLI，不把状态藏在单元格：

```bash
python3 scripts/colab_bootstrap.py --config configs/train/<stage>_pilot.yaml \
  --drive-root /content/drive/MyDrive/qwen3-asr --run-id <run_id> --seed <seed>
python3 scripts/manifest_validate.py --manifest manifests/train.jsonl --check-audio
python3 scripts/manifest_validate.py --manifest manifests/val.jsonl --check-audio
python3 inference/run_inference.py --config configs/inference/base_eval.yaml \
  --manifest manifests/val.jsonl --output runs/<run_id>/predictions/base_val.jsonl --mode eval
python3 evaluation/eval_cer.py --config configs/inference/base_eval.yaml \
  --manifest manifests/val.jsonl --predictions runs/<run_id>/predictions/base_val.jsonl \
  --output-dir runs/<run_id>/eval/base
```

每个命令退出后立即检查返回码和结构化状态；失败不得继续执行下一阶段。

训练进程按单张 A100、单进程设计，不使用 DDP。rollout、更新和评测按顺序复用同一张卡。冻结 reference 可以留在该进程内，不另起推理服务。每个 checkpoint 还要保存 `sampler_state`、`micro_step`、`samples_seen`、`group_visits`、当前累积梯度边界和阶段父 release hash，避免恢复后重复消费数据或重启 scheduler warmup。

## Run 产物

每个 run 的文件名、checkpoint 内容和 `latest.json` 格式见 `04_training_plan.md`。写入顺序是本地临时目录、checksum、复制到 `checkpoints/.incoming/`、读回核对，然后才写新的 `latest.json`。核对失败时状态为 `sync_failed`，不移动恢复指针。

## 测试

本合同的本地测试使用临时目录模拟半成品、截断文件、checksum 不一致和旧 `latest.json`；Colab 测试使用空 runtime、单条 clean/degraded 推理、A100 preflight 和 Phase 2–4 的一次中断恢复。通过标准是坏状态不被恢复、结构化事件可定位、完整 checkpoint 能被读回并继续至少一步。

## 断点恢复验收

从 Phase 2 起，必须模拟一次 runtime 中断，重新启动后加载最近的完整 checkpoint，继续至少 1 step，并确认 global step、micro step、optimizer、scheduler、sampler 游标、manifest hash、LoRA 配置、父 release hash 和 eval 解码 hash 一致。RNG 状态必须能加载；采样 token 逐位相同不是通过条件。Phase 0 只验证推理产物写入 Drive 后能够读回。

## 中断、清理与回滚

状态只允许从 `created → preflight → running → paused/completed/failed/oom/blocked_*` 流转；数据失败为 `data_failed`，Drive 同步失败为 `sync_failed`，恢复核对失败为 `resume_rejected`，门禁失败为 `gate_failed`，发布后才是 `published`。正常结束必须先同步并读回最后一个完整 checkpoint，再写 `gate.json`、`status=completed`；通过 gate 并复制 release 后写 `published`。半成品和 `.incoming` 目录保留用于诊断但不能恢复。

清理只删除本地 `/content/cache` 和 `/content/run` 的可重建内容；在 Drive 上只按 `keep_last` 删除已确认不被 `latest.json`、release 或 run report 引用的 checkpoint。任何回滚都指向上一份 `published` release，新尝试使用新 `run_id`，不覆盖原 run。

## 验收与影响

Colab 合同的验收是：全新 runtime 能安装锁定依赖、记录环境指纹、加载模型、运行 clean/degraded baseline、把结果写入并从 Drive 读回；Phase 2–4 还必须完成一次中断恢复。该合同要求训练代码支持本地 staging、checksum、原子恢复指针和结构化状态，不改变 Drive 中已有历史 run。
