# Colab + Drive 与 Training Gym 思想设计

## 目标

本项目使用 Colab 的单张 A100 完成中文 Qwen3-ASR-0.6B 后训练。Google Drive 负责持久化、版本化和恢复；Colab 本地磁盘负责实际加载音频、模型和 checkpoint。设计借鉴 Modal Training Gym 将 dataset、generate/rollout、reward、evaluation、recipe 和 run metadata 分开的思路，但不依赖其 SDK、Modal 后端或私有 Qwen wrapper。

## 角色映射

| Training Gym 思想 | 本项目模块 | ASR 具体含义 |
| --- | --- | --- |
| DatasetConfig | `scripts/manifest_*` | 固定中文音频、转写、scenario、split 和 source ID |
| generate/rollout | `inference/` 与 `train/` 的 rollout 接口（实现时登记具体模块） | 给同一音频生成候选转写，保存 decode 参数和 raw output |
| custom reward | `train/` 的 reward 接口（实现时登记具体模块） | 以 CER 为主，结合格式、空输出、重复输出和长度惩罚 |
| eval function | `evaluation/` | 固定 held-out manifest，按 scenario 聚合 CER 和健康度 |
| TrainConfig/recipe | `configs/` | 模型、LoRA、batch、梯度累积、rollout 数和停止条件 |
| TrainResult/observability | Drive 的 `runs/<run_id>/` | 状态、指标、checkpoint 索引、环境指纹和错误样本 |

ASR 的 rollout 不需要多轮环境反馈：同一输入音频可以生成 greedy、temperature 或不同候选，reward 直接由参考文本和候选转写计算。若之后需要噪声级别或解码策略作为环境变量，仍必须写入样本 metadata，不能藏在 notebook 状态里。

## Drive 目录合同

```text
<DRIVE_ROOT>/qwen3-asr/
  repo_snapshots/<git_sha>/
  models/Qwen3-ASR-0.6B/
  manifests/<manifest_id>/
    train.jsonl
    val.jsonl
    test.jsonl
    manifest.json
    sha256.json
  runs/<run_id>/
    request.json
    environment.json
    resource_report.json
    resolved_config.yaml
    manifest_sha256.json
    status.json
    training_state.json         # 可选的 run 摘要；可恢复副本在 checkpoint 内
    metrics.jsonl
    events.jsonl
    predictions/
    eval/
    rollouts/
    gate.json
    run_report.md
    checkpoints/
      .incoming/
      step_000010/
    latest.json
  releases/
    base/
    sft/<run_id>/
    dpo/<run_id>/
    rl/<run_id>/
```

`runs/<run_id>/` 内部文件、checkpoint 名称和 `latest.json` 以 `04_training_plan.md` 为准。`run_id` 格式为 `YYYYMMDDTHHMMSS_<stage>_06b_zh_seed<seed>`。已有 run 不覆盖；恢复必须显式指定 `--run-id`。

每个 run 的 `request.json` 固定请求参数，`environment.json` 固定执行环境，`resolved_config.yaml` 固定训练参数；三者以及 manifest、父 release 的 SHA-256 都写进 `training_state.json`。`events.jsonl` 是追加式生命周期日志，至少包含时间、状态、step、事件类型、样本/错误计数和 traceback 摘要；`metrics.jsonl` 是数值日志，不能通过重写历史行修正结果。

## Colab 生命周期

### 启动

1. 挂载 Drive，创建或读取 `run_id`。
2. 固定 Git SHA；若当前 checkout 有未提交修改，写入状态并拒绝正式 run。
3. 安装锁定依赖，记录 Python、PyTorch、CUDA、GPU、`qwen-asr`、Transformers 和 pip freeze。
4. 从 Drive 同步 manifest、音频、模型缓存和 `latest.json` 指向的完整 checkpoint。音频和模型进入 `/content/cache`，运行临时文件进入 `/content/run`。
5. 计算本地文件 checksum，校验配置、manifest、模型名和训练阶段。正式 run 拒绝未提交修改；`allow_dirty` 只允许不能发布的 smoke。
6. 先跑 1 个 batch 的 preflight，再进入 smoke 或正式训练。preflight 失败写 `preflight_failed`，不更新 `latest.json`。

### 运行

- 音频、模型和 checkpoint 从本地磁盘读取；Drive 不参与每 step I/O。
- 每 `save_steps` 写入本地临时 checkpoint，生成 checksum，复制到 Drive 暂存目录 `checkpoints/.incoming/<step_name>/` 并就位至正式目录 `checkpoints/<step_name>/`。
- 从 Drive 读回核对 `sha256.json` 与各文件 checksum，核对通过后原子写入新的 `latest.json`。Drive 改名不作为原子性保证，未通过核对绝不移动恢复指针。
- 每个 step 或固定间隔追加 `metrics.jsonl` 和 `events.jsonl`，不重写整个大文件。
- 单张 A100 上按“rollout → 训练 → 评测”分时执行；不用 DDP，不启动常驻的第二个推理服务。

### 结束与中断

- 正常结束：先同步并读回最后一个完整 checkpoint，再写最终 metrics 和 gate；gate 通过且 release 复制完成后才写 `status=published`，否则写 `completed` 或 `gate_failed`。
- Colab 中断：启动时只认 `latest.json` 指向且文件齐全的 checkpoint；半成品目录一律忽略。
- Drive 同步失败：状态写为 `sync_failed`，不更新 `latest.json`，下一次启动重新同步。
- 从 Phase 2 起，恢复后至少继续 1 step，并比较 global step、optimizer、scheduler、manifest hash、config hash 和 eval 解码 hash。RNG 状态要能加载；采样 token 逐位相同不是通过条件。

## 阶段运行策略

### SFT

smoke 从 train 里取同时包含 clean 和一个 degraded scenario 的小切片，跑 5–20 step，确认显存、loss、保存和恢复。pilot 使用包含五类场景的 train split，门禁只读 val。单张 A100 采用 `04_training_plan.md` 里的 LoRA、BF16、梯度累积和音频/文本长度上限。

### DPO

先按 `04_training_plan.md` 离线生成并审计 chosen/rejected，使用冻结 SFT release 预计算参考 logprob。trainer 训练时直接读取 pair manifest 中的离线 logprob，节省单卡显存。pair manifest 和 reference logprob 的 hash 写入 run。DPO 不与 RL rollout 同时运行。

### RL

以 DPO release 为 policy 初始值。一个训练 step 的生命周期是：载入本地 batch → 用 `decode.rollout` 生成 G 条候选（1 greedy + 3 采样） → 用统一 reward 打分（含截断与空/重复惩罚） → 组内减均值得到 advantage（全组无方差时整组 loss 归零） → 记录混合采样上的 `k3_proxy_train` 并计算 token 平均 policy loss → 写指标 → 按保存间隔 checkpoint。held-out 使用 `decode.eval` 的 greedy，读取 val，不读取 rollout 温度，也不读取训练样本；`raw_kl` 在固定 val 前缀上对词表求精确条件 KL。算法细节以 `04_training_plan.md` 为准。

## 必须先实现的最小组件

1. `scripts/colab_bootstrap.py`：挂载、环境指纹、Git SHA、目录初始化。
2. `scripts/drive_sync.py`：本地 staging、checksum、读回核对后的 checkpoint 同步和恢复索引。
3. `scripts/manifest_validate.py`：中文 manifest schema、source 泄漏和五类 scenario 检查。
4. `inference/run_inference.py`：单条/批量推理和失败 JSONL。
5. `evaluation/eval_cer.py`：CER、scenario 聚合、空/重复/超长输出统计。
6. `notebooks/00_bootstrap_baseline.ipynb`：只编排上述 CLI，验证从空 runtime 到 baseline 结果。

## 验收标准

完成 Phase 0 必须满足：从一个全新的 Colab runtime，挂载同一 Drive，固定同一 Git SHA 和 seed，能够在单张 A100 上重新生成环境摘要、加载 `Qwen/Qwen3-ASR-0.6B`、处理至少一条中文 clean 和一条 degraded 音频，并把 predictions 和 metrics 写入 Drive 后读回。optimizer checkpoint 的保存与恢复属于 Phase 2。

## 测试与失败处理

本设计至少需要三类测试：一是本地用临时目录验证 staging、文件 hash、`.incoming` 隔离和 `latest.json` 不指向半成品；二是 Colab preflight 验证 A100、BF16、模型加载、manifest 本地化和单条推理；三是 Phase 2–4 的中断恢复 smoke，验证 global step、optimizer/scheduler/RNG、manifest/config/eval hash 和 reference checksum。Drive 读回或 checksum 失败时状态为 `sync_failed`，不移动恢复指针；模型/数据/API 错误写入事件并停止当前 run。

## 影响

该设计把 Drive 从训练计算路径移出，降低 runtime 重启和 Drive I/O 对 loss 的影响，但要求本地磁盘有足够空间，并要求每个 checkpoint 通过读回校验后才能恢复。Training Gym 只提供模块边界和可观测性思想；项目不依赖其 SDK、后端或私有 wrapper，所有接口仍以官方 `qwen-asr`/Transformers 和本仓库 JSONL/config 合同为准。
