#!/usr/bin/env bash
# rl_pilot_v23 resumes the read-only v22 step-8 checkpoint.
# After resume, the configured learning rate 5e-6 replaces the restored 1e-5.
# Advantage, beta, the KL ceiling, and the audio-projection freeze stay.
# Never write v22 or the champion directory.
set -u

export PYTHONUNBUFFERED=1

REPO="/data/mega-asr/repo"
PYTHON="/data/mega-asr/venv/bin/python"
TORCHRUN="/data/mega-asr/venv/bin/torchrun"
MANIFEST="/data/mega-asr/manifests/pilot_rl.jsonl"
VAL_MANIFEST="/data/mega-asr/manifests/rl_val_pool.jsonl"
WER_MANIFEST="/data/mega-asr/manifests/validation.jsonl"
CHAMPION="/data/mega-asr/runs/dpo_pilot_v2/merged_base"
V22_RUN="/data/mega-asr/runs/rl_pilot_v22"
V22_STEP8="/data/mega-asr/runs/rl_pilot_v22/checkpoints/step_8"
V23_RUN="/data/mega-asr/runs/rl_pilot_v23"
V23_CONFIG="configs/train/qwen3_asr_rl_v23.yaml"
HORIZON=24
PIPE_LOG="/data/mega-asr/logs/rl_pilot_v23_pipeline.log"

mkdir -p /data/mega-asr/logs
cd "$REPO"
exec >> "$PIPE_LOG" 2>&1
echo "driver start $$ at $(date -u +%Y-%m-%dT%H:%M:%SZ)"

busy() {
  "$PYTHON" - << 'PY'
import sys
from pathlib import Path
needles = ("train/train_rl.py", "parallel_inference.py", "score_rl_greedy_held_out.py")
for entry in Path("/proc").iterdir():
    if not entry.name.isdigit():
        continue
    try:
        cmd = (entry / "cmdline").read_bytes().replace(b"\x00", b" ").decode("utf-8", "replace")
    except OSError:
        continue
    if any(needle in cmd for needle in needles):
        sys.exit(0)
sys.exit(1)
PY
}

refuse_run() {
  local run="$1"
  case "$run" in
    "/data/mega-asr/runs/rl_pilot_v10"|"/data/mega-asr/runs/rl_pilot_v11"|"/data/mega-asr/runs/rl_pilot_v12"|"/data/mega-asr/runs/rl_pilot_v13"|"/data/mega-asr/runs/rl_pilot_v14"|"/data/mega-asr/runs/rl_pilot_v15"|"/data/mega-asr/runs/rl_pilot_v16"|"/data/mega-asr/runs/rl_pilot_v17"|"/data/mega-asr/runs/rl_pilot_v18"|"/data/mega-asr/runs/rl_pilot_v19"|"/data/mega-asr/runs/rl_pilot_v20"|"/data/mega-asr/runs/rl_pilot_v21"|"$V22_RUN"|"$CHAMPION"|"${CHAMPION}"/*)
      echo "REFUSE: run path is protected: $run"
      exit 1
      ;;
  esac
}

if busy; then
  echo "REFUSE: training or scoring is already running"
  exit 1
fi

write_status() {
  local run="$1"
  local status="$2"
  local phase="$3"
  local detail="${4:-}"
  FOLLOWUP_STATUS="${run}/pipeline_followup.json" STATUS_VALUE="$status" PHASE_VALUE="$phase" DETAIL_VALUE="$detail" \
    "$PYTHON" - << 'PY'
import datetime, json, os
from pathlib import Path
path = Path(os.environ["FOLLOWUP_STATUS"])
data = {}
if path.is_file():
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        data = {}
data["status"] = os.environ["STATUS_VALUE"]
data["phase"] = os.environ["PHASE_VALUE"]
data["detail"] = os.environ.get("DETAIL_VALUE", "")
data["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

assert_config() {
  "$PYTHON" - "$REPO/$V23_CONFIG" << 'PY'
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8")
required = (
    "learning_rate: 5.0e-6",
    "apply_learning_rate_on_resume: true",
    "mode: raw_gap",
    "local_max_relative: 0.35",
    "beta: 0.04",
    "max_raw_kl: 5.0e-4",
    "max_steps: 24",
    "train_audio_projections: false",
    "expected_total_targets: 196",
)
missing = [item for item in required if item not in text]
forbidden = (
    "learning_rate: 1.0e-5",
    "mode: unit",
    "mode: capped_gap",
    "train_audio_projections: true",
    "apply_learning_rate_on_resume: false",
)
found = [item for item in forbidden if item in text]
if missing or found:
    raise SystemExit("config missing " + ", ".join(missing) + " forbidden " + ", ".join(found))
PY
}

run_chunk() {
  local end="$1"
  local resume="${2:-}"
  refuse_run "$V23_RUN"
  local log="/data/mega-asr/logs/rl_pilot_v23.log"
  local resume_args=()
  if [ -n "$resume" ]; then
    resume_args=(--resume-from-checkpoint "$resume")
  fi
  write_status "$V23_RUN" "TRAINING" "chunk_${end}"
  "$TORCHRUN" --standalone --nproc_per_node=4 train/train_rl.py \
    --manifest "$MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --wer-manifest "$WER_MANIFEST" \
    --config "$V23_CONFIG" \
    --output-dir "$V23_RUN" \
    --sample-strategy degraded \
    --probe-decision "${V23_RUN}/search_probe.json" \
    --max-steps "$end" \
    --save-steps 4 \
    --eval-steps 4 \
    --no-export-merged-on-finish \
    "${resume_args[@]}" \
    >> "$log" 2>&1
}

score_step() {
  local step="$1"
  write_status "$V23_RUN" "EVALUATING" "validation_step_${step}"
  RL_RUN_DIR="$V23_RUN" RL_CONFIG="${REPO}/${V23_CONFIG}" RL_METHOD="rl_pilot_v23_step_${step}" \
    bash "${REPO}/scripts/score_rl_pilot_checkpoint.sh" "$step"
}

decide() {
  local step="$1"
  "$PYTHON" "${REPO}/train/rl_v16_decision.py" \
    --run-dir "$V23_RUN" \
    --step "$step" \
    --horizon "$HORIZON"
}

export_passed() {
  local step="$1"
  local dest="${V23_RUN}/merged_base"
  refuse_run "$dest"
  if [[ "$dest" == /data/mega-asr/runs/dpo_pilot_v2* ]]; then
    echo "REFUSE: export destination is the champion"
    exit 1
  fi
  "$PYTHON" train/train_rl.py \
    --export-merged \
    --config "$V23_CONFIG" \
    --checkpoint-dir "${V23_RUN}/checkpoints/step_${step}" \
    --output-dir "$dest"
}

finish_action() {
  local step="$1"
  local action="$2"
  if [ "$action" = "passed" ]; then
    export_passed "$step" || {
      write_status "$V23_RUN" "FAILED" "export" "export failed"
      exit 1
    }
    write_status "$V23_RUN" "DONE" "gates_ready" "train_status=PASSED"
    echo "DONE: rl_pilot_v23 gate PASSED"
    return 0
  fi
  write_status "$V23_RUN" "DONE" "gates_ready" "action=${action}"
  echo "DONE: rl_pilot_v23 action ${action}"
}

prepare_v23() {
  if [ -e "$V23_RUN" ]; then
    echo "REFUSE: rl_pilot_v23 already exists"
    exit 1
  fi
  if [ ! -f "${V22_RUN}/search_probe.json" ]; then
    echo "REFUSE: v22 probe is missing"
    exit 1
  fi
  if [ ! -f "${V22_STEP8}/adapter/adapter_model.safetensors" ] || [ ! -f "${V22_STEP8}/optimizer.pt" ] || [ ! -f "${V22_STEP8}/scheduler.pt" ]; then
    echo "REFUSE: v22 step 8 checkpoint is incomplete"
    exit 1
  fi
  if [ ! -f "${V22_RUN}/loss_log.jsonl" ] || [ ! -f "${V22_RUN}/switch_decision.json" ] || [ ! -f "${V22_RUN}/gate_step_8.json" ] || [ ! -f "${V22_RUN}/gate_step_11.json" ]; then
    echo "REFUSE: v22 loss log, decision, or gate is missing"
    exit 1
  fi
  if ! grep -q "apply_configured_learning_rate" "${REPO}/train/train_rl.py"; then
    echo "REFUSE: server trainer does not apply the configured learning rate on resume"
    exit 1
  fi
  mkdir -p "$V23_RUN"
  cp -p "${V22_RUN}/search_probe.json" "${V23_RUN}/search_probe.json"
  if ! "$PYTHON" - << 'PY'
import json
from pathlib import Path

run = Path("/data/mega-asr/runs/rl_pilot_v23")
v22 = Path("/data/mega-asr/runs/rl_pilot_v22")
kept = []
for line in (v22 / "loss_log.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if int(row.get("global_step", -1)) <= 8:
        kept.append(row)
(run / "loss_log.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept),
    encoding="utf-8",
)
probe = json.loads((run / "search_probe.json").read_text(encoding="utf-8"))
if probe.get("decision") != "GO_GRPO" or int(probe.get("n_prompts", 0)) != 128:
    raise SystemExit("v23 probe is not the 128-prompt GO_GRPO file")
decision = json.loads((v22 / "switch_decision.json").read_text(encoding="utf-8"))
if decision.get("action") != "stop":
    raise SystemExit("v22 decision is not stop")
if abs(float(decision["held_out_reward_improvement"]) - (-0.0004)) > 1e-6:
    raise SystemExit("v22 decision reward is not -0.0004")
gate11 = json.loads((v22 / "gate_step_11.json").read_text(encoding="utf-8"))
if float(gate11["metrics"]["robust_error_rate_increase"]) >= 0:
    raise SystemExit("v22 step 11 robust is not below DPO")
gate8 = json.loads((v22 / "gate_step_8.json").read_text(encoding="utf-8"))
metrics8 = gate8["metrics"]
if abs(float(metrics8["held_out_reward_improvement"]) - 0.0001) > 1e-6:
    raise SystemExit("v22 step 8 reward improvement is not +0.0001")
if float(metrics8["robust_error_rate_increase"]) >= 0:
    raise SystemExit("v22 step 8 robust is not below DPO")
step0 = None
step8 = None
mass = None
max_step = -1
for row in kept:
    step = int(row.get("global_step", -1))
    max_step = max(max_step, step)
    if step == 0 and row.get("val_decode") == "greedy":
        step0 = float(row["val_mean_reward"])
    if step == 8 and row.get("val_decode") == "greedy":
        step8 = float(row["val_mean_reward"])
    if "cumulative_reward_mass" in row:
        mass = float(row["cumulative_reward_mass"])
if max_step != 8:
    raise SystemExit(f"truncated log max step is {max_step}")
if step0 is None or abs(step0 - 0.8773) > 1e-6:
    raise SystemExit(f"step0 reward is {step0}")
if step8 is None or abs(step8 - 0.8774) > 1e-6:
    raise SystemExit(f"step8 reward is {step8}")
if mass is None or abs(mass - 11.838534) > 1e-6:
    raise SystemExit(f"copied mass is {mass}")
trainer = Path("/data/mega-asr/repo/train/train_rl.py").read_text(encoding="utf-8")
if trainer.count("apply_configured_learning_rate") < 2 or "apply_learning_rate_on_resume" not in trainer:
    raise SystemExit("server trainer is missing the resume learning-rate hook")
champion = Path("/data/mega-asr/runs/dpo_pilot_v2/merged_base/model.safetensors")
stat = champion.stat()
if stat.st_size != 4076190936:
    raise SystemExit(f"champion bytes changed: {stat.st_size}")
(run / "source.json").write_text(json.dumps({
    "source_model": "/data/mega-asr/runs/dpo_pilot_v2/merged_base",
    "resume_checkpoint": "/data/mega-asr/runs/rl_pilot_v22/checkpoints/step_8",
    "probe_source": "/data/mega-asr/runs/rl_pilot_v22/search_probe.json",
    "champion_bytes": stat.st_size,
    "champion_mtime": stat.st_mtime,
    "local_max_relative": 0.35,
    "learning_rate": 5.0e-6,
    "apply_learning_rate_on_resume": True,
    "advantage_mode": "raw_gap",
    "train_audio_projections": False,
    "expected_lora_targets": 196,
    "horizon": 24,
    "step0_val_reward": step0,
    "copied_reward_mass": mass,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("v23 baseline ok")
PY
  then
    rm -rf "$V23_RUN"
    echo "REFUSE: v23 baseline check failed"
    exit 1
  fi
}

score_and_decide() {
  local step
  step=$("$PYTHON" "${REPO}/scripts/score_rl_greedy_held_out.py" --resolve-step --run-dir "$V23_RUN") || {
    write_status "$V23_RUN" "FAILED" "validation" "saved checkpoint incomplete"
    exit 1
  }
  score_step "$step" || {
    write_status "$V23_RUN" "FAILED" "validation_step_${step}" "score failed"
    exit 1
  }
  local action
  action=$(decide "$step") || {
    write_status "$V23_RUN" "FAILED" "switch" "decision failed"
    exit 1
  }
  LAST_ACTION="$action"
  LAST_STEP="$step"
  echo "rl_pilot_v23 step ${step} action ${action}"
}

assert_config

if [ -f "${V23_RUN}/switch_decision.json" ]; then
  action=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("action",""))' "${V23_RUN}/switch_decision.json")
  echo "Existing v23 decision: ${action}"
  if [ "$action" = "continue" ]; then
    echo "REFUSE: v23 already has a continue decision; not starting another chunk from a rerun"
    exit 1
  fi
  echo "driver finished existing decision ${action}"
  exit 0
fi

if [ -e "$V23_RUN" ]; then
  echo "REFUSE: rl_pilot_v23 already exists without a switch decision"
  exit 1
fi

prepare_v23

LAST_ACTION=""
LAST_STEP=""
end=12
resume="$V22_STEP8"
while true; do
  run_chunk "$end" "$resume"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    write_status "$V23_RUN" "FAILED" "chunk_${end}" "torchrun exit ${rc}"
    echo "FAILED: rl_pilot_v23 chunk ${end} exit ${rc}"
    exit 1
  fi
  score_and_decide
  if [ "$LAST_ACTION" != "continue" ]; then
    break
  fi
  if [ "$LAST_STEP" -ge "$HORIZON" ]; then
    LAST_ACTION="stop"
    break
  fi
  end=$((LAST_STEP + 4))
  if [ "$end" -gt "$HORIZON" ]; then
    end=$HORIZON
  fi
  resume="${V23_RUN}/checkpoints/step_${LAST_STEP}"
done

finish_action "$LAST_STEP" "$LAST_ACTION"
echo "driver finished v23 action ${LAST_ACTION}"
