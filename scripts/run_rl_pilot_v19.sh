#!/usr/bin/env bash
# rl_pilot_v19 resumes the read-only v16 step-8 checkpoint in a new directory.
# Learning rate 1e-5, reward gap capped at 0.05, local corrections. Horizon 48.
# Continue while Robust does not rise and reward mass stays >= 8.50.
# Never write v16, v17, v18, or the champion directory.
set -u

export PYTHONUNBUFFERED=1

REPO="/data/mega-asr/repo"
PYTHON="/data/mega-asr/venv/bin/python"
TORCHRUN="/data/mega-asr/venv/bin/torchrun"
MANIFEST="/data/mega-asr/manifests/pilot_rl.jsonl"
VAL_MANIFEST="/data/mega-asr/manifests/rl_val_pool.jsonl"
WER_MANIFEST="/data/mega-asr/manifests/validation.jsonl"
CHAMPION="/data/mega-asr/runs/dpo_pilot_v2/merged_base"
V12_RUN="/data/mega-asr/runs/rl_pilot_v12"
V16_RUN="/data/mega-asr/runs/rl_pilot_v16"
V16_STEP8="${V16_RUN}/checkpoints/step_8"
V18_RUN="/data/mega-asr/runs/rl_pilot_v18"
V19_RUN="/data/mega-asr/runs/rl_pilot_v19"
V19_CONFIG="configs/train/qwen3_asr_rl_v19.yaml"
HORIZON=48
PIPE_LOG="/data/mega-asr/logs/rl_pilot_v19_pipeline.log"

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
    "$V12_RUN"|"/data/mega-asr/runs/rl_pilot_v10"|"/data/mega-asr/runs/rl_pilot_v11"|"/data/mega-asr/runs/rl_pilot_v13"|"/data/mega-asr/runs/rl_pilot_v14"|"/data/mega-asr/runs/rl_pilot_v15"|"/data/mega-asr/runs/rl_pilot_v16"|"/data/mega-asr/runs/rl_pilot_v17"|"$V18_RUN"|"$CHAMPION"|"${CHAMPION}"/*)
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
  "$PYTHON" - "$REPO/$V19_CONFIG" << 'PY'
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8")
required = (
    "learning_rate: 1.0e-5",
    "mode: capped_gap",
    "cap: 0.05",
    "local_max_relative: 0.35",
    "beta: 0.04",
    "max_raw_kl: 5.0e-4",
    "max_steps: 48",
)
missing = [item for item in required if item not in text]
if missing or "mode: unit" in text or "mode: raw_gap" in text:
    raise SystemExit("config missing " + ", ".join(missing))
PY
}

run_chunk() {
  local end="$1"
  local resume="${2:-}"
  refuse_run "$V19_RUN"
  local log="/data/mega-asr/logs/rl_pilot_v19.log"
  local resume_args=()
  if [ -n "$resume" ]; then
    resume_args=(--resume-from-checkpoint "$resume")
  fi
  write_status "$V19_RUN" "TRAINING" "chunk_${end}"
  "$TORCHRUN" --standalone --nproc_per_node=4 train/train_rl.py \
    --manifest "$MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --wer-manifest "$WER_MANIFEST" \
    --config "$V19_CONFIG" \
    --output-dir "$V19_RUN" \
    --sample-strategy degraded \
    --probe-decision "${V19_RUN}/search_probe.json" \
    --max-steps "$end" \
    --save-steps 4 \
    --eval-steps 4 \
    --no-export-merged-on-finish \
    "${resume_args[@]}" \
    >> "$log" 2>&1
}

score_step() {
  local step="$1"
  write_status "$V19_RUN" "EVALUATING" "validation_step_${step}"
  RL_RUN_DIR="$V19_RUN" RL_CONFIG="${REPO}/${V19_CONFIG}" RL_METHOD="rl_pilot_v19_step_${step}" \
    bash "${REPO}/scripts/score_rl_pilot_checkpoint.sh" "$step"
}

decide() {
  local step="$1"
  "$PYTHON" "${REPO}/train/rl_v19_decision.py" \
    --run-dir "$V19_RUN" \
    --step "$step" \
    --horizon "$HORIZON"
}

export_passed() {
  local step="$1"
  local dest="${V19_RUN}/merged_base"
  refuse_run "$dest"
  if [[ "$dest" == /data/mega-asr/runs/dpo_pilot_v2* ]]; then
    echo "REFUSE: export destination is the champion"
    exit 1
  fi
  "$PYTHON" train/train_rl.py \
    --export-merged \
    --config "$V19_CONFIG" \
    --checkpoint-dir "${V19_RUN}/checkpoints/step_${step}" \
    --output-dir "$dest"
}

finish_action() {
  local step="$1"
  local action="$2"
  if [ "$action" = "passed" ]; then
    export_passed "$step" || {
      write_status "$V19_RUN" "FAILED" "export" "export failed"
      exit 1
    }
    write_status "$V19_RUN" "DONE" "gates_ready" "train_status=PASSED"
    echo "DONE: rl_pilot_v19 gate PASSED"
    return 0
  fi
  write_status "$V19_RUN" "DONE" "gates_ready" "action=${action}"
  echo "DONE: rl_pilot_v19 action ${action}"
}

prepare_v19() {
  if [ -e "$V19_RUN" ]; then
    echo "REFUSE: rl_pilot_v19 already exists"
    exit 1
  fi
  if [ ! -f "${V12_RUN}/search_probe.json" ]; then
    echo "REFUSE: v12 probe is missing"
    exit 1
  fi
  if [ ! -f "${V16_STEP8}/adapter/adapter_model.safetensors" ] || [ ! -f "${V16_STEP8}/optimizer.pt" ]; then
    echo "REFUSE: v16 step 8 checkpoint is incomplete"
    exit 1
  fi
  if [ ! -f "${V16_RUN}/loss_log.jsonl" ] || [ ! -f "${V16_RUN}/gate_step_8.json" ]; then
    echo "REFUSE: v16 step 8 log or gate is missing"
    exit 1
  fi
  mkdir -p "$V19_RUN"
  cp -p "${V12_RUN}/search_probe.json" "${V19_RUN}/search_probe.json"
  cp -p "${V16_RUN}/loss_log.jsonl" "${V19_RUN}/loss_log.jsonl"
  cp -p "${V16_RUN}/gate_step_8.json" "${V19_RUN}/gate_step_8.json"
  if ! "$PYTHON" - << 'PY'
import json
from pathlib import Path

run = Path("/data/mega-asr/runs/rl_pilot_v19")
v16 = Path("/data/mega-asr/runs/rl_pilot_v16")
v18 = Path("/data/mega-asr/runs/rl_pilot_v18/pipeline_followup.json")
probe = json.loads((run / "search_probe.json").read_text(encoding="utf-8"))
if probe.get("decision") != "GO_GRPO" or int(probe.get("n_prompts", 0)) != 128:
    raise SystemExit("v19 probe is not the 128-prompt GO_GRPO file")
decision = json.loads((v16 / "switch_decision.json").read_text(encoding="utf-8"))
if decision.get("action") != "stop":
    raise SystemExit("v16 decision is not stop")
follow = json.loads(v18.read_text(encoding="utf-8"))
if follow.get("status") == "TRAINING":
    raise SystemExit("v18 followup still says TRAINING")
gate = json.loads((run / "gate_step_8.json").read_text(encoding="utf-8"))
metrics = gate["metrics"]
if abs(float(metrics["robust_error_rate_increase"]) - 0.000164) > 1e-9:
    raise SystemExit("copied step-8 robust does not match v16")
if abs(float(metrics["held_out_reward_improvement"]) - 0.0004) > 1e-9:
    raise SystemExit("copied step-8 reward does not match v16")
step0 = None
mass = None
for line in (run / "loss_log.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if row.get("global_step") == 0 and row.get("val_decode") == "greedy":
        step0 = float(row["val_mean_reward"])
    if "cumulative_reward_mass" in row:
        mass = float(row["cumulative_reward_mass"])
if step0 != 0.8773:
    raise SystemExit(f"step0 reward is {step0}")
if mass is None or abs(mass - 11.724898) > 1e-6:
    raise SystemExit(f"copied mass is {mass}")
champion = Path("/data/mega-asr/runs/dpo_pilot_v2/merged_base/model.safetensors")
stat = champion.stat()
if stat.st_size != 4076190936:
    raise SystemExit(f"champion bytes changed: {stat.st_size}")
(run / "source.json").write_text(json.dumps({
    "source_model": "/data/mega-asr/runs/dpo_pilot_v2/merged_base",
    "resume_checkpoint": "/data/mega-asr/runs/rl_pilot_v16/checkpoints/step_8",
    "probe_source": "/data/mega-asr/runs/rl_pilot_v12/search_probe.json",
    "champion_bytes": stat.st_size,
    "champion_mtime": stat.st_mtime,
    "local_max_relative": 0.35,
    "learning_rate": 1.0e-5,
    "advantage_mode": "capped_gap",
    "advantage_cap": 0.05,
    "horizon": 48,
    "step0_val_reward": step0,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("v19 baseline ok")
PY
  then
    rm -rf "$V19_RUN"
    echo "REFUSE: v19 baseline check failed"
    exit 1
  fi
}

score_and_decide() {
  local step
  step=$("$PYTHON" "${REPO}/scripts/score_rl_greedy_held_out.py" --resolve-step --run-dir "$V19_RUN") || {
    write_status "$V19_RUN" "FAILED" "validation" "saved checkpoint incomplete"
    exit 1
  }
  score_step "$step" || {
    write_status "$V19_RUN" "FAILED" "validation_step_${step}" "score failed"
    exit 1
  }
  local action
  action=$(decide "$step") || {
    write_status "$V19_RUN" "FAILED" "switch" "decision failed"
    exit 1
  }
  LAST_ACTION="$action"
  LAST_STEP="$step"
  echo "rl_pilot_v19 step ${step} action ${action}"
}

assert_config

if [ -f "${V19_RUN}/switch_decision.json" ]; then
  action=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("action",""))' "${V19_RUN}/switch_decision.json")
  echo "Existing v19 decision: ${action}"
  if [ "$action" = "continue" ]; then
    echo "REFUSE: v19 already has a continue decision; not starting another chunk from a rerun"
    exit 1
  fi
  echo "driver finished existing decision ${action}"
  exit 0
fi

if [ -e "$V19_RUN" ]; then
  echo "REFUSE: rl_pilot_v19 already exists without a switch decision"
  exit 1
fi

prepare_v19

LAST_ACTION=""
LAST_STEP=""
end=12
resume="$V16_STEP8"
while true; do
  run_chunk "$end" "$resume"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    write_status "$V19_RUN" "FAILED" "chunk_${end}" "torchrun exit ${rc}"
    echo "FAILED: rl_pilot_v19 chunk ${end} exit ${rc}"
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
  resume="${V19_RUN}/checkpoints/step_${LAST_STEP}"
done

finish_action "$LAST_STEP" "$LAST_ACTION"
echo "driver finished v19 action ${LAST_ACTION}"
