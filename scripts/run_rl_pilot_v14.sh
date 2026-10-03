#!/usr/bin/env bash
# Continue rl_pilot_v12 step 4 at the same 1e-5 raw-gap settings.
# If step 8 is BLOCKED_TRANSFER and greedy reward is still non-negative,
# start rl_pilot_v15 from the DPO champion at 2e-5 with the same advantage.
# Never write v10, v11, v12, v13, or the champion directory.
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
V12_CKPT="${V12_RUN}/checkpoints/step_4"
V14_RUN="/data/mega-asr/runs/rl_pilot_v14"
V15_RUN="/data/mega-asr/runs/rl_pilot_v15"
V14_CONFIG="configs/train/qwen3_asr_rl_v14.yaml"
V15_CONFIG="configs/train/qwen3_asr_rl_v15.yaml"
PIPE_LOG="/data/mega-asr/logs/rl_pilot_v14_pipeline.log"

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
    "$V12_RUN"|"/data/mega-asr/runs/rl_pilot_v10"|"/data/mega-asr/runs/rl_pilot_v11"|"/data/mega-asr/runs/rl_pilot_v13"|"$CHAMPION"|"${CHAMPION}"/*)
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
  local config="$1"
  local expected_lr="$2"
  "$PYTHON" - "$REPO/$config" "$expected_lr" << 'PY'
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8")
lr = sys.argv[2]
required = (
    f"learning_rate: {lr}",
    "mode: raw_gap",
    "beta: 0.04",
    "max_raw_kl: 5.0e-4",
    "max_steps: 12",
)
missing = [item for item in required if item not in text]
if missing:
    raise SystemExit("config missing " + ", ".join(missing))
PY
}

run_chunk() {
  local run="$1"
  local config="$2"
  local end="$3"
  local resume="${4:-}"
  refuse_run "$run"
  local log="/data/mega-asr/logs/$(basename "$run").log"
  local resume_args=()
  if [ -n "$resume" ]; then
    resume_args=(--resume-from-checkpoint "$resume")
  fi
  write_status "$run" "TRAINING" "chunk_${end}"
  "$TORCHRUN" --standalone --nproc_per_node=4 train/train_rl.py \
    --manifest "$MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --wer-manifest "$WER_MANIFEST" \
    --config "$config" \
    --output-dir "$run" \
    --sample-strategy degraded \
    --probe-decision "${run}/search_probe.json" \
    --max-steps "$end" \
    --save-steps 4 \
    --eval-steps 4 \
    --no-export-merged-on-finish \
    "${resume_args[@]}" \
    >> "$log" 2>&1
}

score_step() {
  local run="$1"
  local config="$2"
  local step="$3"
  write_status "$run" "EVALUATING" "validation_step_${step}"
  RL_RUN_DIR="$run" RL_CONFIG="${REPO}/${config}" RL_METHOD="$(basename "$run")_step_${step}" \
    bash "${REPO}/scripts/score_rl_pilot_checkpoint.sh" "$step"
}

decide() {
  local run="$1"
  local step="$2"
  local allow_switch="$3"
  "$PYTHON" "${REPO}/train/rl_v14_decision.py" \
    --run-dir "$run" \
    --step "$step" \
    --horizon 12 \
    --allow-switch "$allow_switch"
}

export_passed() {
  local run="$1"
  local config="$2"
  local step="$3"
  local dest="${run}/merged_base"
  refuse_run "$dest"
  if [[ "$dest" == /data/mega-asr/runs/dpo_pilot_v2* ]]; then
    echo "REFUSE: export destination is the champion"
    exit 1
  fi
  "$PYTHON" train/train_rl.py \
    --export-merged \
    --config "$config" \
    --checkpoint-dir "${run}/checkpoints/step_${step}" \
    --output-dir "$dest"
}

finish_action() {
  local run="$1"
  local config="$2"
  local step="$3"
  local action="$4"
  if [ "$action" = "passed" ]; then
    export_passed "$run" "$config" "$step" || {
      write_status "$run" "FAILED" "export" "export failed"
      exit 1
    }
    write_status "$run" "DONE" "gates_ready" "train_status=PASSED"
    echo "DONE: $(basename "$run") gate PASSED"
    return 0
  fi
  write_status "$run" "DONE" "gates_ready" "action=${action}"
  echo "DONE: $(basename "$run") action ${action}"
}

prepare_v14() {
  local required
  for required in \
    adapter/adapter_model.safetensors \
    optimizer.pt \
    scheduler.pt \
    training_state.json \
    rng_state_rank_0.pt \
    rng_state_rank_1.pt \
    rng_state_rank_2.pt \
    rng_state_rank_3.pt
  do
    if [ ! -f "${V12_CKPT}/${required}" ]; then
      echo "REFUSE: missing ${V12_CKPT}/${required}"
      exit 1
    fi
  done
  if [ ! -f "${V12_RUN}/loss_log.jsonl" ] || [ ! -f "${V12_RUN}/search_probe.json" ]; then
    echo "REFUSE: v12 loss log or probe is missing"
    exit 1
  fi
  mkdir -p "$V14_RUN"
  cp -p "${V12_RUN}/loss_log.jsonl" "${V14_RUN}/loss_log.jsonl"
  cp -p "${V12_RUN}/search_probe.json" "${V14_RUN}/search_probe.json"
  if ! "$PYTHON" - << 'PY'
import json
from pathlib import Path
run = Path("/data/mega-asr/runs/rl_pilot_v14")
rows = [json.loads(line) for line in (run / "loss_log.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
step0 = [
    row for row in rows
    if row.get("global_step") == 0
    and row.get("val_decode") == "greedy"
    and row.get("val_eval_scope") == "Full Held-out"
]
if len(step0) != 1 or step0[0].get("val_mean_reward") != 0.8773:
    raise SystemExit("v14 loss log is missing the v12 step-0 reward 0.8773")
mass = sum(float(row["reward_mass_in_step"]) for row in rows if "reward_mass_in_step" in row)
if abs(mass - 10.603235) > 1e-6:
    raise SystemExit(f"v14 copied reward mass {mass} is not 10.603235")
probe = json.loads((run / "search_probe.json").read_text(encoding="utf-8"))
if probe.get("decision") != "GO_GRPO":
    raise SystemExit("v14 probe is not GO_GRPO")
(run / "source.json").write_text(json.dumps({
    "source_run": "/data/mega-asr/runs/rl_pilot_v12",
    "resume_checkpoint": "/data/mega-asr/runs/rl_pilot_v12/checkpoints/step_4",
    "step0_reward": 0.8773,
    "copied_reward_mass": mass,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("v14 baseline ok")
PY
  then
    rm -rf "$V14_RUN"
    echo "REFUSE: v14 baseline check failed"
    exit 1
  fi
}

run_scored_chunks() {
  local run="$1"
  local config="$2"
  local allow_switch="$3"
  local first_end="$4"
  local resume="${5:-}"
  run_chunk "$run" "$config" "$first_end" "$resume"
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    write_status "$run" "FAILED" "chunk_${first_end}" "torchrun exit ${rc}"
    echo "FAILED: $(basename "$run") chunk ${first_end} exit ${rc}"
    exit 1
  fi
  local step action
  step=$("$PYTHON" "${REPO}/scripts/score_rl_greedy_held_out.py" --resolve-step --run-dir "$run") || {
    write_status "$run" "FAILED" "validation" "saved checkpoint incomplete"
    exit 1
  }
  score_step "$run" "$config" "$step" || {
    write_status "$run" "FAILED" "validation_step_${step}" "score failed"
    exit 1
  }
  action=$(decide "$run" "$step" "$allow_switch") || {
    write_status "$run" "FAILED" "switch" "decision failed"
    exit 1
  }
  echo "$(basename "$run") step ${step} action ${action}"
  if [ "$action" = "continue" ]; then
    local next_end=8
    if [ "$step" -ge 8 ]; then
      next_end=12
    fi
    run_chunk "$run" "$config" "$next_end" "${run}/checkpoints/step_${step}"
    rc=$?
    if [ "$rc" -ne 0 ]; then
      write_status "$run" "FAILED" "chunk_${next_end}" "torchrun exit ${rc}"
      exit 1
    fi
    step=$("$PYTHON" "${REPO}/scripts/score_rl_greedy_held_out.py" --resolve-step --run-dir "$run") || exit 1
    score_step "$run" "$config" "$step" || exit 1
    action=$(decide "$run" "$step" "no") || exit 1
    echo "$(basename "$run") step ${step} action ${action}"
    if [ "$action" = "continue" ] && [ "$step" -lt 12 ]; then
      run_chunk "$run" "$config" 12 "${run}/checkpoints/step_${step}"
      rc=$?
      if [ "$rc" -ne 0 ]; then
        write_status "$run" "FAILED" "chunk_12" "torchrun exit ${rc}"
        exit 1
      fi
      step=$("$PYTHON" "${REPO}/scripts/score_rl_greedy_held_out.py" --resolve-step --run-dir "$run") || exit 1
      score_step "$run" "$config" "$step" || exit 1
      action=$(decide "$run" "$step" "no") || exit 1
      echo "$(basename "$run") step ${step} action ${action}"
    fi
  fi
  LAST_ACTION="$action"
  LAST_STEP="$step"
}

assert_config "$V14_CONFIG" "1.0e-5"
assert_config "$V15_CONFIG" "2.0e-5"

LAST_ACTION=""
LAST_STEP=""

if [ -f "${V14_RUN}/switch_decision.json" ]; then
  action=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("action",""))' "${V14_RUN}/switch_decision.json")
  echo "Existing v14 decision: ${action}"
  if [ "$action" = "continue" ]; then
    echo "REFUSE: v14 already has a continue decision; not starting a second chunk from a rerun"
    exit 1
  fi
  if [ "$action" = "switch" ] && [ ! -f "${V15_RUN}/pipeline_state.json" ]; then
    mkdir -p "$V15_RUN"
    cp -p "${V12_RUN}/search_probe.json" "${V15_RUN}/search_probe.json"
    run_scored_chunks "$V15_RUN" "$V15_CONFIG" "no" 4 ""
    finish_action "$V15_RUN" "$V15_CONFIG" "$LAST_STEP" "$LAST_ACTION"
    echo "driver finished v15 action ${LAST_ACTION}"
    exit 0
  fi
  echo "driver finished existing decision ${action}"
  exit 0
fi

if [ -e "$V14_RUN" ]; then
  echo "REFUSE: rl_pilot_v14 already exists without a switch decision"
  exit 1
fi

prepare_v14
run_scored_chunks "$V14_RUN" "$V14_CONFIG" "yes" 8 "$V12_CKPT"
echo "v14 scored step ${LAST_STEP} action ${LAST_ACTION}"
if [ "$LAST_ACTION" = "switch" ]; then
  v14_step="$LAST_STEP"
  finish_action "$V14_RUN" "$V14_CONFIG" "$v14_step" "switch"
  mkdir -p "$V15_RUN"
  cp -p "${V12_RUN}/search_probe.json" "${V15_RUN}/search_probe.json"
  run_scored_chunks "$V15_RUN" "$V15_CONFIG" "no" 4 ""
  finish_action "$V15_RUN" "$V15_CONFIG" "$LAST_STEP" "$LAST_ACTION"
  echo "driver finished v15 action ${LAST_ACTION}"
  exit 0
fi
finish_action "$V14_RUN" "$V14_CONFIG" "$LAST_STEP" "$LAST_ACTION"
echo "driver finished v14 action ${LAST_ACTION}"
exit 0
