#!/usr/bin/env bash
# Fresh rl_pilot_v17 from the DPO champion.
# Learning rate 1e-5, unit advantage, winners limited to local corrections.
# Continue while Robust does not rise and reward mass stays >= 8.50.
# Horizon 48. Never resume rl_pilot_v16 and never write the champion.
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
V17_RUN="/data/mega-asr/runs/rl_pilot_v17"
V17_CONFIG="configs/train/qwen3_asr_rl_v17.yaml"
HORIZON=48
PIPE_LOG="/data/mega-asr/logs/rl_pilot_v17_pipeline.log"

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
    "$V12_RUN"|"/data/mega-asr/runs/rl_pilot_v10"|"/data/mega-asr/runs/rl_pilot_v11"|"/data/mega-asr/runs/rl_pilot_v13"|"/data/mega-asr/runs/rl_pilot_v14"|"/data/mega-asr/runs/rl_pilot_v15"|"/data/mega-asr/runs/rl_pilot_v16"|"$CHAMPION"|"${CHAMPION}"/*)
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
  "$PYTHON" - "$REPO/$V17_CONFIG" << 'PY'
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8")
required = (
    "learning_rate: 1.0e-5",
    "mode: unit",
    "local_max_relative: 0.35",
    "beta: 0.04",
    "max_raw_kl: 5.0e-4",
    "max_steps: 48",
)
missing = [item for item in required if item not in text]
if missing or "mode: raw_gap" in text:
    raise SystemExit("config missing " + ", ".join(missing))
PY
}

run_chunk() {
  local end="$1"
  local resume="${2:-}"
  refuse_run "$V17_RUN"
  local log="/data/mega-asr/logs/rl_pilot_v17.log"
  local resume_args=()
  if [ -n "$resume" ]; then
    resume_args=(--resume-from-checkpoint "$resume")
  fi
  write_status "$V17_RUN" "TRAINING" "chunk_${end}"
  "$TORCHRUN" --standalone --nproc_per_node=4 train/train_rl.py \
    --manifest "$MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --wer-manifest "$WER_MANIFEST" \
    --config "$V17_CONFIG" \
    --output-dir "$V17_RUN" \
    --sample-strategy degraded \
    --probe-decision "${V17_RUN}/search_probe.json" \
    --max-steps "$end" \
    --save-steps 4 \
    --eval-steps 4 \
    --no-export-merged-on-finish \
    "${resume_args[@]}" \
    >> "$log" 2>&1
}

score_step() {
  local step="$1"
  write_status "$V17_RUN" "EVALUATING" "validation_step_${step}"
  RL_RUN_DIR="$V17_RUN" RL_CONFIG="${REPO}/${V17_CONFIG}" RL_METHOD="rl_pilot_v17_step_${step}" \
    bash "${REPO}/scripts/score_rl_pilot_checkpoint.sh" "$step"
}

decide() {
  local step="$1"
  "$PYTHON" "${REPO}/train/rl_v17_decision.py" \
    --run-dir "$V17_RUN" \
    --step "$step" \
    --horizon "$HORIZON"
}

export_passed() {
  local step="$1"
  local dest="${V17_RUN}/merged_base"
  refuse_run "$dest"
  if [[ "$dest" == /data/mega-asr/runs/dpo_pilot_v2* ]]; then
    echo "REFUSE: export destination is the champion"
    exit 1
  fi
  "$PYTHON" train/train_rl.py \
    --export-merged \
    --config "$V17_CONFIG" \
    --checkpoint-dir "${V17_RUN}/checkpoints/step_${step}" \
    --output-dir "$dest"
}

finish_action() {
  local step="$1"
  local action="$2"
  if [ "$action" = "passed" ]; then
    export_passed "$step" || {
      write_status "$V17_RUN" "FAILED" "export" "export failed"
      exit 1
    }
    write_status "$V17_RUN" "DONE" "gates_ready" "train_status=PASSED"
    echo "DONE: rl_pilot_v17 gate PASSED"
    return 0
  fi
  write_status "$V17_RUN" "DONE" "gates_ready" "action=${action}"
  echo "DONE: rl_pilot_v17 action ${action}"
}

prepare_v17() {
  if [ -e "$V17_RUN" ]; then
    echo "REFUSE: rl_pilot_v17 already exists"
    exit 1
  fi
  if [ ! -f "${V12_RUN}/search_probe.json" ]; then
    echo "REFUSE: v12 probe is missing"
    exit 1
  fi
  if ! grep -q "local_max_relative" "${REPO}/train/train_rl.py"; then
    echo "REFUSE: server trainer has no local_max_relative"
    exit 1
  fi
  mkdir -p "$V17_RUN"
  cp -p "${V12_RUN}/search_probe.json" "${V17_RUN}/search_probe.json"
  if ! PYTHONPATH="${REPO}/train:${REPO}" "$PYTHON" - << 'PY'
import json
from pathlib import Path
from rl_local_winner import local_winner_index
from rl_pair_advantage import winner_advantage

run = Path("/data/mega-asr/runs/rl_pilot_v17")
probe = json.loads((run / "search_probe.json").read_text(encoding="utf-8"))
if probe.get("decision") != "GO_GRPO" or int(probe.get("n_prompts", 0)) != 128:
    raise SystemExit("v17 probe is not the 128-prompt GO_GRPO file")
texts = [
    "Ayr has won eight of the last nine meetings in this series.",
    "Iowa has won eight of the last nine meetings in this series.",
    "In the difficult moments we recognize our thirst for fulfillment.",
]
rewards = [0.40, 0.46, 0.95]
index, status = local_winner_index(texts, rewards, language="en")
if index != 1 or status != "update":
    raise SystemExit(f"local winner check failed: {index} {status}")
if winner_advantage(rewards[0], rewards[index], "unit") != 1.0:
    raise SystemExit("unit advantage check failed")
champion = Path("/data/mega-asr/runs/dpo_pilot_v2/merged_base/model.safetensors")
stat = champion.stat()
(run / "source.json").write_text(json.dumps({
    "source_model": "/data/mega-asr/runs/dpo_pilot_v2/merged_base",
    "probe_source": "/data/mega-asr/runs/rl_pilot_v12/search_probe.json",
    "champion_bytes": stat.st_size,
    "champion_mtime": stat.st_mtime,
    "local_max_relative": 0.35,
    "learning_rate": 1.0e-5,
    "advantage_mode": "unit",
    "horizon": 48,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("v17 baseline ok")
PY
  then
    rm -rf "$V17_RUN"
    echo "REFUSE: v17 baseline check failed"
    exit 1
  fi
}

score_and_decide() {
  local step
  step=$("$PYTHON" "${REPO}/scripts/score_rl_greedy_held_out.py" --resolve-step --run-dir "$V17_RUN") || {
    write_status "$V17_RUN" "FAILED" "validation" "saved checkpoint incomplete"
    exit 1
  }
  score_step "$step" || {
    write_status "$V17_RUN" "FAILED" "validation_step_${step}" "score failed"
    exit 1
  }
  local action
  action=$(decide "$step") || {
    write_status "$V17_RUN" "FAILED" "switch" "decision failed"
    exit 1
  }
  LAST_ACTION="$action"
  LAST_STEP="$step"
  echo "rl_pilot_v17 step ${step} action ${action}"
}

assert_config

if [ -f "${V17_RUN}/switch_decision.json" ]; then
  action=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("action",""))' "${V17_RUN}/switch_decision.json")
  echo "Existing v17 decision: ${action}"
  if [ "$action" = "continue" ]; then
    echo "REFUSE: v17 already has a continue decision; not starting another chunk from a rerun"
    exit 1
  fi
  echo "driver finished existing decision ${action}"
  exit 0
fi

if [ -e "$V17_RUN" ]; then
  echo "REFUSE: rl_pilot_v17 already exists without a switch decision"
  exit 1
fi

prepare_v17

LAST_ACTION=""
LAST_STEP=""
end=4
resume=""
while true; do
  run_chunk "$end" "$resume"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    write_status "$V17_RUN" "FAILED" "chunk_${end}" "torchrun exit ${rc}"
    echo "FAILED: rl_pilot_v17 chunk ${end} exit ${rc}"
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
  resume="${V17_RUN}/checkpoints/step_${LAST_STEP}"
done

finish_action "$LAST_STEP" "$LAST_ACTION"
echo "driver finished v17 action ${LAST_ACTION}"
