#!/usr/bin/env bash
# Wait for the already-running rl_pilot_v7 training job, then score every saved
# checkpoint on the frozen 2,867-row validation set and write gate JSON.
# Safe to launch while training is still in Step 0: it will not start a second job.
set -u

REPO="/data/mega-asr/repo"
RUN="${RL_RUN_DIR:-/data/mega-asr/runs/rl_pilot_v7}"
TRAIN_LOG="${RL_TRAIN_LOG:-/data/mega-asr/logs/rl_pilot_v7.log}"
RL_CONFIG="${RL_CONFIG:-configs/train/qwen3_asr_rl.yaml}"
RL_MAX_STEPS="${RL_MAX_STEPS:-30}"
RL_SAVE_STEPS="${RL_SAVE_STEPS:-10}"
RL_EVAL_STEPS="${RL_EVAL_STEPS:-10}"
STATUS_FILE="${RUN}/pipeline_followup.json"
PYTHON="/data/mega-asr/venv/bin/python"
TORCHRUN="/data/mega-asr/venv/bin/torchrun"
MANIFEST="/data/mega-asr/manifests/validation.jsonl"
RL_VAL="/data/mega-asr/manifests/rl_val_pool.jsonl"
BASE_METRICS="/data/mega-asr/runs/eval_validation_base/predictions_eval/metrics.json"
BASE_PREDS="/data/mega-asr/runs/eval_validation_base/predictions.jsonl"
DPO_METRICS="/data/mega-asr/runs/dpo_pilot_v2/predictions_eval/metrics.json"
DPO_PREDS="/data/mega-asr/runs/dpo_pilot_v2/predictions.jsonl"

mkdir -p "$RUN" /data/mega-asr/logs
cd "$REPO"

write_status() {
  local status="$1"
  local phase="$2"
  local detail="${3:-}"
  FOLLOWUP_STATUS="$STATUS_FILE" STATUS_VALUE="$status" PHASE_VALUE="$phase" DETAIL_VALUE="$detail" \
    "$PYTHON" - << 'PY'
import datetime
import json
import os
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
path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

fail() {
  echo "FAILED: $*" >&2
  write_status "FAILED" "error" "$*"
  exit 1
}

training_alive() {
  pgrep -f "[t]rain/train_rl.py" >/dev/null
}

training_finished() {
  [ -f "$TRAIN_LOG" ] && grep -q "GRPO RL Training finished" "$TRAIN_LOG"
}

start_training_if_needed() {
  if training_alive || training_finished; then
    return 0
  fi
  echo "No training process for ${RUN} and no finish marker in ${TRAIN_LOG}. Starting training."
  export GIT_COMMIT="${GIT_COMMIT:-7b66936-dirty}"
  export PYTHONUNBUFFERED=1
  nohup "$TORCHRUN" --standalone --nproc_per_node=4 train/train_rl.py \
    --manifest /data/mega-asr/manifests/pilot_rl.jsonl \
    --val-manifest "$RL_VAL" \
    --config "$RL_CONFIG" \
    --output-dir "$RUN" \
    --sample-strategy balanced \
    --max-steps "$RL_MAX_STEPS" \
    --save-steps "$RL_SAVE_STEPS" \
    --eval-steps "$RL_EVAL_STEPS" \
    >> "$TRAIN_LOG" 2>&1 &
  echo $! > "${RUN}/train.pid"
  sleep 5
  training_alive || fail "training process exited immediately; see ${TRAIN_LOG}"
}

wait_for_training() {
  write_status "WAITING" "training"
  start_training_if_needed
  while training_alive; do
    sleep 30
  done
  if ! training_finished; then
    fail "training process ended without the finish line in ${TRAIN_LOG}"
  fi
  echo "Training process has exited and the finish line is present."
  sleep 15
}

best_step() {
  "$PYTHON" - << PY
import json
from pathlib import Path
state_path = Path("${RUN}/pipeline_state.json")
step = ""
if state_path.is_file():
    state = json.loads(state_path.read_text(encoding="utf-8"))
    best = str(state.get("best_checkpoint") or "")
    name = best.rstrip("/").split("/")[-1]
    if name.startswith("step_"):
        step = name.split("_", 1)[1]
print(step)
PY
}

saved_steps() {
  "$PYTHON" - << PY
from pathlib import Path
root = Path("${RUN}/checkpoints")
steps = []
if root.is_dir():
    for path in root.glob("step_*"):
        if (path / "adapter").is_dir() and path.name.split("_", 1)[-1].isdigit():
            steps.append(int(path.name.split("_", 1)[-1]))
for step in sorted(set(steps)):
    print(step)
PY
}

eval_step() {
  local step="$1"
  local ckpt="${RUN}/checkpoints/step_${step}"
  local preds="${RUN}/predictions_step_${step}.jsonl"
  local eval_dir="${RUN}/predictions_step_${step}_eval"
  local gate="${RUN}/gate_step_${step}.json"
  if [ ! -d "${ckpt}/adapter" ]; then
    echo "Skip step ${step}: adapter missing"
    return 0
  fi
  if [ -f "$gate" ]; then
    echo "Step ${step} gate already exists, skipping."
    return 0
  fi
  write_status "EVALUATING" "validation_step_${step}"
  echo "Evaluating validation.jsonl for step ${step}"
  "$PYTHON" inference/parallel_inference.py \
    --manifest "$MANIFEST" \
    --output "$preds" \
    --model-id /data/mega-asr/runs/dpo_pilot_v2/merged_base \
    --adapter-dir "${ckpt}/adapter" \
    --method "rl_pilot_v7_step_${step}" \
    --gpus 0 1 2 3 \
    --eval
  [ -f "${eval_dir}/metrics.json" ] || fail "metrics missing after step ${step} inference"
  "$PYTHON" evaluation/verify_gate.py \
    --base-metrics "$BASE_METRICS" \
    --base-predictions "$BASE_PREDS" \
    --dpo-metrics "$DPO_METRICS" \
    --dpo-predictions "$DPO_PREDS" \
    --pilot-metrics "${eval_dir}/metrics.json" \
    --pilot-predictions "$preds" \
    --manifest "$MANIFEST" \
    --rl-val-manifest "$RL_VAL" \
    --rl-loss-log "${RUN}/loss_log.jsonl" \
    --rl-step "$step" \
    --stage rl_pilot \
    --held-out-decode sample \
    --max-robust-regression 0.0 \
    --max-empty-rate 0.002 \
    --output "$gate"
  echo "Gate for step ${step} written to ${gate}"
}

wait_for_training

best="$(best_step)"
mapfile -t steps < <(saved_steps)
if [ "${#steps[@]}" -eq 0 ]; then
  fail "no saved checkpoints under ${RUN}/checkpoints"
fi

ordered=()
if [ -n "$best" ]; then
  ordered+=("$best")
fi
for step in "${steps[@]}"; do
  if [ "$step" != "$best" ]; then
    ordered+=("$step")
  fi
done

echo "Evaluation order: ${ordered[*]} (best=${best:-unknown})"
for step in "${ordered[@]}"; do
  eval_step "$step"
done

gate_summary="$("$PYTHON" - << PY
import json
from pathlib import Path
run = Path("${RUN}")
summary = {}
for gate_path in sorted(run.glob("gate_step_*.json")):
    step = gate_path.stem.split("_")[-1]
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    summary[step] = payload.get("gate_status", "UNKNOWN")
print(json.dumps(summary, ensure_ascii=False))
PY
)"

write_status "DONE" "gates_ready" "best_step=${best}; gates=${gate_summary}"
echo "DONE: rl_pilot_v7 follow-up complete. best_step=${best} gates=${gate_summary}"
