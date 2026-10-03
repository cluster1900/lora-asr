#!/usr/bin/env bash
# v11: same anchored degraded GRPO as v10, learning rate 2e-5.
# Probe first. After each chunk exits, score validation.jsonl before resuming.
# Stop the next chunk when Robust increases by >= 0.0005 or the gate is PASSED.
set -u

REPO="/data/mega-asr/repo"
RUN="/data/mega-asr/runs/rl_pilot_v11"
PROBE="${RUN}/search_probe.json"
TRAIN_LOG="/data/mega-asr/logs/rl_pilot_v11.log"
PIPE_LOG="/data/mega-asr/logs/rl_pilot_v11_pipeline.log"
STATUS_FILE="${RUN}/pipeline_followup.json"
PYTHON="/data/mega-asr/venv/bin/python"
TORCHRUN="/data/mega-asr/venv/bin/torchrun"
MANIFEST="/data/mega-asr/manifests/pilot_rl.jsonl"
VAL_MANIFEST="/data/mega-asr/manifests/rl_val_pool.jsonl"
WER_MANIFEST="/data/mega-asr/manifests/validation.jsonl"
CONFIG="configs/train/qwen3_asr_rl_v11.yaml"

mkdir -p "$RUN" /data/mega-asr/logs
cd "$REPO"

if ps -eo args | awk '/python/ && /train\/train_rl.py/ && !/awk/ { found=1 } END { exit !found }'; then
  echo "train_rl.py is already running; not starting v11" | tee -a "$PIPE_LOG"
  exit 1
fi

write_status() {
  FOLLOWUP_STATUS="$STATUS_FILE" STATUS_VALUE="$1" PHASE_VALUE="$2" DETAIL_VALUE="${3:-}" \
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

common_args=(
  --manifest "$MANIFEST"
  --val-manifest "$VAL_MANIFEST"
  --wer-manifest "$WER_MANIFEST"
  --config "$CONFIG"
  --output-dir "$RUN"
  --sample-strategy degraded
)

write_status "WAITING" "probe"
echo "Starting search probe"
"$TORCHRUN" --standalone --nproc_per_node=4 train/train_rl.py \
  "${common_args[@]}" \
  --probe-only \
  >> "$TRAIN_LOG" 2>&1
if [ ! -f "$PROBE" ]; then
  write_status "FAILED" "probe" "search_probe.json missing"
  exit 1
fi
decision=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("decision",""))' "$PROBE")
mass=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("projected_train_reward_mass",""))' "$PROBE")
echo "Probe decision=${decision} mass=${mass}"
if [ "$decision" != "GO_GRPO" ]; then
  write_status "BLOCKED" "probe" "decision=${decision} mass=${mass}"
  echo "Probe did not return GO_GRPO. Not starting the 12-step run."
  exit 0
fi

run_chunk() {
  local end="$1"
  local resume="${2:-}"
  write_status "TRAINING" "chunk_${end}"
  local resume_args=()
  if [ -n "$resume" ]; then
    resume_args=(--resume-from-checkpoint "$resume")
  fi
  "$TORCHRUN" --standalone --nproc_per_node=4 train/train_rl.py \
    "${common_args[@]}" \
    --probe-decision "$PROBE" \
    --max-steps "$end" \
    --save-steps 4 \
    --eval-steps 4 \
    --no-export-merged-on-finish \
    "${resume_args[@]}" \
    >> "$TRAIN_LOG" 2>&1
}

eval_step() {
  local step="$1"
  local ckpt="${RUN}/checkpoints/step_${step}/adapter"
  local preds="${RUN}/predictions_step_${step}.jsonl"
  local gate="${RUN}/gate_step_${step}.json"
  if [ ! -d "$ckpt" ] || [ -f "$gate" ]; then
    return 0
  fi
  write_status "EVALUATING" "validation_step_${step}"
  "$PYTHON" inference/parallel_inference.py \
    --manifest "$WER_MANIFEST" \
    --output "$preds" \
    --model-id /data/mega-asr/runs/dpo_pilot_v2/merged_base \
    --adapter-dir "$ckpt" \
    --method "rl_pilot_v11_step_${step}" \
    --gpus 0 1 2 3 \
    --eval
  "$PYTHON" evaluation/verify_gate.py \
    --base-metrics /data/mega-asr/runs/eval_validation_base/predictions_eval/metrics.json \
    --base-predictions /data/mega-asr/runs/eval_validation_base/predictions.jsonl \
    --dpo-metrics /data/mega-asr/runs/dpo_pilot_v2/predictions_eval/metrics.json \
    --dpo-predictions /data/mega-asr/runs/dpo_pilot_v2/predictions.jsonl \
    --pilot-metrics "${RUN}/predictions_step_${step}_eval/metrics.json" \
    --pilot-predictions "$preds" \
    --manifest "$WER_MANIFEST" \
    --rl-val-manifest "$VAL_MANIFEST" \
    --rl-loss-log "${RUN}/loss_log.jsonl" \
    --rl-step "$step" \
    --stage rl_pilot \
    --held-out-decode greedy \
    --max-robust-regression 0.0 \
    --max-empty-rate 0.002 \
    --output "$gate"
}

# Prints "PASSED" or "CONTINUE" or "STOP_ROBUST". Missing gate is an error.
gate_action() {
  local gate="$1"
  "$PYTHON" - "$gate" << 'PY'
import json, sys
gate = json.load(open(sys.argv[1], encoding="utf-8"))
status = str(gate.get("gate_status", ""))
robust = float(gate["metrics"]["robust_error_rate_increase"])
if status == "PASSED":
    print("PASSED")
elif robust >= 0.0005:
    print("STOP_ROBUST")
else:
    print("CONTINUE")
PY
}

run_chunk 4
state_status=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status",""))' "${RUN}/pipeline_state.json")
echo "After chunk 4: ${state_status}"
eval_step 4 || echo "eval step 4 failed"
# The lines above already ran in the 2026-09-29 process before this tail was
# spliced in. This helper has to live here: that process will not re-read
# earlier lines.
prediction_rows_ok() {
  local step="$1"
  local preds="${RUN}/predictions_step_${step}.jsonl"
  [ -f "$preds" ] || return 1
  "$PYTHON" - "$preds" << 'PY'
import sys
count = sum(1 for line in open(sys.argv[1], encoding="utf-8") if line.strip())
sys.exit(0 if count == 2867 else 1)
PY
}

action="HOLD"
if ! prediction_rows_ok 4 || [ ! -f "${RUN}/gate_step_4.json" ]; then
  write_status "FAILED" "validation_step_4" "step 4 eval incomplete"
  echo "FAILED: step 4 eval incomplete; not starting chunk 8"
  exit 1
fi
action=$(gate_action "${RUN}/gate_step_4.json") || {
  write_status "FAILED" "validation_step_4" "gate_action failed"
  exit 1
}
echo "Step 4 gate action: ${action}"

# Next chunk only when the trainer finished a save step, the checkpoint can be
# resumed, and the 2867-row gate neither passed nor failed robust retention.
continue_to_chunk() {
  local end="$1"
  local prev="$2"
  if [ "$action" != "CONTINUE" ] || [ "$state_status" != "CHUNK_DONE" ]; then
    return 0
  fi
  if [ ! -f "${RUN}/checkpoints/step_${prev}/optimizer.pt" ] || [ ! -f "${RUN}/checkpoints/step_${prev}/adapter/adapter_model.safetensors" ]; then
    write_status "FAILED" "chunk_${end}" "checkpoint step_${prev} incomplete"
    echo "FAILED: checkpoint step_${prev} incomplete"
    exit 1
  fi
  run_chunk "$end" "${RUN}/checkpoints/step_${prev}"
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    write_status "FAILED" "chunk_${end}" "torchrun exit ${rc}"
    echo "FAILED: chunk ${end} torchrun exit ${rc}"
    exit 1
  fi
  state_status=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status",""))' "${RUN}/pipeline_state.json") || {
    write_status "FAILED" "chunk_${end}" "pipeline_state missing"
    exit 1
  }
  echo "After chunk ${end}: ${state_status}"
  case "$state_status" in
    CHUNK_DONE|BLOCKED_TRANSFER|BLOCKED_SEARCH|BLOCKED_NO_WINNERS|STOPPED_KL|STOPPED_REWARD_DROP|STOPPED_ROBUST|FAILED_ZERO_VARIANCE)
      ;;
    *)
      write_status "FAILED" "chunk_${end}" "unexpected status ${state_status}"
      exit 1
      ;;
  esac
  local score_step="$end"
  if [ "$state_status" != "CHUNK_DONE" ]; then
    score_step=$("$PYTHON" "${REPO}/scripts/score_rl_greedy_held_out.py" --resolve-step --run-dir "$RUN") || {
      write_status "FAILED" "validation_stop" "saved checkpoint incomplete"
      echo "FAILED: saved checkpoint incomplete"
      exit 1
    }
  fi
  if [ -d "${RUN}/checkpoints/step_${score_step}/adapter" ]; then
    bash "${REPO}/scripts/score_rl_pilot_v11_checkpoint.sh" "$score_step" || {
      write_status "FAILED" "validation_step_${score_step}" "score script failed"
      echo "FAILED: step ${score_step} score script failed"
      exit 1
    }
    if ! prediction_rows_ok "$score_step" || [ ! -f "${RUN}/gate_step_${score_step}.json" ]; then
      write_status "FAILED" "validation_step_${score_step}" "eval incomplete"
      echo "FAILED: step ${score_step} eval incomplete"
      exit 1
    fi
    action=$(gate_action "${RUN}/gate_step_${score_step}.json") || exit 1
    echo "Step ${score_step} gate action: ${action}"
  fi
  if [ "$state_status" != "CHUNK_DONE" ] && [ "$action" != "PASSED" ] && [ "$action" != "STOP_ROBUST" ]; then
    action="STOP_TRAINER"
  fi
}

continue_to_chunk 8 4
continue_to_chunk 12 8

if [ "$action" = "STOP_ROBUST" ]; then
  write_status "DONE" "gates_ready" "train_status=STOPPED_ROBUST"
  echo "DONE: rl_pilot_v11 stopped because robust increased"
  exit 0
fi
if [ "$action" = "PASSED" ]; then
  write_status "DONE" "gates_ready" "train_status=CANDIDATE"
  echo "DONE: rl_pilot_v11 gate PASSED; weights stay in this run"
  exit 0
fi
if [ "$action" = "STOP_TRAINER" ]; then
  write_status "DONE" "gates_ready" "train_status=${state_status}"
  echo "DONE: rl_pilot_v11 trainer status ${state_status}"
  exit 0
fi

write_status "DONE" "gates_ready" "train_status=${state_status} gate_action=${action}"
echo "DONE: rl_pilot_v11 follow-up train_status=${state_status} gate_action=${action}"
