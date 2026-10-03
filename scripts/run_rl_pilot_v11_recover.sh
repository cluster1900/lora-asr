#!/usr/bin/env bash
# One recovery pass for rl_pilot_v11 after the driver exits.
# Reruns a missing 2867-row gate, or resumes the next 4-step chunk once.
# Does not start while training or eval is still running.
# Does not resume past a designed stop, and does not write the DPO champion.
set -u

REPO="/data/mega-asr/repo"
RUN="/data/mega-asr/runs/rl_pilot_v11"
PROBE="${RUN}/search_probe.json"
TRAIN_LOG="/data/mega-asr/logs/rl_pilot_v11.log"
STATUS_FILE="${RUN}/pipeline_followup.json"
ATTEMPTS="${RUN}/recovery_attempts.json"
PYTHON="/data/mega-asr/venv/bin/python"
TORCHRUN="/data/mega-asr/venv/bin/torchrun"
MANIFEST="/data/mega-asr/manifests/pilot_rl.jsonl"
VAL_MANIFEST="/data/mega-asr/manifests/rl_val_pool.jsonl"
WER_MANIFEST="/data/mega-asr/manifests/validation.jsonl"
CONFIG="configs/train/qwen3_asr_rl_v11.yaml"
DRIVER_PID="${1:-}"

mkdir -p "$RUN" /data/mega-asr/logs
cd "$REPO"

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

v11_busy() {
  ps -eo args | awk '/python/ && (/train\/train_rl.py/ || /parallel_inference.py/ || /score_rl_greedy_held_out.py/) && !/awk/ { found=1 } END { exit !found }'
}

if [ -n "$DRIVER_PID" ]; then
  echo "Waiting for driver pid ${DRIVER_PID}"
  while kill -0 "$DRIVER_PID" 2>/dev/null; do
    sleep 30
  done
fi
while v11_busy; do
  echo "v11 python still running; waiting"
  sleep 30
done

if [ ! -f "$ATTEMPTS" ]; then
  echo '{"resumed_chunks":[],"eval_retries":[]}' > "$ATTEMPTS"
fi

attempted() {
  local kind="$1"
  local key="$2"
  "$PYTHON" - "$ATTEMPTS" "$kind" "$key" << 'PY'
import json, sys
doc = json.load(open(sys.argv[1], encoding="utf-8"))
sys.exit(0 if sys.argv[3] in doc.get(sys.argv[2], []) else 1)
PY
}

mark_attempt() {
  local kind="$1"
  local key="$2"
  "$PYTHON" - "$ATTEMPTS" "$kind" "$key" << 'PY'
import json, sys
path, kind, key = sys.argv[1:]
doc = json.load(open(path, encoding="utf-8"))
doc.setdefault(kind, [])
if key not in doc[kind]:
    doc[kind].append(key)
json.dump(doc, open(path, "w", encoding="utf-8"), indent=2)
PY
}

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

checkpoint_ok() {
  local step="$1"
  [ -f "${RUN}/checkpoints/step_${step}/optimizer.pt" ] && \
    [ -f "${RUN}/checkpoints/step_${step}/adapter/adapter_model.safetensors" ]
}

gate_action() {
  "$PYTHON" - "$1" << 'PY'
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

eval_step() {
  local step="$1"
  local ckpt="${RUN}/checkpoints/step_${step}/adapter"
  local preds="${RUN}/predictions_step_${step}.jsonl"
  local gate="${RUN}/gate_step_${step}.json"
  if [ ! -d "$ckpt" ]; then
    return 1
  fi
  if prediction_rows_ok "$step" && [ -f "$gate" ]; then
    return 0
  fi
  if attempted eval_retries "$step"; then
    echo "Eval for step ${step} was already retried"
    return 1
  fi
  mark_attempt eval_retries "$step"
  rm -f "$preds" "$gate"
  write_status "EVALUATING" "recovery_step_${step}"
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
  prediction_rows_ok "$step" && [ -f "$gate" ]
}

state_status="MISSING"
if [ -f "${RUN}/pipeline_state.json" ]; then
  state_status=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status",""))' "${RUN}/pipeline_state.json")
fi
echo "Recovery sees pipeline status ${state_status}"

case "$state_status" in
  BLOCKED_TRANSFER|BLOCKED_SEARCH|BLOCKED_NO_WINNERS|STOPPED_KL|STOPPED_REWARD_DROP|STOPPED_ROBUST|FAILED_ZERO_VARIANCE)
    designed_stop=1
    ;;
  *)
    designed_stop=0
    ;;
esac

latest=$("$PYTHON" "${REPO}/scripts/score_rl_greedy_held_out.py" --resolve-step --run-dir "$RUN") || {
  write_status "FAILED" "recovery" "no complete checkpoint; not restarting from step 0"
  echo "FAILED: no complete checkpoint"
  exit 1
}

bash "${REPO}/scripts/score_rl_pilot_v11_checkpoint.sh" "$latest" || {
  write_status "FAILED" "recovery" "step ${latest} eval still incomplete"
  echo "FAILED: step ${latest} eval still incomplete"
  exit 1
}
if ! prediction_rows_ok "$latest" || [ ! -f "${RUN}/gate_step_${latest}.json" ]; then
  write_status "FAILED" "recovery" "step ${latest} eval still incomplete"
  echo "FAILED: step ${latest} eval still incomplete"
  exit 1
fi
action=$(gate_action "${RUN}/gate_step_${latest}.json")
echo "Step ${latest} gate action: ${action}"

if [ "$action" = "PASSED" ]; then
  write_status "DONE" "gates_ready" "train_status=CANDIDATE"
  echo "DONE: gate PASSED; weights stay in this run"
  exit 0
fi
if [ "$action" = "STOP_ROBUST" ] || [ "$designed_stop" -eq 1 ]; then
  if [ "$action" = "STOP_ROBUST" ]; then
    write_status "DONE" "gates_ready" "train_status=STOPPED_ROBUST"
  else
    write_status "DONE" "gates_ready" "train_status=${state_status}"
  fi
  echo "DONE: designed stop, no further chunk"
  exit 0
fi

if [ "$state_status" != "CHUNK_DONE" ] || [ "$action" != "CONTINUE" ]; then
  write_status "DONE" "gates_ready" "train_status=${state_status} gate_action=${action}"
  echo "DONE: no resume for status ${state_status} action ${action}"
  exit 0
fi

next=0
if [ "$latest" -eq 4 ]; then
  next=8
elif [ "$latest" -eq 8 ]; then
  next=12
fi
if [ "$next" -eq 0 ]; then
  write_status "DONE" "gates_ready" "train_status=CHUNK_DONE gate_action=CONTINUE"
  echo "DONE: horizon already has the last checkpoint"
  exit 0
fi
if attempted resumed_chunks "$next"; then
  write_status "FAILED" "recovery" "chunk ${next} already resumed once"
  echo "FAILED: chunk ${next} already resumed once"
  exit 1
fi

probe_decision=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("decision",""))' "$PROBE")
if [ "$probe_decision" != "GO_GRPO" ]; then
  write_status "FAILED" "recovery" "probe decision ${probe_decision}"
  exit 1
fi

mark_attempt resumed_chunks "$next"
write_status "TRAINING" "recovery_chunk_${next}"
echo "Resuming chunk ${next} from step_${latest}"
"$TORCHRUN" --standalone --nproc_per_node=4 train/train_rl.py \
  --manifest "$MANIFEST" \
  --val-manifest "$VAL_MANIFEST" \
  --wer-manifest "$WER_MANIFEST" \
  --config "$CONFIG" \
  --output-dir "$RUN" \
  --sample-strategy degraded \
  --probe-decision "$PROBE" \
  --max-steps "$next" \
  --save-steps 4 \
  --eval-steps 4 \
  --no-export-merged-on-finish \
  --resume-from-checkpoint "${RUN}/checkpoints/step_${latest}" \
  >> "$TRAIN_LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then
  write_status "FAILED" "recovery_chunk_${next}" "torchrun exit ${rc}"
  echo "FAILED: recovery chunk ${next} exit ${rc}"
  exit 1
fi
state_status=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status",""))' "${RUN}/pipeline_state.json")
echo "After recovery chunk ${next}: ${state_status}"
saved=$("$PYTHON" "${REPO}/scripts/score_rl_greedy_held_out.py" --resolve-step --run-dir "$RUN") || {
  write_status "FAILED" "recovery" "checkpoint after resume is incomplete"
  exit 1
}
bash "${REPO}/scripts/score_rl_pilot_v11_checkpoint.sh" "$saved" || {
  write_status "FAILED" "recovery" "step ${saved} eval incomplete after resume"
  exit 1
}
action=$(gate_action "${RUN}/gate_step_${saved}.json")
echo "Step ${saved} gate action: ${action}"
if [ "$action" = "PASSED" ]; then
  write_status "DONE" "gates_ready" "train_status=CANDIDATE"
elif [ "$action" = "STOP_ROBUST" ]; then
  write_status "DONE" "gates_ready" "train_status=STOPPED_ROBUST"
else
  write_status "DONE" "gates_ready" "train_status=${state_status} gate_action=${action}"
fi
echo "DONE: recovery finished status ${state_status} action ${action}"
