#!/usr/bin/env bash
# Probe first. Start the 12-step degraded GRPO run only when search_probe.json says GO_GRPO.
# The 2026-09-29 server run finished BLOCKED_TRANSFER. This checkout refuses to start another one.
set -u

if [ "${RL_V10_ALLOW_RERUN:-}" != "1" ]; then
  echo "rl_pilot_v10 finished BLOCKED_TRANSFER on 2026-09-29. Set RL_V10_ALLOW_RERUN=1 to start again. See docs/qwen3-asr/09_rl_v10_design.md"
  exit 0
fi

REPO="/data/mega-asr/repo"
RUN="/data/mega-asr/runs/rl_pilot_v10"
PROBE="${RUN}/search_probe.json"
TRAIN_LOG="/data/mega-asr/logs/rl_pilot_v10.log"
STATUS_FILE="${RUN}/pipeline_followup.json"
PYTHON="/data/mega-asr/venv/bin/python"
TORCHRUN="/data/mega-asr/venv/bin/torchrun"
MANIFEST="/data/mega-asr/manifests/pilot_rl.jsonl"
VAL_MANIFEST="/data/mega-asr/manifests/rl_val_pool.jsonl"
WER_MANIFEST="/data/mega-asr/manifests/validation.jsonl"
CONFIG="configs/train/qwen3_asr_rl.yaml"

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

run_chunk 4
state_status=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status",""))' "${RUN}/pipeline_state.json")
echo "After chunk 4: ${state_status}"
if [ "$state_status" = "CHUNK_DONE" ] && [ -d "${RUN}/checkpoints/step_4" ]; then
  run_chunk 8 "${RUN}/checkpoints/step_4"
  state_status=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status",""))' "${RUN}/pipeline_state.json")
  echo "After chunk 8: ${state_status}"
fi
if [ "$state_status" = "CHUNK_DONE" ] && [ -d "${RUN}/checkpoints/step_8" ]; then
  run_chunk 12 "${RUN}/checkpoints/step_8"
  state_status=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("status",""))' "${RUN}/pipeline_state.json")
  echo "After chunk 12: ${state_status}"
fi

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
    --method "rl_pilot_v10_step_${step}" \
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

for step in 4 8 12; do
  eval_step "$step" || echo "eval step ${step} failed"
done

write_status "DONE" "gates_ready" "train_status=${state_status}"
echo "DONE: rl_pilot_v10 follow-up train_status=${state_status}"
