#!/usr/bin/env bash
# Score one saved rl_pilot_v11 checkpoint.
# Writes a greedy Full Held-out loss-log row, 2867 validation predictions,
# and gate_step_<N>.json. Does not call train_rl.py and does not export weights.
set -u

STEP="${1:?step}"
REPO="/data/mega-asr/repo"
RUN="/data/mega-asr/runs/rl_pilot_v11"
PYTHON="/data/mega-asr/venv/bin/python"
TORCHRUN="/data/mega-asr/venv/bin/torchrun"
VAL_MANIFEST="/data/mega-asr/manifests/rl_val_pool.jsonl"
WER_MANIFEST="/data/mega-asr/manifests/validation.jsonl"
CONFIG="${REPO}/configs/train/qwen3_asr_rl_v11.yaml"
LOSS_LOG="${RUN}/loss_log.jsonl"
CKPT="${RUN}/checkpoints/step_${STEP}"
PREDS="${RUN}/predictions_step_${STEP}.jsonl"
GATE="${RUN}/gate_step_${STEP}.json"
METRICS="${RUN}/predictions_step_${STEP}_eval/metrics.json"
SCORER="${REPO}/scripts/score_rl_greedy_held_out.py"

cd "$REPO"
echo $$ > "${RUN}/score_step_${STEP}.pid"

if [ "$CKPT" = "/data/mega-asr/runs/dpo_pilot_v2/merged_base" ] || [[ "$CKPT" == /data/mega-asr/runs/dpo_pilot_v2/* ]]; then
  echo "REFUSE: checkpoint path is the DPO champion"
  exit 1
fi

state_step=$("$PYTHON" "$SCORER" --resolve-step --run-dir "$RUN") || {
  echo "REFUSE: pipeline checkpoint is incomplete"
  exit 1
}
if [ "$state_step" != "$STEP" ]; then
  echo "REFUSE: pipeline global_step ${state_step} != requested step ${STEP}"
  exit 1
fi

if ps -eo args | awk '/python/ && (/train\/train_rl.py/ || /parallel_inference.py/ || /score_rl_greedy_held_out.py/) && !/awk/ { found=1 } END { exit !found }'; then
  echo "REFUSE: v11 training or scoring is already using the GPUs"
  exit 1
fi

decision=$("$PYTHON" "$SCORER" --decision-only --step "$STEP" --loss-log "$LOSS_LOG") || exit 1
echo "Greedy held-out decision for step ${STEP}: ${decision}"
case "$decision" in
  skip)
    ;;
  append)
    "$TORCHRUN" --standalone --nproc_per_node=4 "$SCORER" \
      --step "$STEP" \
      --config "$CONFIG" \
      --checkpoint "$CKPT" \
      --val-manifest "$VAL_MANIFEST" \
      --loss-log "$LOSS_LOG" \
      --eval-seed 42
    rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "FAILED: greedy held-out scorer exit ${rc}"
      exit 1
    fi
    ;;
  refuse)
    echo "FAILED: greedy held-out row for step ${STEP} is not provenance-valid"
    exit 1
    ;;
  *)
    echo "FAILED: unknown held-out decision ${decision}"
    exit 1
    ;;
esac

prediction_rows_ok() {
  [ -f "$PREDS" ] || return 1
  "$PYTHON" - "$PREDS" << 'PY'
import sys
count = sum(1 for line in open(sys.argv[1], encoding="utf-8") if line.strip())
sys.exit(0 if count == 2867 else 1)
PY
}

if ! prediction_rows_ok; then
  rm -f "$PREDS" "$GATE"
  "$PYTHON" inference/parallel_inference.py \
    --manifest "$WER_MANIFEST" \
    --output "$PREDS" \
    --model-id /data/mega-asr/runs/dpo_pilot_v2/merged_base \
    --adapter-dir "${CKPT}/adapter" \
    --method "rl_pilot_v11_step_${STEP}" \
    --gpus 0 1 2 3 \
    --eval
  if ! prediction_rows_ok; then
    echo "FAILED: predictions_step_${STEP}.jsonl is not 2867 rows"
    exit 1
  fi
fi

if [ ! -f "$METRICS" ]; then
  "$PYTHON" evaluation/eval_wer.py \
    --predictions "$PREDS" \
    --output-dir "${RUN}/predictions_step_${STEP}_eval"
fi

if [ ! -f "$GATE" ]; then
  "$PYTHON" evaluation/verify_gate.py \
    --base-metrics /data/mega-asr/runs/eval_validation_base/predictions_eval/metrics.json \
    --base-predictions /data/mega-asr/runs/eval_validation_base/predictions.jsonl \
    --dpo-metrics /data/mega-asr/runs/dpo_pilot_v2/predictions_eval/metrics.json \
    --dpo-predictions /data/mega-asr/runs/dpo_pilot_v2/predictions.jsonl \
    --pilot-metrics "$METRICS" \
    --pilot-predictions "$PREDS" \
    --manifest "$WER_MANIFEST" \
    --rl-val-manifest "$VAL_MANIFEST" \
    --rl-loss-log "$LOSS_LOG" \
    --rl-step "$STEP" \
    --stage rl_pilot \
    --held-out-decode greedy \
    --max-robust-regression 0.0 \
    --max-empty-rate 0.002 \
    --output "$GATE"
fi

"$PYTHON" - "$GATE" "$STEP" << 'PY'
import json, sys
gate = json.load(open(sys.argv[1], encoding="utf-8"))
metrics = gate["metrics"]
print(
    "SCORE_DONE"
    f" step={sys.argv[2]}"
    f" gate_status={gate.get('gate_status')}"
    f" robust_error_rate_increase={metrics.get('robust_error_rate_increase')}"
    f" held_out_reward_improvement={metrics.get('held_out_reward_improvement')}"
    f" improved_scenarios={metrics.get('degraded_scenario_improvements_count')}"
    f" valid_output_rate={metrics.get('valid_output_rate')}"
)
PY
