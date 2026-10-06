#!/usr/bin/env bash
# Score one saved RL checkpoint in $RL_RUN_DIR.
# Writes a greedy Full Held-out row, 2867 validation predictions, and
# gate_step_<N>.json (or pilot_gate_step_<N>.json for feasibility checks).
# Does not call train_rl.py and does not export weights.
set -u

STEP="${1:?step}"
REPO="/data/mega-asr/repo"
RUN="${RL_RUN_DIR:?RL_RUN_DIR}"
CONFIG="${RL_CONFIG:?RL_CONFIG}"
PYTHON="/data/mega-asr/venv/bin/python"
TORCHRUN="/data/mega-asr/venv/bin/torchrun"
VAL_MANIFEST="/data/mega-asr/manifests/rl_val_pool.jsonl"
WER_MANIFEST="/data/mega-asr/manifests/validation.jsonl"
LOSS_LOG="${RUN}/loss_log.jsonl"
CKPT="${RUN}/checkpoints/step_${STEP}"
PREDS="${RUN}/predictions_step_${STEP}.jsonl"
GATE="${RUN}/gate_step_${STEP}.json"
GATE_PROFILE="${RL_GATE_PROFILE:-release}"
BASE_EVAL_DIR="${RL_BASE_EVAL_DIR:-/data/mega-asr/runs/eval_validation_base}"
DPO_EVAL_DIR="${RL_DPO_EVAL_DIR:-/data/mega-asr/runs/dpo_pilot_v2}"
MAX_NEW_TOKENS="${RL_MAX_NEW_TOKENS:-512}"
case "$GATE_PROFILE" in
  release)
    ;;
  pilot_feasibility)
    GATE="${RUN}/pilot_gate_step_${STEP}.json"
    ;;
  *)
    echo "REFUSE: unsupported RL_GATE_PROFILE=${GATE_PROFILE}"
    exit 1
    ;;
esac
METRICS="${RUN}/predictions_step_${STEP}_eval/metrics.json"
SCORER="${REPO}/scripts/score_rl_greedy_held_out.py"
METHOD="${RL_METHOD:-$(basename "$RUN")_step_${STEP}}"

case "$RUN" in
  /data/mega-asr/runs/rl_pilot_v10|/data/mega-asr/runs/rl_pilot_v11|/data/mega-asr/runs/rl_pilot_v12|/data/mega-asr/runs/dpo_pilot_v2)
    echo "REFUSE: $RUN is a finished diagnostic run or the champion"
    exit 1
    ;;
esac
if [[ "$CKPT" == /data/mega-asr/runs/dpo_pilot_v2/* ]] || [[ "$CONFIG" == *dpo_pilot_v2/merged_base* ]]; then
  echo "REFUSE: path points at the DPO champion"
  exit 1
fi

cd "$REPO"
mkdir -p "$RUN"
echo $$ > "${RUN}/score_step_${STEP}.pid"

if "$PYTHON" - << 'PY'
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
then
  echo "REFUSE: training or scoring is already using the GPUs"
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
    --method "$METHOD" \
    --gpus 0 1 2 3 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
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
    --base-metrics "$BASE_EVAL_DIR/predictions_eval/metrics.json" \
    --base-predictions "$BASE_EVAL_DIR/predictions.jsonl" \
    --dpo-metrics "$DPO_EVAL_DIR/predictions_eval/metrics.json" \
    --dpo-predictions "$DPO_EVAL_DIR/predictions.jsonl" \
    --pilot-metrics "$METRICS" \
    --pilot-predictions "$PREDS" \
    --manifest "$WER_MANIFEST" \
    --rl-val-manifest "$VAL_MANIFEST" \
    --rl-loss-log "$LOSS_LOG" \
    --rl-step "$STEP" \
    --stage rl_pilot \
    --gate-profile "$GATE_PROFILE" \
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
