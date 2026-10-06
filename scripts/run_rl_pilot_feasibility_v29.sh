#!/usr/bin/env bash
# ARCHIVED 2026-10-03: copied verbatim (below this header) from V100 /tmp/rl_pilot_feasibility_v29.sh.
# It already ran once; it refuses to overwrite an existing run dir. Kept for reproducibility only.
# Note: CONFIG points at qwen3_asr_rl_v28.yaml; --max-steps 4 overrides its horizon of 24.
set -u
set -o pipefail

REPO=/data/mega-asr/repo
RUN=/data/mega-asr/runs/rl_pilot_feasibility_v29
CONFIG=/data/mega-asr/repo/configs/train/qwen3_asr_rl_v28.yaml
PYTHON=/data/mega-asr/venv/bin/python
TORCHRUN=/data/mega-asr/venv/bin/torchrun
MANIFEST=/data/mega-asr/manifests/pilot_rl.jsonl
VAL_MANIFEST=/data/mega-asr/manifests/rl_val_pool.jsonl
WER_MANIFEST=/data/mega-asr/manifests/validation.jsonl

if [ -e "$RUN" ]; then
  echo "REFUSE: run already exists: $RUN" >&2
  exit 1
fi

mkdir -p "$RUN"
cat > "$RUN/source.json" <<JSON
{
  "run_id": "rl_pilot_feasibility_v29",
  "source_model": "/data/mega-asr/runs/dpo_pilot_v2/merged_base",
  "config": "$CONFIG",
  "manifest": "$MANIFEST",
  "val_manifest": "$VAL_MANIFEST",
  "wer_manifest": "$WER_MANIFEST",
  "seed": 20260722,
  "learning_rate": 5.0e-6,
  "sample_strategy": "degraded",
  "pilot_steps": 4,
  "gate_profile": "pilot_feasibility",
  "release_eligible": false
}
JSON

printf 'START %s\n' "$(date -Is)" > "$RUN/launcher.log"
printf 'RUN=%s\n' "$RUN" >> "$RUN/launcher.log"

CUDA_VISIBLE_DEVICES=0,1,2,3 "$TORCHRUN" --standalone --nproc_per_node=4 \
  "$REPO/train/train_rl.py" \
  --manifest "$MANIFEST" \
  --val-manifest "$VAL_MANIFEST" \
  --wer-manifest "$WER_MANIFEST" \
  --config "$CONFIG" \
  --output-dir "$RUN" \
  --sample-strategy degraded \
  --probe-only \
  --probe-prompts 128 \
  --no-export-merged-on-finish \
  >> "$RUN/probe.log" 2>&1
probe_rc=$?
if [ "$probe_rc" -ne 0 ] || [ ! -f "$RUN/search_probe.json" ]; then
  printf 'FAILED probe rc=%s\n' "$probe_rc" >> "$RUN/launcher.log"
  exit "$probe_rc"
fi

probe_decision=$($PYTHON - "$RUN/search_probe.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(json.load(f).get("decision", ""))
PY
)
printf 'PROBE decision=%s\n' "$probe_decision" >> "$RUN/launcher.log"
if [ "$probe_decision" != "GO_GRPO" ]; then
  printf 'STOP probe decision=%s\n' "$probe_decision" >> "$RUN/launcher.log"
  exit 1
fi

CUDA_VISIBLE_DEVICES=0,1,2,3 "$TORCHRUN" --standalone --nproc_per_node=4 \
  "$REPO/train/train_rl.py" \
  --manifest "$MANIFEST" \
  --val-manifest "$VAL_MANIFEST" \
  --wer-manifest "$WER_MANIFEST" \
  --config "$CONFIG" \
  --output-dir "$RUN" \
  --sample-strategy degraded \
  --probe-decision "$RUN/search_probe.json" \
  --max-steps 4 \
  --save-steps 4 \
  --eval-steps 4 \
  --no-export-merged-on-finish \
  >> "$RUN/train.log" 2>&1
train_rc=$?
printf 'TRAIN rc=%s\n' "$train_rc" >> "$RUN/launcher.log"
if [ "$train_rc" -ne 0 ]; then
  exit "$train_rc"
fi

step=$($PYTHON - "$RUN/pipeline_state.json" <<'PY'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    print(int(json.load(f)["global_step"]))
PY
)
printf 'SCORE step=%s profile=pilot_feasibility\n' "$step" >> "$RUN/launcher.log"
RL_GATE_PROFILE=pilot_feasibility RL_RUN_DIR="$RUN" RL_CONFIG="$CONFIG" \
  bash "$REPO/scripts/score_rl_pilot_checkpoint.sh" "$step" \
  >> "$RUN/score.log" 2>&1
score_rc=$?
printf 'SCORE rc=%s\n' "$score_rc" >> "$RUN/launcher.log"
printf 'END %s\n' "$(date -Is)" >> "$RUN/launcher.log"
exit "$score_rc"
