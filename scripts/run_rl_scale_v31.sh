#!/usr/bin/env bash
# RL v31 scale experiment (option B). Design: docs/qwen3-asr/29_rl_v31_scale_design.md
#
# Pipeline: verdict manifest -> probe (GO_GRPO required) -> 2,560-step training
# on the full rl_train_pool degraded pool (stop_profile=scale) -> release gate on
# the last checkpoint -> DPO Champion and candidate inference on the extra
# verdict set -> paired significance gate -> scale_verdict.json.
#
# RL_SCALE_SMOKE=1: run dir rl_scale_v31_smoke, 5 steps (save/eval every 5),
# stops after the checkpoint is saved; no scoring, no verdict.
#
# A formal run is executed as 320-step chunks.  An interrupted chunk leaves the
# run directory recoverable; rerun with RL_RESUME=1 and RL_RUN_DIR=<run> to
# continue from the last complete checkpoint.  The launcher refuses an
# existing run unless that explicit resume flag is present.  It also refuses to
# start while another trainer/inference process is on the GPUs. Never writes
# into the DPO Champion directory; the DPO baseline on the extra set goes to
# its own directory and is reused when it is already complete.
set -u
set -o pipefail

REPO=/data/mega-asr/repo
PYTHON=/data/mega-asr/venv/bin/python
TORCHRUN=/data/mega-asr/venv/bin/torchrun
CONFIG=$REPO/configs/train/qwen3_asr_rl_v31_scale.yaml
MANIFESTS=/data/mega-asr/manifests
MANIFEST=$MANIFESTS/rl_train_pool.jsonl
VAL_MANIFEST=$MANIFESTS/rl_val_pool.jsonl
WER_MANIFEST=$MANIFESTS/validation.jsonl
SFT_MANIFEST=$MANIFESTS/sft_train.jsonl
DPO_TRAIN_MANIFEST=$MANIFESTS/dpo_train_pool.jsonl
BENCH_MANIFEST=$MANIFESTS/bench_test.jsonl
EXTRA_MANIFEST=$MANIFESTS/rl_verdict_extra_degraded.jsonl
DPO_MODEL=/data/mega-asr/runs/dpo_pilot_v2/merged_base
SMOKE="${RL_SCALE_SMOKE:-0}"
RUN_OVERRIDE="${RL_RUN_DIR:-}"
RESUME="${RL_RESUME:-0}"
MIN_FREE_GB=100

if [ "$SMOKE" = "1" ]; then
  RUN="${RUN_OVERRIDE:-/data/mega-asr/runs/rl_scale_v31_smoke_$(date +%Y%m%dT%H%M%S)}"
  CHUNK_STEPS=5
else
  RUN="${RUN_OVERRIDE:-/data/mega-asr/runs/rl_scale_v31_$(date +%Y%m%dT%H%M%S)}"
  CHUNK_STEPS=320
fi
case "$RUN" in
  /data/mega-asr/runs/rl_scale_v31* ) ;;
  * ) echo "REFUSE: RL_RUN_DIR must be under /data/mega-asr/runs/rl_scale_v31*" >&2; exit 1 ;;
esac
BASELINE_DIR="$RUN/baselines"
BASE_EVAL_DIR="$BASELINE_DIR/base_validation"
DPO_EVAL_DIR="$BASELINE_DIR/dpo_validation"
DPO_VALIDATION_SCORED="$DPO_EVAL_DIR/predictions_eval/scored.jsonl"
DPO_EXTRA_DIR="$BASELINE_DIR/dpo_extra"

log() { printf '%s %s\n' "$(date -Is)" "$*" | tee -a "$RUN/launcher.log"; }
fail() { log "FAILED $*"; exit 1; }

if [ -e "$RUN" ] && [ "$RESUME" != "1" ]; then
  echo "REFUSE: run already exists: $RUN (set RL_RESUME=1 to continue a complete checkpoint)" >&2
  exit 1
fi
if [ "$RESUME" = "1" ] && [ ! -d "$RUN" ]; then
  echo "REFUSE: RL_RESUME=1 requires an existing run directory: $RUN" >&2
  exit 1
fi
for path in "$CONFIG" "$MANIFEST" "$VAL_MANIFEST" "$WER_MANIFEST" "$SFT_MANIFEST" \
            "$DPO_TRAIN_MANIFEST" "$BENCH_MANIFEST" "$DPO_MODEL" \
            "$MANIFESTS/dpo_val_pool.jsonl"; do
  [ -e "$path" ] || { echo "REFUSE: missing $path" >&2; exit 1; }
done

# The preflight imports the repository's validator. Make it independent of
# the caller's working directory before running any Python code.
cd "$REPO" || exit 1

# The 10x rollout volume is tens of GB even before model checkpoints.  Refuse
# to start when the filesystem cannot hold a complete run and its temporary
# merge/audit files.  This is deliberately a fixed launcher safety floor.
RUN_PARENT=$(dirname "$RUN")
FREE_KB=$(df -Pk "$RUN_PARENT" | awk 'NR == 2 {print $4}')
MIN_FREE_KB=$((MIN_FREE_GB * 1024 * 1024))
[ -n "$FREE_KB" ] && [ "$FREE_KB" -ge "$MIN_FREE_KB" ] || {
  echo "REFUSE: ${RUN_PARENT} has less than ${MIN_FREE_GB} GiB free" >&2
  exit 1
}

if [ "$SMOKE" = "1" ]; then
  POOL_MIN_ROWS=1
  CELL_MIN_ROWS=1
else
  # Formal runs must not be able to weaken the data contract through an
  # environment override. Increase the contract in the repo/config instead.
  POOL_MIN_ROWS=160000
  CELL_MIN_ROWS=640
  MAX_CELL_FRACTION=0.20
fi
POOL_PREFLIGHT=$(
  "$PYTHON" - "$MANIFEST" "$POOL_MIN_ROWS" "$CELL_MIN_ROWS" "${MAX_CELL_FRACTION:-}" "$([ "$SMOKE" = "1" ] && echo 0 || echo 1)" <<'PY'
import json
import sys
from pathlib import Path

from train.rl_sample_strategy import validate_degraded_manifest

manifest, min_rows, min_cell_rows = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
max_cell_fraction = float(sys.argv[4]) if sys.argv[4] else None
require_all_cells = sys.argv[5] == "1"
rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
summary = validate_degraded_manifest(
    rows,
    min_rows=min_rows,
    min_cell_rows=min_cell_rows,
    max_cell_fraction=max_cell_fraction,
    require_all_cells=require_all_cells,
)
print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
PY
) || { echo "REFUSE: RL manifest failed size/coverage contract" >&2; exit 1; }

# Check the three stable identity keys before touching CUDA.  The trainer also
# checks train/val/wer at runtime; this launcher-level audit covers the other
# fixed roles so a malformed rebuilt pool cannot reach the probe.
ISOLATION_CHECK=$(
  "$PYTHON" - "$MANIFEST" "$VAL_MANIFEST" "$SFT_MANIFEST" "$DPO_TRAIN_MANIFEST" \
    "$BENCH_MANIFEST" "$WER_MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

keys = ("sample_id", "source_utterance_id", "audio_sha256")
paths = [Path(value) for value in sys.argv[1:]]
loaded = {}
for path in paths:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    loaded[str(path)] = rows
    for key in keys:
        values = [str(row.get(key) or "").strip() for row in rows]
        if any(not value for value in values):
            raise SystemExit(f"{path} has missing {key}")
        if len(values) != len(set(values)):
            raise SystemExit(f"{path} has duplicate {key}")
train_path = str(paths[0])
train = loaded[train_path]
for other_path, rows in loaded.items():
    if other_path == train_path:
        continue
    for key in keys:
        overlap = {str(row[key]).strip() for row in train} & {str(row[key]).strip() for row in rows}
        if overlap:
            raise SystemExit(f"RL train overlaps {other_path} on {key}: {sorted(overlap)[0]}")
print(json.dumps({"roles_checked": len(paths), "keys": list(keys), "status": "PASSED"}, sort_keys=True))
PY
) || { echo "REFUSE: RL manifest identity isolation failed" >&2; exit 1; }

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "REFUSE: nvidia-smi is unavailable; cannot verify that GPUs are free" >&2
  exit 1
fi
if ! GPU_APPS="$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null)"; then
  echo "REFUSE: nvidia-smi could not query CUDA compute processes" >&2
  exit 1
fi
if [ -n "$(printf '%s' "$GPU_APPS" | tr -d '[:space:],')" ]; then
  echo "REFUSE: CUDA compute processes are already using GPUs:" >&2
  printf '%s\n' "$GPU_APPS" >&2
  exit 1
fi

mkdir -p "$RUN"
if [ "$RESUME" = "1" ] && [ ! -f "$RUN/source.json" ]; then
  echo "REFUSE: resume run is missing source.json: $RUN" >&2
  exit 1
fi
if [ "$RESUME" != "1" ]; then
cat > "$RUN/source.json" <<JSON
{
  "run_id": "$(basename "$RUN")",
  "design": "docs/qwen3-asr/29_rl_v31_scale_design.md",
  "source_model": "$DPO_MODEL",
  "config": "$CONFIG",
  "manifest": "$MANIFEST",
  "val_manifest": "$VAL_MANIFEST",
  "wer_manifest": "$WER_MANIFEST",
  "extra_verdict_manifest": "$EXTRA_MANIFEST",
  "seed": 20260722,
  "sample_strategy": "degraded",
  "min_degraded_rows": $POOL_MIN_ROWS,
  "min_cell_rows": $CELL_MIN_ROWS,
  "max_cell_fraction": ${MAX_CELL_FRACTION:-null},
  "chunk_steps": $CHUNK_STEPS,
  "min_free_gb": $MIN_FREE_GB,
  "pool_preflight": $POOL_PREFLIGHT,
  "isolation_check": $ISOLATION_CHECK,
  "stop_profile": "scale",
  "smoke": $([ "$SMOKE" = "1" ] && echo true || echo false),
  "gate_profile": "release",
  "repo_train_rl_md5": "$(md5sum "$REPO/train/train_rl.py" | cut -d' ' -f1)",
  "config_md5": "$(md5sum "$CONFIG" | cut -d' ' -f1)"
}
JSON
fi
if [ ! -f "$RUN/launcher_script.sh" ] || [ "$RESUME" != "1" ]; then
  cp "$0" "$RUN/launcher_script.sh" 2>/dev/null || true
fi
if [ "$RESUME" = "1" ]; then
  "$PYTHON" - "$RUN/source.json" "$CONFIG" "$REPO/train/train_rl.py" <<'PY' || {
import hashlib
import json
import sys
from pathlib import Path

source = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
config_path = Path(sys.argv[2])
trainer_path = Path(sys.argv[3])
def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
checks = {
    "config_md5": md5(config_path),
    "repo_train_rl_md5": md5(trainer_path),
}
bad = [key for key, value in checks.items() if source.get(key) != value]
if bad:
    raise SystemExit(f"resume source contract changed: {bad}")
if source.get("sample_strategy") != "degraded" or int(source.get("chunk_steps", 0)) != 320:
    raise SystemExit("resume source contract is not the v31 chunked degraded run")
print("resume source contract OK")
PY
    echo "REFUSE: resume source contract does not match current code/config" >&2
    exit 1
  }
fi
log "START run=$RUN smoke=$SMOKE"
log "POOL_PREFLIGHT min_degraded_rows=$POOL_MIN_ROWS min_cell_rows=$CELL_MIN_ROWS summary=$POOL_PREFLIGHT"
log "ISOLATION_CHECK $ISOLATION_CHECK"

# 1. Extra verdict manifest (idempotent: an existing file must have the same sha256).
"$PYTHON" scripts/build_rl_verdict_manifest.py \
  --source "$MANIFESTS/dpo_val_pool.jsonl" \
  --exclude "$SFT_MANIFEST" "$DPO_TRAIN_MANIFEST" "$MANIFEST" "$VAL_MANIFEST" "$WER_MANIFEST" "$BENCH_MANIFEST" \
  --output "$EXTRA_MANIFEST" \
  --check-audio >> "$RUN/launcher.log" 2>&1 || fail "verdict manifest"
EXTRA_ROWS=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1]))['row_count'])" "$EXTRA_MANIFEST.COMPLETE.json") \
  || fail "read verdict manifest row count"
log "VERDICT_MANIFEST rows=$EXTRA_ROWS"

# 2. Probe on the training pool.  A completed probe is reusable on resume;
# re-running it would only add GPU cost and could make the resume audit harder
# to interpret.
if [ ! -f "$RUN/search_probe.json" ]; then
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$TORCHRUN" --standalone --nproc_per_node=4 \
    train/train_rl.py \
    --manifest "$MANIFEST" --val-manifest "$VAL_MANIFEST" --wer-manifest "$WER_MANIFEST" \
    --config "$CONFIG" --output-dir "$RUN" \
    --sample-strategy degraded --allow-subset \
    --probe-only --probe-prompts 128 --no-export-merged-on-finish \
    >> "$RUN/probe.log" 2>&1
  probe_rc=$?
  [ "$probe_rc" -eq 0 ] && [ -f "$RUN/search_probe.json" ] || fail "probe rc=$probe_rc"
else
  log "PROBE_REUSED path=$RUN/search_probe.json"
fi
probe_decision=$("$PYTHON" -c "import json,sys; print(json.load(open(sys.argv[1])).get('decision',''))" "$RUN/search_probe.json")
log "PROBE decision=$probe_decision"
[ "$probe_decision" = "GO_GRPO" ] || fail "probe decision=$probe_decision"

# 3. Training.  Formal mode advances one checkpoint interval at a time so a
# preemption, driver restart, or host maintenance can resume from the last
# complete adapter/optimizer/RNG checkpoint.  The trainer's YAML horizon stays
# 2,560; --max-steps is only this process's chunk end.
run_train_chunk() {
  local chunk_end="$1"
  local from_step="${2:-0}"
  local resume_args=()
  if [ "$from_step" -gt 0 ]; then
    require_checkpoint "$from_step"
    resume_args=(--resume-from-checkpoint "$RUN/checkpoints/step_${from_step}")
  fi
  log "TRAIN_CHUNK start=$from_step end=$chunk_end"
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$TORCHRUN" --standalone --nproc_per_node=4 \
    train/train_rl.py \
    --manifest "$MANIFEST" --val-manifest "$VAL_MANIFEST" --wer-manifest "$WER_MANIFEST" \
    --config "$CONFIG" --output-dir "$RUN" \
    --sample-strategy degraded --allow-subset \
    --probe-decision "$RUN/search_probe.json" \
    --max-steps "$chunk_end" --save-steps "$CHUNK_STEPS" --eval-steps "$CHUNK_STEPS" \
    "${resume_args[@]}" \
    --no-export-merged-on-finish \
    >> "$RUN/train.log" 2>&1
  local rc=$?
  log "TRAIN_CHUNK end=$chunk_end rc=$rc"
  return "$rc"
}

require_checkpoint() {
  local step="$1"
  local checkpoint="$RUN/checkpoints/step_${step}"
  [ -d "$checkpoint/adapter" ] || fail "missing checkpoint adapter step=$step"
  [ -f "$checkpoint/optimizer.pt" ] || fail "missing checkpoint optimizer step=$step"
  [ -f "$checkpoint/scheduler.pt" ] || fail "missing checkpoint scheduler step=$step"
  [ -f "$checkpoint/training_state.json" ] || fail "missing checkpoint metadata step=$step"
  for rank in 0 1 2 3; do
    [ -f "$checkpoint/rng_state_rank_${rank}.pt" ] || fail "missing checkpoint RNG rank=$rank step=$step"
  done
}

materialize_terminal_pipeline_state() {
  local step="$1"
  local status="$2"
  "$PYTHON" - "$RUN" "$step" "$status" <<'PY'
import datetime
import json
import os
import sys
from pathlib import Path

run = Path(sys.argv[1])
step = int(sys.argv[2])
status = sys.argv[3]
pipeline = run / "pipeline_state.json"
try:
    payload = json.loads(pipeline.read_text(encoding="utf-8"))
except (FileNotFoundError, json.JSONDecodeError):
    payload = {}
if not isinstance(payload, dict):
    payload = {}
payload.update(
    {
        "stage": "rl_pilot",
        "global_step": step,
        "world_size": 4,
        "last_valid_checkpoint": str(run / "checkpoints" / f"step_{step}"),
        "status": status,
        "promoted": False,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
)
temporary = pipeline.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
os.replace(temporary, pipeline)
PY
}

if [ "$SMOKE" = "1" ]; then
  run_train_chunk 5 0 || fail "train smoke chunk"
else
  CURRENT_STEP=0
  CURRENT_STATUS="-"
  CURRENT_TERMINAL=0
  if [ "$RESUME" = "1" ]; then
    if ! IFS=$'\t' read -r CURRENT_STEP CURRENT_STATUS CURRENT_TERMINAL < <(
      "$PYTHON" -c 'import sys; from train.train_rl import select_resume_step; choice = select_resume_step(sys.argv[1], world_size=4); print("%s\t%s\t%s" % (int(choice["step"]), choice.get("status") or "-", 1 if choice["terminal"] else 0))' "$RUN"
    ); then
      fail "read resume checkpoint"
    fi
    log "RESUME_STATE step=$CURRENT_STEP status=$CURRENT_STATUS terminal=$CURRENT_TERMINAL source=select_resume_step"
  fi
  if [ "$CURRENT_TERMINAL" = "1" ]; then
    materialize_terminal_pipeline_state "$CURRENT_STEP" "$CURRENT_STATUS" \
      || fail "materialize terminal pipeline state"
    log "TRAIN_REUSE terminal_step=$CURRENT_STEP status=$CURRENT_STATUS"
  else
      while [ "$CURRENT_STEP" -lt 2560 ]; do
        NEXT_STEP=$((CURRENT_STEP + CHUNK_STEPS))
        [ "$NEXT_STEP" -le 2560 ] || NEXT_STEP=2560
        run_train_chunk "$NEXT_STEP" "$CURRENT_STEP" || fail "train chunk end=$NEXT_STEP"
        read -r NEXT_REPORTED NEXT_STATUS < <("$PYTHON" -c "import json,sys; d=json.load(open(sys.argv[1])); print(int(d['global_step']), d.get('status',''))" "$RUN/pipeline_state.json") \
          || fail "read pipeline_state after chunk=$NEXT_STEP"
        [ "$NEXT_REPORTED" -gt "$CURRENT_STEP" ] || fail "chunk did not advance: current=$CURRENT_STEP reported=$NEXT_REPORTED"
        require_checkpoint "$NEXT_REPORTED"
        CURRENT_STEP="$NEXT_REPORTED"
        CURRENT_STATUS="$NEXT_STATUS"
        log "TRAIN_CHUNK_DONE step=$CURRENT_STEP status=$CURRENT_STATUS"
        case "$CURRENT_STATUS" in
          CHUNK_DONE) continue ;;
          *) break ;;
        esac
      done
  fi
  if [ "$CURRENT_TERMINAL" != "1" ] && [ "$CURRENT_STEP" -ge 2560 ] && [ "$CURRENT_STATUS" = "CHUNK_DONE" ]; then
    materialize_terminal_pipeline_state "$CURRENT_STEP" "COMPLETED" \
      || fail "materialize completed pipeline state"
    CURRENT_STATUS="COMPLETED"
  fi
fi

read -r STEP TRAIN_STATUS < <("$PYTHON" -c "
import json, sys
d = json.load(open(sys.argv[1]))
print(int(d['global_step']), d.get('status', ''))
" "$RUN/pipeline_state.json") || fail "read pipeline_state.json"
log "TRAIN_DONE step=$STEP status=$TRAIN_STATUS"
require_checkpoint "$STEP"

if [ "$SMOKE" = "1" ]; then
  log "SMOKE_DONE step=$STEP status=$TRAIN_STATUS (no scoring)"
  exit 0
fi

# Build same-contract base/DPO baselines before candidate release scoring.
run_eval_baseline() {
  local model_id="$1" out_dir="$2" method="$3"
  local pred="$out_dir/predictions.jsonl"
  local scored="$out_dir/predictions_eval/scored.jsonl"
  mkdir -p "$out_dir"
  if [ ! -f "$scored" ] || [ "$(count_rows "$scored")" != "2867" ]; then
    rm -f "$pred"
    "$PYTHON" inference/parallel_inference.py \
      --manifest "$WER_MANIFEST" --output "$pred" \
      --model-id "$model_id" --method "$method" --gpus 0 1 2 3 \
      --max-new-tokens 512 >> "$RUN/score.log" 2>&1 || fail "$method validation inference"
    [ "$(count_rows "$pred")" = "2867" ] || fail "$method validation predictions row count"
    "$PYTHON" evaluation/eval_wer.py --predictions-jsonl "$pred" \
      --output-dir "$out_dir/predictions_eval" >> "$RUN/score.log" 2>&1 || fail "$method validation eval"
  fi
}

count_rows() { "$PYTHON" -c "import sys; print(sum(1 for l in open(sys.argv[1], encoding='utf-8') if l.strip()))" "$1"; }
run_eval_baseline Qwen/Qwen3-ASR-1.7B "$BASE_EVAL_DIR" base
run_eval_baseline "$DPO_MODEL" "$DPO_EVAL_DIR" dpo_pilot

# 4. Release gate on the last checkpoint (validation 2,867 rows; thresholds unchanged).
RL_GATE_PROFILE=release RL_RUN_DIR="$RUN" RL_CONFIG="$CONFIG" \
  RL_BASE_EVAL_DIR="$BASE_EVAL_DIR" RL_DPO_EVAL_DIR="$DPO_EVAL_DIR" RL_MAX_NEW_TOKENS=512 \
  bash scripts/score_rl_pilot_checkpoint.sh "$STEP" >> "$RUN/score.log" 2>&1 || fail "release scoring step=$STEP"
log "RELEASE_GATE written gate_step_$STEP.json"

count_rows() { "$PYTHON" -c "import sys; print(sum(1 for l in open(sys.argv[1], encoding='utf-8') if l.strip()))" "$1"; }

# 5. DPO Champion on the extra verdict set (own directory, reused when complete).
DPO_EXTRA_PRED=$DPO_EXTRA_DIR/predictions.jsonl
DPO_EXTRA_SCORED=$DPO_EXTRA_DIR/predictions_eval/scored.jsonl
if [ ! -f "$DPO_EXTRA_SCORED" ] || [ "$(count_rows "$DPO_EXTRA_SCORED")" != "$EXTRA_ROWS" ]; then
  mkdir -p "$DPO_EXTRA_DIR"
  rm -f "$DPO_EXTRA_PRED"
  "$PYTHON" inference/parallel_inference.py \
    --manifest "$EXTRA_MANIFEST" --output "$DPO_EXTRA_PRED" \
    --model-id "$DPO_MODEL" --method dpo_pilot --gpus 0 1 2 3 \
    --max-new-tokens 512 \
    >> "$RUN/score.log" 2>&1 || fail "DPO extra inference"
  [ "$(count_rows "$DPO_EXTRA_PRED")" = "$EXTRA_ROWS" ] || fail "DPO extra predictions row count"
  "$PYTHON" evaluation/eval_wer.py --predictions-jsonl "$DPO_EXTRA_PRED" \
    --output-dir "$DPO_EXTRA_DIR/predictions_eval" >> "$RUN/score.log" 2>&1 || fail "DPO extra eval"
fi
log "DPO_EXTRA ready $DPO_EXTRA_SCORED"

# 6. Candidate on the extra verdict set.
CAND_EXTRA_PRED=$RUN/predictions_step_${STEP}_extra.jsonl
CAND_EXTRA_DIR=$RUN/predictions_step_${STEP}_extra_eval
"$PYTHON" inference/parallel_inference.py \
  --manifest "$EXTRA_MANIFEST" --output "$CAND_EXTRA_PRED" \
  --model-id "$DPO_MODEL" --adapter-dir "$RUN/checkpoints/step_$STEP/adapter" \
  --method "$(basename "$RUN")_step_${STEP}" --gpus 0 1 2 3 --max-new-tokens 512 \
  >> "$RUN/score.log" 2>&1 || fail "candidate extra inference"
[ "$(count_rows "$CAND_EXTRA_PRED")" = "$EXTRA_ROWS" ] || fail "candidate extra predictions row count"
"$PYTHON" evaluation/eval_wer.py --predictions-jsonl "$CAND_EXTRA_PRED" \
  --output-dir "$CAND_EXTRA_DIR" >> "$RUN/score.log" 2>&1 || fail "candidate extra eval"

# 7. Paired significance gate (validation + extra degraded, DPO Champion baseline).
PAIRED=$RUN/paired_verdict_step_${STEP}.json
"$PYTHON" evaluation/paired_significance.py \
  --baseline "$DPO_VALIDATION_SCORED" "$DPO_EXTRA_SCORED" \
  --candidate "$RUN/predictions_step_${STEP}_eval/scored.jsonl" "$CAND_EXTRA_DIR/scored.jsonl" \
  --gate --tail-gate --require-max-new-tokens 512 --output "$PAIRED" >> "$RUN/score.log" 2>&1 || fail "paired significance"

# 8. Summary.
"$PYTHON" - "$RUN" "$STEP" "$TRAIN_STATUS" <<'PY' | tee -a "$RUN/launcher.log"
import json, sys
from pathlib import Path
run, step, train_status = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
release = json.load(open(run / f"gate_step_{step}.json", encoding="utf-8"))
paired = json.load(open(run / f"paired_verdict_step_{step}.json", encoding="utf-8"))
bad_train = train_status in {"FAILED_ZERO_VARIANCE", "STOPPED_REWARD_DROP"}
passed = (
    release.get("gate_status") == "PASSED"
    and paired["gate"]["status"] == "PASSED"
    and paired.get("tail_gate", {}).get("status") == "PASSED"
    and not bad_train
)
summary = {
    "design": "docs/qwen3-asr/29_rl_v31_scale_design.md",
    "step": step,
    "train_status": train_status,
    "release_gate_status": release.get("gate_status"),
    "paired_gate": paired["gate"],
    "tail_gate": paired.get("tail_gate"),
    "paired_by_condition_group": paired["by_condition_group"],
    "paired_by_language_condition": paired["by_language_condition"],
    "status": "PASSED" if passed else "FAILED",
}
(run / "scale_verdict.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
deg = paired["by_condition_group"].get("degraded", {})
print(
    f"SCALE_VERDICT status={summary['status']} step={step} train_status={train_status} "
    f"release={summary['release_gate_status']} paired={paired['gate']['status']} tail={paired.get('tail_gate', {}).get('status')} "
    f"degraded_mean={deg.get('mean_delta')} degraded_ci95={deg.get('ci95')}"
)
PY
log "END"
