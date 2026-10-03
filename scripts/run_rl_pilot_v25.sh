#!/usr/bin/env bash
# rl_pilot_v25 starts fresh from the DPO champion.
# Same learning rate, raw-gap advantage, and audio projections as v16.
# Deleted greedy tokens receive the negated gap. Identical token ids skip.
# Never write v10 through v24, or the champion directory.
# Do not delete the v24 continue decision and do not rerun that driver.
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
V25_RUN="/data/mega-asr/runs/rl_pilot_v25"
V25_CONFIG="configs/train/qwen3_asr_rl_v25.yaml"
HORIZON=24
PIPE_LOG="/data/mega-asr/logs/rl_pilot_v25_pipeline.log"

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
    "/data/mega-asr/runs/rl_pilot_v10"|"/data/mega-asr/runs/rl_pilot_v11"|"/data/mega-asr/runs/rl_pilot_v12"|"/data/mega-asr/runs/rl_pilot_v13"|"/data/mega-asr/runs/rl_pilot_v14"|"/data/mega-asr/runs/rl_pilot_v15"|"/data/mega-asr/runs/rl_pilot_v16"|"/data/mega-asr/runs/rl_pilot_v17"|"/data/mega-asr/runs/rl_pilot_v18"|"/data/mega-asr/runs/rl_pilot_v19"|"/data/mega-asr/runs/rl_pilot_v20"|"/data/mega-asr/runs/rl_pilot_v21"|"/data/mega-asr/runs/rl_pilot_v22"|"/data/mega-asr/runs/rl_pilot_v23"|"/data/mega-asr/runs/rl_pilot_v24"|"$CHAMPION"|"${CHAMPION}"/*)
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
  "$PYTHON" - "$REPO/$V25_CONFIG" << 'PY'
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8")
required = (
    "learning_rate: 1.0e-5",
    "mode: raw_gap",
    "local_max_relative: 0.35",
    "beta: 0.04",
    "max_raw_kl: 5.0e-4",
    "max_steps: 24",
    "loss_reduction: sequence_sum",
    "policy_token_mask: signed_edits",
    "train_audio_projections: true",
    "expected_total_targets: 199",
)
missing = [item for item in required if item not in text]
forbidden = (
    "5.0e-6",
    "2.0e-5",
    "mode: unit",
    "mode: capped_gap",
    "mode: fixed",
    "train_audio_projections: false",
    "policy_token_mask: all",
    "policy_token_mask: changes_only",
)
found = [item for item in forbidden if item in text]
if missing or found:
    raise SystemExit("config missing " + ", ".join(missing) + " forbidden " + ", ".join(found))
PY
}

run_chunk() {
  local end="$1"
  local resume="${2:-}"
  refuse_run "$V25_RUN"
  local log="/data/mega-asr/logs/rl_pilot_v25.log"
  local resume_args=()
  if [ -n "$resume" ]; then
    resume_args=(--resume-from-checkpoint "$resume")
  fi
  write_status "$V25_RUN" "TRAINING" "chunk_${end}"
  "$TORCHRUN" --standalone --nproc_per_node=4 train/train_rl.py \
    --manifest "$MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --wer-manifest "$WER_MANIFEST" \
    --config "$V25_CONFIG" \
    --output-dir "$V25_RUN" \
    --sample-strategy degraded \
    --probe-decision "${V25_RUN}/search_probe.json" \
    --max-steps "$end" \
    --save-steps 4 \
    --eval-steps 4 \
    --no-export-merged-on-finish \
    "${resume_args[@]}" \
    >> "$log" 2>&1
}

score_step() {
  local step="$1"
  write_status "$V25_RUN" "EVALUATING" "validation_step_${step}"
  RL_RUN_DIR="$V25_RUN" RL_CONFIG="${REPO}/${V25_CONFIG}" RL_METHOD="rl_pilot_v25_step_${step}" \
    bash "${REPO}/scripts/score_rl_pilot_checkpoint.sh" "$step"
}

decide() {
  local step="$1"
  "$PYTHON" "${REPO}/train/rl_v16_decision.py" \
    --run-dir "$V25_RUN" \
    --step "$step" \
    --horizon "$HORIZON"
}

export_passed() {
  local step="$1"
  local dest="${V25_RUN}/merged_base"
  refuse_run "$dest"
  if [[ "$dest" == /data/mega-asr/runs/dpo_pilot_v2* ]]; then
    echo "REFUSE: export destination is the champion"
    exit 1
  fi
  "$PYTHON" train/train_rl.py \
    --export-merged \
    --config "$V25_CONFIG" \
    --checkpoint-dir "${V25_RUN}/checkpoints/step_${step}" \
    --output-dir "$dest"
}

finish_action() {
  local step="$1"
  local action="$2"
  if [ "$action" = "passed" ]; then
    export_passed "$step" || {
      write_status "$V25_RUN" "FAILED" "export" "export failed"
      exit 1
    }
    write_status "$V25_RUN" "DONE" "gates_ready" "train_status=PASSED"
    echo "DONE: rl_pilot_v25 gate PASSED"
    return 0
  fi
  write_status "$V25_RUN" "DONE" "gates_ready" "action=${action}"
  echo "DONE: rl_pilot_v25 action ${action}"
}

prepare_v25() {
  if [ -e "$V25_RUN" ]; then
    echo "REFUSE: rl_pilot_v25 already exists"
    exit 1
  fi
  if [ ! -f "${V12_RUN}/search_probe.json" ]; then
    echo "REFUSE: v12 probe is missing"
    exit 1
  fi
  mkdir -p "$V25_RUN"
  cp -p "${V12_RUN}/search_probe.json" "${V25_RUN}/search_probe.json"
  if ! PYTHONPATH="${REPO}/train:${REPO}" "$PYTHON" - << 'PY'
import json
from pathlib import Path
from rl_policy_mask import signed_edit_loss_args, signed_edit_update

run = Path("/data/mega-asr/runs/rl_pilot_v25")
probe = json.loads((run / "search_probe.json").read_text(encoding="utf-8"))
if probe.get("decision") != "GO_GRPO" or int(probe.get("n_prompts", 0)) != 128:
    raise SystemExit("v25 probe is not the 128-prompt GO_GRPO file")
for name in ("v16", "v20", "v21", "v22", "v23"):
    decision = json.loads(
        Path(f"/data/mega-asr/runs/rl_pilot_{name}/switch_decision.json").read_text(encoding="utf-8")
    )
    if decision.get("action") != "stop":
        raise SystemExit(f"{name} decision is not stop")
v24 = json.loads(
    Path("/data/mega-asr/runs/rl_pilot_v24/switch_decision.json").read_text(encoding="utf-8")
)
if v24.get("action") != "continue":
    raise SystemExit("v24 decision is not continue")
deletion = signed_edit_update([1, 2, 3], [1, 3], 0.25)
if deletion.action != "update":
    raise SystemExit(f"deletion-only action {deletion.action}")
if deletion.winner_token_advantage != (0.0, 0.0):
    raise SystemExit(f"deletion winner advantage {deletion.winner_token_advantage}")
if deletion.anchor_token_advantage != (0.0, -0.25, 0.0):
    raise SystemExit(f"deletion anchor advantage {deletion.anchor_token_advantage}")
same = signed_edit_update([10, 11, 12], [10, 11, 12], 0.25)
if same.action != "skip" or any(same.winner_token_advantage) or any(same.anchor_token_advantage):
    raise SystemExit(f"identical pair did not skip: {same}")
loss = signed_edit_loss_args([1, 2, 3], [1, 3], 0.25)
anchor_token = tuple(loss.sequence_advantage[0] * value for value in loss.anchor_delete)
winner_token = tuple(loss.sequence_advantage[1] * value for value in loss.winner_keep)
if loss.action != "update" or loss.sequence_advantage != (-0.25, 0.25):
    raise SystemExit(f"signed loss args {loss}")
if anchor_token != (0.0, -0.25, 0.0) or winner_token != (0.0, 0.0):
    raise SystemExit(f"signed token advantages {anchor_token} {winner_token}")
trainer = Path("/data/mega-asr/repo/train/train_rl.py").read_text(encoding="utf-8")
if "signed_edit_loss_args(" not in trainer:
    raise SystemExit("server trainer is missing signed_edit_loss_args")
if "def decide_rl_stop" not in trainer:
    raise SystemExit("server trainer is missing decide_rl_stop")
champion = Path("/data/mega-asr/runs/dpo_pilot_v2/merged_base/model.safetensors")
stat = champion.stat()
if stat.st_size != 4076190936:
    raise SystemExit(f"champion bytes changed: {stat.st_size}")
(run / "source.json").write_text(json.dumps({
    "source_model": "/data/mega-asr/runs/dpo_pilot_v2/merged_base",
    "probe_source": "/data/mega-asr/runs/rl_pilot_v12/search_probe.json",
    "champion_bytes": stat.st_size,
    "champion_mtime": stat.st_mtime,
    "local_max_relative": 0.35,
    "learning_rate": 1.0e-5,
    "advantage_mode": "raw_gap",
    "policy_token_mask": "signed_edits",
    "train_audio_projections": True,
    "expected_lora_targets": 199,
    "horizon": 24,
    "copied_loss_log": False,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("v25 baseline ok")
PY
  then
    rm -rf "$V25_RUN"
    echo "REFUSE: v25 baseline check failed"
    exit 1
  fi
}

score_and_decide() {
  local step
  step=$("$PYTHON" "${REPO}/scripts/score_rl_greedy_held_out.py" --resolve-step --run-dir "$V25_RUN") || {
    write_status "$V25_RUN" "FAILED" "validation" "saved checkpoint incomplete"
    exit 1
  }
  score_step "$step" || {
    write_status "$V25_RUN" "FAILED" "validation_step_${step}" "score failed"
    exit 1
  }
  local action
  action=$(decide "$step") || {
    write_status "$V25_RUN" "FAILED" "switch" "decision failed"
    exit 1
  }
  LAST_ACTION="$action"
  LAST_STEP="$step"
  echo "rl_pilot_v25 step ${step} action ${action}"
}

assert_config

if [ -f "${V25_RUN}/switch_decision.json" ]; then
  action=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("action",""))' "${V25_RUN}/switch_decision.json")
  echo "Existing v25 decision: ${action}"
  if [ "$action" = "continue" ]; then
    echo "REFUSE: v25 already has a continue decision; not starting another chunk from a rerun"
    exit 1
  fi
  echo "driver finished existing decision ${action}"
  exit 0
fi

if [ -e "$V25_RUN" ]; then
  echo "REFUSE: rl_pilot_v25 already exists without a switch decision"
  exit 1
fi

prepare_v25

LAST_ACTION=""
LAST_STEP=""
end=4
resume=""
while true; do
  run_chunk "$end" "$resume"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    write_status "$V25_RUN" "FAILED" "chunk_${end}" "torchrun exit ${rc}"
    echo "FAILED: rl_pilot_v25 chunk ${end} exit ${rc}"
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
  resume="${V25_RUN}/checkpoints/step_${LAST_STEP}"
done

finish_action "$LAST_STEP" "$LAST_ACTION"
echo "driver finished v25 action ${LAST_ACTION}"
