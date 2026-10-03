#!/usr/bin/env bash
# v12 uses winner advantage = raw reward gap at lr 1e-5.
# If the logged pre-clip grad_norm stays above 1.5, v13 starts from the DPO
# champion with a fixed winner advantage of 0.10. v10, v11, and the champion
# directory are never written.
set -u

REPO="/data/mega-asr/repo"
PYTHON="/data/mega-asr/venv/bin/python"
TORCHRUN="/data/mega-asr/venv/bin/torchrun"
MANIFEST="/data/mega-asr/manifests/pilot_rl.jsonl"
VAL_MANIFEST="/data/mega-asr/manifests/rl_val_pool.jsonl"
WER_MANIFEST="/data/mega-asr/manifests/validation.jsonl"
CHAMPION="/data/mega-asr/runs/dpo_pilot_v2/merged_base"
V12_RUN="/data/mega-asr/runs/rl_pilot_v12"
V13_RUN="/data/mega-asr/runs/rl_pilot_v13"
V12_CONFIG="configs/train/qwen3_asr_rl_v12.yaml"
V13_CONFIG="configs/train/qwen3_asr_rl_v13.yaml"
PIPE_LOG="/data/mega-asr/logs/rl_pilot_v12_pipeline.log"

mkdir -p /data/mega-asr/logs "$V12_RUN"
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

run_chunk() {
  local run="$1"
  local config="$2"
  local end="$3"
  local resume="${4:-}"
  local log="/data/mega-asr/logs/$(basename "$run").log"
  local resume_args=()
  if [ -n "$resume" ]; then
    resume_args=(--resume-from-checkpoint "$resume")
  fi
  write_status "$run" "TRAINING" "chunk_${end}"
  "$TORCHRUN" --standalone --nproc_per_node=4 train/train_rl.py \
    --manifest "$MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --wer-manifest "$WER_MANIFEST" \
    --config "$config" \
    --output-dir "$run" \
    --sample-strategy degraded \
    --probe-decision "${run}/search_probe.json" \
    --max-steps "$end" \
    --save-steps 4 \
    --eval-steps 4 \
    --no-export-merged-on-finish \
    "${resume_args[@]}" \
    >> "$log" 2>&1
}

score_step() {
  local run="$1"
  local config="$2"
  local step="$3"
  write_status "$run" "EVALUATING" "validation_step_${step}"
  RL_RUN_DIR="$run" RL_CONFIG="${REPO}/${config}" RL_METHOD="$(basename "$run")_step_${step}" \
    bash "${REPO}/scripts/score_rl_pilot_checkpoint.sh" "$step"
}

decide() {
  local run="$1"
  local step="$2"
  local allow_switch="$3"
  RUN_DIR="$run" SCORED_STEP="$step" ALLOW_SWITCH="$allow_switch" "$PYTHON" - << 'PY'
import json, os, statistics
from pathlib import Path
run = Path(os.environ["RUN_DIR"])
step = int(os.environ["SCORED_STEP"])
allow_switch = os.environ["ALLOW_SWITCH"] == "yes"
norms = []
for line in (run / "loss_log.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if "grad_norm" in row:
        norms.append(float(row["grad_norm"]))
gate = json.loads((run / f"gate_step_{step}.json").read_text(encoding="utf-8"))
metrics = gate["metrics"]
improvement = float(metrics["held_out_reward_improvement"])
robust = float(metrics["robust_error_rate_increase"])
state = json.loads((run / "pipeline_state.json").read_text(encoding="utf-8"))
status = str(state.get("status", ""))
median = statistics.median(norms) if norms else None
if gate.get("gate_status") == "PASSED":
    action, reason = "passed", "gate passed"
elif robust >= 0.0005:
    action, reason = "stop", "robust increased"
elif median is None:
    action, reason = "failed", "no grad_norm rows"
elif median > 1.5:
    action = "switch" if allow_switch else "stop"
    reason = "pre-clip grad_norm still above 1.5"
elif status == "CHUNK_DONE" and step < 8 and improvement >= 0.001:
    action, reason = "continue", "unclipped and greedy reward is moving"
else:
    action, reason = "stop", "unclipped without a passing gate"
payload = {
    "action": action,
    "reason": reason,
    "scored_step": step,
    "train_status": status,
    "median_grad_norm": median,
    "held_out_reward_improvement": improvement,
    "gate_status": gate.get("gate_status"),
}
(run / "switch_decision.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(action)
PY
}

export_passed() {
  local run="$1"
  local config="$2"
  local step="$3"
  local dest="${run}/merged_base"
  if [[ "$dest" == /data/mega-asr/runs/dpo_pilot_v2* ]]; then
    echo "REFUSE: export destination is the champion"
    exit 1
  fi
  "$PYTHON" train/train_rl.py \
    --export-merged \
    --config "$config" \
    --checkpoint-dir "${run}/checkpoints/step_${step}" \
    --output-dir "$dest"
}

run_new() {
  local run="$1"
  local config="$2"
  local allow_switch="$3"
  local probe_source="${4:-}"
  mkdir -p "$run"
  if [ -n "$probe_source" ]; then
    cp "$probe_source" "${run}/search_probe.json"
    echo "Reused probe $(basename "$probe_source") for $(basename "$run")"
  fi
  if [ ! -f "${run}/search_probe.json" ]; then
    write_status "$run" "WAITING" "probe"
    echo "Starting search probe for $(basename "$run")"
    "$TORCHRUN" --standalone --nproc_per_node=4 train/train_rl.py \
      --manifest "$MANIFEST" \
      --val-manifest "$VAL_MANIFEST" \
      --wer-manifest "$WER_MANIFEST" \
      --config "$config" \
      --output-dir "$run" \
      --sample-strategy degraded \
      --probe-only \
      >> "/data/mega-asr/logs/$(basename "$run").log" 2>&1
  fi
  if [ ! -f "${run}/search_probe.json" ]; then
    write_status "$run" "FAILED" "probe" "search_probe.json missing"
    echo "FAILED: probe missing for $(basename "$run")"
    exit 1
  fi
  local decision
  decision=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("decision",""))' "${run}/search_probe.json")
  echo "Probe decision for $(basename "$run"): ${decision}"
  if [ "$decision" != "GO_GRPO" ]; then
    write_status "$run" "BLOCKED" "probe" "decision=${decision}"
    echo "Probe did not return GO_GRPO. Not training $(basename "$run")."
    exit 0
  fi

  run_chunk "$run" "$config" 4
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    write_status "$run" "FAILED" "chunk_4" "torchrun exit ${rc}"
    echo "FAILED: $(basename "$run") chunk 4 exit ${rc}"
    exit 1
  fi
  local step
  step=$("$PYTHON" "${REPO}/scripts/score_rl_greedy_held_out.py" --resolve-step --run-dir "$run") || {
    write_status "$run" "FAILED" "validation" "saved checkpoint incomplete"
    exit 1
  }
  score_step "$run" "$config" "$step" || {
    write_status "$run" "FAILED" "validation_step_${step}" "score failed"
    exit 1
  }
  local action
  action=$(decide "$run" "$step" "$allow_switch") || {
    write_status "$run" "FAILED" "switch" "decision failed"
    exit 1
  }
  echo "$(basename "$run") step ${step} action ${action}"
  if [ "$action" = "continue" ]; then
    run_chunk "$run" "$config" 8 "${run}/checkpoints/step_${step}"
    rc=$?
    if [ "$rc" -ne 0 ]; then
      write_status "$run" "FAILED" "chunk_8" "torchrun exit ${rc}"
      exit 1
    fi
    step=$("$PYTHON" "${REPO}/scripts/score_rl_greedy_held_out.py" --resolve-step --run-dir "$run") || exit 1
    score_step "$run" "$config" "$step" || exit 1
    action=$(decide "$run" "$step" "$allow_switch") || exit 1
    echo "$(basename "$run") step ${step} action ${action}"
  fi
  if [ "$action" = "passed" ]; then
    export_passed "$run" "$config" "$step" || {
      write_status "$run" "FAILED" "export" "export failed"
      exit 1
    }
    write_status "$run" "DONE" "gates_ready" "train_status=PASSED"
    echo "DONE: $(basename "$run") gate PASSED"
  else
    write_status "$run" "DONE" "gates_ready" "action=${action}"
    echo "DONE: $(basename "$run") action ${action}"
  fi
}

if [ -f "${V12_RUN}/switch_decision.json" ]; then
  action=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("action",""))' "${V12_RUN}/switch_decision.json")
  echo "Existing v12 decision: ${action}"
  if [ "$action" = "switch" ] && [ ! -f "${V13_RUN}/pipeline_state.json" ]; then
    run_new "$V13_RUN" "$V13_CONFIG" "no" "${V12_RUN}/search_probe.json"
  fi
  echo "driver finished existing decision ${action}"
  exit 0
fi

if [ -f "${V12_RUN}/pipeline_state.json" ]; then
  echo "REFUSE: rl_pilot_v12 already has pipeline_state.json and no switch decision"
  exit 1
fi

run_new "$V12_RUN" "$V12_CONFIG" "yes"
action=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("action",""))' "${V12_RUN}/switch_decision.json")
if [ "$action" = "switch" ]; then
  run_new "$V13_RUN" "$V13_CONFIG" "no" "${V12_RUN}/search_probe.json"
fi
echo "driver finished v12 action ${action}"
exit 0
