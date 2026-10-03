#!/usr/bin/env bash
# rl_pilot_v21 starts fresh from the DPO champion.
# v16 learning rate and raw-gap advantage, without audio-tower LoRA.
# Stop when Robust is still above DPO at step 8, or when greedy reward falls.
# Never write v16 through v20, or the champion directory.
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
V16_RUN="/data/mega-asr/runs/rl_pilot_v16"
V19_RUN="/data/mega-asr/runs/rl_pilot_v19"
V20_RUN="/data/mega-asr/runs/rl_pilot_v20"
V21_RUN="/data/mega-asr/runs/rl_pilot_v21"
V21_CONFIG="configs/train/qwen3_asr_rl_v21.yaml"
HORIZON=24
PIPE_LOG="/data/mega-asr/logs/rl_pilot_v21_pipeline.log"

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
    "$V12_RUN"|"/data/mega-asr/runs/rl_pilot_v10"|"/data/mega-asr/runs/rl_pilot_v11"|"/data/mega-asr/runs/rl_pilot_v13"|"/data/mega-asr/runs/rl_pilot_v14"|"/data/mega-asr/runs/rl_pilot_v15"|"$V16_RUN"|"/data/mega-asr/runs/rl_pilot_v17"|"/data/mega-asr/runs/rl_pilot_v18"|"$V19_RUN"|"$V20_RUN"|"$CHAMPION"|"${CHAMPION}"/*)
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
  "$PYTHON" - "$REPO/$V21_CONFIG" << 'PY'
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
    "train_audio_projections: false",
    "expected_total_targets: 196",
)
missing = [item for item in required if item not in text]
forbidden = ("5.0e-6", "mode: unit", "mode: capped_gap", "train_audio_projections: true")
found = [item for item in forbidden if item in text]
if missing or found:
    raise SystemExit("config missing " + ", ".join(missing) + " forbidden " + ", ".join(found))
PY
}

run_chunk() {
  local end="$1"
  local resume="${2:-}"
  refuse_run "$V21_RUN"
  local log="/data/mega-asr/logs/rl_pilot_v21.log"
  local resume_args=()
  if [ -n "$resume" ]; then
    resume_args=(--resume-from-checkpoint "$resume")
  fi
  write_status "$V21_RUN" "TRAINING" "chunk_${end}"
  "$TORCHRUN" --standalone --nproc_per_node=4 train/train_rl.py \
    --manifest "$MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --wer-manifest "$WER_MANIFEST" \
    --config "$V21_CONFIG" \
    --output-dir "$V21_RUN" \
    --sample-strategy degraded \
    --probe-decision "${V21_RUN}/search_probe.json" \
    --max-steps "$end" \
    --save-steps 4 \
    --eval-steps 4 \
    --no-export-merged-on-finish \
    "${resume_args[@]}" \
    >> "$log" 2>&1
}

score_step() {
  local step="$1"
  write_status "$V21_RUN" "EVALUATING" "validation_step_${step}"
  RL_RUN_DIR="$V21_RUN" RL_CONFIG="${REPO}/${V21_CONFIG}" RL_METHOD="rl_pilot_v21_step_${step}" \
    bash "${REPO}/scripts/score_rl_pilot_checkpoint.sh" "$step"
}

decide() {
  local step="$1"
  "$PYTHON" "${REPO}/train/rl_v16_decision.py" \
    --run-dir "$V21_RUN" \
    --step "$step" \
    --horizon "$HORIZON"
}

export_passed() {
  local step="$1"
  local dest="${V21_RUN}/merged_base"
  refuse_run "$dest"
  if [[ "$dest" == /data/mega-asr/runs/dpo_pilot_v2* ]]; then
    echo "REFUSE: export destination is the champion"
    exit 1
  fi
  "$PYTHON" train/train_rl.py \
    --export-merged \
    --config "$V21_CONFIG" \
    --checkpoint-dir "${V21_RUN}/checkpoints/step_${step}" \
    --output-dir "$dest"
}

finish_action() {
  local step="$1"
  local action="$2"
  if [ "$action" = "passed" ]; then
    export_passed "$step" || {
      write_status "$V21_RUN" "FAILED" "export" "export failed"
      exit 1
    }
    write_status "$V21_RUN" "DONE" "gates_ready" "train_status=PASSED"
    echo "DONE: rl_pilot_v21 gate PASSED"
    return 0
  fi
  write_status "$V21_RUN" "DONE" "gates_ready" "action=${action}"
  echo "DONE: rl_pilot_v21 action ${action}"
}

prepare_v21() {
  if [ -e "$V21_RUN" ]; then
    echo "REFUSE: rl_pilot_v21 already exists"
    exit 1
  fi
  if [ ! -f "${V12_RUN}/search_probe.json" ]; then
    echo "REFUSE: v12 probe is missing"
    exit 1
  fi
  if ! grep -q "filter_lora_targets" "${REPO}/train/train_rl.py"; then
    echo "REFUSE: server trainer does not filter LoRA targets"
    exit 1
  fi
  mkdir -p "$V21_RUN"
  cp -p "${V12_RUN}/search_probe.json" "${V21_RUN}/search_probe.json"
  if ! PYTHONPATH="${REPO}/train:${REPO}" "$PYTHON" - << 'PY'
import json
from pathlib import Path
from rl_local_winner import local_winner_index
from rl_lora_targets import filter_lora_targets

run = Path("/data/mega-asr/runs/rl_pilot_v21")
probe = json.loads((run / "search_probe.json").read_text(encoding="utf-8"))
if probe.get("decision") != "GO_GRPO" or int(probe.get("n_prompts", 0)) != 128:
    raise SystemExit("v21 probe is not the 128-prompt GO_GRPO file")
for name in ("v16", "v19", "v20"):
    decision = json.loads(Path(f"/data/mega-asr/runs/rl_pilot_{name}/switch_decision.json").read_text(encoding="utf-8"))
    if decision.get("action") != "stop":
        raise SystemExit(f"{name} decision is not stop")
names = [
    "audio_tower.conv_out",
    "audio_tower.proj1",
    "audio_tower.proj2",
    "model.layers.0.self_attn.q_proj",
]
if filter_lora_targets(names, False) != ["model.layers.0.self_attn.q_proj"]:
    raise SystemExit("audio projection filter failed")
if filter_lora_targets(names, True) != names:
    raise SystemExit("default LoRA target filter dropped names")
index, status = local_winner_index(
    [
        "Ayr has won eight of the last nine meetings in this series.",
        "Iowa has won eight of the last nine meetings in this series.",
        "In the difficult moments we recognize our thirst for fulfillment.",
    ],
    [0.40, 0.46, 0.95],
    language="en",
)
if index != 1 or status != "update":
    raise SystemExit(f"local winner check failed: {index} {status}")
trainer = Path("/data/mega-asr/repo/train/train_rl.py").read_text(encoding="utf-8")
if "train_audio_projections" not in trainer or trainer.count("filter_lora_targets") < 2:
    raise SystemExit("server trainer is missing the audio-projection filter")
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
    "train_audio_projections": False,
    "expected_lora_targets": 196,
    "horizon": 24,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("v21 baseline ok")
PY
  then
    rm -rf "$V21_RUN"
    echo "REFUSE: v21 baseline check failed"
    exit 1
  fi
}

score_and_decide() {
  local step
  step=$("$PYTHON" "${REPO}/scripts/score_rl_greedy_held_out.py" --resolve-step --run-dir "$V21_RUN") || {
    write_status "$V21_RUN" "FAILED" "validation" "saved checkpoint incomplete"
    exit 1
  }
  score_step "$step" || {
    write_status "$V21_RUN" "FAILED" "validation_step_${step}" "score failed"
    exit 1
  }
  local action
  action=$(decide "$step") || {
    write_status "$V21_RUN" "FAILED" "switch" "decision failed"
    exit 1
  }
  LAST_ACTION="$action"
  LAST_STEP="$step"
  echo "rl_pilot_v21 step ${step} action ${action}"
}

assert_config

if [ -f "${V21_RUN}/switch_decision.json" ]; then
  action=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("action",""))' "${V21_RUN}/switch_decision.json")
  echo "Existing v21 decision: ${action}"
  if [ "$action" = "continue" ]; then
    echo "REFUSE: v21 already has a continue decision; not starting another chunk from a rerun"
    exit 1
  fi
  echo "driver finished existing decision ${action}"
  exit 0
fi

if [ -e "$V21_RUN" ]; then
  echo "REFUSE: rl_pilot_v21 already exists without a switch decision"
  exit 1
fi

prepare_v21

LAST_ACTION=""
LAST_STEP=""
end=4
resume=""
while true; do
  run_chunk "$end" "$resume"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    write_status "$V21_RUN" "FAILED" "chunk_${end}" "torchrun exit ${rc}"
    echo "FAILED: rl_pilot_v21 chunk ${end} exit ${rc}"
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
  resume="${V21_RUN}/checkpoints/step_${LAST_STEP}"
done

finish_action "$LAST_STEP" "$LAST_ACTION"
echo "driver finished v21 action ${LAST_ACTION}"
