#!/usr/bin/env bash
# rl_pilot_v28 resumes the read-only v27 step-4 checkpoint.
# After resume, the configured learning rate replaces the restored rate.
# The reference candidate, signed edits, raw gap, and audio targets stay.
# Never write v27 or the champion directory.
# Do not delete the v27 stop decision or the v24 continue decision.
set -u

export PYTHONUNBUFFERED=1

REPO="/data/mega-asr/repo"
PYTHON="/data/mega-asr/venv/bin/python"
TORCHRUN="/data/mega-asr/venv/bin/torchrun"
MANIFEST="/data/mega-asr/manifests/pilot_rl.jsonl"
VAL_MANIFEST="/data/mega-asr/manifests/rl_val_pool.jsonl"
WER_MANIFEST="/data/mega-asr/manifests/validation.jsonl"
CHAMPION="/data/mega-asr/runs/dpo_pilot_v2/merged_base"
V27_RUN="/data/mega-asr/runs/rl_pilot_v27"
V27_STEP4="/data/mega-asr/runs/rl_pilot_v27/checkpoints/step_4"
V28_RUN="/data/mega-asr/runs/rl_pilot_v28"
V28_CONFIG="configs/train/qwen3_asr_rl_v28.yaml"
HORIZON=24
PIPE_LOG="/data/mega-asr/logs/rl_pilot_v28_pipeline.log"

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
    "/data/mega-asr/runs/rl_pilot_v10"|"/data/mega-asr/runs/rl_pilot_v11"|"/data/mega-asr/runs/rl_pilot_v12"|"/data/mega-asr/runs/rl_pilot_v13"|"/data/mega-asr/runs/rl_pilot_v14"|"/data/mega-asr/runs/rl_pilot_v15"|"/data/mega-asr/runs/rl_pilot_v16"|"/data/mega-asr/runs/rl_pilot_v17"|"/data/mega-asr/runs/rl_pilot_v18"|"/data/mega-asr/runs/rl_pilot_v19"|"/data/mega-asr/runs/rl_pilot_v20"|"/data/mega-asr/runs/rl_pilot_v21"|"/data/mega-asr/runs/rl_pilot_v22"|"/data/mega-asr/runs/rl_pilot_v23"|"/data/mega-asr/runs/rl_pilot_v24"|"/data/mega-asr/runs/rl_pilot_v25"|"/data/mega-asr/runs/rl_pilot_v26"|"$V27_RUN"|"$CHAMPION"|"${CHAMPION}"/*)
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
  "$PYTHON" - "$REPO/$V28_CONFIG" << 'PY'
import sys
from pathlib import Path
text = Path(sys.argv[1]).read_text(encoding="utf-8")
required = (
    "learning_rate: 5.0e-6",
    "apply_learning_rate_on_resume: true",
    "include_reference_candidate: true",
    "policy_token_mask: signed_edits",
    "mode: raw_gap",
    "local_max_relative: 0.35",
    "beta: 0.04",
    "max_raw_kl: 5.0e-4",
    "max_steps: 24",
    "sample_strategy: degraded",
    "train_audio_projections: true",
    "expected_total_targets: 199",
)
missing = [item for item in required if item not in text]
forbidden = (
    "learning_rate: 1.0e-5",
    "learning_rate: 2.0e-5",
    "mode: unit",
    "mode: capped_gap",
    "mode: fixed",
    "train_audio_projections: false",
    "policy_token_mask: all",
    "policy_token_mask: changes_only",
    "degraded_skip_regressed",
    "include_reference_candidate: false",
    "apply_learning_rate_on_resume: false",
)
found = [item for item in forbidden if item in text]
if missing or found:
    raise SystemExit("config missing " + ", ".join(missing) + " forbidden " + ", ".join(found))
PY
}

run_chunk() {
  local end="$1"
  local resume="${2:-}"
  refuse_run "$V28_RUN"
  local log="/data/mega-asr/logs/rl_pilot_v28.log"
  local resume_args=()
  if [ -n "$resume" ]; then
    resume_args=(--resume-from-checkpoint "$resume")
  fi
  write_status "$V28_RUN" "TRAINING" "chunk_${end}"
  "$TORCHRUN" --standalone --nproc_per_node=4 train/train_rl.py \
    --manifest "$MANIFEST" \
    --val-manifest "$VAL_MANIFEST" \
    --wer-manifest "$WER_MANIFEST" \
    --config "$V28_CONFIG" \
    --output-dir "$V28_RUN" \
    --sample-strategy degraded \
    --probe-decision "${V28_RUN}/search_probe.json" \
    --max-steps "$end" \
    --save-steps 4 \
    --eval-steps 4 \
    --no-export-merged-on-finish \
    "${resume_args[@]}" \
    >> "$log" 2>&1
}

score_step() {
  local step="$1"
  write_status "$V28_RUN" "EVALUATING" "validation_step_${step}"
  RL_RUN_DIR="$V28_RUN" RL_CONFIG="${REPO}/${V28_CONFIG}" RL_METHOD="rl_pilot_v28_step_${step}" \
    bash "${REPO}/scripts/score_rl_pilot_checkpoint.sh" "$step"
}

decide() {
  local step="$1"
  "$PYTHON" "${REPO}/train/rl_v16_decision.py" \
    --run-dir "$V28_RUN" \
    --step "$step" \
    --horizon "$HORIZON"
}

export_passed() {
  local step="$1"
  local dest="${V28_RUN}/merged_base"
  refuse_run "$dest"
  if [[ "$dest" == /data/mega-asr/runs/dpo_pilot_v2* ]]; then
    echo "REFUSE: export destination is the champion"
    exit 1
  fi
  "$PYTHON" train/train_rl.py \
    --export-merged \
    --config "$V28_CONFIG" \
    --checkpoint-dir "${V28_RUN}/checkpoints/step_${step}" \
    --output-dir "$dest"
}

finish_action() {
  local step="$1"
  local action="$2"
  if [ "$action" = "passed" ]; then
    export_passed "$step" || {
      write_status "$V28_RUN" "FAILED" "export" "export failed"
      exit 1
    }
    write_status "$V28_RUN" "DONE" "gates_ready" "train_status=PASSED"
    echo "DONE: rl_pilot_v28 gate PASSED"
    return 0
  fi
  write_status "$V28_RUN" "DONE" "gates_ready" "action=${action}"
  echo "DONE: rl_pilot_v28 action ${action}"
}

prepare_v28() {
  if [ -e "$V28_RUN" ]; then
    echo "REFUSE: rl_pilot_v28 already exists"
    exit 1
  fi
  if [ ! -f "${V27_RUN}/search_probe.json" ] || [ ! -f "${V27_RUN}/loss_log.jsonl" ]; then
    echo "REFUSE: v27 probe or loss log is missing"
    exit 1
  fi
  if [ ! -f "${V27_STEP4}/adapter/adapter_model.safetensors" ] || [ ! -f "${V27_STEP4}/optimizer.pt" ] || [ ! -f "${V27_STEP4}/scheduler.pt" ]; then
    echo "REFUSE: v27 step 4 checkpoint is incomplete"
    exit 1
  fi
  if [ ! -f "${V27_RUN}/switch_decision.json" ] || [ ! -f "${V27_RUN}/gate_step_4.json" ] || [ ! -f "${V27_RUN}/gate_step_7.json" ]; then
    echo "REFUSE: v27 decision or gate is missing"
    exit 1
  fi
  mkdir -p "$V28_RUN"
  cp -p "${V27_RUN}/search_probe.json" "${V28_RUN}/search_probe.json"
  if ! PYTHONPATH="${REPO}/train:${REPO}" "$PYTHON" - << 'PY'
import json
from pathlib import Path
from collections import Counter
from rl_policy_mask import signed_edit_loss_args, signed_edit_update
from rl_sample_strategy import is_degraded_row, row_scenario
from rl_reference_candidate import append_reference_candidate
from rl_local_winner import local_winner_index
from train_rl import compute_sequence_reward

run = Path("/data/mega-asr/runs/rl_pilot_v28")
v27 = Path("/data/mega-asr/runs/rl_pilot_v27")
kept = []
for line in (v27 / "loss_log.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    if int(row.get("global_step", -1)) <= 4:
        kept.append(row)
(run / "loss_log.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept),
    encoding="utf-8",
)
probe = json.loads((run / "search_probe.json").read_text(encoding="utf-8"))
if probe.get("decision") != "GO_GRPO" or int(probe.get("n_prompts", 0)) != 128:
    raise SystemExit("v28 probe is not the 128-prompt GO_GRPO file")
for name in ("v16", "v20", "v21", "v22", "v23", "v25", "v26", "v27"):
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
decision = json.loads((v27 / "switch_decision.json").read_text(encoding="utf-8"))
if decision.get("train_status") != "STOPPED_KL" or int(decision.get("scored_step", -1)) != 7:
    raise SystemExit("v27 decision is not the step-7 KL stop")
if abs(float(decision["held_out_reward_improvement"]) - 0.0002) > 1e-9:
    raise SystemExit("v27 decision reward is not +0.0002")
if float(decision["robust_error_rate_increase"]) <= 0:
    raise SystemExit("v27 decision robust is not above DPO")
gate7 = json.loads((v27 / "gate_step_7.json").read_text(encoding="utf-8"))
metrics7 = gate7["metrics"]
if abs(float(metrics7["held_out_reward_improvement"]) - 0.0002) > 1e-9:
    raise SystemExit("v27 step 7 reward improvement is not +0.0002")
if float(metrics7["robust_error_rate_increase"]) <= 0:
    raise SystemExit("v27 step 7 robust is not above DPO")
gate4 = json.loads((v27 / "gate_step_4.json").read_text(encoding="utf-8"))
metrics4 = gate4["metrics"]
if abs(float(metrics4["held_out_reward_improvement"]) - 0.0001) > 1e-9:
    raise SystemExit("v27 step 4 reward improvement is not +0.0001")
if float(metrics4["robust_error_rate_increase"]) >= 0:
    raise SystemExit("v27 step 4 robust is not below DPO")
step0 = None
step4 = None
mass = None
max_step = -1
for row in kept:
    step = int(row.get("global_step", -1))
    max_step = max(max_step, step)
    if step == 0 and row.get("val_decode") == "greedy":
        step0 = float(row["val_mean_reward"])
    if step == 4 and row.get("val_decode") == "greedy":
        step4 = float(row["val_mean_reward"])
    if "cumulative_reward_mass" in row:
        mass = float(row["cumulative_reward_mass"])
if max_step != 4:
    raise SystemExit(f"truncated log max step is {max_step}")
if step0 is None or abs(step0 - 0.8773) > 1e-6:
    raise SystemExit(f"step0 reward is {step0}")
if step4 is None or abs(step4 - 0.8774) > 1e-6:
    raise SystemExit(f"step4 reward is {step4}")
if mass is None or abs(mass - 16.727412) > 1e-6:
    raise SystemExit(f"copied mass is {mass}")
manifest_rows = [
    json.loads(line)
    for line in Path("/data/mega-asr/manifests/pilot_rl.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
selected = [row for row in manifest_rows if is_degraded_row(row)]
counts = Counter(row_scenario(row) for row in selected)
if counts["clean"] or len(selected) != 2000:
    raise SystemExit(f"v28 epoch size {len(selected)} counts {dict(counts)}")
if counts["noise"] < 1 or counts["recording"] < 1 or counts["distortion"] < 1 or counts["dropout"] < 1:
    raise SystemExit(f"v28 epoch dropped a degraded scenario: {dict(counts)}")
close_texts, close_rewards = append_reference_candidate(
    ["the cat sat", "the cat sit"],
    [0.40, 0.41],
    "the cat sits",
    1.0,
)
close_winner, close_status = local_winner_index(
    close_texts,
    close_rewards,
    language="en",
    min_improvement=0.02,
    max_relative=0.35,
)
if close_status != "update" or close_texts[close_winner] != "the cat sits":
    raise SystemExit(f"close reference was not selected: {close_status} {close_winner}")
far_texts, far_rewards = append_reference_candidate(
    ["the cat sat", "the cat sit"],
    [0.40, 0.41],
    "one two three four five six seven eight nine ten",
    1.0,
)
far_winner, far_status = local_winner_index(
    far_texts,
    far_rewards,
    language="en",
    min_improvement=0.02,
    max_relative=0.35,
)
if far_winner is not None or far_status != "no_improvement":
    raise SystemExit(f"far reference was selected: {far_status} {far_winner}")
for language, sentence in (
    ("en", "In the dim hallway, he recognized her purse by the silver clasp and faded ribbon."),
    ("zh", "先把这些证据发给人家。"),
):
    reward, components, error_rate = compute_sequence_reward(sentence, sentence, language)
    penalties = [components[name] for name in ("empty", "repeat", "too_long", "hallucination")]
    if abs(reward - 1.0) > 1e-9 or abs(error_rate) > 1e-9 or any(penalties):
        raise SystemExit(f"exact reference reward {language} {reward} {components} {error_rate}")
deletion = signed_edit_update([1, 2, 3], [1, 3], 0.25)
if deletion.action != "update":
    raise SystemExit(f"deletion-only action {deletion.action}")
same = signed_edit_update([10, 11, 12], [10, 11, 12], 0.25)
if same.action != "skip":
    raise SystemExit(f"identical pair did not skip: {same}")
loss = signed_edit_loss_args([1, 2, 3], [1, 3], 0.25)
if loss.action != "update" or loss.sequence_advantage != (-0.25, 0.25):
    raise SystemExit(f"signed loss args {loss}")
trainer = Path("/data/mega-asr/repo/train/train_rl.py").read_text(encoding="utf-8")
if trainer.count("apply_configured_learning_rate") < 2 or "apply_learning_rate_on_resume" not in trainer:
    raise SystemExit("server trainer is missing the resume learning-rate hook")
if "append_reference_candidate(" not in trainer or "reference_text=reference_text" not in trainer:
    raise SystemExit("server trainer is missing the reference candidate")
if "include_reference_candidate" not in trainer or "def decide_rl_stop" not in trainer:
    raise SystemExit("server trainer is missing the reference flag or stop table")
champion = Path("/data/mega-asr/runs/dpo_pilot_v2/merged_base/model.safetensors")
stat = champion.stat()
if stat.st_size != 4076190936:
    raise SystemExit(f"champion bytes changed: {stat.st_size}")
(run / "source.json").write_text(json.dumps({
    "source_model": "/data/mega-asr/runs/dpo_pilot_v2/merged_base",
    "resume_checkpoint": "/data/mega-asr/runs/rl_pilot_v27/checkpoints/step_4",
    "probe_source": "/data/mega-asr/runs/rl_pilot_v27/search_probe.json",
    "champion_bytes": stat.st_size,
    "champion_mtime": stat.st_mtime,
    "local_max_relative": 0.35,
    "learning_rate": 5.0e-6,
    "apply_learning_rate_on_resume": True,
    "advantage_mode": "raw_gap",
    "policy_token_mask": "signed_edits",
    "sample_strategy": "degraded",
    "include_reference_candidate": True,
    "epoch_rows": len(selected),
    "train_audio_projections": True,
    "expected_lora_targets": 199,
    "horizon": 24,
    "step0_val_reward": step0,
    "copied_reward_mass": mass,
    "copied_loss_log": True,
}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("v28 baseline ok", len(selected), mass)
PY
  then
    rm -rf "$V28_RUN"
    echo "REFUSE: v28 baseline check failed"
    exit 1
  fi
}

score_and_decide() {
  local step
  step=$("$PYTHON" "${REPO}/scripts/score_rl_greedy_held_out.py" --resolve-step --run-dir "$V28_RUN") || {
    write_status "$V28_RUN" "FAILED" "validation" "saved checkpoint incomplete"
    exit 1
  }
  score_step "$step" || {
    write_status "$V28_RUN" "FAILED" "validation_step_${step}" "score failed"
    exit 1
  }
  local action
  action=$(decide "$step") || {
    write_status "$V28_RUN" "FAILED" "switch" "decision failed"
    exit 1
  }
  LAST_ACTION="$action"
  LAST_STEP="$step"
  echo "rl_pilot_v28 step ${step} action ${action}"
}

assert_config

if [ -f "${V28_RUN}/switch_decision.json" ]; then
  action=$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("action",""))' "${V28_RUN}/switch_decision.json")
  echo "Existing v28 decision: ${action}"
  if [ "$action" = "continue" ]; then
    echo "REFUSE: v28 already has a continue decision; not starting another chunk from a rerun"
    exit 1
  fi
  echo "driver finished existing decision ${action}"
  exit 0
fi

if [ -e "$V28_RUN" ]; then
  echo "REFUSE: rl_pilot_v28 already exists without a switch decision"
  exit 1
fi

prepare_v28

LAST_ACTION=""
LAST_STEP=""
end=8
resume="$V27_STEP4"
while true; do
  run_chunk "$end" "$resume"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    write_status "$V28_RUN" "FAILED" "chunk_${end}" "torchrun exit ${rc}"
    echo "FAILED: rl_pilot_v28 chunk ${end} exit ${rc}"
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
  resume="${V28_RUN}/checkpoints/step_${LAST_STEP}"
done

finish_action "$LAST_STEP" "$LAST_ACTION"
echo "driver finished v28 action ${LAST_ACTION}"
