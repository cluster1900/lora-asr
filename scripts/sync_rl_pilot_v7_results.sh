#!/usr/bin/env bash
# Pull rl_pilot_v7 logs, gates, predictions, and adapter configs from the V100.
# Optimizer states and the merged safetensors stay on the server.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${ROOT}/results/rl_pilot_v7"
mkdir -p "$DEST"

rsync -av \
  --exclude 'checkpoints/*/optimizer.pt' \
  --exclude 'checkpoints/*/scheduler.pt' \
  --exclude 'checkpoints/*/rng_state_rank_*.pt' \
  --exclude 'merged_base/model*.safetensors' \
  --exclude 'merged_base/*.bin' \
  v100:/data/mega-asr/runs/rl_pilot_v7/ \
  "${DEST}/"

mkdir -p "${ROOT}/logs"
rsync -av \
  v100:/data/mega-asr/logs/rl_pilot_v7.log \
  v100:/data/mega-asr/logs/rl_pilot_v7_pipeline.log \
  "${ROOT}/logs/" || true

echo "Synced rl_pilot_v7 results to ${DEST}"
