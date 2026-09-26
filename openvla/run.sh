#!/usr/bin/env bash
set -euo pipefail

# conda activate openvla

cd "$(dirname "$0")"

# The checkpoint must be a local folder (see README), e.g. downloaded with:
#   python vla_cache_scripts/download_model_local.py --model_id openvla/openvla-7b-finetuned-libero-spatial
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint checkpoints/openvla-7b-finetuned-libero-spatial \
  --task_suite_name libero_spatial \
  --use_vla_cache True \
  --action_margin_threshold 0.65
