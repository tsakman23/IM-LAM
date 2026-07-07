#!/usr/bin/env bash

set -euo pipefail

# Number of timesteps and seeds for each split (matching the example)
NUM_STEPS_TRAIN=1000000
NUM_STEPS_TEST=100000
TRAIN_SEED=0
TEST_SEED=100000

# Explicit list of the 50 MT50 Meta-World v3 tasks
TASKS=(
  # "assembly-v3"
  # "basketball-v3"
  # "bin-picking-v3"
  # "box-close-v3"
  # "button-press-topdown-v3"
  # "button-press-topdown-wall-v3"
  # "button-press-v3"
  # "button-press-wall-v3"
  # "coffee-button-v3"
  # "coffee-pull-v3"
  # "coffee-push-v3"
  "dial-turn-v3"
  # "disassemble-v3"
  # "door-close-v3"
  # "door-lock-v3"
  "door-open-v3"
  # "door-unlock-v3"
  # "hand-insert-v3"
  # "drawer-close-v3"
  # "drawer-open-v3"
  "faucet-open-v3"
  # "faucet-close-v3"
  # "hammer-v3"
  # "handle-press-side-v3"
  # "handle-press-v3"
  # "handle-pull-side-v3"
  "handle-pull-v3"
  # "lever-pull-v3"
  # "pick-place-wall-v3"
  # "pick-out-of-hole-v3"
  "pick-place-v3"
  "plate-slide-v3"
  # "plate-slide-side-v3"
  # "plate-slide-back-v3"
  # "plate-slide-back-side-v3"
  "peg-insert-side-v3"
  # "peg-unplug-side-v3"
  # "soccer-v3"
  # "stick-push-v3"
  # "stick-pull-v3"
  "push-v3"
  "push-wall-v3"
  # "push-back-v3"
  # "reach-v3"
  # "reach-wall-v3"
  # "shelf-place-v3"
  "sweep-into-v3"
  # "sweep-v3"
  # "window-open-v3"
  # "window-close-v3"
)

for task in "${TASKS[@]}"; do
  ENV_ID="Meta-World/MT1-${task}"

  echo "=== Generating train split for ${ENV_ID} ==="
  python generate_dataset_huggingface.py \
    --env "${ENV_ID}" \
    --split train \
    --num_steps "${NUM_STEPS_TRAIN}" \
    --seed "${TRAIN_SEED}" \
    --push_to_hub

  echo "=== Generating test split for ${ENV_ID} ==="
  python generate_dataset_huggingface.py \
    --env "${ENV_ID}" \
    --split test \
    --num_steps "${NUM_STEPS_TEST}" \
    --seed "${TEST_SEED}" \
    --push_to_hub
done