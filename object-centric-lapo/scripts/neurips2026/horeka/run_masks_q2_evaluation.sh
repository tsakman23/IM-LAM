# Run last stage of object-centric lapo masks pipeline for Q2: Does masking improve downstream performance and sample efficiency?
# Run last stage of object-centric lapo masks pipeline with latent action dim 128 and a sweep of {2k, 4k, 8k, 16k, 32k, 64k, (128k)} GT actions.
# Note, we already have 128k GT actions from Q1 evaluation. We reuse the stage 2 checkpoints for the sweeps.
# Run naming: object_centric_lapo_masks_q2_<env>_action_count<action_count>_seed<seed>
# Stage 4 checkpoint naming: "./checkpoints/object_centric_lapo_masks_q1_<env>_seed<seed>-4"

ACTION_COUNTS=(
    2048
    4096
    8192
    16384
    32768
    64000
    # 128000
)

AVAILABLE_ENVS=(
    # DCS non-distractor environments
    "dm_control/masked-cheetah-run-v0"
    "dm_control/masked-hopper-hop-v0"
    "dm_control/masked-humanoid-walk-v0"
    "dm_control/masked-walker-run-v0"
    # DCS distractor low environments
    "dm_control/masked-cheetah-run-distractor-low-v0"
    "dm_control/masked-hopper-hop-distractor-low-v0"
    "dm_control/masked-humanoid-walk-distractor-low-v0"
    "dm_control/masked-walker-run-distractor-low-v0"
    # DCS distractor hard environments
    # "dm_control/masked-cheetah-run-distractor-hard-v0"
    # "dm_control/masked-hopper-hop-distractor-hard-v0"
    # "dm_control/masked-humanoid-walk-distractor-hard-v0"
    # "dm_control/masked-walker-run-distractor-hard-v0"
    # DMW MT10 non-distractor environments
    "Meta-World/MT1-reach-v3"
    "Meta-World/MT1-push-v3"
    "Meta-World/MT1-pick-place-v3"
    "Meta-World/MT1-door-open-v3"
    "Meta-World/MT1-drawer-open-v3"
    "Meta-World/MT1-drawer-close-v3"
    "Meta-World/MT1-button-press-topdown-v3"
    "Meta-World/MT1-peg-insert-side-v3"
    "Meta-World/MT1-window-open-v3"
    "Meta-World/MT1-window-close-v3"
    # DMW MT10 distractor environments
    "Meta-World/masked-MT1-reach-v3"
    "Meta-World/masked-MT1-push-v3"
    "Meta-World/masked-MT1-pick-place-v3"
    "Meta-World/masked-MT1-door-open-v3"
    "Meta-World/masked-MT1-drawer-open-v3"
    "Meta-World/masked-MT1-drawer-close-v3"
    "Meta-World/masked-MT1-button-press-topdown-v3"
    "Meta-World/masked-MT1-peg-insert-side-v3"
    "Meta-World/masked-MT1-window-open-v3"
    "Meta-World/masked-MT1-window-close-v3"
    # TODO: Robosuite Two-Arm environments
)

# Map from environment name to action dim
declare -A ACTION_DIM_MAP=(
    ["dm_control/masked-cheetah-run-v0"]=6
    ["dm_control/masked-hopper-hop-v0"]=4
    ["dm_control/masked-humanoid-walk-v0"]=21
    ["dm_control/masked-walker-run-v0"]=6
    ["dm_control/masked-cheetah-run-distractor-low-v0"]=6
    ["dm_control/masked-hopper-hop-distractor-low-v0"]=4
    ["dm_control/masked-humanoid-walk-distractor-low-v0"]=21
    ["dm_control/masked-walker-run-distractor-low-v0"]=6
    # ["dm_control/masked-cheetah-run-distractor-hard-v0"]=6
    # ["dm_control/masked-hopper-hop-distractor-hard-v0"]=6
    # ["dm_control/masked-humanoid-walk-distractor-hard-v0"]=6
    # ["dm_control/masked-walker-run-distractor-hard-v0"]=6
    ["Meta-World/MT1-reach-v3"]=4
    ["Meta-World/MT1-push-v3"]=4
    ["Meta-World/MT1-pick-place-v3"]=4
    ["Meta-World/MT1-door-open-v3"]=4
    ["Meta-World/MT1-drawer-open-v3"]=4
    ["Meta-World/MT1-drawer-close-v3"]=4
    ["Meta-World/MT1-button-press-topdown-v3"]=4
    ["Meta-World/MT1-peg-insert-side-v3"]=4
    ["Meta-World/MT1-window-open-v3"]=4
    ["Meta-World/MT1-window-close-v3"]=4
    ["Meta-World/masked-MT1-reach-v3"]=4
    ["Meta-World/masked-MT1-push-v3"]=4
    ["Meta-World/masked-MT1-pick-place-v3"]=4
    ["Meta-World/masked-MT1-door-open-v3"]=4
    ["Meta-World/masked-MT1-drawer-open-v3"]=4
    ["Meta-World/masked-MT1-drawer-close-v3"]=4
    ["Meta-World/masked-MT1-button-press-topdown-v3"]=4
    ["Meta-World/masked-MT1-peg-insert-side-v3"]=4
    ["Meta-World/masked-MT1-window-open-v3"]=4
    ["Meta-World/masked-MT1-window-close-v3"]=4
)

SEEDS=(
    1
    2
    3
)

DCS_CONFIGS=configs/tasks/dcs_base.yaml
DMW_CONFIGS=configs/tasks/dmw_base.yaml

CACHE_DIR="/tmp/datasets"

for env in "${AVAILABLE_ENVS[@]}"; do
    action_dim="${ACTION_DIM_MAP[$env]}"
    if [[ $env == *"Meta-World/"* ]]; then
        CONFIG=${DMW_CONFIGS}
    else
        CONFIG=${DCS_CONFIGS}
    fi
    # enumerate action counts
    for i in "${!ACTION_COUNTS[@]}"; do
        action_count=${ACTION_COUNTS[i]}
        for seed in "${SEEDS[@]}"; do
            echo "Running evaluation for environment: $env and action count: $action_count with seed: $seed"
            run_id="object_centric_lapo_masks_q2_${env//\//_}_action_count${action_count}_seed${seed}"
            python scripts/run_pipeline_masks.py \
                --name $env \
                --config ${CONFIG} \
                --stages 5 \
                --run_id $run_id \
                --seed $seed \
                --wandb_project object-centric-lapo \
                --override notes="NeurIPS 2026 Q2 Evaluation" \
                --videosaur_checkpoint ./checkpoints/object_centric_lapo_masks_q1_${env//\//_}_seed${seed}-1/videosaur_best.pt \
                --slot_selection_path ./checkpoints/object_centric_lapo_masks_q1_${env//\//_}_seed${seed}-2/slot_selection.json \
                --lapo_checkpoint ./checkpoints/object_centric_lapo_masks_q1_${env//\//_}_seed${seed}-3/lapo_masks_latest.pt \
                --bc_phase_a_checkpoint ./checkpoints/object_centric_lapo_masks_q1_${env//\//_}_seed${seed}-4/bc_phase_a_latest.pt \
                --override task.action_dim=${action_dim} \
                --override cache_dir=${CACHE_DIR}/object_centric_lapo_masks/${env} \
                --override bc.subset_size=${action_count} \
                --override torch_compile=true
        done
    done
    rm -rf "${CACHE_DIR}/object_centric_lapo_masks/${env}"
done
