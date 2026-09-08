# Run full object-centric lapo slots pipeline for Q1: does masking improve linear separability / alignment of latent actions by removing distractor content?
# Run full object-centric lapo slots pipeline with latent action dim 128 and 128k GT actions.
# Run naming: object_centric_lapo_slots_q1_<env>_seed<seed>

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
    for seed in "${SEEDS[@]}"; do
        echo "Running evaluation for environment: $env with seed: $seed"
        run_id="object_centric_lapo_slots_q1_${env//\//_}_seed${seed}"
        python scripts/run_pipeline_slots.py \
            --name $env \
            --config ${CONFIG} \
            --stages 1 2 3 4 5 \
            --run_id $run_id \
            --seed $seed \
            --wandb_project object-centric-lapo \
            --override notes="NeurIPS 2026 Q1 Evaluation" \
            --override task.action_dim=${action_dim} \
            --override cache_dir=${CACHE_DIR}/object_centric_lapo_slots/${env} \
            --override torch_compile=true
    done
    rm -rf "${CACHE_DIR}/object_centric_lapo_slots/${env}"
done