# Run stages 1-3 of object-centric lapo slots for Q4: Does masking increase robustness to new visual distractors?
# Run stages 1-3 with latent action dim 128 and 128k GT actions.
# Run naming: object_centric_lapo_slots_q4_<env>_seed<seed>

AVAILABLE_ENVS=(
    # DCS distractor low environments
    "dm_control/masked-cheetah-run-distractor-low-v0"
    # "dm_control/masked-hopper-hop-distractor-low-v0"
    # "dm_control/masked-humanoid-walk-distractor-low-v0"
    # "dm_control/masked-walker-run-distractor-low-v0"
)

# Map from environment name to action dim
declare -A ACTION_DIM_MAP=(
    ["dm_control/masked-cheetah-run-distractor-low-v0"]=6
    ["dm_control/masked-hopper-hop-distractor-low-v0"]=4
    ["dm_control/masked-humanoid-walk-distractor-low-v0"]=21
    ["dm_control/masked-walker-run-distractor-low-v0"]=6
)

SEEDS=(
    1
    #2
    #3
)

DCS_CONFIGS=configs/tasks/dcs_base.yaml
DMW_CONFIGS=configs/tasks/dmw_base.yaml

CACHE_DIR="/tmp/datasets_uuyes"

for env in "${AVAILABLE_ENVS[@]}"; do
    action_dim="${ACTION_DIM_MAP[$env]}"
    if [[ $env == *"Meta-World/"* ]]; then
        CONFIG=${DMW_CONFIGS}
    else
        CONFIG=${DCS_CONFIGS}
    fi
    for seed in "${SEEDS[@]}"; do
        echo "Running evaluation for environment: $env with seed: $seed"
        run_id="object_centric_lapo_slots_q4_${env//\//_}_test2_seed${seed}"
        python scripts/run_pipeline_slots_ood.py \
            --name $env \
            --config ${CONFIG} \
            --stages 1 2 3 \
            --run_id $run_id \
            --seed $seed \
            --wandb_project object-centric-lapo \
            --override notes="NeurIPS 2026 Q4 Evaluation" \
            --override task.action_dim=${action_dim} \
            --override cache_dir=${CACHE_DIR}/object_centric_lapo_slots/${env} \
            --override torch_compile=true
    done
    env_hard="${env/distractor-low/distractor-hard}"
    rm -rf "${CACHE_DIR}/object_centric_lapo_slots/${env}"
    rm -rf "${CACHE_DIR}/object_centric_lapo_slots/${env_hard}"
done