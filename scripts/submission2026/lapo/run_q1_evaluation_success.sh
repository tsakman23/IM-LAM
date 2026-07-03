# Run rollout evaluation for Q1 DMW success rates; this does not retrain.
# Load Stage 3 checkpoints from the corresponding non-_success runs with latent action dim 128 and 128k GT actions.
# Run naming: lapo_q1_<env>_seed<seed>_success

AVAILABLE_ENVS=(
    # DCS non-distractor environments
    # "dm_control/masked-cheetah-run-v0"
    # "dm_control/masked-hopper-hop-v0"
    # "dm_control/masked-humanoid-walk-v0"
    # "dm_control/masked-walker-run-v0"
    # DCS distractor low environments
    # "dm_control/masked-cheetah-run-distractor-low-v0"
    # "dm_control/masked-hopper-hop-distractor-low-v0"
    # "dm_control/masked-humanoid-walk-distractor-low-v0"
    # "dm_control/masked-walker-run-distractor-low-v0"

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

)

SEEDS=(
    1
    2
    3
)

DCS_CONFIGS=(
    "lapo_bc_dcs_stage_1"
    "lapo_bc_dcs_stage_2"
    "lapo_bc_dcs_stage_3"
)
DMW_CONFIGS=(
    "lapo_bc_dmw_stage_1"
    "lapo_bc_dmw_stage_2"
    "lapo_bc_dmw_stage_3"
)

for env in "${AVAILABLE_ENVS[@]}"; do
    if [[ $env == *"Meta-World/"* ]]; then
        CONFIG=("${DMW_CONFIGS[@]}")
    else
        CONFIG=("${DCS_CONFIGS[@]}")
    fi
    for seed in "${SEEDS[@]}"; do
        run_id="lapo_q1_${env//\//_}_seed${seed}"
        echo "Running evaluation for environment: $env with seed: $seed"
        python scripts/submission2026/eval_lapo.py \
            -cn ${CONFIG[2]} \
            run_id=${run_id}_success \
            env.name=$env \
            logger.mode=online \
            logger.notes="Submission 2026 Q1 Evaluation Success" \
            trainer.compile=True \
            fabric.precision=bf16-mixed \
            trainer.previous_stage_checkpoint=./checkpoints/${run_id}-3
    done
done