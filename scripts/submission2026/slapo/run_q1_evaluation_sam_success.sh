# Run rollout evaluation for Q1 DMW success rates; this does not retrain.
# Load Stage 3 checkpoints from the corresponding non-_success runs with latent action dim 128 and 128k GT actions.
# Run naming: slapo_q1_sam_<env>_seed<seed>_success

AVAILABLE_ENVS=(
    # DCS non-distractor environments
    # "dm_control/sam-masked-cheetah-run-v0"
    # "dm_control/sam-masked-hopper-hop-v0"
    # "dm_control/sam-masked-humanoid-walk-v0"
    # "dm_control/sam-masked-walker-run-v0"
    # DCS distractor low environments
    # "dm_control/sam-masked-cheetah-run-distractor-low-v0"
    # "dm_control/sam-masked-hopper-hop-distractor-low-v0"
    # "dm_control/sam-masked-humanoid-walk-distractor-low-v0"
    # "dm_control/sam-masked-walker-run-distractor-low-v0"
    # DMW MT10 non-distractor environments
    "Meta-World/sam-MT1-reach-v3"
    "Meta-World/sam-MT1-push-v3"
    "Meta-World/sam-MT1-pick-place-v3"
    "Meta-World/sam-MT1-door-open-v3"
    "Meta-World/sam-MT1-drawer-open-v3"
    "Meta-World/sam-MT1-drawer-close-v3"
    "Meta-World/sam-MT1-button-press-topdown-v3"
    "Meta-World/sam-MT1-peg-insert-side-v3"
    "Meta-World/sam-MT1-window-open-v3"
    "Meta-World/sam-MT1-window-close-v3"
    # DMW MT10 distractor environments
    "Meta-World/sam-masked-MT1-reach-v3"
    "Meta-World/sam-masked-MT1-push-v3"
    "Meta-World/sam-masked-MT1-pick-place-v3"
    "Meta-World/sam-masked-MT1-door-open-v3"
    "Meta-World/sam-masked-MT1-drawer-open-v3"
    "Meta-World/sam-masked-MT1-drawer-close-v3"
    "Meta-World/sam-masked-MT1-button-press-topdown-v3"
    "Meta-World/sam-masked-MT1-peg-insert-side-v3"
    "Meta-World/sam-masked-MT1-window-open-v3"
    "Meta-World/sam-masked-MT1-window-close-v3"

)

SEEDS=(
    1
    2
    3
)

DCS_CONFIGS=(
    "slapo_default_stage_1"
    "slapo_default_stage_2"
    "slapo_default_stage_3"
)
DMW_CONFIGS=(
    "slapo_dmw_stage_1"
    "slapo_dmw_stage_2"
    "slapo_dmw_stage_3"
)

for env in "${AVAILABLE_ENVS[@]}"; do
    if [[ $env == *"Meta-World/"* ]]; then
        CONFIG=("${DMW_CONFIGS[@]}")
    else
        CONFIG=("${DCS_CONFIGS[@]}")
    fi
    for seed in "${SEEDS[@]}"; do
        echo "Running evaluation for environment: $env with seed: $seed"
        run_id="slapo_q1_sam_${env//\//_}_seed${seed}"
        python scripts/submission2026/eval_slapo.py \
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