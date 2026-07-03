# Run full slapo pipeline for Q1: does masking improve linear separability / alignment of latent actions by removing distractor content?
# Run full slapo pipeline with latent action dim 128 and 128k GT actions.
# Run naming: slapo_q1_<env>_seed<seed>

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
    "slapo_default_stage_1"
    "slapo_default_stage_2"
    "slapo_default_stage_3"
)
DMW_CONFIGS=(
    "slapo_dmw_stage_1"
    "slapo_dmw_stage_2"
    "slapo_dmw_stage_3"
)

CACHE_DIR="/tmp/datasets"

for env in "${AVAILABLE_ENVS[@]}"; do
    if [[ $env == *"Meta-World/"* ]]; then
        CONFIG=("${DMW_CONFIGS[@]}")
    else
        CONFIG=("${DCS_CONFIGS[@]}")
    fi
    for seed in "${SEEDS[@]}"; do
        echo "Running evaluation for environment: $env with seed: $seed"
        run_id="slapo_q1_${env//\//_}_seed${seed}"
        python experiments/run_slapo.py \
            run_id=$run_id \
            env.name=$env \
            logger.mode=online \
            logger.notes="Submission 2026 Q1 Evaluation" \
            dataset.cache_dir="${CACHE_DIR}/slapo/${env}" \
            trainer.compile=True \
            fabric.precision=bf16-mixed \
            --stage stage_1 -cn ${CONFIG[0]} \
            --stage stage_2 -cn ${CONFIG[1]} \
            --stage stage_3 -cn ${CONFIG[2]}
    done
    rm -rf "${CACHE_DIR}/slapo/${env}"
done