# Run stage 1 of slapo pipeline for Q5: How does the approach handle occlusions (or a completely occluded agent)?
# Run stage 1 of slapo pipeline with occlusion level sweep {0.0, 0.25, 0.5, 0.75}
# Run naming: slapo_q5_<env>_mask_loss_<mask_loss>_occlusion_level<occlusion_level>_seed<seed>

AVAILABLE_ENVS=(
    # DCS distractor low environments
    "dm_control/masked-cheetah-run-distractor-low-v0"
    "dm_control/masked-hopper-hop-distractor-low-v0"
    "dm_control/masked-humanoid-walk-distractor-low-v0"
    "dm_control/masked-walker-run-distractor-low-v0"
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

OCCLUSION_LEVELS=(
    # 0.0  # I already ran this in Q3 evaluation @128 action dim
    0.25
    0.5
    0.75
)

MASK_LOSS=(
    True
    False
)
CACHE_DIR="/tmp/datasets"

for env in "${AVAILABLE_ENVS[@]}"; do
    CONFIG=("${DCS_CONFIGS[@]}")
    for occlusion_level in "${OCCLUSION_LEVELS[@]}"; do
        for mask_loss in "${MASK_LOSS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                echo "Running evaluation for environment: $env and mask loss: $mask_loss and occlusion level: $occlusion_level with seed: $seed"
                run_id="slapo_q5_${env//\//_}_mask_loss${mask_loss}_occlusion_level${occlusion_level}_seed${seed}"
                python experiments/run_slapo.py \
                    run_id=$run_id \
                    env.name=$env \
                    logger.mode=online \
                    logger.notes="Submission 2026 Q5 Evaluation" \
                    dataset.cache_dir="${CACHE_DIR}/slapo/${env}" \
                    fabric.precision=bf16-mixed \
                    --selected_stages=[stage_1] \
                    --stage stage_1 -cn ${CONFIG[0]} \
                    module.mask_loss=${mask_loss} \
                    module.occlude_mask_observation=True \
                    module.occlude_mask_observation_fraction=${occlusion_level}
            done
        done
    done
    rm -rf "${CACHE_DIR}/slapo/${env}"
done