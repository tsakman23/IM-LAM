# Run stage 1 of slapo pipeline for Q6 with varying levels of erosion and dilation of the mask: does masking improve linear separability / alignment of latent actions by removing distractor content with degraded mask quality?
# Run stage 1 of slapo pipeline with latent action dim 128 and 128k GT actions.
# Run naming: slapo_q6_q1_<env>_erosion_<erosion>_dilation_<dilation>_radius<radius>_seed<seed>

RADII=(
    0
    1
    2
    3
)

EROSION=(
    True
    False
)
DILATION=(
    False
    True
)

AVAILABLE_ENVS=(
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

CACHE_DIR="/tmp/datasets"

DMW_CONFIGS=(
    "slapo_dmw_stage_1"
    "slapo_dmw_stage_2"
    "slapo_dmw_stage_3"
)

for env in "${AVAILABLE_ENVS[@]}"; do
    for i in "${!EROSION[@]}"; do
        erosion=${EROSION[i]}
        dilation=${DILATION[i]}
        for radius in "${RADII[@]}"; do
            for seed in "${SEEDS[@]}"; do
                echo "Running evaluation for environment: $env and erosion: $erosion and dilation: $dilation and radius: $radius with seed: $seed"
                run_id="slapo_q6_q1_${env//\//_}_erosion_${erosion}_dilation_${dilation}_radius_${radius}_seed${seed}"
                python experiments/run_slapo.py \
                    run_id=$run_id \
                    env.name=$env \
                    logger.mode=online \
                    logger.notes="Submission 2026 Q6 Q1 Evaluation" \
                    dataset.cache_dir="${CACHE_DIR}/slapo/${env}" \
                    trainer.compile=False \
                    fabric.precision=bf16-mixed \
                    module.dilate_mask=${dilation} \
                    module.dilate_mask_radius=${radius} \
                    module.erode_mask=${erosion} \
                    module.erode_mask_radius=${radius} \
                    --selected_stages=[stage_1] \
                    --stage stage_1 -cn ${DMW_CONFIGS[0]}
            done
        done
    done
    rm -rf "${CACHE_DIR}/slapo/${env}"
done