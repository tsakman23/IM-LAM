# Run stage 1 of slapo pipeline for Q3: Does masking enable compact latent actions without performance degradation?
# Run stage 1 of slapo pipeline with latent action dim sweep {16, 32, 64, 128, 256, 512, 1024}
# Run naming: slapo_q3_<env>_mask_loss_<mask_loss>_action_dim<action_dim>_seed<seed>

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

ACTION_DIMS=(
    # 16
    32
    64
    128
    256
    512
    1024
)

MASK_LOSS=(
    True
    False
)
CACHE_DIR="/tmp/datasets"

for env in "${AVAILABLE_ENVS[@]}"; do
    CONFIG=("${DCS_CONFIGS[@]}")
    for action_dim in "${ACTION_DIMS[@]}"; do
        for mask_loss in "${MASK_LOSS[@]}"; do
            for seed in "${SEEDS[@]}"; do
                echo "Running evaluation for environment: $env and mask loss: $mask_loss and action dim: $action_dim with seed: $seed"
                run_id="slapo_q3_${env//\//_}_mask_loss${mask_loss}_action_dim${action_dim}_seed${seed}"
                python experiments/run_slapo.py \
                    run_id=$run_id \
                    env.name=$env \
                    logger.mode=online \
                    logger.notes="Submission 2026 Q3 Evaluation" \
                    dataset.cache_dir="${CACHE_DIR}/slapo/${env}" \
                    trainer.compile=True \
                    fabric.precision=bf16-mixed \
                    net.code_dim=${action_dim} \
                    --stage stage_1 -cn ${CONFIG[0]} \
                    module.mask_loss=${mask_loss}
            done
        done
    done
    rm -rf "${CACHE_DIR}/slapo/${env}"
done