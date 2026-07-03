# Run first stage of slapo pipeline for ablation study.
# Run first stage of slapo pipeline with latent action dim 128 and 128k GT actions.
# Run naming: slapo_ablation_<env>_mask_channel_<mask_channel>_loss_masking_<loss_masking>_input_masking_<input_masking>_k_<k>_seed<seed>

AVAILABLE_ENVS=(
    # DCS non-distractor environments
    # "dm_control/masked-cheetah-run-v0"
    # "dm_control/masked-hopper-hop-v0"
    # "dm_control/masked-humanoid-walk-v0"
    # "dm_control/masked-walker-run-v0"
    # DCS distractor low environments
    "dm_control/masked-cheetah-run-distractor-low-v0"
    "dm_control/masked-hopper-hop-distractor-low-v0"
    "dm_control/masked-humanoid-walk-distractor-low-v0"
    "dm_control/masked-walker-run-distractor-low-v0"
    # DMW MT10 non-distractor environments
    # "Meta-World/MT1-reach-v3"
    # "Meta-World/MT1-push-v3"
    # "Meta-World/MT1-pick-place-v3"
    # "Meta-World/MT1-door-open-v3"
    # "Meta-World/MT1-drawer-open-v3"
    # "Meta-World/MT1-drawer-close-v3"
    # "Meta-World/MT1-button-press-topdown-v3"
    # "Meta-World/MT1-peg-insert-side-v3"
    # "Meta-World/MT1-window-open-v3"
    # "Meta-World/MT1-window-close-v3"
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

MASK_CHANNEL=(
    False
    True
    False
    False
    True
    False

)
LOSS_MASKING=(
    True
    False
    False
    False
    True
    False
)
INPUT_MASKING=(
    False
    False
    False
    True
    False
    False
)
K=(
    10
    10
    10
    10
    1
    1
)
BLOCK_SIZE=(
    13
    13
    13
    13
    4
    4
)

SEEDS=(
    1
    2
    3
)

FUTURE_OBS_SAMPLING=(
    True
    True
    True
    True
    False
    False
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
    for i in "${!MASK_CHANNEL[@]}"; do
        mask_channel=${MASK_CHANNEL[i]}
        loss_masking=${LOSS_MASKING[i]}
        input_masking=${INPUT_MASKING[i]}
        k=${K[i]}
        block_size=${BLOCK_SIZE[i]}
        future_obs_sampling=${FUTURE_OBS_SAMPLING[i]}
        for seed in "${SEEDS[@]}"; do
            echo "Running evaluation for environment: $env with mask channel: $mask_channel, loss masking: $loss_masking, input masking: $input_masking, k: $k, seed: $seed"
            run_id="slapo_ablation_${env//\//_}_mask_channel_${mask_channel}_loss_masking_${loss_masking}_input_masking_${input_masking}_k_${k}_seed${seed}"
            python experiments/run_slapo.py \
                run_id=$run_id \
                env.name=$env \
                logger.mode=online \
                logger.notes="Submission 2026 Ablation Study" \
                dataset.cache_dir="${CACHE_DIR}/slapo/${env}" \
                trainer.compile=True \
                fabric.precision=bf16-mixed \
                --selected_stages=[stage_1] \
                --stage stage_1 -cn ${CONFIG[0]} \
                env.block_size=${block_size} \
                net.add_mask_to_observation=${mask_channel} \
                module.mask_loss=${loss_masking} \
                module.mask_observation=${input_masking} \
                net.future_obs_offset=${k} \
                net.future_obs_sampling=${future_obs_sampling}
        done
    done
    rm -rf "${CACHE_DIR}/slapo/${env}"
done