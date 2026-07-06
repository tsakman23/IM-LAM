# MaskLAM / SLAPO Reproduction Code

This repository contains implementations and experiment code for **MaskLAM**, the method described in the Submission submission *Segment to Focus: Guiding Latent Action Models in the Presence of Distractors*.

In the codebase, **MaskLAM is implemented under the name `SLAPO`**. The repository also contains the **LAPO** baseline used for comparison. The main reproduction scripts for the paper are split by method under `scripts/submission2026/slapo/` and `scripts/submission2026/lapo/`.

## What this code reproduces

The paper studies latent action learning from observation-only video in the presence of visual distractors. It evaluates whether masking the forward-dynamics reconstruction loss helps latent action models learn action-aligned latents and better downstream policies.

The main experiments are organized by research question:

| Paper result | Script family | What it runs |
|---|---|---|
| MaskLAM recovers action-aligned latents under distractors | `run_q1_evaluation*.sh` | Full three-stage SLAPO/MaskLAM pipeline with latent action dimension 128 and 128k ground-truth actions |
| Better alignment translates to better policies | `run_q1_evaluation*.sh` plus `*_success.sh` | Same as above and Meta-World rollout success-rate computation after Q1 training |
| Cleaner latents allow smaller label budgets | `run_q2_evaluation*.sh` plus `*_success.sh` | Stage 3 label-budget sweep over 2k, 4k, 8k, 16k, 32k, and 64k ground-truth actions |
| MaskLAM enables compact latent action spaces | `run_q3_evaluation.sh` | Stage 1 latent-dimension sweep over 32, 64, 128, 256, 512, and 1024 |
| MaskLAM degrades gracefully under agent occlusion | `run_q5_evaluation.sh` | Stage 1 occlusion sweep over 25%, 50%, and 75% agent-mask occlusion |
| MaskLAM is robust to imperfect segmentation masks | `run_q6_evaluation.sh` | Stage 1 mask perturbation sweep using erosion/dilation radii 0, 1, 2, and 3 |
| Ablation study | `run_ablation.sh` | Stage 1 ablations for mask channel, loss masking, input masking, multi-step IDM, and related settings |

The `_success` scripts should be run **after** the corresponding non-`_success` script. They do not retrain the model; they load the checkpoint produced by the training run and perform rollouts to compute downstream success metrics, especially for Distracting Meta-World (DMW).

Note: `experiments/run_slapo.py` now runs this rollout evaluation **inline** at the end of its pipeline (logged under `eval/` in the same run), so for SLAPO the separate `_success` step is only needed to re-score an existing checkpoint or to evaluate with different rollout settings. The LAPO baseline (`run_lapo_bc.py`) still uses the separate `eval_lapo.py` helper.

## Requirements

- Python >= 3.10
- ffmpeg
- A Hugging Face account, if downloading hosted datasets
- A Weights & Biases account, if using `logger.mode=online`
- CUDA-capable GPU recommended with 24GB of VRAM
- 100GB of free disk space per environment dataset
- 64GB of RAM

The experiments in the paper were run on a single NVIDIA A100 40 GB GPU. Peak memory was below 24 GB across methods and stages, so reproduction should also be possible on many high-memory consumer GPUs, although wall-clock time will vary.

## Installation

Create and activate a clean conda environment:

```bash
conda create -n imitation -c conda-forge python=3.10 -y
conda activate imitation
```

Install uv for package management:

```bash
pip install uv
```

Install the package with all environments:

```bash
uv pip install -e .[submission]
```

For AMD GPUs, install with the ROCm PyTorch index:

```bash
uv pip install -e .[submission] --extra-index-url https://download.pytorch.org/whl/rocm6.3
```

For headless MuJoCo / DeepMind Control runs, always set:

```bash
export MUJOCO_GL=egl
```

## Repository structure

```text
.
├── README.md
├── pyproject.toml
├── setup.py
├── ifo/                         # Main implementation package
│   ├── common/                  # Shared collectors, utilities, wrappers, logging, etc.
│   └── modules/                 # Method-specific implementations
├── experiments/                 # Hydra entry points for training and evaluation
│   └── configs/                 # Hydra configs for SLAPO/MaskLAM, LAPO, and baselines
├── scripts/
│   └── submission2026/             # Paper reproduction scripts and success-eval helpers
│       ├── slapo/               # SLAPO/MaskLAM reproduction shell scripts
│       └── lapo/                # LAPO baseline reproduction shell scripts
├── checkpoints/                 # Training outputs and model checkpoints
├── videos/                      # Generated videos, if enabled
├── wandb/                       # Local W&B logs, if enabled
└── resources/                   # Figures and auxiliary assets
```

## Methods included

### MaskLAM / SLAPO

`SLAPO` is the implementation name for MaskLAM. It keeps the LAPO training pipeline but masks the Stage 1 forward-dynamics reconstruction loss so that the latent action is trained from agent pixels rather than distractor pixels.

The main SLAPO entry point runs the **entire pipeline in a single process and a single Weights & Biases run** - Stage 1 (IDM), Stage 2 (latent policy), Stage 3 (behavior cloning), followed by a rollout success evaluation:

```bash
python experiments/run_slapo.py
```

Each stage logs under its own metric prefix with a 0-based step axis (`stage_1/`, `stage_2/`, `stage_3/`), and the final rollout eval logs under `eval/` (e.g. `eval/episode_success_rate`). Per-stage checkpoints are still written to `./checkpoints/<run_id>-<stage>`. The rollout eval runs automatically after Stage 3; pass `eval=false` to skip it.

For evaluating an existing Stage-3 checkpoint on its own (for example, re-scoring a finished run), use the standalone success-rollout helper:

```bash
python scripts/submission2026/eval_slapo.py
```

### LAPO baseline

The repository also contains LAPO baseline code. The main LAPO behavior-cloning pipeline entry point is:

```bash
python experiments/run_lapo_bc.py
```

The success-rollout helper for LAPO checkpoints is:

```bash
python scripts/submission2026/eval_lapo.py
```

Use the LAPO configs, for example `lapo_bc_dcs_stage_1`, `lapo_bc_dcs_stage_2`, `lapo_bc_dcs_stage_3`, `lapo_bc_dmw_stage_1`, `lapo_bc_dmw_stage_2`, and `lapo_bc_dmw_stage_3`, when reproducing LAPO baseline runs.

## General Hydra usage

Experiments use Hydra configs. The full pipeline runs in one process and one W&B run; you provide a per-stage config for each stage:

```bash
python experiments/run_slapo.py \
  run_id=my_run \
  env.name=dm_control/masked-cheetah-run-distractor-low-v0 \
  --stage stage_1 -cn slapo_default_stage_1 \
  --stage stage_2 -cn slapo_default_stage_2 \
  --stage stage_3 -cn slapo_default_stage_3
```

This trains the three stages sequentially in-process and then runs the rollout eval (add `eval=false` to skip). All metrics land in a single run, grouped by stage (`stage_1/`, `stage_2/`, `stage_3/`, `eval/`), each with its own 0-based step axis.

To run only selected stages (checkpoints from earlier stages are resolved from `./checkpoints/<run_id>-<stage>`):

```bash
python experiments/run_slapo.py \
  run_id=my_stage1_only_run \
  env.name=dm_control/masked-cheetah-run-distractor-low-v0 \
  --selected_stages=[stage_1] \
  --stage stage_1 -cn slapo_default_stage_1
```

To continue an interrupted run, reuse the same `run_id`. The experiment code resolves the latest checkpoint in the corresponding checkpoint directory.

## Reproducing the Submission experiments

Run all commands from the repository root. The examples below use the SLAPO/MaskLAM shell scripts in `scripts/submission2026/slapo/`. The LAPO baseline shell scripts live in `scripts/submission2026/lapo/`.

Most scripts use:

```bash
CACHE_DIR=/tmp/datasets
```

The scripts remove cached datasets for some sweeps after each environment. Use fast local SSD storage for this directory.

### Q1: action-aligned latents under distractors, plus downstream policies

Ground-truth-mask SLAPO/MaskLAM:

```bash
bash scripts/submission2026/slapo/run_q1_evaluation.sh
```

SAM-mask SLAPO/MaskLAM:

```bash
bash scripts/submission2026/slapo/run_q1_evaluation_sam.sh
```

These scripts run the full three-stage pipeline over DCS and DMW environments, with seeds `1`, `2`, and `3`, latent action dimension 128, and 128k ground-truth actions for Stage 3.

After the non-`_success` run finishes, compute DMW rollout success rates:

```bash
bash scripts/submission2026/slapo/run_q1_evaluation_success.sh
bash scripts/submission2026/slapo/run_q1_evaluation_sam_success.sh
```

The success scripts load checkpoints named like:

```text
./checkpoints/slapo_q1_<env>_seed<seed>-3
./checkpoints/slapo_q1_sam_<env>_seed<seed>-3
```

and log/print:

```text
val/episode_success_rate
val/episode_return
val/episode_length
```

### Q2: smaller action-label budgets / sample efficiency

Ground-truth-mask SLAPO/MaskLAM:

```bash
bash scripts/submission2026/slapo/run_q2_evaluation.sh
```

SAM-mask SLAPO/MaskLAM:

```bash
bash scripts/submission2026/slapo/run_q2_evaluation_sam.sh
```

These scripts run Stage 3 only and sweep labeled action counts:

```text
2048, 4096, 8192, 16384, 32768, 64000
```

They reuse Stage 2 checkpoints from Q1:

```text
./checkpoints/slapo_q1_<env>_seed<seed>-2
./checkpoints/slapo_q1_sam_<env>_seed<seed>-2
```

After the non-`_success` runs finish, compute DMW rollout success rates:

```bash
bash scripts/submission2026/slapo/run_q2_evaluation_success.sh
bash scripts/submission2026/slapo/run_q2_evaluation_sam_success.sh
```

The success scripts load Stage 3 checkpoints named like:

```text
./checkpoints/slapo_q2_<env>_action_count<action_count>_seed<seed>-3
./checkpoints/slapo_q2_sam_<env>_action_count<action_count>_seed<seed>-3
```

### Q3: compact latent action spaces

```bash
bash scripts/submission2026/slapo/run_q3_evaluation.sh
```

This runs Stage 1 on the DCS distractor environments while sweeping latent action dimension:

```text
32, 64, 128, 256, 512, 1024
```

and comparing:

```text
module.mask_loss=True
module.mask_loss=False
```

Run IDs follow:

```text
slapo_q3_<env>_mask_loss<mask_loss>_action_dim<action_dim>_seed<seed>
```

### Q5: graceful degradation under agent occlusion

```bash
bash scripts/submission2026/slapo/run_q5_evaluation.sh
```

This runs Stage 1 on DCS distractor environments while sweeping:

```text
module.occlude_mask_observation_fraction in {0.25, 0.5, 0.75}
module.mask_loss in {True, False}
```

The 0.0 occlusion setting is produced by the Q3 run at latent action dimension 128.

Run IDs follow:

```text
slapo_q5_<env>_mask_loss<mask_loss>_occlusion_level<occlusion_level>_seed<seed>
```

### Q6: robustness to imperfect segmentation masks

```bash
bash scripts/submission2026/slapo/run_q6_evaluation.sh
```

This runs Stage 1 on DMW distractor environments while perturbing masks by erosion or dilation:

```text
radius in {0, 1, 2, 3}
erosion/dilation pairs: (True, False), (False, True)
```

Run IDs follow:

```text
slapo_q6_q1_<env>_erosion_<erosion>_dilation_<dilation>_radius_<radius>_seed<seed>
```

### Ablation study

```bash
bash scripts/submission2026/slapo/run_ablation.sh
```

This runs Stage 1 ablations for the SLAPO/MaskLAM components, including:

- mask channel
- loss masking
- input masking
- multi-step inverse dynamics model setting `k`
- future-observation sampling

Run IDs follow:

```text
slapo_ablation_<env>_mask_channel_<mask_channel>_loss_masking_<loss_masking>_input_masking_<input_masking>_k_<k>_seed<seed>
```

## Success-rate evaluation scripts

The helper scripts `eval_slapo.py` and `eval_lapo.py` are lightweight evaluation entry points. They:

1. Resolve `trainer.previous_stage_checkpoint`.
2. Load the latest checkpoint from that path.
3. Instantiate the environment and trained policy.
4. Run rollouts with the configured rollout horizon and exploration mode.
5. Print and optionally log:
   - `val/episode_success_rate`
   - `val/episode_return`
   - `val/episode_length`

Example manual SLAPO success evaluation:

```bash
python scripts/submission2026/eval_slapo.py \
  -cn slapo_dmw_stage_3 \
  run_id=manual_success_eval \
  env.name=Meta-World/masked-MT1-reach-v3 \
  trainer.previous_stage_checkpoint=./checkpoints/slapo_q1_Meta-World_masked-MT1-reach-v3_seed1-3 \
  logger.mode=offline
```

Example manual LAPO success evaluation:

```bash
python scripts/submission2026/eval_lapo.py \
  -cn lapo_bc_dmw_stage_3 \
  run_id=manual_lapo_success_eval \
  env.name=Meta-World/masked-MT1-reach-v3 \
  trainer.previous_stage_checkpoint=./checkpoints/lapo_q1_Meta-World_masked-MT1-reach-v3_seed1-3 \
  logger.mode=offline
```

## Data and checkpoints

Datasets are expected to be downloaded or cached through the configured dataset loaders. The scripts set cache paths such as:

```text
/tmp/datasets/slapo/<env>
```

Checkpoints are written under:

```text
./checkpoints/<run_id>-<stage_index>
```

For example:

```text
./checkpoints/slapo_q1_dm_control_masked-cheetah-run-distractor-low-v0_seed1-1
./checkpoints/slapo_q1_dm_control_masked-cheetah-run-distractor-low-v0_seed1-2
./checkpoints/slapo_q1_dm_control_masked-cheetah-run-distractor-low-v0_seed1-3
```

Stage 2 checkpoints are reused by the Q2 label-budget sweeps. Stage 3 checkpoints are used by the `_success` rollout scripts.

## Expected compute

A single end-to-end run consists of Stage 1 LAM pre-training, Stage 2 latent-policy behavior cloning, and Stage 3 action-decoder fine-tuning. The paper reports single-run wall-clock on an A100 40 GB of approximately:

| Method | DCS | DMW |
|---|---:|---:|
| LAPO | ~6 h 21 m | ~5 h 20 m |
| MaskLAM / SLAPO | ~11 h 8 m | ~7 h 8 m |

Large sweeps such as Q2, Q3, Q5, Q6, and the ablation study are substantially more expensive because they multiply environments by seeds and sweep values.


## Troubleshooting

### `dm_control` or MuJoCo fails on a headless machine

Use EGL rendering:

```bash
export MUJOCO_GL=egl
```

### A Q2 script cannot find a checkpoint

Run the corresponding Q1 script first. Q2 reuses Q1 Stage 2 checkpoints:

```text
./checkpoints/slapo_q1_<env>_seed<seed>-2
./checkpoints/slapo_q1_sam_<env>_seed<seed>-2
```

### A `_success` script cannot find a checkpoint

Run the corresponding non-`_success` script first. The `_success` scripts load Stage 3 checkpoints from the training run and only perform rollout evaluation.

### W&B login fails

Set logging to offline/disabled before running, or log into W&B:

```bash
wandb login
```

### Dataset caching is slow

Use a local SSD cache directory. The scripts default to `/tmp/datasets`; change `CACHE_DIR` in the shell scripts if that path is not suitable.