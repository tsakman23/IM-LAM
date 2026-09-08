# IM-LAM: Interaction-Masked Latent Action Models

Code for the MSc thesis ***Interaction-Masked Latent Action Models for Object-Aware Manipulation
under Visual Distractors*** (Georgios Tsakoumakis, Imperial College London).

IM-LAM learns latent actions from observation-only video on **Distracting Meta-World (DMW)** and
adds a **directed agent to object** forward-dynamics model so the world model reasons about the
manipulated object, not just the arm - under a moving video-clip background that breaks naive pixel
reconstruction. The repository builds on the MaskLAM / SLAPO reproduction (MaskLAM is implemented
under the name `SLAPO`) and reuses its three-stage pipeline.

> **Running experiments:** this README is an overview and quickstart. The full, authoritative
> workflow - data generation, staging, per-model run commands, diagnostics, and visualization - is
> in **[`docs/experiments.md`](docs/experiments.md)**.

## Pipeline

Every model runs the same three stages, in one process and one Weights & Biases run:

1. **Stage 1 - LAM pretraining.** An inverse-dynamics model (IDM) infers a latent action `z_t`; a forward-dynamics
   model (FDM) predicts the next frame from the current frame and `z_t`.
2. **Stage 2 - latent policy.** Behavior cloning of the latent actions on the frozen encoder.
3. **Stage 3 - action decoder.** Training a decoder to map latents to true actions, followed by a
   Meta-World rollout success evaluation (`eval/`).

IM-LAM changes **only the Stage-1 FDM**, swapping MaskLAM's monolithic decoder for a directed
interaction predictor (`IMLAMIDM` + `InteractionWorldModel`); the IDM encoder is identical to
MaskLAM's, so Stages 2/3 are shared.

## Models

All commands and per-stage configs are in [`docs/experiments.md`](docs/experiments.md). Stage-1
configs and W&B groups:

| Model | Entry point | Stage-1 config | W&B group |
|---|---|---|---|
| LAPO (baseline) | `experiments/run_lapo_bc.py` | `lapo_bc_dmw_stage_1` | `lapo_reprod` |
| MaskLAM (baseline) | `experiments/run_slapo.py` | `slapo_dmw_stage_1` | `masklam_reprod` |
| Foreground-MaskLAM (union) | `experiments/run_slapo.py` | `foreground_masklam_dmw_stage_1` | `foreground_masklam` |
| Foreground-MaskLAM (dual) | `experiments/run_slapo.py` | `foreground_masklam_dual_dmw_stage_1` | `dual_masklam` |
| **IM-LAM (union)** - main | `experiments/run_slapo.py` | `imlam_dmw_stage_1` | `imlam_union` |
| IM-LAM (dual) | `experiments/run_slapo.py` | `imlam_dual_dmw_stage_1` | `imlam_dual` |
| Direct-z (ablation) | `experiments/run_slapo.py` | `imlam_direct_z_dmw_stage_1` | `imlam_direct_z` |
| OCL / LAPO-Slots (baseline) | `object-centric-lapo/scripts/run_pipeline_slots.py` | `configs/tasks/dmw_base.yaml` | `ocl_slots` |

`union` gates the FDM loss by the agent-and-object union; `dual` scores agent and object terms under
their own mask-area normalization (`L = L_A + lambda_O * L_O`). The IM-LAM encoder/IDM input stays
agent-only in both, so `z_t` remains an embodiment action.

## Datasets

Hosted on the Hugging Face Hub:

| Dataset | Masks | Used by |
|---|---|---|
| [`EpicPinkPenguin/visual_distracting_metaworld`](https://huggingface.co/datasets/EpicPinkPenguin/visual_distracting_metaworld) | agent only | MaskLAM (LAPO ignores masks) |
| [`tsakman23/visual_masked_distracting_metaworld`](https://huggingface.co/datasets/tsakman23/visual_masked_distracting_metaworld) | agent + object (ground truth) + `object_state` | Foreground-MaskLAM, IM-LAM, OCL |
| [`tsakman23/visual_masked_distracting_metaworld_sam`](https://huggingface.co/datasets/tsakman23/visual_masked_distracting_metaworld_sam) | ground-truth **and** SAM-predicted masks | IM-LAM trained on SAM masks (future work) |

Frames are 128x128; each task is a separate config (e.g. `push-v3`) with ~1M train / 100k test steps.
Generating new-task data and staging a fast local copy are covered in
[`docs/experiments.md`](docs/experiments.md) (Sections 1-2).

## Requirements

- Python >= 3.10, `ffmpeg`, a CUDA GPU (~24 GB VRAM recommended)
- Hugging Face account (to pull/push datasets) and, optionally, a Weights & Biases account
- Fast local storage for the staged dataset copy (training reads it every step)

## Setup

Use the in-repo conda environment and always export the headless-render and cache variables:

```bash
conda activate ./conda_env

export MUJOCO_GL=egl                            # headless EGL rendering
export HF_HOME=/data2/masklam/datasets/hf_home  # keep HF caches off the home-dir quota
```

Authenticate to the Hub with a **write** token in `HF_TOKEN` (`hf auth whoami` should print your
username). See [`docs/experiments.md`](docs/experiments.md) Section 0 for the full prerequisites.

## Quickstart

Stage the dataset locally, then launch the full pipeline. Example for **IM-LAM (union)** on one task
and seed:

```bash
# 1. stage the object-mask data to RAM (see docs/experiments.md Section 2)
python scripts/save_local_dataset.py \
  --repo tsakman23/visual_masked_distracting_metaworld --task <task> \
  --splits test train --out-root /tmp/slapo_local

# 2. run Stage 1 (IM-LAM FDM) -> Stage 2 -> Stage 3 + rollout eval, one W&B run
CUDA_VISIBLE_DEVICES=<n> MUJOCO_GL=egl python experiments/run_slapo.py \
  run_id=im-lam_<task>_union_seed1 env.name=Meta-World/masked-MT1-<task> \
  dataset.dataset_path=/tmp/slapo_local \
  logger.mode=online logger.group=imlam_union \
  trainer.compile=True fabric.precision=bf16-mixed trainer.random_seed=1 \
  --stage stage_1 -cn imlam_dmw_stage_1 \
  --stage stage_2 -cn foreground_masklam_dmw_stage_2 \
  --stage stage_3 -cn foreground_masklam_dmw_stage_3
```

Replace `<task>` with a slug such as `push-v3`, `handle-pull-v3`, `pick-place-v3`,
`peg-insert-side-v3`, `door-open-v3`, `sweep-into-v3`, or `dial-turn-v3`. Every other model's exact
command (LAPO, MaskLAM, Foreground union/dual, IM-LAM dual, Direct-z, OCL) is in
[`docs/experiments.md`](docs/experiments.md) Section 4.

## Diagnostics and figures

Stage-1 mechanism diagnostics and figure generation live in `scripts/imlam_diagnostics/`; all figures
are written under `docs/figures/`:

- **`run_diagnostics.py`** - the headline Stage-1 metrics on a frozen checkpoint and pushes them into
  that run's W&B summary: **Object Prediction Ratio** (`E_O / E_O^copy`), the object-dynamics probe,
  and agent-path dependence. Supports `--model imlam | foreground | masklam`.
- **`reconstruction_panel.py`** - FDM reconstructions with per-pixel error heatmaps.
- **`agent_path_panel.py`** - object prediction under normal / no-transition / shuffled agent context.
- **`eigen_cam.py`** - Eigen-CAM saliency of the IDM encoder.
- **`extraction_footprint.py`** - the interaction attention footprint.

Usage and the recommended run order are in [`docs/experiments.md`](docs/experiments.md).

## Repository structure

```text
.
├── ifo/                          # Main implementation package (SLAPO/MaskLAM, IM-LAM, LAPO, ...)
│   ├── common/                   # Shared collectors, nets, datasets, utilities, logging
│   └── modules/                  # Method-specific modules (slapo/, lapo/, ...)
├── experiments/                  # Hydra entry points (run_slapo.py, run_lapo_bc.py, ...)
│   └── configs/                  # Per-stage configs for every model
├── scripts/
│   ├── imlam_diagnostics/        # IM-LAM diagnostics + figure generators
│   ├── save_local_dataset.py     # Stage a fast local dataset copy
│   └── submission2026/           # Upstream MaskLAM/SLAPO submission reproduction scripts
├── object-centric-lapo/          # OCL / LAPO-Slots baseline (separate codebase)
├── distracting-metaworld-dataset-main/   # DMW dataset generator (object masks + object_state)
├── docs/
│   ├── experiments.md            # Full run workflow (start here to run anything)
│   └── figures/                  # Generated diagnostic figures
└── checkpoints/                  # Training outputs, ./checkpoints/<run_id>-<stage>
```

The upstream MaskLAM/SLAPO submission (the "Segment to Focus" reproduction, research questions
Q1-Q6 and ablations) is driven by the shell scripts under `scripts/submission2026/`; those are
separate from the thesis workflow documented in `docs/experiments.md`.

## Citation

```bibtex
@mastersthesis{tsakoumakis2026imlam,
  title  = {Interaction-Masked Latent Action Models for Object-Aware Manipulation under Visual Distractors},
  author = {Tsakoumakis, Georgios},
  school = {Imperial College London},
  year   = {2026},
  type   = {{MSc} thesis}
}
```

Please also cite Meta-World, the DAVIS background dataset, MaskLAM, and LAPO.
