# Object-Centric Latent Action Learning

Implementation vendored from the authors of [MaskLAM](https://arxiv.org/abs/2602.02259). Many thanks for providing the code.

A PyTorch implementation of **Object-Centric LAPO** from the paper:

> **Object-Centric Latent Action Learning**
> arXiv: [2502.09680](https://arxiv.org/abs/2502.09680)

Object-Centric LAPO combines **VideoSAUR** (object-centric slot representations from frozen DINOv2 features) with **LAPO** (latent action models) to learn imitation policies from unlabeled video in the presence of visual distractors. The key insight is that slot attention decomposes scenes into objects, enabling the agent to focus on task-relevant entities and ignore distracting backgrounds or camera motion.

Two variants are implemented:

- **LAPO-slots** --- learns inverse/forward dynamics directly in the slot embedding space via MLPs
- **LAPO-masks** --- uses slot attention masks to filter pixel observations before training a CNN-based latent action model

## Method Overview

```
                         Unlabeled Video
                              |
                    [Stage 1] VideoSAUR
                    Frozen DINOv2 + Slot Attention
                              |
                     slots + attention masks
                        /            \
             [Stage 2]                [Stage 2]
          Slot Selection            Slot Selection
          (PCA + linear probe)      (PCA + linear probe)
                |                         |
         [Stage 3a]                [Stage 3b]
         LAPO-slots                LAPO-masks
         MLP IDM/FDM               CNN IDM/FDM
         in slot space             on masked pixels
                \                       /
                 [Stage 4] Behavior Cloning (Phase A)
                 BC on latent actions
                              |
                 [Stage 5] Decoder Fine-tuning (Phase B)
                 fine-tune on small labeled dataset
                              |
                        Evaluation
```

**Stage 1 --- VideoSAUR Pretraining.**
A frozen DINOv2 ViT-B/14 extracts patch features from individual frames. A learnable Slot Attention module decomposes these features into $K$ object slots, and a cross-attention decoder reconstructs the original features. Training objective: MSE on DINO feature reconstruction.

**Stage 2 --- Slot Selection.**
Each slot is evaluated for task relevance by fitting PCA + linear regression to predict ground-truth actions (5-fold CV). The slot with the lowest prediction error is selected.

**Stage 3 --- Latent Action Learning.**
Either variant trains an inverse dynamics model (IDM) that maps $(o_t, o_{t+k})$ to a latent action $z_t$, and a forward dynamics model (FDM) that predicts the future from $(o_t, z_t)$.

**Stage 4 --- Behavior Cloning (Phase A).**
A BC agent is trained to predict latent actions from observations.

**Stage 5 --- Decoder Fine-tuning (Phase B).**
The Phase A checkpoint is loaded, the encoder is frozen, and the actor is fine-tuned on a small labeled dataset to output real actions.

## Installation

```bash
# Create and activate conda environment
conda create -n object-lapo python=3.10 -y
conda activate object-lapo

# Install uv
pip install uv

# Install dependencies
uv pip install -r requirements.txt
```

## Usage

### Full Pipeline (Recommended)

The simplest way to run everything is the pipeline scripts. They chain all five stages, passing checkpoints between them automatically:

```bash
conda activate object-lapo

# LAPO-slots on Cheetah Run (Distracting Control Suite, 64x64)
python scripts/run_pipeline_slots.py \
    --name dm_control/masked-cheetah-run-v0 \
    --config configs/tasks/dcs_base.yaml

# LAPO-masks on Hammer (Meta-World, 128x128)
python scripts/run_pipeline_masks.py \
    --name Meta-World/distracting-MT1-hammer-v3 \
    --config configs/tasks/dmw_base.yaml
```

Each run is assigned a unique `--run_id` (random 5-char tag if not specified). This controls the checkpoint directory and wandb run name. Reuse the same `--run_id` to resume training:

```bash
# First run (prints "Run ID: a3x7k")
python scripts/run_pipeline_slots.py \
    --name dm_control/masked-cheetah-run-v0 \
    --config configs/tasks/dcs_base.yaml

# Resume the same experiment
python scripts/run_pipeline_slots.py \
    --name dm_control/masked-cheetah-run-v0 \
    --config configs/tasks/dcs_base.yaml \
    --run_id a3x7k

# Or name it explicitly
python scripts/run_pipeline_slots.py \
    --name dm_control/masked-cheetah-run-v0 \
    --config configs/tasks/dcs_base.yaml \
    --run_id my_experiment
```

**Example environment names (`--name`):**

| Env name | Benchmark | Distraction | Image Size |
|---|---|---|---|
| `dm_control/masked-cheetah-run-v0` | Distracting Control Suite | masked | 64x64 |
| `dm_control/masked-walker-run-v0` | Distracting Control Suite | masked | 64x64 |
| `dm_control/masked-hopper-hop-v0` | Distracting Control Suite | masked | 64x64 |
| `dm_control/masked-humanoid-walk-v0` | Distracting Control Suite | masked | 64x64 |
| `Meta-World/distracting-MT1-hammer-v3` | Meta-World | masked | 128x128 |
| `Meta-World/distracting-MT1-bin-picking-v3` | Meta-World | masked | 128x128 |
| `Meta-World/distracting-MT1-basketball-v3` | Meta-World | masked | 128x128 |
| `Meta-World/distracting-MT1-soccer-v3` | Meta-World | masked | 128x128 |

**Pipeline options:**

```bash
python scripts/run_pipeline_slots.py \
    --name dm_control/masked-cheetah-run-v0 \  # which environment (sets task.name)
    --config configs/tasks/dcs_base.yaml \     # base task-family config (DCS/DMW)
    --stages 1 2 3 4 5 \                       # which stages to run (default: all)
    --override lapo_slots.batch_size=4096 \    # override any config field (repeatable)
    --run_id my_exp \                          # experiment ID (random if omitted)
    --seed 42 \                                # random seed
    --checkpoint_dir runs/exp1 \               # where to save checkpoints
    --wandb_project my-project                 # wandb project name
```

### Running Individual Stages

When iterating on a single stage or resuming after a failure, run stages selectively:

```bash
# Train only VideoSAUR (Stage 1)
python scripts/run_pipeline_slots.py \
    --name dm_control/masked-cheetah-run-v0 \
    --config configs/tasks/dcs_base.yaml \
    --stages 1 --run_id exp1

# Run only slot selection (Stage 2), pointing to an existing VideoSAUR checkpoint
python scripts/run_pipeline_slots.py \
    --name dm_control/masked-cheetah-run-v0 \
    --config configs/tasks/dcs_base.yaml \
    --stages 2 --run_id exp1 \
    --videosaur_checkpoint checkpoints/dm_control_masked-cheetah-run-v0/exp1/videosaur/videosaur_best.pt

# Train LAPO + BC + decoder (Stages 3-5), reusing earlier checkpoints
python scripts/run_pipeline_slots.py \
    --name dm_control/masked-cheetah-run-v0 \
    --config configs/tasks/dcs_base.yaml \
    --stages 3 4 5 --run_id exp1 \
    --videosaur_checkpoint checkpoints/dm_control_masked-cheetah-run-v0/exp1/videosaur/videosaur_best.pt \
    --slot_selection_path checkpoints/dm_control_masked-cheetah-run-v0/exp1/slot_selection.json
```

### Standalone Training Scripts

Each stage can also be invoked directly with explicit config files. This is useful when you want full control over which YAML files to merge:

```bash
# Stage 1: VideoSAUR pretraining
python -m src.training.train_videosaur \
    --config configs/stage1_videosaur.yaml configs/tasks/dcs_base.yaml

# Stage 2: Slot selection
python -m src.training.train_slot_selection \
    --config configs/tasks/dcs_base.yaml \
    --videosaur_checkpoint checkpoints/dm_control_masked-cheetah-run-v0/videosaur/videosaur_best.pt

# Stage 3: LAPO (choose variant)
python -m src.training.train_lapo \
    --variant slots \
    --config configs/stage3_lapo_slots.yaml configs/tasks/dcs_base.yaml \
    --videosaur_checkpoint checkpoints/dm_control_masked-cheetah-run-v0/videosaur/videosaur_best.pt \
    --slot_selection_path checkpoints/dm_control_masked-cheetah-run-v0/slot_selection.json

# Stage 4: Behavior cloning (Phase A)
python -m src.training.train_bc \
    --variant slots \
    --config configs/stage4_bc.yaml configs/tasks/dcs_base.yaml \
    --videosaur_checkpoint checkpoints/dm_control_masked-cheetah-run-v0/videosaur/videosaur_best.pt \
    --lapo_checkpoint checkpoints/dm_control_masked-cheetah-run-v0/lapo_slots/lapo_slots_latest.pt \
    --slot_selection_path checkpoints/dm_control_masked-cheetah-run-v0/slot_selection.json

# Stage 5: Decoder fine-tuning (Phase B)
python -m src.training.train_decoder \
    --variant slots \
    --config configs/stage5_decoder.yaml configs/tasks/dcs_base.yaml \
    --phase_a_checkpoint checkpoints/exp1-4/bc_phase_a_latest.pt
```

### Python API

All components can be used directly from Python for custom training loops, analysis, or integration into other projects:

```python
import torch
from src.models.videosaur import VideoSAUR
from src.models.lapo_slots import LAPOSlots
from src.models.slot_selector import SlotSelector
from src.utils.checkpoint import load_checkpoint

device = "cuda"

# --- Load a trained VideoSAUR and extract slots ---
videosaur = VideoSAUR(num_slots=4, slot_dim=128)
load_checkpoint("checkpoints/cheetah_run/videosaur/videosaur_best.pt", videosaur)
videosaur = videosaur.to(device).eval()

images = torch.randn(8, 3, 64, 64, device=device)  # centered [-0.5, 0.5]
slots, attn_masks = videosaur.extract_slots(images)
# slots:      (8, 4, 128)  — 4 slot embeddings per image
# attn_masks: (8, 4, 1369) — attention over 37x37 patches per slot

# --- Infer latent actions with LAPO-slots ---
lapo = LAPOSlots(slot_dim=128, hidden_dim=1024, latent_action_dim=8192)
load_checkpoint("checkpoints/cheetah_run/lapo_slots/lapo_slots_latest.pt", lapo)
lapo = lapo.to(device).eval()

s_t = slots[:, 0]       # select slot 0 at time t
s_future = slots[:, 0]  # slot 0 at time t+k (use different images in practice)

with torch.no_grad():
    z_t = lapo.infer_action(s_t, s_future)
    # z_t: (8, 8192) — latent action embedding
```

```python
# --- Use LAPO-masks with CNN encoder/decoder ---
from gymnasium import spaces
from src.models.lapo_masks import LAPOMasks
from src.models.quantizer import VectorQuantizerEMA
from src.models.world_model import UNetWorldModel

action_space = spaces.Box(low=-1, high=1, shape=(6,))
quantizer = VectorQuantizerEMA(num_codes=256, code_dim=1024, num_codebooks=2)
world_model = UNetWorldModel((9, 64, 64), output_channels=3, condition_dim=1024)

lapo_masks = LAPOMasks(
    observation_shape=(13, 3, 64, 64),  # T=frame_stack+future_offset
    action_space=action_space,
    channel_multiplier=6,
    quantizer=quantizer,
    world_model=world_model,
    latent_action_dim=1024,
)

obs_seq = torch.randn(4, 13, 3, 64, 64)  # (batch, time, C, H, W)
masks = torch.ones(4, 13, 1, 64, 64)     # binary slot masks

next_obs, action_dist, vq_loss, perplexity, latent_action = lapo_masks(obs_seq, mask=masks)
# next_obs:      (4, 3, 64, 64) — predicted future frame
# action_dist:   Distribution    — over action space
# vq_loss:       scalar          — VQ commitment loss
# latent_action: (4, 1024)       — pre-quantization encoder output
```

```python
# --- Slot selection on a custom dataset ---
import numpy as np
from src.models.slot_selector import SlotSelector

# slot_embeddings: (N, K, D) from VideoSAUR, actions: (N, action_dim)
slot_embeddings = np.random.randn(10000, 4, 128)
actions = np.random.randn(10000, 6)

selector = SlotSelector(pca_components=32)
scores = selector.evaluate_slots(slot_embeddings, actions, n_folds=5)
# scores: {0: -0.42, 1: -0.91, 2: -0.38, 3: -0.87}  (neg MSE, higher = better)

selected = selector.select(num_slots_to_select=1)
# selected: [2]  — slot 2 best predicts actions
```

```python
# --- Create evaluation environments ---
from ifo.common.utils.env import make_env

# DM Control masked (requires dm_control + gymnasium bindings)
env = make_env(name="dm_control/masked-cheetah-run-v0", frame_stack=3, image_size=64)
obs, info = env.reset()
# obs shape: (9, 64, 64) — 3 stacked CHW frames, normalized to [-0.5, 0.5]

# Meta-World masked (requires metaworld)
env = make_env(name="Meta-World/masked-MT1-hammer-v3", frame_stack=3, image_size=128)
obs, info = env.reset()
# obs shape: (9, 128, 128)
```

### Checkpoints

Each stage saves checkpoints to `<checkpoint_dir>/<run_id>-<stage_id>/`:

```
checkpoints/
├── exp1-1/
│   ├── videosaur_best.pt
│   └── videosaur_latest.pt
├── exp1-2/
│   └── slot_selection.json
├── exp1-3/
│   └── lapo_slots_latest.pt
├── exp1-4/
│   └── bc_phase_a_latest.pt
└── exp1-5/
    └── bc_final_latest.pt
```

Checkpoints contain model weights, optimizer state, step/epoch counters, and best validation loss. Training automatically resumes from the latest checkpoint if one exists in the target directory.

### Adding a New Task

You no longer need to register tasks in a task map. To run a new environment, pass its env id via `--name` and supply a suitable base config (`configs/tasks/dcs_base.yaml` for DCS-like settings, `configs/tasks/dmw_base.yaml` for DMW-like settings), optionally overriding any fields you need:

```bash
python scripts/run_pipeline_slots.py \
    --name dm_control/masked-walker-run-v0 \
    --config configs/tasks/dcs_base.yaml

python scripts/run_pipeline_masks.py \
    --name Meta-World/distracting-MT1-soccer-v3 \
    --config configs/tasks/dmw_base.yaml \
    --override bc.finetune_updates=20000
```

## Running All Experiments

Below are the commands to reproduce every experiment (both variants x all tasks x all distraction levels).

### DCS --- LAPO-slots

```bash
python scripts/run_pipeline_slots.py --name dm_control/masked-cheetah-run-v0 --config configs/tasks/dcs_base.yaml --run_id slots_cheetah_masked
python scripts/run_pipeline_slots.py --name dm_control/masked-walker-run-v0 --config configs/tasks/dcs_base.yaml --run_id slots_walker_masked
python scripts/run_pipeline_slots.py --name dm_control/masked-hopper-hop-v0 --config configs/tasks/dcs_base.yaml --run_id slots_hopper_masked
python scripts/run_pipeline_slots.py --name dm_control/masked-humanoid-walk-v0 --config configs/tasks/dcs_base.yaml --run_id slots_humanoid_masked
```

### DCS --- LAPO-masks

```bash
python scripts/run_pipeline_masks.py --name dm_control/masked-cheetah-run-v0 --config configs/tasks/dcs_base.yaml --run_id masks_cheetah_masked
python scripts/run_pipeline_masks.py --name dm_control/masked-walker-run-v0 --config configs/tasks/dcs_base.yaml --run_id masks_walker_masked
python scripts/run_pipeline_masks.py --name dm_control/masked-hopper-hop-v0 --config configs/tasks/dcs_base.yaml --run_id masks_hopper_masked
python scripts/run_pipeline_masks.py --name dm_control/masked-humanoid-walk-v0 --config configs/tasks/dcs_base.yaml --run_id masks_humanoid_masked
```

### DMW --- LAPO-slots

```bash
# No distractor
python scripts/run_pipeline_slots.py --name Meta-World/distracting-MT1-hammer-v3 --config configs/tasks/dmw_base.yaml --run_id slots_hammer_none
python scripts/run_pipeline_slots.py --name Meta-World/distracting-MT1-bin-picking-v3 --config configs/tasks/dmw_base.yaml --run_id slots_bin_picking_none
python scripts/run_pipeline_slots.py --name Meta-World/distracting-MT1-basketball-v3 --config configs/tasks/dmw_base.yaml --run_id slots_basketball_none
python scripts/run_pipeline_slots.py --name Meta-World/distracting-MT1-soccer-v3 --config configs/tasks/dmw_base.yaml --run_id slots_soccer_none

# Masked
python scripts/run_pipeline_slots.py --name Meta-World/distracting-MT1-hammer-v3 --config configs/tasks/dmw_base.yaml --run_id slots_hammer_masked
python scripts/run_pipeline_slots.py --name Meta-World/distracting-MT1-bin-picking-v3 --config configs/tasks/dmw_base.yaml --run_id slots_bin_picking_masked
python scripts/run_pipeline_slots.py --name Meta-World/distracting-MT1-basketball-v3 --config configs/tasks/dmw_base.yaml --run_id slots_basketball_masked
python scripts/run_pipeline_slots.py --name Meta-World/distracting-MT1-soccer-v3 --config configs/tasks/dmw_base.yaml --run_id slots_soccer_masked
```

### DMW --- LAPO-masks

```bash
# No distractor
python scripts/run_pipeline_masks.py --name Meta-World/distracting-MT1-hammer-v3 --config configs/tasks/dmw_base.yaml --run_id masks_hammer_none
python scripts/run_pipeline_masks.py --name Meta-World/distracting-MT1-bin-picking-v3 --config configs/tasks/dmw_base.yaml --run_id masks_bin_picking_none
python scripts/run_pipeline_masks.py --name Meta-World/distracting-MT1-basketball-v3 --config configs/tasks/dmw_base.yaml --run_id masks_basketball_none
python scripts/run_pipeline_masks.py --name Meta-World/distracting-MT1-soccer-v3 --config configs/tasks/dmw_base.yaml --run_id masks_soccer_none

# Masked
python scripts/run_pipeline_masks.py --name Meta-World/distracting-MT1-hammer-v3 --config configs/tasks/dmw_base.yaml --run_id masks_hammer_masked
python scripts/run_pipeline_masks.py --name Meta-World/distracting-MT1-bin-picking-v3 --config configs/tasks/dmw_base.yaml --run_id masks_bin_picking_masked
python scripts/run_pipeline_masks.py --name Meta-World/distracting-MT1-basketball-v3 --config configs/tasks/dmw_base.yaml --run_id masks_basketball_masked
python scripts/run_pipeline_masks.py --name Meta-World/distracting-MT1-soccer-v3 --config configs/tasks/dmw_base.yaml --run_id masks_soccer_masked
```

To re-run any experiment with a different seed, append `--seed <N>` and use a different `--run_id`.

## Project Structure

```
object-centric-lapo/
├── configs/
│   ├── stage1_videosaur.yaml          # VideoSAUR hyperparameters
│   ├── stage2_slot_selection.yaml     # Slot selection settings
│   ├── stage3_lapo_slots.yaml         # LAPO-slots hyperparameters
│   ├── stage3_lapo_masks.yaml         # LAPO-masks hyperparameters
│   ├── stage4_bc.yaml                 # Stage 4 behavior cloning settings
│   ├── stage5_decoder.yaml            # Stage 5 decoder fine-tuning settings
│   └── tasks/                         # Base task-family configs
│       ├── dcs_base.yaml
│       └── dmw_base.yaml
├── src/
│   ├── models/
│   │   ├── dino_backbone.py           # Frozen DINOv2 ViT-B/14 feature extractor
│   │   ├── slot_attention.py          # Slot Attention + cross-attention decoder
│   │   ├── videosaur.py               # Full VideoSAUR model
│   │   ├── slot_selector.py           # PCA + linear probe slot selection
│   │   ├── lapo_slots.py              # MLP-based IDM/FDM in slot space
│   │   ├── lapo_masks.py              # CNN-based IDM/FDM on masked pixels
│   │   ├── impala_cnn.py              # IMPALA CNN backbone
│   │   ├── world_model.py             # UNet and IMPALA world model decoders
│   │   ├── quantizer.py               # VQ-EMA, FSQ, Identity quantizers
│   │   ├── action_head.py             # Action distribution heads (Box/Discrete)
│   │   └── bc_agent.py                # Pixel-based BC agent (both variants)
│   ├── data/
│   │   ├── datasets.py                # HuggingFace dataset wrappers
│   │   └── transforms.py              # Data preprocessing transforms
│   ├── envs/
│   │   ├── dcs_env.py                 # DCS environment creation
│   │   ├── dmw_env.py                 # Meta-World environment creation
│   │   └── wrappers.py                # Frame stack, normalize, video recording
│   ├── training/
│   │   ├── trainer.py                 # Generic PyTorch trainer with wandb
│   │   ├── train_videosaur.py         # Stage 1: VideoSAUR pretraining
│   │   ├── train_slot_selection.py    # Stage 2: Slot selection
│   │   ├── train_lapo.py             # Stage 3: LAPO-slots or LAPO-masks
│   │   ├── train_bc.py               # Stage 4: BC on latent actions (Phase A)
│   │   └── train_decoder.py          # Stage 5: decoder fine-tuning (Phase B)
│   └── utils/
│       ├── config.py                  # YAML config loading with dataclasses
│       ├── checkpoint.py              # Checkpoint save/load utilities
│       └── helpers.py                 # merge_tc, orthogonal_init, visualization
└── scripts/
    ├── run_pipeline_slots.py          # End-to-end pipeline (slots)
    └── run_pipeline_masks.py          # End-to-end pipeline (masks)
```

## Datasets

Training uses HuggingFace-hosted datasets (downloaded automatically on first run):

| Dataset | HuggingFace Path | Tasks | Image Size |
|---|---|---|---|
| Distracting Control Suite | `EpicPinkPenguin/visual_distracting_control_suite` | cheetah_run, walker_run, hopper_hop, humanoid_walk | 64 x 64 |
| Distracting Meta-World | `EpicPinkPenguin/visual_distracting_metaworld` | hammer-v3, bin-picking-v3, basketball-v3, soccer-v3 | 128 x 128 |

Each sample contains: `observation` (uint8 CHW), `mask` (uint8 HW), `action`, `state`, `reward`, `terminated`, `truncated`.

## Key Hyperparameters

| Parameter | DCS (64x64) | DMW (128x128) |
|---|---|---|
| Num slots | 4 (8 for humanoid) | 3 |
| Slot dim | 128 | 128 |
| VideoSAUR steps | 100k | 100k |
| VideoSAUR batch | 128 | 128 |
| LAPO-slots hidden | 1024 | 1024 |
| LAPO-slots latent dim | 8192 | 512 |
| LAPO-slots batch | 8192 | 64 |
| LAPO-masks encoder scale | 4 | 4 |
| LAPO-masks latent dim | 1024 | 1024 |
| LAPO-masks batch | 512 | 64 |
| Future offset ($k$) | 10 | 10 |
| Frame stack | 3 | 3 |
| BC fine-tune steps | 2,500 | 15,000 |
| Precision | bfloat16 | bfloat16 |

## Configuration

Configs are layered YAML files that merge left-to-right. The pipeline automatically loads stage defaults first, then applies task-specific overrides on top:

```
configs/stage1_videosaur.yaml       # VideoSAUR defaults (lr, batch_size, warmup, ...)
  + configs/stage3_lapo_slots.yaml  # LAPO-slots defaults (hidden_dim, latent_action_dim, ...)
  + configs/stage4_bc.yaml          # Stage 4 BC defaults
  + configs/stage5_decoder.yaml     # Stage 5 decoder defaults
  + configs/tasks/dcs_base.yaml          # Task-family overrides (num_slots, image_size, batch_size, ...)
  = final merged config
```

Any field in the dataclass configs (`src/utils/config.py`) can be overridden in YAML. Nested fields use dotted keys:

```yaml
# Override DINO model and slot attention iterations
videosaur:
  dino:
    model_name: "vit_small_patch14_dinov2.lvd142m"
  slot_attention:
    num_iterations: 5

# Switch world model decoder
lapo_masks:
  world_model_type: "impala"
  quantizer_type: "fsq"
```

You can also override config fields directly from the pipeline CLI using `--override key=value` (repeatable), for example:

```bash
python scripts/run_pipeline_masks.py \
    --name Meta-World/distracting-MT1-hammer-v3 \
    --config configs/tasks/dmw_base.yaml \
    --override lapo_masks.batch_size=256 \
    --override bc.finetune_updates=20000
```

## Logging

Training metrics and visualizations are logged to [Weights & Biases](https://wandb.ai). Set `--wandb_project` to choose the project name, or set `WANDB_MODE=disabled` to turn logging off.

Tracked metrics per stage:

- **Stage 1**: reconstruction MSE, similarity loss, cosine similarity, slot mask overlays, color-coded slot decomposition, per-slot attention heatmaps, slot entropy/coverage/max attention metrics, learning rate schedule
- **Stage 2**: per-slot action prediction MSE, selected slot indices
- **Stage 3**: FDM loss (slots) or reconstruction + VQ loss (masks), codebook perplexity, `action_decoder_mse` (linear probe from latent action to ground-truth action)
- **Stage 4**: BC loss, `action_decoder_mse`
- **Stage 5**: fine-tuning loss, environment evaluation returns

## Citation

```bibtex
@article{object-centric-lapo,
  title={Object-Centric Latent Action Learning},
  year={2025},
  eprint={2502.09680},
  archivePrefix={arXiv},
}
```

## License

This is a research implementation. Please refer to the original paper for details.
