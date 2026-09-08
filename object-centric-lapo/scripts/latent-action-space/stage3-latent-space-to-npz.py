"""
Script to extract true actions and latent actions from a trained Stage-3
LAPO model (either 'masks' or 'slots' variant) and save them to an NPZ file
for downstream UMAP/latent-space visualization.

Variant is auto-detected from the config files (looks for 'variant' key,
which is set to 'slots' or 'masks' in stage3_lapo_slots.yaml /
stage3_lapo_masks.yaml).  You can also force it with --variant.

Both variants require a frozen VideoSAUR checkpoint and a slot-selection JSON
(same as the training pipeline).  For the 'masks' variant, VideoSAUR is used
to produce binary attention masks that are applied to observations before the
CNN IDM encoder runs.  For the 'slots' variant, VideoSAUR extracts per-slot
embeddings that are passed directly to the MLP IDM.

Usage examples
--------------
# masks variant
python scripts/latent-action-space/stage3-latent-space-to-npz.py \\
    --config configs/tasks/dcs_base.yaml configs/stage3_lapo_masks.yaml \\
    --checkpoint_path checkpoints/<run-id>-3/lapo_masks_latest.pt \\
    --videosaur_checkpoint checkpoints/<run-id>-1/videosaur_latest.pt \\
    --slot_selection_path checkpoints/<run-id>-2/slot_selection.json \\
    --split train \\
    --output_dir outputs/npz

# slots variant
python scripts/latent-action-space/stage3-latent-space-to-npz.py \\
    --config configs/tasks/dcs_base.yaml configs/stage3_lapo_slots.yaml \\
    --checkpoint_path checkpoints/<run-id>-3/lapo_slots_latest.pt \\
    --videosaur_checkpoint checkpoints/<run-id>-1/videosaur_latest.pt \\
    --slot_selection_path checkpoints/<run-id>-2/slot_selection.json \\
    --split train \\
    --output_dir outputs/npz

# override cache dir and limit samples
python scripts/latent-action-space/stage3-latent-space-to-npz.py \\
    --config configs/tasks/dcs_base.yaml configs/stage3_lapo_masks.yaml \\
    --checkpoint_path checkpoints/<run-id>-3/lapo_masks_latest.pt \\
    --videosaur_checkpoint checkpoints/<run-id>-1/videosaur_latest.pt \\
    --slot_selection_path checkpoints/<run-id>-2/slot_selection.json \\
    --cache_dir /tmp/datasets_usugg \\
    --max_samples 10000 \\
    --output_dir outputs/npz

Notes
-----
- Output NPZ keys: 'latent_actions', 'true_actions'.
- The checkpoint can be a .pt file or a directory – in the latter case the
  most recently modified *.pt file is used (same logic as get_latest_checkpoint).
- For 'masks', true_action is taken at timestep frame_stack-1 (current frame).
- For 'slots', true_action is taken at timestep 0 (s_t frame).
"""

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root is importable regardless of CWD.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ifo.common.utils.data import get_dataset
from src.models.lapo_masks import LAPOMasks
from src.models.lapo_slots import LAPOSlots
from src.models.quantizer import FiniteScalarQuantizer, IdentityQuantizer, VectorQuantizerEMA
from src.models.videosaur import VideoSAUR
from src.models.world_model import ImpalaWorldModel, UNetWorldModel
from src.utils.checkpoint import load_checkpoint, get_latest_checkpoint
from src.utils.config import ExperimentConfig, load_config
from gymnasium import spaces


# ---------------------------------------------------------------------------
# Helpers – mirrors helpers from train_lapo.py
# ---------------------------------------------------------------------------

def _resolve_checkpoint(path: str) -> str:
    """Resolve a checkpoint path: file → as-is; directory → latest *.pt."""
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        latest = get_latest_checkpoint(path)
        if latest is None:
            raise FileNotFoundError(f"No *.pt checkpoint found in directory: {path}")
        return latest
    raise FileNotFoundError(f"Checkpoint not found: {path}")


def _build_videosaur(cfg: ExperimentConfig) -> VideoSAUR:
    """Instantiate VideoSAUR from config (not yet loaded)."""
    sa_cfg = cfg.videosaur.slot_attention
    return VideoSAUR(
        num_slots=sa_cfg.num_slots if sa_cfg.num_slots else cfg.task.num_slots,
        slot_dim=sa_cfg.slot_dim,
        num_iterations=sa_cfg.num_iterations,
        dino_model_name=cfg.videosaur.dino.model_name,
        dino_image_size=cfg.videosaur.dino.image_size,
    )


def _get_slot_masks(
    videosaur: VideoSAUR,
    observations: torch.Tensor,
    selected_slots: list,
    device: str,
) -> torch.Tensor:
    """Extract binary slot-attention masks for selected slots.

    Args:
        videosaur: frozen VideoSAUR model.
        observations: (B, T, C, H, W) temporal observation sequence.
        selected_slots: list of slot indices to combine.
        device: torch device string.

    Returns:
        masks: (B, T, 1, H, W) binary mask.
    """
    b, t, c, h, w = observations.shape
    obs_flat = observations.reshape(b * t, c, h, w).to(device)
    _, attn_masks = videosaur.extract_slots(obs_flat)   # (B*T, K, N_patches)

    mask = attn_masks[:, selected_slots].sum(dim=1)     # (B*T, N_patches)
    mask = mask.clamp(0, 1)

    n_patches = mask.shape[1]
    patch_h = int(n_patches ** 0.5)
    mask = mask.reshape(b * t, 1, patch_h, patch_h)
    mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)
    mask = (mask > 0.5).float()
    return mask.reshape(b, t, 1, h, w)


# ifo datasets return already-batched TensorDicts from __getitems__.
def _collate(batch):
    return batch


# ---------------------------------------------------------------------------
# Variant-specific extraction functions
# ---------------------------------------------------------------------------

def _extract_masks(
    cfg: ExperimentConfig,
    checkpoint_path: str,
    videosaur_checkpoint: str,
    slot_selection_path: str,
    split: str,
    output_dir: str,
    max_samples: Optional[int],
    num_workers: int,
    cache_dir: Optional[str],
) -> str:
    """Extract latent + true actions for the **masks** variant (LAPOMasks)."""
    mc = cfg.lapo_masks
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[masks] Device: {device}")

    # --- Slot selection ---
    with open(slot_selection_path) as f:
        selection = json.load(f)
    selected_slots = selection["selected_slots"]
    print(f"[masks] Selected slots: {selected_slots}")

    # --- VideoSAUR ---
    videosaur = _build_videosaur(cfg).to(device)
    vs_ckpt = _resolve_checkpoint(videosaur_checkpoint)
    load_checkpoint(vs_ckpt, videosaur)
    videosaur.eval()
    print(f"[masks] Loaded VideoSAUR from: {vs_ckpt}")

    # --- Dataset ---
    actual_cache_dir = cache_dir if cache_dir is not None else cfg.cache_dir
    block_size = mc.future_obs_offset + mc.frame_stack
    print(f"[masks] Building '{split}' dataset (block_size={block_size})")
    dataset = get_dataset(
        name=cfg.task.name,
        split=split,
        cache_dir=actual_cache_dir,
        block_size=block_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=mc.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=_collate,
    )

    # --- LAPOMasks model ---
    obs_shape = (mc.future_obs_offset + mc.frame_stack, cfg.task.obs_channels, *cfg.task.image_size)
    action_space = spaces.Box(low=-1, high=1, shape=(cfg.task.action_dim,))

    if mc.quantizer_type == "vq_ema":
        quantizer = VectorQuantizerEMA(
            num_codes=mc.num_codes,
            code_dim=mc.latent_action_dim,
            num_codebooks=mc.num_codebooks,
        )
    elif mc.quantizer_type == "fsq":
        quantizer = FiniteScalarQuantizer()
    else:
        quantizer = IdentityQuantizer()

    h, w = cfg.task.image_size
    c = cfg.task.obs_channels
    shape = (mc.frame_stack * c, h, w)
    if mc.world_model_type == "unet":
        world_model = UNetWorldModel(
            shape=shape, output_channels=c,
            condition_dim=mc.latent_action_dim, base_channels=mc.base_channels,
        )
    else:
        world_model = ImpalaWorldModel(
            shape=shape, output_channels=c, condition_dim=mc.latent_action_dim,
        )

    model = LAPOMasks(
        observation_shape=obs_shape,
        action_space=action_space,
        channel_multiplier=mc.channel_multiplier,
        quantizer=quantizer,
        world_model=world_model,
        latent_action_dim=mc.latent_action_dim,
        num_latents=mc.num_latents,
        future_obs_offset=mc.future_obs_offset,
        frame_stack=mc.frame_stack,
        encoder_deep=getattr(mc, "encoder_deep", False),
        encoder_num_res_blocks=getattr(mc, "encoder_num_res_blocks", 2),
    ).to(device)

    ckpt_path = _resolve_checkpoint(checkpoint_path)
    load_checkpoint(ckpt_path, model)
    model.eval()
    print(f"[masks] Loaded LAPOMasks from: {ckpt_path}")

    # --- Extraction loop ---
    all_latent_actions = []
    all_true_actions = []
    samples_processed = 0

    print("[masks] Extracting latent + true actions …")
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[masks/{split}]"):
            obs = batch["observation"]               # (B, T, C, H, W)
            true_actions = batch["action"]           # (B, T, action_dim)

            # Get slot masks (requires VideoSAUR on GPU)
            slot_masks = _get_slot_masks(videosaur, obs, selected_slots, device)

            obs_dev = obs.to(device)
            latent_action = model.label(obs_dev, mask=slot_masks)   # (B, latent_action_dim)

            # Current-frame action: index frame_stack - 1
            true_action_t = true_actions[:, mc.frame_stack - 1].cpu()

            batch_n = latent_action.shape[0]
            if max_samples is not None and samples_processed + batch_n > max_samples:
                limit = max_samples - samples_processed
                latent_action = latent_action[:limit]
                true_action_t = true_action_t[:limit]
                batch_n = limit

            all_latent_actions.append(latent_action.cpu().float().numpy())
            all_true_actions.append(true_action_t.float().numpy())
            samples_processed += batch_n

            if max_samples is not None and samples_processed >= max_samples:
                break

    return _save_npz(
        all_latent_actions, all_true_actions, samples_processed,
        model_tag="lapo_masks",
        env_name=cfg.task.name,
        split=split,
        output_dir=output_dir,
    )


def _extract_slots(
    cfg: ExperimentConfig,
    checkpoint_path: str,
    videosaur_checkpoint: str,
    slot_selection_path: str,
    split: str,
    output_dir: str,
    max_samples: Optional[int],
    num_workers: int,
    cache_dir: Optional[str],
) -> str:
    """Extract latent + true actions for the **slots** variant (LAPOSlots)."""
    sc = cfg.lapo_slots
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[slots] Device: {device}")

    # --- Slot selection ---
    with open(slot_selection_path) as f:
        selection = json.load(f)
    selected_slot = selection["selected_slots"][0]
    print(f"[slots] Selected slot index: {selected_slot}")

    # --- VideoSAUR ---
    videosaur = _build_videosaur(cfg).to(device)
    vs_ckpt = _resolve_checkpoint(videosaur_checkpoint)
    load_checkpoint(vs_ckpt, videosaur)
    videosaur.eval()
    print(f"[slots] Loaded VideoSAUR from: {vs_ckpt}")

    # --- Dataset ---
    actual_cache_dir = cache_dir if cache_dir is not None else cfg.cache_dir
    block_size = sc.future_offset + 1
    print(f"[slots] Building '{split}' dataset (block_size={block_size})")
    dataset = get_dataset(
        name=cfg.task.name,
        split=split,
        cache_dir=actual_cache_dir,
        block_size=block_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=sc.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=_collate,
    )

    # --- LAPOSlots model ---
    model = LAPOSlots(
        slot_dim=sc.slot_dim,
        hidden_dim=sc.hidden_dim,
        latent_action_dim=sc.latent_action_dim,
        num_residual_blocks=sc.num_residual_blocks,
    ).to(device)

    ckpt_path = _resolve_checkpoint(checkpoint_path)
    load_checkpoint(ckpt_path, model)
    model.eval()
    print(f"[slots] Loaded LAPOSlots from: {ckpt_path}")

    # --- Extraction loop ---
    all_latent_actions = []
    all_true_actions = []
    samples_processed = 0

    print("[slots] Extracting latent + true actions …")
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[slots/{split}]"):
            obs = batch["observation"]           # (B, T, C, H, W)
            true_actions = batch["action"]       # (B, T, action_dim)

            obs_t = obs[:, 0].to(device)
            obs_future = obs[:, -1].to(device)

            # Extract slots on the fly
            slots_t, _ = videosaur.extract_slots(obs_t)          # (B, K, D_slot)
            slots_future, _ = videosaur.extract_slots(obs_future)

            s_t = slots_t[:, selected_slot]           # (B, D_slot)
            s_future = slots_future[:, selected_slot]

            latent_action = model.infer_action(s_t, s_future)    # (B, latent_action_dim)

            # Current-frame action (timestep 0 = s_t)
            true_action_t = true_actions[:, 0].cpu()

            batch_n = latent_action.shape[0]
            if max_samples is not None and samples_processed + batch_n > max_samples:
                limit = max_samples - samples_processed
                latent_action = latent_action[:limit]
                true_action_t = true_action_t[:limit]
                batch_n = limit

            all_latent_actions.append(latent_action.cpu().float().numpy())
            all_true_actions.append(true_action_t.float().numpy())
            samples_processed += batch_n

            if max_samples is not None and samples_processed >= max_samples:
                break

    return _save_npz(
        all_latent_actions, all_true_actions, samples_processed,
        model_tag="lapo_slots",
        env_name=cfg.task.name,
        split=split,
        output_dir=output_dir,
    )


# ---------------------------------------------------------------------------
# NPZ save
# ---------------------------------------------------------------------------

def _save_npz(
    latent_actions: list,
    true_actions: list,
    n_samples: int,
    model_tag: str,
    env_name: str,
    split: str,
    output_dir: str,
) -> str:
    la = np.concatenate(latent_actions, axis=0)
    ta = np.concatenate(true_actions, axis=0)

    print(f"Extracted {n_samples} samples.")
    print(f"  latent_actions shape : {la.shape}")
    print(f"  true_actions   shape : {ta.shape}")

    env_slug = env_name.replace("/", "_").replace("-", "_")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model_tag}_{env_slug}_{split}.npz")

    np.savez(
        output_path,
        latent_actions=la,
        idm_actions=la,      # alias expected by some notebooks
        true_actions=ta,
    )
    print(f"Saved → {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract Stage-3 LAPO (masks or slots) latent actions + true actions"
            " into an NPZ file for latent-space visualization."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        nargs="+",
        required=True,
        metavar="YAML",
        help=(
            "One or more YAML config paths (merged left-to-right). "
            "Typically: configs/tasks/dcs_base.yaml configs/stage3_lapo_masks.yaml"
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the Stage-3 LAPO checkpoint (.pt file or directory).",
    )
    parser.add_argument(
        "--videosaur_checkpoint",
        type=str,
        required=True,
        help="Path to the Stage-1 VideoSAUR checkpoint (.pt file or directory).",
    )
    parser.add_argument(
        "--slot_selection_path",
        type=str,
        required=True,
        help="Path to the Stage-2 slot selection JSON file.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["masks", "slots"],
        default=None,
        help=(
            "Force 'masks' or 'slots' variant. "
            "If not set, auto-detected from the 'variant' key in the merged config."
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to extract from.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/npz",
        help="Directory to write the output .npz file.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Stop after extracting this many samples (default: all).",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Dataset cache directory override (default: from config, /tmp/datasets_usugg).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader num_workers.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve config paths relative to project root if needed
    config_paths = []
    for p in args.config:
        if not os.path.isabs(p):
            p = os.path.join(_PROJECT_ROOT, p)
        config_paths.append(p)

    cfg = load_config(*config_paths)

    # Determine variant
    variant = args.variant if args.variant is not None else cfg.variant
    if variant not in ("masks", "slots"):
        raise ValueError(
            f"Unknown variant '{variant}'. "
            "Pass --variant masks|slots or set 'variant: masks/slots' in your config."
        )

    print(f"Task:    {cfg.task.name}")
    print(f"Variant: {variant}")
    print(f"Split:   {args.split}")
    print()

    common_kwargs = dict(
        cfg=cfg,
        checkpoint_path=args.checkpoint_path,
        videosaur_checkpoint=args.videosaur_checkpoint,
        slot_selection_path=args.slot_selection_path,
        split=args.split,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir,
    )

    if variant == "masks":
        _extract_masks(**common_kwargs)
    else:
        _extract_slots(**common_kwargs)


if __name__ == "__main__":
    main()
