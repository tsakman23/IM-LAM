"""Stage 1: VideoSAUR pretraining.

Trains slot attention on DINO features to decompose images into object slots.
"""

import argparse
from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader

import wandb
from ifo.common.utils.data import get_dataset
from src.models.videosaur import VideoSAUR
from src.training.trainer import Trainer
from src.utils.config import ExperimentConfig, load_config, print_config
from src.utils.helpers import (
    compute_slot_metrics,
    visualize_slot_decomposition,
    visualize_slot_heatmaps,
    visualize_slot_masks,
)


def collate_fn(batch):
    """Identity collate — dataset __getitems__ returns batched TensorDict."""
    return batch


def build_dataset(cfg: ExperimentConfig, split: str = "train"):
    """Build the appropriate dataset based on config."""
    return get_dataset(
        name=cfg.task.name,
        split=split,
        cache_dir=cfg.cache_dir,
        block_size=None,
    )


def compute_loss(model: VideoSAUR, batch) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compute VideoSAUR training loss."""
    obs = batch["observation"]
    slots, recon_features, attn_masks, target_features = model(obs)
    loss, metrics = model.compute_loss(recon_features, target_features)
    return loss, metrics


def extra_log(model: VideoSAUR, batch, step: int):
    """Log slot visualizations and metrics to wandb."""
    obs = batch["observation"][:8]
    with torch.no_grad():
        slots, recon_features, attn_masks, target_features = model(obs)

    # Slot mask overlays (per-slot masked images)
    mask_grid = visualize_slot_masks(obs, attn_masks, nrow=8)
    # Color-coded slot decomposition
    decomp_grid = visualize_slot_decomposition(obs, attn_masks, nrow=8)
    # Per-slot attention heatmaps
    heatmap_grid = visualize_slot_heatmaps(attn_masks, nrow=8)
    # Slot statistics
    slot_metrics = compute_slot_metrics(attn_masks)

    wandb.log(
        {
            "train/slot_masks": wandb.Image(mask_grid),
            "train/slot_decomposition": wandb.Image(decomp_grid),
            "train/slot_heatmaps": wandb.Image(heatmap_grid),
            **{f"train/{k}": v for k, v in slot_metrics.items()},
        },
        step=step,
    )


def train_videosaur(cfg: ExperimentConfig) -> str:
    """Run Stage 1: VideoSAUR pretraining.

    Returns:
        Path to best checkpoint.
    """
    print("Task config:")
    print_config(cfg.task)
    print("VideoSAUR config:")
    print_config(cfg.videosaur)
    print()

    # Init wandb
    base_run_id = cfg.task.run_id
    videosaur_run_id = f"{base_run_id}-1"
    run_name = videosaur_run_id
    wandb.init(project=cfg.wandb_project, notes=cfg.notes, name=run_name, id=run_name, resume="allow", config=vars(cfg))

    # Dataset
    train_dataset = build_dataset(cfg, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.videosaur.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_dataset = build_dataset(cfg, split="test")
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.videosaur.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # Model
    sa_cfg = cfg.videosaur.slot_attention
    model = VideoSAUR(
        num_slots=sa_cfg.num_slots if sa_cfg.num_slots else cfg.task.num_slots,
        slot_dim=sa_cfg.slot_dim,
        num_iterations=sa_cfg.num_iterations,
        dino_model_name=cfg.videosaur.dino.model_name,
        dino_image_size=cfg.videosaur.dino.image_size,
        sim_weight=cfg.videosaur.sim_weight,
        sim_temperature=cfg.videosaur.sim_temperature,
    )

    # Optimizer + scheduler (exponential decay with warmup)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.videosaur.lr)

    warmup_steps = cfg.videosaur.warmup_steps
    max_steps = cfg.videosaur.max_steps
    decay_steps = getattr(cfg.videosaur, "decay_steps", max_steps)
    import math

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        # Exponential decay from 1.0 to ~0 over decay_steps
        progress = (step - warmup_steps) / max(decay_steps - warmup_steps, 1)
        return math.exp(-5.0 * progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Train
    checkpoint_dir = f"{cfg.checkpoint_dir}/{videosaur_run_id}"
    trainer = Trainer(
        checkpoint_dir=checkpoint_dir,
        precision=cfg.precision,
    )
    trainer.fit(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        compute_loss=compute_loss,
        max_steps=max_steps,
        grad_clip=cfg.videosaur.grad_clip,
        scheduler=scheduler,
        val_loader=val_loader,
        val_frequency=cfg.videosaur.val_frequency,
        val_unit=cfg.videosaur.val_unit,
        checkpoint_prefix="videosaur",
        extra_log_fn=extra_log,
        torch_compile=cfg.torch_compile,
    )

    wandb.finish()
    return f"{checkpoint_dir}/videosaur_best.pt"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", default=[])
    args = parser.parse_args()

    cfg = load_config(*args.config)
    train_videosaur(cfg)
