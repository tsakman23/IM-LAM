"""Stage 3: LAPO training (slots or masks variant).

Slots: Train SlotIDM+SlotFDM with on-the-fly slot extraction.
Masks: Train CNN IDM+FDM on slot-mask-filtered pixel observations.
"""

import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
from ifo.common.utils.data import get_dataset
from src.models.lapo_masks import LAPOMasks
from src.models.lapo_slots import LAPOSlots
from src.models.quantizer import FiniteScalarQuantizer, IdentityQuantizer, VectorQuantizerEMA
from src.models.videosaur import VideoSAUR
from src.models.world_model import ImpalaWorldModel, UNetWorldModel
from src.training.trainer import Trainer
from src.utils.action_space import action_probe_loss_and_metrics, make_action_space
from src.utils.checkpoint import load_checkpoint
from src.utils.config import ExperimentConfig, load_config, print_config


# NOTE: `ifo` datasets return already-batched TensorDicts from `__getitems__`.
def collate_fn(batch):
    return batch

# ---- LAPO-slots ----

def build_slots_dataset(cfg: ExperimentConfig, split: str = "train"):
    """Build dataset with temporal windowing for LAPO-slots."""
    block_size = cfg.lapo_slots.future_offset + 1
    return get_dataset(
        name=cfg.task.name,
        split=split,
        cache_dir=cfg.cache_dir,
        block_size=block_size,
    )


def train_lapo_slots(
    cfg: ExperimentConfig,
    videosaur_checkpoint: str,
    slot_selection_path: str,
) -> str:
    """Train LAPO-slots IDM+FDM."""
    print("Task config:")
    print_config(cfg.task)
    print("LAPO-slots config:")
    print_config(cfg.lapo_slots)
    print()

    base_run_id = cfg.task.run_id
    lapo_run_id = f"{base_run_id}-3"
    run_name = lapo_run_id
    wandb.init(project=cfg.wandb_project, notes=cfg.notes, name=run_name, id=run_name, resume="allow", config=vars(cfg))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load slot selection
    with open(slot_selection_path) as f:
        selection = json.load(f)
    selected_slot = selection["selected_slots"][0]

    # Load frozen VideoSAUR for slot extraction
    sa_cfg = cfg.videosaur.slot_attention
    videosaur = VideoSAUR(
        num_slots=sa_cfg.num_slots if sa_cfg.num_slots else cfg.task.num_slots,
        slot_dim=sa_cfg.slot_dim,
        num_iterations=sa_cfg.num_iterations,
        dino_model_name=cfg.videosaur.dino.model_name,
        dino_image_size=cfg.videosaur.dino.image_size,
    )
    load_checkpoint(videosaur_checkpoint, videosaur)
    videosaur = videosaur.to(device)
    videosaur.eval()

    # Datasets
    train_dataset = build_slots_dataset(cfg, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.lapo_slots.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_dataset = build_slots_dataset(cfg, split="test")
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.lapo_slots.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # Model
    model = LAPOSlots(
        slot_dim=cfg.lapo_slots.slot_dim,
        hidden_dim=cfg.lapo_slots.hidden_dim,
        latent_action_dim=cfg.lapo_slots.latent_action_dim,
        num_residual_blocks=cfg.lapo_slots.num_residual_blocks,
    )

    # Linear probe: latent action -> ground truth action (diagnostic, separate optimizer)
    action_probe = nn.Linear(cfg.lapo_slots.latent_action_dim, cfg.task.action_dim).to(device)
    probe_optimizer = torch.optim.Adam(action_probe.parameters(), lr=cfg.lapo_slots.lr)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lapo_slots.lr)

    # Warmup scheduler
    warmup_epochs = getattr(cfg.lapo_slots, "warmup_epochs", 3)
    warmup_steps = warmup_epochs * len(train_loader)
    total_steps = cfg.lapo_slots.epochs * len(train_loader)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(warmup_steps, 1)
        if total_steps <= warmup_steps:
            return 1.0

        decay_progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 1.0 - decay_progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    @torch._dynamo.disable
    def _slots_probe_update(z_t_detached, gt_actions):
        action_pred = action_probe(z_t_detached)
        action_loss, metrics = action_probe_loss_and_metrics(
            action_pred, gt_actions, cfg.task.action_space_type
        )
        probe_optimizer.zero_grad()
        action_loss.backward()
        probe_optimizer.step()
        return metrics

    def compute_loss(model, batch):
        obs = batch["observation"]  # (B, T, C, H, W)
        obs_t = obs[:, 0].to(device)
        obs_future = obs[:, -1].to(device)
        gt_actions = batch["action"][:, 0].to(device)

        # Extract slots on the fly from frozen VideoSAUR
        with torch.no_grad():
            slots_t, _ = videosaur.extract_slots(obs_t)
            slots_future, _ = videosaur.extract_slots(obs_future)

        s_t = slots_t[:, selected_slot]
        s_future = slots_future[:, selected_slot]

        z_t, s_hat = model(s_t, s_future)
        loss = model.compute_loss(s_hat, s_future)

        metrics = {"fdm_mse": loss.detach()}

        # Linear probe: only update during training (validate runs under no_grad)
        if model.training:
            probe_metrics = _slots_probe_update(z_t.detach(), gt_actions)
        else:
            action_pred = action_probe(z_t.detach())
            _, probe_metrics = action_probe_loss_and_metrics(
                action_pred, gt_actions, cfg.task.action_space_type
            )
        metrics.update(probe_metrics)

        return loss, metrics

    checkpoint_dir = f"{cfg.checkpoint_dir}/{lapo_run_id}"
    trainer = Trainer(checkpoint_dir=checkpoint_dir, precision=cfg.precision)
    trainer.fit(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        compute_loss=compute_loss,
        max_epochs=cfg.lapo_slots.epochs,
        grad_clip=cfg.lapo_slots.grad_clip,
        scheduler=scheduler,
        val_loader=val_loader,
        val_frequency=cfg.lapo_slots.val_frequency,
        val_unit=cfg.lapo_slots.val_unit,
        checkpoint_prefix="lapo_slots",
        torch_compile=cfg.torch_compile,
    )

    wandb.finish()
    return f"{checkpoint_dir}/lapo_slots_latest.pt"


# ---- LAPO-masks ----

def build_masks_dataset(cfg: ExperimentConfig, split: str = "train"):
    """Build dataset with temporal windowing for LAPO-masks."""
    block_size = cfg.lapo_masks.future_obs_offset + cfg.lapo_masks.frame_stack
    return get_dataset(
        name=cfg.task.name,
        split=split,
        cache_dir=cfg.cache_dir,
        block_size=block_size,
    )


def build_quantizer(cfg: ExperimentConfig):
    """Build the information bottleneck quantizer."""
    mc = cfg.lapo_masks
    if mc.quantizer_type == "vq_ema":
        return VectorQuantizerEMA(
            num_codes=mc.num_codes,
            code_dim=mc.latent_action_dim,
            num_codebooks=mc.num_codebooks,
        )
    elif mc.quantizer_type == "fsq":
        return FiniteScalarQuantizer()
    else:
        return IdentityQuantizer()


def build_world_model(cfg: ExperimentConfig):
    """Build the FDM world model decoder."""
    mc = cfg.lapo_masks
    c = cfg.task.obs_channels
    h, w = cfg.task.image_size
    shape = (mc.frame_stack * c, h, w)

    if mc.world_model_type == "unet":
        return UNetWorldModel(
            shape=shape,
            output_channels=c,
            condition_dim=mc.latent_action_dim,
            base_channels=mc.base_channels,
        )
    else:
        return ImpalaWorldModel(
            shape=shape,
            output_channels=c,
            condition_dim=mc.latent_action_dim,
        )


def get_slot_masks_for_batch(
    videosaur_model: VideoSAUR,
    observations: torch.Tensor,
    selected_slots: list,
    device: str,
) -> torch.Tensor:
    """Extract and apply slot attention masks to observations.

    Args:
        videosaur_model: frozen VideoSAUR.
        observations: (B, T, C, H, W) temporal observation sequence.
        selected_slots: list of slot indices to use.
        device: compute device.

    Returns:
        masks: (B, T, 1, H, W) combined mask for selected slots.
    """
    b, t, c, h, w = observations.shape

    # Process each timestep
    obs_flat = observations.reshape(b * t, c, h, w).to(device)
    _, attn_masks = videosaur_model.extract_slots(obs_flat)  # (B*T, K, N_patches)

    # Sum selected slot masks
    mask = attn_masks[:, selected_slots].sum(dim=1)  # (B*T, N_patches)
    mask = mask.clamp(0, 1)

    # Reshape to spatial
    n_patches = mask.shape[1]
    patch_h = int(n_patches**0.5)
    mask = mask.reshape(b * t, 1, patch_h, patch_h)
    mask = F.interpolate(mask, size=(h, w), mode="bilinear", align_corners=False)

    # Binarize with threshold
    mask = (mask > 0.5).float()
    mask = mask.reshape(b, t, 1, h, w)

    return mask


def train_lapo_masks(
    cfg: ExperimentConfig,
    videosaur_checkpoint: str,
    slot_selection_path: str,
) -> str:
    """Train LAPO-masks CNN IDM+FDM."""
    print("Task config:")
    print_config(cfg.task)
    print("LAPO-masks config:")
    print_config(cfg.lapo_masks)
    print()

    base_run_id = cfg.task.run_id
    lapo_run_id = f"{base_run_id}-3"
    run_name = lapo_run_id
    wandb.init(project=cfg.wandb_project, notes=cfg.notes, name=run_name, id=run_name, resume="allow", config=vars(cfg))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load slot selection
    with open(slot_selection_path) as f:
        selection = json.load(f)
    selected_slots = selection["selected_slots"]

    # Load frozen VideoSAUR for mask extraction
    sa_cfg = cfg.videosaur.slot_attention
    videosaur = VideoSAUR(
        num_slots=sa_cfg.num_slots if sa_cfg.num_slots else cfg.task.num_slots,
        slot_dim=sa_cfg.slot_dim,
        num_iterations=sa_cfg.num_iterations,
        dino_model_name=cfg.videosaur.dino.model_name,
        dino_image_size=cfg.videosaur.dino.image_size,
    )
    load_checkpoint(videosaur_checkpoint, videosaur)
    videosaur = videosaur.to(device)
    videosaur.eval()

    # Dataset
    dataset = build_masks_dataset(cfg, split="train")
    train_loader = DataLoader(
        dataset,
        batch_size=cfg.lapo_masks.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_dataset = build_masks_dataset(cfg, split="test")
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.lapo_masks.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # Build LAPO-masks model
    mc = cfg.lapo_masks
    obs_shape = (mc.future_obs_offset + mc.frame_stack, cfg.task.obs_channels, *cfg.task.image_size)
    action_space = make_action_space(cfg.task.action_dim, cfg.task.action_space_type)

    quantizer = build_quantizer(cfg)
    world_model = build_world_model(cfg)

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

    # Linear probe: latent action -> ground truth action (diagnostic, separate optimizer)
    action_probe = nn.Linear(mc.latent_action_dim, cfg.task.action_dim).to(device)
    probe_optimizer = torch.optim.Adam(action_probe.parameters(), lr=mc.lr)

    optimizer = torch.optim.Adam(model.parameters(), lr=mc.lr)

    # Warmup scheduler
    warmup_epochs = getattr(mc, "warmup_epochs", 3)
    warmup_steps = warmup_epochs * len(train_loader)
    total_steps = mc.epochs * len(train_loader)

    def masks_lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(warmup_steps, 1)
        if total_steps <= warmup_steps:
            return 1.0

        decay_progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.0, 1.0 - decay_progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, masks_lr_lambda)

    @torch._dynamo.disable
    def _masks_probe_update(latent_action_detached, actions):
        action_pred = action_probe(latent_action_detached)
        action_loss, metrics = action_probe_loss_and_metrics(
            action_pred, actions, cfg.task.action_space_type
        )
        probe_optimizer.zero_grad()
        action_loss.backward()
        probe_optimizer.step()
        return metrics

    def compute_loss(model, batch):
        obs = batch["observation"]  # (B, T, C, H, W)

        # Get slot masks from frozen VideoSAUR
        with torch.no_grad():
            slot_masks = get_slot_masks_for_batch(videosaur, obs, selected_slots, device)

        next_obs, vq_loss, perplexity, latent_action = model(obs, mask=slot_masks)

        # Target: future observation (last frame)
        target = obs[:, -1]  # (B, C, H, W)
        recon_loss = F.mse_loss(next_obs, target)

        # Linear probe: separate optimizer, does not affect main model
        actions = batch["action"][:, mc.frame_stack - 1]  # Action at current timestep
        if model.training:
            probe_metrics = _masks_probe_update(latent_action.detach(), actions)
        else:
            action_pred = action_probe(latent_action.detach())
            _, probe_metrics = action_probe_loss_and_metrics(
                action_pred, actions, cfg.task.action_space_type
            )

        loss = recon_loss + vq_loss
        metrics = {
            "recon_loss": recon_loss.detach(),
            "vq_loss": vq_loss.detach(),
            "perplexity": perplexity.detach(),
        }
        metrics.update(probe_metrics)
        return loss, metrics

    assert not (cfg.torch_compile and mc.quantizer_type in ("vq_ema", "fsq")), (
        f"torch.compile is incompatible with quantizer_type='{mc.quantizer_type}' "
        "(in-place buffer updates cause compile errors). "
        "Use quantizer_type='identity' or disable torch_compile."
    )

    checkpoint_dir = f"{cfg.checkpoint_dir}/{lapo_run_id}"
    trainer = Trainer(checkpoint_dir=checkpoint_dir, precision=cfg.precision)
    trainer.fit(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        compute_loss=compute_loss,
        max_epochs=mc.epochs,
        grad_clip=mc.grad_clip,
        scheduler=scheduler,
        val_loader=val_loader,
        val_frequency=mc.val_frequency,
        val_unit=mc.val_unit,
        checkpoint_prefix="lapo_masks",
        torch_compile=cfg.torch_compile,
    )

    wandb.finish()
    return f"{checkpoint_dir}/lapo_masks_latest.pt"


# ---- Unified entry point ----

def train_lapo(
    cfg: ExperimentConfig,
    videosaur_checkpoint: str,
    slot_selection_path: str,
) -> str:
    """Train LAPO (dispatches to slots or masks variant)."""
    if cfg.variant == "slots":
        return train_lapo_slots(cfg, videosaur_checkpoint, slot_selection_path)
    else:
        return train_lapo_masks(cfg, videosaur_checkpoint, slot_selection_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", default=[])
    parser.add_argument("--variant", choices=["slots", "masks"], required=True)
    parser.add_argument("--videosaur_checkpoint", type=str, required=True)
    parser.add_argument("--slot_selection_path", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(*args.config)
    cfg.variant = args.variant
    train_lapo(cfg, args.videosaur_checkpoint, args.slot_selection_path)
