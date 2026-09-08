"""Stage 4: Behavior Cloning on latent actions (Phase A only).

Both LAPO-slots and LAPO-masks variants use a pixel-based BC agent (ImpalaCNN).
The difference is only how latent actions are produced for labeling.
This stage outputs `bc_phase_a_latest.pt`, which is consumed by decoder fine-tuning.
"""

import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb
from src.models.bc_agent import BCAgentPixel
from src.models.impala_cnn import ImpalaCNNBackbone
from src.models.lapo_slots import LAPOSlots
from src.models.videosaur import VideoSAUR
from src.training.trainer import Trainer
from src.training.bc_utils import build_dataset, collate_fn
from src.utils.action_space import action_probe_loss_and_metrics, make_action_space
from src.utils.checkpoint import filter_state_dict_by_prefix, load_checkpoint
from src.utils.config import ExperimentConfig, load_config, print_config
from src.utils.helpers import merge_tc


# ---- BC for LAPO-slots ----

def train_bc_slots(
    cfg: ExperimentConfig,
    videosaur_checkpoint: str,
    lapo_checkpoint: str,
    slot_selection_path: str,
) -> str:
    """Train Phase A BC agent for LAPO-slots variant."""
    print("Task config:")
    print_config(cfg.task)
    print("BC config:")
    print_config(cfg.bc)
    print()

    base_run_id = cfg.task.run_id
    bc_run_id = f"{base_run_id}-4"
    run_name = bc_run_id
    wandb.init(project=cfg.wandb_project, notes=cfg.notes, name=run_name, id=run_name, resume="allow", config=vars(cfg))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frame_stack = cfg.bc.frame_stack
    future_offset = cfg.lapo_slots.future_offset
    latent_action_dim = cfg.lapo_slots.latent_action_dim

    # Load slot selection
    with open(slot_selection_path) as f:
        selection = json.load(f)
    selected_slot = selection["selected_slots"][0]

    # Load frozen VideoSAUR
    sa_cfg = cfg.videosaur.slot_attention
    videosaur = VideoSAUR(
        num_slots=sa_cfg.num_slots if sa_cfg.num_slots else cfg.task.num_slots,
        slot_dim=sa_cfg.slot_dim,
        num_iterations=sa_cfg.num_iterations,
        dino_model_name=cfg.videosaur.dino.model_name,
        dino_image_size=cfg.videosaur.dino.image_size,
    )
    load_checkpoint(videosaur_checkpoint, videosaur)
    videosaur = videosaur.to(device).eval()

    # Load frozen LAPO-slots IDM
    lapo = LAPOSlots(
        slot_dim=cfg.lapo_slots.slot_dim,
        hidden_dim=cfg.lapo_slots.hidden_dim,
        latent_action_dim=latent_action_dim,
        num_residual_blocks=cfg.lapo_slots.num_residual_blocks,
    )
    load_checkpoint(lapo_checkpoint, lapo)
    lapo = lapo.to(device).eval()

    print("Phase A: Training BC on latent actions...")
    block_size = frame_stack + future_offset
    dataset = build_dataset(cfg.task.name, cfg.cache_dir, split="train", block_size=block_size)
    train_loader = DataLoader(
        dataset, batch_size=cfg.bc.batch_size,
        shuffle=True, num_workers=cfg.num_workers, drop_last=True,
        collate_fn=collate_fn,
    )
    val_dataset = build_dataset(cfg.task.name, cfg.cache_dir, split="test", block_size=block_size)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.bc.batch_size,
        shuffle=False, num_workers=cfg.num_workers, drop_last=True,
        collate_fn=collate_fn,
    )

    # BC agent (pixel-based)
    action_space = make_action_space(cfg.task.action_dim, cfg.task.action_space_type)
    obs_shape = (frame_stack, cfg.task.obs_channels, *cfg.task.image_size)
    bc_agent = BCAgentPixel(
        observation_shape=obs_shape,
        action_space=action_space,
        channel_multiplier=cfg.bc.channel_multiplier,
        latent_action_dim=latent_action_dim,
        hidden_dim=cfg.bc.hidden_dim,
        action_head_dim=cfg.bc.action_head_dim,
        encoder_deep=cfg.bc.encoder_deep,
        encoder_num_res_blocks=cfg.bc.encoder_num_res_blocks,
    ).to(device)

    # Linear probe: latent action -> ground truth action (diagnostic, separate optimizer)
    action_probe = nn.Linear(latent_action_dim, cfg.task.action_dim).to(device)
    probe_optimizer = torch.optim.Adam(action_probe.parameters(), lr=cfg.bc.lr)

    optimizer = torch.optim.Adam(bc_agent.encoder.parameters(), lr=cfg.bc.lr)

    @torch._dynamo.disable
    def _slots_bc_probe_update(latent_action_detached, gt_actions):
        action_pred = action_probe(latent_action_detached)
        action_loss, metrics = action_probe_loss_and_metrics(
            action_pred, gt_actions, cfg.task.action_space_type
        )
        probe_optimizer.zero_grad()
        action_loss.backward()
        probe_optimizer.step()
        return metrics

    def bc_loss_fn(model, batch):
        obs = batch["observation"]  # (B, T, C, H, W)
        current = obs[:, :frame_stack].to(device)
        obs_current = obs[:, frame_stack - 1].to(device)
        obs_future = obs[:, frame_stack - 1 + future_offset].to(device)
        gt_actions = batch["action"][:, frame_stack - 1].to(device)

        with torch.no_grad():
            slots_t, _ = videosaur.extract_slots(obs_current)
            slots_future, _ = videosaur.extract_slots(obs_future)
            s_t = slots_t[:, selected_slot]
            s_future = slots_future[:, selected_slot]
            latent_action = lapo.infer_action(s_t, s_future)

        predicted_latent = model.forward_latent(current)
        bc_loss = F.mse_loss(predicted_latent, latent_action)

        # Linear probe: separate optimizer, does not affect main model
        if model.training:
            probe_metrics = _slots_bc_probe_update(latent_action.detach(), gt_actions)
        else:
            action_pred = action_probe(latent_action.detach())
            _, probe_metrics = action_probe_loss_and_metrics(
                action_pred, gt_actions, cfg.task.action_space_type
            )

        metrics = {"bc_loss": bc_loss.detach()}
        metrics.update(probe_metrics)
        return bc_loss, metrics

    checkpoint_dir = f"{cfg.checkpoint_dir}/{bc_run_id}"
    trainer = Trainer(checkpoint_dir=checkpoint_dir, precision=cfg.precision)
    trainer.fit(
        model=bc_agent,
        optimizer=optimizer,
        train_loader=train_loader,
        compute_loss=bc_loss_fn,
        max_epochs=cfg.bc.bc_epochs,
        grad_clip=cfg.bc.grad_clip,
        val_loader=val_loader,
        val_frequency=cfg.bc.bc_val_frequency,
        val_unit=cfg.bc.bc_val_unit,
        checkpoint_prefix="bc_phase_a",
        torch_compile=cfg.torch_compile,
    )

    wandb.finish()
    return f"{checkpoint_dir}/bc_phase_a_latest.pt"


# ---- BC for LAPO-masks ----

def train_bc_masks(
    cfg: ExperimentConfig,
    lapo_checkpoint: str,
) -> str:
    """Train Phase A BC agent for LAPO-masks variant."""
    print("Task config:")
    print_config(cfg.task)
    print("BC config:")
    print_config(cfg.bc)
    print()

    base_run_id = cfg.task.run_id
    bc_run_id = f"{base_run_id}-4"
    run_name = bc_run_id
    wandb.init(project=cfg.wandb_project, notes=cfg.notes, name=run_name, id=run_name, resume="allow", config=vars(cfg))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mc = cfg.lapo_masks
    frame_stack = mc.frame_stack
    latent_action_dim = mc.latent_action_dim

    print("Phase A: BC on latent actions...")
    block_size = frame_stack + mc.future_obs_offset
    dataset = build_dataset(cfg.task.name, cfg.cache_dir, split="train", block_size=block_size)
    train_loader = DataLoader(
        dataset, batch_size=cfg.bc.batch_size,
        shuffle=True, num_workers=cfg.num_workers, drop_last=True,
        collate_fn=collate_fn,
    )
    val_dataset = build_dataset(cfg.task.name, cfg.cache_dir, split="test", block_size=block_size)
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.bc.batch_size,
        shuffle=False, num_workers=cfg.num_workers, drop_last=True,
        collate_fn=collate_fn,
    )

    # Load frozen LAPO-masks encoder for labeling
    lapo_state = torch.load(lapo_checkpoint, map_location=device, weights_only=False)
    encoder_state = filter_state_dict_by_prefix(lapo_state["model"], "encoder")

    c = cfg.task.obs_channels
    h, w = cfg.task.image_size
    frozen_encoder = ImpalaCNNBackbone(
        (2 * frame_stack * c, h, w), mc.channel_multiplier, mc.latent_action_dim,
        num_res_blocks=mc.encoder_num_res_blocks,
        encoder_deep=mc.encoder_deep,
    ).to(device)
    frozen_encoder.load_state_dict(encoder_state)
    frozen_encoder.eval()

    # BC agent (pixel-based, separate encoder from LAPO)
    action_space = make_action_space(cfg.task.action_dim, cfg.task.action_space_type)
    obs_shape = (frame_stack, cfg.task.obs_channels, *cfg.task.image_size)
    bc_agent = BCAgentPixel(
        observation_shape=obs_shape,
        action_space=action_space,
        channel_multiplier=cfg.bc.channel_multiplier,
        latent_action_dim=latent_action_dim,
        hidden_dim=cfg.bc.hidden_dim,
        action_head_dim=cfg.bc.action_head_dim,
        encoder_deep=cfg.bc.encoder_deep,
        encoder_num_res_blocks=cfg.bc.encoder_num_res_blocks,
    ).to(device)

    # Linear probe: latent action -> ground truth action (diagnostic, separate optimizer)
    action_probe = nn.Linear(latent_action_dim, cfg.task.action_dim).to(device)
    probe_optimizer = torch.optim.Adam(action_probe.parameters(), lr=cfg.bc.lr)

    optimizer = torch.optim.Adam(bc_agent.encoder.parameters(), lr=cfg.bc.lr)

    @torch._dynamo.disable
    def _masks_bc_probe_update(latent_action_detached, gt_actions):
        action_pred = action_probe(latent_action_detached)
        action_loss, metrics = action_probe_loss_and_metrics(
            action_pred, gt_actions, cfg.task.action_space_type
        )
        probe_optimizer.zero_grad()
        action_loss.backward()
        probe_optimizer.step()
        return metrics

    def bc_loss_fn(model, batch):
        obs = batch["observation"]  # (B, T, C, H, W)
        current = obs[:, :frame_stack].to(device)
        future = obs[:, -frame_stack:].to(device)
        gt_actions = batch["action"][:, frame_stack - 1].to(device)

        with torch.no_grad():
            current_flat = merge_tc(current)
            future_flat = merge_tc(future)
            latent_action = frozen_encoder(torch.cat([current_flat, future_flat], dim=1))

        predicted_latent = model.forward_latent(current)
        bc_loss = F.mse_loss(predicted_latent, latent_action)

        # Linear probe: separate optimizer, does not affect main model
        if model.training:
            probe_metrics = _masks_bc_probe_update(latent_action.detach(), gt_actions)
        else:
            action_pred = action_probe(latent_action.detach())
            _, probe_metrics = action_probe_loss_and_metrics(
                action_pred, gt_actions, cfg.task.action_space_type
            )

        metrics = {"bc_loss": bc_loss.detach()}
        metrics.update(probe_metrics)
        return bc_loss, metrics

    checkpoint_dir = f"{cfg.checkpoint_dir}/{bc_run_id}"
    trainer = Trainer(checkpoint_dir=checkpoint_dir, precision=cfg.precision)
    trainer.fit(
        model=bc_agent,
        optimizer=optimizer,
        train_loader=train_loader,
        compute_loss=bc_loss_fn,
        max_epochs=cfg.bc.bc_epochs,
        grad_clip=cfg.bc.grad_clip,
        val_loader=val_loader,
        val_frequency=cfg.bc.bc_val_frequency,
        val_unit=cfg.bc.bc_val_unit,
        checkpoint_prefix="bc_phase_a",
        torch_compile=cfg.torch_compile,
    )

    wandb.finish()
    return f"{checkpoint_dir}/bc_phase_a_latest.pt"


# ---- Unified entry point ----

def train_bc(
    cfg: ExperimentConfig,
    videosaur_checkpoint: str = "",
    lapo_checkpoint: str = "",
    slot_selection_path: str = "",
) -> str:
    if cfg.variant == "slots":
        return train_bc_slots(cfg, videosaur_checkpoint, lapo_checkpoint, slot_selection_path)
    else:
        return train_bc_masks(cfg, lapo_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", default=[])
    parser.add_argument("--variant", choices=["slots", "masks"], required=True)
    parser.add_argument("--videosaur_checkpoint", type=str, default="")
    parser.add_argument("--lapo_checkpoint", type=str, required=True)
    parser.add_argument("--slot_selection_path", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(*args.config)
    cfg.variant = args.variant
    train_bc(cfg, args.videosaur_checkpoint, args.lapo_checkpoint, args.slot_selection_path)
