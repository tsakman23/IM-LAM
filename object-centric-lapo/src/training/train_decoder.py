"""Stage 5: Decoder fine-tuning (Phase B only).

Loads a Phase A BC checkpoint (`bc_phase_a_latest.pt`), freezes the encoder,
and fine-tunes the action decoder on labeled data.
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb
from ifo.common.utils.env import make_env
from src.models.bc_agent import BCAgentPixel
from src.training.bc_utils import (
    build_labeled_dataset,
    collate_fn,
    maybe_subset_dataset,
    rollout_bc_agent_in_env,
)
from src.training.trainer import Trainer
from src.utils.action_space import make_action_space, prepare_action_targets
from src.utils.checkpoint import load_checkpoint
from src.utils.config import ExperimentConfig, load_config, print_config


def _build_bc_agent(cfg: ExperimentConfig, frame_stack: int, latent_action_dim: int, device: str) -> BCAgentPixel:
    action_space = make_action_space(cfg.task.action_dim, cfg.task.action_space_type)
    obs_shape = (frame_stack, cfg.task.obs_channels, *cfg.task.image_size)
    return BCAgentPixel(
        observation_shape=obs_shape,
        action_space=action_space,
        channel_multiplier=cfg.bc.channel_multiplier,
        latent_action_dim=latent_action_dim,
        hidden_dim=cfg.bc.hidden_dim,
        action_head_dim=cfg.bc.action_head_dim,
        encoder_deep=cfg.bc.encoder_deep,
        encoder_num_res_blocks=cfg.bc.encoder_num_res_blocks,
    ).to(device)


def _train_decoder_impl(
    cfg: ExperimentConfig,
    *,
    frame_stack: int,
    latent_action_dim: int,
    phase_a_checkpoint: str,
) -> str:
    print("Task config:")
    print_config(cfg.task)
    print("BC config:")
    print_config(cfg.bc)
    print()

    base_run_id = cfg.task.run_id
    decoder_run_id = f"{base_run_id}-5"
    run_name = decoder_run_id
    wandb.init(project=cfg.wandb_project, notes=cfg.notes, name=run_name, id=run_name, resume="allow", config=vars(cfg))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    bc_agent = _build_bc_agent(cfg, frame_stack, latent_action_dim, device)
    load_checkpoint(phase_a_checkpoint, bc_agent, map_location=device)

    labeled_dataset = build_labeled_dataset(cfg.task.name, cfg.cache_dir, split="train", frame_stack=frame_stack)
    labeled_dataset = maybe_subset_dataset(labeled_dataset, cfg.bc.subset_size)
    labeled_loader = DataLoader(
        labeled_dataset,
        batch_size=cfg.bc.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )
    labeled_val_dataset = build_labeled_dataset(cfg.task.name, cfg.cache_dir, split="test", frame_stack=frame_stack)
    labeled_val_dataset = maybe_subset_dataset(labeled_val_dataset, cfg.bc.subset_size)
    labeled_val_loader = DataLoader(
        labeled_val_dataset,
        batch_size=cfg.bc.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=True,
        collate_fn=collate_fn,
    )

    # Freeze encoder, train only actor.
    for param in bc_agent.encoder.parameters():
        param.requires_grad = False

    optimizer_ft = torch.optim.Adam(bc_agent.actor.parameters(), lr=cfg.bc.finetune_lr)

    finetune_warmup_steps = cfg.bc.finetune_warmup_epochs * len(labeled_loader)
    finetune_total_steps = cfg.bc.finetune_updates

    def ft_lr_lambda(step):
        if finetune_warmup_steps > 0 and step < finetune_warmup_steps:
            return step / max(finetune_warmup_steps, 1)
        if finetune_total_steps <= finetune_warmup_steps:
            return 1.0
        decay_progress = (step - finetune_warmup_steps) / max(
            finetune_total_steps - finetune_warmup_steps, 1
        )
        return max(0.0, 1.0 - decay_progress)

    scheduler_ft = torch.optim.lr_scheduler.LambdaLR(optimizer_ft, ft_lr_lambda)

    def finetune_loss_fn(model, batch):
        obs = batch["observation"].to(device)
        actions = batch["action"][:, frame_stack - 1].to(device)
        actions = prepare_action_targets(actions, cfg.task.action_space_type)
        dist = model(obs)
        loss = -dist.log_prob(actions).mean()
        return loss, {"finetune_loss": loss.detach()}

    rollout_env = make_env(
        cfg.task.name,
        num_envs=cfg.bc.rollout_num_envs,
        record_video=False,
        block_size=frame_stack,
        action_repeat=1,
    )

    checkpoint_dir = f"{cfg.checkpoint_dir}/{decoder_run_id}"
    trainer_ft = Trainer(checkpoint_dir=checkpoint_dir, precision=cfg.precision)

    def extra_val_fn(model: nn.Module, val_loss: float, global_step: int, global_epoch: int) -> None:
        episode_return, episode_length, num_episodes, success_rate = rollout_bc_agent_in_env(
            rollout_env,
            model,
            device,
            rollout_steps=cfg.bc.rollout_steps,
            rollout_sample=cfg.bc.rollout_sample,
            seed=cfg.seed,
        )
        wandb.log(
            {
                "eval/episode_return": episode_return,
                "eval/episode_length": episode_length,
                "eval/num_episodes": num_episodes,
                "eval/success_rate": success_rate,
            },
            step=global_step,
        )

    trainer_ft.fit(
        model=bc_agent,
        optimizer=optimizer_ft,
        train_loader=labeled_loader,
        compute_loss=finetune_loss_fn,
        max_steps=cfg.bc.finetune_updates,
        grad_clip=cfg.bc.grad_clip,
        scheduler=scheduler_ft,
        val_loader=labeled_val_loader,
        val_frequency=cfg.bc.ft_val_frequency,
        val_unit=cfg.bc.ft_val_unit,
        checkpoint_prefix="bc_final",
        torch_compile=cfg.torch_compile,
        extra_val_fn=extra_val_fn,
    )

    rollout_env.close()
    wandb.finish()
    return f"{checkpoint_dir}/bc_final_latest.pt"


def train_decoder_slots(cfg: ExperimentConfig, phase_a_checkpoint: str) -> str:
    """Fine-tune decoder for LAPO-slots from a Phase A checkpoint."""
    return _train_decoder_impl(
        cfg,
        frame_stack=cfg.bc.frame_stack,
        latent_action_dim=cfg.lapo_slots.latent_action_dim,
        phase_a_checkpoint=phase_a_checkpoint,
    )


def train_decoder_masks(cfg: ExperimentConfig, phase_a_checkpoint: str) -> str:
    """Fine-tune decoder for LAPO-masks from a Phase A checkpoint."""
    return _train_decoder_impl(
        cfg,
        frame_stack=cfg.lapo_masks.frame_stack,
        latent_action_dim=cfg.lapo_masks.latent_action_dim,
        phase_a_checkpoint=phase_a_checkpoint,
    )


def train_decoder(cfg: ExperimentConfig, phase_a_checkpoint: str) -> str:
    if cfg.variant == "slots":
        return train_decoder_slots(cfg, phase_a_checkpoint)
    return train_decoder_masks(cfg, phase_a_checkpoint)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", default=[])
    parser.add_argument("--variant", choices=["slots", "masks"], required=True)
    parser.add_argument("--phase_a_checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(*args.config)
    cfg.variant = args.variant
    train_decoder(cfg, args.phase_a_checkpoint)
