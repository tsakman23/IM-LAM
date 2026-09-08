"""Stage 2: Slot selection via action probing.

Evaluates each slot's ability to predict continuous or discrete actions after PCA.
"""

import argparse
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

import wandb
from ifo.common.utils.data import get_dataset
from src.models.slot_selector import SlotSelector
from src.models.videosaur import VideoSAUR
from src.utils.checkpoint import load_checkpoint
from src.utils.config import ExperimentConfig, load_config, print_config


def collate_fn(batch):
    """Identity collate — dataset __getitems__ returns batched TensorDict."""
    return batch


def build_dataset(cfg: ExperimentConfig, split: str = "train"):
    """Build dataset with both observations and actions."""
    return get_dataset(
        name=cfg.task.name,
        split=split,
        cache_dir=cfg.cache_dir,
        block_size=None,
    )


def train_slot_selection(
    cfg: ExperimentConfig,
    videosaur_checkpoint: str,
    max_samples: int = 50000,
) -> Dict:
    """Run Stage 2: Slot selection.

    Args:
        cfg: experiment config.
        videosaur_checkpoint: path to trained VideoSAUR checkpoint.
        max_samples: max samples for linear probing.

    Returns:
        Dict with selected_slots and per-slot scores.
    """
    print("Task config:")
    print_config(cfg.task)
    print()

    base_run_id = cfg.task.run_id
    slot_selection_run_id = f"{base_run_id}-2"
    run_name = slot_selection_run_id
    wandb.init(project=cfg.wandb_project, notes=cfg.notes, name=run_name, id=run_name, resume="allow", config=vars(cfg))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load VideoSAUR
    sa_cfg = cfg.videosaur.slot_attention
    model = VideoSAUR(
        num_slots=sa_cfg.num_slots if sa_cfg.num_slots else cfg.task.num_slots,
        slot_dim=sa_cfg.slot_dim,
        num_iterations=sa_cfg.num_iterations,
        dino_model_name=cfg.videosaur.dino.model_name,
        dino_image_size=cfg.videosaur.dino.image_size,
    )
    load_checkpoint(videosaur_checkpoint, model)
    model = model.to(device)
    model.eval()

    # Dataset (small subset)
    dataset = build_dataset(cfg, split="train")
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    # Extract slots and actions
    all_slots = []
    all_actions = []
    n_collected = 0

    with torch.no_grad():
        for batch in loader:
            obs = batch["observation"].to(device)
            actions = batch["action"]

            slots, _ = model.extract_slots(obs)
            all_slots.append(slots.cpu().numpy())
            all_actions.append(actions.numpy())

            n_collected += obs.shape[0]
            if n_collected >= max_samples:
                break

    slot_embeddings = np.concatenate(all_slots, axis=0)[:max_samples]
    actions = np.concatenate(all_actions, axis=0)[:max_samples]

    # Evaluate and select
    selector = SlotSelector(
        pca_components=32,
        action_space_type=cfg.task.action_space_type,
    )
    scores = selector.evaluate_slots(slot_embeddings, actions)
    selected = selector.select(num_slots_to_select=1)

    # Log results
    print(f"Slot scores: {scores}")
    print(f"Selected slots: {selected}")

    wandb.log({f"slot_selection/slot_{k}_score": v for k, v in scores.items()})
    wandb.log({"slot_selection/selected_slots": selected})

    result = {
        "selected_slots": selected,
        "slot_scores": scores,
    }

    # Save selection metadata
    import json
    import os
    save_path = f"{cfg.checkpoint_dir}/{slot_selection_run_id}/slot_selection.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(result, f)

    wandb.finish()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+", default=[])
    parser.add_argument("--videosaur_checkpoint", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=50000)
    args = parser.parse_args()

    cfg = load_config(*args.config)
    train_slot_selection(cfg, args.videosaur_checkpoint, args.max_samples)
