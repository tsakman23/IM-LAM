#!/usr/bin/env python3
"""End-to-end pipeline for Object-Centric LAPO-masks.

Usage examples:
    # DCS (64x64) with base config
    python scripts/run_pipeline_masks.py \
        --name dm_control/masked-cheetah-run-v0 \
        --config configs/tasks/dcs_base.yaml

    # DMW (128x128) with base config
    python scripts/run_pipeline_masks.py \
        --name Meta-World/distracting-MT1-hammer-v3 \
        --config configs/tasks/dmw_base.yaml

    # Override individual fields from the CLI
    python scripts/run_pipeline_masks.py \
        --name dm_control/masked-humanoid-walk-v0 \
        --config configs/tasks/dcs_base.yaml \
        --override task.num_slots=8 \
        --override lapo_masks.batch_size=256
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config  # noqa: E402
from src.utils.helpers import random_tag  # noqa: E402

STAGE_CONFIG_MAP = {
    "videosaur": "configs/stage1_videosaur.yaml",
    "slot_selection": "configs/stage2_slot_selection.yaml",
    "lapo_masks": "configs/stage3_lapo_masks.yaml",
    "bc": "configs/stage4_bc.yaml",
    "decoder": "configs/stage5_decoder.yaml",
}


def _apply_override(cfg: Any, key: str, value_str: str) -> None:
    """Apply a dotted-key override like 'lapo_masks.batch_size=256' to cfg."""
    try:
        value = yaml.safe_load(value_str)
    except Exception:
        value = value_str

    parts = key.split(".")
    target = cfg
    for part in parts[:-1]:
        target = getattr(target, part)
    setattr(target, parts[-1], value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Object-Centric LAPO Pipeline (masks variant)")
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Environment name (sets task.name in the config).",
    )
    parser.add_argument(
        "--stages",
        type=str,
        nargs="+",
        default=["1", "2", "3", "4", "5"],
        help="Stages to run (1=VideoSAUR, 2=SlotSelection, 3=LAPO, 4=BC Phase A, 5=Decoder Phase B).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--wandb_project", type=str, default="object-centric-lapo")
    parser.add_argument(
        "--config",
        nargs="*",
        default=[],
        help="YAML config files to layer on top of stage defaults (e.g. base task configs).",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override config fields with dotted keys, e.g. 'lapo_masks.batch_size=256'. "
        "Can be specified multiple times.",
    )
    parser.add_argument("--run_id", type=str, default="", help="Experiment run ID. Random if not specified.")
    parser.add_argument("--videosaur_checkpoint", type=str, default="")
    parser.add_argument("--slot_selection_path", type=str, default="")
    parser.add_argument("--lapo_checkpoint", type=str, default="")
    parser.add_argument("--bc_phase_a_checkpoint", type=str, default="")

    args = parser.parse_args()

    # Build config from stage defaults + user-specified YAML configs
    config_files = [
        STAGE_CONFIG_MAP["videosaur"],
        STAGE_CONFIG_MAP["slot_selection"],
        STAGE_CONFIG_MAP["lapo_masks"],
        STAGE_CONFIG_MAP["bc"],
        STAGE_CONFIG_MAP["decoder"],
        *args.config,
    ]

    cfg = load_config(*config_files)
    cfg.variant = "masks"
    cfg.seed = args.seed
    cfg.checkpoint_dir = args.checkpoint_dir
    cfg.task.run_id = args.run_id or random_tag()
    cfg.wandb_project = args.wandb_project
    cfg.task.name = args.name

    # Apply dotted-key overrides from the CLI
    for ov in args.override:
        if "=" not in ov:
            raise ValueError(f"Invalid override '{ov}'. Expected format: key=value")
        key, value_str = ov.split("=", 1)
        _apply_override(cfg, key, value_str)

    print("=== Object-Centric LAPO Pipeline (masks) ===")
    print(f"Env name: {cfg.task.name}")
    print(f"Stages: {args.stages}")
    videosaur_run_id = f"{cfg.task.run_id}-1"
    slot_selection_run_id = f"{cfg.task.run_id}-2"
    lapo_run_id = f"{cfg.task.run_id}-3"
    bc_run_id = f"{cfg.task.run_id}-4"
    decoder_run_id = f"{cfg.task.run_id}-5"
    print(f"Base Run ID: {cfg.task.run_id}")
    print(
        f"Checkpoint dirs: {cfg.checkpoint_dir}/{videosaur_run_id}, "
        f"{cfg.checkpoint_dir}/{slot_selection_run_id}, {cfg.checkpoint_dir}/{lapo_run_id}, "
        f"{cfg.checkpoint_dir}/{bc_run_id}, {cfg.checkpoint_dir}/{decoder_run_id}"
    )
    print()

    videosaur_ckpt = args.videosaur_checkpoint
    slot_selection_path = args.slot_selection_path or f"{cfg.checkpoint_dir}/{slot_selection_run_id}/slot_selection.json"
    lapo_ckpt = args.lapo_checkpoint
    bc_phase_a_ckpt = args.bc_phase_a_checkpoint

    # Stage 1: VideoSAUR
    if "1" in args.stages:
        print("=" * 60)
        print("Stage 1: VideoSAUR Pretraining")
        print("=" * 60)
        from src.training.train_videosaur import train_videosaur

        videosaur_ckpt = train_videosaur(cfg)
        print(f"VideoSAUR checkpoint: {videosaur_ckpt}")
        print()

    if not videosaur_ckpt:
        videosaur_ckpt = f"{cfg.checkpoint_dir}/{videosaur_run_id}/videosaur_best.pt"

    # Stage 2: Slot Selection
    if "2" in args.stages:
        print("=" * 60)
        print("Stage 2: Slot Selection")
        print("=" * 60)
        from src.training.train_slot_selection import train_slot_selection

        result = train_slot_selection(cfg, videosaur_ckpt)
        slot_selection_path = f"{cfg.checkpoint_dir}/{slot_selection_run_id}/slot_selection.json"
        print(f"Selected slots: {result['selected_slots']}")
        print()

    # Stage 3: LAPO-masks Training
    if "3" in args.stages:
        print("=" * 60)
        print("Stage 3: LAPO-masks Training")
        print("=" * 60)
        from src.training.train_lapo import train_lapo

        lapo_ckpt = train_lapo(cfg, videosaur_ckpt, slot_selection_path)
        print(f"LAPO checkpoint: {lapo_ckpt}")
        print()

    if not lapo_ckpt:
        lapo_ckpt = f"{cfg.checkpoint_dir}/{lapo_run_id}/lapo_masks_latest.pt"

    # Stage 4: Behavior Cloning (Phase A)
    if "4" in args.stages:
        print("=" * 60)
        print("Stage 4: Behavior Cloning (Phase A)")
        print("=" * 60)
        from src.training.train_bc import train_bc

        bc_phase_a_ckpt = train_bc(cfg, videosaur_ckpt, lapo_ckpt, slot_selection_path)
        print(f"BC Phase A checkpoint: {bc_phase_a_ckpt}")
        print()

    if not bc_phase_a_ckpt:
        bc_phase_a_ckpt = f"{cfg.checkpoint_dir}/{bc_run_id}/bc_phase_a_latest.pt"

    # Stage 5: Decoder Fine-tuning (Phase B)
    if "5" in args.stages:
        print("=" * 60)
        print("Stage 5: Decoder Fine-tuning (Phase B)")
        print("=" * 60)
        from src.training.train_decoder import train_decoder

        bc_final_ckpt = train_decoder(cfg, bc_phase_a_ckpt)
        print(f"Decoder checkpoint: {bc_final_ckpt}")
        print()

    print("=== Pipeline Complete ===")


if __name__ == "__main__":
    main()

