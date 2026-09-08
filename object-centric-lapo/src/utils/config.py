import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import yaml


def print_config(cfg, indent: int = 0) -> None:
    """Pretty-print a dataclass config."""
    prefix = "  " * indent
    for f in dataclasses.fields(cfg):
        val = getattr(cfg, f.name)
        if dataclasses.is_dataclass(val):
            print(f"{prefix}{f.name}:")
            print_config(val, indent + 1)
        else:
            print(f"{prefix}{f.name}: {val}")


@dataclass
class DinoConfig:
    model_name: str = "vit_base_patch14_dinov2.lvd142m"
    image_size: int = 518
    feature_dim: int = 768


@dataclass
class SlotAttentionConfig:
    num_slots: int = 4
    slot_dim: int = 128
    num_iterations: int = 3
    feature_dim: int = 768


@dataclass
class VideoSAURConfig:
    dino: DinoConfig = field(default_factory=DinoConfig)
    slot_attention: SlotAttentionConfig = field(default_factory=SlotAttentionConfig)
    sim_weight: float = 0.1
    sim_temperature: float = 0.075
    lr: float = 3e-4
    warmup_steps: int = 2500
    max_steps: int = 100_000
    batch_size: int = 128
    grad_clip: float = 0.05
    lr_scheduler: str = "exp_decay_with_warmup"
    decay_steps: int = 100_000
    episode_length: int = 3
    max_video_length: int = 1000
    val_frequency: int = 1
    val_unit: str = "epoch"


@dataclass
class LAPOSlotsConfig:
    slot_dim: int = 128
    hidden_dim: int = 1024
    latent_action_dim: int = 8192
    num_residual_blocks: int = 3
    future_offset: int = 10
    frame_stack: int = 1
    lr: float = 3e-4
    warmup_epochs: int = 3
    batch_size: int = 8192
    epochs: int = 30
    grad_clip: float = 1.0
    val_frequency: int = 1
    val_unit: str = "epoch"


@dataclass
class LAPOMasksConfig:
    channel_multiplier: int = 4
    latent_action_dim: int = 1024
    num_latents: int = 1
    quantizer_type: Literal["vq_ema", "fsq", "identity"] = "vq_ema"
    num_codes: int = 256
    num_codebooks: int = 2
    world_model_type: Literal["unet", "impala"] = "unet"
    base_channels: int = 24
    future_obs_offset: int = 10
    frame_stack: int = 3
    encoder_deep: bool = False
    encoder_num_res_blocks: int = 2
    lr: float = 3e-4
    warmup_epochs: int = 3
    batch_size: int = 512
    epochs: int = 10
    grad_clip: float = 1.0
    val_frequency: int = 1
    val_unit: str = "epoch"


@dataclass
class BCConfig:
    channel_multiplier: int = 32
    latent_action_dim: int = 1024
    frame_stack: int = 3
    hidden_dim: int = 256
    action_head_dim: int = 64
    encoder_deep: bool = False
    encoder_num_res_blocks: int = 2
    bc_epochs: int = 10
    finetune_updates: int = 2500
    finetune_hidden_dim: int = 256
    subset_size: int = 128000
    lr: float = 1e-4
    finetune_lr: float = 3e-4
    finetune_warmup_epochs: int = 0
    batch_size: int = 512
    grad_clip: float = 1.0
    bc_val_frequency: int = 1
    ft_val_frequency: int = 1
    bc_val_unit: str = "epoch"
    ft_val_unit: str = "epoch"
    rollout_steps: int = 10000
    rollout_num_envs: int = 8
    rollout_sample: bool = False


@dataclass
class TaskConfig:
    name: str = "dm_control/masked-cheetah-run-v0"
    run_id: str = "my_run_id"
    image_size: Tuple[int, int] = (64, 64)
    obs_channels: int = 3
    action_dim: int = 6
    action_space_type: Literal["continuous", "discrete"] = "continuous"
    num_slots: int = 4


@dataclass
class ExperimentConfig:
    task: TaskConfig = field(default_factory=TaskConfig)
    videosaur: VideoSAURConfig = field(default_factory=VideoSAURConfig)
    lapo_slots: LAPOSlotsConfig = field(default_factory=LAPOSlotsConfig)
    lapo_masks: LAPOMasksConfig = field(default_factory=LAPOMasksConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    variant: Literal["slots", "masks"] = "slots"
    seed: int = 0
    precision: str = "bfloat16"
    torch_compile: bool = False
    num_workers: int = 8
    cache_dir: str = "/tmp/datasets"
    checkpoint_dir: str = "checkpoints"
    run_id: str = ""
    wandb_project: str = "object-centric-lapo"
    notes: str = ""


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base dict."""
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def _dict_to_dataclass(cls, d: Dict[str, Any]):
    """Recursively convert a dict to a nested dataclass."""
    if not isinstance(d, dict):
        return d
    import dataclasses
    fieldtypes = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for k, v in d.items():
        if k in fieldtypes:
            ft = fieldtypes[k]
            # Resolve string annotations
            if isinstance(ft, str):
                ft = eval(ft)
            if dataclasses.is_dataclass(ft):
                kwargs[k] = _dict_to_dataclass(ft, v)
            elif hasattr(ft, '__origin__') and ft.__origin__ is tuple and isinstance(v, list):
                kwargs[k] = tuple(v)
            else:
                kwargs[k] = v
    return cls(**kwargs)


def load_config(*yaml_paths: str) -> ExperimentConfig:
    """Load and merge YAML config files into an ExperimentConfig.

    Later files override earlier ones.
    """
    merged: Dict[str, Any] = {}
    for path in yaml_paths:
        p = Path(path)
        if p.exists():
            with open(p) as f:
                data = yaml.safe_load(f) or {}
            merged = _merge_dicts(merged, data)
    if not merged:
        return ExperimentConfig()
    return _dict_to_dataclass(ExperimentConfig, merged)
