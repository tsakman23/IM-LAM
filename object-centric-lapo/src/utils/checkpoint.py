from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    step: int,
    epoch: int,
    path: str,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "step": step,
        "epoch": epoch,
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if extra:
        state.update(extra)
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = "cpu",
) -> Dict[str, Any]:
    """Load a training checkpoint."""
    state = torch.load(path, map_location=map_location, weights_only=False)
    if model is not None and "model" in state:
        model.load_state_dict(state["model"])
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    return state


def get_latest_checkpoint(checkpoint_dir: str, prefix: str = "") -> Optional[str]:
    """Find the latest checkpoint in a directory by modification time."""
    p = Path(checkpoint_dir)
    if not p.exists():
        return None
    pattern = f"{prefix}*.pt" if prefix else "*.pt"
    checkpoints = sorted(p.glob(pattern), key=lambda x: x.stat().st_mtime)
    return str(checkpoints[-1]) if checkpoints else None


def filter_state_dict_by_prefix(state_dict: Dict[str, Any], prefix: str) -> OrderedDict:
    """Filter state_dict to keys starting with prefix, removing the prefix."""
    if not prefix.endswith("."):
        prefix += "."
    filtered = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith(prefix):
            filtered[key[len(prefix):]] = value
    return filtered
