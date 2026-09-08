import random
import string
from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch import Tensor


def random_tag(length: int = 5) -> str:
    """Generate a short random alphanumeric tag for run names."""
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))


def merge_tc(observation: Tensor) -> Tensor:
    """Merge time and channel dimensions.

    (B, T, C, H, W) -> (B, T*C, H, W)
    """
    b, t, c, h, w = observation.shape
    return observation.reshape(b, t * c, h, w)


def orthogonal_init(
    module: nn.Module,
    modules_to_apply: Optional[List[Type[nn.Module]]] = None,
    gain: float = 1.0,
) -> None:
    """Orthogonal initialization of weights."""
    if modules_to_apply is None:
        modules_to_apply = [nn.Linear, nn.Conv2d, nn.ConvTranspose2d]
    if isinstance(module, tuple(modules_to_apply)):
        nn.init.orthogonal_(module.weight.data, gain)
        if hasattr(module.bias, "data"):
            module.bias.data.fill_(0.0)


def center(x: Tensor) -> Tensor:
    """Center pixel values: [0, 255] -> [-0.5, 0.5]."""
    return x / 255.0 - 0.5


def uncenter(x: Tensor) -> Tensor:
    """Uncenter pixel values: [-0.5, 0.5] -> [0, 1]."""
    return x + 0.5


@torch.no_grad()
def make_comparison_grid(
    images: Tensor,
    reconstructions: Tensor,
    nrow: int = 8,
) -> Tensor:
    """Create a comparison grid of original vs reconstructed images.

    Args:
        images: (B, C, H, W) original images in [-0.5, 0.5].
        reconstructions: (B, C, H, W) reconstructed images in [-0.5, 0.5].
        nrow: Number of images per row.

    Returns:
        Grid tensor suitable for wandb.Image.
    """
    from torchvision.utils import make_grid

    n = min(images.shape[0], nrow)
    imgs = (images[:n] + 0.5).clamp(0, 1)
    recons = (reconstructions[:n] + 0.5).clamp(0, 1)
    grid = make_grid(torch.cat([imgs, recons], dim=0), nrow=n)
    return grid


@torch.no_grad()
def visualize_slot_masks(
    images: Tensor,
    attn_masks: Tensor,
    nrow: int = 8,
) -> Tensor:
    """Visualize slot attention masks overlaid on images.

    Args:
        images: (B, C, H, W) in [-0.5, 0.5].
        attn_masks: (B, K, N_patches) attention masks.
        nrow: Number of images per row.

    Returns:
        Grid tensor.
    """
    from torchvision.utils import make_grid
    import torch.nn.functional as F

    b, c, h, w = images.shape
    k = attn_masks.shape[1]
    n = min(b, nrow)
    imgs = (images[:n] + 0.5).clamp(0, 1)

    # Reshape masks to spatial
    n_patches = attn_masks.shape[2]
    patch_h = int(n_patches**0.5)
    masks = attn_masks[:n].reshape(n, k, patch_h, patch_h)
    masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)

    # Create colored overlays
    rows = [imgs]
    for s in range(k):
        mask_s = masks[:, s:s+1]  # (n, 1, H, W)
        overlay = imgs * mask_s
        rows.append(overlay)

    grid = make_grid(torch.cat(rows, dim=0), nrow=n)
    return grid


# Fixed slot colors (saturated, perceptually distinct).
SLOT_COLORS = [
    (1.0, 0.0, 0.0),   # red
    (0.0, 0.7, 0.0),   # green
    (0.2, 0.4, 1.0),   # blue
    (1.0, 0.8, 0.0),   # yellow
    (1.0, 0.0, 1.0),   # magenta
    (0.0, 0.9, 0.9),   # cyan
    (1.0, 0.5, 0.0),   # orange
    (0.5, 0.0, 1.0),   # purple
    (0.0, 1.0, 0.5),   # spring green
    (0.8, 0.4, 0.4),   # salmon
    (0.4, 0.8, 0.4),   # light green
    (0.4, 0.4, 0.8),   # light blue
    (0.9, 0.9, 0.5),   # khaki
    (0.9, 0.5, 0.9),   # orchid
    (0.5, 0.9, 0.9),   # pale cyan
]


@torch.no_grad()
def visualize_slot_decomposition(
    images: Tensor,
    attn_masks: Tensor,
    nrow: int = 8,
) -> Tensor:
    """Color-coded slot decomposition: each pixel colored by its dominant slot.

    Args:
        images: (B, C, H, W) in [-0.5, 0.5].
        attn_masks: (B, K, N_patches) attention masks.
        nrow: Number of images per row.

    Returns:
        Grid tensor with original images and color-coded decomposition.
    """
    from torchvision.utils import make_grid
    import torch.nn.functional as F

    b, c, h, w = images.shape
    k = attn_masks.shape[1]
    n = min(b, nrow)
    device = images.device
    imgs = (images[:n] + 0.5).clamp(0, 1)

    # Reshape masks to spatial
    n_patches = attn_masks.shape[2]
    patch_h = int(n_patches ** 0.5)
    masks = attn_masks[:n].reshape(n, k, patch_h, patch_h)
    masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)  # (n, K, H, W)

    # Build color map tensor (K, 3)
    colors = torch.tensor(SLOT_COLORS[:k], device=device, dtype=imgs.dtype)  # (K, 3)

    # Soft color mix: weighted sum of slot colors by attention weight
    # masks: (n, K, H, W) -> (n, K, H, W, 1), colors: (K, 3) -> (1, K, 1, 1, 3)
    color_map = (masks.unsqueeze(-1) * colors[None, :, None, None, :]).sum(dim=1)  # (n, H, W, 3)
    color_map = color_map.permute(0, 3, 1, 2)  # (n, 3, H, W)

    # Blend: 60% color map + 40% original grayscale for context
    gray = imgs.mean(dim=1, keepdim=True).expand_as(imgs)
    blended = 0.6 * color_map + 0.4 * gray

    grid = make_grid(torch.cat([imgs, blended], dim=0), nrow=n)
    return grid


@torch.no_grad()
def visualize_slot_heatmaps(
    attn_masks: Tensor,
    nrow: int = 8,
) -> Tensor:
    """Render each slot's attention mask as a grayscale heatmap.

    Args:
        attn_masks: (B, K, N_patches) attention masks.
        nrow: Number of images per row.

    Returns:
        Grid tensor: rows = slots, columns = samples.
    """
    from torchvision.utils import make_grid

    b, k, n_patches = attn_masks.shape
    n = min(b, nrow)
    patch_h = int(n_patches ** 0.5)

    masks = attn_masks[:n].reshape(n, k, patch_h, patch_h)  # (n, K, ph, ph)

    # Stack all slots as separate single-channel images
    panels = []
    for s in range(k):
        slot_mask = masks[:, s:s + 1]  # (n, 1, ph, ph)
        # Expand to 3-channel for grid
        slot_mask = slot_mask.expand(-1, 3, -1, -1)
        panels.append(slot_mask)

    grid = make_grid(torch.cat(panels, dim=0), nrow=n)
    return grid


@torch.no_grad()
def compute_slot_metrics(attn_masks: Tensor) -> dict:
    """Compute per-slot statistics from attention masks.

    Args:
        attn_masks: (B, K, N_patches) attention masks (sum to 1 over K per patch).

    Returns:
        Dict with slot_entropy, slot_coverage, slot_max_attn.
    """
    b, k, n = attn_masks.shape

    # Attention entropy per patch: -sum_k p_k log p_k, then average over patches and batch
    eps = 1e-8
    entropy = -(attn_masks * (attn_masks + eps).log()).sum(dim=1)  # (B, N)
    mean_entropy = entropy.mean().item()
    max_entropy = torch.log(torch.tensor(float(k))).item()

    # Coverage: fraction of patches where each slot is dominant
    dominant = attn_masks.argmax(dim=1)  # (B, N)
    coverage = []
    for s in range(k):
        frac = (dominant == s).float().mean().item()
        coverage.append(frac)

    # Mean max attention per patch (how peaked the assignment is)
    max_attn = attn_masks.max(dim=1).values.mean().item()

    return {
        "slot_entropy": mean_entropy,
        "slot_entropy_normalized": mean_entropy / max(max_entropy, eps),
        "slot_max_attn": max_attn,
        **{f"slot_{s}_coverage": c for s, c in enumerate(coverage)},
    }
