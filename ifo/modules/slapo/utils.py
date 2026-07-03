from typing import Union

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import Tensor


@torch.compiler.disable()
@torch.no_grad()
def agent_occlusion(batch: TensorDict, occlusion_fraction: float, fill_value: Union[float, Tensor]) -> TensorDict:
    """Occlude a random rectangle inside the agent's tight bounding box.

    The function computes a per-sample agent bounding box from the foreground
    mask, then draws a random sub-rectangle whose area is approximately
    ``occlusion_fraction`` of that bounding box. Pixels inside the rectangle
    are overwritten with ``fill_value`` in the observation and zeroed in the
    mask. The same rectangle is applied identically across all time steps.

    Algorithm:
       1. Compute the agent footprint by OR-ing the mask over time.
       2. Find the tight axis-aligned bounding box of that footprint.
       3. Sample a random rectangle inside the bounding box:
          a. Pick a random aspect ratio.
          b. Derive height and width so that ``h * w ≈ occlusion_fraction * bbox_area``.
          c. Clamp both dimensions to fit the bounding box, re-deriving one
             dimension after clamping the other to preserve area.
          d. Sample a random top-left corner within the remaining slack.
       4. Build a spatial boolean mask from the rectangle and apply it to all
          frames: fill the observation with ``fill_value`` and zero the mask.

       When ``occlusion_fraction >= 1`` the entire bounding box is filled.
       When ``occlusion_fraction <= 0`` the batch is returned unchanged.

    Args:
        batch: TensorDict containing:
            - ``"observation"`` shaped ``(B, T, C, H, W)``
            - ``"mask"`` shaped ``(B, T, H, W)`` or ``(B, T, 1, H, W)``
            Both tensors are modified in place.
        occlusion_fraction: Desired ratio of occluded area to agent bounding-box
            area. Values ``<= 0`` are a no-op; values ``>= 1`` fill the whole
            bounding box.
        fill_value: Constant used to overwrite observation pixels inside the
            occluded region. Either a scalar or a length-``C`` tensor (one value
            per channel). Mask pixels in the region are always set to ``0``.

    Returns:
        The same ``batch`` TensorDict (modified in place).
    """
    observation = batch["observation"]
    mask = batch["mask"]

    if occlusion_fraction <= 0:
        return batch

    # Normalise mask to (B, T, H, W), squeezing a singleton channel dim.
    if mask.dim() == 5 and mask.shape[2] == 1:
        mask_bt_hw = mask.squeeze(2)
    else:
        mask_bt_hw = mask

    if mask_bt_hw.dim() != 4:
        raise ValueError(f"mask must be (B, T, H, W) or (B, T, 1, H, W), got {tuple(mask.shape)}")

    b, t, c, h, w = observation.shape
    if mask_bt_hw.shape != (b, t, h, w):
        raise ValueError(
            f"observation shape {tuple(observation.shape)} incompatible with "
            f"mask shape {tuple(mask.shape)}"
        )

    device = observation.device
    dtype_obs = observation.dtype

    # Prepare the fill colour as a broadcastable (1, 1, C, 1, 1) tensor.
    if isinstance(fill_value, Tensor):
        fill_vec = fill_value.to(device=device, dtype=dtype_obs).reshape(1, 1, c, 1, 1)
    else:
        fill_vec = torch.full((1, 1, c, 1, 1), fill_value, device=device, dtype=dtype_obs)

    # Compute agent footprint: OR the mask over time → (B, H, W).
    agent_union = (mask_bt_hw > 0.5).any(dim=1)
    # Mark samples that have at least one foreground pixel as valid.
    valid = agent_union.any(dim=(1, 2))

    # Find the tight axis-aligned bounding box per sample.
    # For rows: collapse columns to get a (B, H) indicator, then find the first
    # and last active row index to get ay0 (top) and ay1 (bottom, exclusive).
    h_idx = torch.arange(h, device=device, dtype=torch.long).view(1, h).expand(b, h)
    row_active = agent_union.any(dim=2)
    ay0 = torch.where(row_active, h_idx, h).amin(dim=1)
    ay1 = torch.where(row_active, h_idx, -1).amax(dim=1) + 1

    # For columns: collapse rows to get a (B, W) indicator, then find the first
    # and last active column index to get ax0 (left) and ax1 (right, exclusive).
    w_idx = torch.arange(w, device=device, dtype=torch.long).view(1, w).expand(b, w)
    col_active = agent_union.any(dim=1)
    ax0 = torch.where(col_active, w_idx, w).amin(dim=1)
    ax1 = torch.where(col_active, w_idx, -1).amax(dim=1) + 1

    # Bounding-box height, width, and area (all shape (B,)).
    bh = ay1 - ay0
    bw = ax1 - ax0
    bbox_area = bh * bw
    bbox_area_nonneg = bbox_area.clamp(min=0)

    # Compute the target occlusion area in pixels.
    frac = float(occlusion_fraction)
    oc_area = torch.round(frac * bbox_area.float()).long().clamp(min=1)
    oc_area = torch.minimum(oc_area, bbox_area_nonneg)

    # Samples where a partial (sub-bbox) occlusion rectangle is needed.
    partial = valid & (frac < 1.0) & (bbox_area_nonneg > 0) & (oc_area < bbox_area_nonneg)

    # Sample a random occlusion rectangle (h_occ × w_occ).
    # Draw a log-uniform aspect ratio so tall and wide rectangles are equally likely.
    aspect = torch.exp(torch.empty(b, device=device).uniform_(-0.5, 0.5))

    # Derive initial height from area and aspect, then clamp to bbox height.
    h_occ = torch.round((oc_area.float() * aspect).sqrt()).long().clamp(min=1)
    h_occ = torch.minimum(h_occ, bh.clamp(min=0))

    # Derive width from target area and clamped height, then clamp to bbox width.
    w_occ = torch.round(oc_area.float() / h_occ.float().clamp(min=1)).long().clamp(min=1)
    w_occ = torch.minimum(w_occ, bw.clamp(min=0))

    # Re-derive height from target area and clamped width to recover area that
    # was lost when w_occ was clamped down to the bbox width.
    h_occ = torch.round(oc_area.float() / w_occ.float().clamp(min=1)).long().clamp(min=1)
    h_occ = torch.minimum(h_occ, bh.clamp(min=0))

    # Sample a random top-left corner within the remaining bbox slack.
    bh_c = bh.clamp(min=0)
    bw_c = bw.clamp(min=0)
    upper_y = (bh_c - h_occ + 1).clamp(min=1)
    upper_x = (bw_c - w_occ + 1).clamp(min=1)
    iy = torch.floor(torch.rand(b, device=device) * upper_y.float()).long()
    ix = torch.floor(torch.rand(b, device=device) * upper_x.float()).long()
    # Safety clamp so the offset never exceeds the valid range.
    iy = torch.minimum(iy, (upper_y - 1).clamp(min=0))
    ix = torch.minimum(ix, (upper_x - 1).clamp(min=0))

    # Select the final occlusion rectangle per sample.
    # Default to the full agent bounding box (used when frac >= 1).
    y0 = ay0
    y1 = ay1
    x0 = ax0
    x1 = ax1
    # Compute the partial sub-box: top-left = bbox origin + random offset.
    y0_p = ay0 + iy
    y1_p = y0_p + h_occ
    x0_p = ax0 + ix
    x1_p = x0_p + w_occ
    # Use the partial sub-box only where a sub-bbox occlusion is needed.
    y0 = torch.where(partial, y0_p, y0)
    y1 = torch.where(partial, y1_p, y1)
    x0 = torch.where(partial, x0_p, x0)
    x1 = torch.where(partial, x1_p, x1)

    # Build a boolean spatial mask (B, H, W) for the occlusion rectangle.
    yy = torch.arange(h, device=device, dtype=torch.long).view(1, h, 1)
    xx = torch.arange(w, device=device, dtype=torch.long).view(1, 1, w)
    y0_e = y0.view(b, 1, 1)
    y1_e = y1.view(b, 1, 1)
    x0_e = x0.view(b, 1, 1)
    x1_e = x1.view(b, 1, 1)
    occ_hw = (yy >= y0_e) & (yy < y1_e) & (xx >= x0_e) & (xx < x1_e)
    # Suppress occlusion for invalid samples (no foreground at all).
    occ_hw &= valid[:, None, None]

    # Apply the occlusion to every frame.
    # Broadcast the spatial mask to (B, T, H, W) and (B, T, C, H, W).
    occ_bthw = occ_hw.unsqueeze(1).expand(-1, t, -1, -1)
    occ_obs = occ_bthw.unsqueeze(2).expand(-1, -1, c, -1, -1)

    # Overwrite observation pixels with the fill colour inside the rectangle.
    fill_exp = fill_vec.expand(b, t, c, h, w)
    observation.copy_(torch.where(occ_obs, fill_exp, observation))

    # Zero the mask inside the rectangle so downstream code treats these pixels
    # as background.
    zeros_m = torch.zeros_like(mask_bt_hw)
    mask_bt_hw.copy_(torch.where(occ_bthw, zeros_m, mask_bt_hw))

    return batch

@torch.compiler.disable()
@torch.no_grad()
def dilate_mask(mask: Tensor, radius: int) -> Tensor:
    """Morphological dilation of a binary mask using a square structuring element.

    Simulates over-segmentation: the mask grows outward, including nearby
    background pixels. All original agent pixels are preserved.

    Args:
        mask (Tensor): Binary mask tensor of shape (B, T, 1, H, W), values in {0, 1}.
        radius (int): Dilation radius in pixels. Kernel size = 2 * radius + 1.
              radius=0 returns the mask unchanged.

    Returns:
        Dilated binary mask of shape (B, T, 1, H, W).
    """
    if radius <= 0:
        return mask

    kernel_size = 2 * radius + 1
    padding = radius

    b, t, c, h, w = mask.shape

    # Dilation = max-pool with same-padding
    return F.max_pool2d(mask.view(b * t, c, h, w), kernel_size=kernel_size, stride=1, padding=padding).reshape_as(mask)


@torch.compiler.disable()
@torch.no_grad()
def erode_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Morphological erosion of a binary mask using a square structuring element.

    Simulates under-segmentation: the mask shrinks inward, removing boundary
    pixels (limb tips, extremities). Agent pixels near the boundary are lost.

    If erosion eliminates the mask entirely for a frame, returns an all-zero
    mask for that frame (the loss will be zero, equivalent to skipping it).

    Args:
        mask (Tensor): Binary mask tensor of shape (B, T, 1, H, W), values in {0, 1}.
        radius (int): Erosion radius in pixels. Kernel size = 2 * radius + 1.
              radius=0 returns the mask unchanged.

    Returns:
        Eroded binary mask of shape (B, T, 1, H, W).
    """
    if radius <= 0:
        return mask

    kernel_size = 2 * radius + 1
    padding = radius

    b, t, c, h, w = mask.shape

    # Erosion = min-pool = negated max-pool of negated input
    return -F.max_pool2d(-mask.view(b * t, c, h, w), kernel_size=kernel_size, stride=1, padding=padding).reshape_as(mask)
