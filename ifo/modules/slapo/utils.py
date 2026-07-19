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


def area_normalized_masked_mse(error: Tensor, mask: Tensor) -> Tensor:
    """Area-normalized masked MSE: mean squared error over the pixels inside ``mask``.

    Sums ``error`` (typically the elementwise output of
    ``F.mse_loss(..., reduction="none")``) over the region where ``mask`` is
    nonzero, and normalizes by the mask's own area (``mask.sum()``) rather than
    the total number of pixels. This keeps the loss magnitude comparable across
    samples with differently sized masks, and is the shared building block for
    both MaskLAM's single-mask loss and IM-LAM's dual loss below.

    The denominator is floored at one pixel: a batch whose mask is empty
    everywhere (e.g. the object fully occluded in every sampled target frame,
    conceivable on the occlusion stress task peg-insert-side) yields a zero
    loss contribution instead of a 0/0 NaN that would poison training. For
    binary masks the floor changes nothing else, since any nonempty mask has
    area >= 1.

    Args:
        error (Tensor): Elementwise error, broadcastable against ``mask``.
        mask (Tensor): Binary mask, broadcastable against ``error``.

    Returns:
        Scalar tensor: ``(error * mask).sum() / mask.sum().clamp(min=1)``.
    """
    return (error * mask).sum() / mask.sum().clamp(min=1.0)


def dual_masked_reconstruction_loss(
    error: Tensor, agent_mask: Tensor, object_mask: Tensor, lambda_o: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Dual loss: agent and object reconstruction error, each independently normalized.

    L = L_A + lambda_o * L_O, where L_A and L_O are each an
    :func:`area_normalized_masked_mse` over their own mask. This differs from
    the alternative union loss (one mask, one shared normalization) in
    that a small object mask does not have its contribution diluted by the much
    larger agent mask - each region gets its own budget, sized by ``lambda_o``
    rather than by relative pixel count.

    Args:
        error (Tensor): Elementwise error, broadcastable against both masks.
        agent_mask (Tensor): Binary agent mask.
        object_mask (Tensor): Binary manipulated-object mask.
        lambda_o (float): Weight on the object term.

    Returns:
        Tuple of ``(loss, loss_agent, loss_object)`` scalar tensors, so callers
        can log each term separately.
    """
    loss_agent = area_normalized_masked_mse(error, agent_mask)
    loss_object = area_normalized_masked_mse(error, object_mask)
    loss = loss_agent + lambda_o * loss_object
    return loss, loss_agent, loss_object


def dual_loss_grad_norms(
    loss_agent: Tensor, loss_object: Tensor, params: list
) -> tuple[Tensor, Tensor]:
    """Per-term gradient L2 norms of the dual loss w.r.t. shared ``params``.

    Even with per-term area normalization, ``loss_agent`` and ``loss_object`` both
    backpropagate into the same shared parameters (e.g. the FDM decoder), and
    normalization alone does not guarantee comparable *gradient* influence there -
    a term can be unit-scale-normalized and still be gradient-starved if its path
    through the network has a smaller Jacobian. This computes each term's gradient
    independently via ``torch.autograd.grad`` (not ``.backward()``), so neither call
    touches ``.grad`` or disturbs the real optimization backward pass that follows
    on the combined loss. Both calls use ``retain_graph=True`` since that combined
    backward pass still needs the graph afterwards.

    Intended as a periodic (not every-step) training-time diagnostic, since each
    call is an extra backward pass through ``params``.

    Args:
        loss_agent (Tensor): Scalar agent-region loss term.
        loss_object (Tensor): Scalar object-region loss term.
        params (list[Tensor]): Shared parameters both terms flow through.

    Returns:
        Tuple of scalar tensors ``(grad_norm_agent, grad_norm_object)``, each the
        global L2 norm of that term's gradient over ``params`` (unused parameters
        contribute zero).
    """
    grads_agent = torch.autograd.grad(loss_agent, params, retain_graph=True, allow_unused=True)
    grads_object = torch.autograd.grad(loss_object, params, retain_graph=True, allow_unused=True)
    grad_norm_agent = torch.linalg.vector_norm(
        torch.stack([g.norm() for g in grads_agent if g is not None])
    )
    grad_norm_object = torch.linalg.vector_norm(
        torch.stack([g.norm() for g in grads_object if g is not None])
    )
    return grad_norm_agent, grad_norm_object


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
