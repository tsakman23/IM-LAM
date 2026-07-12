#!/usr/bin/env python
"""Post-hoc MaskLAM vs Foreground-MaskLAM reconstruction comparison.

Loads two Stage-1 checkpoints and compares their forward-dynamics-model (FDM)
``pred_next_obs`` on identical frames, to make the core Foreground-MaskLAM claim
visible: **MaskLAM reconstructs the agent well but not the manipulated object,
whereas Foreground-MaskLAM reconstructs both** (selfnote: see Proposal Draft 4, S7.5 / Eq. 4-6).

Each method has two nets. The IDM is an encoder that maps a stack of current+future frames
to a latent action ``z``. The FDM is a world model - itself an encoder-decoder: it encodes
the current frame stack into a low-resolution spatial bottleneck, injects ``z`` there, and
decodes back to the predicted next frame. (SLAPO's code names these two nets ``self.encoder``
and ``self.decoder``, which is just shorthand - the FDM has its own internal encoder and
decoder.) The observation fed to both nets has a mask channel attached.

This script rebuilds each model from its own Stage-1 config, so at test time each gets
exactly the mask input it was trained with:

  - MaskLAM: the agent mask.
  - Foreground-MaskLAM (as trained here - loss-only, ``object_mask_input=false``): the
    SAME agent mask. Its agent-plus-object union changed only the training LOSS, not the
    input. (An optional ablation may feed the union as input too; the code handles that,
    but it is off by default.)

Both models therefore get identical inputs, so any difference in the predicted next frame
comes purely from what each FDM was trained to reproduce: FG's FDM was graded on object
pixels, so it learned to redraw the object; MaskLAM's FDM was graded on agent pixels only,
so it never learned to - it renders the arm well but smears or misses the manipulated
object. Making that gap visible is the whole point of the figure.

Frames are drawn from the push dataset (same ``get_dataset`` path, ``block_size=13``)
where the object mask is non-empty and the object is visibly moving, so reconstruction
differences are meaningful. Errors are strictly silhouette-gated (no background leakage):
for each model and frame, ``err_region = (perpixel_MSE * M).sum() / M.sum()`` computed
separately for ``M`` = object silhouette and ``M`` = agent silhouette. The agent-region
error is the control (both models should be low); the headline is MaskLAM's object-region
error >> Foreground-MaskLAM's.

Per frame a panel is saved under ``docs/figures/fg_vs_masklam/``:
``[GT next-obs | MaskLAM pred | FG pred | union-mask overlay | error-difference map]``
where the error-difference map = ``perpixel_MSE(MaskLAM) - perpixel_MSE(FG)`` shown only
on object-silhouette pixels (background zeroed), so bright = MaskLAM reconstructs the
object worse than FG.

Example
-------
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl \
python scripts/compare_reconstruction.py \
  --env Meta-World/masked-MT1-push-v3 \
  --masklam-ckpt checkpoints/slapo_q1_Meta-World_masked-MT1-push-v3_seed1-1 \
  --fg-ckpt checkpoints/fg_masklam_push-v3_seed1-1 \
  --num-frames 6 \
  --cache-dir ./datasets/slapo/Meta-World/masked-MT1-push-v3
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

# The checkpoint-loading helpers (load_checkpoint_state_dict, filter_state_dict_by_prefix)
# live in scripts/submission2026/eval_slapo.py; put that dir on the path so we can import them.
sys.path.insert(0, os.path.join(os.getcwd(), "scripts", "submission2026"))

CONFIG_DIR = os.path.join(os.getcwd(), "experiments", "configs")


# ---------------------------------------------------------------------------
# Pure, importable helpers (unit-tested in test_compare_reconstruction.py)
# ---------------------------------------------------------------------------
def per_pixel_mse(pred: Tensor, gt: Tensor) -> Tensor:
    """Per-pixel MSE between predicted and ground-truth next observations.

    Args:
        pred (Tensor): Predicted next obs, shape (B, C, H, W).
        gt (Tensor): Ground-truth next obs, shape (B, C, H, W).

    Returns:
        Tensor: Per-pixel squared error averaged over channels, shape (B, H, W).
    """
    assert pred.shape == gt.shape and pred.dim() == 4, f"expected (B,C,H,W), got {pred.shape}"
    return ((pred - gt) ** 2).mean(dim=1)


def region_error(perpixel: Tensor, mask: Tensor, eps: float = 1e-6) -> Tensor:
    """Silhouette-gated mean error: ``(perpixel * M).sum() / M.sum()`` per sample.

    Args:
        perpixel (Tensor): Per-pixel error, shape (B, H, W).
        mask (Tensor): Binary silhouette in {0, 1}, shape (B, H, W).
        eps (float): Guard against empty masks.

    Returns:
        Tensor: Region error per sample, shape (B,). NaN-free (eps-guarded).
    """
    assert perpixel.shape == mask.shape and perpixel.dim() == 3
    num = (perpixel * mask).sum(dim=(1, 2))
    den = mask.sum(dim=(1, 2)).clamp_min(eps)
    return num / den


def _centroid(mask_hw: Tensor) -> Tensor:
    """Return the (row, col) centroid of a binary (H, W) mask; zeros if empty."""
    h, w = mask_hw.shape
    total = mask_hw.sum()
    if total <= 0:
        return torch.zeros(2, device=mask_hw.device)
    rows = torch.arange(h, device=mask_hw.device, dtype=mask_hw.dtype)
    cols = torch.arange(w, device=mask_hw.device, dtype=mask_hw.dtype)
    r = (mask_hw.sum(dim=1) * rows).sum() / total
    c = (mask_hw.sum(dim=0) * cols).sum() / total
    return torch.stack([r, c])


def object_motion_score(object_mask_block: Tensor, frame_stack: int) -> float:
    """Object motion over the current window, as centroid displacement (pixels).

    Measures how far the object centroid moves between the first frame of the window
    and the reconstructed target frame (index ``frame_stack``), so ranking by this
    surfaces frames where the object is actively being pushed.

    Args:
        object_mask_block (Tensor): Object mask window, shape (T, 1, H, W) in {0, 1}.
        frame_stack (int): Index of the reconstructed target frame.

    Returns:
        float: L2 centroid displacement in pixels (0.0 if the object is absent).
    """
    start = _centroid(object_mask_block[0, 0])
    target = _centroid(object_mask_block[frame_stack, 0])
    if object_mask_block[0, 0].sum() <= 0 or object_mask_block[frame_stack, 0].sum() <= 0:
        return 0.0
    return float((target - start).pow(2).sum().sqrt())


def select_moving_object_frames(
    dataset,
    frame_stack: int,
    num_frames: int,
    scan_limit: int,
    min_object_pixels: int,
    min_separation: int,
) -> List[int]:
    """Pick dataset window indices where the object is non-empty and visibly moving.

    Scans up to ``scan_limit`` evenly-spaced windows, keeps those whose target-frame
    object silhouette has at least ``min_object_pixels`` pixels, ranks them by
    object-centroid displacement (descending), and greedily selects ``num_frames``
    indices at least ``min_separation`` apart (so panels are not near-duplicate frames).

    Args:
        dataset: A windowed dataset whose ``[i]`` yields a TensorDict with an
            ``object_mask`` of shape (T, 1, H, W).
        frame_stack (int): Index of the reconstructed target frame.
        num_frames (int): Number of frames to select.
        scan_limit (int): Max number of candidate windows to inspect.
        min_object_pixels (int): Minimum object silhouette area at the target frame.
        min_separation (int): Minimum index gap between selected windows.

    Returns:
        List[int]: Selected window start indices (highest object motion first).
    """
    n = len(dataset)
    stride = max(1, n // scan_limit)
    candidates: List[Tuple[float, int]] = []
    for idx in range(0, n, stride):
        sample = dataset[idx]
        object_mask = sample["object_mask"]  # (T, 1, H, W)
        if object_mask[frame_stack, 0].sum() < min_object_pixels:
            continue
        score = object_motion_score(object_mask, frame_stack)
        if score > 0.0:
            candidates.append((score, idx))
    candidates.sort(key=lambda x: x[0], reverse=True)

    selected: List[int] = []
    for _, idx in candidates:
        if all(abs(idx - s) >= min_separation for s in selected):
            selected.append(idx)
        if len(selected) >= num_frames:
            break
    return selected


def resolve_input_mask(agent_mask: Tensor, object_mask: Tensor, object_mask_input: bool) -> Tensor:
    """Return the mask a model receives at inference, per its config flags.

    Replicates ``SLAPOIDMModule._forward``: the input mask is the agent mask unless
    ``object_mask_input`` is set, in which case it is the agent U object union.

    Args:
        agent_mask (Tensor): Agent mask, shape (..., 1, H, W) in {0, 1}.
        object_mask (Tensor): Object mask, same shape.
        object_mask_input (bool): Whether the config feeds the union to the encoder.

    Returns:
        Tensor: The mask to pass to ``net.forward``.
    """
    if object_mask_input:
        return (agent_mask + object_mask).clamp(0.0, 1.0)
    return agent_mask


def error_difference_map(
    perpixel_a: Tensor, perpixel_b: Tensor, object_silhouette: Tensor
) -> Tensor:
    """MaskLAM-minus-FG per-pixel error, zeroed off the object silhouette.

    Args:
        perpixel_a (Tensor): MaskLAM per-pixel MSE, shape (H, W).
        perpixel_b (Tensor): FG per-pixel MSE, shape (H, W).
        object_silhouette (Tensor): Object silhouette in {0, 1}, shape (H, W).

    Returns:
        Tensor: ``max(a - b, 0) * M`` (background zeroed; bright = MaskLAM worse).
    """
    diff = (perpixel_a - perpixel_b).clamp_min(0.0)
    return diff * object_silhouette


# ---------------------------------------------------------------------------
# Config / model loading (mirrors experiment.stage_1 + eval_slapo)
# ---------------------------------------------------------------------------
def compose_config(config_name: str, overrides: List[str]):
    """Compose a Stage-1 Hydra config in-process (mirrors run_slapo._compose_stage)."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name=config_name, overrides=overrides)


def seed_all(seed: int) -> None:
    """Seed torch (+CUDA) so both models draw the same future-obs sample ``k``."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_net(cfg, device: torch.device):
    """Instantiate the Stage-1 net from ``cfg`` and load its checkpoint weights."""
    import hydra

    from eval_slapo import filter_state_dict_by_prefix, load_checkpoint_state_dict

    env = hydra.utils.instantiate(cfg.env, num_envs=1)
    try:
        observation_space = env.single_observation_space
        action_space = env.single_action_space
    finally:
        env.close()

    net = hydra.utils.instantiate(cfg.net)(observation_space=observation_space, action_space=action_space)
    assert getattr(net, "segmentation_net", None) is None, (
        "compare_reconstruction expects ground-truth masks (segmentation_net=null); "
        "a learned segmentation net is not supported here."
    )
    state_dict = load_checkpoint_state_dict(cfg.trainer.previous_stage_checkpoint)
    net.load_state_dict(filter_state_dict_by_prefix(state_dict, "net"), strict=False)
    net.to(device).eval()
    return net


@torch.no_grad()
def run_model(net, cfg, observation: Tensor, agent_mask: Tensor, object_mask: Tensor, seed: int) -> Tensor:
    """Run one model's FDM on a batch, feeding the mask its config dictates.

    Args:
        net: Loaded ``SLAPOIDM``.
        cfg: That model's Stage-1 config (for ``module.object_mask_input`` / ``mask_observation``).
        observation (Tensor): Centered obs window, shape (B, T, C, H, W).
        agent_mask (Tensor): Agent mask window, shape (B, T, 1, H, W).
        object_mask (Tensor): Object mask window, shape (B, T, 1, H, W).
        seed (int): Seed applied before the forward so future-obs sampling matches across models.

    Returns:
        Tensor: Predicted next observation, shape (B, C, H, W), centered.
    """
    # object_mask_input / mask_observation may be absent on the plain MaskLAM config
    # (the module defaults both to False); read them defensively.
    object_mask_input = bool(cfg.module.get("object_mask_input", False))
    mask_observation = bool(cfg.module.get("mask_observation", False))
    input_mask = resolve_input_mask(agent_mask, object_mask, object_mask_input)
    obs = observation
    if mask_observation:
        obs = obs * input_mask
    seed_all(seed)
    next_observation, _, _, _ = net(obs, input_mask)
    return next_observation


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def _to_display(centered_chw: Tensor):
    """Centered (C, H, W) in [-0.5, 0.5] -> uint8 HWC array in [0, 255]."""
    return ((centered_chw + 0.5) * 255.0).clamp(0, 255).byte().cpu().permute(1, 2, 0).numpy()


def render_panel(
    out_path: str,
    gt_next: Tensor,
    pred_masklam: Tensor,
    pred_fg: Tensor,
    union_overlay: Tensor,
    diff_map: Tensor,
    object_silhouette: Tensor,
    errors: Dict[str, float],
    labels: Tuple[str, str],
) -> None:
    """Render and save the 5-column comparison panel for one frame."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    masklam_label, fg_label = labels
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.6))

    axes[0].imshow(_to_display(gt_next))
    axes[0].set_title("GT next-obs")
    axes[1].imshow(_to_display(pred_masklam))
    axes[1].set_title(f"{masklam_label} pred")
    axes[2].imshow(_to_display(pred_fg))
    axes[2].set_title(f"{fg_label} pred")
    axes[3].imshow(_to_display(union_overlay))
    axes[3].set_title("union-mask overlay")

    diff = diff_map.cpu().numpy()
    vmax = float(diff.max()) if diff.max() > 0 else 1.0
    im = axes[4].imshow(diff, cmap="inferno", vmin=0.0, vmax=vmax)
    axes[4].set_title(f"{masklam_label} extra error (object)")
    cbar = fig.colorbar(im, ax=axes[4], fraction=0.046, pad=0.04)
    cbar.set_label(f"{masklam_label} - {fg_label} per-pixel MSE")

    for ax in axes:
        ax.axis("off")

    caption = (
        f"object-region MSE:  {masklam_label}={errors['obj_masklam']:.4f}  "
        f"{fg_label}={errors['obj_fg']:.4f}   |   "
        f"agent-region MSE (control):  {masklam_label}={errors['agent_masklam']:.4f}  "
        f"{fg_label}={errors['agent_fg']:.4f}"
    )
    fig.suptitle(caption, y=0.02, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="Meta-World/masked-MT1-push-v3")
    p.add_argument("--masklam-label", default="MaskLAM")
    p.add_argument("--masklam-ckpt", default="checkpoints/slapo_q1_Meta-World_masked-MT1-push-v3_seed1-1")
    p.add_argument("--masklam-config", default="slapo_dmw_stage_1")
    p.add_argument("--fg-label", default="FG-MaskLAM")
    p.add_argument("--fg-ckpt", default="checkpoints/fg_masklam_push-v3_seed1-1")
    p.add_argument("--fg-config", default="foreground_masklam_dmw_stage_1")
    p.add_argument("--num-frames", type=int, default=6)
    p.add_argument("--split", default="test")
    p.add_argument("--cache-dir", default="./datasets/slapo/Meta-World/masked-MT1-push-v3")
    p.add_argument("--out", default="docs/figures/fg_vs_masklam")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scan-limit", type=int, default=1500, help="Max candidate windows to inspect for object motion.")
    p.add_argument("--min-object-pixels", type=int, default=20, help="Min object silhouette area at the target frame.")
    p.add_argument("--min-separation", type=int, default=64, help="Min index gap between selected frames.")
    return p.parse_args()


def main() -> None:
    import hydra

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    common_overrides = [f"env.name={args.env}", f"dataset.cache_dir={args.cache_dir}"]
    masklam_cfg = compose_config(
        args.masklam_config, common_overrides + [f"trainer.previous_stage_checkpoint={args.masklam_ckpt}"]
    )
    fg_cfg = compose_config(
        args.fg_config, common_overrides + [f"trainer.previous_stage_checkpoint={args.fg_ckpt}"]
    )

    frame_stack = int(masklam_cfg.net.frame_stack)

    # Frames come from the object-mask (FG) dataset so silhouette gating is available;
    # observations are IID with the authors' repo, so feeding them to both nets is fair.
    print(f"[compare] loading frames from FG dataset ({fg_cfg.dataset.dataset_path}, split={args.split}) ...")
    dataset = hydra.utils.instantiate(fg_cfg.dataset, split=args.split)
    indices = select_moving_object_frames(
        dataset,
        frame_stack=frame_stack,
        num_frames=args.num_frames,
        scan_limit=args.scan_limit,
        min_object_pixels=args.min_object_pixels,
        min_separation=args.min_separation,
    )
    if not indices:
        raise RuntimeError("No frames with a moving, non-empty object mask were found; relax --min-object-pixels.")
    print(f"[compare] selected {len(indices)} frame windows: {indices}")

    batch = torch.stack([dataset[i] for i in indices], dim=0)  # TensorDict (B, T, ...)
    observation = batch["observation"].to(device)
    agent_mask = batch["mask"].to(device)
    object_mask = batch["object_mask"].to(device)

    print(f"[compare] loading {args.masklam_label} net from {args.masklam_ckpt} ...")
    masklam_net = load_net(masklam_cfg, device)
    print(f"[compare] loading {args.fg_label} net from {args.fg_ckpt} ...")
    fg_net = load_net(fg_cfg, device)

    pred_masklam = run_model(masklam_net, masklam_cfg, observation, agent_mask, object_mask, args.seed)
    pred_fg = run_model(fg_net, fg_cfg, observation, agent_mask, object_mask, args.seed)

    gt_next = observation[:, frame_stack]  # (B, C, H, W)
    ppm_masklam = per_pixel_mse(pred_masklam, gt_next)  # (B, H, W)
    ppm_fg = per_pixel_mse(pred_fg, gt_next)

    object_sil = object_mask[:, frame_stack, 0]  # (B, H, W)
    agent_sil = agent_mask[:, frame_stack, 0]
    union_sil = (agent_mask[:, frame_stack] + object_mask[:, frame_stack]).clamp(0.0, 1.0)  # (B, 1, H, W)

    obj_err_masklam = region_error(ppm_masklam, object_sil)
    obj_err_fg = region_error(ppm_fg, object_sil)
    agent_err_masklam = region_error(ppm_masklam, agent_sil)
    agent_err_fg = region_error(ppm_fg, agent_sil)

    # Union overlay: green tint on the union region over the GT next observation.
    alpha = 0.5
    green = torch.zeros_like(gt_next)
    green[:, 1] = 0.5  # centered space: +0.5 == 255 after un-centering
    union_overlay = gt_next * (1.0 - alpha * union_sil) + green * (alpha * union_sil)

    # Tee the MSE log to both stdout and <out>/fg_vs_masklam_mse.txt.
    report: List[str] = []

    def emit(line: str = "") -> None:
        print(line)
        report.append(line)

    header = (
        f"{'frame':>8} | {'obj MSE ' + args.masklam_label:>18} {'obj MSE ' + args.fg_label:>18} "
        f"| {'agent MSE ' + args.masklam_label:>20} {'agent MSE ' + args.fg_label:>20}"
    )
    emit("\n" + header)
    emit("-" * len(header))
    for b, idx in enumerate(indices):
        errors = {
            "obj_masklam": float(obj_err_masklam[b]),
            "obj_fg": float(obj_err_fg[b]),
            "agent_masklam": float(agent_err_masklam[b]),
            "agent_fg": float(agent_err_fg[b]),
        }
        emit(
            f"{idx:>8} | {errors['obj_masklam']:>18.5f} {errors['obj_fg']:>18.5f} "
            f"| {errors['agent_masklam']:>20.5f} {errors['agent_fg']:>20.5f}"
        )
        diff_map = error_difference_map(ppm_masklam[b], ppm_fg[b], object_sil[b])
        out_path = os.path.join(args.out, f"frame_{idx:07d}.png")
        render_panel(
            out_path,
            gt_next[b],
            pred_masklam[b],
            pred_fg[b],
            union_overlay[b],
            diff_map,
            object_sil[b],
            errors,
            labels=(args.masklam_label, args.fg_label),
        )

    emit(
        f"\n[compare] object-region mean MSE: {args.masklam_label}={float(obj_err_masklam.mean()):.5f}  "
        f"{args.fg_label}={float(obj_err_fg.mean()):.5f}"
    )
    emit(
        f"[compare] agent-region mean MSE (control): {args.masklam_label}={float(agent_err_masklam.mean()):.5f}  "
        f"{args.fg_label}={float(agent_err_fg.mean()):.5f}"
    )
    emit(f"[compare] wrote {len(indices)} panels to {args.out}/")

    report_path = os.path.join(args.out, "fg_vs_masklam_mse.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report) + "\n")
    print(f"[compare] wrote MSE log to {report_path}")


if __name__ == "__main__":
    main()
