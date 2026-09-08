"""Multi-model FDM reconstruction panel (Phase 9): MaskLAM vs FG-union vs FG-dual vs IM-LAM.

Makes the object-reconstruction progression visible on ONE figure: the agent-only MaskLAM FDM smears
or misses the manipulated object, FG (object-in-loss) redraws it, and IM-LAM's directed FDM predicts
where it MOVES to. Each model's frozen Stage-1 FDM predicts the same target frame; we crop to the
object and print the object-region MSE per model.

Frame choice is principled, not cosmetic (see the design discussion): the target is a
MAX-PER-STEP-OBJECT-MOTION frame, because that is the discriminative "predict vs copy" regime. On a
static-object frame the target object sits where the input object was, so copy-forward is trivially
correct and the figure cannot separate "renders the object's appearance" from "predicts its dynamics".
On a moving-object frame the target object is at a NEW location, so a copy lands in the wrong place and
only a model that predicts dynamics scores well. The frame must also leave the full block_size window
inside ONE episode (else the IDM's future_obs_offset frame leaks across the episode boundary and
corrupts z_t for every model), which is why e.g. handle-pull uses the burst-3 peak (70), not the
final-burst peak (89) that falls too close to the episode end.

All four models take a uniform call: SLAPOIDM.forward (MaskLAM / FG) accepts-and-ignores object_mask,
IMLAMIDM requires it, so net(obs, agent_mask, object_mask=...) is correct for every model. Every model
receives the agent mask as INPUT (all have object_mask_input=false); the object mask reaches only
IM-LAM's FDM and is used here purely for scoring/cropping the object region.

Env (per experiments.md): conda_env/bin/python, MUJOCO_GL not needed (spaces are derived from a data
batch, no env instantiation). Default reads the RAM-staged /tmp/slapo_local; override --data-path.

Usage:
    conda_env/bin/python scripts/imlam_diagnostics/reconstruction_panel.py \
        --task handle-pull-v3 --data-path /tmp/slapo_local
    
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # for compare_reconstruction

import hydra  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402

from ifo.common.utils.utility import tensordict_collate  # noqa: E402
from compare_reconstruction import per_pixel_mse, region_error, seed_all, _to_display  # noqa: E402
from scripts.imlam_diagnostics.run_diagnostics import load_frozen_net  # noqa: E402

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "experiments", "configs"))

# One model column = (label, config_name, checkpoint). config_name selects the net architecture
# (SLAPOIDM vs IMLAMIDM); the checkpoint supplies the trained net.* weights. Each entry is the best
# available Stage-1 checkpoint (the saver keeps the val-best step, so these are early-stopped, not
# truncated - e.g. MaskLAM's 15000 is its best; val loss rose after).
MODEL_SETS = {
    "push-v3": {
        "seed1": [
            ("MaskLAM",                    "slapo_dmw_stage_1",                   "checkpoints/masklam_push_seed1-1/step-000031248.ckpt"),
            ("Foreground-MaskLAM (Union)", "foreground_masklam_dmw_stage_1",      "checkpoints/fg_masklam_push-v3_seed1-1/step-000031248.ckpt"),
            ("Foreground-MaskLAM (Dual)",  "foreground_masklam_dual_dmw_stage_1", "checkpoints/dual_masklam_push_seed1-1/step-000010000.ckpt"),
            ("IM-LAM",                     "imlam_dmw_stage_1",                   "checkpoints/im-lam_push_union_seed1-1/step-000031248.ckpt"),
        ],
        "seed2": [
            ("Foreground-MaskLAM (Union)", "foreground_masklam_dmw_stage_1",      "checkpoints/fg_masklam_push_seed2_retry4-1/step-000015000.ckpt"),
            ("Foreground-MaskLAM (Dual)",  "foreground_masklam_dual_dmw_stage_1", "checkpoints/dual_masklam_push_seed2-1/step-000031248.ckpt"),
            ("IM-LAM",                     "imlam_dmw_stage_1",                   "checkpoints/im-lam_push_union_seed2-1/step-000031248.ckpt"),
        ],
        "seed3": [
            ("Foreground-MaskLAM (Union)", "foreground_masklam_dmw_stage_1",      "checkpoints/fg_masklam_push_seed3_retry-1/step-000031248.ckpt"),
            ("Foreground-MaskLAM (Dual)",  "foreground_masklam_dual_dmw_stage_1", "checkpoints/dual_masklam_push_seed3-1/step-000031248.ckpt"),
            ("IM-LAM",                     "imlam_dmw_stage_1",                   "checkpoints/im-lam_push_union_seed3-1/step-000015000.ckpt")
        ]
    },
    "sweep-into-v3": {
        "seed1": [
            ("MaskLAM",                    "slapo_dmw_stage_1",                   "checkpoints/slapo_sweepinto_seed1_reproduction-1/step-000015000.ckpt"),
            ("Foreground-MaskLAM (Union)", "foreground_masklam_dmw_stage_1",      "checkpoints/fg_masklam_sweep-into_seed1-1/step-000015000.ckpt"),
            ("Foreground-MaskLAM (Dual)",  "foreground_masklam_dual_dmw_stage_1", "checkpoints/dual_masklam_sweep-into_seed1-1/step-000010000.ckpt"),
            ("IM-LAM",                     "imlam_dmw_stage_1",                   "checkpoints/im-lam_sweep-into_union_seed1-1/step-000020000.ckpt"),
        ],
        "seed2": [
            ("MaskLAM",                    "slapo_dmw_stage_1",                   "checkpoints/masklam_sweep-into_seed2_retry3-1/step-000031248.ckpt"),
            ("Foreground-MaskLAM (Union)", "foreground_masklam_dmw_stage_1",      "checkpoints/fg_masklam_sweep-into_seed2-1/step-000031248.ckpt"),
            ("Foreground-MaskLAM (Dual)",  "foreground_masklam_dual_dmw_stage_1", "checkpoints/dual_masklam_sweep-into_seed2_replay-1/step-000005000.ckpt"),
            ("IM-LAM",                     "imlam_dmw_stage_1",                   "checkpoints/im-lam_sweep-into_union_seed2-1/step-000031248.ckpt"),
        ],
        "seed3": [
            ("MaskLAM",                    "slapo_dmw_stage_1",                   "checkpoints/masklam_sweep-into_seed3_retry3-1/step-000015000.ckpt"),
            ("Foreground-MaskLAM (Union)", "foreground_masklam_dmw_stage_1",      "checkpoints/fg_masklam_sweep-into_seed3-1/step-000015000.ckpt"),
            ("Foreground-MaskLAM (Dual)",  "foreground_masklam_dual_dmw_stage_1", "checkpoints/dual_masklam_sweep-into_seed3-1/step-000010000.ckpt"),
            ("IM-LAM",                     "imlam_dmw_stage_1",                   "checkpoints/im-lam_sweep-into_union_seed3-1/step-000020000.ckpt"),
        ]
    },
    "door-open-v3": {
        "seed1": [
            ("MaskLAM",                    "slapo_dmw_stage_1",                   "checkpoints/masklam_door-open_seed1-1/step-000010000.ckpt"),
            ("Foreground-MaskLAM (Union)", "foreground_masklam_dmw_stage_1",      "checkpoints/fg_masklam_door-open_seed1-1/step-000031248.ckpt"),
            ("Foreground-MaskLAM (Dual)",  "foreground_masklam_dual_dmw_stage_1", "checkpoints/dual_masklam_door-open_seed1-1/step-000010000.ckpt"),
            ("IM-LAM",                     "imlam_dmw_stage_1",                   "checkpoints/im-lam_door-open_union_seed1-1/step-000020000.ckpt"),
        ],
        "seed2": [
            ("Foreground-MaskLAM (Union)", "foreground_masklam_dmw_stage_1",      "checkpoints/fg_masklam_door-open_seed2-1/step-000015000.ckpt"),
            ("Foreground-MaskLAM (Dual)",  "foreground_masklam_dual_dmw_stage_1", "checkpoints/dual_masklam_door-open_seed2-1/step-000031248.ckpt"),
            ("IM-LAM",                     "imlam_dmw_stage_1",                   "checkpoints/im-lam_door-open_union_seed2_retry-1/step-000020000.ckpt"),
        ],
        "seed3": [
            ("Foreground-MaskLAM (Union)", "foreground_masklam_dmw_stage_1",      "checkpoints/fg_masklam_door-open_seed3-1/step-000031248.ckpt"),
            ("Foreground-MaskLAM (Dual)",  "foreground_masklam_dual_dmw_stage_1", "checkpoints/dual_masklam_door-open_seed3-1/step-000031248.ckpt"),
            ("IM-LAM",                     "imlam_dmw_stage_1",                   "checkpoints/im-lam_door-open_union_seed3-1/step-000031248.ckpt")
        ]
    },
    "handle-pull-v3": {
        "seed1": [
            ("MaskLAM",                    "slapo_dmw_stage_1",                   "checkpoints/masklam_handle-pull-v3_seed1-1/step-000015000.ckpt"),
            ("Foreground-MaskLAM (Union)", "foreground_masklam_dmw_stage_1",      "checkpoints/fg_masklam_handle-pull_seed1-1/step-000031248.ckpt"),
            ("Foreground-MaskLAM (Dual)",  "foreground_masklam_dual_dmw_stage_1", "checkpoints/dual_masklam_handle-pull_seed1-1/step-000031248.ckpt"),
            ("IM-LAM",                     "imlam_dmw_stage_1",                   "checkpoints/im-lam_handle-pull_union_seed1-retry-1/step-000031248.ckpt"),
        ],
        "seed2": [
            ("MaskLAM",                    "slapo_dmw_stage_1",                   "checkpoints/masklam_handle-pull-v3_seed2-1/step-000015000.ckpt"),
            ("Foreground-MaskLAM (Union)", "foreground_masklam_dmw_stage_1",      "checkpoints/fg_masklam_handle-pull_seed2-1/step-000015000.ckpt"),
            ("Foreground-MaskLAM (Dual)",  "foreground_masklam_dual_dmw_stage_1", "checkpoints/dual_masklam_handle-pull_seed2-1/step-000015000.ckpt"),
            ("IM-LAM",                     "imlam_dmw_stage_1",                   "checkpoints/im-lam_handle-pull_union_seed2-1/step-000025000.ckpt"),
        ],
        "seed3": [
            ("Foreground-MaskLAM (Union)", "foreground_masklam_dmw_stage_1",      "checkpoints/fg_masklam_handle-pull_seed3-1/step-000015000.ckpt"),
            ("Foreground-MaskLAM (Dual)",  "foreground_masklam_dual_dmw_stage_1", "checkpoints/dual_masklam_handle-pull_seed3-1/step-000015000.ckpt"),
            ("IM-LAM",                     "imlam_dmw_stage_1",                   "checkpoints/im-lam_handle-pull_union_seed3_retry2-1/step-000015000.ckpt"),
        ]
    },
    "pick-place-v3": {
        "seed1": [
            ("MaskLAM",                    "slapo_dmw_stage_1",                   "checkpoints/masklam_pick-place_seed1-1/step-000015000.ckpt"),
            ("Foreground-MaskLAM (Union)", "foreground_masklam_dmw_stage_1",      "checkpoints/fg_masklam_pick-place_seed1-1/step-000010000.ckpt"),
            ("Foreground-MaskLAM (Dual)",  "foreground_masklam_dual_dmw_stage_1", "checkpoints/dual_masklam_pick-place_seed1-1/step-000010000.ckpt"),
            ("IM-LAM",                     "imlam_dmw_stage_1",                   "checkpoints/im-lam_pick-place_union_seed1_retry5-1/step-000020000.ckpt"),
        ],
        "seed2": [
            ("Foreground-MaskLAM (Union)", "foreground_masklam_dmw_stage_1",      "checkpoints/fg_masklam_pick-place_seed2-1/step-000015000.ckpt"),
            ("Foreground-MaskLAM (Dual)",  "foreground_masklam_dual_dmw_stage_1", "checkpoints/dual_masklam_pick-place_seed2-1/step-000031248.ckpt"),
            ("IM-LAM",                     "imlam_dmw_stage_1",                   "checkpoints/im-lam_pick-place_union_seed2-1/step-000031248.ckpt"),
        ],
        "seed3": [
            ("Foreground-MaskLAM (Union)", "foreground_masklam_dmw_stage_1",      "checkpoints/fg_masklam_pick-place_seed3-1/step-000010000.ckpt"),
            ("Foreground-MaskLAM (Dual)",  "foreground_masklam_dual_dmw_stage_1", "checkpoints/dual_masklam_pick-place_seed3-redo3-1/step-000010000.ckpt"),
            ("IM-LAM",                     "imlam_dmw_stage_1",                   "checkpoints/im-lam_pick-place_union_seed3-1/step-000015000.ckpt"),
        ]
    },
    "peg-insert-side-v3": {
        "seed1": [
            ("MaskLAM",                    "slapo_dmw_stage_1",                   "checkpoints/masklam_peg-insert-side_seed1-1/step-000010000.ckpt"),
            ("Foreground-MaskLAM (Union)", "foreground_masklam_dmw_stage_1",      "checkpoints/fg_masklam_peg-insert-side_seed1_retry-1/step-000010000.ckpt"),
            ("Foreground-MaskLAM (Dual)",  "foreground_masklam_dual_dmw_stage_1", "checkpoints/dual_masklam_peg-insert-side_seed1-1/step-000025000.ckpt"),
            ("IM-LAM",                     "imlam_dmw_stage_1",                   "checkpoints/im-lam_peg-insert-side_union_seed1_retry-1/step-000015000.ckpt"),
        ],
        "seed2": [
            ("Foreground-MaskLAM (Union)", "foreground_masklam_dmw_stage_1",      "checkpoints/fg_masklam_peg-insert-side_seed2-1/step-000015000.ckpt"),
            ("Foreground-MaskLAM (Dual)",  "foreground_masklam_dual_dmw_stage_1", "checkpoints/dual_masklam_peg-insert-side_seed2-1/step-000010000.ckpt"),
            ("IM-LAM",                     "imlam_dmw_stage_1",                   "checkpoints/im-lam_peg-insert-side_union_seed2_retry-1/step-000020000.ckpt"),
        ],
        "seed3": [
            ("Foreground-MaskLAM (Union)", "foreground_masklam_dmw_stage_1",      "checkpoints/fg_masklam_peg-insert-side_seed3-1/step-000010000.ckpt"),
            ("Foreground-MaskLAM (Dual)",  "foreground_masklam_dual_dmw_stage_1", "checkpoints/dual_masklam_peg-insert-side_seed3-1/step-000031248.ckpt"),
            ("IM-LAM",                     "imlam_dmw_stage_1",                   "checkpoints/im-lam_peg-insert-side_union_seed3-1/step-000015000.ckpt"),
        ]
    }
    
}

# Constrained max-per-step-object-motion target frames, episode 0 (computed from object_state; the full
# block_size=13 window stays inside episode 0). Used when --target-frames is not given.
DEFAULT_TARGET_FRAMES = {
    "handle-pull-v3": [30, 50, 70],
    "push-v3": [45, 51],
    "sweep-into-v3": [18, 29, 35],
    "door-open-v3": [55, 61, 67],
    "pick-place-v3": [32, 38, 44],
    "peg-insert-side-v3": [64, 70, 76],
}


def _compose(config_name, task, data_path):
    GlobalHydra.instance().clear()
    overrides = [f"env.name=Meta-World/masked-MT1-{task}", "++module.log_dual_loss_grad_every=0"]
    if data_path:
        overrides.append(f"dataset.dataset_path={data_path}")
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name=config_name, overrides=overrides)


def _object_bbox(object_sil, margin):
    """(rmin, rmax, cmin, cmax) bounding box of a binary (H, W) mask, padded by `margin` and clamped."""
    ys, xs = np.nonzero(object_sil > 0)
    h, w = object_sil.shape
    if len(ys) == 0:
        return 0, h, 0, w
    rmin, rmax = max(0, ys.min() - margin), min(h, ys.max() + 1 + margin)
    cmin, cmax = max(0, xs.min() - margin), min(w, xs.max() + 1 + margin)
    return rmin, rmax, cmin, cmax


def _crop(chw, bbox):
    rmin, rmax, cmin, cmax = bbox
    return chw[:, rmin:rmax, cmin:cmax]


def _square_bbox(bbox, h, w):
    """Expand a bbox to a square (side = max(height, width)), centred and clamped to the image.

    Keeps the object aspect ratio uniform ACROSS tasks so every crop fills its cell identically - a
    tall-narrow object (push, pick-place) otherwise gets a tall axes box that shoves the objMSE xlabel
    down into the row below. Does not change any score: objMSE is over the full-frame silhouette, not
    the crop."""
    rmin, rmax, cmin, cmax = bbox
    side = max(rmax - rmin, cmax - cmin)

    def expand(lo, hi, full):
        pad = side - (hi - lo)
        lo -= pad // 2
        hi += pad - pad // 2
        if lo < 0:
            hi -= lo; lo = 0
        if hi > full:
            lo -= hi - full; hi = full
        return max(0, lo), min(full, hi)

    rmin, rmax = expand(rmin, rmax, h)
    cmin, cmax = expand(cmin, cmax, w)
    return rmin, rmax, cmin, cmax


@torch.no_grad()
def _predict(net, obs, agent_mask, object_mask, seed):
    """Uniform FDM call for all four models: SLAPOIDM ignores object_mask, IMLAMIDM requires it.

    Every model here has object_mask_input=false, so its INPUT mask is the agent mask; the object mask
    is passed through for IM-LAM's FDM only. Seed before the forward so the IDM's future-obs sampling
    (z_t) is identical across models.
    """
    net.future_obs_sampling = getattr(net, "future_obs_sampling", True) and False  # deterministic z_t
    seed_all(seed)
    return net(obs, agent_mask, object_mask=object_mask)[0]


def _silhouette_masked_display(chw, sil_hw):
    """RGB uint8 crop with everything outside the object silhouette blacked out.

    Shows exactly the pixels the object-region MSE is computed over (no distractor-background noise
    competing for the eye), so a model that smears the object is visibly worse, not just numerically."""
    disp = _to_display(chw)                       # (h, w, 3) uint8
    return disp * (sil_hw[..., None] > 0)


def collect_task(task, models, data_path, split, frame_stack, seed, device, target_frames_override=None):
    """Load one task's (single) target frame and run every model's frozen FDM on it. Returns a row dict
    for :func:`render`. ``models = [(label, config_name, checkpoint)]``; crops are formed in render."""
    from gymnasium import spaces
    frames = ([int(x) for x in target_frames_override.split(",")] if target_frames_override
              else DEFAULT_TARGET_FRAMES.get(task))
    if not frames:
        raise SystemExit(f"no default target frames for {task}; pass --target-frames")
    frames = frames[:1]  # single frame (one row); pass --target-frames <one> to choose which
    window_idx = [t - frame_stack for t in frames]

    ds_cfg = _compose("imlam_dmw_stage_1", task, data_path)
    dataset = hydra.utils.instantiate(ds_cfg.dataset, split=split)
    batch = tensordict_collate([dataset[i] for i in window_idx]).to(device)
    obs, agent_mask, object_mask = batch["observation"], batch["mask"], batch["object_mask"]
    gt_next = obs[:, frame_stack]                     # (1, C, H, W)
    object_sils = object_mask[:, frame_stack, 0]      # (1, H, W)

    _, t, c, h, w = obs.shape
    obs_space = spaces.Box(-np.inf, np.inf, (t, c, h, w), np.float32)
    act_space = spaces.Box(-1.0, 1.0, (batch["action"].shape[-1],), np.float32)

    labels, preds, ppms, obj_mse, steps = [], [], [], [], []
    for label, config_name, ckpt in models:
        print(f"[{task}] loading {label} from {ckpt} ...", flush=True)
        cfg = _compose(config_name, task, data_path)
        net = load_frozen_net(cfg, ckpt, obs_space, act_space, device)
        pred = _predict(net, obs, agent_mask, object_mask, seed)       # (1, C, H, W)
        ppm = per_pixel_mse(pred, gt_next)                             # (1, H, W)
        labels.append(label)
        preds.append(pred[0])                                          # (C, H, W)
        ppms.append(ppm[0])                                            # (H, W)
        obj_mse.append(region_error(ppm, object_sils)[0].item())       # scalar
        steps.append(int(os.path.basename(ckpt).replace("step-", "").replace(".ckpt", "")))
    return {"task": task, "frame": frames[0], "gt_next": gt_next[0], "object_sil": object_sils[0],
            "labels": labels, "preds": preds, "ppms": ppms, "obj_mse": obj_mse, "steps": steps}


def render(rows, out_path, margin, mode="rgb", silhouette=True, double=False, err_cmap="Reds", square=True):
    """One figure. Columns = [locator | GT | one per model].

    Single mode (default): one matplotlib row per task (its single target frame). mode='rgb' shows
    silhouette-masked object reconstruction crops; mode='heatmap' shows silhouette-gated per-pixel
    error crops.

    double=True: TWO matplotlib rows per task - RGB reconstructions on top, the corresponding
    per-pixel error heatmaps directly underneath (each model's error sits below its reconstruction).

    Error colours are a sequential white(low) -> dark-red(high) map (``err_cmap``, default "Reds"),
    with the scale SHARED across models within a task (per-task ``vmax``); in double mode a colourbar
    for that scale sits in the error row's GT column."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = rows[0]["labels"]
    n_tasks, n_models = len(rows), len(labels)
    ncols = 2 + n_models          # locator | GT crop | one per model
    rpt = 2 if double else 1       # matplotlib rows per task
    fig, axes = plt.subplots(n_tasks * rpt, ncols,
                             figsize=(ncols * 2.0, n_tasks * rpt * 2.25), squeeze=False)

    for r, row in enumerate(rows):
        sil = row["object_sil"].cpu().numpy()
        bbox = _object_bbox(sil, margin)
        if square:
            bbox = _square_bbox(bbox, sil.shape[0], sil.shape[1])
        rmin, rmax, cmin, cmax = bbox
        sil_crop = sil[rmin:rmax, cmin:cmax]

        # Per-pixel error crop (2D (H,W)), silhouette-gated (0 outside the object -> white). Shared
        # vmax per task so colours are comparable across models within the task.
        def _err_crop(m):
            e = row["ppms"][m][rmin:rmax, cmin:cmax].cpu().numpy()
            return e * sil_crop if silhouette else e
        vmax = max(float(_err_crop(m).max()) for m in range(n_models)) or 1.0

        rgb_r = r * rpt                          # RGB (or single) row index
        err_r = rgb_r + (1 if double else 0)     # error row index (== rgb_r for single-heatmap)
        show_rgb = double or mode == "rgb"
        show_err = double or mode == "heatmap"
        top_r = rgb_r if show_rgb else err_r     # row carrying the locator/GT and the column titles

        # Column 0: full target frame with the object bbox drawn (locator).
        ax = axes[top_r][0]
        ax.imshow(_to_display(row["gt_next"]))
        ax.add_patch(plt.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin, fill=False,
                                    edgecolor="lime", linewidth=1.2))
        ax.set_ylabel(f"{row['task']}", fontsize=10)
        if top_r == 0:
            ax.set_title("target", fontsize=10)

        # Column 1: GT object crop (always RGB, silhouette-masked to match the model columns).
        ax = axes[top_r][1]
        gt_crop = _crop(row["gt_next"], bbox)
        ax.imshow(_silhouette_masked_display(gt_crop, sil_crop) if silhouette
                  else _to_display(gt_crop), interpolation="nearest")
        if top_r == 0:
            ax.set_title("GT (object)", fontsize=10)

        # Model columns: reconstruction (top) and/or error heatmap (bottom).
        err_im = None
        for m in range(n_models):
            if show_rgb:
                ax = axes[rgb_r][2 + m]
                crop = _crop(row["preds"][m], bbox)
                ax.imshow(_silhouette_masked_display(crop, sil_crop) if silhouette
                          else _to_display(crop), interpolation="nearest")
                ax.set_xlabel(f"objMSE={row['obj_mse'][m]:.4f}", fontsize=9, labelpad=4)
                if rgb_r == 0:
                    ax.set_title(labels[m], fontsize=9)
            if show_err:
                ax = axes[err_r][2 + m]
                err_im = ax.imshow(_err_crop(m), cmap=err_cmap, vmin=0.0, vmax=vmax,
                                   interpolation="nearest")
                if not show_rgb:                 # heatmap-only figure: carry labels/scores here
                    ax.set_xlabel(f"objMSE={row['obj_mse'][m]:.4f}", fontsize=9, labelpad=4)
                    if err_r == 0:
                        ax.set_title(labels[m], fontsize=9)

        # double: the error row's locator/GT cells are free - label the row and host the colourbar.
        if double:
            a0 = axes[err_r][0]
            a0.axis("off")  # error row's locator cell is unused (quantity is named on the colourbar)
            host = axes[err_r][1]
            host.axis("off")
            if err_im is not None:
                cb = fig.colorbar(err_im, ax=host, fraction=0.6, pad=0.02)
                cb.ax.tick_params(labelsize=7)
                cb.set_label("per-pixel MSE", fontsize=9)

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=1.6)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="handle-pull-v3")
    p.add_argument("--all", action="store_true",
                   help="One row per task in MODEL_SETS (that has --model-set), instead of a single --task.")
    p.add_argument("--data-path", default="/tmp/slapo_local", help="Local dataset root (or HF repo id).")
    p.add_argument("--split", default="test")
    p.add_argument("--target-frames", default=None,
                    help="Comma-separated global (episode-0) target-frame indices for --task (only the "
                         "first is used). Ignored under --all (each task uses its own default frame).")
    p.add_argument("--frame-stack", type=int, default=3)
    p.add_argument("--crop-margin", type=int, default=10, help="Pixels of context around the object bbox.")
    p.add_argument("--model-set", default="seed1", help="Which checkpoint set (e.g. seed1, seed2) from MODEL_SETS[task].")
    p.add_argument("--no-silhouette", action="store_true", help="Show full crops (with distractor bg) instead of silhouette-masked.")
    p.add_argument("--heatmap", action="store_true", help="Also write a companion per-pixel object-error heatmap figure.")
    p.add_argument("--double", action="store_true",
                   help="Combined two-row panel per task: RGB reconstructions on top, the corresponding "
                        "per-pixel error heatmaps (white->dark red, shared scale per task) underneath.")
    p.add_argument("--err-cmap", default="Reds",
                   help="Sequential matplotlib colormap for the error heatmaps (default: Reds; "
                        "white=low error -> dark red=high). Good alternatives: YlOrRd, OrRd.")
    p.add_argument("--no-square", action="store_true",
                   help="Keep the raw object bbox aspect ratio instead of squaring crops (squaring is "
                        "the default; it keeps a uniform aspect across tasks so the objMSE label spacing "
                        "is consistent).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    tasks = list(MODEL_SETS) if args.all else [args.task]
    rows = []
    for task in tasks:
        try:
            models = MODEL_SETS[task][args.model_set]
        except KeyError:
            if args.all:
                print(f"skip {task}: no MODEL_SETS[{task!r}][{args.model_set!r}]")
                continue
            raise SystemExit(f"no MODEL_SETS[{task!r}][{args.model_set!r}] - add its checkpoints to the script")
        rows.append(collect_task(task, models, args.data_path, args.split, args.frame_stack,
                                 args.seed, device, None if args.all else args.target_frames))
    if not rows:
        raise SystemExit(f"no tasks with model-set {args.model_set!r} in MODEL_SETS")

    tag = "all" if args.all else args.task
    default_name = f"reconstruction_panel_{tag}_{args.model_set}{'_double' if args.double else ''}.png"
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                    "docs", "figures", "reconstruction_panels", default_name)
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if args.double:
        # Combined figure already contains both the reconstructions and their error heatmaps.
        render(rows, out, args.crop_margin, silhouette=not args.no_silhouette, double=True,
               err_cmap=args.err_cmap, square=not args.no_square)
    else:
        render(rows, out, args.crop_margin, mode="rgb", silhouette=not args.no_silhouette,
               err_cmap=args.err_cmap, square=not args.no_square)
        if args.heatmap:
            render(rows, out.replace(".png", "_heatmap.png"), args.crop_margin, mode="heatmap",
                   err_cmap=args.err_cmap, square=not args.no_square)


if __name__ == "__main__":
    main()

"""
Notes: 
1. FG-dual winning object-MSE is almost tautological. The dual loss gives the object its own area-normalized 
term at λ_O=1.0. On a small-object task like handle-pull, the union loss (FG-union, and the IM-LAM checkpoint 
here - both union) dilutes the object: it shares one normalization with the much larger agent mask, so the 
object contributes little gradient. FG-dual removes that dilution and trains the object at full strength. 
So FG-dual is optimizing exactly the quantity we're plotting, harder than anyone else. It winning object-MSE 
is close to definitional - not evidence of a better world model.

2. Object reconstruction and policy quality are different objectives that can pull apart. The whole 
three-stage design hinges on z_t being a clean embodiment action. Stage 1 freezes the encoder; 
Stages 2/3 build the policy on z_t. If Stage-1's pressure to reconstruct the object pushes 
*object-appearance* information into z_t (the FDM's conditioning signal), then z_t stops being a clean agent 
action -> the downstream policy inherits a worse action space -> worse NSR. FG-dual's aggressive object loss 
is precisely the pressure that risks this contamination. It can buy object-reconstruction at the cost of z_t 
cleanliness - which surfaces as worst NSR. That's the trade the number is showing you.
"""