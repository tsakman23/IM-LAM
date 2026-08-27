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
    CUDA_VISIBLE_DEVICES=0 conda_env/bin/python scripts/imlam_diagnostics/reconstruction_panel.py \
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
# truncated - e.g. MaskLAM's 15000 is its best; val loss rose after). The -2/-3 sibling dirs are
# Stages 2/3, not later Stage-1 steps.
MODEL_SETS = {
    "handle-pull-v3": {
        "seed1": [
            ("MaskLAM",  "slapo_dmw_stage_1",                   "checkpoints/masklam_handle-pull-v3_seed1-1/step-000015000.ckpt"),
            ("FG-union", "foreground_masklam_dmw_stage_1",      "checkpoints/fg_masklam_handle-pull_seed1-1/step-000031248.ckpt"),
            ("FG-dual",  "foreground_masklam_dual_dmw_stage_1", "checkpoints/dual_masklam_handle-pull_seed1-1/step-000031248.ckpt"),
            ("IM-LAM",   "imlam_dmw_stage_1",                   "checkpoints/im-lam_handle-pull_union_seed1-retry-1/step-000031248.ckpt"),
        ],
        # seed2 checkpoints are at mixed/earlier steps (15k/15k/15k/25k); no seed2 Stage-1 reached 31248.
        # Whether these are val-best or truncated is unverified - read seed2 as a robustness check, not a
        # matched-training comparison.
        "seed2": [
            ("MaskLAM",  "slapo_dmw_stage_1",                   "checkpoints/masklam_handle-pull-v3_seed2-1/step-000015000.ckpt"),
            ("FG-union", "foreground_masklam_dmw_stage_1",      "checkpoints/fg_masklam_handle-pull_seed2-1/step-000015000.ckpt"),
            ("FG-dual",  "foreground_masklam_dual_dmw_stage_1", "checkpoints/dual_masklam_handle-pull_seed2-1/step-000015000.ckpt"),
            ("IM-LAM",   "imlam_dmw_stage_1",                   "checkpoints/im-lam_handle-pull_union_seed2-1/step-000025000.ckpt"),
        ],
    },
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


def render(out_path, task, target_frames, gt_next, preds, ppms, object_sils, obj_mse, labels,
           seed_note, margin, mode="rgb", silhouette=True):
    """One figure. mode='rgb': silhouette-masked object crops per model. mode='heatmap':
    silhouette-gated per-pixel error crops (inferno), shared vmax per frame across models."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_frames, n_models = len(target_frames), len(labels)
    ncols = 2 + n_models  # locator | GT crop | one per model
    fig, axes = plt.subplots(n_frames, ncols, figsize=(ncols * 2.0, n_frames * 2.25), squeeze=False)

    for r in range(n_frames):
        sil = object_sils[r].cpu().numpy()
        bbox = _object_bbox(sil, margin)
        sil_crop = sil[bbox[0]:bbox[1], bbox[2]:bbox[3]]

        # Column 0: full target frame, object bbox drawn (locator).
        ax = axes[r][0]
        ax.imshow(_to_display(gt_next[r]))
        rmin, rmax, cmin, cmax = bbox
        ax.add_patch(plt.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin, fill=False,
                                    edgecolor="lime", linewidth=1.2))
        ax.set_ylabel(f"target frame {target_frames[r]}", fontsize=9)
        if r == 0:
            ax.set_title("target (locator)", fontsize=9)

        # Column 1: GT object crop (always RGB, silhouette-masked to match the model columns).
        ax = axes[r][1]
        gt_crop = _crop(gt_next[r], bbox)
        ax.imshow(_silhouette_masked_display(gt_crop, sil_crop) if silhouette
                  else _to_display(gt_crop), interpolation="nearest")
        if r == 0:
            ax.set_title("GT (object)", fontsize=9)

        # Per-pixel error crop (2D (H,W)), silhouette-gated. Shared vmax/frame for comparable colours.
        def _err_crop(m):
            e = ppms[m][r][bbox[0]:bbox[1], bbox[2]:bbox[3]].cpu().numpy()
            return e * sil_crop
        vmax = max(float(_err_crop(m).max()) for m in range(n_models)) if mode == "heatmap" else None

        for m in range(n_models):
            ax = axes[r][2 + m]
            if mode == "heatmap":
                ax.imshow(_err_crop(m), cmap="inferno", vmin=0.0, vmax=vmax or 1.0, interpolation="nearest")
            else:
                crop = _crop(preds[m][r], bbox)
                ax.imshow(_silhouette_masked_display(crop, sil_crop) if silhouette
                          else _to_display(crop), interpolation="nearest")
            ax.set_xlabel(f"objMSE={obj_mse[m][r]:.4f}", fontsize=8)
            if r == 0:
                ax.set_title(labels[m], fontsize=9)

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    mean_line = "   ".join(f"{labels[m]}={float(obj_mse[m].mean()):.4f}" for m in range(n_models))
    kind = "per-pixel object error (inferno, shared scale/frame)" if mode == "heatmap" \
        else ("silhouette-masked object crops" if silhouette else "object crops")
    fig.suptitle(f"{task}  -  FDM {kind} at max-motion frames  |  mean object-region MSE:  "
                 f"{mean_line}\n{seed_note}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="handle-pull-v3")
    p.add_argument("--data-path", default="/tmp/slapo_local", help="Local dataset root (or HF repo id).")
    p.add_argument("--split", default="test")
    p.add_argument("--target-frames", default=None,
                    help="Comma-separated global (episode-0) target-frame indices. Default: precomputed "
                         "max-motion frames for the task.")
    p.add_argument("--frame-stack", type=int, default=3)
    p.add_argument("--crop-margin", type=int, default=10, help="Pixels of context around the object bbox.")
    p.add_argument("--model-set", default="seed1", help="Which checkpoint set (e.g. seed1, seed2) from MODEL_SETS[task].")
    p.add_argument("--no-silhouette", action="store_true", help="Show full crops (with distractor bg) instead of silhouette-masked.")
    p.add_argument("--heatmap", action="store_true", help="Also write a companion per-pixel object-error heatmap figure.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    try:
        models = MODEL_SETS[args.task][args.model_set]
    except KeyError:
        raise SystemExit(f"no MODEL_SETS[{args.task!r}][{args.model_set!r}] - add its checkpoints to the script")

    frames = ([int(x) for x in args.target_frames.split(",")] if args.target_frames
              else DEFAULT_TARGET_FRAMES.get(args.task))
    if not frames:
        raise SystemExit(f"no default target frames for {args.task}; pass --target-frames")
    window_idx = [t - args.frame_stack for t in frames]  # dataset[i] target is at within-window frame_stack

    # Load the object-mask dataset once (from the IM-LAM config) and pull the target windows as a batch.
    ds_cfg = _compose("imlam_dmw_stage_1", args.task, args.data_path)
    dataset = hydra.utils.instantiate(ds_cfg.dataset, split=args.split)
    batch = tensordict_collate([dataset[i] for i in window_idx]).to(device)
    obs = batch["observation"]
    agent_mask = batch["mask"]
    object_mask = batch["object_mask"]
    gt_next = obs[:, args.frame_stack]                     # (B, C, H, W)
    object_sils = object_mask[:, args.frame_stack, 0]      # (B, H, W)

    _, t, c, h, w = obs.shape
    from gymnasium import spaces
    obs_space = spaces.Box(-np.inf, np.inf, (t, c, h, w), np.float32)
    act_space = spaces.Box(-1.0, 1.0, (batch["action"].shape[-1],), np.float32)

    labels, preds, ppms, obj_mse, steps = [], [], [], [], []
    for label, config_name, ckpt in models:
        print(f"loading {label} from {ckpt} ...", flush=True)
        cfg = _compose(config_name, args.task, args.data_path)
        net = load_frozen_net(cfg, ckpt, obs_space, act_space, device)
        pred = _predict(net, obs, agent_mask, object_mask, args.seed)     # (B, C, H, W)
        ppm = per_pixel_mse(pred, gt_next)                                # (B, H, W)
        labels.append(label)
        preds.append(pred)
        ppms.append(ppm)
        obj_mse.append(region_error(ppm, object_sils))                    # (B,)
        steps.append(int(os.path.basename(ckpt).replace("step-", "").replace(".ckpt", "")))

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                    "scratchpad", f"reconstruction_panel_{args.task}_{args.model_set}.png")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    seed_note = (f"{args.model_set}, frozen Stage-1.  steps: "
                 + ", ".join(f"{lab}={st}" for lab, st in zip(labels, steps)))
    render(out, args.task, frames, gt_next, preds, ppms, object_sils, obj_mse, labels, seed_note,
           args.crop_margin, mode="rgb", silhouette=not args.no_silhouette)
    if args.heatmap:
        heat_out = out.replace(".png", "_heatmap.png")
        render(heat_out, args.task, frames, gt_next, preds, ppms, object_sils, obj_mse, labels,
               seed_note, args.crop_margin, mode="heatmap")


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