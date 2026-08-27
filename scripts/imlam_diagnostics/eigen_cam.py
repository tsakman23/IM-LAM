"""Eigen-CAM on the IDM (Phase 9): is z_t agent/embodiment-centered, or background-centered?

Sanity check following MaskLAM's saliency analysis (proposal S7.5): the IDM encoder (SLAPOIDM.encoder,
an ImpalaCNNBackbone) produces z_t; if z_t is a clean embodiment action its salient input regions should
be the AGENT, not the DAVIS distractor background. Eigen-CAM (Muhammad & Yeasin 2020) gives a
gradient-free, class-free saliency: project the last conv feature map onto its first principal component.

We hook the last spatial layer (net.encoder.backbone[2] -> 192x16x16 for DMW), run the net forward (which
runs the IDM encoder internally), take the captured activations A in (C, HW), compute the first right
singular vector v1, and the saliency = |A @ v1| reshaped to 16x16, upsampled and overlaid on the current
observation. Quantitative complement: the fraction of saliency mass inside the agent mask (high =
agent-centered) - the actual sanity-check number, the heatmap is the illustration.

Caveat: the IDM input is a 24-channel current+future+mask
stack, so the single spatial saliency is over a fused spatiotemporal input, not one frame; and it probes
the ACTION ENCODER, not the FDM/interaction mechanism that is the IM-LAM story. It is a z_t sanity check,
not evidence for the causal prior.

IM-LAM by default (direct-z via --config-name); the IDM encoder architecture is shared across model
types but the trained weights differ, so run it on the checkpoint of interest.

Usage:
    conda_env/bin/python scripts/imlam_diagnostics/eigen_cam.py --task door-open-v3 --data-path /tmp/slapo_local
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(__file__))

import hydra  # noqa: E402
from gymnasium import spaces  # noqa: E402

from ifo.common.utils.utility import tensordict_collate  # noqa: E402
from compare_reconstruction import seed_all, _to_display  # noqa: E402
from scripts.imlam_diagnostics.run_diagnostics import load_frozen_net  # noqa: E402
from reconstruction_panel import _compose, DEFAULT_TARGET_FRAMES  # noqa: E402
from agent_path_panel import TASK_CHECKPOINTS  # noqa: E402
from attention_panel import _overlay  # noqa: E402

BOTTLENECK = 16


def eigen_cam(activation):
    """Eigen-CAM saliency for one sample: |A @ v1| over the (H*W) positions, reshaped to (H, W).

    A is the (C, H, W) last-conv activation; flatten to (H*W, C), take the first right singular vector
    v1 (principal component), project. abs() resolves the SVD sign ambiguity."""
    c, h, w = activation.shape
    a = activation.reshape(c, h * w).T                      # (HW, C)
    _, _, vh = torch.linalg.svd(a, full_matrices=False)     # a = U S Vh; vh[0] = first PC (C,)
    sal = (a @ vh[0]).abs().reshape(h, w)                   # (H, W)
    return sal


# task -> [(label, config_name, checkpoint)] for the multi-model comparison (--compare). The IDM
# encoder is architecturally identical across model types, so the same hook works for all; only the
# trained weights differ. door-open has no plain MaskLAM checkpoint, so it's FG-union/FG-dual/IM-LAM.
COMPARE_MODELS = {
    "door-open-v3": [
        ("FG-union", "foreground_masklam_dmw_stage_1",      "checkpoints/fg_masklam_door-open_seed3-1/step-000031248.ckpt"),
        ("FG-dual",  "foreground_masklam_dual_dmw_stage_1", "checkpoints/dual_masklam_door-open_seed3-1/step-000031248.ckpt"),
        ("IM-LAM",   "imlam_dmw_stage_1",                   "checkpoints/im-lam_door-open_union_seed3-1/step-000031248.ckpt"),
    ],
}


@torch.no_grad()
def _cam_for_model(net, obs, agent_mask, object_mask, seed):
    """Hook the IDM encoder's last conv block, run the net, return the Eigen-CAM (16x16) for frame 0."""
    captured = {}
    handle = net.encoder.backbone[2].register_forward_hook(  # last ImpalaEncoderBlock -> (B, 192, 16, 16)
        lambda m, i, o: captured.__setitem__("act", o.detach()))
    net.future_obs_sampling = getattr(net, "future_obs_sampling", True) and False
    seed_all(seed)
    net(obs, agent_mask, object_mask=object_mask)
    handle.remove()
    return eigen_cam(captured["act"][0]).cpu().numpy()


def collect_task(task, models, data_path, split, frame_stack, seed, device):
    """Load the task's max-motion batch once, compute the IDM Eigen-CAM for each model.

    models = [(label, config_name, checkpoint)]. The dataset is loaded from the first model's config
    (all share the object-mask repo/dir); the uniform net(obs, agent_mask, object_mask=...) call works
    for SLAPOIDM (ignores object_mask) and IMLAMIDM (uses it)."""
    frames = DEFAULT_TARGET_FRAMES.get(task)
    if not frames:
        raise SystemExit(f"no default max-motion frames for {task}")
    dataset = hydra.utils.instantiate(_compose(models[0][1], task, data_path).dataset, split=split)
    batch = tensordict_collate([dataset[f - frame_stack] for f in frames]).to(device)
    obs, agent_mask, object_mask = batch["observation"], batch["mask"], batch["object_mask"]
    _, t, c, h, w = obs.shape
    obs_space = spaces.Box(-np.inf, np.inf, (t, c, h, w), np.float32)
    act_space = spaces.Box(-1.0, 1.0, (batch["action"].shape[-1],), np.float32)

    cams = []
    for label, config_name, ckpt in models:
        print(f"[{task}] {label}: loading {ckpt} ...", flush=True)
        net = load_frozen_net(_compose(config_name, task, data_path), ckpt, obs_space, act_space, device)
        cams.append((label, _cam_for_model(net, obs, agent_mask, object_mask, seed)))
    return {"task": task, "frame": frames[0], "obs": obs[0, frame_stack - 1],
            "agent": agent_mask[0, frame_stack - 1, 0], "cams": cams}


def render(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_models = max(len(r["cams"]) for r in rows)
    ncols = 1 + n_models  # observation | one Eigen-CAM per model
    fig, axes = plt.subplots(len(rows), ncols, figsize=(ncols * 2.3, len(rows) * 2.3), squeeze=False)
    for r, row in enumerate(rows):
        ax = axes[r][0]
        ax.imshow(_to_display(row["obs"]))
        ax.contour(row["agent"].cpu().numpy(), levels=[0.5], colors="lime", linewidths=0.8)
        ax.set_ylabel(f"{row['task']}\nframe {row['frame']}", fontsize=8)
        if r == 0:
            ax.set_title("observation (agent=lime)", fontsize=9)
        for j, (label, sal) in enumerate(row["cams"]):
            _overlay(axes[r][1 + j], sal, row["obs"], row["agent"])
            if r == 0:
                axes[r][1 + j].set_title(label, fontsize=9)

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("IDM Eigen-CAM", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="door-open-v3")
    p.add_argument("--all", action="store_true")
    p.add_argument("--compare", action="store_true",
                   help="Compare the IDM Eigen-CAM across models (COMPARE_MODELS[task]) instead of one model.")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--config-name", default=None)
    p.add_argument("--data-path", default="/tmp/slapo_local")
    p.add_argument("--split", default="test")
    p.add_argument("--frame-stack", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    if args.compare:
        tasks = list(COMPARE_MODELS) if args.all else [args.task]
        rows = []
        for task in tasks:
            if task not in COMPARE_MODELS:
                raise SystemExit(f"no COMPARE_MODELS[{task}] - add its per-model checkpoints")
            rows.append(collect_task(task, COMPARE_MODELS[task], args.data_path, args.split,
                                     args.frame_stack, args.seed, device))
        model_tag = "compare"
    else:
        tasks = list(TASK_CHECKPOINTS) if args.all else [args.task]
        rows, config_names = [], set()
        for task in tasks:
            default_cfg, default_ckpt = TASK_CHECKPOINTS.get(task, (None, None))
            config_name = args.config_name or default_cfg
            checkpoint = args.checkpoint if (not args.all and args.checkpoint) else default_ckpt
            if not (config_name and checkpoint):
                raise SystemExit(f"no config/checkpoint for {task}; pass --checkpoint (and --config-name)")
            config_names.add(config_name)
            label = "IM-LAM-direct-z" if "direct_z" in config_name else "IM-LAM"
            rows.append(collect_task(task, [(label, config_name, checkpoint)], args.data_path, args.split,
                                     args.frame_stack, args.seed, device))
        model_tag = "imlam-direct-z" if any("direct_z" in c for c in config_names) else "imlam"
    tag = "all" if args.all else args.task
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                    "scratchpad", "eigen_cam", f"eigen_cam_{model_tag}_{tag}.png")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    render(rows, out)


if __name__ == "__main__":
    main()
