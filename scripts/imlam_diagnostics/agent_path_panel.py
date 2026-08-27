"""Agent-path corruption panel (Phase 9): the qualitative companion to R_no_transition / R_shuffled.

Shows the object prediction visibly degrade as the agent pathway feeding the object branch is corrupted
at eval time (no retraining), per proposal S7.5 / S7.4.3. IM-LAM only (and the direct-z ablation via
--config-name) - a within-model mechanism demo, since only IMLAMIDM has an agent_ctx_mode to corrupt.

Layout: one ROW per task (its max-per-step-object-motion frame, the same principled frame the
reconstruction panel uses); columns = [target locator | GT object crop | normal | no_transition |
shuffled], each model column an object crop annotated with its object-region MSE. --heatmap adds a
companion per-pixel object-error figure (shared error scale per row) so the corrupted modes visibly
light up relative to normal.

shuffled needs B >= 2 (a single-sample batch is a no-op, S7.4.3): each task's forward is run on a batch
of ALL its max-motion frames so the shuffle has a real other-sample source, but only the first frame's
row is displayed. The shuffle source is therefore intra-task (another max-motion frame of the same
task), noted in the caption.

Single-task now (quick analysis while experiments finish), all-task later (--all): the full figure is
one row per task once every IM-LAM checkpoint exists.

Env: conda_env/bin/python. Default reads --data-path (local staged dir or HF repo).

Usage:
    conda_env/bin/python scripts/imlam_diagnostics/agent_path_panel.py --task door-open-v3 \
        --data-path /tmp/slapo_local --heatmap
    conda_env/bin/python scripts/imlam_diagnostics/agent_path_panel.py --all --data-path /tmp/slapo_local
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # compare_reconstruction
sys.path.insert(0, os.path.dirname(__file__))  # reconstruction_panel

import hydra  # noqa: E402
from gymnasium import spaces  # noqa: E402

from ifo.common.utils.utility import tensordict_collate  # noqa: E402
from compare_reconstruction import per_pixel_mse, region_error, seed_all, _to_display  # noqa: E402
from scripts.imlam_diagnostics.run_diagnostics import load_frozen_net  # noqa: E402
from reconstruction_panel import _compose, _object_bbox, _crop, DEFAULT_TARGET_FRAMES  # noqa: E402

MODES = ["normal", "no_transition", "shuffled"]

# task -> (config_name, IM-LAM union checkpoint). config_name selects IMLAMIDM; direct-z via
# --config-name. Populated with the checkpoints that exist so far; extend as experiments finish.
TASK_CHECKPOINTS = {
    "door-open-v3":   ("imlam_dmw_stage_1", "checkpoints/im-lam_door-open_union_seed2-1/step-000031248.ckpt"),
    "handle-pull-v3": ("imlam_dmw_stage_1", "checkpoints/im-lam_handle-pull_union_seed1-retry-1/step-000031248.ckpt"),
    "push-v3":        ("imlam_dmw_stage_1", "checkpoints/im-lam_push_union_seed1-1/step-000031248.ckpt"),
    "sweep-into-v3":  ("imlam_dmw_stage_1", "checkpoints/im-lam_sweep-into_union_seed2-1/step-000031248.ckpt"),
    # TODO: pick-place-v3 / peg-insert-side-v3: no IM-LAM checkpoint yet.
}


@torch.no_grad()
def _predict_mode(net, obs, agent_mask, object_mask, mode, seed):
    """FDM forward under one agent_ctx_mode. future_obs_sampling off so z_t is deterministic; seeded so
    the shuffle permutation is reproducible."""
    net.future_obs_sampling = getattr(net, "future_obs_sampling", True) and False
    seed_all(seed)
    return net(obs, agent_mask, object_mask=object_mask, agent_ctx_mode=mode)[0]


def collect_task_row(task, config_name, checkpoint, data_path, split, frame_stack, seed, device):
    """Run the three modes for one task and return the (first) frame's row data. The forward batch is
    all of the task's max-motion frames (>=2) so shuffled is meaningful; only frame 0 is displayed."""
    frames = DEFAULT_TARGET_FRAMES.get(task)
    if not frames:
        raise SystemExit(f"no default max-motion frames for {task}")
    window_idx = [f - frame_stack for f in frames]

    cfg = _compose(config_name, task, data_path)
    dataset = hydra.utils.instantiate(cfg.dataset, split=split)
    batch = tensordict_collate([dataset[i] for i in window_idx]).to(device)
    obs, agent_mask, object_mask = batch["observation"], batch["mask"], batch["object_mask"]
    gt_next = obs[:, frame_stack]                    # (B, C, H, W)
    object_sils = object_mask[:, frame_stack, 0]     # (B, H, W)

    _, t, c, h, w = obs.shape
    obs_space = spaces.Box(-np.inf, np.inf, (t, c, h, w), np.float32)
    act_space = spaces.Box(-1.0, 1.0, (batch["action"].shape[-1],), np.float32)
    print(f"[{task}] loading {checkpoint} ...", flush=True)
    net = load_frozen_net(cfg, checkpoint, obs_space, act_space, device)

    preds, ppms, objmse = {}, {}, {}
    for mode in MODES:
        pred = _predict_mode(net, obs, agent_mask, object_mask, mode, seed)  # (B, C, H, W)
        preds[mode] = pred[0]
        ppms[mode] = per_pixel_mse(pred, gt_next)[0]                          # (H, W)
        objmse[mode] = region_error(per_pixel_mse(pred, gt_next), object_sils)[0].item()
    return {"task": task, "frame": frames[0], "gt_next": gt_next[0], "object_sil": object_sils[0],
            "preds": preds, "ppms": ppms, "objmse": objmse}


def render(rows, out_path, margin, mode="rgb"):
    """One row per task. mode='rgb': object crops per agent_ctx_mode. mode='heatmap': silhouette-gated
    per-pixel object-error crops (inferno, shared vmax per row)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 2 + len(MODES)  # locator | GT crop | normal | no_transition | shuffled
    fig, axes = plt.subplots(len(rows), ncols, figsize=(ncols * 2.0, len(rows) * 2.25), squeeze=False)

    for r, row in enumerate(rows):
        sil = row["object_sil"].cpu().numpy()
        bbox = _object_bbox(sil, margin)
        sil_crop = sil[bbox[0]:bbox[1], bbox[2]:bbox[3]]

        ax = axes[r][0]
        ax.imshow(_to_display(row["gt_next"]))
        rmin, rmax, cmin, cmax = bbox
        ax.add_patch(plt.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin, fill=False,
                                    edgecolor="lime", linewidth=1.2))
        ax.set_ylabel(f"{row['task']}\nframe {row['frame']}", fontsize=8)
        if r == 0:
            ax.set_title("target (locator)", fontsize=9)

        ax = axes[r][1]
        ax.imshow(_to_display(_crop(row["gt_next"], bbox)), interpolation="nearest")
        if r == 0:
            ax.set_title("GT (object)", fontsize=9)

        vmax = (max(float((row["ppms"][m][bbox[0]:bbox[1], bbox[2]:bbox[3]].cpu().numpy() * sil_crop).max())
                    for m in MODES) if mode == "heatmap" else None)
        for j, m in enumerate(MODES):
            ax = axes[r][2 + j]
            if mode == "heatmap":
                err = row["ppms"][m][bbox[0]:bbox[1], bbox[2]:bbox[3]].cpu().numpy() * sil_crop
                ax.imshow(err, cmap="inferno", vmin=0.0, vmax=vmax or 1.0, interpolation="nearest")
            else:
                ax.imshow(_to_display(_crop(row["preds"][m], bbox)), interpolation="nearest")
            ratio = row["objmse"][m] / (row["objmse"]["normal"] + 1e-8)
            label = f"objMSE={row['objmse'][m]:.4f}" + ("" if m == "normal" else f"\n(x{ratio:.2f} vs normal)")
            ax.set_xlabel(label, fontsize=7)
            if r == 0:
                ax.set_title(m, fontsize=9)

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    kind = "per-pixel object error (inferno, shared/row)" if mode == "heatmap" else "object crops"
    fig.suptitle("Agent-path corruption (IM-LAM): " + kind + "  |  single-frame illustration of "
                 "R_no_transition / R_shuffled; shuffle source is intra-task", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="door-open-v3", help="Single task to render (one row).")
    p.add_argument("--all", action="store_true", help="Render every task in TASK_CHECKPOINTS as rows.")
    p.add_argument("--checkpoint", default=None, help="Override the IM-LAM checkpoint for --task.")
    p.add_argument("--config-name", default=None, help="Override config (e.g. imlam_direct_z_dmw_stage_1).")
    p.add_argument("--data-path", default="/tmp/slapo_local", help="Local dataset root or HF repo id.")
    p.add_argument("--split", default="test")
    p.add_argument("--frame-stack", type=int, default=3)
    p.add_argument("--crop-margin", type=int, default=10)
    p.add_argument("--heatmap", action="store_true", help="Also write a per-pixel object-error companion figure.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    tasks = list(TASK_CHECKPOINTS) if args.all else [args.task]
    rows, config_names = [], set()
    for task in tasks:
        default_cfg, default_ckpt = TASK_CHECKPOINTS.get(task, (None, None))
        config_name = args.config_name or default_cfg
        checkpoint = args.checkpoint if (not args.all and args.checkpoint) else default_ckpt
        if not (config_name and checkpoint):
            raise SystemExit(f"no config/checkpoint for {task}; pass --checkpoint (and --config-name)")
        config_names.add(config_name)
        rows.append(collect_task_row(task, config_name, checkpoint, args.data_path, args.split,
                                     args.frame_stack, args.seed, device))

    # Model tag for the filename: 'imlam-direct-z' if any row used the direct-z config, else 'imlam'.
    model_tag = "imlam-direct-z" if any("direct_z" in c for c in config_names) else "imlam"
    tag = "all" if args.all else args.task
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                    "scratchpad", "agent_path_panels", f"agent_path_panel_{model_tag}_{tag}.png")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    render(rows, out, args.crop_margin, mode="rgb")
    if args.heatmap:
        render(rows, out.replace(".png", "_heatmap.png"), args.crop_margin, mode="heatmap")


if __name__ == "__main__":
    main()
