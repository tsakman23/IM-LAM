"""Extraction attention footprint (Phase 9, Option A): where do the entity read-outs actually read?

The object-dynamics probe's SeparationRatio is non-discriminative (structurally ~1 on coupled tasks),
so it says nothing usable about A_t / O_t. This figure asks a different, non-circular question - not
"is the read-out pure?" but "WHERE does it gather content?" - by visualizing the mask-biased extraction
attention of MSA_A / MSA_O.

For each read-out we show a triptych over the 16x16 bottleneck:
  1. the pooled occupancy mask W (from pool_mask_occupancy) - the *bias* the read-out is handed;
  2. the marginal extraction attention - where the read-out actually attends (averaged over heads and
     query tokens; the beta*W_j bias enters each key uniformly across queries, so the query-marginal is
     the honest thing to compare against W);
  3. their difference (attention - mask, each max-normalized to [0,1]) - the *learned, content-driven*
     deviation from the bias. This is the point of the figure: panel 1 is what the mask hands you (partly
     by construction), panel 3 is what the extraction learned on top (does attention sharpen onto the
     contact point / spread toward motion?).

IM-LAM only (the extraction MSAs live in the interaction FDM; a monolithic FG-MaskLAM has none).

Usage:
    conda_env/bin/python scripts/imlam_diagnostics/extraction_footprint.py --task door-open-v3 \
        --data-path /tmp/slapo_local
"""
import argparse
import math
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

from ifo.common.nets.interaction import pool_mask_occupancy  # noqa: E402
from ifo.common.utils.utility import tensordict_collate  # noqa: E402
from compare_reconstruction import seed_all, _to_display  # noqa: E402
from scripts.imlam_diagnostics.run_diagnostics import load_frozen_net  # noqa: E402
from reconstruction_panel import _compose, DEFAULT_TARGET_FRAMES  # noqa: E402
from agent_path_panel import TASK_CHECKPOINTS  # noqa: E402


def footprint_from_attn(attn, side, query_weight=None):
    """Extraction attention as a (side, side) key-map, from one sample's ``(heads, Nq, Nk)`` weights.

    Averages over heads, then reduces over query tokens:
      - ``query_weight=None`` (uniform): the plain query-marginal - "where the average output token
        attends." The ``beta*W_j`` bias enters each key identically across queries, so this is directly
        comparable to the pooled mask, but it mixes entity-region output tokens with background ones.
      - ``query_weight`` an occupancy ``(Nq,)``: weights the query-average by that occupancy - "where the
        ENTITY-region read-out attends," i.e. the tokens the write-back gate actually keeps.
    Row-major reshape matches ``pool_mask_occupancy``'s flatten order."""
    per_query = attn.mean(0)                       # (Nq, Nk), mean over heads
    if query_weight is None:
        key_attention = per_query.mean(0)          # uniform over queries -> (Nk,)
    else:
        w = query_weight / (query_weight.sum() + 1e-8)
        key_attention = (per_query * w.unsqueeze(-1)).sum(0)  # occupancy-weighted over queries -> (Nk,)
    return key_attention.reshape(side, side)


def _norm01(x):
    return x / (float(np.abs(x).max()) + 1e-8)


def _overlay(ax, map_hw, base_chw, contour_hw, base_brightness=0.5, max_alpha=0.9):
    """Dimmed grayscale observation with an inferno heatmap overlay: ``map_hw`` is max-normalized and
    nearest-upsampled, then drawn with a per-pixel alpha driven by its value (high = opaque heatmap,
    low = shows the gray base), plus the entity-mask contour. Kept local so the committed diagnostic is
    self-contained (no dependency on other panel scripts)."""
    h, w = base_chw.shape[-2:]
    a = torch.from_numpy(map_hw)[None, None]
    up = F.interpolate(a, size=(h, w), mode="nearest")[0, 0].numpy()
    up = up / (up.max() + 1e-8)
    gray = _to_display(base_chw).astype(float).mean(-1) / 255.0 * base_brightness  # dim grayscale (H, W)
    ax.imshow(gray, cmap="gray", vmin=0.0, vmax=1.0)
    ax.imshow(up, cmap="inferno", vmin=0.0, vmax=1.0, alpha=np.clip(up ** 0.7, 0.0, 1.0) * max_alpha)
    ax.contour(contour_hw.cpu().numpy(), levels=[0.5], colors="lime", linewidths=0.8)


@torch.no_grad()
def collect_task(task, config_name, checkpoint, data_path, split, frame_stack, seed, device,
                 query_weight="occupancy"):
    """For the task's first max-motion frame: the pooled masks, extraction attention footprints, and the
    obs/contours needed to render the agent and object triptychs."""
    frames = DEFAULT_TARGET_FRAMES.get(task)
    if not frames:
        raise SystemExit(f"no default max-motion frames for {task}")

    cfg = _compose(config_name, task, data_path)
    dataset = hydra.utils.instantiate(cfg.dataset, split=split)
    batch = tensordict_collate([dataset[frames[0] - frame_stack]]).to(device)
    obs, agent_mask, object_mask = batch["observation"], batch["mask"], batch["object_mask"]

    _, t, c, h, w = obs.shape
    obs_space = spaces.Box(-np.inf, np.inf, (t, c, h, w), np.float32)
    act_space = spaces.Box(-1.0, 1.0, (batch["action"].shape[-1],), np.float32)
    print(f"[{task}] loading {checkpoint} ...", flush=True)
    net = load_frozen_net(cfg, checkpoint, obs_space, act_space, device)
    if not hasattr(net, "extract_entities"):
        raise SystemExit(f"{checkpoint} is not an IM-LAM checkpoint (no extraction MSAs to visualize).")

    net.future_obs_sampling = getattr(net, "future_obs_sampling", True) and False
    seed_all(seed)
    _, _, attn_a, attn_o = net.extract_entities(obs, agent_mask, object_mask, return_attn=True)

    side = int(math.isqrt(attn_a.shape[-1]))
    w_agent_flat = pool_mask_occupancy(agent_mask[:, frame_stack - 1], side)[0]   # (Nq,)
    w_object_flat = pool_mask_occupancy(object_mask[:, frame_stack - 1], side)[0]
    qw_a = w_agent_flat if query_weight == "occupancy" else None                  # weight queries by own occupancy
    qw_o = w_object_flat if query_weight == "occupancy" else None
    w_agent, w_object = w_agent_flat.reshape(side, side), w_object_flat.reshape(side, side)
    fp_a = footprint_from_attn(attn_a[0], side, query_weight=qw_a)
    fp_o = footprint_from_attn(attn_o[0], side, query_weight=qw_o)

    return {
        "task": task, "frame": frames[0],
        "obs": obs[0, frame_stack - 1],
        "agent_contour": agent_mask[0, frame_stack - 1, 0],
        "object_contour": object_mask[0, frame_stack - 1, 0],
        # per read-out: (label, pooled mask, attention footprint, contour)
        "readouts": [
            ("agent read-out (MSA_A)", w_agent.cpu().numpy(), fp_a.cpu().numpy(), agent_mask[0, frame_stack - 1, 0]),
            ("object read-out (MSA_O)", w_object.cpu().numpy(), fp_o.cpu().numpy(), object_mask[0, frame_stack - 1, 0]),
        ],
    }


def _diff_panel(ax, diff_hw, base_chw, contour_hw):
    """Signed attention-minus-mask deviation: diverging heatmap (upsampled, nearest) + entity contour."""
    h, w = base_chw.shape[-2:]
    up = F.interpolate(torch.from_numpy(diff_hw)[None, None].float(), size=(h, w), mode="nearest")[0, 0].numpy()
    m = float(np.abs(up).max()) + 1e-8
    ax.imshow(up, cmap="coolwarm", vmin=-m, vmax=m)
    ax.contour(contour_hw.cpu().numpy(), levels=[0.5], colors="lime", linewidths=0.8)


def render(rows, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # One block of 2 rows (agent, object) per task; 4 columns: obs | pooled mask | attention | diff.
    n_readouts = 2
    total_rows = len(rows) * n_readouts
    fig, axes = plt.subplots(total_rows, 4, figsize=(4 * 2.3, total_rows * 2.3), squeeze=False)
    col_titles = ["observation", "pooled mask $\\widehat{W}$", "extraction attention", "attention - mask (learned)"]

    for t, row in enumerate(rows):
        for j, (label, mask_hw, fp_hw, contour) in enumerate(row["readouts"]):
            r = t * n_readouts + j
            axes[r][0].imshow(_to_display(row["obs"]))
            axes[r][0].contour(contour.cpu().numpy(), levels=[0.5], colors="lime", linewidths=0.8)
            axes[r][0].set_ylabel(f"{row['task']}\nframe {row['frame']}\n{label}", fontsize=7)
            _overlay(axes[r][1], _norm01(mask_hw), row["obs"], contour)
            _overlay(axes[r][2], _norm01(fp_hw), row["obs"], contour)
            _diff_panel(axes[r][3], _norm01(fp_hw) - _norm01(mask_hw), row["obs"], contour)
            if r == 0:
                for c, title in enumerate(col_titles):
                    axes[r][c].set_title(title, fontsize=9)

    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("IM-LAM extraction attention footprint (lime = entity mask; red/blue = attention over/under the mask)",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="door-open-v3")
    p.add_argument("--all", action="store_true", help="Render every task in TASK_CHECKPOINTS.")
    p.add_argument("--checkpoint", default=None, help="Override the IM-LAM checkpoint for --task.")
    p.add_argument("--config-name", default=None, help="Override config (e.g. imlam_direct_z_dmw_stage_1).")
    p.add_argument("--data-path", default="/tmp/slapo_local")
    p.add_argument("--split", default="test")
    p.add_argument("--frame-stack", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--query-weight", choices=["uniform", "occupancy"], default="occupancy",
                   help="Reduce the extraction attention over query tokens by: 'occupancy' (weight each "
                        "query by its entity occupancy - where the entity-region read-out attends, the "
                        "default) or 'uniform' (the plain query-marginal).")
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
        rows.append(collect_task(task, config_name, checkpoint, args.data_path, args.split,
                                 args.frame_stack, args.seed, device, query_weight=args.query_weight))

    model_tag = "imlam-direct-z" if any("direct_z" in c for c in config_names) else "imlam"
    qw_tag = "occ" if args.query_weight == "occupancy" else "uniform"
    tag = "all" if args.all else args.task
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scratchpad",
                                    "extraction_footprint", f"extraction_footprint_{model_tag}_{qw_tag}_{tag}.png")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    render(rows, out)


if __name__ == "__main__":
    main()
