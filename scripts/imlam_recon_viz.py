"""IM-LAM overfit reconstruction viz (Phase 7 integration check A).

Overfits the interaction FDM on a fixed batch (same setup as scripts/imlam_overfit_gate.py, dual loss),
then renders per-sample panels - current frame o_t, true next frame o_{t+1}, IM-LAM prediction, and the
object-region error - so you can SEE the object being predicted, not just read L_O -> 0. Samples with
the largest object footprint are shown (most informative for the object branch).

This is a diagnostic on MEMORIZED frames (it confirms the machinery can reconstruct the object);
held-out generalization is a Phase-8 check that needs a real trained checkpoint.

Usage:
    conda_env/bin/python scripts/imlam_recon_viz.py --out imlam_overfit_recon.png
    conda_env/bin/python scripts/imlam_recon_viz.py --steps 1500 --n-show 4 --task sweep-into-v3
"""
import argparse
import os
import sys

import numpy as np
import torch
from gymnasium import spaces
from torch.utils.data import DataLoader, Subset

import hydra
from hydra import compose, initialize_config_dir

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from ifo.common.utils.utility import tensordict_collate  # noqa: E402

CONFIG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments", "configs"))


def _load_batch(cfg, args, device):
    if args.batch_cache and os.path.exists(args.batch_cache):
        return torch.load(args.batch_cache, weights_only=False).to(device)
    print(f"loading {args.split} split (one-time)...", flush=True)
    ds = hydra.utils.instantiate(cfg.dataset, split=args.split)
    sub = Subset(ds, list(range(min(args.n_transitions, len(ds)))))
    batch = next(iter(DataLoader(sub, batch_size=len(sub), collate_fn=tensordict_collate, shuffle=False)))
    if args.batch_cache:
        torch.save(batch, args.batch_cache)
    return batch.to(device)


def _build_and_overfit(cfg, batch, args, device):
    _, t, c, h, w = batch["observation"].shape
    obs = spaces.Box(-np.inf, np.inf, (t, c, h, w), np.float32)
    act = spaces.Box(-1.0, 1.0, (batch["action"].shape[-1],), np.float32)
    net = hydra.utils.instantiate(cfg.net)(observation_space=obs, action_space=act)
    module = hydra.utils.instantiate(cfg.module, net=net).to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    for step in range(args.steps):
        loss = module.training_step(batch, 0, 1)["loss"]
        opt.zero_grad(); loss.backward(); opt.step()
    return net


def _img(x):
    """(C,H,W) in [-0.5, 0.5] -> (H,W,C) in [0,1] for display."""
    return (x.detach().float().cpu() + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="sweep-into-v3")
    p.add_argument("--data-path", default="datasets/slapo_local/sweep-into-v3")
    p.add_argument("--split", default="test")
    p.add_argument("--batch-cache", default=None, help="Optional .pt cache for the fixed batch (fast reuse).")
    p.add_argument("--n-transitions", type=int, default=64)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-show", type=int, default=4)
    p.add_argument("--out", default="imlam_overfit_recon.png")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    device = torch.device(args.device)

    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="imlam_dual_dmw_stage_1", overrides=[
            f"env.name=Meta-World/masked-MT1-{args.task}",
            f"dataset.dataset_path={args.data_path}",
            "++module.log_dual_loss_grad_every=0"])

    torch.manual_seed(args.seed)
    batch = _load_batch(cfg, args, device)
    print(f"overfitting {args.steps} steps (dual, lr={args.lr})...", flush=True)
    net = _build_and_overfit(cfg, batch, args, device)

    fs = net.frame_stack
    net.eval()
    with torch.no_grad():
        pred = net(batch["observation"], batch["mask"], object_mask=batch["object_mask"])[0]
    current = batch["observation"][:, fs - 1]           # o_t
    target = batch["observation"][:, fs]                # o_{t+1}
    obj_mask = batch["object_mask"][:, fs]              # (B,1,H,W) at t+1

    # Show the samples with the largest object footprint (where the object branch matters most).
    order = obj_mask.flatten(1).sum(1).argsort(descending=True)
    idx = order[:args.n_show].tolist()

    def _obj_crop(hw_or_hwc, mask_hw, side=40):
        """Crop a `side`x`side` window centred on the object (for a visible object-scale view)."""
        ys, xs = np.where(mask_hw)
        H = mask_hw.shape[0]
        if len(ys) == 0:
            cy = cx = H // 2
        else:
            cy, cx = int(ys.mean()), int(xs.mean())
        half = side // 2
        y0 = min(max(cy - half, 0), H - side); x0 = min(max(cx - half, 0), H - side)
        sl = (slice(y0, y0 + side), slice(x0, x0 + side))
        return hw_or_hwc[sl] if hw_or_hwc.ndim == 2 else hw_or_hwc[sl[0], sl[1], :]

    cols = ["current o_t", "true o_{t+1}", "IM-LAM pred", "object error",
            "true (object zoom)", "pred (object zoom)"]
    fig, axes = plt.subplots(len(idx), 6, figsize=(6 * 2.3, len(idx) * 2.3))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(idx):
        err = (pred[i] - target[i]).abs().mean(0).cpu().numpy()          # (H,W) full-frame abs error
        m = obj_mask[i, 0].cpu().numpy() > 0.5
        tgt_img, pred_img = _img(target[i]), _img(pred[i])
        panels = [_img(current[i]), tgt_img, pred_img, err,
                  _obj_crop(tgt_img, m), _obj_crop(pred_img, m)]
        for cidx, (ax, panel) in enumerate(zip(axes[r], panels)):
            if cidx == 3:
                ax.imshow(panel, cmap="magma", vmin=0, vmax=max(err.max(), 1e-6))
                ax.contour(m, levels=[0.5], colors="lime", linewidths=0.8)
            else:
                ax.imshow(panel, interpolation="nearest" if cidx >= 4 else None)
                if cidx in (1, 2):
                    ax.contour(m, levels=[0.5], colors="lime", linewidths=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[cidx], fontsize=10)
        obj_err = float((pred[i] - target[i]).abs().mean(0)[m].mean()) if m.any() else float("nan")
        axes[r, 0].set_ylabel(f"obj |err|={obj_err:.4f}", fontsize=9)

    fig.suptitle(f"IM-LAM overfit reconstruction ({args.task}, dual, {args.steps} steps) - "
                 f"masked loss trains agent+object only; distractor bg is high-error by design",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
