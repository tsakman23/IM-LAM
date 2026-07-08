"""IID check: compare our generated DMW data against the MaskLAM authors' release
on the 8 shared columns, using two-sample Kolmogorov-Smirnov statistics.

Only the columns present in BOTH datasets are compared (our object_mask / object_state
have no counterpart in the release and are ignored here).

"Ours" is, by default, generated in-process (no push required); pass --ours-repo to
stream an already-pushed HF dataset instead. "Theirs" is streamed from the authors'
released repo.

Usage:
  # generate ours on the fly (seed matches the split) and compare to the release
  MUJOCO_GL=egl PYTHONPATH=. python verify/iid_check.py \
      --config push-v3 --split train --n 3000

  # or compare an already-pushed dataset to the release
  PYTHONPATH=. python verify/iid_check.py \
      --config push-v3 --split train --n 3000 \
      --ours-repo tsakman23/visual_masked_distracting_metaworld
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HOME", "/data2/masklam/datasets/hf_home")
# Make the repo root importable so `python verify/iid_check.py` works without PYTHONPATH=.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SHARED = ["observation", "observation_distracted", "mask",
          "state", "action", "reward", "terminated", "truncated"]
# KS D below this is "indistinguishable" at n~few-thousand (KS 0.05-crit ~ 1.36*sqrt(2/n)).
KS_FLAG = 0.10


def ks_d(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic D (max CDF gap). No scipy needed."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def _img_channel_means(pil_img) -> np.ndarray:
    return np.asarray(pil_img.convert("RGB"), dtype=np.float32).reshape(-1, 3).mean(0)


def collect(rows_iter, n):
    """Pull n rows, extracting the comparable features from the 8 shared columns."""
    out = {k: [] for k in ("state", "action", "reward", "mask_area",
                           "obs_mean", "obsd_mean", "terminated", "truncated")}
    ep_returns, ep_lengths = [], []
    cur_ret, cur_len = 0.0, 0
    for i, row in enumerate(rows_iter):
        if i >= n:
            break
        out["state"].append(np.asarray(row["state"], dtype=np.float32))
        out["action"].append(np.asarray(row["action"], dtype=np.float32))
        r = float(row["reward"])
        out["reward"].append(r)
        out["mask_area"].append(int((np.asarray(row["mask"].convert("L")) > 127).sum()))
        out["obs_mean"].append(_img_channel_means(row["observation"]))
        out["obsd_mean"].append(_img_channel_means(row["observation_distracted"]))
        term, trunc = bool(row["terminated"]), bool(row["truncated"])
        out["terminated"].append(term)
        out["truncated"].append(trunc)
        cur_ret += r
        cur_len += 1
        if term or trunc:
            ep_returns.append(cur_ret)
            ep_lengths.append(cur_len)
            cur_ret, cur_len = 0.0, 0
    for k in out:
        out[k] = np.asarray(out[k])
    out["ep_return"] = np.asarray(ep_returns)
    out["ep_length"] = np.asarray(ep_lengths)
    return out


def ours_iter(args):
    if args.ours_repo:
        from datasets import load_dataset
        ds = load_dataset(args.ours_repo, name=args.config, split=args.split, streaming=True)
        return iter(ds)
    # generate in-process; seed matches the split convention
    from generate_dataset_huggingface import episode_generator
    seed = 0 if args.split == "train" else 100000
    return episode_generator(
        env_name=f"Meta-World/MT1-{args.config}", split=args.split,
        max_num_episodes=None, max_num_steps=args.n, base_seed=seed,
        camera_name="corner", background_dataset_path="datasets/DAVIS",
    )


def theirs_iter(args):
    from datasets import load_dataset
    ds = load_dataset(args.theirs_repo, name=args.config, split=args.split, streaming=True)
    return iter(ds)


def report(ours, theirs):
    print(f"\n{'metric':<26}{'KS D':>9}   {'ours mean':>12}{'theirs mean':>13}   flag")
    print("-" * 74)
    worst = 0.0
    lines = []

    def row(name, a, b):
        nonlocal worst
        d = ks_d(a, b)
        worst = max(worst, 0.0 if np.isnan(d) else d)
        flag = "  <-- DIFF" if (not np.isnan(d) and d > KS_FLAG) else ""
        lines.append(f"{name:<26}{d:>9.4f}   {np.mean(a):>12.4f}{np.mean(b):>13.4f}{flag}")

    for j in range(ours["state"].shape[1]):
        row(f"state[{j}]", ours["state"][:, j], theirs["state"][:, j])
    for j in range(ours["action"].shape[1]):
        row(f"action[{j}]", ours["action"][:, j], theirs["action"][:, j])
    row("reward", ours["reward"], theirs["reward"])
    row("agent mask area (px)", ours["mask_area"], theirs["mask_area"])
    for c, ch in enumerate("RGB"):
        row(f"obs mean {ch}", ours["obs_mean"][:, c], theirs["obs_mean"][:, c])
    for c, ch in enumerate("RGB"):
        row(f"obs_dist mean {ch}", ours["obsd_mean"][:, c], theirs["obsd_mean"][:, c])
    row("episode return", ours["ep_return"], theirs["ep_return"])
    row("episode length", ours["ep_length"], theirs["ep_length"])
    print("\n".join(lines))
    print("-" * 74)
    n_diff = sum(1 for ln in lines if "DIFF" in ln)
    print(f"worst KS D = {worst:.4f}  (flag threshold {KS_FLAG});  {n_diff} metric(s) flagged")
    print(f"ours: {len(ours['reward'])} rows / {len(ours['ep_return'])} episodes;  "
          f"theirs: {len(theirs['reward'])} rows / {len(theirs['ep_return'])} episodes")
    verdict = "PASS - distributions match (IID-consistent)" if n_diff == 0 else \
              "REVIEW - some metrics exceed the flag threshold (see DIFF rows)"
    print(f"\nVERDICT: {verdict}\n")
    return n_diff == 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    p.add_argument("--config", default="push-v3", help="task config, e.g. push-v3")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--n", type=int, default=3000, help="rows to sample from each side")
    p.add_argument("--ours-repo", default=None,
                   help="HF repo to stream OURS from; if omitted, ours is generated in-process")
    p.add_argument("--theirs-repo", default="EpicPinkPenguin/visual_distracting_metaworld")
    a = p.parse_args()
    print(f"IID check | config={a.config} split={a.split} n={a.n}")
    print(f"  ours   = {a.ours_repo or 'generated in-process'}")
    print(f"  theirs = {a.theirs_repo}")
    ours = collect(ours_iter(a), a.n)
    theirs = collect(theirs_iter(a), a.n)
    ok = report(ours, theirs)
    # Bypass interpreter finalization: MuJoCo/EGL + datasets streaming threads can
    # segfault during teardown (a cosmetic shutdown crash that fires after results
    # are already printed). Flush and hard-exit with the real status instead.
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0 if ok else 1)
