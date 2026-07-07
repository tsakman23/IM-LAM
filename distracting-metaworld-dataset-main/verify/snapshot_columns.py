"""Snapshot the first N transitions of the (unmodified) generator to an .npz.

Run this BEFORE modifying generate_dataset_huggingface.py so it captures the
released 8-column baseline used by the additivity invariant.
"""
import argparse
import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from generate_dataset_huggingface import episode_generator  # noqa: E402


def snapshot(env_name: str, split: str, seed: int, n: int, out: str) -> None:
    cols = {k: [] for k in (
        "observation", "observation_distracted", "mask",
        "state", "action", "reward", "terminated", "truncated",
    )}
    gen = episode_generator(
        env_name=env_name, split=split,
        max_num_episodes=None, max_num_steps=n,
        base_seed=seed, camera_name="corner",
        background_dataset_path="datasets/DAVIS",
    )
    for i, row in enumerate(gen):
        if i >= n:
            break
        cols["observation"].append(np.asarray(row["observation"], dtype=np.uint8))
        cols["observation_distracted"].append(np.asarray(row["observation_distracted"], dtype=np.uint8))
        cols["mask"].append(np.asarray(row["mask"], dtype=np.uint8))
        cols["state"].append(np.asarray(row["state"], dtype=np.float32))
        cols["action"].append(np.asarray(row["action"], dtype=np.float32))
        cols["reward"].append(np.float32(row["reward"]))
        cols["terminated"].append(bool(row["terminated"]))
        cols["truncated"].append(bool(row["truncated"]))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **{k: np.asarray(v) for k, v in cols.items()})
    print(f"Saved {len(cols['observation'])} rows to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Meta-World/MT1-push-v3")
    p.add_argument("--split", default="train")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", default="datasets/verify/golden_push-v3_seed0.npz")
    a = p.parse_args()
    snapshot(a.env, a.split, a.seed, a.n, a.out)
