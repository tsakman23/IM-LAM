"""Stream the first N rows of the authors' released push-v3 config to an .npz."""
import argparse
import os

import numpy as np

os.environ.setdefault("HF_HOME", "/data2/masklam/datasets/hf_home")

from datasets import load_dataset  # noqa: E402


def stream(repo: str, config: str, split: str, n: int, out: str) -> None:
    ds = load_dataset(repo, name=config, split=split, streaming=True)
    cols = {k: [] for k in (
        "observation", "observation_distracted", "mask",
        "state", "action", "reward", "terminated", "truncated",
    )}
    for i, row in enumerate(ds):
        if i >= n:
            break
        cols["observation"].append(np.asarray(row["observation"].convert("RGB"), dtype=np.uint8))
        cols["observation_distracted"].append(np.asarray(row["observation_distracted"].convert("RGB"), dtype=np.uint8))
        cols["mask"].append(np.asarray(row["mask"].convert("L"), dtype=np.uint8))
        cols["state"].append(np.asarray(row["state"], dtype=np.float32))
        cols["action"].append(np.asarray(row["action"], dtype=np.float32))
        cols["reward"].append(np.float32(row["reward"]))
        cols["terminated"].append(bool(row["terminated"]))
        cols["truncated"].append(bool(row["truncated"]))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out, **{k: np.asarray(v) for k, v in cols.items()})
    print(f"Saved {len(cols['observation'])} released rows to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="EpicPinkPenguin/visual_distracting_metaworld")
    p.add_argument("--config", default="push-v3")
    p.add_argument("--split", default="train")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--out", default="datasets/verify/released_push-v3.npz")
    a = p.parse_args()
    stream(a.repo, a.config, a.split, a.n, a.out)
