"""Save agent/object mask overlays and report object_state motion for eyeballing."""
import argparse
import os

import numpy as np
from PIL import Image

os.environ.setdefault("MUJOCO_GL", "egl")

from generate_dataset_huggingface import episode_generator  # noqa: E402


def run(env_name: str, n: int, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    gen = episode_generator(
        env_name=env_name, split="train",
        max_num_episodes=None, max_num_steps=n,
        base_seed=0, camera_name="corner",
        background_dataset_path="datasets/DAVIS",
    )
    states = []
    for i, row in enumerate(gen):
        if i >= n:
            break
        obs = np.asarray(row["observation"]).astype(np.float32)
        agent = np.asarray(row["mask"].convert("L")) > 0
        obj = np.asarray(row["object_mask"].convert("L")) > 0
        overlay = obs.copy()
        overlay[agent] = 0.5 * overlay[agent] + 0.5 * np.array([0, 255, 0])   # agent green
        overlay[obj]   = 0.5 * overlay[obj]   + 0.5 * np.array([255, 0, 0])   # object red
        Image.fromarray(overlay.astype(np.uint8)).save(os.path.join(out_dir, f"overlay_{i:04d}.png"))
        states.append(np.asarray(row["object_state"], dtype=np.float32))
    states = np.asarray(states)
    print(f"object_state dim={states.shape[1]}  "
          f"pos-range per axis (first 3 dims): "
          f"{np.ptp(states[:, :3], axis=0)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Meta-World/MT1-push-v3")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--out", default="datasets/verify/overlays/push-v3")
    a = p.parse_args()
    run(a.env, a.n, a.out)
