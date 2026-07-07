"""Render per-frame panels [observation | agent mask | object mask | overlay]
for visual inspection of the GT masks. Panels are upscaled (nearest) so small
object masks (e.g. the push puck) are clearly visible.

Usage:
  MUJOCO_GL=egl PYTHONPATH=. python verify/render_mask_panels.py \
      --env Meta-World/MT1-push-v3 --steps 150 --num_frames 8 \
      --out datasets/verify/mask_panels/push-v3
"""
import argparse
import os

import numpy as np
from PIL import Image

os.environ.setdefault("MUJOCO_GL", "egl")

from generate_dataset_huggingface import episode_generator  # noqa: E402


def _to_rgb(img_l: np.ndarray) -> np.ndarray:
    """(H,W) 0/255 mask -> (H,W,3) white-on-black."""
    return np.stack([img_l, img_l, img_l], axis=-1).astype(np.uint8)


def _overlay(obs: np.ndarray, agent: np.ndarray, obj: np.ndarray) -> np.ndarray:
    out = obs.astype(np.float32).copy()
    a = agent > 0
    o = obj > 0
    out[a] = 0.5 * out[a] + 0.5 * np.array([0, 255, 0])   # agent green
    out[o] = 0.5 * out[o] + 0.5 * np.array([255, 0, 0])   # object red
    return out.astype(np.uint8)


def render(env_name: str, steps: int, num_frames: int, scale: int, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    stride = max(1, steps // num_frames)
    gen = episode_generator(
        env_name=env_name, split="train",
        max_num_episodes=None, max_num_steps=steps,
        base_seed=0, camera_name="corner",
        background_dataset_path="datasets/DAVIS",
    )
    saved = 0
    for i, row in enumerate(gen):
        if i >= steps:
            break
        if i % stride != 0:
            continue
        obs = np.asarray(row["observation"].convert("RGB"), dtype=np.uint8)
        agent = np.asarray(row["mask"].convert("L"), dtype=np.uint8)
        obj = np.asarray(row["object_mask"].convert("L"), dtype=np.uint8)
        tiles = [obs, _to_rgb(agent), _to_rgb(obj), _overlay(obs, agent, obj)]
        panel = np.concatenate(tiles, axis=1)  # (H, 4W, 3)
        img = Image.fromarray(panel)
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
        path = os.path.join(out_dir, f"panel_{i:04d}.png")
        img.save(path)
        obj_px = int((obj > 0).sum())
        agent_px = int((agent > 0).sum())
        print(f"frame {i:4d}: agent_px={agent_px:5d} object_px={obj_px:4d} -> {path}")
        saved += 1
    print(f"saved {saved} panels to {out_dir}  (columns: observation | agent mask | object mask | overlay)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="Meta-World/MT1-push-v3")
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--scale", type=int, default=3)
    p.add_argument("--out", default="datasets/verify/mask_panels/push-v3")
    a = p.parse_args()
    render(a.env, a.steps, a.num_frames, a.scale, a.out)
