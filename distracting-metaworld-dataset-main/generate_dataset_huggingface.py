#!/usr/bin/env python3
"""
Generate a standard HuggingFace dataset for MetaWorld with three observation variants:
  - observation        → vanilla RGB render          (PIL Image, mode="RGB")
  - observation_dist   → DAVIS background distracted (PIL Image, mode="RGB")
  - mask               → robot-arm segmentation mask (PIL Image, mode="L", 1 channel)

Dataset structure on the Hub
-----------------------------
  repo_id   : EpicPinkPenguin/visual_distracting_metaworld
  config    : task slug, e.g. "bin-picking-v3"
  split     : "train" | "test"

This mirrors the EpicPinkPenguin/visual_distracting_control_suite layout.

DAVIS split mapping
-------------------
  --split train  →  DAVIS "train" videos
  --split test   →  DAVIS "val"   videos

Usage
-----
# dry-run (no upload, 20 frames)
python generate_dataset.py \
    --env Meta-World/MT1-bin-picking-v3 \
    --split train --num_episodes 2 --seed 0 --dry_run

# real upload
python generate_dataset.py \
    --env Meta-World/MT1-bin-picking-v3 \
    --split train --num_episodes 200 --seed 0 --push_to_hub

# test split (different seed to avoid overlap)
python generate_dataset.py \
    --env Meta-World/MT1-bin-picking-v3 \
    --split test --num_episodes 50 --seed 10000 --push_to_hub
"""

import argparse
import os
import shutil
from typing import Any, Dict, Generator, Literal, Optional

import metaworld
import metaworld.policies
import numpy as np
from PIL import Image
from rich.console import Console
from tqdm.auto import tqdm

from datasets import Dataset, Features, Sequence, Value
from datasets import Image as HFImage
from make_env import make_metaworld_env

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────

# REPO_ID     = "EpicPinkPenguin/visual_distracting_metaworld"
REPO_ID     = "tsakman23/visual_masked_distracting_metaworld"
IMG_HW      = 128
STATE_DIM   = 39
ACTION_DIM  = 4

# HuggingFace feature schema.
# mask is stored as single-channel PIL "L" images – HFImage() accepts any PIL mode.
FEATURES = Features({
    "observation":            HFImage(),
    "observation_distracted": HFImage(),
    "mask":                   HFImage(),
    "object_mask":            HFImage(),
    "state":                  Sequence(Value("float32"), length=STATE_DIM),
    "object_state":           Sequence(Value("float32")),   # variable: block-concat [all obj pos; all obj quat] (push-v3: pos3+quat4=7)
    "action":                 Sequence(Value("float32"), length=ACTION_DIM),
    "reward":                 Value("float32"),
    "terminated":             Value("bool"),
    "truncated":              Value("bool"),
})

# ────────────────────────────────────────────────────────────────────────────
# Image helpers
# ────────────────────────────────────────────────────────────────────────────

def _to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """Return a (H, W, 3) uint8 array regardless of input layout."""
    if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[1] > 3:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.dtype != np.uint8:
        arr = (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def _extract_image(obs: dict) -> np.ndarray:
    """Pull the pixel array from a VectorEnv obs dict (num_envs=1)."""
    for key in ("pixels", "image", "rgb", "observation"):
        if key in obs:
            arr = np.asarray(obs[key])
            if arr.ndim == 4:       # (N, H, W, C) – strip batch dim
                arr = arr[0]
            return _to_rgb_uint8(arr)
    raise KeyError(
        f"No image key in obs dict. Keys: {list(obs.keys())}\n"
        "Expected one of: 'pixels', 'image', 'rgb', 'observation'."
    )


def _make_mask(seg_arr: np.ndarray) -> np.ndarray:
    """
    Binary arm mask from the segmentation render.
    Returns (H, W) uint8: 255 = arm foreground, 0 = background.
    """
    if seg_arr.ndim == 3:
        foreground = np.any(seg_arr > 0, axis=-1)
    else:
        foreground = seg_arr > 0
    return (foreground * 255).astype(np.uint8)


# ────────────────────────────────────────────────────────────────────────────
# Policy lookup
# ────────────────────────────────────────────────────────────────────────────

def _get_policy(base_env: str) -> Any:
    policy_map = metaworld.policies.ENV_POLICY_MAP
    if base_env in policy_map:
        return policy_map[base_env]()
    for task_key in metaworld.ALL_V3_ENVIRONMENTS:
        if base_env.endswith(task_key) and task_key in policy_map:
            print(f"Policy: {policy_map[task_key].__name__}")
            return policy_map[task_key]()
    raise ValueError(
        f"No expert policy for '{base_env}'.\n"
        f"Available: {sorted(policy_map)}"
    )


def _task_slug(env_id: str) -> str:
    """'Meta-World/MT1-bin-picking-v3'  →  'bin-picking-v3'"""
    part = env_id.split("/")[-1]        # MT1-bin-picking-v3
    return part[len("MT1-"):] if part.startswith("MT1-") else part


# Per-task object-body overrides for the object mask. Empty = use the automatic
# movable-w.r.t.-world heuristic. Populate only when an overlay check shows the
# heuristic selects the wrong body (find names via env.unwrapped.model.body(i).name).
TASK_OBJECT_OVERRIDES: dict[str, list[str]] = {
    # e.g. "door-open-v3": ["door_link"],
}


# ────────────────────────────────────────────────────────────────────────────
# Paired environment
# ────────────────────────────────────────────────────────────────────────────

class PairedEnv:
    """
    Three synchronised MetaWorld VectorEnvs (num_envs=1):
      _vanilla      – clean background   → RGB observation
      _distracting  – DAVIS background   → RGB observation_distracted
      _segmentation – segmentation mode  → binary arm mask (mode L)

    All three receive identical seeds and actions so physics stays in sync.
    Images are read directly from the obs dicts – no extra render() calls.

    DAVIS split:
      train split → DAVIS "train" videos
      test  split → DAVIS "val"   videos
    """

    def __init__(
        self,
        env_name: str,
        split: Literal["train", "test"] = "train",
        camera_name: str = "corner",
        background_dataset_path: str = "datasets/DAVIS",
        seed: Optional[int] = None,
        object_body_names: Optional[list] = None,
    ) -> None:
        davis_split = "train" if split == "train" else "val"

        common = dict(
            env_name=env_name,
            num_envs=1,
            camera_name=camera_name,
            seed=seed,
            height=IMG_HW,
            width=IMG_HW,
        )

        self._vanilla      = make_metaworld_env(**common, distracting=False, segmentation=False)
        self._distracting  = make_metaworld_env(
            **common,
            distracting=True,
            segmentation=False,
            background_dataset_path=background_dataset_path,
            dataset_videos=davis_split,
        )
        self._segmentation = make_metaworld_env(
            **common, distracting=False, segmentation=True,
            object_body_names=object_body_names,
        )

    def reset(self, seed: Optional[int] = None) -> dict:
        obs_v, info_v = self._vanilla.reset(seed=seed)
        obs_d, _      = self._distracting.reset(seed=seed)
        obs_s, _      = self._segmentation.reset(seed=seed)
        return self._pack(obs_v, obs_d, obs_s)

    def step(self, action: np.ndarray):
        a = action[np.newaxis]                              # (1, 4) for VectorEnv
        obs_v, rew, term, trunc, _ = self._vanilla.step(a)
        obs_d, *_                  = self._distracting.step(a)
        obs_s, *_                  = self._segmentation.step(a)
        return self._pack(obs_v, obs_d, obs_s), float(rew[0]), bool(term[0]), bool(trunc[0])

    def close(self) -> None:
        for e in (self._vanilla, self._distracting, self._segmentation):
            try:
                e.close()
            except Exception:
                pass

    def _pack(self, obs_v, obs_d, obs_s) -> dict:
        img_van  = _extract_image(obs_v)
        img_dis  = _extract_image(obs_d)
        seg_arr  = _extract_image(obs_s)
        state    = obs_v["state"][0].astype(np.float32)
        mask_arr = _make_mask(seg_arr)                     # agent, UNCHANGED

        # Object mask: read-only segmentation render on the seg env's current state.
        masks = self._segmentation.call("render_masks")[0]
        object_mask_arr = masks["object"][:, :, 0].astype(np.uint8)  # (H, W) 0/255

        # Object state: world-frame pos + quat from the vanilla env's current state.
        obj_pos  = np.asarray(self._vanilla.call("_get_pos_objects")[0], dtype=np.float32).ravel()
        obj_quat = np.asarray(self._vanilla.call("_get_quat_objects")[0], dtype=np.float32).ravel()
        object_state = np.concatenate([obj_pos, obj_quat]).astype(np.float32)

        return {
            "img_van":      img_van,
            "img_dis":      img_dis,
            "mask_arr":     mask_arr,
            "object_mask":  object_mask_arr,
            "state":        state,
            "object_state": object_state,
        }


# ────────────────────────────────────────────────────────────────────────────
# Dataset generator
# ────────────────────────────────────────────────────────────────────────────

def episode_generator(
    env_name: str,
    split: Literal["train", "test"],
    max_num_episodes: Optional[int],
    max_num_steps: Optional[int],
    base_seed: int,
    camera_name: str,
    background_dataset_path: str,
) -> Generator[Dict[str, Any], None, None]:
    """
    Yields one sample dict per timestep, matching FEATURES exactly.
    PIL Images are created here so Dataset.from_generator() can encode them.
    """
    policy      = _get_policy(env_name)
    env         = PairedEnv(
        env_name=env_name,
        split=split,
        camera_name=camera_name,
        background_dataset_path=background_dataset_path,
        seed=base_seed,
        object_body_names=TASK_OBJECT_OVERRIDES.get(_task_slug(env_name)),
    )
    _first_log   = True
    total_steps  = 0
    total_episodes = 0
    episode_returns = []

    # Estimated total number of frames so the progress bar has a defined end.
    if max_num_steps is not None:
        pbar = tqdm(total=max_num_steps, desc=f"Steps ({split})", unit="step")
    else:
        pbar = tqdm(total=max_num_episodes, desc=f"Episodes ({split})", unit="episode")

    obs = env.reset(seed=base_seed)
    reward = 0.0
    terminated = truncated = False
    ep_return = 0.0

    try:
        while (max_num_episodes is None or total_episodes < max_num_episodes) and \
              (max_num_steps is None or total_steps < max_num_steps):
            # PIL conversion – observation at time t
            pil_obs  = Image.fromarray(obs["img_van"],  mode="RGB")
            pil_dist = Image.fromarray(obs["img_dis"],  mode="RGB")
            pil_mask = Image.fromarray(obs["mask_arr"], mode="L")   # single channel
            pil_object_mask = Image.fromarray(obs["object_mask"], mode="L")

            if _first_log:
                print(f"first frame: obs={obs['img_van'].shape}  mask={obs['mask_arr'].shape} (L)")
                _first_log = False

            action = policy.get_action(obs["state"]).astype(np.float32)
            action = action.clip(-1, 1)

            yield {
                "observation":      pil_obs,
                "observation_distracted": pil_dist,
                "mask":             pil_mask,
                "object_mask":      pil_object_mask,
                "state":            obs["state"].tolist(),
                "object_state":     obs["object_state"].tolist(),
                "action":           action.tolist(),
                "reward":           float(reward),
                "terminated":       bool(terminated),
                "truncated":        bool(truncated),
            }

            done = terminated or truncated
            total_steps += 1
            ep_return  += float(reward)

            if done:
                total_episodes += 1
                episode_returns.append(ep_return)
                ep_return = 0.0

            if max_num_steps is not None:
                pbar.update(1)
            else:
                pbar.update(done)

            next_obs, reward, terminated, truncated = env.step(action)
            obs = next_obs
    finally:
        # Finalise progress bar with the exact number of frames actually generated.
        if pbar is not None:
            pbar.total = pbar.n
            pbar.refresh()
            pbar.close()
        if episode_returns:
            # Print episode returns
            mean_return = sum(episode_returns) / len(episode_returns)
            print(f"\nAverage episode return over {len(episode_returns)} episodes: {mean_return:.3f}")
            # Save episode returns to a file; sanitise env_name so it never creates directories
            safe_env_name = env_name.replace("/", "_")
            filename = f"episode_returns_{safe_env_name}_{split}.txt"
            with open(filename, "w") as f:
                f.write(f"Average episode return: {mean_return:.3f}\n")
        env.close()


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main(
    env_name: str,
    split: Literal["train", "test"],
    num_episodes: Optional[int],
    num_steps: Optional[int],
    base_seed: int,
    camera_name: str,
    background_dataset_path: str,
    push_to_hub: bool,
    dry_run: bool,
    cache_dir: str,
    max_shard_size: str,
    private: bool,
) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    console = Console()

    task = _task_slug(env_name)

    limit_str = f"max_steps={num_steps}" if num_steps else f"episodes={num_episodes}"
    console.print(
        f"\n[bold cyan]"
        f"env={env_name}  task={task}  split={split}  "
        f"{limit_str}  res={IMG_HW}x{IMG_HW}  "
        f"seed {base_seed}"
        f"[/bold cyan]\n"
    )

    # gen_kwargs contains only plain serialisable values (str, int, …).
    # Passing the top-level function + gen_kwargs lets datasets pickle
    # just the kwargs for cache-fingerprinting, avoiding the dill crash
    # that happens when a closure captures complex objects.
    gen_kwargs = dict(
        env_name=env_name,
        split=split,
        max_num_episodes=num_episodes,
        max_num_steps=num_steps,
        base_seed=base_seed,
        camera_name=camera_name,
        background_dataset_path=background_dataset_path,
    )

    # ── dry-run ─────────────────────────────────────────────────────────────
    if dry_run:
        console.print("[yellow][DRY RUN] Testing generator – no upload[/yellow]")
        count = 0
        for sample in episode_generator(**gen_kwargs):
            count += 1
            if count <= 3:
                console.print(
                    f"  sample {count}: "
                    f"obs={sample['observation'].size}  "
                    f"mask_mode={sample['mask'].mode}  "
                    f"mask={sample['mask'].size}  "
                    f"reward={sample['reward']:.3f}"
                )
            if count >= 20:
                break
        console.print(f"[green]✓ Generator OK – {count} samples tested[/green]")
        return

    # ── build dataset ────────────────────────────────────────────────────────
    # Dataset.from_generator caches to disk, then push_to_hub commits reliably.
    # Cache is deleted automatically after a successful upload.
    console.print("[1/3] Building dataset from generator (cached to disk)…")
    # Auto-generate unique cache dir per task+split to avoid NFS conflicts
    # when multiple jobs run in parallel.
    # Always wipe any existing cache so stale data from previous (buggy) runs
    # can never be reused.
    cache_dir = os.path.join("datasets", "hf_cache", f"{task}_{split}")
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)

    dataset = Dataset.from_generator(
        episode_generator,
        gen_kwargs=gen_kwargs,
        features=FEATURES,
        cache_dir=cache_dir,
    )
    console.print(f"      ✓ {len(dataset):,} frames collected")

    # ── upload ───────────────────────────────────────────────────────────────
    if push_to_hub:
        console.print(
            f"\n[3/3] Uploading  repo={REPO_ID}  "
            f"config={task}  split={split} …"
        )
        dataset.push_to_hub(
            repo_id=REPO_ID,
            config_name=task,
            split=split,
            private=private,
        )
        console.print(
            f"\n[bold green]✓✓✓ Done![/bold green]  "
            f"https://huggingface.co/datasets/{REPO_ID}"
        )
        # Delete cache now that upload succeeded
        shutil.rmtree(cache_dir, ignore_errors=True)
        console.print(f"[dim]Cache deleted: {cache_dir}[/dim]")
    else:
        console.print(
            f"\n[yellow]--push_to_hub not set – "
            f"dataset cached locally in {cache_dir}[/yellow]"
        )


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--env", type=str, required=True, metavar="GYM_ID",
                        help="e.g.  Meta-World/MT1-bin-picking-v3")
    parser.add_argument("--split", choices=["train", "test"], default="train",
                        help="train → DAVIS train  |  test → DAVIS val")
    parser.add_argument("--num_episodes", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None,
                        help="Stop after this many total steps (overrides episode count)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base seed; episode i uses seed+i")
    parser.add_argument("--camera", type=str, default="corner",
                        choices=["corner", "corner2", "corner3",
                                 "topview", "behindGripper", "gripperPOV"])
    parser.add_argument("--background_dataset_path", type=str,
                        default="datasets/DAVIS")
    parser.add_argument("--cache_dir", type=str, default="datasets/hf_cache",
                        help="Local HuggingFace dataset cache directory")
    parser.add_argument("--max_shard_size", type=str, default="1GB",
                        help="Max parquet shard size (e.g. 500MB, 2GB)")
    parser.add_argument("--push_to_hub", action="store_true",
                        help=f"Upload to {REPO_ID}")
    parser.add_argument("--private", action="store_true",
                        help="Make the HuggingFace repo private")
    parser.add_argument("--dry_run", action="store_true",
                        help="Test generator without uploading (20 frames max)")

    args = parser.parse_args()
    assert args.num_episodes is not None or args.num_steps is not None, \
        "Either --num_episodes or --num_steps must be set"
    assert not (args.num_episodes is not None and args.num_steps is not None), \
        "Only one of --num_episodes or --num_steps can be set"

    main(
        env_name=args.env,
        split=args.split,
        num_episodes=args.num_episodes,
        num_steps=args.num_steps,
        base_seed=args.seed,
        camera_name=args.camera,
        background_dataset_path=args.background_dataset_path,
        push_to_hub=args.push_to_hub,
        dry_run=args.dry_run,
        cache_dir=args.cache_dir,
        max_shard_size=args.max_shard_size,
        private=args.private,
    )
