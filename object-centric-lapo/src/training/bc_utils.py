from typing import Any

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces
from torch.utils.data import Subset
from tqdm.auto import tqdm

from ifo.common.utils.data import get_dataset


def build_dataset(task_name: str, cache_dir: str, split: str, block_size: int):
    """Build a dataset with temporal windowing."""
    return get_dataset(name=task_name, split=split, cache_dir=cache_dir, block_size=block_size)


def build_labeled_dataset(task_name: str, cache_dir: str, split: str, frame_stack: int):
    """Build a labeled dataset for decoder fine-tuning."""
    return get_dataset(name=task_name, split=split, cache_dir=cache_dir, block_size=frame_stack)


def maybe_subset_dataset(dataset: Any, subset_size: int):
    """Optionally slice a dataset to a fixed subset size."""
    if subset_size is None or subset_size <= 0:
        return dataset
    dataset_len = len(dataset)
    if dataset_len <= subset_size:
        return dataset
    indices = list(range(int(subset_size)))
    return Subset(dataset, indices)


def collate_fn(batch):
    """Identity collate for TensorDict batches."""
    return batch


@torch.no_grad()
def rollout_bc_agent_in_env(
    env,
    model: nn.Module,
    device: str,
    *,
    rollout_steps: int,
    rollout_sample: bool,
    seed: int,
) -> tuple[float, float, int, float]:
    """Roll out a model in a vectorized env and report episode stats."""
    num_envs = int(env.num_envs)
    rollout_steps = int(rollout_steps)
    steps_per_env = max(1, (rollout_steps + num_envs - 1) // num_envs)

    obs, _info = env.reset(seed=seed)
    episode_returns: list[float] = []
    episode_lengths: list[float] = []
    episode_successes: list[float] = []
    episode_had_success = np.zeros(num_envs, dtype=bool)
    has_success_key = False

    was_training = model.training
    model.eval()
    for _ in tqdm(range(steps_per_env), desc="BC rollout", leave=False):
        obs_t = torch.as_tensor(obs, device=device)
        if obs_t.dtype != torch.float32:
            obs_t = obs_t.float()

        dist = model(obs_t)
        action = dist.sample() if rollout_sample else dist.mode
        action_np = action.detach().cpu().numpy()
        if isinstance(env.single_action_space, spaces.Discrete):
            action_np = action_np.astype(np.int64, copy=False)
        else:
            action_np = action_np.astype(np.float32, copy=False)

        obs, _reward, terminated, truncated, info = env.step(action_np)
        done = np.asarray(terminated, dtype=bool) | np.asarray(truncated, dtype=bool)

        if isinstance(info, dict) and "success" in info:
            has_success_key = True
            success_np = np.asarray(info["success"], dtype=bool)
            success_np = np.ravel(success_np)
            assert success_np.size == num_envs, f"Success array size mismatch: {success_np.size} != {num_envs}"
            episode_had_success |= success_np

        if done.any() and isinstance(info, dict):
            stats = info.get("episode_statistics")
            stats_mask = info.get("_episode_statistics")
            if stats is not None and stats_mask is not None:
                stats_mask_np = np.asarray(stats_mask, dtype=bool)
                if stats_mask_np.any():
                    returns = np.asarray(stats["r"], dtype=np.float32)[stats_mask_np]
                    lengths = np.asarray(stats["l"], dtype=np.float32)[stats_mask_np]
                    episode_returns.extend(returns.astype(float).tolist())
                    episode_lengths.extend(lengths.astype(float).tolist())
                    episode_successes.extend(episode_had_success[stats_mask_np].astype(float).tolist())
            episode_had_success[done] = False

    if was_training:
        model.train()

    episode_return_mean = float(np.mean(episode_returns)) if episode_returns else float("nan")
    episode_length_mean = float(np.mean(episode_lengths)) if episode_lengths else float("nan")
    success_rate = (
        float(np.mean(episode_successes)) if (has_success_key and episode_successes) else 0.0
    )
    return episode_return_mean, episode_length_mean, int(len(episode_returns)), success_rate
