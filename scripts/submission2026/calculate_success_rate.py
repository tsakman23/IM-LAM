"""Success-rate rollout evaluation for SLAPO/LAPO checkpoints.

Runs closed-loop rollouts with a trained policy and reports the task success rate
alongside mean episode return and length. Success is read from the ``terminated``
signal: Meta-World eval envs are wrapped with ``MetaworldTerminateOnSuccessWrapper``
(``ifo/common/utils/env.py``), which sets ``terminated = bool(info["success"])``, so a
finished episode is a success iff it terminated (rather than truncating at the time
limit). Episode returns/lengths come from the ``RecordEpisodeStatistics`` wrapper
(``stats_key="episode_statistics"``), the same source the training-time RolloutCallback uses.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from tensordict import TensorDict
from torchmetrics import MeanMetric

from ifo.common.collector import Collector, ExplorationMode


def _select_action(policy, observation: np.ndarray, exploration_mode: ExplorationMode) -> np.ndarray:
    """Select an action from the policy for a batch of observations.

    Mirrors the action-selection in ``Collector.rollout`` for the deterministic
    ("mode") and stochastic ("sample") exploration modes used at evaluation time.
    """
    with torch.no_grad():
        obs_tensor = torch.as_tensor(observation, device=policy.device)
        prediction = policy(TensorDict({"observation": obs_tensor}, (obs_tensor.shape[0],)))
        assert "action_distribution" in prediction, "Policy must output action distribution"
        distribution = prediction["action_distribution"]
        if exploration_mode == ExplorationMode.MODE:
            action = distribution.mode
        elif exploration_mode == ExplorationMode.SAMPLE:
            action = distribution.sample()
        else:
            raise ValueError(f"Unsupported exploration mode for success evaluation: {exploration_mode}")
        # Upcast if using mixed precision, matching Collector.rollout.
        if action.dtype in (torch.float16, torch.bfloat16):
            action = action.float()
        return action.cpu().numpy()


def calculate_success_rate(
    collector: Collector,
    max_steps: int,
    exploration_mode: ExplorationMode = ExplorationMode.MODE,
    reset_before_rollout: bool = True,
) -> Tuple[float, float, float]:
    """Roll out the collector's policy and compute the task success rate.

    Args:
        collector: A ``Collector`` holding the vector ``env`` and trained ``policy``.
        max_steps: Total environment-step budget across all envs. The loop runs
            ``max_steps // num_envs`` iterations (each stepping every env once), matching
            the RolloutCallback convention so ``rollout_steps`` maps to the same coverage.
        exploration_mode: ``MODE`` (deterministic, default for eval) or ``SAMPLE``.
        reset_before_rollout: Reset the env before collecting; otherwise continue from
            the collector's last stored observation.

    Returns:
        ``(success_rate, episode_return, episode_length)`` averaged over completed
        episodes. Returns ``(0.0, 0.0, 0.0)`` if no episode completes within the budget.
    """
    env = collector.env
    policy = collector.policy
    num_envs = env.num_envs
    num_iterations = max(1, int(max_steps) // num_envs)

    if reset_before_rollout or not getattr(collector, "_env_is_reset", False):
        observation, _ = env.reset(seed=getattr(collector, "_random_seed", None))
    else:
        observation = collector._last_step["observation"]

    episode_returns, episode_lengths = MeanMetric(), MeanMetric()
    successes, episodes = 0, 0

    for _ in range(num_iterations):
        action = _select_action(policy, observation, exploration_mode)
        observation, _reward, terminated, truncated, info = env.step(action)
        terminated = np.asarray(terminated, dtype=bool)
        truncated = np.asarray(truncated, dtype=bool)
        done = terminated | truncated
        if done.any():
            finished = np.asarray(info["_episode_statistics"], dtype=bool)
            episode_returns(info["episode_statistics"]["r"][finished])
            episode_lengths(info["episode_statistics"]["l"][finished])
            successes += int(np.count_nonzero(terminated[finished]))
            episodes += int(np.count_nonzero(finished))

    if episodes == 0:
        return 0.0, 0.0, 0.0

    success_rate = successes / episodes
    return success_rate, float(episode_returns.compute()), float(episode_lengths.compute())
