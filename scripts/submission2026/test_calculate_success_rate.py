"""Tests for calculate_success_rate.

Run:  MUJOCO_GL=egl conda_env/bin/python scripts/submission2026/test_calculate_success_rate.py
(the script's own directory is on sys.path, so `from calculate_success_rate import ...` resolves,
matching how eval_slapo.py / eval_lapo.py import it).
"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calculate_success_rate import calculate_success_rate  # noqa: E402

from ifo.common.collector import ExplorationMode  # noqa: E402


class _FakeDist:
    def __init__(self, action: torch.Tensor):
        self._action = action

    @property
    def mode(self) -> torch.Tensor:
        return self._action

    def sample(self) -> torch.Tensor:
        return self._action


class _FakePolicy:
    """Returns a fixed action distribution; ignores the observation."""

    def __init__(self, num_envs: int, action_dim: int = 1):
        self.device = torch.device("cpu")
        self._action = torch.zeros(num_envs, action_dim)

    def __call__(self, _td):
        return {"action_distribution": _FakeDist(self._action)}


class _ScriptedVectorEnv:
    """Vector env replaying a scripted list of per-step episode-end events.

    Each event is a dict with optional keys 'terminated'/'truncated' (bool lists of
    length num_envs) and 'r'/'l' (float lists) giving returns/lengths for finished envs.
    On any step where an env finishes, we populate info like gymnasium's
    RecordEpisodeStatistics(stats_key="episode_statistics") does.
    """

    def __init__(self, num_envs: int, obs_dim: int, events: list[dict]):
        self.num_envs = num_envs
        self._obs_dim = obs_dim
        self._events = events
        self._i = 0

    def reset(self, seed=None):
        return np.zeros((self.num_envs, self._obs_dim), dtype=np.float32), {}

    def step(self, action):
        ev = self._events[self._i]
        self._i += 1
        obs = np.zeros((self.num_envs, self._obs_dim), dtype=np.float32)
        reward = np.zeros(self.num_envs, dtype=np.float32)
        terminated = np.asarray(ev.get("terminated", [False] * self.num_envs), dtype=bool)
        truncated = np.asarray(ev.get("truncated", [False] * self.num_envs), dtype=bool)
        mask = terminated | truncated
        info: dict = {}
        if mask.any():
            info["_episode_statistics"] = mask
            info["episode_statistics"] = {
                "r": np.asarray(ev.get("r", [0.0] * self.num_envs), dtype=np.float32),
                "l": np.asarray(ev.get("l", [0.0] * self.num_envs), dtype=np.float32),
            }
        return obs, reward, terminated, truncated, info


class _FakeCollector:
    def __init__(self, env, policy):
        self.env = env
        self.policy = policy
        self._random_seed = 0
        self._env_is_reset = False


class CalculateSuccessRateTest(unittest.TestCase):
    def test_success_is_fraction_of_terminated_over_finished_episodes(self):
        num_envs = 2
        F, T = False, True
        events = [
            {},                                                            # step 0: nothing finishes
            {},                                                            # step 1
            {"terminated": [T, F], "r": [10.0, 0.0], "l": [3.0, 0.0]},     # step 2: env0 success
            {"truncated":  [F, T], "r": [0.0, 5.0],  "l": [0.0, 4.0]},     # step 3: env1 timeout (fail)
            {},                                                            # step 4
            {"terminated": [T, F], "r": [12.0, 0.0], "l": [3.0, 0.0]},     # step 5: env0 success
        ]
        env = _ScriptedVectorEnv(num_envs, obs_dim=4, events=events)
        collector = _FakeCollector(env, _FakePolicy(num_envs))

        # calculate_success_rate loops max_steps // num_envs iterations.
        success_rate, episode_return, episode_length = calculate_success_rate(
            collector=collector,
            max_steps=num_envs * len(events),
            exploration_mode=ExplorationMode.MODE,
            reset_before_rollout=True,
        )

        self.assertAlmostEqual(success_rate, 2 / 3, places=6)
        self.assertAlmostEqual(episode_return, (10.0 + 5.0 + 12.0) / 3, places=4)
        self.assertAlmostEqual(episode_length, (3.0 + 4.0 + 3.0) / 3, places=4)

    def test_no_finished_episodes_returns_zeros(self):
        num_envs = 2
        env = _ScriptedVectorEnv(num_envs, obs_dim=4, events=[{}, {}])
        collector = _FakeCollector(env, _FakePolicy(num_envs))

        success_rate, episode_return, episode_length = calculate_success_rate(
            collector=collector,
            max_steps=num_envs * 2,
            exploration_mode=ExplorationMode.MODE,
            reset_before_rollout=True,
        )

        self.assertEqual(success_rate, 0.0)
        self.assertEqual(episode_return, 0.0)
        self.assertEqual(episode_length, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
