"""Environment wrappers for evaluation."""

from collections import deque
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np


class FrameStackWrapper(gym.Wrapper):
    """Stack the last k frames as observation."""

    def __init__(self, env: gym.Env, num_stack: int = 3) -> None:
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = deque(maxlen=num_stack)

        obs_space = env.observation_space
        low = np.repeat(obs_space.low, num_stack, axis=0)
        high = np.repeat(obs_space.high, num_stack, axis=0)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=obs_space.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.concatenate(list(self.frames), axis=0)


class TransposeObservation(gym.ObservationWrapper):
    """Transpose observation from HWC to CHW format."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(obs_shape[2], obs_shape[0], obs_shape[1]),
            dtype=np.uint8,
        )

    def observation(self, obs):
        return np.transpose(obs, (2, 0, 1))


class NormalizeObservation(gym.ObservationWrapper):
    """Normalize observations to [-0.5, 0.5]."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-0.5, high=0.5,
            shape=self.observation_space.shape,
            dtype=np.float32,
        )

    def observation(self, obs):
        return obs.astype(np.float32) / 255.0 - 0.5


class ClipAction(gym.ActionWrapper):
    """Clip actions to action space bounds."""

    def action(self, action):
        return np.clip(action, self.action_space.low, self.action_space.high)


class VideoRecorder(gym.Wrapper):
    """Record video frames for wandb logging."""

    def __init__(self, env: gym.Env, record_every: int = 1) -> None:
        super().__init__(env)
        self.record_every = record_every
        self.episode_count = 0
        self.frames = []
        self.recording = False

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode_count += 1
        self.recording = (self.episode_count % self.record_every == 0)
        self.frames = []
        if self.recording:
            frame = self.env.render()
            if frame is not None:
                self.frames.append(frame)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if self.recording:
            frame = self.env.render()
            if frame is not None:
                self.frames.append(frame)
        return obs, reward, terminated, truncated, info

    def get_video(self) -> Optional[np.ndarray]:
        """Get recorded frames as (T, H, W, C) array."""
        if self.frames:
            return np.stack(self.frames)
        return None
