"""Meta-World environment creation for evaluation."""

from typing import Optional

import gymnasium as gym
import numpy as np

from src.envs.wrappers import (
    ClipAction,
    FrameStackWrapper,
    NormalizeObservation,
    TransposeObservation,
    VideoRecorder,
)


class MetaWorldImageWrapper(gym.ObservationWrapper):
    """Convert Meta-World state observation to rendered image observation."""

    def __init__(self, env: gym.Env, image_size: int = 128) -> None:
        super().__init__(env)
        self.image_size = image_size
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(image_size, image_size, 3),
            dtype=np.uint8,
        )

    def observation(self, obs):
        img = self.env.render()
        if img is None:
            return np.zeros(self.observation_space.shape, dtype=np.uint8)
        return img


def make_dmw_env(
    task_name: str = "hammer-v3",
    image_size: int = 128,
    frame_stack: int = 3,
    seed: int = 0,
    record_video: bool = False,
    record_every: int = 10,
) -> gym.Env:
    """Create a Meta-World environment for evaluation.

    Requires metaworld to be installed.

    Args:
        task_name: Meta-World task (hammer-v3, bin-picking-v3, etc.).
        image_size: rendered image size.
        frame_stack: number of frames to stack.
        seed: random seed.
        record_video: whether to wrap with VideoRecorder.
        record_every: record every N episodes.

    Returns:
        Wrapped gymnasium environment.
    """
    try:
        import metaworld
    except ImportError:
        raise ImportError(
            "metaworld required for DMW envs. "
            "Install with: pip install git+https://github.com/Farama-Foundation/Metaworld.git"
        )

    # Create Meta-World environment
    ml1 = metaworld.ML1(task_name, seed=seed)
    env_cls = ml1.train_classes[task_name]
    env = env_cls(render_mode="rgb_array")
    task = ml1.train_tasks[0]
    env.set_task(task)

    # Image rendering wrapper
    env = MetaWorldImageWrapper(env, image_size=image_size)

    # Standard wrappers
    env = TransposeObservation(env)
    env = NormalizeObservation(env)
    env = FrameStackWrapper(env, num_stack=frame_stack)
    env = ClipAction(env)

    if record_video:
        env = VideoRecorder(env, record_every=record_every)

    return env
