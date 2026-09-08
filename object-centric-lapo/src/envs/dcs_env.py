"""Distracting Control Suite environment creation for evaluation."""

from typing import Optional

import gymnasium as gym

from src.envs.wrappers import (
    ClipAction,
    FrameStackWrapper,
    NormalizeObservation,
    TransposeObservation,
    VideoRecorder,
)


def make_dcs_env(
    domain_name: str = "cheetah",
    task_name: str = "run",
    image_size: int = 64,
    frame_stack: int = 3,
    seed: int = 0,
    record_video: bool = False,
    record_every: int = 10,
) -> gym.Env:
    """Create a DCS environment for evaluation.

    Requires dm_control, mujoco, and shimmy to be installed.

    Args:
        domain_name: DM Control domain (cheetah, walker, hopper, humanoid).
        task_name: DM Control task (run, hop, walk).
        image_size: rendered image size.
        frame_stack: number of frames to stack.
        seed: random seed.
        record_video: whether to wrap with VideoRecorder.
        record_every: record every N episodes.

    Returns:
        Wrapped gymnasium environment.
    """
    try:
        from dm_control import suite
        from shimmy import DmControlCompatibilityV0
    except ImportError:
        raise ImportError(
            "dm_control and shimmy required for DCS envs. "
            "Install with: pip install dm_control shimmy mujoco"
        )

    # Create base dm_control env
    dm_env = suite.load(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs={"random": seed},
    )

    # Wrap for gymnasium compatibility
    env = DmControlCompatibilityV0(dm_env, render_mode="rgb_array")

    # Add pixel rendering
    env = gym.wrappers.AddRenderObservation(
        env,
        render_only=True,
        render_kwargs={"width": image_size, "height": image_size},
    )

    # Standard wrappers
    env = TransposeObservation(env)
    env = NormalizeObservation(env)
    env = FrameStackWrapper(env, num_stack=frame_stack)
    env = ClipAction(env)

    if record_video:
        env = VideoRecorder(env, record_every=record_every)

    return env
