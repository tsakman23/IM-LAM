from typing import Callable, Optional

import gymnasium as gym
import numpy as np
from gymnasium.vector import VectorEnv

from ifo.common.env.wrapper import DiscretizeMineRLActionWrapper, NormalizeReward
from ifo.common.utils.functions import Clip, center


def atari(env: VectorEnv) -> VectorEnv:
    """Apply environment wrappers for Atari environment."""
    env = gym.wrappers.vector.DtypeObservation(env, dtype=np.float32)
    env = gym.wrappers.vector.TransformObservation(env, center)
    return env


def procgen(env: VectorEnv) -> VectorEnv:
    """Apply environment wrappers for Procgen environment.

    Args:
        env (VectorEnv): Vectorized Procgen environment.

    Returns:
        VectorEnv: Wrapped Procgen environment.
    """
    env = NormalizeReward(env, gamma=0.99)
    env = gym.wrappers.vector.TransformReward(env, Clip(-10, 10))
    env = gym.wrappers.vector.DtypeObservation(env, dtype=np.float32)
    env = gym.wrappers.vector.TransformObservation(env, center)
    return env


def minigrid(env: VectorEnv) -> VectorEnv:
    """Apply environment wrappers for Minigrid environment.

    Args:
        env (VectorEnv): Vectorized Minigrid environment.

    Returns:
        VectorEnv: Wrapped Minigrid environment.
    """
    env = NormalizeReward(env, gamma=0.99)
    env = gym.wrappers.vector.TransformReward(env, Clip(-10, 10))
    env = gym.wrappers.vector.DtypeObservation(env, dtype=np.float32)
    env = gym.wrappers.vector.TransformObservation(env, center)
    return env


def vizdoom(env: VectorEnv) -> VectorEnv:
    """Apply environment wrappers for Vizdoom environment.

    Args:
        env (VectorEnv): Vectorized Vizdoom environment.

    Returns:
        VectorEnv: Wrapped Vizdoom environment.
    """
    env = NormalizeReward(env, gamma=0.99)
    env = gym.wrappers.vector.TransformReward(env, Clip(-10, 10))
    env = gym.wrappers.vector.DtypeObservation(env, dtype=np.float32)
    env = gym.wrappers.vector.TransformObservation(env, center)
    return env


def minerl(env: VectorEnv) -> VectorEnv:
    """Apply environment wrappers for MineRL environment.

    Args:
        env (VectorEnv): Vectorized MineRL environment.

    Returns:
        VectorEnv: Wrapped MineRL environment.
    """
    env = NormalizeReward(env, gamma=0.99)
    env = gym.wrappers.vector.TransformReward(env, Clip(-10, 10))
    env = gym.wrappers.vector.ResizeObservation(env, (320, 180))
    env.single_observation_space = gym.spaces.Box(
        low=env.observation_space.low[0],
        high=env.observation_space.high[0],
        shape=env.observation_space.shape[1:],
        dtype=env.observation_space.dtype,
    )
    env = gym.wrappers.vector.DtypeObservation(env, dtype=np.float32)
    env = gym.wrappers.vector.TransformObservation(env, center)
    env = DiscretizeMineRLActionWrapper(env)
    return env


def dmc(env: VectorEnv) -> VectorEnv:
    """Apply environment wrappers for DMC environment.

    Args:
        env (VectorEnv): Vectorized DMC environment.

    Returns:
        VectorEnv: Wrapped DMC environment.
    """
    env = gym.wrappers.vector.ClipAction(env)
    env = gym.wrappers.vector.DtypeObservation(env, dtype=np.float32)
    env = gym.wrappers.vector.TransformObservation(env, center)
    return env


def dmc_distracting(env: VectorEnv) -> VectorEnv:
    """Apply environment wrappers for Distracting DMC environment.

    Args:
        env (VectorEnv): Vectorized Distracting DMC environment.

    Returns:
        VectorEnv: Wrapped Distracting DMC environment.
    """
    env = gym.wrappers.vector.ClipAction(env)
    env = gym.wrappers.vector.DtypeObservation(env, dtype=np.float32)
    env = gym.wrappers.vector.TransformObservation(env, center)
    return env


def metaworld(env: VectorEnv) -> VectorEnv:
    """Apply environment wrappers for Meta-World environment.

    Args:
        env (VectorEnv): Vectorized Meta-World environment.

    Returns:
        VectorEnv: Wrapped Meta-World environment.
    """
    env = gym.wrappers.vector.DtypeObservation(env, dtype=np.float32)
    env = gym.wrappers.vector.TransformObservation(env, center)
    return env

def distracted_metaworld(env: VectorEnv) -> VectorEnv:
    """Apply environment wrappers for Distracted Meta-World environment.

    Args:
        env (VectorEnv): Vectorized Distracted Meta-World environment.

    Returns:
        VectorEnv: Wrapped Distracted Meta-World environment.
    """
    env = gym.wrappers.vector.DtypeObservation(env, dtype=np.float32)
    env = gym.wrappers.vector.TransformObservation(env, center)
    return env

def get_env_transform(env_name: Optional[str]) -> Optional[Callable]:
    """Get environment transformations for the specified environment.

    Args:
        env_name (str): Name of the environment.

    Returns:
        Optional[Callable]: Transformations for the specified environment.
    """
    if env_name is None:
        return None
    if "ALE" in env_name:
        return atari
    if "procgen" in env_name:
        return procgen
    elif "MiniGrid" in env_name:
        return minigrid
    elif "Vizdoom" in env_name:
        return vizdoom
    elif "MineRL" in env_name:
        return minerl
    elif "dm_control" in env_name:
        if "distracting" in env_name or "masked" in env_name:
            return dmc_distracting
        else:
            return dmc
    elif "Meta-World" in env_name:
        if "distracting" in env_name or "masked" in env_name:
            return distracted_metaworld
        else:
            return metaworld
    else:
        raise NotImplementedError("Environment is currently not supported.")
