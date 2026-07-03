from ifo.common.utils.imports import _IS_MINIGRID_AVAILABLE

if not _IS_MINIGRID_AVAILABLE:
    raise ModuleNotFoundError(_IS_MINIGRID_AVAILABLE)

import gymnasium as gym
import minigrid


def register_minigrid_envs():
    """Register custom Minigrid environments."""
    minigrid_empty_registry = {
        "MiniGrid-Empty-Random-16x16-v0": {"size": 16, "agent_start_pos": None},
        "MiniGrid-Empty-Random-12x12-v0": {"size": 12, "agent_start_pos": None},
        "MiniGrid-Empty-Random-8x8-v0": {"size": 8, "agent_start_pos": None},
    }

    minigrid_dynamic_registry = {
        "MiniGrid-Dynamic-Obstacles-Random-16x16-v0": {"size": 16, "agent_start_pos": None, "n_obstacles": 5},
        "MiniGrid-Dynamic-Obstacles-Random-12x12-v0": {"size": 12, "agent_start_pos": None, "n_obstacles": 3},
        "MiniGrid-Dynamic-Obstacles-Random-8x8-v0": {"size": 8, "agent_start_pos": None, "n_obstacles": 1},
    }

    for id, kwargs in minigrid_empty_registry.items():
        if id not in gym.envs.registry:
            gym.register(id=id, entry_point="minigrid.envs:EmptyEnv", kwargs=kwargs)

    for id, kwargs in minigrid_dynamic_registry.items():
        if id not in gym.envs.registry:
            gym.register(id=id, entry_point="minigrid.envs:DynamicObstaclesEnv", kwargs=kwargs)
