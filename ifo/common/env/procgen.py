from ifo.common.utils.imports import _IS_PROCGEN_AVAILABLE

if not _IS_PROCGEN_AVAILABLE:
    raise ModuleNotFoundError(_IS_PROCGEN_AVAILABLE)

from copy import deepcopy
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from gym3 import types_np
from gym3.types import Discrete, Real, TensorType, ValType, multimap
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, RenderFrame
from gymnasium.vector import AutoresetMode, VectorEnv
from gymnasium.vector.utils import batch_space
from gymnasium.vector.vector_env import ArrayType
from procgen.env import ProcgenGym3Env

from ifo.common.utils.render import tile_images


def gym3_to_gymnasium(vt: ValType) -> spaces.Space:
    """
    Convert a gym3 ValType to a gymnasium Space.

    Args:
        vt (ValType): The gym3 ValType.

    Returns:
        spaces.Space: The gymnasium Space.
    """

    def tt2space(tt: TensorType) -> spaces.Space:
        """
        Convert a gym3 TensorType to a gymnasium Space.

        Args:
            tt (TensorType): The gym3 TensorType.

        Returns:
            spaces.Space: The gymnasium Space.
        """
        if isinstance(tt.eltype, Discrete):
            if tt.ndim == 0:
                return spaces.Discrete(tt.eltype.n)
            else:
                return spaces.Box(
                    low=0,
                    high=tt.eltype.n - 1,
                    shape=tt.shape,
                    dtype=types_np.dtype(tt),
                )
        elif isinstance(tt.eltype, Real):
            return spaces.Box(
                shape=tt.shape,
                dtype=types_np.dtype(tt),
                low=float("-inf"),
                high=float("inf"),
            )
        else:
            raise NotImplementedError

    space = multimap(tt2space, vt)

    def dict2dict_space(d: Dict[str, Any]) -> spaces.Space:
        """
        Convert a dictionary to a gymnasium Dict space.

        Args:
            d (Dict[str, Any]): The dictionary.

        Returns:
            spaces.Space: The gymnasium Dict space.
        """
        if isinstance(d, dict):
            return spaces.Dict({k: dict2dict_space(v) for k, v in d.items()})
        else:
            return d

    return dict2dict_space(space)


class ProcgenEnv(VectorEnv):
    """
    Create a Gymnasium VecEnv environment from a gym3 environment.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15, "autoreset_mode": AutoresetMode.NEXT_STEP}

    def __init__(self, env_name: str, num_envs: int = 1, **kwargs) -> None:
        """
        Initialize the ProcgenEnv.

        Args:
            env_name (str): The name of the Procgen environment.
            num_envs (int): The number of environments to create.
            **kwargs: Additional keyword arguments.
        """
        self.env_name = env_name
        self.num_envs = num_envs
        self.env_kwargs = kwargs.copy()

        self.env = ProcgenGym3Env(num_envs, env_name, **kwargs)
        self.single_observation_space = gym3_to_gymnasium(self.env.ob_space)["rgb"]
        self.single_action_space = gym3_to_gymnasium(self.env.ac_space)
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.render_mode = "rgb_array"
        self._prev_observations = None
        self._prev_infos = None
        self._dones = np.zeros(self.num_envs, dtype=bool)

    def reset(
        self,
        *,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        Reset the environment.

        Args:
            seed (int | list[int] | None): The seed for the environment.
            options (dict[str, Any] | None): Additional options.

        Returns:
            Tuple[ObsType, dict[str, Any]]: The observation and information dictionary.
        """
        # Handle seeding by recreating the environment if a seed is provided
        if seed is not None:
            if isinstance(seed, list):
                # For vectorized environments, use the first seed
                seed = seed[0] if seed else self._current_seed
                raise Warning("Procgen environments do not support vectorized seeding. Using the first seed.")

            # Always recreate the environment when a seed is explicitly provided
            # Close the old environment
            self.env.close()

            # Create new environment with the new seed
            new_kwargs = self.env_kwargs.copy()
            new_kwargs["start_level"] = seed

            self.env = ProcgenGym3Env(self.num_envs, self.env_name, **new_kwargs)

        # Sending -1 as action causes the environment to reset.
        self.env.act(np.full((self.num_envs,), -1))
        rewards, observations, firsts = self.env.observe()
        observations = observations["rgb"]

        # Warning if the reset is not done on all environments.
        if not firsts.all():
            print("Warning: manual reset ignored")

        # Update the previous observations and dones.
        self._prev_observations = deepcopy(observations)
        self._prev_infos = {}
        self._dones[:] = False
        return observations, {}

    def _convert_info(self, infos: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert the information dictionary from list of dicts to dict of np.ndarrays.

        Args:
            infos (List[Dict[str, Any]]): The list of information dictionaries.

        Returns:
            Dict[str, Any]: The converted information dictionary.
        """
        converted_infos = {}
        for i, info in enumerate(infos):
            converted_infos = self._add_info(converted_infos, info, i)
        return converted_infos

    def step(self, env_actions: ActType) -> tuple[ObsType, ArrayType, ArrayType, ArrayType, dict[str, Any]]:
        """
        Perform a step in the environment.

        Args:
            actions (np.ndarray): The actions to take in the environment.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
                The observations, rewards, terminations, truncations, info dictionary.
        """
        # Ignore actions from ended environments and reset them manually.
        if self._dones.any():
            actions = deepcopy(env_actions)
            actions[self._dones] = -1
        else:
            actions = env_actions

        # Perform the actions.
        self.env.act(actions)
        rewards, observations, firsts = self.env.observe()
        observations = observations["rgb"]
        rewards[self._dones] = 0.0  # Set the reward for the initial observation to 0.0.

        # Replace the first observation for each resetted environment with the last observation.
        # But not those that we manually reset.
        true_firsts = firsts & ~self._dones
        if true_firsts.any():
            observations[true_firsts] = self._prev_observations[true_firsts]

        # Update the previous observations and dones.
        self._prev_observations = deepcopy(observations)
        self._dones = true_firsts

        # Note: As we converted this AutoresetMode.SAME_STEP to AutoresetMode.NEXT_STEP, the info is partially
        # incorrect, since we manually reset the environment again. This means that the prev_level_seed and level_seed
        # is incorrect for a short moment in between resets. But this should be fine, as nobody should use this info.
        info = self._convert_info(self.env.get_info())

        # Get the true terminations and truncations.
        terminations = true_firsts & (info["prev_level_complete"] == 1)
        truncations = true_firsts & (info["prev_level_complete"] == 0)
        return observations, rewards, terminations, truncations, info

    def render(self) -> tuple[RenderFrame, ...] | None:
        """Returns the render mode from the base vector environment."""
        """
        Render the environment.

        Args:
            mode (str): The rendering mode.

        Returns:
            np.ndarray: The rendered image.
        """
        images = []
        if self.render_mode == "rgb_array":
            # If any environment is done, use the previous observation.
            if self._dones.any():
                images = [self._prev_observations[i] for i in range(self.num_envs)]
            else:
                _, observations, _ = self.env.observe()
                observations = observations["rgb"]
                images = [observations[i] for i in range(self.num_envs)]
        else:
            raise ValueError(f"Invalid render mode: {self.render_mode}")
        big_image = tile_images(images)
        return big_image


### Environment Registration ###
ENV_NAMES = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
]


def register_procgen_envs() -> None:
    """Register Procgen environments."""
    for env_name in ENV_NAMES:
        id = f"procgen-{env_name}-v0"
        if id not in gym.envs.registry:
            gym.register(
                id=id,
                entry_point="ifo.common.env.procgen:ProcgenEnv",
                kwargs={"env_name": env_name, "distribution_mode": "easy", "num_levels": 0, "start_level": 0},
            )
