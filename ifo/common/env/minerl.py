from ifo.common.utils.imports import _IS_MINERL_AVAILABLE

if not _IS_MINERL_AVAILABLE:
    raise ModuleNotFoundError(_IS_MINERL_AVAILABLE)

from typing import Any, List, Optional, Sequence, SupportsFloat

import gym
import gymnasium
from gymnasium.core import WrapperActType, WrapperObsType
from gymnasium.utils.step_api_compatibility import step_api_compatibility
from minerl.env import _singleagent
from minerl.herobraine.env_specs.basalt_specs import MAKE_HOUSE_VILLAGE_INVENTORY, BasaltBaseEnvSpec
from minerl.herobraine.env_specs.human_controls import HumanControlEnvSpec
from minerl.herobraine.env_specs.treechop_specs import TREECHOP_WORLD_GENERATOR_OPTIONS
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero.mc import ALL_ITEMS

MINUTE = 20 * 60
BASALT_GYMNASIUM_ENTRYPOINT = "ifo.common.env.minerl:_basalt_gymnasium_entrypoint"


class DoneOnESCWrapper(gymnasium.Wrapper):
    """
    Use the "ESC" action of the MineRL 1.0.0 to end
    an episode (if 1, step will return done=True)
    """

    def __init__(self, env):
        super().__init__(env)
        self.episode_over = False

    def reset(self, **kwargs) -> tuple[WrapperObsType, dict[str, Any]]:
        self.episode_over = False
        return self.env.reset(**kwargs)

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.episode_over:
            raise RuntimeError("Expected `reset` after episode terminated, not `step`.")
        observation, reward, terminated, truncated, info = self.env.step(action)
        terminated = terminated or bool(action["ESC"])
        self.episode_over = terminated
        return observation, reward, terminated, truncated, info


class GymnasiumCompatibility(gymnasium.Wrapper):
    """A wrapper which can transform an environment from old step API to new.

    New step API refers to step() method returning (observation, reward, terminated, truncated, info)
    """

    def __init__(self, env):
        """Wraps an environment to allow a modular transformation of the :meth:`step` and :meth:`reset` methods.

        Args:
            env: The environment to wrap
        """
        self.env = env
        self._action_space = self._convert_space(env.action_space)
        self._observation_space = self._convert_space(env.observation_space)
        self._metadata = env.metadata

        self._cached_spec = env.spec

        self.is_vector_env = isinstance(env.unwrapped, gymnasium.vector.VectorEnv)
        self._render_mode = "rgb_array"

    def _convert_space(self, space: gym.spaces.Space) -> gymnasium.spaces.Space:
        """Converts a gym.spaces.Space to a gymnasium.spaces.Space.

        Args:
            space (gym.spaces.Space): The gym space to convert.

        Raises:
            ValueError: Raised if the space type is not supported.
        Returns:
            gymnasium.spaces.Space: The converted space.
        """
        if isinstance(space, gym.spaces.Dict):
            return gymnasium.spaces.Dict({key: self._convert_space(value) for key, value in space.spaces.items()})
        elif isinstance(space, gym.spaces.Tuple):
            return gymnasium.spaces.Tuple([self._convert_space(subspace) for subspace in space.spaces])
        elif isinstance(space, gym.spaces.MultiDiscrete):
            return gymnasium.spaces.MultiDiscrete(space.nvec)
        elif isinstance(space, gym.spaces.MultiBinary):
            return gymnasium.spaces.MultiBinary(space.n)
        elif isinstance(space, gym.spaces.Discrete):
            return gymnasium.spaces.Discrete(space.n)
        elif isinstance(space, gym.spaces.Box):
            return gymnasium.spaces.Box(low=space.low, high=space.high, dtype=space.dtype)
        elif isinstance(space, gym.spaces.Space):
            return space
        else:
            raise ValueError(f"Unsupported space type: {type(space)}")

    @property
    def render_mode(self) -> str | None:
        """The render mode of the environment.

        Returns:
            str | None: The render mode of the environment.
        """
        return self._render_mode

    def step(self, action: WrapperActType) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return step_api_compatibility(self.env.step(action))

    def reset(self, seed: int | None = None, **kwargs) -> tuple[WrapperObsType, dict[str, Any]]:
        observation = self.env.reset()
        self.env.seed(seed)
        self.action_space.seed(seed)
        self.observation_space.seed(seed)
        return observation, {}


def _basalt_gymnasium_entrypoint(env_spec: "ExtendedBasaltBaseEnvSpec", **kwargs) -> _singleagent._SingleAgentEnv:
    """Used as entrypoint for `gym.make`."""
    envs_without_esc_wrapper = (TreeChopEnvSpec, ObtainEnvSpec)
    env = _singleagent._SingleAgentEnv(env_spec=env_spec)
    env = GymnasiumCompatibility(env)
    if not isinstance(env_spec, envs_without_esc_wrapper):
        env = DoneOnESCWrapper(env)
    return env


class ExtendedBasaltBaseEnvSpec(BasaltBaseEnvSpec):
    """Extended version of the BasaltBaseEnvSpec that allows for more customization
    and is compatible with the new Gymnasium API.
    """

    def __init__(
        self,
        name: str,
        demo_server_experiment_name: str,
        max_episode_steps=2400,
        inventory: Sequence[dict] = (),
        preferred_spawn_biome: str = "plains",
        fov_range: Sequence[int] = (70, 70),
        resolution: Sequence[int] = (640, 320),
        gamma_range: Sequence[int] = (2, 2),
        guiscale_range: Sequence[int] = (1, 1),
        cursor_size_range: Sequence[float] = (16.0, 16.0),
        reward_threshold: Optional[float] = None,
    ) -> None:
        """Instantiate ExtendedBasaltBaseEnvSpec.

        Args:
            name (str): The name of the environment.
            demo_server_experiment_name (str): The name of the experiment on the demo server.
            max_episode_steps (int, optional): The maximum number of steps before the environment is reset.
            inventory (Sequence[dict], optional): The inventory the agent starts with.
            preferred_spawn_biome (str, optional): The preferred biome to spawn in.
            fov_range (Sequence[int], optional): The range of the field of view.
            resolution (Sequence[int], optional): The resolution of the observation.
            gamma_range (Sequence[int], optional): The gamma range of the observation.
            guiscale_range (Sequence[int], optional): The GUI scale range of the observation.
            cursor_size_range (Sequence[float], optional): The cursor size range of the observation.
            reward_threshold (float, optional): The reward threshold of the environment. If None, no threshold is set.
        """
        self.inventory = inventory
        self.preferred_spawn_biome = preferred_spawn_biome
        self.demo_server_experiment_name = demo_server_experiment_name
        self.reward_threshold = reward_threshold
        HumanControlEnvSpec.__init__(
            self,
            name=name,
            max_episode_steps=max_episode_steps,
            fov_range=fov_range,
            resolution=resolution,
            gamma_range=gamma_range,
            guiscale_range=guiscale_range,
            cursor_size_range=cursor_size_range,
        )

    def _entry_point(self) -> str:
        return BASALT_GYMNASIUM_ENTRYPOINT

    def create_observables(self):
        observables = [handlers.ObservationFromCurrentLocation(), handlers.ObservationFromLifeStats()]
        return super().create_observables() + observables

    def register(self):
        reg_spec = dict(
            id=self.name,
            entry_point=self._entry_point(),
            kwargs=self._env_kwargs(),
            max_episode_steps=self.max_episode_steps,
        )
        if self.reward_threshold:
            reg_spec.update(dict(reward_threshold=self.reward_threshold))

        gymnasium.register(**reg_spec)


class FindCaveEnvSpec(ExtendedBasaltBaseEnvSpec):
    """
    After spawning in a plains biome, explore and find a cave. When inside a cave, end
    the episode by setting the "ESC" action to 1.

    You are not allowed to dig down from the surface to find a cave.
    """

    def __init__(self, **kwargs) -> None:
        """Instantiate FindCaveEnvSpec."""
        super().__init__(
            name="MineRLBasaltFindCave-v0",
            demo_server_experiment_name="findcaves",
            max_episode_steps=3 * MINUTE,
            preferred_spawn_biome="plains",
            inventory=[],
            **kwargs,
        )


class MakeWaterfallEnvSpec(ExtendedBasaltBaseEnvSpec):
    """
    After spawning in an extreme hills biome, use your waterbucket to make a beautiful waterfall.
    Then take an aesthetic "picture" of it by moving to a good location, positioning
    player's camera to have a nice view of the waterfall, and ending the episode by
    setting "ESC" action to 1.
    """

    def __init__(self, **kwargs) -> None:
        """Instantiate MakeWaterfallEnvSpec."""
        super().__init__(
            name="MineRLBasaltMakeWaterfall-v0",
            demo_server_experiment_name="waterfall",
            max_episode_steps=5 * MINUTE,
            preferred_spawn_biome="extreme_hills",
            inventory=[
                dict(type="water_bucket", quantity=1),
                dict(type="cobblestone", quantity=20),
                dict(type="stone_shovel", quantity=1),
                dict(type="stone_pickaxe", quantity=1),
            ],
            **kwargs,
        )


class PenAnimalsVillageEnvSpec(ExtendedBasaltBaseEnvSpec):
    """
    After spawning in a village, build an animal pen next to one of the houses in a village.
    Use your fence posts to build one animal pen that contains at least two of the same animal.
    (You are only allowed to pen chickens, cows, pigs, or sheep.)
    There should be at least one gate that allows players to enter and exit easily.
    The animal pen should not contain more than one type of animal.
    (You may kill any extra types of animals that accidentally got into the pen.)

    Do not harm villagers or existing village structures in the process.

    Send 1 for "ESC" key to end the episode.
    """

    def __init__(self, **kwargs):
        """Instantiate PenAnimalsVillageEnvSpec."""
        super().__init__(
            name="MineRLBasaltCreateVillageAnimalPen-v0",
            demo_server_experiment_name="village_pen_animals",
            max_episode_steps=5 * MINUTE,
            preferred_spawn_biome="plains",
            inventory=[
                dict(type="oak_fence", quantity=64),
                dict(type="oak_fence_gate", quantity=64),
                dict(type="carrot", quantity=1),
                dict(type="wheat_seeds", quantity=1),
                dict(type="wheat", quantity=1),
            ],
            **kwargs,
        )

    def create_agent_start(self) -> List[handlers.Handler]:
        return super().create_agent_start() + [handlers.SpawnInVillage()]


class VillageMakeHouseEnvSpec(ExtendedBasaltBaseEnvSpec):
    """
    Build a house in the style of the village without damaging the village. It
    should be in an appropriate location  (e.g. next to the path through the village)
    Then, give a brief tour of the house (i.e. spin around slowly such that all of the
    walls and the roof are visible).
    Finally, end the episode by setting the "ESC" action to 1.


    Tip:
        You can find detailed information on which materials are used in each biome-specific
        village (plains, savannah, taiga, desert) here:
        https://minecraft.wiki/w/Village/Structure/Blueprints
    """

    def __init__(self, **kwargs):
        """Instantiate VillageMakeHouseEnvSpec."""
        super().__init__(
            name="MineRLBasaltBuildVillageHouse-v0",
            demo_server_experiment_name="village_make_house",
            max_episode_steps=12 * MINUTE,
            preferred_spawn_biome="plains",
            inventory=MAKE_HOUSE_VILLAGE_INVENTORY,
            **kwargs,
        )

    def create_agent_start(self) -> List[handlers.Handler]:
        return super().create_agent_start() + [handlers.SpawnInVillage()]


class TreeChopEnvSpec(ExtendedBasaltBaseEnvSpec):
    """
    In treechop, the agent must collect 64 `minercaft:log`. This replicates a common scenario in Minecraft,
    as logs are necessary to craft a large amount of items in the game, and are a key resource in Minecraft.

    The agent begins in a forest biome (near many trees) with an iron axe for cutting trees.
    The agent is given +1 reward for obtaining each unit of wood, and the episode terminates
    once the agent obtains 64 units.
    """

    def __init__(self, **kwargs):
        """Instantiate TreeChopEnvSpec."""
        super().__init__(
            name="MineRLTreechop-v0",
            demo_server_experiment_name="survivaltreechop",
            max_episode_steps=7 * MINUTE,
            preferred_spawn_biome="plains",
            inventory=[dict(type="iron_axe", quantity=1)],
            reward_threshold=64,
            **kwargs,
        )

    def create_rewardables(self) -> List[handlers.Handler]:
        return [handlers.RewardForCollectingItems([dict(type="log", amount=1, reward=1.0)])]

    def create_agent_handlers(self) -> List[handlers.Handler]:
        return [handlers.AgentQuitFromPossessingItem([dict(type="log", amount=64)])]

    def create_server_world_generators(self) -> List[handlers.Handler]:
        return [handlers.DefaultWorldGenerator(force_reset="true", generator_options=TREECHOP_WORLD_GENERATOR_OPTIONS)]

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) >= self.reward_threshold


class ObtainEnvSpec(ExtendedBasaltBaseEnvSpec):
    """
    In this environment the agent is required to obtain a a target item.
    The agent begins in a random starting location on a random survival map
    without any items, matching the normal starting conditions for human players in Minecraft.

    During an episode the agent is rewarded according to the requisite item
    hierarchy needed to obtain a diamond shovel. The rewards for each item are
    given here::

        - Item reward="1" type="log"
        - Item reward="2" type="planks"
        - Item reward="4" type="stick"
        - Item reward="4" type="crafting_table"
        - Item reward="8" type="wooden_pickaxe"
        - Item reward="16" type="cobblestone"
        - Item reward="32" type="furnace"
        - Item reward="32" type="stone_pickaxe"
        - Item reward="64" type="iron_ore"
        - Item reward="128" type="iron_ingot"
        - Item reward="256" type="iron_pickaxe"
        - Item reward="1024" type="diamond"
        - Item reward="2048" type="diamond_shovel"
    """

    dense = True
    full_reward_schedule = [
        dict(type="log", amount=1, reward=1),
        dict(type="planks", amount=1, reward=2),
        dict(type="stick", amount=1, reward=4),
        dict(type="crafting_table", amount=1, reward=4),
        dict(type="wooden_pickaxe", amount=1, reward=8),
        dict(type="cobblestone", amount=1, reward=16),
        dict(type="furnace", amount=1, reward=32),
        dict(type="stone_pickaxe", amount=1, reward=32),
        dict(type="iron_ore", amount=1, reward=64),
        dict(type="iron_ingot", amount=1, reward=128),
        dict(type="iron_pickaxe", amount=1, reward=256),
        dict(type="diamond", amount=1, reward=1024),
        dict(type="diamond_shovel", amount=1, reward=2048),
        dict(type="diamond", amount=1, reward=4096),
        dict(type="diamond_pickaxe", amount=1, reward=8192),
    ]

    def __init__(self, **kwargs):
        self.target_item = kwargs.pop("target_item", "diamond")
        # Make sure the target item is in the reward schedule.
        possible_target_items = {v["type"] for v in self.full_reward_schedule}
        assert (
            self.target_item in possible_target_items
        ), f"Target item {self.target_item} not supported, must be one of {possible_target_items}"
        camel_case_target_item = self.snake_case_to_camel_case(self.target_item)
        target_item_without_underscore = self.target_item.replace("_", "")
        self.reward_schedule = []
        # Create the reward schedule up (including) to the target item.
        for reward in self.full_reward_schedule:
            self.reward_schedule.append(reward)
            if reward["type"] == self.target_item:
                break

        super().__init__(
            name=f"MineRLObtain{camel_case_target_item}-v0",
            demo_server_experiment_name=f"survivalobtain{target_item_without_underscore}",
            max_episode_steps=30 * MINUTE,
            **kwargs,
        )

    def snake_case_to_camel_case(self, snake_str: str) -> str:
        """Converts a snake_case string to a CamelCase string.

        For example "snake_case" -> "SnakeCase".

        Args:
            snake_str (str): The snake_case string to convert.

        Returns:
            str: The CamelCase string.
        """
        return "".join(word.title() for word in snake_str.split("_"))

    def determine_success_from_rewards(self, rewards: list) -> bool:
        rewards = set(rewards)
        allow_missing_ratio = 0.1
        max_missing = round(len(self.reward_schedule) * allow_missing_ratio)

        # Get a list of the rewards from the reward_schedule.
        reward_values = [s["reward"] for s in self.reward_schedule]

        return len(rewards.intersection(reward_values)) >= len(reward_values) - max_missing

    def create_rewardables(self) -> List[handlers.Handler]:
        reward_handler = handlers.RewardForCollectingItems if self.dense else handlers.RewardForCollectingItemsOnce

        return [reward_handler(self.reward_schedule if self.reward_schedule else {self.target_item: 1})]

    def create_agent_handlers(self) -> List[handlers.Handler]:
        return [handlers.AgentQuitFromPossessingItem([dict(type=self.target_item, amount=1)])]

    def create_observables(self) -> List[handlers.Handler]:
        observables = [
            handlers.FlatInventoryObservation(ALL_ITEMS),
            handlers.EquippedItemObservation(ALL_ITEMS, mainhand=True, offhand=True, _default="air", _other="air"),
        ]
        return super().create_observables() + observables


def register_minerl_envs() -> None:
    """Register MineRL environments."""
    try:
        import minerl
    except ImportError:
        raise ImportError("Please install the MineRL environment using `pip install minerl`.")

    env_specs = [
        FindCaveEnvSpec,
        MakeWaterfallEnvSpec,
        PenAnimalsVillageEnvSpec,
        VillageMakeHouseEnvSpec,
        TreeChopEnvSpec,
        ObtainEnvSpec,  # Defaults to diamond.
    ]

    for env_spec in env_specs:
        env_spec(resolution=(640, 360)).register()
