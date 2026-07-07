import datetime
import uuid
from collections import deque
from copy import deepcopy
from logging import warn
from typing import Any, Dict, Final, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import Env, ObservationWrapper
from gymnasium.vector import AutoresetMode, VectorActionWrapper, VectorEnv, VectorObservationWrapper, VectorWrapper
from gymnasium.vector.utils import batch_space, create_empty_array
from gymnasium.wrappers import FrameStackObservation
from gymnasium.wrappers.utils import RunningMeanStd, create_zero_array


class FilterObservation(ObservationWrapper):
    """
    A wrapper that filters a Dict observation space to only include a specific key.

    Example:
        >>> import gymnasium as gym
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> obs, _ = env.reset()
        >>> obs.keys()
        dict_keys(['image', 'direction', 'mission'])
        >>> env = FilterObservation(env, key='image')
        >>> obs, _ = env.reset()
        >>> obs.shape
        (7, 7, 3)
    """

    def __init__(self, env: Env, key: str):
        """A wrapper that selects a key from a Dict observation space.

        Args:
            env: The environment to apply the wrapper.
            key: The key to select from the observation.
        """
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Dict), "Wrapper requires a Dict observation space."
        self.key = key
        self.observation_space = env.observation_space.spaces[key]

    def observation(self, observation: Dict[str, Any]) -> Any:
        return observation[self.key]


class OriginalWrapper(VectorWrapper):
    """A wrapper that saves the original step output before they are modified by other wrappers.

    The original step output is saved in the info dictionary under the key "original".

    Example:
    >>> import gymnasium as gym
    >>> from gymnasium.vector import SyncVectorEnv
    >>> from env.wrapper import OriginalWrapper
    >>>
    >>> # Create a vector environment
    >>> env = SyncVectorEnv([lambda: gym.make("CartPole-v1")])
    >>>
    >>> # Wrap with OriginalWrapper to preserve original observations and rewards
    >>> wrapped_env = OriginalWrapper(env)
    >>>
    >>> # Reset the environment
    >>> obs, info = wrapped_env.reset()
    >>>
    >>> # Take a step
    >>> action = [0]  # Example action
    >>> next_obs, reward, terminated, truncated, info = wrapped_env.step(action)
    >>>
    >>> # Access the original (unmodified) observation and reward
    >>> original_obs = info["original"]["observation"]
    >>> original_reward = info["original"]["reward"]
    >>>
    >>> # Get original observation space
    >>> original_obs_space = wrapped_env.original_observation_space
    """

    def __init__(self, env: VectorEnv) -> None:
        """Initializes the wrapper.

        Args:
            env (VectorEnv): The environment to wrap.
        """
        super().__init__(env)
        self._original_observation_space = deepcopy(env.observation_space)
        self._original_single_observation_space = deepcopy(env.single_observation_space)
        self._original_action_space = deepcopy(env.action_space)
        self._original_single_action_space = deepcopy(env.single_action_space)

    def reset(self, **kwargs):
        observations, infos = self.env.reset(**kwargs)
        original_step_output = deepcopy({"observation": observations, "info": infos})
        original_step_output["reward"] = np.zeros(self.env.num_envs)
        original_step_output["terminated"] = np.zeros(self.env.num_envs)
        original_step_output["truncated"] = np.zeros(self.env.num_envs)
        infos["original"] = original_step_output
        return observations, infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        infos["original"] = deepcopy(
            {
                "observation": observations,
                "action": actions,
                "reward": rewards,
                "terminated": terminations,
                "truncated": truncations,
                "info": infos,
            }
        )
        return observations, rewards, terminations, truncations, infos

    @property
    def original_observation_space(self):
        return self._original_observation_space

    @property
    def original_single_observation_space(self):
        return self._original_single_observation_space

    @property
    def original_action_space(self):
        return self._original_action_space

    @property
    def original_single_action_space(self):
        return self._original_single_action_space


class NormalizeReward(VectorWrapper, gym.utils.RecordConstructorArgs):
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

    The exponential moving average will have variance :math:`(1 - \gamma)^2`.
    This implementation fixes the gymnasium implementation and adheres to the original gym implementation.

    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """

    def __init__(
        self,
        env: VectorEnv,
        gamma: float = 0.99,
        epsilon: float = 1e-8,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.

        Args:
            env (VectorEnv): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        gym.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        VectorWrapper.__init__(self, env)
        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.env.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        self.returns = self.returns * self.gamma + rewards
        rewards = self.normalize(rewards)
        dones = terminations | truncations
        self.returns[dones] = 0
        return observations, rewards, terminations, truncations, infos

    def normalize(self, rewards: np.ndarray) -> np.ndarray:
        """Normalizes the rewards with the running mean rewards and their variance.

        Args:
            rews (np.ndarray): The rewards to normalize.

        Returns:
            np.ndarray: The normalized rewards.
        """
        self.return_rms.update(self.returns)
        return rewards / np.sqrt(self.return_rms.var + self.epsilon)


class VectorFrameStackObservation(FrameStackObservation, VectorWrapper):
    """A vectorized version of the FrameStackObservation wrapper.

    The wrapper can be used for vectorized environments.
    """

    def __init__(self, env: VectorEnv, stack_size: int, *, padding_type: str = "reset") -> None:
        """Initializes the wrapper.

        Args:
            env (gym.Env): The environment to wrap.
            stack_size: The number of frames to stack.
            padding_type: The padding type to use when stacking the observations, options: "reset", "zero", custom obs
        """
        VectorWrapper.__init__(self, env)
        gym.utils.RecordConstructorArgs.__init__(self, stack_size=stack_size, padding_type=padding_type)

        if not np.issubdtype(type(stack_size), np.integer):
            raise TypeError(f"The stack_size is expected to be an integer, actual type: {type(stack_size)}")
        if not 0 < stack_size:
            raise ValueError(f"The stack_size needs to be greater than zero, actual value: {stack_size}")
        if isinstance(padding_type, str) and (padding_type == "reset" or padding_type == "zero"):
            self.padding_value = create_zero_array(env.observation_space)
        elif padding_type in env.observation_space:
            self.padding_value = padding_type
            padding_type = "_custom"
        else:
            if isinstance(padding_type, str):
                raise ValueError(  # we are guessing that the user just entered the "reset" or "zero" wrong
                    f"Unexpected `padding_type`, expected 'reset', 'zero' or a custom observation space, actual value: {padding_type!r}"
                )
            else:
                raise ValueError(
                    f"Unexpected `padding_type`, expected 'reset', 'zero' or a custom observation space, actual value: {padding_type!r} not an instance of env observation ({env.observation_space})"
                )

        self.observation_space = batch_space(env.observation_space, n=stack_size)
        self.single_observation_space = batch_space(env.single_observation_space, n=stack_size)
        self.stack_size: Final[int] = stack_size
        self.padding_type: Final[str] = padding_type

        self.obs_queue = deque([self.padding_value for _ in range(self.stack_size)], maxlen=self.stack_size)
        self.stacked_obs = create_empty_array(env.observation_space, n=self.stack_size)


class TransposeObservation(VectorObservationWrapper):
    """Transpose the observation space of the environment.

    The wrapper can be used for regular and vectorized environments.
    """

    def __init__(self, env: VectorEnv, permutation: Sequence[int]) -> None:
        """Initializes the wrapper.

        Args:
            env (VectorEnv): The environment to wrap.
            permutation (Sequence[int]): The permutation of the axes.
        """
        super().__init__(env)
        assert isinstance(env, (VectorWrapper, VectorEnv)), "The environment must be a vectorized environment."
        self.permutation = permutation

        # Transpose the observation space.
        assert isinstance(env.observation_space, spaces.Box), "The observation space must be a Box."
        low = np.transpose(env.observation_space.low, permutation)
        high = np.transpose(env.observation_space.high, permutation)
        shape = [env.observation_space.shape[i] for i in permutation]
        self.observation_space = spaces.Box(low=low, high=high, shape=shape, dtype=env.observation_space.dtype)

        # Transpose the single observation space.
        # Make sure all rows of low and high are the same, since the single observation space is the same for all envs.
        assert np.all(low[0] == low), "The rows of the low array must be the same."
        assert np.all(high[0] == high), "The rows of the high array must be the same."
        self.single_observation_space = spaces.Box(
            low=low[0], high=high[0], shape=shape[1:], dtype=env.observation_space.dtype
        )

    def observations(self, observations):
        """Transpose the observations.

        Args:
            observations (np.ndarray): The observations to transpose.

        Returns:
            np.ndarray: The transposed observations.
        """
        return np.transpose(observations, self.permutation)


class DiscretizeMineRLActionWrapper(VectorActionWrapper):
    """Wrapper that transforms MineRL actions to a discrete space and removes senseless actions."""

    def __init__(
        self,
        env: VectorEnv,
        camera_maxval: int = 10,
        camera_binsize: int = 2,
        camera_quantization_scheme: str = "mu_law",
        camera_mu: int = 10,
        n_camera_bins: int = 11,
    ) -> None:
        """
        Initializes DiscretizeMineRLActionWrapper.

        Args:
            env (VectorEnv): Vectorized MineRL environment.
            camera_maxval (int, optional): The maximum value for the camera quantizer.
            camera_binsize (int, optional): The bin size for the camera quantizer.
            camera_quantization_scheme (str, optional): The quantization scheme for the camera quantizer.
            camera_mu (float, optional): The mu parameter for the camera quantizer.
            n_camera_bins (int, optional): The number of camera bins in the factored space.
        """
        super().__init__(env)
        self.transform = DiscretizeMineRLAction(
            camera_maxval=camera_maxval,
            camera_binsize=camera_binsize,
            camera_quantization_scheme=camera_quantization_scheme,
            camera_mu=camera_mu,
            n_camera_bins=n_camera_bins,
        )
        self.single_action_space = self.transform.single_action_space
        self.action_space = batch_space(self.single_action_space, self.num_envs)

    def actions(self, actions):
        return self.transform.reverse(actions)


class UUIDWrapper(VectorWrapper):
    def __init__(self, env: VectorEnv) -> None:
        """Wrapper that adds a unique ID to the environment.

        Args:
            env (VectorEnv): The environment to wrap.
        """
        super().__init__(env)
        if "autoreset_mode" not in env.metadata:
            warn(f"Vector environment ({env}) is missing `autoreset_mode` metadata key.")
        else:
            assert (
                env.metadata["autoreset_mode"] == AutoresetMode.NEXT_STEP
                or env.metadata["autoreset_mode"] == AutoresetMode.DISABLED
            )
        self._ids = np.array([self._get_id() for _ in range(env.num_envs)])
        self._previous_dones = np.zeros(env.num_envs, dtype=bool)

    def _get_id(self) -> str:
        """Get a unique ID for the environment.

        Returns:
            str: A unique ID for the environment.
        """
        return f"{datetime.datetime.now().strftime('%Y%m%dT%H%M%S')}-{str(uuid.uuid4().hex)}"

    def reset(self, **kwargs):
        self._ids = np.array([self._get_id() for _ in range(self.env.num_envs)])
        self._previous_dones = np.zeros(self.env.num_envs, dtype=bool)

        observations, infos = self.env.reset(**kwargs)
        infos["id"] = self._ids.copy()

        return observations, infos

    def step(self, actions):
        if self._previous_dones.any():
            for i in np.where(self._previous_dones)[0].tolist():
                self._ids[i] = self._get_id()

        observations, rewards, terminations, truncations, infos = self.env.step(actions)
        infos["id"] = self._ids.copy()
        self._previous_dones = terminations | truncations

        return observations, rewards, terminations, truncations, infos


class DictObservationWrapper(VectorObservationWrapper):
    """A wrapper that converts observations to a dictionary format, if the observation is not already a dictionary.

    If the observation is not already a dictionary, it creates a dictionary where
    the original observation gets the key 'observation'. Additionally, it can add
    'is_first' and 'is_terminal' to the observation dictionary if enabled.

    Args:
        env (VectorEnv): The environment to wrap.
        add_is_first (bool): Whether to add 'is_first' to the observation. Defaults to False.
        add_is_terminal (bool): Whether to add 'is_terminal' to the observation. Defaults to False.
    """

    def __init__(
        self,
        env: VectorEnv,
        add_is_first: bool = False,
        add_is_terminal: bool = False,
        add_continue: bool = False,
    ):
        super().__init__(env)
        if "autoreset_mode" not in env.metadata:
            warn(f"Vector environment ({env}) is missing `autoreset_mode` metadata key.")
        else:
            assert (
                env.metadata["autoreset_mode"] == AutoresetMode.NEXT_STEP
                or env.metadata["autoreset_mode"] == AutoresetMode.DISABLED
            )

        self.add_is_first = add_is_first
        self.add_is_terminal = add_is_terminal
        self.add_continue = add_continue

        # Track previous dones for is_first calculation
        self._previous_dones = np.zeros(env.num_envs, dtype=bool)
        self._first_step = np.ones(env.num_envs, dtype=bool)

        # Update observation space
        if isinstance(env.single_observation_space, spaces.Dict):
            # Already a dict, keep the same structure
            dict_spaces = env.single_observation_space.spaces.copy()
        else:
            # Convert to dict with 'observation' key
            dict_spaces = {"image": env.single_observation_space}

        # Add additional keys if enabled
        if self.add_is_first:
            dict_spaces["is_first"] = spaces.Box(low=0, high=1, shape=(), dtype=bool)
        if self.add_is_terminal:
            dict_spaces["is_terminal"] = spaces.Box(low=0, high=1, shape=(), dtype=bool)
        if self.add_continue:
            dict_spaces["continue"] = spaces.Box(low=0, high=1, shape=(), dtype=bool)

        self.single_observation_space = spaces.Dict(dict_spaces)
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)

    def observations(self, observations):
        """Convert observations to dictionary format."""
        # Convert to dict if not already
        observations = observations if isinstance(observations, dict) else {"image": observations}

        # Add is_first if enabled
        if self.add_is_first:
            observations["is_first"] = self._first_step.copy()

        # Add is_terminal if enabled (initialize as False, will be updated in step)
        if self.add_is_terminal:
            observations["is_terminal"] = np.zeros(self.num_envs, dtype=bool)

        # Add continue if enabled
        if self.add_continue:
            observations["continue"] = np.zeros(self.num_envs, dtype=bool)

        return observations

    def reset(self, **kwargs):
        """Reset the environment and update tracking variables."""
        observations, infos = self.env.reset(**kwargs)

        # Reset tracking variables
        self._previous_dones = np.zeros(self.num_envs, dtype=bool)
        self._first_step = np.ones(self.num_envs, dtype=bool)

        # Convert observations
        observations = self.observations(observations)

        return observations, infos

    def step(self, actions):
        """Step the environment and update observations with terminal info."""
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        # Update first step tracking
        self._first_step = self._previous_dones.copy()

        # Convert observations
        observations = self.observations(observations)

        # Update is_terminal with current terminations/truncations
        if self.add_is_terminal:
            observations["is_terminal"] = terminations | truncations

        # Update continue with current terminations/truncations
        if self.add_continue:
            observations["continue"] = 1 - observations["is_terminal"]

        # Update previous dones for next step
        self._previous_dones = terminations | truncations

        return observations, rewards, terminations, truncations, infos


class ActionRepeat(gym.Wrapper):
    """Action repeat wrapper that repeats the same action for multiple steps.

    This wrapper executes the same action for a specified number of steps,
    accumulating rewards and stopping early if the episode terminates.

    Args:
        env: The environment to wrap.
        repeat (int): Number of times to repeat each action. Default is 4.
    """

    def __init__(self, env, repeat=4):
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        """Execute the given action for multiple steps.

        Args:
            action: The action to repeat.

        Returns:
            tuple: A tuple containing:
                - obs: The final observation after all repeated steps
                - total_reward (float): Sum of rewards from all repeated steps
                - terminated (bool): Whether the episode terminated
                - truncated (bool): Whether the episode was truncated
                - info (dict): Info from the final step
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for _ in range(self.repeat):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class ClipAction(gym.ActionWrapper):
    """Clips actions to specified bounds.

    This wrapper clips actions to the specified low and high bounds before
    passing them to the environment. This is useful for ensuring actions
    stay within valid ranges.

    Example:
        >>> import gymnasium as gym
        >>> from env.wrapper import ClipAction
        >>>
        >>> # Create an environment
        >>> env = gym.make("Pendulum-v1")
        >>>
        >>> # Wrap with ClipAction to clip actions to [-1, 1]
        >>> wrapped_env = ClipAction(env, low=-1.0, high=1.0)
        >>>
        >>> # Actions will be clipped to the specified bounds
        >>> obs, info = wrapped_env.reset()
        >>> action = 2.0  # This will be clipped to 1.0
        >>> obs, reward, terminated, truncated, info = wrapped_env.step(action)
    """

    def __init__(self, env: gym.Env, low: float = -1.0, high: float = 1.0):
        """Initializes the ClipAction wrapper.

        Args:
            env (gym.Env): The environment to wrap.
            low (float): The lower bound for clipping actions. Default is -1.0.
            high (float): The upper bound for clipping actions. Default is 1.0.
        """
        super().__init__(env)
        self.low = low
        self.high = high

    def action(self, action):
        """Clip action to the specified bounds.

        Args:
            action: The action to clip.

        Returns:
            The clipped action.
        """
        return np.clip(action, self.low, self.high)


class OneHotActionWrapper(VectorActionWrapper):
    """Converts one-hot encoded actions to categorical actions for vectorized environments.

    This wrapper transforms one-hot encoded actions (represented as vectors with a single 1
    and the rest 0s) into categorical action indices. This is useful when working with
    algorithms that output one-hot encoded actions but the environment expects discrete
    action indices.

    The action spaces remain Discrete, but the sample() method is patched to return
    one-hot encoded vectors instead of categorical indices.

    Example:
        >>> import gymnasium as gym
        >>> from gymnasium.vector import SyncVectorEnv
        >>> from env.wrapper import OneHotActionWrapper
        >>>
        >>> # Create a vectorized environment with discrete actions
        >>> env = SyncVectorEnv([lambda: gym.make("CartPole-v1") for _ in range(2)])
        >>>
        >>> # Wrap with OneHotActionWrapper
        >>> wrapped_env = OneHotActionWrapper(env)
        >>>
        >>> # Action space is still Discrete, but sample() returns one-hot vectors
        >>> obs, info = wrapped_env.reset()
        >>> sampled_actions = wrapped_env.action_space.sample()  # Returns one-hot vectors
        >>>
        >>> # Provide one-hot encoded actions
        >>> onehot_actions = np.array([[1, 0], [0, 1]])  # First env: action 0, second env: action 1
        >>> obs, reward, terminated, truncated, info = wrapped_env.step(onehot_actions)
    """

    def __init__(self, env: VectorEnv):
        """Initializes the OneHotActionWrapper.

        Args:
            env (VectorEnv): The vectorized environment to wrap. Must have a Discrete action space.
        """
        super().__init__(env)

        assert isinstance(
            env.single_action_space, spaces.Discrete
        ), "OneHotActionWrapper only supports Discrete action spaces"

        # Patch the sample method of single_action_space
        self._patch_single_sample_method(self.single_action_space)

        # Patch the sample method of action_space (batched)
        self._patch_batched_sample_method(self.action_space, self.num_envs)

    def _patch_single_sample_method(self, action_space):
        """Patch the sample method of a single action space to return one-hot vectors.

        Args:
            action_space: The single action space to patch (Discrete).
        """
        original_sample = action_space.sample

        def sample(mask=None, *, probability=None):
            idx = original_sample(mask, probability=probability)
            one_hot = np.zeros(action_space.n, dtype=np.float32)
            one_hot[idx] = 1.0
            return one_hot

        action_space.sample = sample

    def _patch_batched_sample_method(self, action_space, num_envs):
        """Patch the sample method of a batched action space to return one-hot vectors.

        Args:
            action_space: The batched action space to patch.
            num_envs: Number of parallel environments.
        """
        original_sample = action_space.sample

        def sample(mask=None, *, probability=None):
            indices = original_sample(mask, probability=probability)
            one_hot_batch = np.zeros((num_envs, action_space.nvec[0]), dtype=np.float32)
            one_hot_batch[np.arange(num_envs), indices] = 1.0
            return one_hot_batch

        action_space.sample = sample

    def actions(self, actions):
        """Transform one-hot encoded actions to categorical action indices.

        Args:
            actions (np.ndarray): One-hot encoded actions with shape (num_envs, n_actions).

        Returns:
            np.ndarray: Categorical action indices with shape (num_envs,).
        """
        # Convert one-hot to categorical by finding the index of the maximum value
        # This handles both perfect one-hot encoding and soft distributions
        return np.argmax(actions, axis=-1)


class MetaWorldImageWrapper(gym.ObservationWrapper):
    """Wrapper that converts MetaWorld observations into a Dict with image and state keys.

    MetaWorld environments natively output state-based observations (joint positions,
    object positions, etc.). This wrapper renders images from the MuJoCo simulation
    and returns a Dict observation containing both the rendered image and the original state vector.

    Example:
        >>> import gymnasium as gym
        >>> from env.wrapper import MetaWorldImageWrapper
        >>>
        >>> env = gym.make('Meta-World/MT1', env_name="reach-v3", render_mode="rgb_array", camera_name="corner2", width=84, height=84)
        >>> env = MetaWorldImageWrapper(env)
        >>> obs, info = env.reset()
        >>> obs["image"].shape
        (64, 64, 3)
        >>> obs["state"].shape
        (39,)
    """

    def __init__(
        self,
        env: gym.Env,
        auto_rotate: bool = True,
    ):
        """Initializes the MetaWorldImageWrapper.

        Args:
            env (gym.Env): The environment to wrap.
            auto_rotate (bool, optional): Whether to flip upside down images automatically. Defaults to True.
        """
        super().__init__(env)

        assert env.render_mode == "rgb_array", \
            "MetaWorldImageWrapper only supports environments with render_mode='rgb_array'"

        self._height = env.unwrapped.height
        self._width = env.unwrapped.width
        self._camera_id = env.unwrapped.camera_name
        self._auto_rotate = auto_rotate
        self._rot_180 = True if self._camera_id.startswith("corner") and auto_rotate else False

        # Ensure consistent goal/task randomisation across all three variants
        # (vanilla, distracting, segmentation).  DistractingMetaworldWrapper
        # and SegmentationMetaworldWrapper already set these; setting them here
        # too makes vanilla behave identically.
        # see: https://github.com/Farama-Foundation/Metaworld/pull/370
        env.unwrapped.seeded_rand_vec = True
        env.unwrapped._freeze_rand_vec = False


        # Build a Dict observation space with "image" and "state" keys.
        image_space = spaces.Box(
            low=0, high=255, shape=(self._height, self._width, 3), dtype=np.uint8
        )
        state_space = env.observation_space
        self.observation_space = spaces.Dict({"image": image_space, "state": state_space})

    def observation(self, observation):
        """Render the current MuJoCo scene and combine with the state observation.

        Args:
            observation: The original state-based observation from MetaWorld.

        Returns:
            dict: A dictionary with keys:
                - "image": An RGB image of shape (height, width, 3) with dtype uint8.
                - "state": The original state observation.
        """
        image_or_tuple = self.env.render()
        if isinstance(image_or_tuple, tuple):
            image = np.stack(image_or_tuple, axis=0)
        else:
            image = image_or_tuple

        image = np.rot90(image, 2) if self._rot_180 else image

        return {"image": image, "state": observation}


class NormalizeAction(gym.ActionWrapper):
    """Normalizes actions from [-1, 1] to the original action space bounds.

    This wrapper assumes that the input actions are in the range [-1, 1] and
    transforms them to the original action space bounds. This is useful when
    working with algorithms that output normalized actions.

    The transformation is:
    original_action = (normalized_action + 1) / 2 * (high - low) + low

    For infinite bounds, the actions are passed through unchanged.

    Example:
        >>> import gymnasium as gym
        >>> from env.wrapper import NormalizeAction
        >>>
        >>> # Create an environment
        >>> env = gym.make("Pendulum-v1")
        >>>
        >>> # Wrap with NormalizeAction to transform [-1,1] actions to original bounds
        >>> wrapped_env = NormalizeAction(env)
        >>>
        >>> # Actions in [-1, 1] will be transformed to the original action space
        >>> obs, info = wrapped_env.reset()
        >>> action = 0.5  # Will be transformed based on original action space
        >>> obs, reward, terminated, truncated, info = wrapped_env.step(action)
        >>>
        >>> # Check the normalized action space (should be [-1, 1])
        >>> print(wrapped_env.action_space)
    """

    def __init__(self, env: gym.Env):
        """Initializes the NormalizeAction wrapper.

        Args:
            env (gym.Env): The environment to wrap.
        """
        super().__init__(env)

        assert isinstance(env.action_space, spaces.Box), "NormalizeAction only supports Box action spaces"

        # Store the original action space bounds for later transformation
        self._original_low = env.action_space.low
        self._original_high = env.action_space.high

        # Create a boolean mask to identify which action dimensions have finite bounds
        # Some environments may have infinite bounds (e.g., velocity controls), which we handle differently
        # Only dimensions with both finite low and high bounds will be normalized to [-1, 1]
        self._mask = np.isfinite(self._original_low) & np.isfinite(self._original_high)

        # For finite bounds, use the original values; for infinite bounds, use [-1, 1] as fallback
        # This ensures we have valid bounds for the transformation calculations
        self._low = np.where(self._mask, self._original_low, -1)
        self._high = np.where(self._mask, self._original_high, 1)

        # Create the new normalized action space where finite bounds are mapped to [-1, 1]
        # Infinite bounds are left unchanged to preserve their unbounded nature
        # This allows algorithms to work with a standardized [-1, 1] range for most actions
        # while still supporting unbounded actions where appropriate
        normalized_low = np.where(self._mask, -np.ones_like(self._low), self._low)
        normalized_high = np.where(self._mask, np.ones_like(self._high), self._high)

        # Update the wrapper's action space to reflect the normalized bounds
        # Algorithms will now see actions in the [-1, 1] range for bounded dimensions
        self.action_space = spaces.Box(
            low=normalized_low, high=normalized_high, shape=env.action_space.shape, dtype=np.float64
        )

    def action(self, action):
        """Transform normalized action [-1, 1] to original action space bounds.

        Args:
            action: The normalized action in [-1, 1].

        Returns:
            The action transformed to original bounds.
        """
        # Transform from [-1, 1] to original bounds
        original = (action + 1) / 2 * (self._high - self._low) + self._low
        # Only apply transformation to finite bounds, pass through infinite bounds unchanged
        return np.where(self._mask, original, action)


class MetaworldTerminateOnSuccessWrapper(gym.Wrapper):
    """Gymnasium wrapper that terminates an episode when the task succeeds in Metaworld environments.

    The ``info["success"]`` flag produced by Metaworld environments is used
    to set the ``terminated`` signal.
    """

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Take one environment step, terminating on success.

        Args:
            action: Action to execute in the environment.

        Returns:
            A ``(observation, reward, terminated, truncated, info)`` tuple
            where *terminated* is ``True`` when ``info["success"]`` is truthy.
        """
        obs, reward, terminated, truncated, info = super().step(action)
        terminated = bool(info["success"])
        return obs, reward, terminated, truncated, info

