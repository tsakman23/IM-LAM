from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import Env
from gymnasium.vector import AutoresetMode, SyncVectorEnv, VectorEnv
from tensordict import TensorDict
from torchmetrics import MeanMetric

from ifo.common.utils.utility import is_wrapped


class ExplorationMode(str, Enum):
    """Enum for different exploration modes in the collector.

    Modes:
        RANDOM: Select actions randomly from the action space
        SAMPLE: Sample actions from the policy's distribution
        MODE: Always select the action with highest probability
        EPSILON_GREEDY: Use greedy action selection with epsilon probability of random action
    """

    RANDOM = "random"
    SAMPLE = "sample"
    MODE = "mode"
    EPSILON_GREEDY = "epsilon_greedy"


class Collector:
    """A collector for gathering rollouts from gymnasium environments.

    This collector can work with both single and vector environments, and supports
    collecting rollouts using either a policy or random actions. The collector
    can be configured to collect a specific number of steps or episodes.
    """

    def __init__(
        self,
        env: Union[Env, VectorEnv],
        policy: Optional[nn.Module] = None,
        return_log_prob: bool = False,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize the collector.

        Args:
            env: The environment to collect from. Can be either a single environment
                or a vector environment.
            policy: Optional policy to use for action selection. If None, random actions
                will be sampled from the environment's action space.
            return_log_prob: Whether to return the log probability of the action under key "log_prob".
            random_seed: Optional random seed to set for the environment for each reset.
        """
        self.policy = policy
        self._env_is_reset = False
        self._action_space = env.action_space
        self._return_log_prob = return_log_prob
        self._random_seed = random_seed
        assert not return_log_prob or policy is not None, "Policy must be provided if return_log_prob is True"
        assert is_wrapped(env, gym.wrappers.RecordEpisodeStatistics) or is_wrapped(
            env, gym.wrappers.vector.RecordEpisodeStatistics
        ), "Environment must be wrapped with RecordEpisodeStatistics"

        # Wrap single env in SyncVectorEnv for consistent handling
        if not isinstance(env, VectorEnv):
            self.env = SyncVectorEnv([lambda: env])
            self.num_envs = 1
        else:
            self.env = env
            self.num_envs = env.num_envs
        assert (
            self.env.unwrapped.metadata["autoreset_mode"] == AutoresetMode.NEXT_STEP
        ), "Only NEXT_STEP autoreset mode is supported"

    @property
    def last_step(self) -> TensorDict:
        """
        Last step of the rollout.

        Returns:
            TensorDict: Last step of the rollout.
        """
        return TensorDict(self._last_step, batch_size=(self.num_envs,))

    def rollout(
        self,
        max_steps: int,
        callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        exploration_mode: ExplorationMode = ExplorationMode.SAMPLE,
        epsilon_greedy: float = 0.0,
        reset_before_rollout: bool = False,
        store_data: bool = True,
        store_info: bool = False,
    ) -> Tuple[TensorDict, Optional[float], Optional[float]]:
        """Collect rollouts from the environment.

        Args:
            max_steps: Maximum number of steps to collect.
            callback: Optional callback function that will be called after each step
                with the step data.
            exploration_mode: Mode to use for exploration. Can be "random", "sample", or "mode".
                - "random": Sample random actions.
                - "sample": Sample actions from the policy.
                - "mode": Take the mode of the action distribution.
                - "egreedy": Take epsilon-greedy actions.
            epsilon_greedy: Epsilon for ExplorationMode.EPSILON_GREEDY.
            reset_before_rollout: Whether to reset the environment before collecting
                rollouts. If False, the environment will not be reset and just continue
                from the last observation.
            store_data: Whether to store the data of the rollout in a TensorDict. If False, an empty
                TensorDict will be returned.
            store_info: Whether to store the info of the rollout in a TensorDict.

        Returns:
            - data: TensorDict containing the collected data
            - episode_returns: Mean of the episode returns or None if not collected
            - episode_lengths: Mean of the episode lengths or None if not collected

        Note:
            Each transition is at least composed of:
            - observation (before step)
            - action (before step)
            - reward (after step)
            - terminated (after step)
            - truncated (after step)
        """
        assert max_steps > 0, "Maximum number of steps must be greater than 0"
        # Initialize transition.
        transition: Dict[str, np.ndarray] = {}
        data = TensorDict({}, batch_size=(self.num_envs, max_steps), device=torch.device("cpu"))
        # Initialize environment
        if reset_before_rollout or not self._env_is_reset:
            observation, info = self.env.reset(seed=self._random_seed)
            transition["observation"] = observation
            transition["reward"] = np.zeros(self.num_envs, dtype=np.float32)
            transition["terminated"] = np.zeros(self.num_envs, dtype=bool)
            transition["truncated"] = np.zeros(self.num_envs, dtype=bool)
            if store_info:
                transition["info"] = info
            self._env_is_reset = True
        else:
            observation = self._last_step["observation"]
            transition["observation"] = observation
            transition["reward"] = self._last_step["reward"]
            transition["terminated"] = self._last_step["terminated"]
            transition["truncated"] = self._last_step["truncated"]
            if store_info:
                transition["info"] = self._last_step["info"]

        # Initialize tracking variables
        episode_returns = MeanMetric()
        episode_lengths = MeanMetric()

        # Main collection loop
        for step_count in range(max_steps):
            # Get actions from policy or sample randomly
            prediction = {}
            if self.policy is not None or exploration_mode == ExplorationMode.RANDOM:
                with torch.no_grad():
                    obs_tensor = torch.as_tensor(observation, device=self.policy.device)
                    prediction = self.policy(TensorDict({"observation": obs_tensor}, (self.num_envs,)))
                    assert "action_distribution" in prediction, "Policy must output action distribution"
                    if exploration_mode == ExplorationMode.MODE:
                        action = prediction["action_distribution"].mode
                    elif exploration_mode == ExplorationMode.EPSILON_GREEDY:
                        greedy_action = prediction["action_distribution"].mode
                        random_action = torch.as_tensor(self._action_space.sample(), device=greedy_action.device)
                        uniform_samples = torch.rand(self.num_envs, device=greedy_action.device)
                        action = torch.where(uniform_samples < epsilon_greedy, random_action, greedy_action)
                    elif exploration_mode == ExplorationMode.SAMPLE:
                        action = prediction["action_distribution"].sample()
                    else:
                        raise ValueError(f"Invalid exploration mode: {exploration_mode}")
                    if self._return_log_prob:
                        prediction["log_prob"] = prediction["action_distribution"].log_prob(action)
                    # Upcast if using mixed precision.
                    if action.dtype == torch.float16 or action.dtype == torch.bfloat16:
                        action = action.float()
                    action = action.cpu().numpy()
            else:
                action = self._action_space.sample()

            # Step environment
            next_observation, reward, terminated, truncated, info = self.env.step(action)
            done = terminated | truncated

            # Update tracking variables
            if done.any():
                episode_returns(info["episode_statistics"]["r"][info["_episode_statistics"]])
                episode_lengths(info["episode_statistics"]["l"][info["_episode_statistics"]])

            # Store action and other policy output.
            prediction["action"] = action
            additional_keys = set(prediction.keys()) - {"action_distribution"}
            for k in additional_keys:
                transition[k] = prediction[k]

            # Store info.
            if store_info:
                transition["info"] = info

            # Store complete transition.
            if store_data:
                data[:, step_count] = transition

            # Store data from last step for next rollout.
            if step_count == max_steps - 1:
                self._last_step = {
                    "observation": next_observation,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                }
                if store_info:
                    self._last_step["info"] = info
            else:
                transition["observation"] = next_observation
                transition["action"] = action
                transition["reward"] = reward
                transition["terminated"] = terminated
                transition["truncated"] = truncated
                if store_info:
                    transition["info"] = info

            # Call callback if provided
            if callback is not None:
                callback(data)

            # Update observation
            observation = next_observation

        episode_return = episode_returns.compute() if episode_returns.weight > 0 else None
        episode_length = episode_lengths.compute() if episode_lengths.weight > 0 else None
        return data, episode_return, episode_length
