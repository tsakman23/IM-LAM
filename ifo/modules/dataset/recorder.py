# DEPRECATED for DMW: this generic PPO/Lightning recorder is NOT the pipeline used
# to build the Distracting Meta-World datasets. Use the standalone repo
# `distracting-metaworld-dataset-main` (generate_dataset_huggingface.py) instead.
# Kept importable for the DCS/PPO path only.

from itertools import chain
from typing import Dict, Optional, Union

import gymnasium as gym
import hydra
import numpy as np
import torch
from datasets.splits import Split
from lightning.fabric import Fabric
from omegaconf import DictConfig
from tensordict import TensorDict
from torchmetrics import MeanMetric
from tqdm import tqdm

from datasets import Dataset, Features
from ifo.common.collector import ExplorationMode
from ifo.common.trainer import LightningTrainer
from ifo.common.utils.utility import is_wrapped
from ifo.modules.ppo.module import PPOModule


class Recorder:
    """
    Recorder class for dataset generation.
    """

    def __init__(
        self,
        fabric: Fabric,
        compile: bool = False,
        exploration_mode: ExplorationMode = ExplorationMode.SAMPLE,
        max_train_steps: Union[int, float] = 1e6,
        max_test_steps: Union[int, float] = 1e6,
        cache_dir: str = "",
        repo_id: str = "",
        config_name: Optional[str] = None,
        max_shard_size: str = "1GB",
        private: bool = True,
        dataset_features: Features = Features(),
        random_seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Initialize the PPO trainer.

        Args:
            fabric (Fabric): The fabric instance to use for training.
            exploration_mode (ExplorationMode): Mode to use for exploration during recording.
            max_train_steps (Union[int, float]): The maximum number of environment steps for the training dataset.
            max_test_steps (Union[int, float]): The maximum number of environment steps for the test dataset.
            cache_dir (str): The directory to cache the dataset.
            repo_id (str): The repository ID to push the dataset to.
            config_name (Optional[str]): The name of the config to push the dataset to.
            max_shard_size (str): The maximum size of a shard to push to the dataset.
            private (bool): Whether the dataset is private.
            dataset_features (Features): The features of the dataset.
            random_seed (Optional[int]): The random seed to use for resetting the environment.
            **kwargs: Additional keyword arguments
        """
        self.fabric = fabric
        self.compile = compile
        self.max_train_steps = int(max_train_steps)
        self.max_test_steps = int(max_test_steps)
        self.exploration_mode = exploration_mode
        self.cache_dir = cache_dir
        self.repo_id = repo_id
        self.config_name = config_name
        self.max_shard_size = max_shard_size
        self.private = private
        self.dataset_features = dataset_features
        self.random_seed = random_seed

    def create_dataset(
        self,
        model: PPOModule,
        env_cfg: DictConfig,
        checkpoint: str,
    ) -> None:
        """Create the dataset.

        Args:
            model (PPOModule): The model used for recording. This should be an instance of a LightningModule.
            env_cfg (DictConfig): The environment configuration.
            checkpoint (str): Path to a specific checkpoint to load or directory to load the latest
                checkpoint from.
        """
        # Setup model and optimizer.
        model.training_step = torch.compile(model.training_step, mode="max-autotune", disable=not self.compile)
        model.validation_step = torch.compile(model.validation_step, mode="max-autotune", disable=not self.compile)
        model.forward = torch.compile(model.forward, mode="max-autotune", disable=not self.compile)
        model = self.fabric.setup(model)

        # Assemble checkpoint state (current epoch and global step will be added in save).
        state = {"model": model}

        # Load last checkpoint if available.
        _, _, _ = LightningTrainer.load_checkpoint(self.fabric, state, None, checkpoint)

        # Generate the dataset splits.
        train_test_split = {Split.TRAIN: self.max_train_steps, Split.TEST: self.max_test_steps}

        for split, total_steps in train_test_split.items():
            print(f"Generate dataset for {split} split with {total_steps} examples.")
            gen_kwargs = {
                "policy": model,
                "env_cfg": env_cfg,
                "max_steps": total_steps,
                "exploration_mode": self.exploration_mode,
            }
            dataset = Dataset.from_generator(
                generator=self.record,
                cache_dir=self.cache_dir,
                features=self.dataset_features,
                split=split,
                gen_kwargs=gen_kwargs,
            )
            dataset.push_to_hub(
                repo_id=self.repo_id,
                config_name=self.config_name,
                max_shard_size=self.max_shard_size,
                private=self.private,
            )

    def _convert_transition_dtype(self, transition: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Convert the transition dtype to match the dataset features.

        Args:
            transition (Dict[str, np.ndarray]): The transition to convert.

        Returns:
            Dict[str, np.ndarray]: The converted transition.
        """
        transition["observation"] = transition["observation"].astype(np.uint8)
        transition["reward"] = transition["reward"].astype(np.float32)
        transition["terminated"] = transition["terminated"].astype(bool)
        transition["truncated"] = transition["truncated"].astype(bool)
        return transition

    @torch.no_grad()
    def record(
        self,
        policy: PPOModule,
        env_cfg: DictConfig,
        max_steps: int,
        exploration_mode: ExplorationMode,
        epsilon_greedy: float = 0.0,
    ):
        """Record a dataset of transitions from the environment.

        This generator function records transitions from the environment using the provided policy
        and yields them one by one. It continues until the specified maximum number of steps is reached.

        Args:
            policy (PPOModule): The policy to use for action selection.
            env_cfg (DictConfig): The environment configuration.
            max_steps (int): The maximum number of environment steps to record.
            exploration_mode (ExplorationMode): The mode to use for exploration.
            epsilon_greedy (float): The epsilon value for epsilon-greedy exploration.

        Yields:
            dict: A dictionary containing a single transition with keys:
                - observation: The observation at the current step.
                - action: The action taken at the current step.
                - reward: The reward received at the current step.
                - terminated: Whether the episode terminated at the current step.
                - truncated: Whether the episode was truncated at the current step.
        """
        episode_return, episode_length = MeanMetric(), MeanMetric()
        env = hydra.utils.instantiate(env_cfg, preserve_original=True)
        assert is_wrapped(env, gym.wrappers.RecordEpisodeStatistics) or is_wrapped(
            env, gym.wrappers.vector.RecordEpisodeStatistics
        ), "Environment must be wrapped with RecordEpisodeStatistics"
        action_repeat = env.get_wrapper_attr("action_repeat") if hasattr(env, "get_wrapper_attr") else 1
        assert action_repeat == 1, "Action repeat is not supported for dataset generation."
        transitions = [[] for _ in range(env.num_envs)]
        transition = {}
        env_is_reset = False
        steps = 0
        recorder_pbar = tqdm(range(max_steps), total=max_steps, desc="Recording")
        while steps < max_steps:
            if not env_is_reset:
                observation, info = env.reset(seed=self.random_seed)
                transition["observation"] = info["original"]["observation"]
                transition["reward"] = np.zeros(env.num_envs, dtype=np.float32)
                transition["terminated"] = np.zeros(env.num_envs, dtype=bool)
                transition["truncated"] = np.zeros(env.num_envs, dtype=bool)
                done = np.zeros(env.num_envs, dtype=bool)
                env_is_reset = True

            # Get actions from policy or sample randomly
            if exploration_mode == ExplorationMode.RANDOM:
                action = env.action_space.sample()
            else:
                obs_tensor = torch.as_tensor(observation, device=policy.device)
                prediction = policy(TensorDict({"observation": obs_tensor}, (env.num_envs,)))
                if exploration_mode == ExplorationMode.MODE:
                    action = prediction["action_distribution"].mode
                elif exploration_mode == ExplorationMode.EPSILON_GREEDY:
                    greedy_action = prediction["action_distribution"].mode
                    random_action = torch.as_tensor(env.action_space.sample(), device=greedy_action.device)
                    uniform_samples = torch.rand(env.num_envs, device=greedy_action.device)
                    action = torch.where(uniform_samples < epsilon_greedy, random_action, greedy_action)
                elif exploration_mode == ExplorationMode.SAMPLE:
                    action = prediction["action_distribution"].sample()
                else:
                    raise ValueError(f"Invalid exploration mode: {exploration_mode}")
                # Upcast if using mixed precision.
                if action.dtype == torch.float16 or action.dtype == torch.bfloat16:
                    action = action.float()
                action = action.cpu().numpy()

            # Step environment
            next_observation, reward, terminated, truncated, info = env.step(action)
            next_done = terminated | truncated

            # Store action
            transition["action"] = info["original"]["action"]

            # Convert transition dtype to match dataset features
            transition = self._convert_transition_dtype(transition)

            # Store complete transition.
            for i in range(env.num_envs):
                transitions[i].append({k: v[i] for k, v in transition.items()})

            # Update tracking variables
            if next_done.any():
                episode_return(info["episode_statistics"]["r"][info["_episode_statistics"]])
                episode_length(info["episode_statistics"]["l"][info["_episode_statistics"]])

            # Yield complete trajectory if finished.
            if done.any():
                # Filter for all completed trajectories
                complete_trajectories = [trajectory for trajectory, d in zip(transitions, done) if d]
                complete_trajectories = list(chain(*complete_trajectories))
                # Reset collected transitions for completed trajectories
                transitions = [trajectory if not d else [] for trajectory, d in zip(transitions, done)]

                # Yield all complete transitions till now.
                for i in range(len(complete_trajectories)):
                    if steps + i < max_steps:
                        recorder_pbar.update(1)
                        yield complete_trajectories[i]
                    else:
                        break
                steps += len(complete_trajectories)

            # Store parts of next transition
            transition = {}
            transition["observation"] = info["original"]["observation"]
            transition["reward"] = info["original"]["reward"]
            transition["terminated"] = info["original"]["terminated"]
            transition["truncated"] = info["original"]["truncated"]

            # Update observation and done
            done = next_done
            observation = next_observation

        recorder_pbar.close()
        env.close()
        print()
        print("Finished recording.")
        print(f"Mean episode return: {episode_return.compute()}")
        print(f"Mean episode length: {episode_length.compute()}")
