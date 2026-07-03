import os
from time import time
from typing import Any, Dict, Iterator, Literal, Optional, Tuple, Union

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from lightning.fabric import Fabric
from tensordict import TensorDict
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import MeanMetric

import wandb
from ifo.common.collector import ExplorationMode
from ifo.common.trainer import LightningTrainer
from ifo.common.utils.utility import get_latest_checkpoint, rename_key
from ifo.modules.dreamerv3.buffer import ReplayBuffer
from ifo.modules.dreamerv3.module import DreamerV3Module
from ifo.modules.dreamerv3.prefetch import Prefetch
from ifo.modules.dreamerv3.utils import Ratio


class DreamerV3Trainer(LightningTrainer):
    """
    Trainer class for reinforcement learning with DreamerV3.

    Note: This implementation is missing the replay context and the replay loss from the original implementation.

    Paper: https://arxiv.org/abs/2301.04104
    """

    def __init__(
        self,
        fabric: Fabric,
        compile: bool = False,
        checkpoint_dir: str = "./checkpoints",
        validation_criteria: Literal["min", "max"] = "max",
        validation_frequency: Union[int, float] = 1,
        validation_unit: Literal["iteration", "step"] = "step",
        random_seed: int = 0,
        validation_steps: Union[int, float] = 1e4,
        validation_exploration_mode: ExplorationMode = ExplorationMode.SAMPLE,
        max_steps: Union[int, float] = 1e6,
        **kwargs,
    ) -> None:
        """Initialize the DreamerV3 trainer.

        Args:
            fabric (Fabric): The fabric instance to use for training.
            compile (bool): Whether to compile the model using torch.compile.
            checkpoint_dir (str): Directory to store checkpoints to and to automatically load the latest checkpoint
                from, when available. Can be overwritten by the `checkpoint_path` argument in :meth:`fit`.
            validation_criteria (Literal["min", "max"]): The criteria to use for validation, either minimizing or
                maximizing the validation metric.
            validation_frequency (Union[int, float]): How many iterations or environment steps to run before each
                validation.
            validation_unit (Literal["iteration", "step"]): Whether to validate on iterations or environment steps.
            random_seed (int): The random seed to use for resetting the environment.
            validation_steps (Union[int, float]): The number of environment steps to run validation.
            validation_exploration_mode (ExplorationMode): Mode to use for exploration during validation.
            max_steps (Union[int, float]): The maximum number of environment steps to train.
            **kwargs: Additional keyword arguments passed to the parent class.
        """
        self.max_steps = int(max_steps)
        self.validation_steps = int(validation_steps)
        self.validation_exploration_mode = validation_exploration_mode
        self.validation_frequency = int(validation_frequency)
        self.policy_steps_per_iteration = 0  # This is set in the fit method.
        self.action_repeat = 0  # This is set in the fit method.
        self.global_step_action_repeat = 0  # This is set in the fit method.
        self.next_validation_step = 0
        self.per_rank_gradient_steps = 0
        self.supported_action_spaces = (spaces.Box, spaces.Discrete)
        super().__init__(fabric, compile, checkpoint_dir, validation_criteria, validation_unit, random_seed, **kwargs)
        assert validation_exploration_mode in [
            ExplorationMode.MODE,
            ExplorationMode.SAMPLE,
        ], "Validation exploration mode must be MODE or SAMPLE"

    def fit(
        self,
        model: DreamerV3Module,
        train_env: VectorEnv,
        val_env: Optional[VectorEnv] = None,
        checkpoint: Optional[str] = None,
    ) -> None:
        """Fit the model.

        Args:
            model (DreamerV3Module): The model to be trained. This should be an instance of a LightningModule.
            train_env (VectorEnv): The environment for training the agent.
            val_env (Optional[VectorEnv]): The environment for validating the agent. If not provided,
                validation will be skipped.
            checkpoint (Optional[str], optional): Path to a specific checkpoint to load or directory to load the latest
                checkpoint from. If provided, this will override automatic loading from the checkpoint directory.
        """
        assert isinstance(
            train_env.single_action_space, self.supported_action_spaces
        ), f"Action space {train_env.single_action_space} is not supported. Supported spaces: {self.supported_action_spaces}"

        # Setup optimizer and scheduler.
        optimizer, lr_scheduler = model.configure_optimizers()

        # Setup model and optimizer.
        model.training_step = torch.compile(model.training_step, mode="max-autotune", disable=not self.compile)
        model.forward = torch.compile(model.forward, mode="max-autotune", disable=not self.compile)
        model, optimizer = self.fabric.setup(model, optimizer)

        # Setup ratio for dreaming or gathering real experience.
        ratio = model.configure_ratio()

        # Setup replay buffer.
        replay_buffer = model.configure_replay_buffer(
            self.fabric.world_size,
            self.fabric.global_rank,
            self.checkpoint_dir,
        )

        # Setup sampling with prefetching.
        sample_train, sample_val = self._setup_sampling(replay_buffer, model)

        # Assemble checkpoint state (current epoch and global step will be added in save).
        state = {
            "model": model,
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "ratio": ratio,
            "replay_buffer": replay_buffer,
        }

        # Load last checkpoint if available.
        self.global_step, self.global_epoch, self.global_iteration = self.load_checkpoint(
            self.fabric, state, self.checkpoint_dir, checkpoint
        )

        # Check if we even need to train here.
        if self.global_step >= self.max_steps:
            return

        # Steps done in the environment per iteration
        self.action_repeat = (
            train_env.get_wrapper_attr("action_repeat") if hasattr(train_env, "get_wrapper_attr") else 1
        )
        self.global_step_action_repeat = self.global_step * self.action_repeat
        self.policy_steps_per_iteration = int(train_env.num_envs * self.fabric.world_size)

        # Initialize agent and environment.
        step_data = self._initialize_step_data(train_env, model)
        step_data["dreamer_state"] = model.net.init_states(step_data["is_first"][:, 0])

        trainer_pbar = self.tqdm_wrapper(
            range(
                self.global_step // self.policy_steps_per_iteration, self.max_steps // self.policy_steps_per_iteration
            ),
            initial=self.global_step,
            total=self.max_steps,
            position=0,
            desc="Training",
        )

        for steps in trainer_pbar:
            self.fabric.log("trainer/iteration", self.global_iteration, step=self.global_step_action_repeat)

            # Environment interaction
            step_data = self.rollout_loop(model, train_env, replay_buffer, step_data)
            self.fabric.call("rollout_loop_end", **rename_key(locals(), "self", "trainer"))

            # Dynamics and behavior learning
            enough_data = self.global_step >= train_env.num_envs * model.block_size * self.fabric.world_size
            if enough_data:
                self.train_loop(model, optimizer, sample_train, ratio, lr_scheduler)
                self.fabric.call("train_loop_end", **rename_key(locals(), "self", "trainer"))

            if self.should_validate(self.validation_unit):
                self.val_loop(model, val_env, sample_val)
                self.save_best(state=state, on_validation=val_env is not None)
                self.fabric.call("val_loop_end", **rename_key(locals(), "self", "trainer"))

            if self.fabric.is_global_zero:
                trainer_pbar.update(self.policy_steps_per_iteration)
            metrics = {"train_return": self._current_train_loss, "val_return": self._current_val_loss}
            self._format_iterable(trainer_pbar, metrics)

            # Stop training if max_steps is reached.
            if self.max_steps is not None and self.global_step >= self.max_steps:
                self.should_stop = True

            self.global_iteration += 1

        # Reset for next fit call.
        self.should_stop = False
        self.global_iteration = 0
        self.global_step = 0
        self.global_epoch = 0
        self.next_validation_step = 0
        self._current_val_loss = float("inf") if self.validation_criteria == "min" else -float("inf")
        self._prev_val_loss = float("inf") if self.validation_criteria == "min" else -float("inf")
        self._current_train_loss = float("inf") if self.validation_criteria == "min" else -float("inf")
        self._prev_train_loss = float("inf") if self.validation_criteria == "min" else -float("inf")
        self.close_tqdm_wrapper(trainer_pbar, self.max_steps)

    @torch.no_grad()
    def rollout_loop(
        self,
        model: DreamerV3Module,
        train_env: VectorEnv,
        replay_buffer: ReplayBuffer,
        step_data: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Collect rollout data from the environment and store it in the replay buffer.

        Args:
            model (DreamerV3Module): The DreamerV3 model to use for collecting data.
            train_env (VectorEnv): The training environment to collect data from.
            replay_buffer (ReplayBuffer): The replay buffer to store collected data.
            step_data (Dict[str, np.ndarray]): Current step data containing observations and other state information.

        Returns:
            Dict[str, np.ndarray]: Updated step data for the next rollout iteration.
        """
        start_time = time()
        episode_returns = MeanMetric().to(self.fabric.device)
        episode_lengths = MeanMetric().to(self.fabric.device)
        model.eval()

        obs = TensorDict(step_data.pop("next_obs"))
        dreamer_state = step_data.pop("dreamer_state")
        # Sample an action given the observation received by the environment
        if self.global_step < (model.batch_size * model.block_size * self.fabric.world_size):
            actions = train_env.action_space.sample()
        else:
            actions, dreamer_state = model(obs.to(self.fabric.device), dreamer_state, greedy=False)
            actions = actions.cpu().numpy(force=True)

        step_data["actions"] = actions
        for env_idx in range(train_env.num_envs):
            replay_buffer.add({k: v[env_idx] for k, v in step_data.items()}, worker=env_idx)

        # Step the environment
        next_obs, rewards, terminated, truncated, infos = train_env.step(actions)
        dones = terminated | truncated

        # Update tracking variables
        if dones.any():
            episode_returns(infos["episode_statistics"]["r"][infos["_episode_statistics"]])
            episode_lengths(infos["episode_statistics"]["l"][infos["_episode_statistics"]])

        dreamer_state = model.net.update_states(dones, dreamer_state)

        for k in model.net.obs_keys:
            step_data[k] = next_obs[k]

        step_data["next_obs"] = next_obs  # Save for the next rollout
        step_data["is_first"] = next_obs["is_first"][:, None]
        step_data["terminated"] = terminated[:, None]
        step_data["truncated"] = truncated[:, None]
        step_data["rewards"] = rewards[:, None]
        step_data["dreamer_state"] = dreamer_state

        # Log episode return and length.
        self.global_step += self.policy_steps_per_iteration
        self.global_step_action_repeat = self.global_step * self.action_repeat
        if self.fabric.all_reduce(dones.sum()) > 0:
            episode_return, episode_length = episode_returns.compute(), episode_lengths.compute()
            self.fabric.log_dict(
                {"train/episode_return": episode_return, "train/episode_length": episode_length},
                step=self.global_step_action_repeat,
            )
            self._prev_train_loss = self._current_train_loss
            self._current_train_loss = episode_return

        # Log throughput.
        throughput = self.policy_steps_per_iteration / (time() - start_time)
        self.fabric.log("train/env_throughput", throughput, step=self.global_step_action_repeat)
        return step_data

    def train_loop(
        self,
        model: DreamerV3Module,
        optimizer: Optimizer,
        sample_train: Iterator[TensorDict],
        ratio: Ratio,
        lr_scheduler: Optional[LRScheduler] = None,
    ) -> None:
        """The training loop running a single training epoch.

        Args:
            model (DreamerV3Module): The DreamerV3 model to be trained.
            optimizer (Optimizer): The optimizer for updating the model parameters.
            sample_train (Iterator[TensorDict]): The iterator to sample training data from replay buffer.
            ratio (Ratio): The ratio object determining how many training steps to perform.
            lr_scheduler (Optional[LRScheduler]): The learning rate scheduler. If not provided, a constant
                learning rate will be used.
        """
        train_steps = ratio(self.global_step)
        if train_steps == 0:
            return
        self.fabric.call("train_epoch_start", **rename_key(locals(), "self", "trainer"))
        model.train()
        train_pbar = self.tqdm_wrapper(
            range(train_steps), total=train_steps, desc="Training Loop", position=1, leave=False
        )
        mean_metrics, throughput = None, MeanMetric().to(self.fabric.device)

        for batch_idx in train_pbar:
            start_time = time()
            batch = next(sample_train)

            # Update target critic periodically.
            if self.per_rank_gradient_steps % model.target_network_update_freq == 0:
                self._update_target_network(model)

            # Train one step.
            self.fabric.call("train_batch_start", **rename_key(locals(), "self", "trainer"))
            step_dict = model.training_step(batch, batch_idx=batch_idx, batch_len=train_steps)

            self.fabric.call("before_zero_grad", **rename_key(locals(), "self", "trainer"))
            optimizer.zero_grad()

            self.fabric.call("before_backward", **rename_key(locals(), "self", "trainer"))
            self.fabric.backward(step_dict["loss"])

            if model.max_grad_norm is not None:
                self.fabric.call("before_clip_grad", **rename_key(locals(), "self", "trainer"))  #
                self.fabric.clip_gradients(module=model, optimizer=optimizer, max_norm=model.max_grad_norm)

            self.fabric.call("before_optimizer_step", **rename_key(locals(), "self", "trainer"))
            optimizer.step()

            if lr_scheduler is not None:
                self.fabric.call("before_step_scheduler", **rename_key(locals(), "self", "trainer"))
                lr_scheduler.step()

            # Update mean metrics.
            if not mean_metrics:
                mean_metrics = {k: MeanMetric().to(self.fabric.device) for k in step_dict.keys()}
            for k, metric in mean_metrics.items():
                metric(step_dict[k])

            # Update progress bar.
            self._format_iterable(train_pbar, mean_metrics["loss"].compute(), "train")

            # Log throughput.
            # throughput(batch.shape[0] / (time() - start_time))
            throughput(batch["actions"].shape[0] / (time() - start_time))
            self.fabric.call("train_batch_end", **rename_key(locals(), "self", "trainer"))
            self.per_rank_gradient_steps += 1

        self.global_epoch += 1
        self.fabric.log(
            "train/throughput", throughput.compute() * self.fabric.world_size, step=self.global_step_action_repeat
        )
        assert mean_metrics is not None, "No training metrics were computed."
        self.fabric.log_dict(
            {f"train/{k}": v.compute() for k, v in mean_metrics.items()}, step=self.global_step_action_repeat
        )

        self.close_tqdm_wrapper(train_pbar, train_steps)

    @torch.no_grad()
    def val_loop(
        self,
        model: DreamerV3Module,
        val_env: Optional[VectorEnv],
        sample_val: Iterator[TensorDict],
    ) -> None:
        """Run validation loop to evaluate the model performance.

        Args:
            model (DreamerV3Module): The DreamerV3 model to evaluate.
            val_env (Optional[VectorEnv]): The validation environment. If None, validation is skipped.
            sample_val (Iterator[TensorDict]): The iterator to sample validation data from replay buffer.
        """
        # Skip validation if no validation environment is provided.
        if val_env is None:
            return

        start_time = time()
        episode_returns = MeanMetric().to(self.fabric.device)
        episode_lengths = MeanMetric().to(self.fabric.device)

        val_pbar = self.tqdm_wrapper(
            range(self.validation_steps // self.policy_steps_per_iteration),
            total=self.validation_steps,
            position=1,
            desc="Validation Loop",
            unit="steps",
            leave=False,
        )

        model.eval()
        greedy = self.validation_exploration_mode == ExplorationMode.MODE
        obs, _ = val_env.reset(seed=self.random_seed + self.fabric.global_rank * val_env.num_envs)
        obs = TensorDict(obs)
        dreamer_state = model.net.init_states(obs["is_first"])

        for _ in val_pbar:
            actions, dreamer_state = model(obs.to(self.fabric.device), dreamer_state, greedy=greedy)
            actions = actions.cpu().numpy(force=True)

            next_obs, rewards, terminated, truncated, infos = val_env.step(actions)
            dones = terminated | truncated

            # Update tracking variables
            if dones.any():
                episode_returns(infos["episode_statistics"]["r"][infos["_episode_statistics"]])
                episode_lengths(infos["episode_statistics"]["l"][infos["_episode_statistics"]])
                self._format_iterable(val_pbar, {"return": episode_returns.compute()}, "val")

            dreamer_state = model.net.update_states(dones, dreamer_state)

            obs = TensorDict(next_obs)

            if self.fabric.is_global_zero:
                val_pbar.update(self.policy_steps_per_iteration)

        if self.fabric.all_reduce(episode_returns.weight) > 0:
            episode_return, episode_length = episode_returns.compute(), episode_lengths.compute()
            self.fabric.log_dict(
                {"val/episode_return": episode_return, "val/episode_length": episode_length},
                step=self.global_step_action_repeat,
            )
            self._prev_val_loss = self._current_val_loss
            self._current_val_loss = episode_return

        # Log throughput.
        throughput = self.validation_steps * val_env.num_envs / (time() - start_time)
        self.fabric.log("val/throughput", throughput * self.fabric.world_size, step=self.global_step_action_repeat)

        # Log open-loop performance of world model only on rank 0 and if there is enough data in the replay buffer.
        enough_data = self.global_step >= val_env.num_envs * model.block_size * self.fabric.world_size
        if self.fabric.is_global_zero and enough_data and model.net.cnn_keys:
            batch = next(sample_val)

            video = model.visualize_world_model(batch)
            self.fabric.log(
                "val/world_model_video",
                wandb.Video(video.numpy(force=True), fps=5, format="gif"),
                step=self.global_step_action_repeat,
            )
        self.close_tqdm_wrapper(val_pbar, self.validation_steps)

    def should_validate(self, unit: Literal["iteration", "step"]) -> bool:
        """Whether to currently run validation.

        Args:
            unit (Literal["iteration", "step"]): The unit of the validation frequency.

        Returns:
            bool: Whether to currently run validation.
        """
        if unit == "iteration" and self.validation_unit == "iteration":
            return self.global_iteration % self.validation_frequency == 0
        elif unit == "step" and self.validation_unit == "step":
            if self.global_step >= self.next_validation_step:
                self.next_validation_step = self.global_step + self.validation_frequency
                return True
        return False

    @staticmethod
    def load_checkpoint(
        fabric: Fabric,
        state: Dict[str, Union[nn.Module, Optimizer, ReplayBuffer, Any]] = {},
        default_checkpoint_dir: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Tuple[int, int, int]:
        replay_buffer = None
        if "replay_buffer" in state:
            replay_buffer = state.pop("replay_buffer")
            assert isinstance(replay_buffer, ReplayBuffer), "Replay buffer must be of type ReplayBuffer"
            replay_buffer = DreamerV3Trainer.load_replay_buffer(
                fabric, replay_buffer, default_checkpoint_dir, checkpoint_path
            )

        global_step, global_epoch, global_iteration = LightningTrainer.load_checkpoint(
            fabric, state, default_checkpoint_dir, checkpoint_path
        )

        if replay_buffer:
            state["replay_buffer"] = replay_buffer

        return global_step, global_epoch, global_iteration

    @staticmethod
    def load_replay_buffer(
        fabric: Fabric,
        replay_buffer: ReplayBuffer,
        default_checkpoint_dir: Optional[str],
        checkpoint_path: Optional[str],
    ) -> ReplayBuffer:
        """Load the replay buffer from the checkpoint.

        Args:
            fabric (Fabric): The fabric instance used for distributed training.
            replay_buffer (ReplayBuffer): The replay buffer instance to load data into.
            default_checkpoint_dir (Optional[str]): Default directory to search for checkpoints if no specific path is provided.
            checkpoint_path (Optional[str]): Specific path to a checkpoint file or directory containing checkpoints.

        Returns:
            ReplayBuffer: The replay buffer with loaded data from the checkpoint, or the original buffer if no checkpoint is found.
        """
        # Try to load from a specific checkpoint path.
        checkpoint_path = get_latest_checkpoint(checkpoint_path, extension="ckpt")

        # If no checkpoint path is provided, try to continue training from a previous checkpoint.
        if checkpoint_path is None:
            checkpoint_path = get_latest_checkpoint(default_checkpoint_dir, extension="ckpt")

        # Get directory of replay buffer
        replay_buffer_dirs = []
        if checkpoint_path:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            for i in range(fabric.world_size):
                replay_path = os.path.join(checkpoint_dir, f"replay_buffer_{i}")
                if os.path.exists(replay_path):
                    replay_buffer_dirs.append(replay_path)

        if replay_buffer_dirs:
            assert (
                len(replay_buffer_dirs) == fabric.world_size
            ), f"Expected {fabric.world_size} replay buffer directories, got {len(replay_buffer_dirs)}"
            replay_buffer.load(replay_buffer_dirs[fabric.global_rank])
            print(f"Successfully loaded replay buffer directories {replay_buffer_dirs}")
        return replay_buffer

    def save(
        self,
        state: Dict[str, Union[nn.Module, Optimizer, ReplayBuffer, Any]] = {},
        post_fix: Literal["iteration", "epoch", "step"] = "epoch",
        keep_last: bool = False,
    ) -> None:
        replay_buffer = None
        if state and "replay_buffer" in state:
            assert isinstance(state["replay_buffer"], ReplayBuffer), "Replay buffer must be of type ReplayBuffer"
            replay_buffer = state.pop("replay_buffer")
            replay_buffer.save()

        super().save(state, post_fix, keep_last)

        if replay_buffer:
            state["replay_buffer"] = replay_buffer

    def _initialize_step_data(self, env: VectorEnv, model: DreamerV3Module) -> Dict[str, np.ndarray]:
        """Initialize step data dictionary with observations and default values.

        Args:
            env (VectorEnv): The training environment to reset and get initial observations.
            model (DreamerV3Module): The DreamerV3 model to get observation keys from.

        Returns:
            Dict[str, np.ndarray]: Initial step data containing observations and default values.
        """
        step_data = {}
        obs, _ = env.reset(seed=self.random_seed + self.fabric.global_rank * env.num_envs)
        for k in model.net.obs_keys:
            step_data[k] = obs[k]
        step_data["next_obs"] = obs  # Save for the next rollout
        step_data["rewards"] = np.zeros((env.num_envs, 1))
        step_data["truncated"] = np.zeros((env.num_envs, 1))
        step_data["terminated"] = np.zeros((env.num_envs, 1))
        step_data["is_first"] = np.ones_like(step_data["terminated"])
        return step_data

    def _update_target_network(self, model: DreamerV3Module) -> None:
        """Update the target critic network using exponential moving average.

        The target network is updated using a soft update mechanism where the target
        parameters are slowly moved towards the current critic parameters. This helps
        stabilize training by providing a more stable target for value estimation.

        Args:
            model (DreamerV3Module): The DreamerV3 model containing both the critic
                and target critic networks.
        """
        # Use full update (tau=1) for the first gradient step, otherwise use the configured tau
        tau = 1 if self.per_rank_gradient_steps == 0 else model.tau

        # Perform soft update: target_param = tau * current_param + (1 - tau) * target_param
        for cp, tcp in zip(model.net.critic.parameters(), model.net.target_critic.parameters()):
            tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)

    def _setup_sampling(
        self, replay_buffer: ReplayBuffer, model: DreamerV3Module, prefetch_factor: int = 1
    ) -> Tuple[Iterator[TensorDict], Iterator[TensorDict]]:
        """
        Setup sampling with prefetching.

        Args:
            replay_buffer (ReplayBuffer): The replay buffer to sample training data from.
            model (DreamerV3Module): The DreamerV3 model to get batch size and block size from.
            prefetch_factor (int): The prefetch factor for the sampling.

        Returns:
            Tuple[Iterator[TensorDict], Iterator[TensorDict]]: The iterator to sample training data from replay buffer.
        """
        prefetch_sample_train_fn = lambda: TensorDict(
            replay_buffer.sample(batch=model.batch_size, mode="train"),
            batch_size=(model.block_size, model.batch_size),
            device=self.fabric.device,
            non_blocking=True,
        ).to(torch.float32)
        prefetch_sample_val_fn = lambda: TensorDict(
            replay_buffer.sample(batch=model.batch_size, mode="report"),
            batch_size=(model.block_size, model.batch_size),
            device=self.fabric.device,
            non_blocking=True,
        ).to(torch.float32)
        prefetch_sample_train = iter(Prefetch(prefetch_sample_train_fn, prefetch_factor=prefetch_factor))
        prefetch_sample_val = iter(Prefetch(prefetch_sample_val_fn, prefetch_factor=prefetch_factor))
        return prefetch_sample_train, prefetch_sample_val
