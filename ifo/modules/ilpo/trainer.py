from time import time
from typing import Literal, Optional, Union

import torch
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from lightning.fabric import Fabric
from tensordict import TensorDict
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import MeanMetric

from ifo.common.buffer import ReplayBuffer
from ifo.common.collector import Collector, ExplorationMode
from ifo.common.trainer import LightningTrainer
from ifo.common.utils.utility import rename_key
from ifo.modules.ilpo.module import ILPORemappingModule


class ILPOTrainer(LightningTrainer):
    """
    Trainer class for stage 2 of ILPO.

    Paper: https://arxiv.org/abs/1805.07914
    """

    def __init__(
        self,
        fabric: Fabric,
        compile: bool = False,
        checkpoint_dir: str = "./checkpoints",
        validation_criteria: Literal["min", "max"] = "max",
        validation_frequency: Union[int, float] = 1,
        validation_unit: Literal["iteration", "step"] = "iteration",
        random_seed: int = 0,
        validation_steps: Union[int, float] = 1e4,
        validation_exploration_mode: ExplorationMode = ExplorationMode.SAMPLE,
        max_steps: Union[int, float] = 1e6,
        **kwargs,
    ) -> None:
        """Initialize the ILPO trainer.

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
        self.action_repeat = 1  # This is set in the fit method.
        self.global_step_action_repeat = 0  # This is set in the fit method.
        self.next_validation_step = 0
        self.supported_action_spaces = (spaces.Box, spaces.Discrete)
        super().__init__(fabric, compile, checkpoint_dir, validation_criteria, validation_unit, random_seed, **kwargs)

    def fit(
        self,
        model: ILPORemappingModule,
        train_env: VectorEnv,
        val_env: Optional[VectorEnv] = None,
        checkpoint: Optional[str] = None,
    ) -> None:
        """Fit the model.

        Args:
            model (ILPORemappingModule): The model to be trained. This should be an instance of a LightningModule.
            train_env (VectorEnv): The environment for training the agent.
            val_env (Optional[VectorEnv]): The environment for validating the agent. If not provided,
                validation will be skipped.
            checkpoint (Optional[str], optional): Path to a specific checkpoint to load or directory to load the latest
                checkpoint from. If provided, this will override automatic loading from the checkpoint directory.
        """
        assert isinstance(
            train_env.single_action_space, self.supported_action_spaces
        ), f"Action space {train_env.single_action_space} is not supported. Supported spaces: {self.supported_action_spaces}"
        assert self.fabric.world_size == 1, "ILPO only supports single-device training."

        # Setup optimizer and scheduler.
        optimizer, lr_scheduler = model.configure_optimizers()

        # Setup model and optimizer.
        model.training_step = torch.compile(model.training_step, mode="max-autotune", disable=not self.compile)
        model.validation_step = torch.compile(model.validation_step, mode="max-autotune", disable=not self.compile)
        model.forward = torch.compile(model.forward, mode="max-autotune", disable=not self.compile)
        model, optimizer = self.fabric.setup(model, optimizer)

        # Assemble checkpoint state (current epoch and global step will be added in save).
        state = {"model": model, "optimizer": optimizer, "lr_scheduler": lr_scheduler}

        # Load last checkpoint if available.
        self.global_step, self.global_epoch, self.global_iteration = self.load_checkpoint(
            self.fabric, state, self.checkpoint_dir, checkpoint
        )

        # Check if we even need to train here.
        if self.global_step >= self.max_steps:
            return

        # Setup data buffer and collector.
        data_buffer = model.configure_data_buffer(num_envs=train_env.num_envs)
        train_collector = model.training_collector(train_env, random_seed=self.random_seed)
        val_collector = None if val_env is None else model.validation_collector(val_env, random_seed=self.random_seed)

        self.action_repeat = (
            train_env.get_wrapper_attr("action_repeat") if hasattr(train_env, "get_wrapper_attr") else 1
        )
        self.global_step_action_repeat = self.global_step * self.action_repeat
        self.policy_steps_per_iteration = model.num_steps

        trainer_pbar = self.tqdm_wrapper(
            range(self.global_step, self.max_steps, self.policy_steps_per_iteration),
            initial=self.global_step,
            total=self.max_steps,
            position=0,
            desc="Training",
        )

        for steps in trainer_pbar:
            self.fabric.log("trainer/iteration", self.global_iteration, step=self.global_step_action_repeat)
            rollout_data = self.rollout_loop(model, train_collector, model.num_steps)
            self.fabric.call("rollout_loop_end", **rename_key(locals(), "self", "trainer"))
            data_buffer.extend(rollout_data)

            if len(data_buffer) * train_env.num_envs >= model.batch_size:
                self.train_loop(model, optimizer, lr_scheduler, data_buffer)
                self.fabric.call("train_loop_end", **rename_key(locals(), "self", "trainer"))

            if self.should_validate(self.validation_unit):
                self.val_loop(model, val_collector)
                self.save_best(state=state, on_validation=val_env is not None)
                self.fabric.call("val_loop_end", **rename_key(locals(), "self", "trainer"))

            trainer_pbar.update(self.policy_steps_per_iteration)
            metrics = {"train_return": self._current_train_loss, "val_return": self._current_val_loss}
            self._format_iterable(trainer_pbar, metrics)

            # Stop training if max_steps is reached.
            if self.max_steps is not None and self.global_step >= self.max_steps:
                self.should_stop = True

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
    def rollout_loop(self, model: ILPORemappingModule, collector: Collector, max_steps: int) -> TensorDict:
        """Collect rollout data from the environment using the collector.

        Args:
            model (ILPORemappingModule): The PPO model to use for collecting data.
            collector (Collector): The collector to use for collecting data.
            max_steps (int): The maximum number of steps to collect.

        Returns:
            TensorDict: The collected rollout data.
        """
        start_time = time()
        model.eval()

        rollout_pbar = self.tqdm_wrapper(
            range(max_steps), total=max_steps, position=1, desc="Rollout Loop", leave=False
        )

        rollout_data, episode_return, episode_length = collector.rollout(
            max_steps=max_steps // collector.env.num_envs,
            callback=lambda x: rollout_pbar.update(collector.env.num_envs),
            exploration_mode=ExplorationMode.EPSILON_GREEDY,
            epsilon_greedy=model.epsilon_greedy,
        )

        # Log episode return and length.
        self.global_step += max_steps
        self.global_step_action_repeat = self.global_step * self.action_repeat

        if episode_return is not None:
            self.fabric.log_dict(
                {"train/episode_return": episode_return, "train/episode_length": episode_length},
                step=self.global_step_action_repeat,
            )
            self._prev_train_loss = self._current_train_loss
            self._current_train_loss = episode_return
            self._format_iterable(rollout_pbar, {"return": episode_return}, "train")
        self.fabric.log("train/env_throughput", max_steps / (time() - start_time), step=self.global_step_action_repeat)
        self.close_tqdm_wrapper(rollout_pbar, max_steps)
        return rollout_data

    def train_loop(
        self,
        model: ILPORemappingModule,
        optimizer: Optimizer,
        lr_scheduler: Optional[LRScheduler],
        data_buffer: ReplayBuffer,
    ) -> None:
        """Train the model.

        Args:
            model (ILPORemappingModule): The model to train.
            optimizer (Optimizer): The optimizer to use for training.
            lr_scheduler (Optional[LRScheduler]): The learning rate scheduler to use for training.
            data_buffer (ReplayBuffer): The data buffer to use for training.
        """
        # Train policy and value model.
        mean_metrics = None
        throughput = MeanMetric().to(self.fabric.device)
        train_loss = MeanMetric().to(self.fabric.device)
        self.fabric.log("train/learning_rate", optimizer.param_groups[0]["lr"], step=self.global_step_action_repeat)
        dataset = data_buffer.dataset
        dataloader = model.training_dataloader(dataset)
        dataloader = self.fabric.setup_dataloaders(dataloader)
        train_pbar = self.tqdm_wrapper(
            range(model.update_epochs * len(dataloader)),
            total=model.update_epochs * len(dataloader),
            position=1,
            desc="Training Loop",
            leave=False,
        )
        for epoch in range(model.update_epochs):
            self.fabric.call("train_epoch_start", **rename_key(locals(), "self", "trainer"))
            for batch_idx, batch in enumerate(dataloader):
                start_time = time()
                # Train one step.
                step_dict = model.training_step(batch, batch_idx=batch_idx, batch_len=len(data_buffer))
                optimizer.zero_grad()
                self.fabric.backward(step_dict["loss"])

                if model.max_grad_norm is not None:
                    total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), model.max_grad_norm)
                    self.fabric.log("train/total_grad_norm", total_grad_norm, step=self.global_step_action_repeat)

                optimizer.step()

                # Update mean metrics.
                if not mean_metrics:
                    mean_metrics = {k: MeanMetric().to(self.fabric.device) for k in step_dict.keys()}
                for k, metric in mean_metrics.items():
                    metric(step_dict[k])
                throughput(batch.shape[0] / (time() - start_time))
                train_loss(step_dict["loss"])

                if lr_scheduler is not None:
                    lr_scheduler.step()

                # Update progress bar.
                self._format_iterable(train_pbar, train_loss.compute(), "train")
                train_pbar.update(1)

            self.global_epoch += model.update_epochs
            self.fabric.call("train_epoch_end", **rename_key(locals(), "self", "trainer"))

        # Log training metrics.
        assert mean_metrics is not None, "No training metrics were computed."
        self.fabric.log_dict(
            {f"train/{k}": v.compute() for k, v in mean_metrics.items()}, step=self.global_step_action_repeat
        )
        self.fabric.log(
            "train/throughput", throughput.compute() * self.fabric.world_size, step=self.global_step_action_repeat
        )
        self.global_iteration += 1
        self.close_tqdm_wrapper(train_pbar, model.update_epochs * len(dataloader))

    @torch.no_grad()
    def val_loop(self, model: ILPORemappingModule, val_collector: Optional[Collector] = None) -> None:
        """Run validation to evaluate the current model performance.

        Args:
            model (ILPORemappingModule): The ILPO model to evaluate.
            val_collector (Optional[Collector], optional): Environment collector for
                validation. If None, validation is skipped. Defaults to None.
        """
        # Skip validation if no validation environment is provided.
        if val_collector is None:
            return

        val_pbar = self.tqdm_wrapper(
            range(self.validation_steps),
            total=self.validation_steps,
            position=1,
            desc="Validation Loop",
            unit="steps",
            leave=False,
        )

        model.eval()
        _, episode_return, episode_length = val_collector.rollout(
            max_steps=self.validation_steps // val_collector.env.num_envs,
            callback=lambda x: val_pbar.update(val_collector.env.num_envs),
            exploration_mode=self.validation_exploration_mode,
            reset_before_rollout=True,
        )

        # Get and log training rewards and episode lengths
        self._prev_val_loss = self._current_val_loss
        if episode_return is not None:
            self.fabric.log_dict(
                {"val/episode_return": episode_return, "val/episode_length": episode_length},
                step=self.global_step_action_repeat,
            )
            self._format_iterable(val_pbar, {"return": episode_return}, "val")
            self._current_val_loss = episode_return

        self.close_tqdm_wrapper(val_pbar, self.validation_steps)

    def should_validate(self, unit: Literal["iteration", "step"]) -> bool:
        """Determine whether validation should be run at the current training state.

        Args:
            unit (Literal["iteration", "step"]): The unit to check for validation timing.
                Must match the trainer's configured validation_unit for validation to
                be triggered.

        Returns:
            bool: True if validation should be run now, False otherwise.
        """
        if unit == "iteration" and self.validation_unit == "iteration":
            return self.global_iteration % self.validation_frequency == 0
        elif unit == "step" and self.validation_unit == "step":
            if self.global_step >= self.next_validation_step:
                self.next_validation_step = self.global_step + self.validation_frequency
                return True
        return False
