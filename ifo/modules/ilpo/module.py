from typing import Callable, Dict, Optional, Union

import torch
from gymnasium.vector import VectorEnv
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import Tensor, nn
from torch.distributions import Distribution

import wandb
from ifo.common.buffer import ReplayBuffer
from ifo.common.collector import Collector
from ifo.common.module import SupervisedLightningModule
from ifo.common.utils.render import make_comp_grid


class ILPOLatentPolicyModule(SupervisedLightningModule):
    """Module for learning the ILPO latent policy."""

    def __init__(
        self,
        net: nn.Module,
        batch_size: int,
        optimizer: DictConfig,
        lr_scheduler: Optional[DictConfig] = None,
        num_workers: int = 0,
        max_grad_norm: Optional[float] = None,
        debug_transform: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        """Instantiate module for learning a latent policy.

        Args:
            net (nn.Module): Policy neural network.
            batch_size (int): Batch size.
            optimizer (DictConfig): The optimizer to use.
            lr_scheduler (Optional[DictConfig]): The learning rate scheduler to use.
            num_workers (int, optional): Number of workers for dataloader.
            max_grad_norm (Optional[float], optional): Maximum gradient norm. If not provided, no clipping will be done.
            debug_transform (Optional[Callable], optional): Transform to convert observations back to a human
                interpretable format.
        """
        super().__init__(net, batch_size, optimizer, lr_scheduler, num_workers, max_grad_norm, **kwargs)
        self.debug_transform = debug_transform

    @torch.no_grad()
    @torch.compiler.disable()
    def _log_debug_images(
        self,
        next_observation: Tensor,
        batch: TensorDict,
        log_prefix: str,
        log_name: str,
        caption: str,
    ) -> None:
        """Log debug images.

        Args:
            next_observation (Tensor): Predicted next observation.
            batch (TensorDict): Batch with ground truth next observation.
            log_prefix (str): Prefix for logging.
            log_name (str): Name of the log.
            caption (str): Caption for the log.
        """
        assert self.debug_transform is not None, "Debug transform is not provided"
        next_obs = self.debug_transform(next_observation)
        gt_next_obs = self.debug_transform(batch["observation"][:, -1])
        self.logger.experiment.log(
            {f"{log_prefix}/{log_name}": wandb.Image(make_comp_grid(next_obs, gt_next_obs), caption=caption)}
        )

    def _forward(self, batch: TensorDict, batch_idx: int, batch_len: int, prefix: str) -> TensorDict:
        """Forward pass.

        Args:
            batch (TensorDict): Batch of observations.
            batch_idx (int): Index of batch.
            batch_len (int): Length of batch.
            prefix (str): Prefix for logging.

        Returns:
            TensorDict: Dictionary with loss and other metrics to log.
        """
        min_gen_loss, exp_gen_loss, exp_next_obs, min_next_obs = self.net(batch["observation"])
        loss = min_gen_loss + exp_gen_loss

        # Log predicted expected observation vs ground truth on the last batch.
        if batch_idx == batch_len - 1 and self.debug_transform is not None:
            self._log_debug_images(
                exp_next_obs, batch, prefix, "exp_next_obs", "Predicted expected next state vs ground truth"
            )
            self._log_debug_images(
                min_next_obs, batch, prefix, "min_next_obs", "Predicted minimum next state vs ground truth"
            )

        return TensorDict(
            {
                "loss": loss,
                "min_gen_loss": min_gen_loss,
                "exp_gen_loss": exp_gen_loss,
            }
        )

    def training_step(self, batch: TensorDict, batch_idx: int, batch_len: int) -> TensorDict:
        return self._forward(batch, batch_idx, batch_len, "train")

    def validation_step(self, batch: TensorDict, batch_idx: int, batch_len: int) -> TensorDict:
        return self._forward(batch, batch_idx, batch_len, "val")

    def predict_step(self, batch: TensorDict, batch_idx: int, batch_len: int) -> TensorDict:
        """Predict latent action with latent policy.

        Args:
            batch (TensorDict): Batch of observations.
            batch_idx (int): Index of batch.
            batch_len (int): Length of batch.

        Returns:
            TensorDict: Dictionary with latent actions.
        """
        latent_action = self.net.label(batch["observation"])
        return TensorDict({"latent_action": latent_action})


class ILPORemappingModule(SupervisedLightningModule):
    """Module for ILPO remapping policy."""

    def __init__(
        self,
        net: nn.Module,
        batch_size: int,
        optimizer: DictConfig,
        lr_scheduler: Optional[DictConfig] = None,
        num_workers: int = 0,
        max_grad_norm: Optional[float] = None,
        num_steps: Union[int, float] = 16,
        epsilon_greedy: float = 0.0,
        update_epochs: int = 1,
        replay_buffer_capacity: Union[int, float] = 10000,
        **kwargs,
    ) -> None:
        """Instantiate module for learning a remapping policy.

        Args:
            net (nn.Module): The neural network model of the agent.
            batch_size (int): The batch size for training. Must be divisible by num_steps.
            optimizer (DictConfig): The optimizer to use for the ILPO module.
            lr_scheduler (Optional[DictConfig], optional): The learning rate scheduler to use for the optimizer.
            num_workers (int, optional): The number of workers for the dataloader.
            max_grad_norm (float, optional): The maximum gradient norm for gradient clipping.
            num_steps (Union[int, float], optional): The number of steps to rollout the environment.
            epsilon_greedy (float, optional): The epsilon value for epsilon-greedy exploration.
            update_epochs (int, optional): The number of epochs to update the policy.
            replay_buffer_capacity (Union[int, float], optional): The capacity of the replay buffer.
        """
        super().__init__(
            net=net,
            batch_size=batch_size,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            num_workers=num_workers,
            max_grad_norm=max_grad_norm,
        )
        self.num_steps = int(num_steps)
        self.epsilon_greedy = epsilon_greedy
        self.update_epochs = update_epochs
        self.replay_buffer_capacity = int(replay_buffer_capacity)

    def training_step(self, batch: TensorDict, batch_idx: int, batch_len: int) -> TensorDict:
        """Train for one step.

        Args:
            batch (TensorDict): Batch with observation and action.
            batch_idx (int): Index of batch.
            batch_len (int): Length of batches in dataloader.
        Returns:
            TensorDict: Dictionary with loss.
        """
        action_distribution = self.net.predict(batch["observation"])
        loss = -action_distribution.log_prob(batch["action"][:, -2]).mean()

        return TensorDict({"loss": loss})

    def forward(self, batch: TensorDict) -> Dict[str, Distribution]:
        """Predict action distribution and value.

        Args:
            batch (TensorDict): Batch with observation.

        Returns:
            Dict[str, Distribution]: Dictionary with action distribution.
        """
        action_distribution = self.net(batch["observation"])
        return {"action_distribution": action_distribution}

    def configure_data_buffer(self, num_envs: int = 1) -> ReplayBuffer:
        """
        Configure the replay buffer for storing rollout data.

        Args:
            num_envs (int): Number of parallel environments. Defaults to 1.

        Returns:
            ReplayBuffer: A data buffer configured with the specified parameters.
        """
        data_buffer = ReplayBuffer(
            capacity=self.replay_buffer_capacity,
            num_envs=num_envs,
            block_size=self.net.world_model_block_size,
            device=torch.device("cpu"),
            is_frame_stacked=True,
        )
        return data_buffer

    def training_collector(self, env: VectorEnv, random_seed: Optional[int] = None) -> Collector:
        """Configure the collector for training the PPO module.

        Args:
            env (VectorEnv): The environment to collect data from.
            random_seed (Optional[int]): The random seed to use for resetting the environment.

        Returns:
            Collector: The collector for the PPO module.
        """
        collector = Collector(
            env=env,
            policy=self,
            random_seed=random_seed,
        )
        return collector

    def validation_collector(self, env: VectorEnv, random_seed: Optional[int] = None) -> Collector:
        """Configure the collector for validating the PPO module.

        Args:
            env (VectorEnv): The environment to collect data from.
            random_seed (Optional[int]): The random seed to use for resetting the environment.
        """
        collector = Collector(
            env=env,
            policy=self,
            random_seed=random_seed,
        )
        return collector
