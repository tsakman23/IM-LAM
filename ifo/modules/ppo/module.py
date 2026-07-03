from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from gymnasium.vector import VectorEnv
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import Tensor
from torch.distributions import Distribution

from ifo.common.buffer import ReplayBuffer
from ifo.common.collector import Collector
from ifo.common.module import SupervisedLightningModule


class PPOModule(SupervisedLightningModule):
    """
    PPO (Proximal Policy Optimization) module for reinforcement learning.

    This module implements the PPO algorithm for training reinforcement learning agents.
    It handles the collection of experience, computation of advantages, and policy updates
    using the PPO objective. The module supports both continuous and discrete action spaces
    and includes features like GAE (Generalized Advantage Estimation), value function clipping,
    and entropy regularization.
    """

    def __init__(
        self,
        net: nn.Module,
        batch_size: int,
        optimizer: DictConfig,
        lr_scheduler: Optional[DictConfig] = None,
        num_workers: int = 0,
        max_grad_norm: Optional[float] = 0.5,
        num_steps: int = 4096,
        gamma: float = 0.999,
        gae_lambda: float = 0.95,
        update_epochs: int = 3,
        norm_advantage: bool = True,
        clip_coef: float = 0.2,
        clip_vloss: bool = True,
        entropy_coef: float = 0.01,
        value_loss_coef: float = 0.5,
    ) -> None:
        """
        Initialize the PPO module.

        Args:
            net (nn.Module): The neural network model of the agent.
            batch_size (int): The batch size for training. Must be divisible by num_steps.
            optimizer (DictConfig): The optimizer to use for the PPO module.
            lr_scheduler (Optional[DictConfig], optional): The learning rate scheduler to use for the optimizer.
            num_workers (int, optional): The number of workers for the dataloader.
            max_grad_norm (float, optional): The maximum gradient norm for gradient clipping.
            num_steps (int, optional): The number of steps to collect data for each update.
            gamma (float, optional): The discount factor for future rewards.
            gae_lambda (float, optional): The lambda parameter for Generalized Advantage Estimation.
            update_epochs (int, optional): The number epochs to perform on a single replay buffer.
            norm_advantage (bool, optional): Whether to normalize advantages.
            clip_coef (float, optional): The clipping coefficient for PPO.
            clip_vloss (bool, optional): Whether to clip the value loss.
            entropy_coef (float, optional): The coefficient for the entropy loss.
            value_loss_coef (float, optional): The coefficient for the value function loss.
        """

        super().__init__(
            net=net,
            batch_size=batch_size,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            num_workers=num_workers,
            max_grad_norm=max_grad_norm,
        )
        self.num_steps = num_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.update_epochs = update_epochs
        self.norm_advantage = norm_advantage
        self.clip_coef = clip_coef
        self.clip_vloss = clip_vloss
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef

    def training_step(self, batch: TensorDict, batch_idx: int, batch_len: int) -> TensorDict:
        """Train for one step.

        Args:
            batch (TensorDict): Batch with observation and action.
            batch_idx (int): Index of batch.
            batch_len (int): Length of batches in dataloader.
        Returns:
            TensorDict: Dictionary with loss and other metrics to log.
        """
        action_distribution, value = self.net(batch["observation"])

        # Clipped surrogate loss
        log_ratio = action_distribution.log_prob(batch["action"][:, -1]) - batch["log_prob"][:, -1]
        ratio = log_ratio.exp()
        policy_loss = batch["advantage"][:, -1] * ratio
        policy_loss_clipped = batch["advantage"][:, -1] * ratio.clamp(1 - self.clip_coef, 1 + self.clip_coef)
        policy_loss = -torch.min(policy_loss, policy_loss_clipped).mean()

        # Value loss
        if self.clip_vloss:
            value_clipped = batch["value"][:, -1] + (value - batch["value"][:, -1]).clamp(
                -self.clip_coef, self.clip_coef
            )
            value_loss = (value - batch["return"][:, -1]).pow(2)
            value_loss_clipped = (value_clipped - batch["return"][:, -1]).pow(2)
            value_loss = torch.max(value_loss, value_loss_clipped).mean()
        else:
            value_loss = (batch["return"][:, -1] - value).pow(2).mean()

        # Entropy loss
        entropy_loss = action_distribution.entropy().mean()
        loss = policy_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss

        # Compute metrics
        return_var = batch["return"][:, -1].var()
        explained_var = 1 - (batch["return"][:, -1] - batch["value"][:, -1]).var() / (return_var + 1e-8)
        clip_fraction = ((ratio - 1.0).abs() > self.clip_coef).float()
        approx_kl_div = ((ratio - 1) - log_ratio).mean()

        return TensorDict(
            {
                "loss": loss,
                "value_loss": value_loss,
                "policy_loss": policy_loss,
                "entropy_loss": entropy_loss,
                "approx_kl_div": approx_kl_div,
                "explained_variance": explained_var,
                "clip_fraction": clip_fraction,
            }
        )

    def forward(self, batch: TensorDict) -> Dict[str, Union[Distribution, Tensor]]:
        """Predict action distribution and value.

        Args:
            batch (TensorDict): Batch with observation.

        Returns:
            Dict[str, Union[Distribution, Tensor]]: Dictionary with action distribution and value.
        """
        action_distribution, value = self.net(batch["observation"])
        return {"action_distribution": action_distribution, "value": value}

    def configure_data_buffer(self, num_envs: int = 1, block_size: int = 1) -> ReplayBuffer:
        """
        Configure the replay buffer for storing rollout data.

        Args:
            num_envs (int): Number of parallel environments. Defaults to 1.
            block_size (int): Size of the block to stack observations. Defaults to 1.

        Returns:
            ReplayBuffer: A data buffer configured with the specified parameters.
        """
        data_buffer = ReplayBuffer(
            capacity=self.num_steps//num_envs,
            num_envs=num_envs,
            block_size=block_size,
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
            return_log_prob=True,
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
