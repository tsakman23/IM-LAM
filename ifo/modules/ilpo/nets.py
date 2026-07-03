from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch import Tensor, nn
from torch.distributions import Distribution

from ifo.common.nets.actor import ActionHead
from ifo.common.nets.base import FullyConnectedNeuralNetwork
from ifo.common.nets.impala_cnn import ImpalaCNNBackbone
from ifo.common.nets.world_model import UNetWorldModel
from ifo.common.utils.functions import merge_tc


class ILPOLatentPolicy(nn.Module):
    """Latent policy class that encapsulates the latent policy and forwards dynamics model."""

    def __init__(
        self,
        observation_space: spaces.Box,
        channel_multiplier: int,
        code_dim: int,
        policy_block_size: Optional[int] = None,
    ) -> None:
        """Instantiate ILPOLatentPolicy.

        Args:
            observation_space (spaces.Box): The observation space of the environment.
            channel_multiplier (int): Channel multiplier of ImpalaCNN encoder.
            code_dim (int): Dimension of latent action.
            policy_block_size (int, optional): Block size of the policy. If None, the block size is set to the block
                size of the observations minus one.
        """
        super().__init__()
        assert len(observation_space.shape) == 4, "Observation space must be 4-dimensional."
        t, c, h, w = observation_space.shape
        self.obs_channels = c
        self.code_dim = code_dim
        self.policy_block_size = t - 1 if policy_block_size is None else policy_block_size
        self.encoder = ImpalaCNNBackbone((self.policy_block_size * c, h, w), channel_multiplier, 128)
        self.mlp = nn.Sequential(nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, self.code_dim))
        self.latent_policy = nn.Sequential(self.encoder, self.mlp)
        self.world_model = UNetWorldModel(((t - 1) * c, h, w), c, self.code_dim)
        self.start_dim = (t - 1 - self.policy_block_size) * c
        self.stop_dim = -1 * c
        self.register_buffer("code", F.one_hot(torch.arange(self.code_dim)).float())  # (A, A)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward method of the latent policy.

        Args:
            x (Tensor): Observation o_{t-block_size+1}, o_t, o_{t+1}.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: Minimum generator loss, expected generator loss,
                expected next observation, minimum next observation.
        """
        x = merge_tc(x)
        b, c, h, w = x.shape

        # This is the legacy implementation of conditioning the world model on each code.
        # next_obs = []
        # for i in range(self.code_dim):
        #     next_obs.append(self.world_model(x[:, :self.stop_dim], self.code[None, i].expand(b, -1)))
        # next_obs = torch.stack(next_obs, dim=1)  # [B, Z, C, H, W]
        # This is the fast implementation of conditioning the world model on each code.
        obs = x[:, None, : self.stop_dim].expand(-1, self.code_dim, -1, -1, -1).reshape(b * self.code_dim, -1, h, w)
        code = self.code[None].expand(b, -1, -1).reshape(b * self.code_dim, self.code_dim)
        next_obs = self.world_model(obs, code)
        next_obs = next_obs.view(b, self.code_dim, -1, h, w)  # [B, Z, C, H, W]

        gen_loss = torch.mean((next_obs - x[:, None, self.stop_dim :]) ** 2, dim=(-1, -2, -3))  # [B, Z]
        min_gen_loss, min_gen_loss_idx = torch.min(gen_loss, dim=-1)  # Take smallest loss over actions.
        min_gen_loss = min_gen_loss.mean()
        min_next_obs = next_obs[torch.arange(b, device=next_obs.device), min_gen_loss_idx]

        latent_action = self.latent_policy(x[:, self.start_dim : self.stop_dim])
        latent_action_probs = F.softmax(latent_action, dim=-1)  # [B, Z]
        exp_next_obs = torch.sum(latent_action_probs[..., None, None, None] * next_obs.detach(), dim=1)  # [B, C, H, W]
        exp_gen_loss = F.mse_loss(exp_next_obs, x[:, self.stop_dim :])
        # exp_gen_loss = F.cross_entropy(latent_action, min_gen_loss_idx.detach())
        return min_gen_loss, exp_gen_loss, exp_next_obs, min_next_obs

    def label(self, x: Tensor) -> Tensor:
        """Label the latent action."""
        x = merge_tc(x)
        z = self.latent_policy(x[:, self.start_dim : self.stop_dim])
        z_ids = torch.argmax(z, dim=-1)  # [B]
        z = self.code[z_ids]  # [B, Z]
        return z


class ILPOActorAgent(nn.Module):
    """Policy class that maps the latent action to the real action of the environment."""

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Space,
        channel_multiplier: int,
        code_dim: int,
        world_model_block_size: Optional[int] = None,
    ) -> None:
        """Instantiate policy.

        Args:
            observation_space (spaces.Box): The observation space of the environment.
            action_space (spaces.Space): The action space of the environment.
            channel_multiplier (int, optional): Multiplier for the number of channels in the Impala feature extractor.
            code_dim (int): Code dimension of VQ-VAE respectively latent size of IDM.
            world_model_block_size (int, optional): Block size for the world model. If None, the block
                size is set to the block size of the observations plus one.
        """
        super().__init__()
        assert len(observation_space.shape) == 4, "Observation space must be 4-dimensional (T, C, H, W)."
        t, c, h, w = observation_space.shape
        self.obs_channels = c
        self.code_dim = code_dim
        # Encoder and MLP form the latent policy.
        self.encoder = ImpalaCNNBackbone((t * c, h, w), channel_multiplier, 128)
        self.mlp = nn.Sequential(nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, self.code_dim))

        self.world_model_block_size = t + 1 if world_model_block_size is None else world_model_block_size
        assert self.world_model_block_size > t, "World model block size must be in [t+1, inf]."
        self.world_model = UNetWorldModel((self.world_model_block_size - 1) * c, c, self.code_dim)
        self.stop_dim = -1 * c
        self.start_dim = (self.world_model_block_size - 1 - t) * c

        # RL policy head.
        self.actor = nn.Sequential(
            FullyConnectedNeuralNetwork(code_dim + 128, 64, [192, 128]), ActionHead(64, action_space)
        )
        self.register_buffer("code", F.one_hot(torch.arange(self.code_dim)).float())  # (A, A)

    def min_latent_action(self, x: Tensor) -> Tensor:
        """Compute the minimum latent action for a given observation.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The minimum latent action.
        """
        b, c, h, w = x.shape

        # This is the legacy implementation of conditioning the world model on each code.
        # next_obs = []
        # for i in range(self.code_dim):
        #     next_obs.append(self.encoder(self.world_model(x[:, :self.stop_dim], self.code[None, i].expand(b, -1))))
        # next_obs = torch.stack(next_obs, dim=1)  # [B, Z, 128]
        # This is the fast implementation of conditioning the world model on each code.
        obs = x[:, None, : self.stop_dim].expand(-1, self.code_dim, -1, -1, -1).reshape(b * self.code_dim, -1, h, w)
        code = self.code[None, :, :].expand(b, -1, -1).reshape(b * self.code_dim, self.code_dim)
        wm_output = self.world_model(obs, code)
        next_encoded_obs = self.encoder(wm_output)
        next_encoded_obs = next_encoded_obs.view(b, self.code_dim, -1)  # [B, Z, 128]

        gen_loss = torch.mean((next_encoded_obs - self.encoder(x[:, self.stop_dim :])[:, None]) ** 2, dim=-1)  # [B, Z]
        min_latent_action_ids = torch.argmin(gen_loss, dim=-1)  # Take smallest loss over actions.
        min_latent_action = self.code[min_latent_action_ids]  # [B, Z]
        return min_latent_action

    def predict(self, x: Tensor) -> Distribution:
        """Predict action distribution.

        Args:
            x (Tensor): The input tensor with observation o_{t-world_model_block_size+2}, o_t, o_{t+1}.

        Returns:
            Distribution: Action distribution.
        """
        with torch.no_grad():
            x = merge_tc(x)
            h = self.encoder(x[:, self.start_dim : self.stop_dim])
            z = self.min_latent_action(x)
        action_dist = self.actor(torch.cat([z, h], dim=-1))
        return action_dist

    def forward(self, x: Tensor) -> Distribution:
        """Forward pass

        Args:
            x (Tensor): The input tensor with observation o_{t-block_size+1}, o_t.

        Returns:
            Distribution: Action distribution.
        """
        x = merge_tc(x)
        h = self.encoder(x)
        z = self.mlp(h)  # [B, Z]
        z_ids = torch.argmax(z, dim=-1)  # [B]
        z = self.code[z_ids]  # [B, Z]
        action_dist = self.actor(torch.cat([z, h], dim=-1))
        return action_dist
