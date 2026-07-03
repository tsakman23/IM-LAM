from typing import Literal

import torch
from gymnasium import spaces
from torch import Tensor, nn
from torch.distributions import Distribution

from ifo.common.nets.actor import ActionHead
from ifo.common.nets.base import FullyConnectedNeuralNetwork
from ifo.common.nets.impala_cnn import ImpalaCNNBackbone
from ifo.common.utils.functions import merge_tc


class LAPOBCActorAgent(nn.Module):
    """
    Actor class for behavior cloning fine-tuning of LAPO.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Space,
        channel_multiplier: int,
        code_dim: int,
        hidden_dim: int = 256,
        box_head_distribution: Literal["normal", "tanh_normal", "scaled_normal", "mse"] = "scaled_normal",
        categorical_head_distribution: Literal["categorical", "one_hot_categorical"] = "categorical",
    ) -> None:
        """
        Initializes the LAPOBCActorAgent.

        Args:
            observation_space (spaces.Box): The observation space of the environment.
            action_space (spaces.Space): The action space of the environment.
            channel_multiplier (int): Multiplier for the number of channels in the Impala feature extractor.
            code_dim (int): Code dimension of VQ-VAE respectively latent size of IDM.
            hidden_dim (int): Dimension of the hidden layers for the action decoder.
            box_head_distribution (Literal["normal", "tanh_normal", "scaled_normal", "mse"]): The distribution to
                use for the box head.
            categorical_head_distribution (Literal["categorical", "one_hot_categorical"]): The distribution to use
                for the categorical head.
        """
        super().__init__()
        assert len(observation_space.shape) == 4, "Observation space must be 4-dimensional."
        t, c, h, w = observation_space.shape
        self.obs_channels = c
        self.latent_policy = ImpalaCNNBackbone((t * c, h, w), channel_multiplier, code_dim)
        self.hidden_dim = hidden_dim
        self.actor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(code_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            ActionHead(
                self.hidden_dim,
                action_space,
                box_head_distribution=box_head_distribution,
                categorical_head_distribution=categorical_head_distribution,
            ),
        )

    def forward(self, x: Tensor) -> Distribution:
        """Forward pass.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Distribution: Action distribution.
        """
        x = merge_tc(x)
        with torch.no_grad():
            self.latent_policy.eval()
            x = self.latent_policy(x)
        return self.actor(x)
