import torch
import torch.nn.functional as F
from gymnasium import spaces
from torch import Tensor, nn
from torch.distributions import Distribution
from torch.nn import TransformerEncoder, TransformerEncoderLayer

from ifo.common.nets.actor import ActionHead
from ifo.common.nets.impala_cnn import ImpalaCNNBackbone
from ifo.common.utils.functions import merge_tc


class ImpalaActorAgent(nn.Module):
    """
    Agent class with actor network and Impala feature extractor.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Space,
        channel_multiplier: int = 1,
    ) -> None:
        """
        Initializes the ImpalaActorAgent.

        Args:
            observation_space (spaces.Box): The observation space of the environment.
            action_space (spaces.Space): The action space of the environment.
            channel_multiplier (int, optional): Multiplier for the number of channels in the Impala feature extractor.
        """
        super().__init__()
        assert len(observation_space.shape) == 4, "Observation space must be 4-dimensional."
        t, c, h, w = observation_space.shape
        self.obs_channels = c
        self.feature_extractor = ImpalaCNNBackbone((t * c, h, w), channel_multiplier)
        self.actor = nn.Sequential(nn.ReLU(), ActionHead(self.feature_extractor.out_features, action_space))

    def forward(self, x: Tensor) -> Distribution:
        """Forward pass.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Distribution: Action distribution.
        """
        x = merge_tc(x)
        x = self.feature_extractor(x)
        return self.actor(x)


class ImpalaTransformerActorAgent(nn.Module):
    """
    Agent class with Transformer actor network and Impala feature extractor.

    The Impala feature extractor is a CNN that extracts features from the input tensor. These features are encoded
    with a learned positional embedding and passed to a causal Transformer decoder.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Space,
        channel_multiplier: int = 1,
        nhead: int = 8,
        num_layers: int = 6,
    ) -> None:
        """
        Initializes the ImpalaTransformerActorAgent.

        Args:
            observation_space (spaces.Box): The observation space of the environment.
            action_space (spaces.Space): The action space of the environment.
            channel_multiplier (int, optional): Multiplier for the number of channels in the Impala feature extractor.
            nhead (int, optional): Number of attention heads in the Transformer.
            num_layers (int, optional): Number of layers in the Transformer.
        """
        super().__init__()
        assert len(observation_space.shape) == 4, "Observation space must be 4-dimensional."
        t, c, h, w = observation_space.shape
        self.obs_channels = c
        self.feature_extractor = ImpalaCNNBackbone((c, h, w), channel_multiplier)
        self.position_embedding = nn.Embedding(t, self.feature_extractor.out_features)
        self.transformer = TransformerEncoder(
            encoder_layer=TransformerEncoderLayer(
                d_model=self.feature_extractor.out_features,
                nhead=nhead,
                activation=F.gelu,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
        )
        self.actor = nn.Sequential(nn.ReLU(), ActionHead(self.feature_extractor.out_features, action_space))

    def forward(self, x: Tensor) -> Distribution:
        """Forward pass.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Distribution: Action distribution.
        """
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)  # [B*T, C, H, W]
        x = self.feature_extractor(x)  # [B*T, D]
        x = x.view(b, t, -1) + self.position_embedding(torch.arange(t, device=x.device))  # [B, T, D]
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(t, device=x.device)
        x = self.transformer(x, mask=causal_mask, is_causal=True)  # [B, T, D]
        return self.actor(x)

    def predict(self, x: Tensor) -> Distribution:
        """Predict the action distribution for the given observation.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Distribution: Action distribution.
        """
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)  # [B*T, C, H, W]
        x = self.feature_extractor(x)  # [B*T, D]
        x = x.view(b, t, -1) + self.position_embedding(torch.arange(t, device=x.device))  # [B, T, D]
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(t, device=x.device)
        x = self.transformer(x, mask=causal_mask, is_causal=True)  # [B, T, D]
        return self.actor(x[:, -1])  # [B, A]
