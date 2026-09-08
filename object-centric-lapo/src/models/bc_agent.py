import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from torch import Tensor
from torch.distributions import Distribution

from src.models.action_head import ActionHead, FullyConnectedNeuralNetwork
from src.models.impala_cnn import ImpalaCNNBackbone
from src.utils.helpers import merge_tc


class BCAgentPixel(nn.Module):
    """Pixel-based behavior cloning agent for both LAPO variants.

    Phase A: encoder + latent_head predict latent actions from frame-stacked images.
    Phase B: encoder frozen, actor predicts real actions.
    """

    def __init__(
        self,
        observation_shape: tuple,
        action_space: spaces.Space,
        channel_multiplier: int,
        latent_action_dim: int,
        hidden_dim: int = 256,
        action_head_dim: int = 64,
        encoder_deep: bool = False,
        encoder_num_res_blocks: int = 2,
    ) -> None:
        """
        Args:
            observation_shape: (T, C, H, W) observation shape.
            action_space: gymnasium action space.
            channel_multiplier: IMPALA backbone channel multiplier.
            latent_action_dim: dimension of latent actions (from LAPO IDM).
            hidden_dim: MLP hidden layer size for actor.
            action_head_dim: final MLP output / action head input dim.
            encoder_deep: use deeper encoder architecture.
            encoder_num_res_blocks: number of residual blocks per encoder block.
        """
        super().__init__()
        assert len(observation_shape) == 4
        t, c, h, w = observation_shape
        self.encoder = ImpalaCNNBackbone(
            (t * c, h, w), channel_multiplier, latent_action_dim,
            num_res_blocks=encoder_num_res_blocks, encoder_deep=encoder_deep,
        )
        self.hidden_dim = hidden_dim
        # Phase B: predict real actions
        self.actor = nn.Sequential(
            nn.ReLU(),
            nn.Linear(latent_action_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            ActionHead(
                self.hidden_dim,
                action_space,
                box_head_distribution="mse",
            ),
        )

    def forward_latent(self, x: Tensor) -> Tensor:
        """Phase A: predict latent action from observation.

        Args:
            x: (B, T, C, H, W) observation sequence.

        Returns:
            (B, latent_action_dim) predicted latent action.
        """
        x = merge_tc(x)
        return self.encoder(x)

    def forward(self, x: Tensor) -> Distribution:
        """Phase B: predict action distribution (encoder frozen).

        Args:
            x: (B, T, C, H, W) observation sequence.

        Returns:
            Action distribution.
        """
        x = merge_tc(x)
        with torch.no_grad():
            self.encoder.eval()
            latent_action = self.encoder(x)
        return self.actor(latent_action)
