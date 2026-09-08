from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from torch import Tensor
from torch.distributions import Distribution

from src.models.action_head import ActionHead
from src.models.impala_cnn import ImpalaCNNBackbone
from src.models.quantizer import FiniteScalarQuantizer, IdentityQuantizer, VectorQuantizerEMA
from src.models.world_model import ImpalaWorldModel, UNetWorldModel
from src.utils.helpers import merge_tc, orthogonal_init


class LAPOMasks(nn.Module):
    """LAPO-masks: Latent action model operating in masked pixel space.

    CNN encoder maps (current_obs, future_obs) -> latent action -> quantize -> decode -> predicted future obs.
    Slot attention masks are applied externally before calling forward.
    """

    def __init__(
        self,
        observation_shape: Tuple[int, int, int, int],
        action_space: spaces.Space,
        channel_multiplier: int,
        quantizer: Union[VectorQuantizerEMA, FiniteScalarQuantizer, IdentityQuantizer],
        world_model: Union[UNetWorldModel, ImpalaWorldModel],
        latent_action_dim: int,
        num_latents: int = 1,
        future_obs_offset: int = 10,
        frame_stack: int = 3,
        encoder_deep: bool = False,
        encoder_num_res_blocks: int = 2,
    ) -> None:
        """
        Args:
            observation_shape: (T, C, H, W) shape of temporal observation sequence.
            action_space: gymnasium action space for action decoder.
            channel_multiplier: IMPALA CNN channel multiplier.
            quantizer: information bottleneck.
            world_model: decoder (UNet or Impala).
            latent_action_dim: latent action dimension.
            num_latents: number of discrete latents for VQ factorization.
            future_obs_offset: temporal offset for future observations.
            frame_stack: number of stacked frames.
            encoder_deep: use deeper encoder architecture.
            encoder_num_res_blocks: number of residual blocks per encoder block.
        """
        super().__init__()
        assert len(observation_shape) == 4
        t, c, h, w = observation_shape
        assert t == future_obs_offset + frame_stack

        self.obs_channels = c
        self.latent_action_dim = latent_action_dim
        self.num_latents = num_latents
        self.future_obs_offset = future_obs_offset
        self.frame_stack = frame_stack

        # Encoder: concat of current + future filtered frames
        self.encoder = ImpalaCNNBackbone(
            (2 * frame_stack * c, h, w), channel_multiplier, latent_action_dim,
            num_res_blocks=encoder_num_res_blocks, encoder_deep=encoder_deep,
        )

        # Decoder: world model
        self.decoder = world_model

        # Information bottleneck
        self.information_bottleneck = quantizer

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            x: (B, T, C, H, W) observation sequence.
            mask: (B, T, 1, H, W) optional masks applied to observations.

        Returns:
            next_observation: (B, C, H, W) predicted future frame.
            vq_loss: quantization loss.
            perplexity: codebook usage metric.
            latent_action: (B, latent_action_dim) pre-quantization encoder output.
        """
        if mask is not None:
            idm_observation = x * mask
        else:
            idm_observation = x

        idm_current = merge_tc(idm_observation[:, :self.frame_stack])
        idm_future = merge_tc(idm_observation[:, -self.frame_stack:])

        latent_action = self.encoder(torch.cat([idm_current, idm_future], dim=1))

        # Factorize and quantize
        b, c = latent_action.shape
        factorized = latent_action.view(b, c // self.num_latents, self.num_latents)
        quantized, vq_loss, perplexity = self.information_bottleneck(factorized)
        quantized = quantized.view(b, c)

        # FDM: predict future from current obs + latent action
        fdm_current = merge_tc(x[:, :self.frame_stack])
        next_observation = self.decoder(fdm_current, quantized)

        return next_observation, vq_loss, perplexity, latent_action

    def label(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Extract latent action (encoder only).

        Args:
            x: (B, T, C, H, W) observation sequence.
            mask: (B, T, 1, H, W) optional masks.

        Returns:
            Latent action (B, latent_action_dim).
        """
        if mask is not None:
            idm_observation = x * mask
        else:
            idm_observation = x

        idm_current = merge_tc(idm_observation[:, :self.frame_stack])
        idm_future = merge_tc(idm_observation[:, -self.frame_stack:])
        return self.encoder(torch.cat([idm_current, idm_future], dim=1))
