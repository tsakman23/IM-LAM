import math
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ifo.common.nets.impala_cnn import ImpalaDecoderBlock, ImpalaEncoderBlock
from ifo.common.nets.u_net import DownsampleBlock, ResidualBlock, UpsampleBlock


class UNetWorldModel(nn.Module):
    """UNet based world model with concatenation-based conditioning from LAPO paper.

    Paper: https://arxiv.org/abs/2312.10812

    Based on implementation: https://github.com/schmidtdominik/LAPO/tree/main
    """

    def __init__(
        self,
        shape: Tuple[int, int, int],
        output_channels: int,
        condition_dim: int,
        base_channels: int = 24,
    ) -> None:
        """UNet with concatenation-based conditioning from LAPO paper.

        Args:
            shape (Tuple[int, int, int]): Shape of the input tensor (C, H, W).
            output_channels (int): Number of output channels.
            condition_dim (int): Dimension of the condition tensor.
            base_channels (int, optional): Number of base channels to use for processing.
        """
        super().__init__()
        assert len(shape) == 3, "Shape must be (C, H, W)."
        input_channels, _, _ = shape
        # Inject action condition on first layer.
        self.encoder = nn.ModuleList(
            [
                DownsampleBlock(input_channels + condition_dim, base_channels * 1),
                DownsampleBlock(base_channels * 1, base_channels * 2),
                DownsampleBlock(base_channels * 2, base_channels * 4),
                DownsampleBlock(base_channels * 4, base_channels * 8),
                DownsampleBlock(base_channels * 8, base_channels * 16),
                DownsampleBlock(base_channels * 16, base_channels * 32),
            ]
        )

        self.decoder = nn.ModuleList(
            [
                UpsampleBlock(base_channels * 32 + condition_dim, base_channels * 16),
                UpsampleBlock(base_channels * 32, base_channels * 8),
                UpsampleBlock(base_channels * 16, base_channels * 4),
                UpsampleBlock(base_channels * 8, base_channels * 2),
                UpsampleBlock(base_channels * 4, base_channels * 1),
                UpsampleBlock(base_channels * 2, base_channels * 1),
            ]
        )

        self.final = nn.Sequential(
            nn.Conv2d(base_channels * 1 + input_channels, base_channels * 1, kernel_size=(3, 3), padding=1),
            ResidualBlock(base_channels * 1, base_channels * 1, base_channels * 1 // 2),
            nn.ReLU(),
            nn.Conv2d(base_channels * 1, output_channels, kernel_size=(1, 1)),
        )

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor.
            condition (Tensor): Condition tensor.

        Returns:
            Tensor: Output tensor.
        """
        condition = condition[:, :, None, None]  # Expand by spatial dimensions.

        # Inject action at the beginning.
        _, _, h, w = x.shape
        x_ = torch.cat((x, condition.expand(-1, -1, h, w)), dim=1)
        skip_connections = []
        for layer in self.encoder:
            x_ = layer(x_)
            skip_connections.append(x_)

        # Inject action in the middle.
        _, _, h, w = skip_connections[-1].shape
        skip_connections[-1] = condition.expand(-1, -1, h, w)

        for layer in self.decoder:
            x_ = torch.cat((x_, skip_connections.pop()), dim=1)
            x_ = layer(x_)

        # Inject original input at the end.
        x_ = self.final(torch.cat((x_, x), dim=1))
        return F.tanh(x_) / 2  # Normalize to [-0.5, 0.5] range.


class ImpalaWorldModel(nn.Module):
    """Impala world model with concatenation-based conditioning from LAOM paper.

    Paper: https://arxiv.org/abs/2502.00379

    Based on implementation: https://github.com/dunnolab/laom/tree/main
    """

    def __init__(
        self,
        shape: Tuple[int, int, int],
        output_channels: int,
        condition_dim: int = 128,
        channel_multiplier: int = 1,
        encoder_channels: Sequence[int] = (16, 32, 64, 128, 256),
        encoder_num_res_blocks: int = 1,
    ):
        """Instantiate ImpalaWorldModel.

        Args:
            shape (Tuple[int, int, int]): Shape of the input tensor.
            output_channels (int): Number of output channels.
            condition_dim (int, optional): Dimension of the condition tensor.
            channel_multiplier (int, optional): Multiplier for the number of channels in the encoder and decoder.
            encoder_channels (Sequence[int], optional): List of channels for the encoder.
            encoder_num_res_blocks (int, optional): Number of residual blocks for the encoder and decoder.
        """
        super().__init__()
        assert len(shape) == 3, "Shape must be (C, H, W)."

        # Encoder
        encoder_layers = []
        for out_channels in encoder_channels:
            encoder_block = ImpalaEncoderBlock(shape[0], channel_multiplier * out_channels, encoder_num_res_blocks)
            shape = encoder_block.get_output_shape(*shape[1:])
            encoder_layers.append(encoder_block)

        self.encoder = nn.Sequential(*encoder_layers)
        self.final_encoder_shape = shape

        # Decoder
        decoder_layers = []
        shape = (shape[0] * 2, *shape[1:])  # Double number of channels because of concatenated actions.
        for out_channels in encoder_channels[::-1]:
            decoder_block = ImpalaDecoderBlock(shape[0], channel_multiplier * out_channels, encoder_num_res_blocks)
            shape = decoder_block.get_output_shape(*shape[1:])
            decoder_layers.append(decoder_block)

        self.decoder = nn.Sequential(
            *decoder_layers,
            nn.GELU(),
            nn.Conv2d(encoder_channels[0] * channel_multiplier, output_channels, kernel_size=1),
        )
        self.act_proj = nn.Linear(condition_dim, math.prod(self.final_encoder_shape))

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor.
            condition (Tensor): Condition tensor.

        Returns:
            Tensor: Output tensor.
        """
        x_ = self.encoder(x)

        # Inject action by concatenating condition to the final encoder output.
        condition_ = self.act_proj(condition).reshape(-1, *self.final_encoder_shape)
        x_ = torch.concat([x_, condition_], dim=1)

        x_ = self.decoder(x_)
        return F.tanh(x_) / 2  # Normalize to [-0.5, 0.5] range.

