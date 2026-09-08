import math
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from src.models.impala_cnn import ImpalaDecoderBlock, ImpalaEncoderBlock


# --- UNet building blocks ---

class ResidualBlock(nn.Module):
    """ResNet block with optional channel projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        kernel_size: Tuple[int, int] = (3, 3),
    ) -> None:
        super().__init__()
        hidden_channels = hidden_channels if hidden_channels else out_channels
        self.residual_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=kernel_size, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=kernel_size, padding=1),
        )
        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), padding=0)

    def forward(self, x: Tensor) -> Tensor:
        return self.residual_block(x) + self.residual_layer(x)


class DownsampleBlock(nn.Module):
    """Downsample by factor 2 via MaxPool."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (3, 3)) -> None:
        super().__init__()
        self.downsample_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            ResidualBlock(out_channels, out_channels, out_channels // 2, kernel_size),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.downsample_block(x)


class UpsampleBlock(nn.Module):
    """Upsample by factor 2 via ConvTranspose."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (3, 3)) -> None:
        super().__init__()
        self.upsample_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(out_channels),
            ResidualBlock(out_channels, out_channels, out_channels // 2, kernel_size=kernel_size),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.upsample_block(x)


# --- World Models ---

class UNetWorldModel(nn.Module):
    """UNet world model with concatenation-based conditioning.

    From LAPO paper: https://arxiv.org/abs/2312.10812
    """

    def __init__(
        self,
        shape: Tuple[int, int, int],
        output_channels: int,
        condition_dim: int,
        base_channels: int = 24,
    ) -> None:
        super().__init__()
        assert len(shape) == 3
        input_channels, _, _ = shape

        self.encoder = nn.ModuleList([
            DownsampleBlock(input_channels + condition_dim, base_channels * 1),
            DownsampleBlock(base_channels * 1, base_channels * 2),
            DownsampleBlock(base_channels * 2, base_channels * 4),
            DownsampleBlock(base_channels * 4, base_channels * 8),
            DownsampleBlock(base_channels * 8, base_channels * 16),
            DownsampleBlock(base_channels * 16, base_channels * 32),
        ])

        self.decoder = nn.ModuleList([
            UpsampleBlock(base_channels * 32 + condition_dim, base_channels * 16),
            UpsampleBlock(base_channels * 32, base_channels * 8),
            UpsampleBlock(base_channels * 16, base_channels * 4),
            UpsampleBlock(base_channels * 8, base_channels * 2),
            UpsampleBlock(base_channels * 4, base_channels * 1),
            UpsampleBlock(base_channels * 2, base_channels * 1),
        ])

        self.final = nn.Sequential(
            nn.Conv2d(base_channels * 1 + input_channels, base_channels * 1, kernel_size=(3, 3), padding=1),
            ResidualBlock(base_channels * 1, base_channels * 1, base_channels * 1 // 2),
            nn.ReLU(),
            nn.Conv2d(base_channels * 1, output_channels, kernel_size=(1, 1)),
        )

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        condition = condition[:, :, None, None]

        # Inject action at input
        _, _, h, w = x.shape
        x_ = torch.cat((x, condition.expand(-1, -1, h, w)), dim=1)
        skip_connections = []
        for layer in self.encoder:
            x_ = layer(x_)
            skip_connections.append(x_)

        # Inject action at bottleneck
        _, _, h, w = skip_connections[-1].shape
        skip_connections[-1] = condition.expand(-1, -1, h, w)

        for layer in self.decoder:
            x_ = torch.cat((x_, skip_connections.pop()), dim=1)
            x_ = layer(x_)

        # Inject original input at end
        x_ = self.final(torch.cat((x_, x), dim=1))
        return F.tanh(x_) / 2


class ImpalaWorldModel(nn.Module):
    """Impala-based world model with concatenation-based conditioning.

    From LAOM paper: https://arxiv.org/abs/2502.00379
    """

    def __init__(
        self,
        shape: Tuple[int, int, int],
        output_channels: int,
        condition_dim: int = 128,
        channel_multiplier: int = 1,
        encoder_channels: Sequence[int] = (16, 32, 64, 128, 256),
        encoder_num_res_blocks: int = 1,
    ) -> None:
        super().__init__()
        assert len(shape) == 3

        # Encoder
        encoder_layers = []
        current_shape = shape
        for out_channels in encoder_channels:
            encoder_block = ImpalaEncoderBlock(current_shape[0], channel_multiplier * out_channels, encoder_num_res_blocks)
            current_shape = encoder_block.get_output_shape(*current_shape[1:])
            encoder_layers.append(encoder_block)

        self.encoder = nn.Sequential(*encoder_layers)
        self.final_encoder_shape = current_shape

        # Decoder
        decoder_layers = []
        dec_shape = (current_shape[0] * 2, *current_shape[1:])
        for out_channels in encoder_channels[::-1]:
            decoder_block = ImpalaDecoderBlock(dec_shape[0], channel_multiplier * out_channels, encoder_num_res_blocks)
            dec_shape = decoder_block.get_output_shape(*dec_shape[1:])
            decoder_layers.append(decoder_block)

        self.decoder = nn.Sequential(
            *decoder_layers,
            nn.GELU(),
            nn.Conv2d(encoder_channels[0] * channel_multiplier, output_channels, kernel_size=1),
            nn.Tanh(),
        )
        self.act_proj = nn.Linear(condition_dim, math.prod(self.final_encoder_shape))

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        x_ = self.encoder(x)
        condition_ = self.act_proj(condition).reshape(-1, *self.final_encoder_shape)
        x_ = torch.concat([x_, condition_], dim=1)
        x_ = self.decoder(x_)
        return x_
