from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class ResidualBlock(nn.Module):
    """ResNet block with batch norm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        kernel_size: Tuple[int, int] = (3, 3),
    ) -> None:
        """ResNet block with group norm.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            hidden_channels (int, optional): Hidden channels. If not specified, defaults to `out_channels`.
            kernel_size (int, optional): Kernel size. Defaults to 3.
        """
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
        """Forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.residual_block(x) + self.residual_layer(x)


class DownsampleBlock(nn.Module):
    """Downsample block that downsamples the spatial size by factor 2."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (3, 3)) -> None:
        """Downsample block that downsamples the spatial size by factor 2.

        Downsampling is done via max pooling.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            kernel_size (int, optional): Kernel size. Defaults to 3.
        """
        super().__init__()
        self.downsample_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            ResidualBlock(out_channels, out_channels, out_channels // 2, kernel_size),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.downsample_block(x)


class UpsampleBlock(nn.Module):
    """Upsample by factor 2."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Tuple[int, int] = (3, 3)) -> None:
        """Upsample by factor 2.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of convolutional channels.
            kernel_size (Tuple[int, int], optional): Kernel size. Defaults to (3, 3).
        """
        super().__init__()
        self.upsample_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(2, 2), stride=2),
            nn.BatchNorm2d(out_channels),
            ResidualBlock(out_channels, out_channels, out_channels // 2, kernel_size=kernel_size),
            nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Upsampled tensor
        """
        return self.upsample_block(x)


class UNet(nn.Module):
    """UNet model."""

    def __init__(
        self,
        shape: Tuple[int, int, int],
        output_channels: int,
        base_channels: int = 24,
    ) -> None:
        """UNet model from LAPO paper.

        Args:
            shape (Tuple[int, int, int]): Shape of the input tensor (C, H, W).
            output_channels (int): Number of output channels.
            base_channels (int, optional): Number of base channels to use for processing.
        """
        super().__init__()
        assert len(shape) == 3, "Shape must be (C, H, W)."
        input_channels, _, _ = shape
        self.encoder = nn.ModuleList(
            [
                DownsampleBlock(input_channels, base_channels * 1),
                DownsampleBlock(base_channels * 1, base_channels * 2),
                DownsampleBlock(base_channels * 2, base_channels * 4),
                DownsampleBlock(base_channels * 4, base_channels * 8),
                DownsampleBlock(base_channels * 8, base_channels * 16),
                DownsampleBlock(base_channels * 16, base_channels * 32),
            ]
        )

        self.decoder = nn.ModuleList(
            [
                UpsampleBlock(base_channels * 32, base_channels * 16),
                UpsampleBlock(base_channels * 32, base_channels * 8),
                UpsampleBlock(base_channels * 16, base_channels * 4),
                UpsampleBlock(base_channels * 8, base_channels * 2),
                UpsampleBlock(base_channels * 4, base_channels * 1),
                UpsampleBlock(base_channels * 2, base_channels * 1),
                nn.Sequential(
                    nn.Conv2d(base_channels * 1 + input_channels, base_channels * 1, kernel_size=(3, 3), padding=1),
                    ResidualBlock(base_channels * 1, base_channels * 1, base_channels * 1 // 2),
                    nn.ReLU(),
                    nn.Conv2d(base_channels * 1, output_channels, kernel_size=(1, 1)),
                    )
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        skip_connections = [x.clone()]
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)

        x = self.decoder[0](skip_connections.pop())

        for layer in self.decoder[1:]:
            x = torch.cat((x, skip_connections.pop()), dim=1)
            x = layer(x)

        return x
