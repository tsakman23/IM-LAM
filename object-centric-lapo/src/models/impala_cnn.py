from typing import Sequence, Tuple

import torch.nn as nn
from torch import Tensor


class ImpalaResidualBlock(nn.Module):
    """Residual block for IMPALA CNN."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.residual_block(x) + x


class ImpalaEncoderBlock(nn.Module):
    """Encoder block: Conv -> MaxPool(stride=2) -> ResBlocks."""

    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int = 2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
            *[ImpalaResidualBlock(out_channels) for _ in range(num_res_blocks)],
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

    def get_output_shape(self, h: int, w: int) -> Tuple[int, int, int]:
        h = (h + 2 * 1 - 3) // 2 + 1
        w = (w + 2 * 1 - 3) // 2 + 1
        return (self.out_channels, h, w)


class ImpalaDecoderBlock(nn.Module):
    """Decoder block: ConvTranspose(stride=2) -> ResBlocks."""

    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int = 2) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            *[ImpalaResidualBlock(out_channels) for _ in range(num_res_blocks)],
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)

    def get_output_shape(self, h: int, w: int) -> Tuple[int, int, int]:
        return (self.out_channels, h * 2, w * 2)


class ImpalaCNNBackbone(nn.Module):
    """IMPALA CNN backbone: encoder blocks + flatten + linear.

    Paper: https://arxiv.org/abs/1802.01561
    """

    def __init__(
        self,
        input_size: Sequence[int],
        channel_multiplier: int,
        output_dim: int = 256,
        num_res_blocks: int = 2,
        encoder_deep: bool = False,
    ) -> None:
        assert len(input_size) == 3
        c, h, w = input_size
        super().__init__()

        # Standard: 3 blocks (16x, 32x, 32x). Deep: adds 4th block (64x).
        blocks = [
            ImpalaEncoderBlock(c, 16 * channel_multiplier, num_res_blocks),
            ImpalaEncoderBlock(16 * channel_multiplier, 32 * channel_multiplier, num_res_blocks),
            ImpalaEncoderBlock(32 * channel_multiplier, 32 * channel_multiplier, num_res_blocks),
        ]
        self._num_blocks = 3
        if encoder_deep:
            blocks.append(ImpalaEncoderBlock(32 * channel_multiplier, 64 * channel_multiplier, num_res_blocks))
            self._num_blocks = 4

        self.backbone = nn.Sequential(*blocks, nn.ReLU(), nn.Flatten())
        self._channel_multiplier = channel_multiplier
        self._encoder_deep = encoder_deep
        self.fc_input_dim = self._calculate_input_dim(input_size)
        self.final_fc = nn.Linear(self.fc_input_dim, output_dim)
        self.out_features = output_dim

    def _calculate_input_dim(self, input_size: Sequence[int]) -> int:
        c, h, w = input_size
        cm = self._channel_multiplier

        def impala_transform(channels, height, width):
            height, width = (height + 1) // 2, (width + 1) // 2
            return channels, height, width

        c, h, w = impala_transform(16 * cm, h, w)
        c, h, w = impala_transform(32 * cm, h, w)
        c, h, w = impala_transform(32 * cm, h, w)
        if self._encoder_deep:
            c, h, w = impala_transform(64 * cm, h, w)
        return c * h * w

    def forward(self, x: Tensor) -> Tensor:
        return self.final_fc(self.backbone(x))
