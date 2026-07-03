from typing import Optional, Sequence, Tuple

import torch.nn as nn
from torch import Tensor


class ImpalaResidualBlock(nn.Module):
    """Residual Block for the IMPALA CNN."""

    def __init__(self, channels: int) -> None:
        """Instantiate module.

        Args:
            channels (int): Number of convolutional channels.
        """
        super().__init__()
        self.residual_block = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input

        Returns:
            Tensor: Output
        """
        return self.residual_block(x) + x


class ImpalaEncoderBlock(nn.Module):
    """Encoder block based on IMPALA CNN."""

    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int = 2) -> None:
        """Instantiate module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_res_blocks (int): Number of residual blocks.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=1),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),  # Remove?
            *[ImpalaResidualBlock(out_channels) for _ in range(num_res_blocks)],
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input

        Returns:
            Tensor: Output
        """
        return self.block(x)

    def get_output_shape(self, h: int, w: int) -> Tuple[int, int, int]:
        """Get output shape.

        Args:
            h (int): Height.
            w (int): Width.

        Returns:
            Tuple[int, int]: Output shape [C, H, W].
        """
        # MaxPool2d with kernel_size=3, stride=2, padding=1
        h = (h + 2 * 1 - 3) // 2 + 1
        w = (w + 2 * 1 - 3) // 2 + 1
        return (self.out_channels, h, w)


class ImpalaDecoderBlock(nn.Module):
    """Decoder block based on IMPALA CNN that upsamples by factor 2."""

    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int = 2) -> None:
        """Instantiate module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_res_blocks (int): Number of residual blocks.
        """
        super().__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            *[ImpalaResidualBlock(out_channels) for _ in range(num_res_blocks)],
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input

        Returns:
            Tensor: Output
        """
        return self.block(x)

    def get_output_shape(self, h: int, w: int) -> Tuple[int, int, int]:
        """Get output shape.

        Args:
            h (int): Height.
            w (int): Width.

        Returns:
            Tuple[int, int]: Output shape [C, H, W].
        """
        return (self.out_channels, h * 2, w * 2)

class ImpalaCNNBackbone(nn.Module):
    """Backbone of IMPALA CNN Model.

    Paper: https://arxiv.org/abs/1802.01561
    """

    def __init__(
        self,
        input_size: Sequence[int],
        channel_multiplier: int,
        output_dim: int = 256,
        num_res_blocks: int = 2,
    ) -> None:
        """Instantiate module.

        Args:
            input_size (Sequence[int]): Sequence of input sizes in format [C, H, W]
            channel_multiplier (int): Multiplier for number of covolutional channels.
            output_dim (int): Dimension of last linear layer.
            num_res_blocks (int): Number of residual blocks.
        """
        assert len(input_size) == 3
        c, h, w = input_size
        super().__init__()
        self.backbone = nn.Sequential(
            ImpalaEncoderBlock(c, 16 * channel_multiplier, num_res_blocks),
            ImpalaEncoderBlock(16 * channel_multiplier, 32 * channel_multiplier, num_res_blocks),
            ImpalaEncoderBlock(32 * channel_multiplier, 32 * channel_multiplier, num_res_blocks),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Calculate the input_dim for the final fully connected layer
        self.fc_input_dim = self._calculate_input_dim(input_size, channel_multiplier)
        self.final_fc = nn.Linear(self.fc_input_dim, output_dim)
        self.out_features = output_dim

    def _calculate_input_dim(self, input_size: Sequence[int], channel_multiplier: int) -> int:
        """Calculate the number of features after the backbone.

        Args:
            input_size (Sequence[int]): Input size [C, H, W].
            channel_multiplier (int): Multiplier for convolutional channels.

        Returns:
            int: Flattened size after the backbone.
        """
        c, h, w = input_size

        # Define transformations per ImpalaBlock
        def impala_transform(channels, height, width):
            # Conv2d keeps spatial dimensions with padding=1
            # MaxPool2d reduces spatial dimensions by half (stride=2)
            height, width = (height + 1) // 2, (width + 1) // 2
            return channels, height, width

        # Apply transformations through the backbone
        c, h, w = impala_transform(16 * channel_multiplier, h, w)
        c, h, w = impala_transform(32 * channel_multiplier, h, w)
        c, h, w = impala_transform(32 * channel_multiplier, h, w)

        # Calculate flattened size
        return c * h * w

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input

        Returns:
            Tensor: Output
        """
        return self.final_fc(self.backbone(x))


class ImpalaCNN(nn.Module):
    """Impala CNN Model.

    Paper: https://arxiv.org/abs/1802.01561

    Note: This implementation is without the text encoder (Embedding 20 -> LSTM 64) in Figure 3 of the paper.
    """

    def __init__(self, input_size: Sequence[int], channel_multiplier: int) -> None:
        """Instantiate module.

        Args:
            input_size (Sequence[int]): List of input sizes in format [C, H, W]
            channel_multiplier (int): Multiplier for number of covolutional channels.
        """
        super().__init__()
        assert len(input_size) == 3
        self.backbone = nn.Sequential(ImpalaCNNBackbone(input_size, channel_multiplier), nn.ReLU())
        self.lstm = nn.LSTMCell(256, 256)

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            input (Tensor): Input tensor.
            hx (Optional[Tuple[Tensor, Tensor]], optional): Tuple containing the hidden state and cell state tensor.

        Returns:
            Tuple[Tensor, Tensor]: Tuple containing the next hidden state and cell state tensor.
        """
        input = self.backbone(input)
        return self.lstm(input, hx)
