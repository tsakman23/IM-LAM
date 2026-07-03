from typing import Union

import torch
from torch import Tensor, nn


class LayerScale(nn.Module):
    """Layer-wise scaling module for Transformer models.

    Applies learnable per-dimension scaling to the input tensor. This is commonly
    used in vision Transformers to stabilize training by scaling residual branches.
    """

    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        """Initialize the LayerScale module.

        Args:
            dim (int): Dimension of the input features.
            init_values (Union[float, Tensor]): Initial value for the scaling parameter.
                Defaults to 1e-5.
            inplace (bool): Whether to perform in-place multiplication. Defaults to False.
        """
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.empty(dim))
        self.init_values = init_values

    def reset_parameters(self) -> None:
        """Reset the scaling parameter to its initial value.

        Initializes gamma with the constant value specified by init_values.
        """
        nn.init.constant_(self.gamma, self.init_values)  # type: ignore[arg-type]

    def forward(self, x: Tensor) -> Tensor:
        """Apply layer-wise scaling to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (..., dim).

        Returns:
            Tensor: Scaled tensor of the same shape as input.
        """
        return x.mul_(self.gamma) if self.inplace else x * self.gamma
