from typing import Callable, List, Optional

import torch.nn.functional as F
from torch import Tensor, nn

from ifo.common.utils.functions import cat_keep_shapes, uncat_with_shapes


class ListForwardMixin(object):
    """Mixin class that provides list-based forward pass functionality.

    This mixin adds the ability to process a list of tensors with different shapes
    by concatenating them, applying the forward pass, and then splitting them back
    to their original shapes.
    """

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for a single tensor.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Processed tensor.

        Raises:
            NotImplementedError: Must be implemented by subclass.
        """
        raise NotImplementedError

    def forward_list(self, x_list: List[Tensor]) -> List[Tensor]:
        """Forward pass for a list of tensors with potentially different shapes.

        This method concatenates the input tensors, applies the forward pass,
        and then splits the result back to match the original shapes.

        Args:
            x_list (List[Tensor]): List of input tensors with potentially different shapes.

        Returns:
            List[Tensor]: List of processed tensors with the same shapes as the input.
        """
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        x_flat = self.forward(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)


class MLP(nn.Module, ListForwardMixin):
    """Multi-Layer Perceptron (MLP) with two linear layers and dropout.

    A standard feedforward network consisting of two linear transformations
    with an activation function and dropout in between.
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        """Initialize the MLP.

        Args:
            in_features (int): Number of input features.
            hidden_features (Optional[int]): Number of hidden features. Defaults to in_features if None.
            out_features (Optional[int]): Number of output features. Defaults to in_features if None.
            act_layer (Callable[..., nn.Module]): Activation function class. Defaults to nn.GELU.
            drop (float): Dropout probability. Defaults to 0.0.
            bias (bool): Whether to use bias in linear layers. Defaults to True.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the MLP.

        Args:
            x (Tensor): Input tensor of shape (..., in_features).

        Returns:
            Tensor: Output tensor of shape (..., out_features).
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module, ListForwardMixin):
    """SwiGLU Feed-Forward Network.

    Implements the SwiGLU (Swish-Gated Linear Unit) activation function,
    which uses element-wise multiplication of SiLU-activated and linear
    projections. This architecture has been shown to improve performance
    in Transformer models.

    The hidden dimension is computed as (2/3 * hidden_features) and then
    aligned to a multiple of align_to for computational efficiency.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[Callable[..., nn.Module]] = None,
        drop: float = 0.0,
        bias: bool = True,
        align_to: int = 8,
    ) -> None:
        """Initialize the SwiGLU FFN.

        Args:
            in_features (int): Number of input features.
            hidden_features (Optional[int]): Number of hidden features before alignment.
                Defaults to in_features if None.
            out_features (Optional[int]): Number of output features. Defaults to in_features if None.
            act_layer (Optional[Callable[..., nn.Module]]): Unused parameter, kept for API compatibility.
            drop (float): Unused parameter, kept for API compatibility.
            bias (bool): Whether to use bias in linear layers. Defaults to True.
            align_to (int): Align hidden dimension to this multiple for efficiency.
                Defaults to 8.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % align_to)
        self.w1 = nn.Linear(in_features, swiglu_hidden_features, bias=bias)
        self.w2 = nn.Linear(in_features, swiglu_hidden_features, bias=bias)
        self.w3 = nn.Linear(swiglu_hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the SwiGLU FFN.

        Applies the SwiGLU activation: SiLU(w1(x)) * w2(x), followed by
        the output projection w3.

        Args:
            x (Tensor): Input tensor of shape (..., in_features).

        Returns:
            Tensor: Output tensor of shape (..., out_features).
        """
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)
