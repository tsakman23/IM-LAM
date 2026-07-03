import warnings
from math import prod
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, no_type_check

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ifo.common.utils.model import cnn_forward, create_layers, miniblock


class MLP(nn.Module):
    """Simple MLP backbone."""

    def __init__(
        self,
        input_dims: Union[int, Sequence[int]],
        output_dim: Optional[int] = None,
        hidden_sizes: Sequence[int] = (),
        layer_args: Optional[Any] = None,
        dropout_layer: Optional[Union[nn.Module, Sequence[nn.Module]]] = None,
        dropout_args: Optional[Any] = None,
        norm_layer: Optional[Union[nn.Module, Sequence[nn.Module]]] = None,
        norm_args: Optional[Any] = None,
        activation: Optional[Union[nn.Module, Sequence[nn.Module]]] = nn.ReLU,
        act_args: Optional[Any] = None,
        flatten_dim: Optional[int] = None,
    ) -> None:
        """Initialize the MLP.

        Args:
            input_dims (Union[int, Sequence[int]]): dimensions of the input vector.
            output_dim (int, optional): dimension of the output vector. If set to None, there
                is no final linear layer. Else, a final linear layer is added.
                Defaults to None.
            hidden_sizes (Sequence[int], optional): shape of MLP passed in as a list, not including
                input_dims and output_dim.
            layer_args (Optional[Any]): arguments for the linear layers.
            dropout_layer (Union[nn.Module, Sequence[nn.Module]], optional): which dropout layer to be used
                before activation (possibly before the normalization layer), e.g., ``nn.Dropout``.
                You can also pass a list of dropout modules with the same length
                of hidden_sizes to use different dropout modules in different layers.
                If None, then no dropout layer is used.
                Defaults to None.
            dropout_args (Optional[Any]): arguments for the dropout layers.
            norm_layer (Union[nn.Module, Sequence[nn.Module]], optional): which normalization layer to be used
                before activation, e.g., ``nn.LayerNorm`` and ``nn.BatchNorm1d``.
                You can also pass a list of normalization modules with the same length
                of hidden_sizes to use different normalization modules in different layers.
                If None, then no normalization layer is used.
                Defaults to None.
            norm_args (Optional[Any]): arguments for the normalization layers.
            activation (Union[nn.Module, Sequence[nn.Module]], optional): which activation to use after each layer,
                can be both the same activation for all layers if a single ``nn.Module`` is passed, or different
                activations for different layers if a list is passed.
                Defaults to ``nn.ReLU``.
            act_args (Optional[Any]): arguments for the activation layers.
            flatten_dim (int, optional): whether to flatten input data. The flatten dimension starts from 1.
                Defaults to None.
        """
        super().__init__()
        num_layers = len(hidden_sizes)
        if num_layers < 1 and output_dim is None:
            raise ValueError("The number of layers should be at least 1.")

        if isinstance(input_dims, Sequence) and flatten_dim is None:
            warnings.warn(
                "input_dims is a sequence, but flatten_dim is not specified. "
                "Be careful to flatten the input data correctly before the forward."
            )

        dropout_layer_list, dropout_args_list = create_layers(dropout_layer, dropout_args, num_layers)
        norm_layer_list, norm_args_list = create_layers(norm_layer, norm_args, num_layers)
        activation_list, act_args_list = create_layers(activation, act_args, num_layers)

        if isinstance(layer_args, list):
            layer_args_list = layer_args
        else:
            layer_args_list = [layer_args] * num_layers

        if isinstance(input_dims, int):
            input_dims = [input_dims]
        hidden_sizes = [prod(input_dims)] + list(hidden_sizes)
        model = []
        for in_dim, out_dim, l_args, drop, drop_args, norm, norm_args, activ, act_args in zip(
            hidden_sizes[:-1],
            hidden_sizes[1:],
            layer_args_list,
            dropout_layer_list,
            dropout_args_list,
            norm_layer_list,
            norm_args_list,
            activation_list,
            act_args_list,
        ):
            model += miniblock(in_dim, out_dim, nn.Linear, l_args, drop, drop_args, norm, norm_args, activ, act_args)
        if output_dim is not None:
            model += [nn.Linear(hidden_sizes[-1], output_dim)]

        self._output_dim = output_dim or hidden_sizes[-1]
        self._model = nn.Sequential(*model)
        self._flatten_dim = flatten_dim

    @property
    def model(self) -> nn.Module:
        """Get the underlying sequential model.

        Returns:
            nn.Module: The sequential model containing all MLP layers.
        """
        return self._model

    @property
    def output_dim(self) -> int:
        """Get the output dimension of the MLP.

        Returns:
            int: The dimension of the output layer.
        """
        return self._output_dim

    @property
    def flatten_dim(self) -> Optional[int]:
        """Get the flattening dimension.

        Returns:
            Optional[int]: The dimension from which to flatten input, None if no flattening.
        """
        return self._flatten_dim

    def forward(self, obs: Tensor) -> Tensor:
        """Forward pass through the MLP.

        Args:
            obs (Tensor): Input tensor to be processed by the MLP.

        Returns:
            Tensor: Output tensor after passing through all MLP layers.
        """
        if self.flatten_dim is not None:
            obs = obs.flatten(self.flatten_dim)
        return self.model(obs)


class CNN(nn.Module):
    """Simple CNN backbone."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: Sequence[int],
        cnn_layer: nn.Module = nn.Conv2d,
        layer_args: Any = None,
        dropout_layer: Optional[Union[nn.Module, Sequence[nn.Module]]] = None,
        dropout_args: Optional[Any] = None,
        norm_layer: Optional[Union[nn.Module, Sequence[nn.Module]]] = None,
        norm_args: Optional[Any] = None,
        activation: Optional[Union[nn.Module, Sequence[nn.Module]]] = nn.ReLU,
        act_args: Optional[Any] = None,
    ) -> None:
        """Initialize the CNN.

        Args:
            input_channels (int): dimensions of the input channels.
            hidden_channels (Sequence[int]): intermediate number of channels for the CNN,
                including the output channels.
            cnn_layer (nn.Module): the convolutional layer type to use. Defaults to nn.Conv2d.
            layer_args (Any): arguments for the convolutional layers.
            dropout_layer (Union[nn.Module, Sequence[nn.Module]], optional): which dropout layer to be used
                before activation (possibly before the normalization layer), e.g., ``nn.Dropout``.
                You can also pass a list of dropout modules with the same length
                of hidden_channels to use different dropout modules in different layers.
                If None, then no dropout layer is used.
                Defaults to None.
            dropout_args (Optional[Any]): arguments for the dropout layers.
            norm_layer (Union[nn.Module, Sequence[nn.Module]], optional): which normalization layer to be used
                before activation, e.g., ``nn.LayerNorm`` and ``nn.BatchNorm1d``.
                You can also pass a list of normalization modules with the same length
                of hidden_channels to use different normalization modules in different layers.
                If None, then no normalization layer is used.
                Defaults to None.
            norm_args (Optional[Any]): arguments for the normalization layers.
            activation (Union[nn.Module, Sequence[nn.Module]], optional): which activation to use after each layer,
                can be both the same activation for all layers if a single ``nn.Module`` is passed, or different
                activations for different layers if a list is passed.
                Defaults to ``nn.ReLU``.
            act_args (Optional[Any]): arguments for the activation layers.
        """
        super().__init__()
        num_layers = len(hidden_channels)
        if num_layers < 1:
            raise ValueError("The number of layers should be at least 1.")

        dropout_layer_list, dropout_args_list = create_layers(dropout_layer, dropout_args, num_layers)
        norm_layer_list, norm_args_list = create_layers(norm_layer, norm_args, num_layers)
        activation_list, act_args_list = create_layers(activation, act_args, num_layers)

        if isinstance(layer_args, list):
            layer_args_list = layer_args
        else:
            layer_args_list = [layer_args] * num_layers

        hidden_sizes = [input_channels] + list(hidden_channels)
        model = []
        for in_dim, out_dim, l_args, drop, drop_args, norm, norm_args, activ, act_args in zip(
            hidden_sizes[:-1],
            hidden_sizes[1:],
            layer_args_list,
            dropout_layer_list,
            dropout_args_list,
            norm_layer_list,
            norm_args_list,
            activation_list,
            act_args_list,
        ):
            model += miniblock(in_dim, out_dim, cnn_layer, l_args, drop, drop_args, norm, norm_args, activ, act_args)

        self._output_dim = hidden_sizes[-1]
        self._model = nn.Sequential(*model)

    @property
    def model(self) -> nn.Module:
        """Get the underlying sequential CNN model.

        Returns:
            nn.Module: The sequential model containing all CNN layers.
        """
        return self._model

    @property
    def output_dim(self) -> int:
        """Get the output dimension of the CNN.

        Returns:
            int: The number of output channels from the final CNN layer.
        """
        return self._output_dim

    @no_type_check
    def forward(self, obs: Tensor) -> Tensor:
        """Forward pass through the CNN.

        Args:
            obs (Tensor): Input tensor to be processed by the CNN layers.

        Returns:
            Tensor: Output tensor after passing through all CNN layers.
        """
        return self.model(obs)


class DeCNN(nn.Module):
    """Simple deconvolutional CNN backbone."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: Sequence[int] = (),
        cnn_layer: nn.Module = nn.ConvTranspose2d,
        layer_args: Any = None,
        dropout_layer: Optional[Union[nn.Module, Sequence[nn.Module]]] = None,
        dropout_args: Optional[Any] = None,
        norm_layer: Optional[Union[nn.Module, Sequence[nn.Module]]] = None,
        norm_args: Optional[Any] = None,
        activation: Optional[Union[nn.Module, Sequence[nn.Module]]] = nn.ReLU,
        act_args: Optional[Any] = None,
    ) -> None:
        """Initialize the deconvolutional CNN.

        Args:
            input_channels (int): dimensions of the input channels.
            hidden_channels (Sequence[int], optional): intermediate number of channels for the DeCNN,
                including the output channels. Defaults to ().
            cnn_layer (nn.Module): the deconvolutional layer type to use. Defaults to nn.ConvTranspose2d.
            layer_args (Any): arguments for the deconvolutional layers.
            dropout_layer (Union[nn.Module, Sequence[nn.Module]], optional): which dropout layer to be used
                before activation (possibly before the normalization layer), e.g., ``nn.Dropout``.
                You can also pass a list of dropout modules with the same length
                of hidden_channels to use different dropout modules in different layers.
                If None, then no dropout layer is used.
                Defaults to None.
            dropout_args (Optional[Any]): arguments for the dropout layers.
            norm_layer (Union[nn.Module, Sequence[nn.Module]], optional): which normalization layer to be used
                before activation, e.g., ``nn.LayerNorm`` and ``nn.BatchNorm1d``.
                You can also pass a list of normalization modules with the same length
                of hidden_channels to use different normalization modules in different layers.
                If None, then no normalization layer is used.
                Defaults to None.
            norm_args (Optional[Any]): arguments for the normalization layers.
            activation (Union[nn.Module, Sequence[nn.Module]], optional): which activation to use after each layer,
                can be both the same activation for all layers if a single ``nn.Module`` is passed, or different
                activations for different layers if a list is passed.
                Defaults to ``nn.ReLU``.
            act_args (Optional[Any]): arguments for the activation layers.
        """
        super().__init__()
        num_layers = len(hidden_channels)
        if num_layers < 1:
            raise ValueError("The number of layers should be at least 1.")

        dropout_layer_list, dropout_args_list = create_layers(dropout_layer, dropout_args, num_layers)
        norm_layer_list, norm_args_list = create_layers(norm_layer, norm_args, num_layers)
        activation_list, act_args_list = create_layers(activation, act_args, num_layers)

        if isinstance(layer_args, list):
            layer_args_list = layer_args
        else:
            layer_args_list = [layer_args] * num_layers

        hidden_sizes = [input_channels] + list(hidden_channels)
        model = []
        for in_dim, out_dim, l_args, drop, drop_args, norm, norm_args, activ, act_args in zip(
            hidden_sizes[:-1],
            hidden_sizes[1:],
            layer_args_list,
            dropout_layer_list,
            dropout_args_list,
            norm_layer_list,
            norm_args_list,
            activation_list,
            act_args_list,
        ):
            model += miniblock(in_dim, out_dim, cnn_layer, l_args, drop, drop_args, norm, norm_args, activ, act_args)

        self._output_dim = hidden_sizes[-1]
        self._model = nn.Sequential(*model)

    @property
    def model(self) -> nn.Module:
        """Get the underlying sequential deconvolutional model.

        Returns:
            nn.Module: The sequential model containing all deconvolutional layers.
        """
        return self._model

    @property
    def output_dim(self) -> int:
        """Get the output dimension of the DeCNN.

        Returns:
            int: The number of output channels from the final deconvolutional layer.
        """
        return self._output_dim

    @no_type_check
    def forward(self, obs: Tensor) -> Tensor:
        """Forward pass through the DeCNN.

        Args:
            obs (Tensor): Input tensor to be processed by the deconvolutional layers.

        Returns:
            Tensor: Output tensor after passing through all deconvolutional layers.
        """
        return self.model(obs)


class NatureCNN(CNN):
    """CNN from DQN Nature paper: Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning."
    Nature 518.7540 (2015): 529-533.
    """

    def __init__(self, in_channels: int, features_dim: int, screen_size: int = 64) -> None:
        """Initialize the NatureCNN.

        Args:
            in_channels (int): the input channels to the first convolutional layer
            features_dim (int): the features dimension in output from the last convolutional layer
            screen_size (int, optional): the dimension of the input image as a single integer.
                Needed to extract the features and compute the output dimension after all the
                convolutional layers.
                Defaults to 64.
        """
        super().__init__(
            in_channels,
            [32, 64, 64],
            layer_args=[
                {"kernel_size": 8, "stride": 4},
                {"kernel_size": 4, "stride": 2},
                {"kernel_size": 3, "stride": 1},
            ],
        )

        with torch.no_grad():
            x = self.model(torch.rand(1, in_channels, screen_size, screen_size, device=self.model[0].weight.device))
            out_dim = x.flatten(1).shape[1]
        self._output_dim = out_dim
        self.fc = None
        if features_dim is not None:
            self._output_dim = features_dim
            self.fc = nn.Linear(out_dim, features_dim)

    @property
    def output_dim(self) -> int:
        """Get the output dimension of the NatureCNN.

        Returns:
            int: The dimension of the final feature layer (after optional fully connected layer).
        """
        return self._output_dim

    def forward(self, obs: Tensor) -> Tensor:
        """Forward pass through the NatureCNN.

        Args:
            obs (Tensor): Input tensor (typically images) to be processed.

        Returns:
            Tensor: Feature representation after CNN processing and optional fully connected layer.
        """
        obs = cnn_forward(self.model, obs, input_dim=obs.shape[-3:], output_dim=(-1,))
        obs = F.relu(self.fc(obs))
        return obs


class LayerNormGRUCell(nn.Module):
    """A GRU cell with a LayerNorm, taken
    from https://github.com/danijar/dreamerv2/blob/main/dreamerv2/common/nets.py#L317.

    This particular GRU cell accepts 3-D inputs, with a sequence of length 1, and applies
    a LayerNorm after the projection of the inputs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        batch_first: bool = False,
        layer_norm_cls: Callable[..., nn.Module] = nn.Identity,
        layer_norm_kw: Dict[str, Any] = {},
    ) -> None:
        """Initialize the LayerNormGRUCell.

        Args:
            input_size (int): the input size.
            hidden_size (int): the hidden state size
            bias (bool, optional): whether to apply a bias to the input projection.
                Defaults to True.
            batch_first (bool, optional): whether the first dimension represent the batch dimension or not.
                Defaults to False.
            layer_norm_cls (Callable[..., nn.Module]): the layer norm to apply after the input projection.
                Defaults to nn.Identiy.
            layer_norm_kw (Dict[str, Any]): the kwargs of the layer norm.
                Default to {}.
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.batch_first = batch_first
        self.linear = nn.Linear(input_size + hidden_size, 3 * hidden_size, bias=self.bias)
        # Avoid multiple values for the `normalized_shape` argument
        layer_norm_kw.pop("normalized_shape", None)
        self.layer_norm = layer_norm_cls(3 * hidden_size, **layer_norm_kw)

    def forward(self, input: Tensor, hx: Optional[Tensor] = None) -> Tensor:
        """Forward pass through the LayerNormGRUCell.

        Args:
            input (Tensor): Input tensor (1D, 2D, or 3D with sequence length 1).
            hx (Optional[Tensor]): Hidden state tensor. If None, initialized with zeros.

        Returns:
            Tensor: Updated hidden state after processing the input.

        Raises:
            AssertionError: If 3D input doesn't have sequence length 1, or if input is not 1D/2D.
        """
        is_3d = input.dim() == 3
        if is_3d:
            if input.shape[int(self.batch_first)] == 1:
                input = input.squeeze(int(self.batch_first))
            else:
                raise AssertionError(
                    "LayerNormGRUCell: Expected input to be 3-D with sequence length equal to 1 but received "
                    f"a sequence of length {input.shape[int(self.batch_first)]}"
                )
        if hx.dim() == 3:
            hx = hx.squeeze(0)
        assert input.dim() in (
            1,
            2,
        ), f"LayerNormGRUCell: Expected input to be 1-D or 2-D but received {input.dim()}-D tensor"

        is_batched = input.dim() == 2
        if not is_batched:
            input = input.unsqueeze(0)

        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, dtype=input.dtype, device=input.device)
        else:
            hx = hx.unsqueeze(0) if not is_batched else hx

        input = torch.cat((hx, input), -1)
        x = self.linear(input)
        x = self.layer_norm(x)
        reset, cand, update = torch.chunk(x, 3, -1)
        reset = torch.sigmoid(reset)
        cand = torch.tanh(reset * cand)
        update = torch.sigmoid(update - 1)
        hx = update * cand + (1 - update) * hx

        if not is_batched:
            hx = hx.squeeze(0)
        elif is_3d:
            hx = hx.unsqueeze(0)

        return hx


class LayerNormChannelLast(nn.LayerNorm):
    """Layer normalization for 4D tensors in NCHW format.

    This class extends PyTorch's LayerNorm to work with 4D tensors by permuting dimensions
    to apply normalization in channel-last format (NHWC) and then permuting back to NCHW.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialize LayerNormChannelLast.

        Args:
            *args: Arguments passed to nn.LayerNorm.
            **kwargs: Keyword arguments passed to nn.LayerNorm.
        """
        super().__init__(*args, **kwargs)

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass applying layer normalization to 4D tensor.

        Args:
            input (Tensor): Input tensor in NCHW format (4D).

        Returns:
            Tensor: Layer-normalized tensor in NCHW format with original dtype preserved.

        Raises:
            ValueError: If input tensor is not 4D.
        """
        if input.dim() != 4:
            raise ValueError(f"Input tensor must be 4D (NCHW), received {len(input.shape)}D instead: {input.shape}")
        input = input.permute(0, 2, 3, 1).contiguous()
        input = super().forward(input)
        input = input.permute(0, 3, 1, 2).contiguous()
        return input


class FullyConnectedNeuralNetwork(nn.Module):
    """Fully connected neural network with ReLU activation function."""

    def __init__(self, in_features: int, out_features: int, hidden_sizes: List[int] = []) -> None:
        """Instantiate FullyConnectedNeuralNetwork.

        If hidden_sizes is empty, the network will be a simple linear layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features
            hidden_sizes (List[int]): List of hidden layer sizes.
        """
        super().__init__()

        # Define NN layers.
        layer_sizes = [in_features, *hidden_sizes, out_features]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i + 1]), nn.ReLU()])

        self.mlp = nn.Sequential(*layers[:-1])  # Remove last ReLU.

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.mlp(x)
