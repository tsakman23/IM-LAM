from typing import Callable, List, Tuple, Type, Union

import numpy as np
import torch
from torch import Tensor, nn


def named_apply(
    fn: Callable,
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    """Apply a function to all modules in a module tree with their names.

    Recursively traverses a PyTorch module hierarchy and applies a function to each
    module along with its fully qualified name. The traversal order and whether to
    include the root module can be controlled.

    Args:
        fn (Callable): Function to apply to each module. Should accept keyword arguments
            'module' (nn.Module) and 'name' (str).
        module (nn.Module): The root module to traverse.
        name (str, optional): The name prefix for the root module. Defaults to "".
        depth_first (bool, optional): If True, applies function after visiting children
            (post-order). If False, applies before visiting children (pre-order).
            Defaults to True.
        include_root (bool, optional): Whether to apply the function to the root module.
            Defaults to False.

    Returns:
        nn.Module: The input module (unchanged, but function has been applied to its tree).
    """
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_name = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_name,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


def count_parameters(module: nn.Module) -> int:
    """Count the total number of parameters in a PyTorch module.

    Args:
        module (nn.Module): The PyTorch module to count parameters for.

    Returns:
        int: Total number of parameters in the module.
    """
    c = 0
    for m in module.parameters():
        c += m.nelement()
    return c


def cat_keep_shapes(x_list: List[Tensor]) -> Tuple[Tensor, List[Tuple[int, ...]], List[int]]:
    """Concatenate tensors while preserving information to restore original shapes.

    Flattens all tensors except the last dimension and concatenates them along dim 0.
    Returns the flattened concatenated tensor along with metadata to restore shapes.

    Args:
        x_list (List[Tensor]): List of tensors with shape [..., D] where ... can be any number of dimensions.

    Returns:
        Tuple[Tensor, List[Tuple[int, ...]], List[int]]: A tuple containing:
            - flattened (Tensor): Concatenated tensor with shape [total_tokens, D]
            - shapes (List[Tuple[int, ...]]): Original shapes of each tensor in x_list
            - num_tokens (List[int]): Number of tokens (flattened elements) for each tensor
    """
    shapes = [x.shape for x in x_list]
    num_tokens = [x.select(dim=-1, index=0).numel() for x in x_list]
    flattened = torch.cat([x.flatten(0, -2) for x in x_list])
    return flattened, shapes, num_tokens


def uncat_with_shapes(flattened: Tensor, shapes: List[Tuple[int, ...]], num_tokens: List[int]) -> List[Tensor]:
    """Restore original tensor shapes after concatenation.

    Reverse operation of cat_keep_shapes. Splits a flattened tensor and reshapes
    each piece according to the provided shape metadata.

    Args:
        flattened (Tensor): Concatenated tensor with shape [total_tokens, D]
        shapes (List[Tuple[int, ...]]): Original shapes of tensors before flattening
        num_tokens (List[int]): Number of tokens for each original tensor

    Returns:
        List[Tensor]: List of tensors restored to their original shapes (except last dimension
            which is determined by flattened.shape[-1])
    """
    outputs_splitted = torch.split_with_sizes(flattened, num_tokens, dim=0)
    shapes_adjusted = [shape[:-1] + torch.Size([flattened.shape[-1]]) for shape in shapes]
    outputs_reshaped = [o.reshape(shape) for o, shape in zip(outputs_splitted, shapes_adjusted)]
    return outputs_reshaped


def center(x: Tensor) -> Tensor:
    """Center the image.

    Args:
        x (Tensor): Image tensor.

    Returns:
        Tensor: Centered image tensor.
    """
    return x / 255 - 0.5


def uncenter(x: Tensor) -> Tensor:
    """Uncenter the image.

    Args:
        x (Tensor): Image tensor.

    Returns:
        Tensor: Uncentered image tensor.
    """
    return (x + 0.5) * 255


class Clip:
    """Clip a value to a specified range."""

    def __init__(self, min_val: float, max_val: float) -> None:
        """
        Initializes Clip.

        Args:
            min_val (float): Minimum value.
            max_val (float): Maximum value.
        """
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
        """
        Clips the value to the specified range.

        Args:
            x (Union[Tensor, np.ndarray]): Value to be clipped.

        Returns:
            Union[Tensor, np.ndarray]: Clipped value.
        """
        if isinstance(x, Tensor):
            return x.clamp(self.min_val, self.max_val)
        elif isinstance(x, np.ndarray):
            return np.clip(x, self.min_val, self.max_val)
        else:
            raise TypeError("Input must be a Tensor or ndarray")


def merge_tc(observation: Tensor) -> Tensor:
    """Merge time and channel dimensions
    Observation: (Batch, Time, Channel, Height, Width) -> (Batch, Time * Channel, Height, Width)

    Args:
        observation (Tensor): Observation.

    Returns:
        Tensor: Observation with merged time and channel dimension.
    """
    b, t, c, h, w = observation.shape
    return observation.reshape(b, t * c, h, w)


def orthogonal_init(
    module: nn.Module,
    modules_to_apply: List[Type[nn.Module]] = [nn.Linear, nn.Conv2d, nn.ConvTranspose2d],
    gain: float = 1.0
    ) -> None:
    """Orthogonal initialization of the weights.

    Args:
        module (nn.Module): The module to initialize.
        modules_to_apply (List[Type[nn.Module]], optional): The modules to apply the orthogonal initialization to.
        gain (float, optional): The gain for the orthogonal initialization.
    """
    if isinstance(module, tuple(modules_to_apply)):
        nn.init.orthogonal_(module.weight.data, gain)
        if hasattr(module.bias, "data"):
            module.bias.data.fill_(0.0)


def symlog(x: Tensor) -> Tensor:
    """
    Symmetric logarithm function that handles both positive and negative values.

    This function applies a logarithmic transformation that preserves the sign
    of the input while compressing large values. It's defined as:
    symlog(x) = sign(x) * log(1 + |x|)

    Args:
        x: Input tensor of any shape

    Returns:
        Tensor with the same shape as input, with symmetric log transformation applied
    """
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: Tensor) -> Tensor:
    """
    Symmetric exponential function, inverse of symlog.

    This function is the inverse of symlog, expanding compressed values back
    to their original scale while preserving sign. It's defined as:
    symexp(x) = sign(x) * (exp(|x|) - 1)

    Args:
        x: Input tensor of any shape, typically output from symlog

    Returns:
        Tensor with the same shape as input, with symmetric exp transformation applied
    """
    return torch.sign(x) * torch.expm1(torch.abs(x))
