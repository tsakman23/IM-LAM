from typing import Dict, Union

import torch
from tensordict import TensorDict
from torch import Tensor
from torchvision.transforms.v2 import Compose, Lambda, ToDtype, Transform


def center(x: Tensor) -> Tensor:
    """Center pixel values: uint8 [0, 255] -> float32 [-0.5, 0.5]."""
    return x / 255.0 - 0.5


def uncenter(x: Tensor) -> Tensor:
    """Uncenter pixel values: [-0.5, 0.5] -> [0, 1]."""
    return x + 0.5


class DictTransform:
    """Applies a set of transforms to a Dict of Tensors or a TensorDict."""

    def __init__(self, transform: Dict[str, Transform]) -> None:
        self.transform = transform

    def __call__(self, x: Union[TensorDict, Dict[str, Tensor]]) -> Union[TensorDict, Dict[str, Tensor]]:
        assert isinstance(x, (TensorDict, dict))
        for key, transform in self.transform.items():
            if key in x:
                x[key] = transform(x[key])
        return x


class Permute(Transform):
    """Permute tensor dimensions from HWC to CHW format.

    Handles arbitrary leading batch/time dimensions.
    """

    def forward(self, inpt: Tensor) -> Tensor:
        ndim = inpt.dim()
        assert ndim >= 3, "Input tensor must have at least 3 dimensions."
        perm = tuple(range(ndim - 3)) + (ndim - 1, ndim - 3, ndim - 2)
        return inpt.permute(perm)


class Clip(Transform):
    """Clip tensor values to a specified range."""

    def __init__(self, min_val: float, max_val: float) -> None:
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, inpt: Tensor) -> Tensor:
        return torch.clamp(inpt, min=self.min_val, max=self.max_val)


class ProcessMask(Transform):
    """Convert uint8 mask {0, 255} to float32 {0, 1}."""

    def forward(self, inpt: Tensor) -> Tensor:
        return (inpt > 0).to(torch.float32)


def dcs_masked_transform() -> DictTransform:
    """Transform for Masked Distracting Control Suite."""
    return DictTransform({
        "observation": Compose([ToDtype(torch.float32), Lambda(center)]),
        "mask": ProcessMask(),
        "action": Clip(min_val=-1, max_val=1),
    })


def metaworld_transform() -> DictTransform:
    """Transform for Meta-World."""
    return DictTransform({
        "observation": Compose([ToDtype(torch.float32), Lambda(center)]),
        "mask": ProcessMask(),
        "action": Clip(min_val=-1, max_val=1),
    })
