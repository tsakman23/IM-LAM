from typing import Dict, Optional, Union

import torch
from tensordict import TensorDict
from torch import Tensor
from torchvision.transforms.v2 import Compose, Lambda, Resize, ToDtype, Transform

from ifo.common.utils.actions import DiscretizeMineRLAction
from ifo.common.utils.functions import center


class DictTransform:
    """Applies a set of transforms to a Dict of Tensors or a TensorDict."""

    def __init__(self, transform: Dict[str, Transform]) -> None:
        """
        Initializes DictTransform.

        Args:
            transform_dict (Dict[str, Transform]): Dictionary of transformations, where the key is the name of the field
                to be transformed and the value is the transformation to be applied.
        """
        self.transform = transform

    def __call__(self, x: Union[TensorDict, Dict[str, Tensor]]) -> Union[TensorDict, Dict[str, Tensor]]:
        """
        Applies the specified transforms to the data dictionary.

        Args:
            x (Union[TensorDict, Dict[str, Tensor]): Dictionary containing the data to be transformed.

        Returns:
            Union[TensorDict, Dict[str, Tensor]: Dictionary containing the transformed data.
        """
        assert isinstance(x, (TensorDict, dict))
        for key, transform in self.transform.items():
            if key in x:
                x[key] = transform(x[key])
            else:
                raise ValueError(f"Key {key} not found in dictionary.")
        return x


class DiscretizeMineRLActionTransform(Transform):
    """Transforms MineRL actions to a discrete factored space."""

    def __init__(
        self,
        camera_maxval: int = 10,
        camera_binsize: int = 2,
        camera_quantization_scheme: str = "mu_law",
        camera_mu: int = 10,
        n_camera_bins: int = 11,
    ) -> None:
        """
        Initializes DiscreteMineRLAction transformation.

        Args:
            camera_maxval (int, optional): The maximum value for the camera quantizer.
            camera_binsize (int, optional): The bin size for the camera quantizer.
            camera_quantization_scheme (str, optional): The quantization scheme for the camera quantizer.
            camera_mu (float, optional): The mu parameter for the camera quantizer.
            n_camera_bins (int, optional): The number of camera bins in the factored space.
        """
        super().__init__()
        self.transform = DiscretizeMineRLAction(
            camera_maxval=camera_maxval,
            camera_binsize=camera_binsize,
            camera_quantization_scheme=camera_quantization_scheme,
            camera_mu=camera_mu,
            n_camera_bins=n_camera_bins,
        )

    def forward(self, inpt: TensorDict) -> TensorDict:
        """Convert MineRL actions to a discrete factored space.

        Args:
            inpt (TensorDict): TensorDict containing the actions.

        Returns:
            TensorDict: TensorDict containing the transformed actions.
        """
        output = {k: v.numpy() for k, v in inpt.items()}
        output = self.transform.forward(output)
        return TensorDict(output, batch_size=inpt.batch_size)


class Permute(Transform):
    """Permute the dimensions of a tensor from HWC to CHW format.

    This transform is useful for converting image data from height-width-channel format
    to channel-height-width format, which is the standard format for PyTorch.
    """

    def __init__(self) -> None:
        """Initialize the Permute transform."""
        super().__init__()

    def forward(self, inpt: torch.Tensor) -> torch.Tensor:
        """Permute the dimensions of the input tensor from HWC to CHW.

        Args:
            inpt (torch.Tensor): Input tensor in HWC format.

        Returns:
            torch.Tensor: Output tensor in CHW format.
        """
        # Get the number of dimensions
        ndim = inpt.dim()
        assert ndim >= 3, "Input tensor must have at least 3 dimensions."
        # Create permutation tuple: keep all dimensions except last 3, then permute last 3
        perm = tuple(range(ndim - 3)) + (ndim - 1, ndim - 3, ndim - 2)
        return inpt.permute(perm)


class Clip(Transform):
    """Clip values in a tensor to a specified range.

    This transform clips the values in a tensor to be within the specified minimum and maximum values.
    Values below min_val are set to min_val, and values above max_val are set to max_val.
    """

    def __init__(self, min_val: float, max_val: float) -> None:
        """Initialize the Clip transform.

        Args:
            min_val (float): Minimum value to clip to.
            max_val (float): Maximum value to clip to.
        """
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, inpt: Tensor) -> Tensor:
        """Clip the input tensor values to the specified range.

        Args:
            inpt (Tensor): Input tensor to clip.

        Returns:
            Tensor: Clipped tensor with values between min_val and max_val.
        """
        return torch.clamp(inpt, min=self.min_val, max=self.max_val)


class ProcessMask(Transform):
    """Convert uint8 mask (H, W) ∈ {0,255} to float32 (1, H, W) ∈ {0,1}"""

    def forward(self, inpt: Tensor) -> Tensor:
        """Convert uint8 mask (H, W) ∈ {0,255} to float32 (1, H, W) ∈ {0,1}
        Args:
            inpt (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        # uint8 {0,255} -> float32 {0,1}
        mask = (inpt > 0).to(torch.float32)

        return mask


def procgen() -> DictTransform:
    """Apply dataset transformations for Procgen environment.

    Returns:
        DictTransform: Transformations for Procgen environment.
    """
    transform = DictTransform(
        {
            "observation": Compose([ToDtype(torch.float32), Lambda(center)]),
        }
    )
    return transform


def minerl() -> DictTransform:
    """Apply dataset transformations for MineRL environment.

    Returns:
        DictTransform: Transformations for MineRL environment.
    """
    transform = DictTransform(
        {
            "observation": Compose([Resize((320, 180)), ToDtype(torch.float32), Lambda(center)]),
            "action": DiscretizeMineRLActionTransform(),
        }
    )
    return transform


def dcs() -> DictTransform:
    """Apply dataset transformations for Distracting Control Suite environment.

    Returns:
        DictTransform: Transformations for Distracting Control Suite environment.
    """
    transform = DictTransform(
        {
            "observation": Compose([Permute(), ToDtype(torch.float32), Lambda(center)]),
            "action": Clip(min_val=-1, max_val=1),
        }
    )
    return transform


def dm_control() -> DictTransform:
    """Apply dataset transformations for DM Control environment.

    Returns:
        DictTransform: Transformations for DM Control environment.
    """
    transform = DictTransform(
        {
            "observation": Compose([ToDtype(torch.float32), Lambda(center)]),
            "action": Clip(min_val=-1, max_val=1),
        }
    )
    return transform


def dcs_masked() -> DictTransform:
    """Apply dataset transformations for Masked Distracting Control Suite Segmentation environment.

    Returns:
       DictTransform: Transformations for Masked Distracting Control Suite Segmentation environment.
    """

    transform = DictTransform(
        {
            "observation": Compose([ToDtype(torch.float32), Lambda(center)]),
            "mask": ProcessMask(),
            "action": Clip(min_val=-1, max_val=1),
        }
    )
    return transform



def metaworld(with_object_mask: bool = False, with_object_state: bool = False) -> DictTransform:
    """Apply dataset transformations for Meta-World environment.

    Args:
        with_object_mask (bool): If True, also process an ``object_mask`` column
            with the same mask transform as ``mask`` (used by the
            Foreground-MaskLAM baseline, which needs the manipulated-object mask).
        with_object_state (bool): If True, also cast an ``object_state`` column
            (world-frame object position + orientation, present only in the
            object-mask repos) to float32. Needed by the object-dynamics probe,
            which regresses k-step object-state deltas from frozen features.

    Returns:
       DictTransform: Transformations for Meta-World environment.
    """

    transforms = {
        "observation": Compose([ToDtype(torch.float32), Lambda(center)]),
        "mask": ProcessMask(),
        "action": Clip(min_val=-1, max_val=1),
    }
    if with_object_mask:
        transforms["object_mask"] = ProcessMask()
    if with_object_state:
        transforms["object_state"] = ToDtype(torch.float32)
    return DictTransform(transforms)

def get_dataset_transform(
    dataset_name: Optional[str], with_object_mask: bool = False, with_object_state: bool = False
) -> DictTransform:
    """Get dataset transformations for the specified dataset.

    Args:
        dataset_name (str): Name of the dataset.
        with_object_mask (bool): If True, Meta-World transforms also process an
            ``object_mask`` column (Foreground-MaskLAM). Ignored for other datasets.
        with_object_state (bool): If True, Meta-World transforms also process an
            ``object_state`` column (object-dynamics probe). Ignored for other datasets.

    Returns:
        DictTransform: Transformations for the specified dataset.
    """
    if dataset_name is None:
        return None
    if "procgen" in dataset_name:
        return procgen()
    elif "MineRL" in dataset_name:
        return minerl()
    elif "dm_control" in dataset_name:
        if "masked" in dataset_name:
            return dcs_masked()
        elif "distracting" in dataset_name:
            return dcs()
        else:
            return dm_control()
    elif "Meta-World" in dataset_name:
        return metaworld(with_object_mask=with_object_mask, with_object_state=with_object_state)
    else:
        raise NotImplementedError("Dataset is currently not supported.")
