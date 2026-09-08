from typing import Optional, Sequence

import numpy as np
import torch
import torchvision
from torch import Tensor


def make_comp_grid(prediction: Tensor, target: Tensor, padding: int = 2) -> Tensor:
    """Create a tensor grid of images, where predictions and targets are vertically stacked.

    Transformations can be applied optionally.

    Args:
        prediction (Tensor): Prediction tensor.
        target (Tensor): Target tensor.
        padding (int): Padding (pixels) between images.

    Returns:
        Tensor: Predictions and targets aligned in a grid for convenient comparison.
    """
    assert prediction.shape == target.shape
    assert len(prediction.shape) > 3  # Make sure we have images and a batch size.
    B, C, _, W = prediction.shape
    pad_tensor = torch.zeros((B, C, padding, W)).to(prediction.device)
    grid = torchvision.utils.make_grid(
        torch.cat((prediction, pad_tensor, target), dim=-2), nrow=int(B**0.5) * 2, padding=padding
    )
    return grid


def tile_images(images_nhwc: Sequence[np.ndarray]) -> np.ndarray:
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.

    Args:
        images_nhwc (Sequence[np.ndarray]): List or array of images, ndim=4 once turned into array.
            n = batch index, h = height, w = width, c = channel

    Returns:
        np.ndarray: The tiled image with shape (P*height, Q*width, c).

    """
    img_nhwc = np.asarray(images_nhwc)
    n_images, height, width, n_channels = img_nhwc.shape
    new_height = int(np.ceil(np.sqrt(n_images)))
    new_width = int(np.ceil(float(n_images) / new_height))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0] * 0 for _ in range(n_images, new_height * new_width)])
    out_image = img_nhwc.reshape((new_height, new_width, height, width, n_channels))
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    out_image = out_image.reshape((new_height * height, new_width * width, n_channels))
    return out_image


def vec_render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
    """
    Renders the environment and returns the rendered image as a numpy array.

    Args:
        mode (str, optional): The rendering mode. Defaults to None.

    Returns:
        np.ndarray or None: The rendered image as a numpy array if mode is 'rgb_array',
        otherwise None.

    Raises:
        None

    """
    if mode == "rgb_array":
        # call the render method of the environments
        images = self.unwrapped.call("render")
        # Create a big image by tiling images from subprocesses
        bigimg = tile_images(images)  # type: ignore[arg-type]
        return bigimg
    else:
        print(f"Warning: mode {mode} not supported in VecEnvWrapper")
    return None
