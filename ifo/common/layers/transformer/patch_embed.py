import math
from typing import Callable, Optional, Tuple, Union

from torch import Tensor, nn


def make_2tuple(x: int | Tuple[int, int]) -> Tuple[int, int]:
    """Convert an integer or 2-tuple to a 2-tuple.

    Args:
        x (int | Tuple[int, int]): Input value, either an integer or a 2-tuple of integers.

    Returns:
        Tuple[int, int]: A 2-tuple of integers. If input is an integer, returns (x, x).

    Raises:
        AssertionError: If input is a tuple but not of length 2, or if input is neither
            an integer nor a tuple.
    """
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """2D image to patch embedding layer.

    Converts a 2D image tensor from shape (B, C, H, W) to patch embeddings of shape
    (B, N, D), where N is the number of patches and D is the embedding dimension.
    This is typically the first layer in Vision Transformer architectures.

    The layer uses a 2D convolution with kernel size and stride equal to the patch size
    to efficiently extract non-overlapping patches and project them to the embedding space.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        flatten_embedding: bool = True,
    ) -> None:
        """Initialize the PatchEmbed layer.

        Args:
            img_size (Union[int, Tuple[int, int]]): Input image size. If int, assumes
                square image. Defaults to 224.
            patch_size (Union[int, Tuple[int, int]]): Patch token size. If int, assumes
                square patches. Defaults to 16.
            in_channels (int): Number of input image channels. Defaults to 3.
            embed_dim (int): Number of linear projection output channels (embedding dimension).
                Defaults to 768.
            norm_layer (Optional[Callable[..., nn.Module]]): Normalization layer to apply
                after projection. If None, no normalization is applied. Defaults to None.
            flatten_embedding (bool): Whether to flatten spatial dimensions. If True, output
                shape is (B, N, D). If False, output shape is (B, H, W, D). Defaults to True.
        """
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass to convert image to patch embeddings.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            Tensor: Patch embeddings. If flatten_embedding is True, shape is (B, N, D)
                where N is the number of patches. If False, shape is (B, H, W, D) where
                H and W are the patch grid dimensions.
        """
        _, _, H, W = x.shape
        # patch_H, patch_W = self.patch_size
        # assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        # assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        """Calculate the number of FLOPs (floating point operations) for this layer.

        Returns:
            float: Approximate number of FLOPs required for a forward pass.
        """
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_channels * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

    def reset_parameters(self) -> None:
        """Reset the parameters of the patch projection layer.

        Initializes the convolutional projection weights and biases using uniform
        distribution with bounds derived from the input channels and patch size,
        following standard initialization practices for convolutional layers.
        """
        k = 1 / (self.in_channels * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))
