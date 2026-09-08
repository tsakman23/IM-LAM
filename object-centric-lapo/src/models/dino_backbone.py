import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DINOv2Backbone(nn.Module):
    """Frozen DINOv2 ViT backbone for feature extraction.

    Loads a pretrained DINOv2 model from timm, freezes all parameters,
    and extracts patch tokens. Internally handles preprocessing:
    input centered [-0.5, 0.5] -> uncenter to [0, 1] -> ImageNet normalize -> resize to target.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model_name: str = "vit_base_patch14_dinov2.lvd142m",
        image_size: int = 518,
    ) -> None:
        super().__init__()
        import timm

        self.model = timm.create_model(model_name, pretrained=True, img_size=image_size)
        self.image_size = image_size
        self.feature_dim = self.model.embed_dim  # 768 for base
        self.patch_size = self.model.patch_embed.patch_size[0]  # 14

        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        # Register normalization buffers
        self.register_buffer(
            "pixel_mean",
            torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "pixel_std",
            torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1),
        )

    def train(self, mode: bool = True):
        """Override to keep backbone always in eval mode."""
        super().train(mode)
        self.model.eval()
        return self

    def preprocess(self, x: Tensor) -> Tensor:
        """Preprocess centered images for DINO.

        Args:
            x: (B, C, H, W) in [-0.5, 0.5].

        Returns:
            (B, C, target_size, target_size) ImageNet-normalized.
        """
        # Uncenter to [0, 1]
        x = x + 0.5
        # Resize to DINO input size
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        # ImageNet normalize
        x = (x - self.pixel_mean) / self.pixel_std
        return x

    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        """Extract patch tokens from images.

        Args:
            x: (B, C, H, W) in [-0.5, 0.5].

        Returns:
            (B, N_patches, feature_dim) patch token features.
        """
        x = self.preprocess(x)
        # Use timm's forward_features which returns (B, 1+N, D) with CLS token
        features = self.model.forward_features(x)
        # Remove CLS token (first position)
        patch_tokens = features[:, 1:, :]
        return patch_tokens

    @property
    def num_patches(self) -> int:
        """Number of patch tokens for the configured image size."""
        n = self.image_size // self.patch_size
        return n * n
