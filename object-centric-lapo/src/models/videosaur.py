from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.models.dino_backbone import DINOv2Backbone
from src.models.slot_attention import SlotAttention, SlotAttentionDecoder


class VideoSAUR(nn.Module):
    """VideoSAUR: DINO backbone + Slot Attention + Decoder.

    Trains slot attention to decompose DINO features into object slots.
    The DINO backbone is frozen; only slot attention + decoder are trained.
    """

    def __init__(
        self,
        num_slots: int = 4,
        slot_dim: int = 128,
        num_iterations: int = 3,
        dino_model_name: str = "vit_base_patch14_dinov2.lvd142m",
        dino_image_size: int = 518,
        sim_weight: float = 0.1,
        sim_temperature: float = 0.075,
    ) -> None:
        super().__init__()
        self.sim_weight = sim_weight
        self.sim_temperature = sim_temperature

        # Frozen DINO backbone
        self.backbone = DINOv2Backbone(
            model_name=dino_model_name,
            image_size=dino_image_size,
        )
        feature_dim = self.backbone.feature_dim
        num_patches = self.backbone.num_patches

        # Trainable slot attention
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            input_dim=feature_dim,
            slot_dim=slot_dim,
            num_iterations=num_iterations,
        )

        # Trainable decoder
        self.decoder = SlotAttentionDecoder(
            num_patches=num_patches,
            slot_dim=slot_dim,
            feature_dim=feature_dim,
        )

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass for training.

        Args:
            images: (B, C, H, W) in [-0.5, 0.5].

        Returns:
            slots: (B, K, D_slot).
            recon_features: (B, N, D_feature).
            attn_masks: (B, K, N).
            target_features: (B, N, D_feature).
        """
        # Extract DINO features (frozen)
        with torch.no_grad():
            target_features = self.backbone(images)  # (B, N, D)

        # Slot attention
        slots = self.slot_attention(target_features)  # (B, K, D_slot)

        # Decode
        recon_features, attn_masks = self.decoder(slots)  # (B, N, D), (B, K, N)

        return slots, recon_features, attn_masks, target_features

    def compute_loss(
        self,
        recon_features: Tensor,
        target_features: Tensor,
    ) -> Tuple[Tensor, dict]:
        """Compute reconstruction loss.

        Args:
            recon_features: (B, N, D) predicted features.
            target_features: (B, N, D) DINO target features.

        Returns:
            Tuple of (total_loss, metrics_dict).
        """
        mse_loss = F.mse_loss(recon_features, target_features)
        metrics = {"mse": mse_loss.detach()}

        if self.sim_weight > 0:
            recon_norm = F.normalize(recon_features, dim=-1)
            target_norm = F.normalize(target_features, dim=-1)
            sim = (recon_norm * target_norm).sum(dim=-1) / self.sim_temperature
            sim_loss = 1 - sim.mean()
            total = mse_loss + self.sim_weight * sim_loss
            metrics["sim_loss"] = sim_loss.detach()
            metrics["cosine_sim"] = (recon_norm * target_norm).sum(dim=-1).mean().detach()
            return total, metrics

        return mse_loss, metrics

    @torch.no_grad()
    def extract_slots(self, images: Tensor) -> Tuple[Tensor, Tensor]:
        """Extract slots and attention masks (inference mode).

        Uses deterministic slot initialization (mu only, no noise).

        Args:
            images: (B, C, H, W) in [-0.5, 0.5].

        Returns:
            slots: (B, K, D_slot).
            attn_masks: (B, K, N).
        """
        was_training = self.training
        self.eval()

        features = self.backbone(images)
        slots = self.slot_attention(features)
        _, attn_masks = self.decoder(slots)

        if was_training:
            self.train()

        return slots, attn_masks
