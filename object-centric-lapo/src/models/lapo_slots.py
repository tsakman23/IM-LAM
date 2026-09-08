from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ResidualMLPBlock(nn.Module):
    """Residual MLP block: x + fc2(gelu(fc1(x)))."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.fc2(F.gelu(self.fc1(x)))


class SlotIDM(nn.Module):
    """Inverse Dynamics Model for slot embeddings.

    Maps (s_t, s_future) -> z_t (latent action).
    """

    def __init__(
        self,
        slot_dim: int = 128,
        hidden_dim: int = 1024,
        latent_action_dim: int = 8192,
        num_residual_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(2 * slot_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim) for _ in range(num_residual_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, latent_action_dim)

    def forward(self, s_t: Tensor, s_future: Tensor) -> Tensor:
        """Infer latent action from current and future slot embeddings.

        Args:
            s_t: (B, D_slot) current slot embedding.
            s_future: (B, D_slot) future slot embedding.

        Returns:
            z_t: (B, D_latent) latent action.
        """
        x = self.input_proj(F.gelu(torch.cat([s_t, s_future], dim=-1)))
        x = self.blocks(x)
        return self.output_proj(x)


class SlotFDM(nn.Module):
    """Forward Dynamics Model for slot embeddings.

    Maps (s_t, z_t) -> s_hat_{t+k} (predicted future slot).
    """

    def __init__(
        self,
        slot_dim: int = 128,
        hidden_dim: int = 1024,
        latent_action_dim: int = 8192,
        num_residual_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(slot_dim + latent_action_dim, hidden_dim)
        self.blocks = nn.Sequential(
            *[ResidualMLPBlock(hidden_dim) for _ in range(num_residual_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, slot_dim)

    def forward(self, s_t: Tensor, z_t: Tensor) -> Tensor:
        """Predict future slot from current slot and latent action.

        Args:
            s_t: (B, D_slot) current slot embedding.
            z_t: (B, D_latent) latent action.

        Returns:
            s_hat: (B, D_slot) predicted future slot.
        """
        x = self.input_proj(F.gelu(torch.cat([s_t, z_t], dim=-1)))
        x = self.blocks(x)
        return self.output_proj(x)


class LAPOSlots(nn.Module):
    """LAPO-slots: Latent action model operating in slot embedding space.

    Composes IDM + FDM. Trained with MSE loss on future slot prediction.
    """

    def __init__(
        self,
        slot_dim: int = 128,
        hidden_dim: int = 1024,
        latent_action_dim: int = 8192,
        num_residual_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.idm = SlotIDM(slot_dim, hidden_dim, latent_action_dim, num_residual_blocks)
        self.fdm = SlotFDM(slot_dim, hidden_dim, latent_action_dim, num_residual_blocks)

    def forward(self, s_t: Tensor, s_future: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass: infer latent action and predict future slot.

        Args:
            s_t: (B, D_slot) current slot.
            s_future: (B, D_slot) target future slot.

        Returns:
            z_t: (B, D_latent) inferred latent action.
            s_hat: (B, D_slot) predicted future slot.
        """
        z_t = self.idm(s_t, s_future)
        s_hat = self.fdm(s_t, z_t)
        return z_t, s_hat

    def infer_action(self, s_t: Tensor, s_future: Tensor) -> Tensor:
        """Infer latent action only (no FDM)."""
        return self.idm(s_t, s_future)

    def compute_loss(self, s_hat: Tensor, s_future: Tensor) -> Tensor:
        """MSE loss between predicted and target future slot."""
        return F.mse_loss(s_hat, s_future)
