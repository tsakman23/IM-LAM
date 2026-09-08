import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SlotAttention(nn.Module):
    """Slot Attention module with learnable per-slot initialization.

    Uses GRU update + MLP residual per iteration.
    """

    def __init__(
        self,
        num_slots: int = 4,
        input_dim: int = 768,
        slot_dim: int = 128,
        num_iterations: int = 3,
    ) -> None:
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations

        # Learnable per-slot initialization
        self.slot_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim) * (slot_dim ** -0.5))
        self.slot_log_sigma = nn.Parameter(torch.zeros(1, num_slots, slot_dim))

        # Input projection
        self.norm_input = nn.LayerNorm(input_dim)
        self.project_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(input_dim, slot_dim, bias=False)

        # Slot projection
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)

        # Update
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, slot_dim * 4),
            nn.GELU(),
            nn.Linear(slot_dim * 4, slot_dim),
        )

        self.scale = slot_dim ** -0.5

    def forward(self, inputs: Tensor) -> Tensor:
        """Run slot attention.

        Args:
            inputs: (B, N, D_input) patch features.

        Returns:
            slots: (B, K, D_slot) slot representations.
        """
        b = inputs.shape[0]

        # Initialize slots
        if self.training:
            slots = self.slot_mu + torch.exp(self.slot_log_sigma) * torch.randn_like(self.slot_mu)
        else:
            slots = self.slot_mu.clone()
        slots = slots.expand(b, -1, -1)

        # Project inputs once
        inputs = self.norm_input(inputs)
        k = self.project_k(inputs)  # (B, N, D_slot)
        v = self.project_v(inputs)  # (B, N, D_slot)

        for _ in range(self.num_iterations):
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.project_q(slots)  # (B, K, D_slot)

            # Attention: softmax over slots (competition)
            attn_logits = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, K, N)
            attn = F.softmax(attn_logits, dim=1)  # Normalize over slots

            # Weighted mean
            attn_sum = attn.sum(dim=-1, keepdim=True) + 1e-8
            attn_norm = attn / attn_sum
            updates = torch.bmm(attn_norm, v)  # (B, K, D_slot)

            # GRU update
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            ).reshape(b, self.num_slots, self.slot_dim)

            # MLP residual
            slots = slots + self.mlp(slots)

        return slots


class SlotAttentionDecoder(nn.Module):
    """Decoder that reconstructs features from slots using cross-attention.

    Slot representations serve as keys/values; learned patch position embeddings
    serve as queries. Attention masks indicate which slot explains each patch.
    """

    def __init__(
        self,
        num_patches: int,
        slot_dim: int = 128,
        feature_dim: int = 768,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches
        self.slot_dim = slot_dim
        self.feature_dim = feature_dim

        # Position embeddings for patches (queries)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, slot_dim) * 0.02)

        # Slot projection for keys/values
        self.slot_proj = nn.Linear(slot_dim, slot_dim)

        # Output projection: slot_dim -> feature_dim per patch
        self.output_proj = nn.Sequential(
            nn.LayerNorm(slot_dim),
            nn.Linear(slot_dim, feature_dim),
        )

        self.scale = slot_dim ** -0.5

    def forward(self, slots: Tensor) -> tuple:
        """Decode slots into reconstructed features.

        Args:
            slots: (B, K, D_slot) slot representations.

        Returns:
            recon_features: (B, N_patches, D_feature) reconstructed features.
            attn_masks: (B, K, N_patches) attention masks per slot.
        """
        b, k, _ = slots.shape

        # Queries from position embeddings
        queries = self.pos_embed.expand(b, -1, -1)  # (B, N, D_slot)

        # Keys from slots
        keys = self.slot_proj(slots)  # (B, K, D_slot)

        # Cross-attention: query=positions, key/value=slots
        attn_logits = torch.bmm(queries, keys.transpose(1, 2)) * self.scale  # (B, N, K)
        attn_masks = F.softmax(attn_logits, dim=-1)  # (B, N, K) — softmax over slots

        # Weighted combination of slot features
        recon = torch.bmm(attn_masks, self.output_proj[1](self.output_proj[0](slots)))  # (B, N, D_feature)

        # Transpose attention for (B, K, N) format
        attn_masks = attn_masks.transpose(1, 2)  # (B, K, N)

        return recon, attn_masks
