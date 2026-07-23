from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn


class MaskBiasedAttention(nn.Module):
    """Multi-head attention with a learnable per-head additive mask bias.

    Borrows Mask2Former's idea of a mask-derived attention bias, but as a soft
    additive logit bias rather than a hard ``-inf`` gate. For query ``i``, key ``j``
    and head ``h`` the pre-softmax score is

        s_ij^h = (q_i * k_j) / sqrt(d_h) + beta_h * W_j

    where ``W_j in [0, 1]`` is a soft occupancy (the pooled current-frame mask) and
    ``beta_h`` is a learnable per-head scalar initialized to 2.0. ``beta_h -> inf``
    recovers hard masked attention; ``beta_h = 0`` recovers unbiased attention.

    Unlike ``ifo.common.layers.transformer.attention.SelfAttention`` (which fuses
    QKV into one projection, forbids an attention bias, and never returns weights),
    this primitive is built for IM-LAM's interaction module, which needs: separate
    Q/K/V sources (self- and cross-attention over different tensors), the additive
    per-head mask bias, and optional attention-map return for diagnostics. The
    softmax is upcast to fp32 and cast back, since training runs in bf16 where
    attention logits are a known instability source.
    """

    def __init__(self, dim: int, num_heads: int, proj_bias: bool = True) -> None:
        """Instantiate mask-biased attention.

        Args:
            dim (int): Model / token width. Must be divisible by ``num_heads``.
            num_heads (int): Number of attention heads.
            proj_bias (bool, optional): Whether the Q/K/V/output projections use bias.
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.k_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.v_proj = nn.Linear(dim, dim, bias=proj_bias)
        self.out_proj = nn.Linear(dim, dim, bias=proj_bias)

        # Learnable per-head bias scale, initialized to 2.0 (proposal S5.5 / A.2 N3).
        self.beta = nn.Parameter(torch.full((num_heads,), 2.0))

    def _split_heads(self, x: Tensor) -> Tensor:
        """(B, N, dim) -> (B, num_heads, N, head_dim)."""
        b, n, _ = x.shape
        return x.reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        q_in: Tensor,
        k_in: Tensor,
        v_in: Tensor,
        mask_bias: Optional[Tensor] = None,
        return_attn: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass.

        Args:
            q_in (Tensor): Query source of shape ``(B, Nq, dim)``.
            k_in (Tensor): Key source of shape ``(B, Nk, dim)``.
            v_in (Tensor): Value source of shape ``(B, Nk, dim)``.
            mask_bias (Optional[Tensor]): Soft occupancy of shape ``(B, Nk)`` in
                ``[0, 1]``, added per head as ``beta_h * mask_bias_j``. ``None``
                (or an all-zero tensor) yields unbiased attention.
            return_attn (bool, optional): If True, also return the attention weights
                of shape ``(B, num_heads, Nq, Nk)``. Kept off by default so the
                training hot path neither allocates them nor triggers recompilation.

        Returns:
            Tensor ``(B, Nq, dim)``, or ``(output, attn_weights)`` when ``return_attn``.
        """
        b, nq, _ = q_in.shape
        nk = k_in.shape[1]

        q = self._split_heads(self.q_proj(q_in))  # (B, H, Nq, hd)
        k = self._split_heads(self.k_proj(k_in))  # (B, H, Nk, hd)
        v = self._split_heads(self.v_proj(v_in))  # (B, H, Nk, hd)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, H, Nq, Nk)
        if mask_bias is not None:
            # beta_h * W_j, broadcast over batch and queries.
            bias = self.beta.reshape(1, self.num_heads, 1, 1) * mask_bias.reshape(b, 1, 1, nk)
            scores = scores + bias

        # Upcast the softmax to fp32 for numerical stability under bf16 training.
        attn = torch.softmax(scores.float(), dim=-1).to(v.dtype)
        ctx = torch.matmul(attn, v)  # (B, H, Nq, hd)
        ctx = ctx.transpose(1, 2).reshape(b, nq, self.dim)
        out = self.out_proj(ctx)

        if return_attn:
            return out, attn
        return out
