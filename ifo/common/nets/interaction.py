from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class InteractionEmbeddings(nn.Module):
    """Token embeddings for the IM-LAM interaction bottleneck.

    Provides exactly two embedding families, added to attention queries and keys
    (never to values, which carry unmodified content):

    - a **spatial positional** embedding ``p``, one vector per bottleneck cell, so
      each token carries its own location. This matters because the write-back is a
      position-aligned residual, and because ``p`` must be re-added at every attention
      operation - the residual convolutions of ``F_A`` would otherwise transform away
      whatever positional signal had been baked into the features.
    - a **temporal** embedding, ``temporal_cur`` / ``temporal_pred``, marking whether a
      token describes the current or the predicted-next state.

    The temporal embedding is load-bearing rather than cosmetic. Because
    ``\hat{A}_{t+1} = F_A(A_t, z_t)`` is a *residual* update, the current and predicted agent
    token blocks are numerically near-identical early in training; without a temporal
    tag the object query could not reliably tell them apart, and attending to the
    *predicted agent transition* is precisely the hypothesis under test. Both vectors
    get a small non-zero init (truncated normal, std 0.02), so they are distinguishable
    from step 0 rather than only after learning.

    Deliberately absent is a per-entity embedding ``e_A``/``e_O``. In entity extraction
    the agent and object read-outs are already separated by their distinct mask biases
    and by separate attention weight sets; in the directed cross-attention an entity tag
    would be added to *all* keys (agent tokens by construction) and *all* queries (object
    tokens), making it constant across the softmax and unable to discriminate anything.
    """

    def __init__(self, dim: int, num_tokens: int) -> None:
        """Instantiate the interaction-module token embeddings.

        Both arguments are derived from the FDM bottleneck geometry at construction time -
        neither is a free hyperparameter, and neither has a default, so the caller must
        compute them from the encoder's ``final_encoder_shape`` rather than hardcoding. For
        DMW they are ``dim = 192`` and ``num_tokens = 16 * 16 = 256``, but relocating the
        interaction module to an earlier encoder stage (e.g. a ``32 x 32``, 1024-token
        bottleneck, per the proposal's small-object option) changes both.

        Args:
            dim (int): Token width = bottleneck channel count (``C`` of ``final_encoder_shape``).
            num_tokens (int): Number of bottleneck cells = ``H' * W'`` of ``final_encoder_shape``.
        """
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens

        self.spatial = nn.Parameter(torch.empty(num_tokens, dim))
        self.temporal_cur = nn.Parameter(torch.empty(dim))
        self.temporal_pred = nn.Parameter(torch.empty(dim))

        # Small, non-zero init. Non-zero is load-bearing: F_A is a residual update, so
        # A_t ~= \hat{A}_{t+1} early in training, and e_cur/e_pred must be distinguishable from
        # step 0. Truncated normal (bounded at +/-2 sigma) is the standard choice for 
        # learned token embeddings - it removes the rare large tail a
        # plain Gaussian would produce, giving a more predictable initial scale. The
        # explicit bounds matter: torch's default a=-2, b=2 would never truncate at std=0.02.
        std = 0.02
        nn.init.trunc_normal_(self.spatial, std=std, a=-2 * std, b=2 * std)
        nn.init.trunc_normal_(self.temporal_cur, std=std, a=-2 * std, b=2 * std)
        nn.init.trunc_normal_(self.temporal_pred, std=std, a=-2 * std, b=2 * std)

    def tag(self, x: Tensor, temporal: Optional[str] = None) -> Tensor:
        """Add the spatial (and optionally temporal) embedding to a token grid.

        Args:
            x (Tensor): Token grid of shape ``(B, num_tokens, dim)``.
            temporal (Optional[str]): ``"cur"`` for current-state tokens, ``"pred"`` for
                predicted-next-state tokens, or ``None`` to add the spatial embedding only.

        Returns:
            Tensor: Tagged tokens of shape ``(B, num_tokens, dim)``.
        """
        out = x + self.spatial
        if temporal is None:
            return out
        if temporal == "cur":
            return out + self.temporal_cur
        if temporal == "pred":
            return out + self.temporal_pred
        raise ValueError(f"temporal must be one of None, 'cur', 'pred'; got {temporal!r}.")


def pool_mask_occupancy(mask: Tensor, grid_hw: Union[int, Tuple[int, int]]) -> Tensor:
    """Average-pool a binary current-frame mask to bottleneck resolution.

    Pools the binary ``{0, 1}`` mask (NOT a resized or re-thresholded image) so each output
    cell holds the true fractional occupancy in ``[0, 1]`` of its input patch - the soft
    ``W_j`` that :class:`MaskBiasedAttention` adds as ``beta_h * W_j``. Bilinear resizing of an
    already-thresholded mask, or thresholding after pooling, would both discard that soft
    occupancy.

    The pooling factor is *derived* from the input and bottleneck sizes (``input_H // H'``),
    never hardcoded, so relocating the interaction module to a finer bottleneck (e.g. a
    ``32 x 32`` grid, the proposal's small-object option) needs no change here.

    Args:
        mask (Tensor): Binary mask ``(B, 1, H, W)`` or ``(B, H, W)``, values in ``{0, 1}``.
        grid_hw (Union[int, Tuple[int, int]]): Bottleneck grid size - an int ``g`` meaning
            ``(g, g)``, or an explicit ``(H', W')``. ``H`` and ``W`` must be divisible by the
            corresponding grid dimension.

    Returns:
        Tensor: ``(B, H' * W')`` fractional occupancy in ``[0, 1]``, flattened row-major so it
        aligns with the row-major flatten of the bottleneck token grid ``B_t``.
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)
    b, c, h, w = mask.shape
    if c != 1:
        raise ValueError(f"expected a single-channel mask, got {c} channels.")
    gh, gw = (grid_hw, grid_hw) if isinstance(grid_hw, int) else grid_hw
    if h % gh != 0 or w % gw != 0:
        raise ValueError(f"input {h}x{w} not divisible by bottleneck grid {gh}x{gw}.")
    occupancy = F.avg_pool2d(mask.float(), kernel_size=(h // gh, w // gw))  # (B, 1, gh, gw)
    return occupancy.reshape(b, gh * gw)
