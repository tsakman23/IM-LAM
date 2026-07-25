import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ifo.common.nets.impala_cnn import ImpalaResidualBlock
from ifo.common.nets.mask_biased_attention import MaskBiasedAttention


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


class AgentDynamics(nn.Module):
    """``F_A``: predict the next agent state ``\hat{A}_{t+1}`` from ``A_t`` and the latent action ``z_t``.

    Reuses MaskLAM's latent-injection mechanism: ``z_t`` is projected by a dedicated
    ``act_proj`` (``Linear(code_dim, dim * num_tokens)``) to a full bottleneck-shaped tensor and
    concatenated channel-wise with ``A_t``, then a small residual convolutional stack fuses the
    two. The update is a residual on ``A_t``: ``\hat{A}_{t+1} = A_t + ResBlocks(fuse([A_t, z]))``.

    The ``act_proj`` here is a *separate* instance from any stock ``ImpalaWorldModel.act_proj``: the
    interaction FDM removes MaskLAM's channel-wise ``z_t`` concat at the decoder input and injects
    ``z_t`` through ``F_A`` instead. Keeping the same projection mechanism (rather than a cheaper
    FiLM/additive scheme) means Foreground-MaskLAM and IM-LAM share an identical latent-injection
    path, so any difference between them is attributable to the interaction module, not to how
    ``z_t`` enters the FDM.

    Structurally ``F_A`` is one MaskLAM FDM block minus the downsample: a channel-changing conv
    (``fuse``, analogous to an ``ImpalaEncoderBlock``'s leading conv) followed by residual blocks, at
    fixed bottleneck resolution. ``num_res_blocks`` therefore mirrors MaskLAM's FDM
    ``encoder_num_res_blocks`` (``= 2`` in the DMW config), using the same ``ImpalaResidualBlock``
    class - so ``F_A`` introduces no residual-depth confound versus the Foreground-MaskLAM baseline.
    ``num_res_blocks`` is a **required keyword-only argument with no default**, so it cannot silently
    diverge from the FDM: Phase 3 threads the FDM's actual ``encoder_num_res_blocks`` in (a forgotten
    thread is a loud ``TypeError``, not a hidden ``2``) and asserts ``f_a.num_res_blocks ==
    encoder_num_res_blocks`` as a guard.

    Assumes a square bottleneck grid (``H' = W' = sqrt(num_tokens)``); DMW's ``16x16`` and the
    proposal's ``32x32`` small-object option are both square.
    """

    def __init__(self, dim: int, num_tokens: int, code_dim: int = 128, *, num_res_blocks: int) -> None:
        """Instantiate F_A.

        Args:
            dim (int): Bottleneck channel width.
            num_tokens (int): Number of bottleneck cells ``H' * W'`` (must be a perfect square).
            code_dim (int, optional): Latent-action dimension ``z_t`` (``128`` for DMW).
            num_res_blocks (int): IMPALA residual blocks in the fuse stack. **Required, no default** -
                it must equal MaskLAM's FDM ``encoder_num_res_blocks`` to avoid a residual-depth
                confound, so the caller (Phase 3's ``InteractionWorldModel``) threads that config value
                in. Removing the default makes a forgotten thread a loud ``TypeError``, not a silent 2.
        """
        super().__init__()
        side = math.isqrt(num_tokens)
        if side * side != num_tokens:
            raise ValueError(
                f"AgentDynamics assumes a square bottleneck grid; num_tokens={num_tokens} "
                "is not a perfect square."
            )
        self.dim = dim
        self.side = side
        self.num_res_blocks = num_res_blocks  # exposed so InteractionWorldModel can assert == encoder_num_res_blocks
        # Same mechanism as MaskLAM's FDM (own instance): project z_t to the full bottleneck tensor.
        self.act_proj = nn.Linear(code_dim, dim * num_tokens)
        self.fuse = nn.Conv2d(2 * dim, dim, kernel_size=3, padding=1) # channel-changing conv, analogous to an ImpalaEncoderBlock's leading conv
        self.res_blocks = nn.Sequential(*[ImpalaResidualBlock(dim) for _ in range(num_res_blocks)])

    def forward(self, a_t: Tensor, z: Tensor) -> Tensor:
        """Args: ``a_t`` ``(B, N, dim)`` agent tokens, ``z`` ``(B, code_dim)``. Returns ``(B, N, dim)``."""
        b, n, d = a_t.shape
        a_sp = a_t.transpose(1, 2).reshape(b, d, self.side, self.side)        # (B, dim, H', W')
        z_sp = self.act_proj(z).reshape(b, self.dim, self.side, self.side)    # (B, dim, H', W')
        delta = self.res_blocks(self.fuse(torch.cat([a_sp, z_sp], dim=1)))    # (B, dim, H', W')
        a_hat_sp = a_sp + delta                                              # residual on A_t
        return a_hat_sp.flatten(2).transpose(1, 2)                            # (B, N, dim) tokens


class InteractionModule(nn.Module):
    """The IM-LAM interaction bottleneck.

    Operates on the FDM bottleneck token grid ``B_t`` of shape ``(B, N, dim)``. Implements so far
    mask-biased entity extraction and directed agent dynamics. The object dynamics 
    (directed cross-attention + ``F_O``) and gated write-back are added in subsequent 
    tasks, after which a ``forward`` will compose them.

    ``dim`` and ``num_tokens`` are the bottleneck geometry (channel width and ``H' * W'``),
    derived by the caller from the FDM encoder's ``final_encoder_shape`` - not free
    hyperparameters. A square grid (``H' = W'``) is assumed by ``F_A``.
    """

    def __init__(
        self,
        dim: int,
        num_tokens: int,
        num_heads: int = 4,
        code_dim: int = 128,
        *,
        num_res_blocks: int,
    ) -> None:
        """Instantiate the interaction module.

        Args:
            dim (int): Bottleneck channel width (token dim), e.g. ``192`` for DMW.
            num_tokens (int): Number of bottleneck cells ``H' * W'``, e.g. ``256`` for DMW.
            num_heads (int, optional): Attention heads per entity read-out head. Must divide ``dim``.
            code_dim (int, optional): Latent-action dimension ``z_t`` (set to 128 in alignment with MaskLAM).
            num_res_blocks (int): IMPALA residual blocks in ``F_A``'s fuse stack. **Required, no default**;
                the caller must pass the FDM's ``encoder_num_res_blocks`` so the two cannot drift.
        """
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.embeddings = InteractionEmbeddings(dim, num_tokens)
        # Two independently parameterized read-out heads. Separate weight sets, together
        # with the distinct mask biases, are what separate the agent and object read-outs -
        # which is why the design deliberately carries no per-entity embedding.
        self.msa_agent = MaskBiasedAttention(dim, num_heads)
        self.msa_object = MaskBiasedAttention(dim, num_heads)
        self.f_a = AgentDynamics(dim, num_tokens, code_dim, num_res_blocks=num_res_blocks)

    def extract(self, b_t: Tensor, w_agent: Tensor, w_object: Tensor) -> Tuple[Tensor, Tensor]:
        """Mask-biased entity extraction: read agent/object tokens out of ``B_t``.

        Both read-outs are mask-biased *self*-attention over ``B_t``: queries and keys carry
        the spatial embedding (and the current-state temporal tag on the query), while the
        values are the untagged ``B_t`` content. The agent head is biased toward the agent
        occupancy and the object head toward the object occupancy, so each bottleneck
        location produces an agent read-out and an object read-out, each reweighted toward
        its own entity's region.

        Args:
            b_t (Tensor): Bottleneck token grid ``(B, N, dim)``.
            w_agent (Tensor): Agent occupancy ``(B, N)`` in ``[0, 1]`` (pooled current-frame mask).
            w_object (Tensor): Object occupancy ``(B, N)`` in ``[0, 1]``.

        Returns:
            Tuple[Tensor, Tensor]: agent read-out ``A_t`` and object read-out ``O_t``, each ``(B, N, dim)``.
        """
        query = self.embeddings.tag(b_t, temporal="cur")  # B_t + p + e_cur
        key = self.embeddings.tag(b_t, temporal=None)      # B_t + p
        value = b_t                                         # untagged content
        a_t = self.msa_agent(query, key, value, mask_bias=w_agent)
        o_t = self.msa_object(query, key, value, mask_bias=w_object)
        return a_t, o_t

    def agent_dynamics(self, a_t: Tensor, z: Tensor) -> Tensor:
        """``\hat{A}_{t+1} = F_A(A_t, z_t)``: inject the latent action into the agent pathway.

        The object branch has no direct ``z_t`` input; it will attend to this predicted agent
        transition instead, so this is the sole entry point for ``z_t`` into the FDM's spatial path.
        """
        return self.f_a(a_t, z)
