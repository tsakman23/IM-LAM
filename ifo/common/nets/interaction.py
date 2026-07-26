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
        mlp_ratio: int = 4,
        direct_z_to_object: bool = False,
        dilate_iters: int = 1,
        *,
        num_res_blocks: int,
    ) -> None:
        """Instantiate the interaction module.

        Args:
            dim (int): Bottleneck channel width (token dim), e.g. ``192`` for DMW.
            num_tokens (int): Number of bottleneck cells ``H' * W'``, e.g. ``256`` for DMW.
            num_heads (int, optional): Attention heads (entity read-outs and directed cross-attention). Must divide ``dim``.
            code_dim (int, optional): Latent-action dimension ``z_t`` (set to 128 in alignment with MaskLAM).
            mlp_ratio (int, optional): Hidden-width multiplier for ``F_O``'s position-wise FFN.
            direct_z_to_object (bool, optional): Matched-ablation switch (proposal S8.3). When True, ``z_t``
                is injected into the object query directly (reusing ``F_A``'s ``act_proj`` so the parameter
                count is unchanged), breaking the ``z_t -> A_hat -> O_hat`` restriction. Default False.
            dilate_iters (int, optional): How many times the write-back applies the 3x3 max-pool dilation
                to the current-frame gates. Each pass expands the gate outward by one bottleneck cell so it
                can cover positions the entity occupies at t+1 but not at t (~8 input px per pass). Default 1.
            num_res_blocks (int): IMPALA residual blocks in ``F_A``'s fuse stack. **Required, no default**;
                the caller must pass the FDM's ``encoder_num_res_blocks`` so the two cannot drift.
        """
        super().__init__()
        if dilate_iters < 1:
            raise ValueError(
                f"dilate_iters must be >= 1 (0 reopens the t+1-coverage gap Dilate exists to close); got {dilate_iters}."
            )
        self.dim = dim
        self.num_tokens = num_tokens
        self.side = math.isqrt(num_tokens)  # square grid guaranteed by AgentDynamics below
        self.direct_z_to_object = direct_z_to_object
        self.dilate_iters = dilate_iters
        self.embeddings = InteractionEmbeddings(dim, num_tokens)
        # Two independently parameterized read-out heads. Separate weight sets, together
        # with the distinct mask biases, are what separate the agent and object read-outs -
        # which is why the design deliberately carries no per-entity embedding.
        self.msa_agent = MaskBiasedAttention(dim, num_heads)
        self.msa_object = MaskBiasedAttention(dim, num_heads)
        # LayerNorm goes exactly where the tags go - on the attention Q/K, never on V. B_t comes from a
        # norm-free IMPALA encoder with unbounded activation magnitude, so without this the content
        # logits can dwarf the mask bias beta*W (mask ~ [0, 2] at init), making the mask - the mechanism
        # that separates the agent and object read-outs - a rounding error, and rendering the logged
        # beta uninterpretable (healthy-looking beta while masks are effectively ignored). Values stay
        # unnormalized: A_t/O_t must keep IMPALA scale for F_A's residual conv blocks and for F_O's
        # residual add. (QK-norm is the documented escalation if extraction still proves unstable.)
        self.norm_extract = nn.LayerNorm(dim)  # extraction Q/K (shared by both entity heads)
        self.f_a = AgentDynamics(dim, num_tokens, code_dim, num_res_blocks=num_res_blocks)
        # F_O: the genuinely cross-attentive op (object queries attend to agent keys/values), plus a
        # position-wise FFN, in the standard pre-norm transformer arrangement. No mask bias here - the
        # keys are agent tokens by construction.
        self.cross_attn = MaskBiasedAttention(dim, num_heads)
        self.norm_obj = nn.LayerNorm(dim)    # F_O: object query stream
        self.norm_ctx = nn.LayerNorm(dim)    # F_O: agent K context (keys only; values stay unnormalized)
        self.norm_ffn = nn.LayerNorm(dim)    # F_O: pre-norm on the FFN sublayer
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim),
            nn.GELU(),
            nn.Linear(mlp_ratio * dim, dim),
        )
        # Gated write-back. Proj_A/Proj_O are the learnable gate on the residual update, ZERO-INITIALIZED
        # so B_hat = B_t exactly at init (the whole module is an identity residual until it learns). They
        # still receive gradient (a zero-conv: grad w.r.t. weight depends on the input, not the value), so
        # the interaction path opens from step 1. Their norms are logged from step 0 (must grow off zero).
        self.proj_a = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj_o = nn.Conv2d(dim, dim, kernel_size=1)
        self.zero_init_write_back()
        # Soft dilation of the current-frame gates (max-pool keeps them in [0, 1]).
        self.dilate = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def zero_init_write_back(self) -> None:
        """Zero the write-back projections so ``B_hat == B_t`` at init (identity / graceful degradation).

        ControlNet-style zero-conv (https://arxiv.org/abs/2302.05543): the output is exactly ``B_t`` at
        init but the projections still receive gradient (grad w.r.t. weight depends on the input, not the
        zero value), so the interaction path opens from step 1. **Must be re-callable, and re-called after
        any global init:** ``orthogonal_init`` (which the DMW config runs via ``use_orthogonal_init=True``)
        re-inits every ``Conv2d`` and would otherwise make ``Proj`` non-zero and silently destroy the
        identity. Phase 3's ``InteractionWorldModel`` calls this last, after its orthogonal init.
        """
        for proj in (self.proj_a, self.proj_o):
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def _to_spatial(self, tokens: Tensor) -> Tensor:
        """(B, N, dim) -> (B, dim, H', W') (row-major, matching the occupancy/pooling order)."""
        b, n, d = tokens.shape
        return tokens.transpose(1, 2).reshape(b, d, self.side, self.side)

    def _to_tokens(self, spatial: Tensor) -> Tensor:
        """(B, dim, H', W') -> (B, N, dim)."""
        return spatial.flatten(2).transpose(1, 2)

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
        b_norm = self.norm_extract(b_t)                     # normalize Q/K so beta*W is not swamped
        query = self.embeddings.tag(b_norm, temporal="cur")  # norm(B_t) + p + e_cur
        key = self.embeddings.tag(b_norm, temporal=None)     # norm(B_t) + p
        value = b_t                                          # untagged, UNNORMALIZED content
        a_t = self.msa_agent(query, key, value, mask_bias=w_agent)
        o_t = self.msa_object(query, key, value, mask_bias=w_object)
        return a_t, o_t

    def agent_dynamics(self, a_t: Tensor, z: Tensor) -> Tensor:
        """``\hat{A}_{t+1} = F_A(A_t, z_t)``: inject the latent action into the agent pathway.

        The object branch has no direct ``z_t`` input; it will attend to this predicted agent
        transition instead, so this is the sole entry point for ``z_t`` into the FDM's spatial path.
        """
        return self.f_a(a_t, z)

    def object_dynamics(
        self,
        o_t: Tensor,
        a_t: Tensor,
        a_hat: Tensor,
        z: Optional[Tensor] = None,
        agent_ctx_mode: str = "normal",
        return_attn: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """``\hat{O}_{t+1} = F_O(O_t, CrossAttn(O_t -> [A_t, \hat{A}_{t+1}]))``: directed object dynamics.

        The object query attends to the current and predicted agent tokens (keys tagged with the
        current/predicted temporal embeddings; values are the untagged agent content), then a
        position-wise FFN follows, in the standard transformer arrangement (attention residual on
        ``O_t``, then FFN residual). This is the one genuinely cross-attentive op in the model, and the
        only route by which ``z_t`` reaches the object prediction (through ``\hat{A}_{t+1} = F_A(A_t, z_t)``) -
        the architectural restriction the hypothesis tests.

        Args:
            o_t (Tensor): Object read-out ``(B, N, dim)``.
            a_t (Tensor): Current agent read-out ``(B, N, dim)``.
            a_hat (Tensor): Predicted next-agent tokens ``\hat{A}_{t+1}`` ``(B, N, dim)``.
            z (Optional[Tensor]): Latent action ``(B, code_dim)``. Only used when
                ``direct_z_to_object`` is set (the matched ablation); otherwise ignored.
            agent_ctx_mode (str): Eval-time substitution for the agent-path diagnostic (no retraining):
                ``"normal"`` uses ``[\tilde{A}_t, \tilde{A}_{t+1}]``; ``"no_transition"`` replaces the predicted block
                with the current one (``[\tilde{A}_t, \tilde{A}_t]``); ``"shuffled"`` replaces the predicted block with a
                batch-rolled copy (each sample gets another sample's predicted context).
            return_attn (bool): If True, also return the object->agent attention weights
                ``(B, num_heads, N, 2N)`` for the qualitative diagnostics. Off by default.

        Returns:
            Tensor ``(B, N, dim)``, or ``(o_hat, attn)`` when ``return_attn``.
        """
        if agent_ctx_mode == "normal":
            a2, a2_temporal = a_hat, "pred"
        elif agent_ctx_mode == "no_transition":
            a2, a2_temporal = a_t, "cur"
        elif agent_ctx_mode == "shuffled":
            # Roll by 1 along the batch: a deterministic derangement (no sample keeps its own
            # predicted context), unlike a random permutation which can leave fixed points.
            a2, a2_temporal = torch.roll(a_hat, shifts=1, dims=0), "pred"
        else:
            raise ValueError(
                f"agent_ctx_mode must be one of 'normal', 'no_transition', 'shuffled'; got {agent_ctx_mode!r}."
            )

        query = self.embeddings.tag(self.norm_obj(o_t), temporal="cur")  # \tilde{O}_t (query pre-normed)
        if self.direct_z_to_object:
            if z is None:
                raise ValueError("direct_z_to_object=True requires a latent action z.")
            # Reuse F_A's act_proj (no new parameters) so IM-LAM and this ablation match in param count.
            b = z.shape[0]
            z_tokens = (
                self.f_a.act_proj(z)
                .reshape(b, self.dim, self.f_a.side, self.f_a.side)
                .flatten(2)
                .transpose(1, 2)
            )
            query = query + z_tokens
        # Keys are pre-normed and temporal-tagged; values carry untagged, unnormalized agent content
        # (kept at IMPALA scale so the residual add below is like-for-like).
        a_t_n, a2_n = self.norm_ctx(a_t), self.norm_ctx(a2)
        key = torch.cat([self.embeddings.tag(a_t_n, "cur"), self.embeddings.tag(a2_n, a2_temporal)], dim=1)
        value = torch.cat([a_t, a2], dim=1)
        # no mask bias: keys are agent tokens by construction.
        cross = self.cross_attn(query, key, value, mask_bias=None, return_attn=return_attn)
        context, attn = cross if return_attn else (cross, None)
        h = o_t + context                       # attention residual on the untagged object content
        o_hat = h + self.ffn(self.norm_ffn(h))  # pre-norm position-wise FFN residual
        return (o_hat, attn) if return_attn else o_hat

    def _dilate_gate(self, occupancy: Tensor) -> Tensor:
        """Soft-dilate a flat occupancy ``(B, N)`` to a spatial gate ``(B, 1, H', W')``."""
        gate = occupancy.reshape(occupancy.shape[0], 1, self.side, self.side)
        for _ in range(self.dilate_iters):
            gate = self.dilate(gate)
        return gate

    def write_back(self, b_t: Tensor, a_hat: Tensor, o_hat: Tensor, w_agent: Tensor, w_object: Tensor) -> Tensor:
        """Mask-gated, zero-init residual write-back into the FDM bottleneck.

        ``B_hat_{t+1} = B_t + Dilate(W^A) . Proj_A(A_hat) + Dilate(W^O) . Proj_O(O_hat)``. Each entity's
        update is confined to its own (dilated) current-frame region, and the projections are
        zero-initialized, so at init ``B_hat == B_t`` exactly: the (untested, randomly-initialized)
        interaction machinery starts dormant and is learned as a residual, so it cannot disrupt the base
        encoder-decoder at init. NB this is identity at IM-LAM's *own* init - the encoder/decoder are
        freshly initialized, not a warm start from a trained MaskLAM, and since ``Proj_A`` sits on the
        only ``z_t`` path, at init the FDM is action-blind (predicts o_{t+1} from o_t alone). ``Proj`` is
        still a zero-conv (its gradient depends on its input, not its value), so it leaves zero on the
        first step and the ``z_t`` pathway opens from step 1. The dilation covers positions the entity
        moves into at ``t+1`` but not present at ``t`` (current-frame masks route; target-frame masks
        supervise).

        Args:
            b_t (Tensor): Bottleneck feature map ``(B, dim, H', W')`` (the residual stream).
            a_hat (Tensor): Predicted agent tokens ``(B, N, dim)``.
            o_hat (Tensor): Predicted object tokens ``(B, N, dim)``.
            w_agent (Tensor): Agent occupancy ``(B, N)`` in ``[0, 1]`` (current-frame).
            w_object (Tensor): Object occupancy ``(B, N)`` in ``[0, 1]``.

        Returns:
            Tensor: predicted bottleneck ``B_hat_{t+1}`` ``(B, dim, H', W')``.
        """
        delta_a = self._dilate_gate(w_agent) * self.proj_a(self._to_spatial(a_hat))
        delta_o = self._dilate_gate(w_object) * self.proj_o(self._to_spatial(o_hat))
        return b_t + delta_a + delta_o

    def forward(
        self,
        b_t: Tensor,
        w_agent: Tensor,
        w_object: Tensor,
        z: Tensor,
        agent_ctx_mode: str = "normal",
        return_attn: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Full interaction bottleneck: ``B_t -> B_hat_{t+1}`` via the directed agent->object pathway.

        Composes extraction, agent dynamics (``F_A``), directed object dynamics (``F_O``), and the gated
        write-back. Because the write-back is zero-initialized, this is an exact identity on ``B_t`` at init.

        Args:
            b_t (Tensor): Bottleneck feature map ``(B, dim, H', W')`` from the FDM encoder.
            w_agent (Tensor): Agent occupancy ``(B, N)`` in ``[0, 1]`` (pooled current-frame mask).
            w_object (Tensor): Object occupancy ``(B, N)`` in ``[0, 1]``.
            z (Tensor): Latent action ``(B, code_dim)``.
            agent_ctx_mode (str): Eval-time agent-path substitution (see `object_dynamics`).
            return_attn (bool): If True, also return the object->agent attention weights.

        Returns:
            Tensor ``(B, dim, H', W')``, or ``(b_hat, attn)`` when ``return_attn``.
        """
        tokens = self._to_tokens(b_t)
        a_t, o_t = self.extract(tokens, w_agent, w_object)
        a_hat = self.agent_dynamics(a_t, z)
        obj = self.object_dynamics(o_t, a_t, a_hat, z=z, agent_ctx_mode=agent_ctx_mode, return_attn=return_attn)
        o_hat, attn = obj if return_attn else (obj, None)
        b_hat = self.write_back(b_t, a_hat, o_hat, w_agent, w_object)
        return (b_hat, attn) if return_attn else b_hat
