from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ifo.common.nets.impala_cnn import ImpalaDecoderBlock, ImpalaEncoderBlock
from ifo.common.nets.interaction import InteractionModule, pool_mask_occupancy
from ifo.common.utils.functions import orthogonal_init


class InteractionWorldModel(nn.Module):
    """IM-LAM forward dynamics model: IMPALA encoder -> interaction bottleneck -> IMPALA decoder.

    Drop-in for :class:`ifo.common.nets.world_model.ImpalaWorldModel` (same constructor call from
    ``SLAPOIDM``: ``world_model((C, H, W), output_channels=c, condition_dim=code_dim)``), with the
    directed interaction module inserted at the bottleneck. Two structural differences from MaskLAM's FDM:

    - **No channel-wise ``z_t`` concat at the bottleneck.** MaskLAM projects ``z_t`` to the bottleneck and
      concatenates it, doubling the decoder's input width; IM-LAM injects ``z_t`` through the interaction
      module's ``F_A`` instead, so ``B_hat`` carries ``d`` channels and the decoder is built from ``d`` (not
      ``2d``). There is therefore no ``act_proj`` on this module - it lives inside ``F_A``.
    - **Action-blind at init.** The interaction module's zero-init write-back makes it an identity residual,
      so at init the FDM predicts ``o_{t+1}`` from ``o_t`` alone and the ``z_t`` pathway opens from step 1.

    The interaction module's ``dim`` and ``num_tokens`` are derived from the encoder's
    ``final_encoder_shape``, and its ``F_A`` residual depth is threaded from ``encoder_num_res_blocks`` so it
    matches the FDM (no residual-depth confound versus Foreground-MaskLAM).
    """

    def __init__(
        self,
        shape: Tuple[int, int, int],
        output_channels: int,
        condition_dim: int = 128,
        channel_multiplier: int = 6,
        encoder_channels: Sequence[int] = (16, 32, 32),
        encoder_num_res_blocks: int = 2,
        num_heads: int = 6,
        mlp_ratio: int = 4,
        dilate_iters: int = 1,
        use_orthogonal_init: bool = False,
        direct_z_to_object: bool = False,
    ) -> None:
        """Instantiate the IM-LAM FDM.

        Args:
            shape (Tuple[int, int, int]): Input shape ``(C, H, W)`` (``C = frame_stack*(c+2)`` for IM-LAM).
            output_channels (int): Number of output channels (the RGB channel count ``c``).
            condition_dim (int, optional): Latent-action dimension ``z_t`` (``128`` for DMW).
            channel_multiplier (int, optional): IMPALA channel multiplier.
            encoder_channels (Sequence[int], optional): Per-block IMPALA channel counts.
            encoder_num_res_blocks (int, optional): Residual blocks per IMPALA block; also threaded into ``F_A``.
            num_heads (int, optional): Attention heads in the interaction module. Must divide the bottleneck width.
            mlp_ratio (int, optional): Hidden-width multiplier for ``F_O``'s position-wise FFN.
            dilate_iters (int, optional): Times the write-back dilates the pooled masks (>=1; the proposal's
                "applied once or twice" - raise for a faster-moving object).
            use_orthogonal_init (bool, optional): Orthogonally initialize submodules, then re-zero the write-back.
            direct_z_to_object (bool, optional): Matched-ablation switch (feeds ``z_t`` to the object branch).
        """
        super().__init__()
        assert len(shape) == 3, "Shape must be (C, H, W)."

        # Encoder (identical construction to ImpalaWorldModel) -> B_t.
        encoder_layers = []
        for out_channels in encoder_channels:
            block = ImpalaEncoderBlock(shape[0], channel_multiplier * out_channels, encoder_num_res_blocks)
            shape = block.get_output_shape(*shape[1:])
            encoder_layers.append(block)
        self.encoder = nn.Sequential(*encoder_layers)
        self.final_encoder_shape = shape  # (C', H', W')
        c_bottleneck, h_prime, w_prime = shape
        if h_prime != w_prime:
            raise ValueError(f"interaction bottleneck must be square; encoder produced {h_prime}x{w_prime}.")
        self.side = h_prime

        # Interaction bottleneck. dim / num_tokens derived from the encoder geometry; num_res_blocks
        # threaded from encoder_num_res_blocks so F_A's residual depth cannot drift from the FDM.
        self.interaction = InteractionModule(
            dim=c_bottleneck,
            num_tokens=h_prime * w_prime,
            num_heads=num_heads,
            code_dim=condition_dim,
            mlp_ratio=mlp_ratio,
            dilate_iters=dilate_iters,
            direct_z_to_object=direct_z_to_object,
            num_res_blocks=encoder_num_res_blocks,
        )
        assert self.interaction.f_a.num_res_blocks == encoder_num_res_blocks, (
            "F_A residual depth must equal the FDM encoder_num_res_blocks (no depth confound)."
        )

        # Decoder built from the UN-doubled bottleneck width: z_t is not concatenated here (it enters via F_A).
        decoder_layers = []
        shape = self.final_encoder_shape
        for out_channels in encoder_channels[::-1]:
            block = ImpalaDecoderBlock(shape[0], channel_multiplier * out_channels, encoder_num_res_blocks)
            shape = block.get_output_shape(*shape[1:])
            decoder_layers.append(block)
        self.decoder = nn.Sequential(
            *decoder_layers,
            nn.GELU(),
            nn.Conv2d(encoder_channels[0] * channel_multiplier, output_channels, kernel_size=1),
        )

        if use_orthogonal_init:
            # orthogonal_init re-inits every Conv2d, including the write-back projections; re-zero them
            # afterward so the identity-at-init (graceful-degradation) guarantee survives.
            self.apply(orthogonal_init)
            self.interaction.zero_init_write_back()

    def forward(
        self,
        x: Tensor,
        condition: Tensor,
        agent_mask_t: Tensor,
        object_mask_t: Tensor,
        agent_ctx_mode: str = "normal",
        return_attn: bool = False,
    ):
        """Forward pass.

        Args:
            x (Tensor): Current frame stack ``(B, C, H, W)`` (RGB + agent + object mask channels).
            condition (Tensor): Latent action ``z_t`` ``(B, condition_dim)``.
            agent_mask_t (Tensor): Current-frame agent mask ``(B, 1, H, W)`` in ``{0, 1}``.
            object_mask_t (Tensor): Current-frame object mask ``(B, 1, H, W)`` in ``{0, 1}``.
            agent_ctx_mode (str): Eval-time agent-path substitution (see ``InteractionModule.object_dynamics``).
            return_attn (bool): If True, also return the object->agent attention weights.

        Returns:
            Predicted next observation ``(B, output_channels, H, W)`` in ``[-0.5, 0.5]``, or ``(y, attn)``
            when ``return_attn``.
        """
        b_t = self.encoder(x)
        w_agent = pool_mask_occupancy(agent_mask_t, self.side)
        w_object = pool_mask_occupancy(object_mask_t, self.side)
        out = self.interaction(
            b_t, w_agent, w_object, condition, agent_ctx_mode=agent_ctx_mode, return_attn=return_attn
        )
        b_hat, attn = out if return_attn else (out, None)
        y = F.tanh(self.decoder(b_hat)) / 2  # normalize to [-0.5, 0.5], matching ImpalaWorldModel
        return (y, attn) if return_attn else y
