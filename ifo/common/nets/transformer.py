"""Transformer implementation for 1D sequences based on DINOv3 architecture.

This module provides a baseline transformer implementation adapted from the DINOv3
Vision Transformer architecture for processing 1D input sequences. It includes:
- Flexible FFN architectures (MLP or SwiGLU variants)
- Optional LayerScale for training stability
- Rotary Position Embeddings (RoPE) support
- Causal and non-causal self-attention blocks

The architecture follows the design principles from DINOv3 but is adapted for
general sequence modeling tasks rather than vision-specific applications.

Reference: DINOv3: Scaling Self-Supervised Vision Foundation Models
(https://arxiv.org/pdf/2508.10104)
"""


from functools import partial
from typing import Dict, List, Literal, Optional, Tuple

import torch
import torch.nn.init
from torch import Tensor, nn

from ifo.common.layers.transformer import (
    MLP,
    CausalSelfAttentionBlock,
    ImageRopePositionEmbedding,
    LayerScale,
    RopePositionEmbedding,
    SelfAttentionBlock,
    SwiGLUFFN,
)
from ifo.common.utils.functions import named_apply

ffn_layer_dict = {
    "mlp": MLP,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": lambda dim: nn.RMSNorm([dim], eps=1e-5),
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_transformer(module: nn.Module, name: str = "") -> None:
    """Initialize weights for Transformer modules.

    Applies appropriate initialization strategies for different module types:
    - Linear layers: truncated normal with std=0.02
    - LayerNorm/RMSNorm: calls reset_parameters()
    - LayerScale: calls reset_parameters()

    For Linear layers with bias_mask (used in masked attention), sets the middle
    third of the mask to 0 (corresponding to key biases).

    Args:
        module (nn.Module): Module to initialize.
        name (str): Name of the module (unused, kept for compatibility). Defaults to "".
    """
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)  # type: ignore[union-attr]
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)  # type: ignore[index]
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, nn.RMSNorm):
        module.reset_parameters()


class Transformer(nn.Module):
    """(Causal)Self-Attention Transformer with RoPE positional embeddings.

    A (Causal) Self-Attention Transformer implementation with several key features:
    - Rotary Position Embeddings (RoPE) instead of learned positional embeddings
    - Flexible FFN architectures (MLP or SwiGLU variants)
    - Support for storage/register tokens
    - Mask token for masked input tokens
    - (Causal) Self-Attention blocks
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: Optional[float] = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        pos_embed_rope_base: float = 10000.0,
        pos_embed_rope_dtype: Optional[str | torch.dtype] = None,
        is_causal: bool = False,
    ) -> None:
        """Initialize the Transformer.

        Args:
            input_dim (int): Input dimension of the tokens before embedding.
            embed_dim (int): Embedding dimension. Defaults to 512.
            depth (int): Number of transformer blocks. Defaults to 12.
            num_heads (int): Number of attention heads. Defaults to 8.
            ffn_ratio (float): Ratio of FFN hidden dimension to embedding dimension.
                Defaults to 4.0.
            qkv_bias (bool): Whether to use bias in QKV projection. Defaults to True.
            drop_path_rate (float): Stochastic depth rate. Defaults to 0.0.
            layerscale_init (Optional[float]): Initial value for LayerScale. If None,
                LayerScale is not used. Defaults to None.
            norm_layer (str): Normalization layer type ("layernorm", "layernormbf16", or "rmsnorm").
                Defaults to "layernorm".
            ffn_layer (str): FFN layer type ("mlp", "swiglu", "swiglu32", etc.).
                Defaults to "mlp".
            ffn_bias (bool): Whether to use bias in FFN layers. Defaults to True.
            proj_bias (bool): Whether to use bias in attention projection. Defaults to True.
            n_storage_tokens (int): Number of storage/register tokens. Defaults to 0.
            mask_k_bias (bool): Whether to mask key bias in attention. Defaults to False.
            pos_embed_rope_base (float): Base for RoPE frequency sequence. Defaults to 10000.0.
            pos_embed_rope_dtype (Optional[str | torch.dtype]): Data type for RoPE buffers ("fp32", "fp16", or "bf16").
                If None, use a global default.
            is_causal (bool): Whether to use causal self-attention. Defaults to False.
        """
        super().__init__()
        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.n_storage_tokens = n_storage_tokens
        self.is_causal = is_causal

        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim))

        self.pos_embed_rope_dtype = pos_embed_rope_dtype
        if isinstance(self.pos_embed_rope_dtype, str):
            self.pos_embed_rope_dtype = dtype_dict[self.pos_embed_rope_dtype]

        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            dtype=self.pos_embed_rope_dtype,
        )

        self.token_embed = nn.Linear(input_dim, embed_dim)

        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        block_cls = CausalSelfAttentionBlock if is_causal else SelfAttentionBlock
        blocks_list = [
            block_cls(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
            )
            for i in range(depth)
        ]

        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything.
        self.norm = norm_layer_cls(embed_dim)
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim))
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize all weights in the model.

        Initializes:
        - Storage tokens with normal distribution (std=0.02) if present
        - Mask token with zeros
        - All other modules via named_apply with init_weights_transformer
        """
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_transformer, self)

    def prepare_tokens_with_masks(self, x: Tensor, masks: Optional[Tensor] = None) -> Tuple[Tensor, int]:
        """Prepare input tokens with storage (register) tokens, and optional masking.

        Args:
            x (Tensor): Input tensor of shape (B, N, input_dim).
            masks (Optional[Tensor]): Boolean mask tensor of shape (B, N) where True
                indicates positions to mask. Defaults to None.

        Returns:
            Tuple[Tensor, int]: A tuple containing:
                - Prepared token sequence of shape (B, n_storage + N, embed_dim)
                where positions are [storage_tokens, tokens]
                - Number of tokens N
        """
        x = self.token_embed(x)  # [B, N, input_dim] -> [B, N, embed_dim]
        B, N, C = x.shape

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
        else:
            x = x + 0 * self.mask_token  # Fix for DDP handle parameters that do not contribute to the loss.

        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(1, 1, C, dtype=x.dtype)

        x = torch.cat([storage_tokens.expand(B, -1, -1), x], dim=1)
        return x, N

    def forward_features_list(
        self, x_list: List[Tensor], masks_list: List[Optional[Tensor]]
    ) -> List[Dict[str, Tensor]]:
        """Forward pass for a list of inputs.

        Args:
            x_list (List[Tensor]): List of input tensors, each of shape (B, N, embed_dim).
            masks_list (List[Optional[Tensor]]): List of optional mask tensors corresponding
                to each input.

        Returns:
            List[Dict[str, Tensor]]: List of dictionaries, one per input, each containing:
                - "x_storage_tokens": Normalized storage tokens of shape (B, n_storage, embed_dim)
                - "x_norm_patchtokens": Normalized patch tokens of shape (B, N, embed_dim)
                - "x_prenorm": Pre-normalization features of shape (B, n_storage+N, embed_dim)
                - "masks": The input masks
        """
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, L = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(L)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(L=L) for L in rope]
            else:
                rope_sincos = [None for _ in x]
            x = blk(x, rope_sincos)
        all_x = x
        output = []
        n_storage_tokens = self.n_storage_tokens if self.n_storage_tokens > 0 else 1
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            x_norm = self.norm(x)
            x_norm_storage_tokens = x_norm[:, : n_storage_tokens]
            x_norm_tokens = x_norm[:, n_storage_tokens :]
            output.append(
                {
                    "x_storage_tokens": x_norm_storage_tokens,  # Normalized storage tokens (B, n_storage, embed_dim)
                    "x_norm_tokens": x_norm_tokens,  # Normalized tokens (B, N, embed_dim)
                    "x_prenorm": x,  # Pre-normalization features (B, n_storage+N, embed_dim)
                    "masks": masks,  # Input masks
                }
            )
        return output

    def forward_features(
        self, x: Tensor | List[Tensor], masks: Optional[Tensor | List[Optional[Tensor]]] = None
    ) -> Dict[str, Tensor] | List[Dict[str, Tensor]]:
        """Forward pass through the transformer backbone.

        Args:
            x (Tensor | List[Tensor]): Input image(s). Single tensor of shape (B, C, H, W)
                or list of tensors for multi-crop.
            masks (Optional[Tensor | List[Optional[Tensor]]]): Optional mask(s) corresponding
                to the input(s). Defaults to None.

        Returns:
            Dict[str, Tensor] | List[Dict[str, Tensor]]: Output features. Single dictionary
                if input is a single tensor, list of dictionaries if input is a list.
        """
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]  # type: ignore[list-item]
        else:
            # Type ignore needed because masks could be Tensor or List, but we know it matches x type
            return self.forward_features_list(x, masks)  # type: ignore[arg-type]

    def forward(self, *args, **kwargs) -> List[Dict[str, Tensor]] | Dict[str, Tensor]:
        """Forward pass through the entire model.

        Args:
            *args: Positional arguments passed to forward_features.
            **kwargs: Keyword arguments passed to forward_features.

        Returns:
            List[Dict[str, Tensor]] | Dict[str, Tensor]: Output features. Single dictionary
                if input is a single tensor, list of dictionaries if input is a list.
        """
        return self.forward_features(*args, **kwargs)


class AxialTransformer(nn.Module):
    """(Causal)Self-Attention Space-Time Transformer.

    A (Causal) Self-Attention Space-Time Transformer implementation with several key features:
    - Rotary Position Embeddings (RoPE) for temporal dimension
    - Learned positional embeddings for spatial dimensions
    - Flexible FFN architectures (MLP or SwiGLU variants)
    - Support for storage/register tokens
    - Mask token for masked input tokens
    - (Causal) Self-Attention blocks for temporal dimension
    - Self-Attention blocks for spatial dimensions
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 512,
        depth: int = 12,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: Optional[float] = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        time_pos_embed_rope_base: float = 1000.0,
        space_pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: Optional[float] = None,
        pos_embed_rope_max_period: Optional[float] = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: Optional[float] = None,
        pos_embed_rope_jitter_coords: Optional[float] = None,
        pos_embed_rope_rescale_coords: Optional[float] = None,
        pos_embed_rope_dtype: Optional[str | torch.dtype] = None,
        is_causal: bool = False,
        time_block_every_n_blocks: int = 4,
    ) -> None:
        """Initialize the Transformer.

        Args:
            input_dim (int): Input dimension of the tokens before embedding.
            embed_dim (int): Embedding dimension. Defaults to 512.
            depth (int): Number of transformer blocks. Defaults to 12.
            num_heads (int): Number of attention heads. Defaults to 8.
            ffn_ratio (float): Ratio of FFN hidden dimension to embedding dimension.
                Defaults to 4.0.
            qkv_bias (bool): Whether to use bias in QKV projection. Defaults to True.
            drop_path_rate (float): Stochastic depth rate. Defaults to 0.0.
            layerscale_init (Optional[float]): Initial value for LayerScale. If None,
                LayerScale is not used. Defaults to None.
            norm_layer (str): Normalization layer type ("layernorm", "layernormbf16", or "rmsnorm").
                Defaults to "layernorm".
            ffn_layer (str): FFN layer type ("mlp", "swiglu", "swiglu32", etc.).
                Defaults to "mlp".
            ffn_bias (bool): Whether to use bias in FFN layers. Defaults to True.
            proj_bias (bool): Whether to use bias in attention projection. Defaults to True.
            n_storage_tokens (int): Number of storage/register tokens. Defaults to 0.
            mask_k_bias (bool): Whether to mask key bias in attention. Defaults to False.
            time_pos_embed_rope_base (float): Base for RoPE frequency sequence. Defaults to 10000.0.
            space_pos_embed_rope_base (float): Base for RoPE frequency sequence. Defaults to 10000.0.
            pos_embed_rope_min_period (Optional[float]): Minimum period for RoPE.
                Defaults to None.
            pos_embed_rope_max_period (Optional[float]): Maximum period for RoPE.
                Defaults to None.
            pos_embed_rope_normalize_coords (Literal["min", "max", "separate"]):
                Coordinate normalization strategy for RoPE. Defaults to "separate".
            pos_embed_rope_shift_coords (Optional[float]): Random shift augmentation
                for RoPE during training. Defaults to None.
            pos_embed_rope_jitter_coords (Optional[float]): Random jitter augmentation
                for RoPE during training. Defaults to None.
            pos_embed_rope_rescale_coords (Optional[float]): Random rescale augmentation
                for RoPE during training. Defaults to None.
            pos_embed_rope_dtype (Optional[str | torch.dtype]): Data type for RoPE buffers ("fp32", "fp16", or "bf16").
                If None, use a global default.
            is_causal (bool): Whether to use causal self-attention. Defaults to False.
            time_block_every_n_blocks (int): Every n blocks, a temporal self-attention block is used. Defaults to 4.
        """
        super().__init__()
        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.n_storage_tokens = n_storage_tokens
        self.is_causal = is_causal

        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, 1, n_storage_tokens, embed_dim))

        self.pos_embed_rope_dtype = pos_embed_rope_dtype
        if isinstance(self.pos_embed_rope_dtype, str):
            self.pos_embed_rope_dtype = dtype_dict[self.pos_embed_rope_dtype]

        self.rope_time_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=time_pos_embed_rope_base,
            dtype=self.pos_embed_rope_dtype,
        )

        self.rope_space_embed = ImageRopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=space_pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=self.pos_embed_rope_dtype,
        )

        self.token_embed = nn.Linear(input_dim, embed_dim)

        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth

        assert depth % time_block_every_n_blocks == 0, "depth must be divisible by time_block_every_n_blocks"
        self.time_block_every_n_blocks = time_block_every_n_blocks

        time_block_cls = CausalSelfAttentionBlock if is_causal else SelfAttentionBlock
        space_block_cls = SelfAttentionBlock

        blocks_list = []
        for i in range(depth):
            block_cls = space_block_cls
            # Insert a time block every time_block_every_n_blocks blocks, but not at the beginning.
            if i % time_block_every_n_blocks == 0 and i != 0:
                block_cls = time_block_cls

            blocks_list.append(block_cls(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
            ))

        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything.
        self.norm = norm_layer_cls(embed_dim)
        self.mask_token = nn.Parameter(torch.empty(1, 1, 1, embed_dim))
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize all weights in the model.

        Initializes:
        - Storage tokens with normal distribution (std=0.02) if present
        - Mask token with zeros
        - All other modules via named_apply with init_weights_transformer
        """
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_transformer, self)

    def prepare_tokens_with_masks(
        self, x: Tensor, masks: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tuple[int, int, int]]:
        """Prepare input tokens with storage (register) tokens, and optional masking.

        Args:
            x (Tensor): Input tensor of shape (B, T, H, W, input_dim).
            masks (Optional[Tensor]): Boolean mask tensor of shape (B, T, N) where True
                indicates positions to mask. Defaults to None.

        Returns:
            Tuple[Tensor, int]: A tuple containing:
                - Prepared token sequence of shape (B, T, n_storage + N, embed_dim)
                where positions are [storage_tokens, tokens]
                - Tuple (T, H, W) indicating patch grid dimensions and temporal dimension
        """
        assert x.dim() == 5, "Input must be a 5D tensor (B, T, H, W, input_dim)"
        B, T, H, W, C = x.shape
        x = self.token_embed(x).view(B, T, -1, self.embed_dim)  # [B, T, H, W, input_dim] -> [B, T, N, embed_dim]

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype), x)
        else:
            x = x + 0 * self.mask_token  # Fix for DDP handle parameters that do not contribute to the loss.

        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(1, 1, 1, self.embed_dim, dtype=x.dtype)

        x = torch.cat([storage_tokens.expand(B, T, -1, -1), x], dim=2)
        return x, (T, H, W)

    def forward_features_list(
        self, x_list: List[Tensor], masks_list: List[Optional[Tensor]]
    ) -> List[Dict[str, Tensor]]:
        """Forward pass for a list of inputs.

        Args:
            x_list (List[Tensor]): List of input tensors, each of shape (B, T, H, W, input_dim).
            masks_list (List[Optional[Tensor]]): List of optional mask tensors corresponding
                to each input.

        Returns:
            List[Dict[str, Tensor]]: List of dictionaries, one per input, each containing:
                - "x_storage_tokens": Normalized storage tokens of shape (B, T, n_storage, embed_dim)
                - "x_norm_patchtokens": Normalized patch tokens of shape (B, T, H, W, embed_dim)
                - "x_prenorm": Pre-normalization features of shape (B, n_storage+H*W, embed_dim)
                - "masks": The input masks
        """
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, thw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(thw_tuple)

        for i, blk in enumerate(self.blocks):
            is_time_block = i % self.time_block_every_n_blocks == 0 and i != 0
            if self.rope_time_embed is not None:
                if is_time_block:
                    rope_sincos = [self.rope_time_embed(L=T) for T, _, _ in rope]
                else:
                    rope_sincos = [self.rope_space_embed(H=H, W=W) for _, H, W in rope]
            else:
                rope_sincos = [None for _ in x]

            if is_time_block:
                x = [x_i.transpose(-2, -3) for x_i in x]  # Switch from spatial to temporal dimension
                x = blk(x, rope_sincos)
                x = [x_i.transpose(-2, -3) for x_i in x]  # Switch back to spatial dimension
            else:
                x = blk(x, rope_sincos)

        all_x = x
        output = []

        n_storage_tokens = self.n_storage_tokens if self.n_storage_tokens > 0 else 1
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            T, H, W = rope[idx]
            B, _, _, C = x.shape
            x_norm = self.norm(x)
            x_norm_storage_tokens = x_norm[:, :, : n_storage_tokens]
            x_norm_tokens = x_norm[:, :, n_storage_tokens :]
            x_norm_tokens = x_norm_tokens.view(B, T, H, W, C)
            output.append(
                {
                    "x_storage_tokens": x_norm_storage_tokens,  # Normalized storage tokens (B, T, n_storage, embed_dim)
                    "x_norm_tokens": x_norm_tokens,  # Normalized tokens (B, T, H*W, embed_dim)
                    "x_prenorm": x,  # Pre-normalization features (B, T, n_storage+N, embed_dim)
                    "masks": masks,  # Input masks
                }
            )
        return output

    def forward_features(
        self, x: Tensor | List[Tensor], masks: Optional[Tensor | List[Optional[Tensor]]] = None
    ) -> Dict[str, Tensor] | List[Dict[str, Tensor]]:
        """Forward pass through the transformer backbone.

        Args:
            x (Tensor | List[Tensor]): Input image(s). Single tensor of shape (B, T, H, W, input_dim)
                or list of tensors for multi-crop.
            masks (Optional[Tensor | List[Optional[Tensor]]]): Optional mask(s) corresponding
                to the input(s). Defaults to None.

        Returns:
            Dict[str, Tensor] | List[Dict[str, Tensor]]: Output features. Single dictionary
                if input is a single tensor, list of dictionaries if input is a list.
        """
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]  # type: ignore[list-item]
        else:
            # Type ignore needed because masks could be Tensor or List, but we know it matches x type
            return self.forward_features_list(x, masks)  # type: ignore[arg-type]

    def forward(self, *args, **kwargs) -> List[Dict[str, Tensor]] | Dict[str, Tensor]:
        """Forward pass through the entire model.

        Args:
            *args: Positional arguments passed to forward_features.
            **kwargs: Keyword arguments passed to forward_features.

        Returns:
            List[Dict[str, Tensor]] | Dict[str, Tensor]: Output features. Single dictionary
                if input is a single tensor, list of dictionaries if input is a list.
        """
        return self.forward_features(*args, **kwargs)


# =============================================================================
# Model Size Factory Functions
# =============================================================================
# These factory functions create Transformer models with predefined configurations
# matching standard model sizes (Small, Base, Large, Huge, 7B).
# Naming convention follows DINOv3 ViT models:
# - "s" = Small (384 dim, 12 layers, 6 heads)
# - "splus" = Small+ with SwiGLU FFN
# - "b" = Base (768 dim, 12 layers, 12 heads)
# - "l" = Large (1024 dim, 24 layers, 16 heads)
# - "lplus" = Large+ with SwiGLU FFN
# - "hplus" = Huge+ with SwiGLU FFN (1280 dim, 32 layers, 20 heads)
# - "7b" = 7B scale (4096 dim, 40 layers, 32 heads)
# =============================================================================


def transformer_s(
    input_dim: int,
    *,
    n_storage_tokens: int = 4,
    is_causal: bool = False,
    **kwargs,
) -> Transformer:
    """Create a Small Transformer model.

    Configuration matches DINOv3 ViT-S architecture (without image-specific components):
    - embed_dim: 384
    - depth: 12
    - num_heads: 6
    - ffn_ratio: 4.0
    - ffn_layer: MLP
    - n_storage_tokens: 4

    Args:
        input_dim (int): Input dimension of the tokens before embedding.
        n_storage_tokens (int): Number of storage/register tokens. Defaults to 4.
        is_causal (bool): Whether to use causal self-attention. Defaults to False.
        **kwargs: Additional keyword arguments passed to Transformer.

    Returns:
        Transformer: A Small Transformer model.

    Example:
        >>> model = transformer_s(input_dim=64)
        >>> x = torch.randn(4, 16, 64)
        >>> output = model(x)
        >>> output["x_norm_tokens"].shape
        torch.Size([4, 16, 384])
    """
    return Transformer(
        input_dim=input_dim,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=n_storage_tokens,
        mask_k_bias=True,
        is_causal=is_causal,
        **kwargs,
    )


def transformer_splus(
    input_dim: int,
    *,
    n_storage_tokens: int = 4,
    is_causal: bool = False,
    **kwargs,
) -> Transformer:
    """Create a Small+ Transformer model with SwiGLU activation.

    Configuration matches DINOv3 ViT-S/16+ architecture (without image-specific components):
    - embed_dim: 384
    - depth: 12
    - num_heads: 6
    - ffn_ratio: 6.0
    - ffn_layer: SwiGLU
    - n_storage_tokens: 4

    The "plus" variant uses SwiGLU instead of MLP and a larger FFN ratio (6 vs 4).

    Args:
        input_dim (int): Input dimension of the tokens before embedding.
        n_storage_tokens (int): Number of storage/register tokens. Defaults to 4.
        is_causal (bool): Whether to use causal self-attention. Defaults to False.
        **kwargs: Additional keyword arguments passed to Transformer.

    Returns:
        Transformer: A Small+ Transformer model.

    Example:
        >>> model = transformer_splus(input_dim=64)
        >>> x = torch.randn(4, 16, 64)
        >>> output = model(x)
        >>> output["x_norm_tokens"].shape
        torch.Size([4, 16, 384])
    """
    return Transformer(
        input_dim=input_dim,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=6.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=n_storage_tokens,
        mask_k_bias=True,
        is_causal=is_causal,
        **kwargs,
    )


def transformer_b(
    input_dim: int,
    *,
    n_storage_tokens: int = 4,
    is_causal: bool = False,
    **kwargs,
) -> Transformer:
    """Create a Base Transformer model.

    Configuration matches DINOv3 ViT-B architecture (without image-specific components):
    - embed_dim: 768
    - depth: 12
    - num_heads: 12
    - ffn_ratio: 4.0
    - ffn_layer: MLP
    - n_storage_tokens: 4

    Args:
        input_dim (int): Input dimension of the tokens before embedding.
        n_storage_tokens (int): Number of storage/register tokens. Defaults to 4.
        is_causal (bool): Whether to use causal self-attention. Defaults to False.
        **kwargs: Additional keyword arguments passed to Transformer.

    Returns:
        Transformer: A Base Transformer model.

    Example:
        >>> model = transformer_b(input_dim=64)
        >>> x = torch.randn(4, 16, 64)
        >>> output = model(x)
        >>> output["x_norm_tokens"].shape
        torch.Size([4, 16, 768])
    """
    return Transformer(
        input_dim=input_dim,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=n_storage_tokens,
        mask_k_bias=True,
        is_causal=is_causal,
        **kwargs,
    )


def transformer_l(
    input_dim: int,
    *,
    n_storage_tokens: int = 4,
    is_causal: bool = False,
    **kwargs,
) -> Transformer:
    """Create a Large Transformer model.

    Configuration matches DINOv3 ViT-L architecture (without image-specific components):
    - embed_dim: 1024
    - depth: 24
    - num_heads: 16
    - ffn_ratio: 4.0
    - ffn_layer: MLP
    - n_storage_tokens: 4

    Args:
        input_dim (int): Input dimension of the tokens before embedding.
        n_storage_tokens (int): Number of storage/register tokens. Defaults to 4.
        is_causal (bool): Whether to use causal self-attention. Defaults to False.
        **kwargs: Additional keyword arguments passed to Transformer.

    Returns:
        Transformer: A Large Transformer model.

    Example:
        >>> model = transformer_l(input_dim=64)
        >>> x = torch.randn(4, 16, 64)
        >>> output = model(x)
        >>> output["x_norm_tokens"].shape
        torch.Size([4, 16, 1024])
    """
    return Transformer(
        input_dim=input_dim,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=n_storage_tokens,
        mask_k_bias=True,
        is_causal=is_causal,
        **kwargs,
    )


def transformer_lplus(
    input_dim: int,
    *,
    n_storage_tokens: int = 4,
    is_causal: bool = False,
    **kwargs,
) -> Transformer:
    """Create a Large+ Transformer model with SwiGLU activation.

    Configuration matches DINOv3 ViT-L/16+ architecture (without image-specific components):
    - embed_dim: 1024
    - depth: 24
    - num_heads: 16
    - ffn_ratio: 6.0
    - ffn_layer: SwiGLU
    - n_storage_tokens: 4

    The "plus" variant uses SwiGLU instead of MLP and a larger FFN ratio (6 vs 4).

    Args:
        input_dim (int): Input dimension of the tokens before embedding.
        n_storage_tokens (int): Number of storage/register tokens. Defaults to 4.
        is_causal (bool): Whether to use causal self-attention. Defaults to False.
        **kwargs: Additional keyword arguments passed to Transformer.

    Returns:
        Transformer: A Large+ Transformer model.

    Example:
        >>> model = transformer_lplus(input_dim=64)
        >>> x = torch.randn(4, 16, 64)
        >>> output = model(x)
        >>> output["x_norm_tokens"].shape
        torch.Size([4, 16, 1024])
    """
    return Transformer(
        input_dim=input_dim,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=6.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=n_storage_tokens,
        mask_k_bias=True,
        is_causal=is_causal,
        **kwargs,
    )


def transformer_hplus(
    input_dim: int,
    *,
    n_storage_tokens: int = 4,
    is_causal: bool = False,
    **kwargs,
) -> Transformer:
    """Create a Huge+ Transformer model with SwiGLU activation.

    Configuration matches DINOv3 ViT-H/16+ architecture (without image-specific components):
    - embed_dim: 1280
    - depth: 32
    - num_heads: 20
    - ffn_ratio: 6.0
    - ffn_layer: SwiGLU
    - n_storage_tokens: 4

    Args:
        input_dim (int): Input dimension of the tokens before embedding.
        n_storage_tokens (int): Number of storage/register tokens. Defaults to 4.
        is_causal (bool): Whether to use causal self-attention. Defaults to False.
        **kwargs: Additional keyword arguments passed to Transformer.

    Returns:
        Transformer: A Huge+ Transformer model.

    Example:
        >>> model = transformer_hplus(input_dim=64)
        >>> x = torch.randn(4, 16, 64)
        >>> output = model(x)
        >>> output["x_norm_tokens"].shape
        torch.Size([4, 16, 1280])
    """
    return Transformer(
        input_dim=input_dim,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=6.0,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=n_storage_tokens,
        mask_k_bias=True,
        is_causal=is_causal,
        **kwargs,
    )


def transformer_7b(
    input_dim: int,
    *,
    n_storage_tokens: int = 4,
    is_causal: bool = False,
    **kwargs,
) -> Transformer:
    """Create a 7B-scale Transformer model with SwiGLU activation.

    Configuration matches DINOv3 ViT-7B architecture (without image-specific components):
    - embed_dim: 4096
    - depth: 40
    - num_heads: 32
    - ffn_ratio: 3.0
    - ffn_layer: SwiGLU with 64-byte alignment
    - n_storage_tokens: 4

    Note: This is a very large model (~7B parameters) and requires significant
    GPU memory to train and run inference.

    Args:
        input_dim (int): Input dimension of the tokens before embedding.
        n_storage_tokens (int): Number of storage/register tokens. Defaults to 4.
        is_causal (bool): Whether to use causal self-attention. Defaults to False.
        **kwargs: Additional keyword arguments passed to Transformer.

    Returns:
        Transformer: A 7B-scale Transformer model.

    Example:
        >>> model = transformer_7b(input_dim=64)
        >>> x = torch.randn(1, 16, 64)  # Small batch due to model size
        >>> output = model(x)
        >>> output["x_norm_tokens"].shape
        torch.Size([1, 16, 4096])
    """
    return Transformer(
        input_dim=input_dim,
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3.0,
        qkv_bias=False,  # 7B model doesn't use qkv bias
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu64",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=n_storage_tokens,
        mask_k_bias=True,
        is_causal=is_causal,
        **kwargs,
    )
