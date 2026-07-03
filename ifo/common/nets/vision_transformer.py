"""Vision Transformer implementation based on DINOv3.

This module implements the Vision Transformer architecture as described in the DINOv3 paper:
DINOv3: Scaling Self-Supervised Vision Foundation Models" (https://arxiv.org/pdf/2508.10104)

DINOv3 is a self-supervised learning approach that produces high-quality dense visual features
without requiring manual annotations. It introduces several key innovations:
- Effective scaling strategies for both dataset and model size
- Gram anchoring to prevent dense feature map degradation during long training
- Post-hoc strategies for enhanced flexibility in resolution, model size, and text alignment

Repository: https://github.com/facebookresearch/dinov3

The models in this module provide versatile vision foundation models that achieve state-of-the-art
performance across a broad range of vision tasks without fine-tuning.
"""
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn

from ifo.common.layers.transformer import ImageRopePositionEmbedding, PatchEmbed, SelfAttentionBlock
from ifo.common.nets.transformer import dtype_dict, ffn_layer_dict, norm_layer_dict
from ifo.common.utils.dino import (
    Weights,
    _add_missing_keys,
    _load_safetensors_from_hub,
    _rename_huggingface_to_legacy_keys,
    init_weights_vit,
)
from ifo.common.utils.functions import named_apply


class DinoVisionTransformer(nn.Module):
    """DINO Vision Transformer (ViT) with RoPE positional embeddings.

    A Vision Transformer implementation based on DINOv3 with several key features:
    - Rotary Position Embeddings (RoPE) instead of learned positional embeddings
    - Flexible FFN architectures (MLP or SwiGLU variants)
    - Optional LayerScale for training stability
    - Support for storage/register tokens
    - Optional separate normalization for CLS and patch tokens
    - Mask token for masked image modeling
    """

    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        pos_embed_rope_base: Optional[float] = 100.0,
        pos_embed_rope_min_period: Optional[float] = None,
        pos_embed_rope_max_period: Optional[float] = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: Optional[float] = None,
        pos_embed_rope_jitter_coords: Optional[float] = None,
        pos_embed_rope_rescale_coords: Optional[float] = None,
        pos_embed_rope_dtype: Optional[str | torch.dtype] = None,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
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
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        **ignored_kwargs,
    ) -> None:
        """Initialize the DINO Vision Transformer.

        Args:
            img_size (int): Input image size (assumes square images). Defaults to 224.
            patch_size (int): Patch size for patch embedding. Defaults to 16.
            in_channels (int): Number of input image channels. Defaults to 3.
            pos_embed_rope_base (Optional[float]): Base for RoPE frequency sequence.
                Defaults to 100.0.
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
            embed_dim (int): Embedding dimension. Defaults to 768.
            depth (int): Number of transformer blocks. Defaults to 12.
            num_heads (int): Number of attention heads. Defaults to 12.
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
            untie_cls_and_patch_norms (bool): Whether to use separate norms for CLS and
                patch tokens. Defaults to False.
            untie_global_and_local_cls_norm (bool): Whether to use separate norms for
                global and local CLS tokens. Defaults to False.
            **ignored_kwargs: Additional keyword arguments that will be ignored with a warning.
        """
        super().__init__()
        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim))

        self.pos_embed_rope_dtype = pos_embed_rope_dtype
        if isinstance(self.pos_embed_rope_dtype, str):
            self.pos_embed_rope_dtype = dtype_dict[self.pos_embed_rope_dtype]

        self.rope_embed = ImageRopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=self.pos_embed_rope_dtype,
        )
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
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

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim))
        self.init_weights()

    def init_weights(self) -> None:
        """Initialize all weights in the model.

        Initializes:
        - RoPE position embeddings
        - CLS token with normal distribution (std=0.02)
        - Storage tokens with normal distribution (std=0.02) if present
        - Mask token with zeros
        - All other modules via named_apply with init_weights_vit
        """
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(self, x: Tensor, masks: Optional[Tensor] = None) -> Tuple[Tensor, Tuple[int, int]]:
        """Prepare input tokens with CLS token, storage (register) tokens, and optional masking.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).
            masks (Optional[Tensor]): Boolean mask tensor of shape (B, N) where True
                indicates positions to mask. Defaults to None.

        Returns:
            Tuple[Tensor, Tuple[int, int]]: A tuple containing:
                - Prepared token sequence of shape (B, 1 + n_storage + H*W, embed_dim)
                  where positions are [CLS, storage_tokens, patch_tokens]
                - Tuple (H, W) indicating patch grid dimensions
        """
        x = self.patch_embed(x)  # [B, C, H, W] -> [B, H', W', embed_dim]
        B, H, W, _ = x.shape
        x = x.flatten(1, 2) # [B, H', W', embed_dim] -> [B, H'*W', embed_dim] = [B, N, embed_dim]

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token  # Fix for DDP handle parameters that do not contribute to the loss.

        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(1, 1, cls_token.shape[-1], dtype=cls_token.dtype)

        x = torch.cat([cls_token.expand(B, -1, -1), storage_tokens.expand(B, -1, -1), x], dim=1)
        return x, (H, W)

    def forward_features_list(
        self, x_list: List[Tensor], masks_list: List[Optional[Tensor]]
    ) -> List[Dict[str, Tensor]]:
        """Forward pass for a list of inputs (e.g., multi-crop training).

        Processes multiple image crops simultaneously, useful for contrastive learning
        methods like DINO that use global and local crops.

        Args:
            x_list (List[Tensor]): List of input image tensors, each of shape (B, C, H, W).
            masks_list (List[Optional[Tensor]]): List of optional mask tensors corresponding
                to each input.

        Returns:
            List[Dict[str, Tensor]]: List of dictionaries, one per input, each containing:
                - "x_norm_clstoken": Normalized CLS token of shape (B, embed_dim)
                - "x_storage_tokens": Normalized storage tokens of shape (B, n_storage, embed_dim)
                - "x_norm_patchtokens": Normalized patch tokens of shape (B, N, embed_dim)
                - "x_prenorm": Pre-normalization features of shape (B, 1+n_storage+N, embed_dim)
                - "masks": The input masks
        """
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])  # type: ignore[misc]
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])  # type: ignore[misc]
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],  # Normalized CLS token (B, embed_dim)
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],  # Normalized storage tokens (B, n_storage, embed_dim)
                    "x_norm_patchtokens": x_norm_patch,  # Normalized patch tokens (B, N, embed_dim)
                    "x_prenorm": x,  # Pre-normalization features (B, 1+n_storage+N, embed_dim)
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

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int | Sequence[int] = 1) -> List[Tensor]:
        """Get intermediate layer outputs without chunking.

        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W).
            n (int | Sequence[int]): If int, returns the last n layers. If sequence,
                returns the specified layer indices. Defaults to 1.

        Returns:
            List[Tensor]: List of intermediate layer outputs, each of shape
                (B, 1+n_storage+N, embed_dim).
        """
        x, (H, W) = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence[int]] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[torch.Tensor | Tuple[torch.Tensor, ...], ...]:
        """Get intermediate layer features with flexible output options.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            n (Union[int, Sequence[int]]): If int, returns the last n layers. If sequence,
                returns the specified layer indices. Defaults to 1.
            reshape (bool): If True, reshape patch tokens to spatial grid (B, D, H, W).
                Defaults to False.
            return_class_token (bool): If True, include CLS tokens in output.
                Defaults to False.
            return_extra_tokens (bool): If True, include storage tokens in output.
                Defaults to False.
            norm (bool): If True, apply normalization to outputs. Defaults to True.

        Returns:
            Tuple[torch.Tensor | Tuple[torch.Tensor, ...], ...]: Tuple of outputs for each
                requested layer. Each element is:
                - Just patch tokens if return_class_token=False and return_extra_tokens=False
                - (patch_tokens, class_tokens) if return_class_token=True only
                - (patch_tokens, extra_tokens) if return_extra_tokens=True only
                - (patch_tokens, class_tokens, extra_tokens) if both are True
        """
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])  # type: ignore[misc]
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        else:  # return_class_token and return_extra_tokens
            return tuple(zip(outputs, class_tokens, extra_tokens))

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


def _make_dinov3_vit(
    *,
    img_size: int = 224,
    patch_size: int = 16,
    in_chans: int = 3,
    compact_arch_name: str = "vitb",
    pos_embed_rope_base: float = 100.0,
    pos_embed_rope_min_period: float | None = None,
    pos_embed_rope_max_period: float | None = None,
    pos_embed_rope_normalize_coords: str = "separate",
    pos_embed_rope_shift_coords: float | None = None,
    pos_embed_rope_jitter_coords: float | None = None,
    pos_embed_rope_rescale_coords: float | None = None,
    pos_embed_rope_dtype: Optional[str | torch.dtype] = None,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    ffn_ratio: float = 4.0,
    qkv_bias: bool = True,
    drop_path_rate: float = 0.0,
    layerscale_init: float | None = None,
    norm_layer: str = "layernorm",
    ffn_layer: str = "mlp",
    ffn_bias: bool = True,
    proj_bias: bool = True,
    n_storage_tokens: int = 0,
    mask_k_bias: bool = False,
    pretrained: bool = True,
    version: Optional[str] = None,
    weights: Union[Weights, str] = Weights.LVD1689M,
    hash: Optional[str] = None,
    check_hash: bool = True,
    **kwargs,
) -> DinoVisionTransformer:
    """Create a DINOv3 Vision Transformer model with specified configuration.

    Args:
        img_size (int): Input image size. Defaults to 224.
        patch_size (int): Patch size for tokenization. Defaults to 16.
        in_chans (int): Number of input channels. Defaults to 3.
        compact_arch_name (str): Compact architecture name. Defaults to "vitb".
        pos_embed_rope_base (float): Base value for RoPE positional embeddings. Defaults to 100.0.
        pos_embed_rope_min_period (float | None): Minimum period for RoPE. Defaults to None.
        pos_embed_rope_max_period (float | None): Maximum period for RoPE. Defaults to None.
        pos_embed_rope_normalize_coords (str): Coordinate normalization method. Defaults to "separate".
        pos_embed_rope_shift_coords (float | None): Coordinate shift for RoPE. Defaults to None.
        pos_embed_rope_jitter_coords (float | None): Coordinate jitter for RoPE. Defaults to None.
        pos_embed_rope_rescale_coords (float | None): Coordinate rescaling factor. Defaults to None.
        pos_embed_rope_dtype (Optional[str | torch.dtype]): Data type for RoPE computations. Defaults to None.
        If None, use a global default.
        embed_dim (int): Embedding dimension. Defaults to 768.
        depth (int): Number of transformer blocks. Defaults to 12.
        num_heads (int): Number of attention heads. Defaults to 12.
        ffn_ratio (float): Ratio for feedforward network hidden dimension. Defaults to 4.0.
        qkv_bias (bool): Whether to use bias in QKV projections. Defaults to True.
        drop_path_rate (float): Drop path rate for stochastic depth. Defaults to 0.0.
        layerscale_init (float | None): Initial value for layer scale. Defaults to None.
        norm_layer (str): Normalization layer type. Defaults to "layernorm".
        ffn_layer (str): Feedforward network layer type. Defaults to "mlp".
        ffn_bias (bool): Whether to use bias in FFN. Defaults to True.
        proj_bias (bool): Whether to use bias in projections. Defaults to True.
        n_storage_tokens (int): Number of storage tokens. Defaults to 0.
        mask_k_bias (bool): Whether to mask key bias. Defaults to False.
        pretrained (bool): Whether to load pretrained weights. Defaults to True.
        version (Optional[str]): Model version string. Defaults to None.
        weights (Union[Weights, str]): Pretrained weights to load. Defaults to Weights.LVD1689M.
        hash (Optional[str]): Hash for weight verification. Defaults to None.
        check_hash (bool): Whether to verify hash when loading weights. Defaults to True.
        **kwargs: Additional keyword arguments passed to DinoVisionTransformer.

    Returns:
        DinoVisionTransformer: The constructed DINOv3 ViT model.

    Raises:
        ValueError: If unsupported weights are specified.
    """
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        pos_embed_rope_base=pos_embed_rope_base,
        pos_embed_rope_min_period=pos_embed_rope_min_period,
        pos_embed_rope_max_period=pos_embed_rope_max_period,
        pos_embed_rope_normalize_coords=pos_embed_rope_normalize_coords,
        pos_embed_rope_shift_coords=pos_embed_rope_shift_coords,
        pos_embed_rope_jitter_coords=pos_embed_rope_jitter_coords,
        pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords,
        pos_embed_rope_dtype=pos_embed_rope_dtype,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        ffn_ratio=ffn_ratio,
        qkv_bias=qkv_bias,
        drop_path_rate=drop_path_rate,
        layerscale_init=layerscale_init,
        norm_layer=norm_layer,
        ffn_layer=ffn_layer,
        ffn_bias=ffn_bias,
        proj_bias=proj_bias,
        n_storage_tokens=n_storage_tokens,
        mask_k_bias=mask_k_bias,
    )
    vit_kwargs.update(**kwargs)
    model = DinoVisionTransformer(**vit_kwargs)
    if pretrained:
        assert patch_size == 16, "Only 16px patch size is supported for pretrained weights"
        if weights not in {Weights.LVD1689M, Weights.SAT493M}:
            raise ValueError(f"Unsupported weights for the backbone: {weights}")

        repo_id = f"facebook/dinov3-{compact_arch_name}-pretrain-{weights.value.lower()}"
        state_dict = _load_safetensors_from_hub(repo_id)
        state_dict = _rename_huggingface_to_legacy_keys(state_dict)
        state_dict = _add_missing_keys(state_dict, model)
        model.load_state_dict(state_dict, strict=True)
    return model


def dinov3_vits16(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = True,
    **kwargs,
) -> DinoVisionTransformer:
    """Create a DINOv3 ViT-S/16 model.

    Args:
        pretrained (bool): Whether to load pretrained weights. Defaults to True.
        weights (Union[Weights, str]): Pretrained weights to load. Defaults to Weights.LVD1689M.
        check_hash (bool): Whether to verify hash when loading weights. Defaults to True.
        **kwargs: Additional keyword arguments passed to _make_dinov3_vit.

    Returns:
        DinoVisionTransformer: The DINOv3 ViT-S/16 model.
    """
    if "hash" not in kwargs:
        kwargs["hash"] = "08c60483"
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32" if pretrained else None,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vits16",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vits16plus(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = True,
    **kwargs,
) -> DinoVisionTransformer:
    """Create a DINOv3 ViT-S/16+ model with SwiGLU activation.

    Args:
        pretrained (bool): Whether to load pretrained weights. Defaults to True.
        weights (Union[Weights, str]): Pretrained weights to load. Defaults to Weights.LVD1689M.
        check_hash (bool): Whether to verify hash when loading weights. Defaults to True.
        **kwargs: Additional keyword arguments passed to _make_dinov3_vit.

    Returns:
        DinoVisionTransformer: The DINOv3 ViT-S/16+ model.
    """
    if "hash" not in kwargs:
        kwargs["hash"] = "4057cbaa"
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32" if pretrained else None,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=6,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vits16plus",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vitb16(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = True,
    **kwargs,
) -> DinoVisionTransformer:
    """Create a DINOv3 ViT-B/16 model.

    Args:
        pretrained (bool): Whether to load pretrained weights. Defaults to True.
        weights (Union[Weights, str]): Pretrained weights to load. Defaults to Weights.LVD1689M.
        check_hash (bool): Whether to verify hash when loading weights. Defaults to True.
        **kwargs: Additional keyword arguments passed to _make_dinov3_vit.

    Returns:
        DinoVisionTransformer: The DINOv3 ViT-B/16 model.
    """
    if "hash" not in kwargs:
        kwargs["hash"] = "73cec8be"
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32" if pretrained else None,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitb16",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vitl16(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = True,
    **kwargs,
) -> DinoVisionTransformer:
    """Create a DINOv3 ViT-L/16 model.

    Args:
        pretrained (bool): Whether to load pretrained weights. Defaults to True.
        weights (Union[Weights, str]): Pretrained weights to load. Defaults to Weights.LVD1689M.
        check_hash (bool): Whether to verify hash when loading weights. Defaults to True.
        **kwargs: Additional keyword arguments passed to _make_dinov3_vit.

    Returns:
        DinoVisionTransformer: The DINOv3 ViT-L/16 model.

    Raises:
        ValueError: If unexpected weights specification is provided.
    """
    untie_global_and_local_cls_norm = False
    if weights == Weights.LVD1689M:
        if "hash" not in kwargs:
            kwargs["hash"] = "8aa4cbdd"
    elif weights == Weights.SAT493M:
        if "hash" not in kwargs:
            kwargs["hash"] = "eadcf0ff"
        untie_global_and_local_cls_norm = True
    elif type(weights) is str:
        import re

        pattern = r"-(.{8}).pth"
        matches = re.findall(pattern, weights)
        if len(matches) != 1:
            raise ValueError(f"Unexpected weights specification for the ViT-L backbone: {weights}")
        hash = matches[0]
        if hash == "eadcf0ff":
            untie_global_and_local_cls_norm = True
    kwargs["version"] = None
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32" if pretrained else None,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=untie_global_and_local_cls_norm,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vitl16",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vith16plus(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = True,
    **kwargs,
) -> DinoVisionTransformer:
    """Create a DINOv3 ViT-H/16+ model with SwiGLU activation.

    Args:
        pretrained (bool): Whether to load pretrained weights. Defaults to True.
        weights (Union[Weights, str]): Pretrained weights to load. Defaults to Weights.LVD1689M.
        check_hash (bool): Whether to verify hash when loading weights. Defaults to True.
        **kwargs: Additional keyword arguments passed to _make_dinov3_vit.

    Returns:
        DinoVisionTransformer: The DINOv3 ViT-H/16+ model.
    """
    if "hash" not in kwargs:
        kwargs["hash"] = "7c1da9a5"

    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32" if pretrained else None,
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
        n_storage_tokens=4,
        mask_k_bias=True,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vith16plus",
        check_hash=check_hash,
        **kwargs,
    )


def dinov3_vit7b16(
    *,
    pretrained: bool = True,
    weights: Union[Weights, str] = Weights.LVD1689M,
    check_hash: bool = True,
    **kwargs,
) -> DinoVisionTransformer:
    """Create a DINOv3 ViT-7B/16 model with SwiGLU activation.

    Args:
        pretrained (bool): Whether to load pretrained weights. Defaults to True.
        weights (Union[Weights, str]): Pretrained weights to load. Defaults to Weights.LVD1689M.
        check_hash (bool): Whether to verify hash when loading weights. Defaults to True.
        **kwargs: Additional keyword arguments passed to _make_dinov3_vit.

    Returns:
        DinoVisionTransformer: The DINOv3 ViT-7B/16 model.
    """
    if weights == Weights.LVD1689M:
        if "hash" not in kwargs:
            kwargs["hash"] = "a955f4ea"
    elif weights == Weights.SAT493M:
        if "hash" not in kwargs:
            kwargs["hash"] = "a6675841"
    kwargs["version"] = None
    untie_global_and_local_cls_norm = True
    return _make_dinov3_vit(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32" if pretrained else None,
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        qkv_bias=False,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="swiglu64",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        untie_global_and_local_cls_norm=untie_global_and_local_cls_norm,
        pretrained=pretrained,
        weights=weights,
        compact_arch_name="vit7b16",
        check_hash=check_hash,
        **kwargs,
    )
