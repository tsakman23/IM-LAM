from functools import partial
from typing import Callable, List, Optional

import torch
from torch import Tensor, nn

from ifo.common.layers.transformer.attention import BlockCausalSelfAttention, CausalSelfAttention, SelfAttention
from ifo.common.layers.transformer.ffn import MLP
from ifo.common.layers.transformer.scale import LayerScale
from ifo.common.utils.functions import cat_keep_shapes, uncat_with_shapes


class SelfAttentionBlock(nn.Module):
    """Self-attention Transformer block with optional sample dropping.

    Implements a standard Transformer block with self-attention and feed-forward layers,
    optionally with LayerScale and sample dropping for efficiency during training.
    Supports both single tensor and list of tensors as input.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = MLP,
        mask_k_bias: bool = False,
    ) -> None:
        """Initialize the SelfAttentionBlock.

        Args:
            dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            ffn_ratio (float): Ratio of FFN hidden dimension to embedding dimension.
                Defaults to 4.0.
            qkv_bias (bool): Whether to use bias in QKV projection. Defaults to False.
            proj_bias (bool): Whether to use bias in attention output projection.
                Defaults to True.
            ffn_bias (bool): Whether to use bias in FFN layers. Defaults to True.
            drop (float): Dropout probability for FFN. Defaults to 0.0.
            attn_drop (float): Dropout probability for attention. Defaults to 0.0.
            init_values (Optional[float]): Initial value for LayerScale. If None,
                LayerScale is not used. Defaults to None.
            drop_path (float): Sample drop ratio for stochastic depth. Defaults to 0.0.
            act_layer (Callable[..., nn.Module]): Activation function class.
                Defaults to nn.GELU.
            norm_layer (Callable[..., nn.Module]): Normalization layer class.
                Defaults to nn.LayerNorm.
            attn_class (Callable[..., nn.Module]): Attention class to use.
                Defaults to SelfAttention.
            ffn_layer (Callable[..., nn.Module]): FFN class to use. Defaults to MLP.
            mask_k_bias (bool): Whether to mask key bias in attention. Defaults to False.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()

        self.sample_drop_ratio = drop_path

    @staticmethod
    def _maybe_index_rope(
        rope: Optional[tuple[Tensor, Tensor]], indices: Tensor
    ) -> Optional[tuple[Tensor, Tensor]]:
        """Index into RoPE embeddings if they have a batch dimension.

        Args:
            rope (Optional[tuple[Tensor, Tensor]]): Tuple of (sin, cos) RoPE embeddings,
                or None if RoPE is not used.
            indices (Tensor): Indices to select from the batch dimension.

        Returns:
            Optional[tuple[Tensor, Tensor]]: Indexed RoPE embeddings or None.
        """
        if rope is None:
            return None

        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            # If the rope embedding has a batch dimension (is different for each batch element), index into it
            return sin[indices], cos[indices]  # [batch, heads, patches, embed_dim]
        else:
            # No batch dimension, do not index
            return sin, cos  # [heads, patches, embed_dim] or [patches, embed_dim]

    def _forward(self, x: Tensor, rope: Optional[tuple[Tensor, Tensor]] = None) -> Tensor:
        """Forward pass for a single tensor (reference implementation).

        This is the reference implementation for a single tensor, matching what is done
        in _forward_list. In practice, we call the list op on [x] instead of this function.

        Args:
            x (Tensor): Input tensor of shape (batch, seq_len, dim).
            rope (Optional[tuple[Tensor, Tensor]]): Optional RoPE embeddings (sin, cos).
                Defaults to None.

        Returns:
            Tensor: Output tensor of shape (batch, seq_len, dim).
        """
        b, _, _ = x.shape
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)

            x_attn = torch.index_add(
                x,
                dim=0,
                source=self.ls1(residual_1),
                index=indices_1,
                alpha=residual_scale_factor,
            )

            indices_2 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_2 = x_attn[indices_2]
            residual_2 = self.mlp(self.norm2(x_subset_2))

            x_ffn = torch.index_add(
                x_attn,
                dim=0,
                source=self.ls2(residual_2),
                index=indices_2,
                alpha=residual_scale_factor,
            )
        else:
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

        return x_ffn

    def _forward_list(
        self, x_list: List[Tensor], rope_list: Optional[List[Optional[tuple[Tensor, Tensor]]]] = None
    ) -> List[Tensor]:
        """Forward pass for a list of tensors with optimized concatenation.

        This list operator concatenates the tokens from the list of inputs together to save
        on the elementwise operations. Torch-compile memory-planning allows hiding the overhead
        related to concat ops.

        Args:
            x_list (List[Tensor]): List of input tensors, each of shape (batch_i, seq_len_i, dim).
            rope_list (Optional[List[Optional[tuple[Tensor, Tensor]]]]): Optional list of RoPE
                embeddings for each input tensor. Defaults to None.

        Returns:
            List[Tensor]: List of output tensors with the same shapes as inputs.
        """
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
        residual_scale_factors = [b / sample_subset_size for b, sample_subset_size in zip(b_list, sample_subset_sizes)]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)  # type: ignore[operator]

            x_attn_list = [
                torch.index_add(
                    x,
                    dim=0,
                    source=self.ls1(residual_1),
                    index=indices_1,
                    alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]

            indices_2_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)  # type: ignore[operator]

            x_ffn = [
                torch.index_add(
                    x_attn,
                    dim=0,
                    source=self.ls2(residual_2),
                    index=indices_2,
                    alpha=residual_scale_factor,
                )
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                )
            ]
        else:
            x_out = []
            rope_list_iter = rope_list if rope_list is not None else [None] * len(x_list)
            for x, rope in zip(x_list, rope_list_iter):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                x_out.append(x_ffn)
            x_ffn = x_out

        return x_ffn

    def forward(
        self,
        x_or_x_list: Tensor | List[Tensor],
        rope_or_rope_list: Optional[tuple[Tensor, Tensor]] | List[Optional[tuple[Tensor, Tensor]]] | None = None,
    ) -> Tensor | List[Tensor]:
        """Forward pass supporting both single tensor and list of tensors.

        Args:
            x_or_x_list (Tensor | List[Tensor]): Input tensor(s) of shape (batch, seq_len, dim).
            rope_or_rope_list (Optional[tuple[Tensor, Tensor]] | List[Optional[tuple[Tensor, Tensor]]] | None):
                Optional RoPE embeddings. Can be a single tuple, list of tuples, or None. Defaults to None.

        Returns:
            Tensor | List[Tensor]: Output tensor(s) with the same shape(s) as input.

        Raises:
            AssertionError: If input type is neither Tensor nor list.
        """
        if isinstance(x_or_x_list, Tensor):
            # for reference:
            # return self._forward(x_or_x_list, rope=rope_or_rope_list)
            # in order to match implementations we call the list op:
            rope_list_arg = [rope_or_rope_list] if not isinstance(rope_or_rope_list, list) else rope_or_rope_list  # type: ignore[list-item]
            return self._forward_list([x_or_x_list], rope_list=rope_list_arg)[0]
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_list_arg = [None for _ in x_or_x_list]
            elif isinstance(rope_or_rope_list, list):
                rope_list_arg = rope_or_rope_list
            else:
                rope_list_arg = [rope_or_rope_list for _ in x_or_x_list]
            return self._forward_list(x_or_x_list, rope_list=rope_list_arg)  # type: ignore[arg-type]
        else:
            raise AssertionError


class CausalSelfAttentionBlock(SelfAttentionBlock):
    """Causal self-attention Transformer block with optional sample dropping.

    Implements a standard Transformer block with causal self-attention and feed-forward
    layers, optionally with LayerScale and sample dropping for efficiency during
    training. Supports both single tensor and list of tensors as input.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = CausalSelfAttention,
        ffn_layer: Callable[..., nn.Module] = MLP,
        mask_k_bias: bool = False,
    ) -> None:
        """Initialize the CausalSelfAttentionBlock.

        Args:
            dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            ffn_ratio (float): Ratio of FFN hidden dimension to embedding dimension.
                Defaults to 4.0.
            qkv_bias (bool): Whether to use bias in QKV projection. Defaults to False.
            proj_bias (bool): Whether to use bias in attention output projection.
                Defaults to True.
            ffn_bias (bool): Whether to use bias in FFN layers. Defaults to True.
            drop (float): Dropout probability for FFN. Defaults to 0.0.
            attn_drop (float): Dropout probability for attention. Defaults to 0.0.
            init_values (Optional[float]): Initial value for LayerScale. If None,
                LayerScale is not used. Defaults to None.
            drop_path (float): Sample drop ratio for stochastic depth. Defaults to 0.0.
            act_layer (Callable[..., nn.Module]): Activation function class.
                Defaults to nn.GELU.
            norm_layer (Callable[..., nn.Module]): Normalization layer class.
                Defaults to nn.LayerNorm.
            attn_class (Callable[..., nn.Module]): Attention class to use.
                Defaults to CausalSelfAttention.
            ffn_layer (Callable[..., nn.Module]): FFN class to use. Defaults to MLP.
            mask_k_bias (bool): Whether to mask key bias in attention. Defaults to False.
        """
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            drop=drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            attn_class=attn_class,
            ffn_layer=ffn_layer,
            mask_k_bias=mask_k_bias,
            )


class BlockCausalSelfAttentionBlock(SelfAttentionBlock):
    """Block-causal self-attention Transformer block with optional sample dropping.

    Implements a standard Transformer block with block-causal self-attention and feed-forward
    layers, optionally with LayerScale and sample dropping for efficiency during
    training. Supports both single tensor and list of tensors as input.
    """

    def __init__(
        self,
        block_size: int,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = BlockCausalSelfAttention,
        ffn_layer: Callable[..., nn.Module] = MLP,
        mask_k_bias: bool = False,
        prefix_block_size: int = 0,
    ) -> None:
        """Initialize the CausalSelfAttentionBlock.

        Args:
            block_size (int): Size of each attention block.
            dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            ffn_ratio (float): Ratio of FFN hidden dimension to embedding dimension.
                Defaults to 4.0.
            qkv_bias (bool): Whether to use bias in QKV projection. Defaults to False.
            proj_bias (bool): Whether to use bias in attention output projection.
                Defaults to True.
            ffn_bias (bool): Whether to use bias in FFN layers. Defaults to True.
            drop (float): Dropout probability for FFN. Defaults to 0.0.
            attn_drop (float): Dropout probability for attention. Defaults to 0.0.
            init_values (Optional[float]): Initial value for LayerScale. If None,
                LayerScale is not used. Defaults to None.
            drop_path (float): Sample drop ratio for stochastic depth. Defaults to 0.0.
            act_layer (Callable[..., nn.Module]): Activation function class.
                Defaults to nn.GELU.
            norm_layer (Callable[..., nn.Module]): Normalization layer class.
                Defaults to nn.LayerNorm.
            attn_class (Callable[..., nn.Module]): Attention class to use.
                Defaults to BlockCausalSelfAttention.
            ffn_layer (Callable[..., nn.Module]): FFN class to use. Defaults to MLP.
            mask_k_bias (bool): Whether to mask key bias in attention. Defaults to False.
            prefix_block_size (int): Size of the first (prefix) block. If 0, all blocks have size
                `block_size`. Defaults to 0.
        """
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            ffn_ratio=ffn_ratio,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            ffn_bias=ffn_bias,
            drop=drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            attn_class=partial(attn_class, block_size=block_size, prefix_block_size=prefix_block_size),
            ffn_layer=ffn_layer,
            mask_k_bias=mask_k_bias,
            )

