import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ifo.common.utils.functions import cat_keep_shapes, uncat_with_shapes


def rope_rotate_half(x: Tensor) -> Tensor:
    """Rotate half of the features for RoPE (Rotary Position Embedding).

    Splits the last dimension in half, swaps the halves, and negates the second half.
    This is a key operation in applying rotary position embeddings.

    Args:
        x (Tensor): Input tensor with shape [..., D] where D is even.

    Returns:
        Tensor: Rotated tensor with same shape as input.
            Example: [x0, x1, x2, x3, x4, x5] -> [-x3, -x4, -x5, x0, x1, x2]
    """
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    """Apply RoPE (Rotary Position Embedding) to input tensor.

    Applies rotary position embeddings using precomputed sine and cosine values.
    The operation is: x * cos + rotate_half(x) * sin

    Args:
        x (Tensor): Input tensor with shape [..., D]
        sin (Tensor): Sine values with shape [..., D], with repeated frequency components
        cos (Tensor): Cosine values with shape [..., D], with repeated frequency components

    Returns:
        Tensor: Tensor with RoPE applied, same shape as input.
    """
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):
    """Linear layer with masked bias for K (key) in QKV projection.

    This is a specialized linear layer where the bias for the K (key) component
    is masked with NaN values. Used in attention mechanisms where K bias should
    be ignored. Assumes output features are divisible by 3 (for Q, K, V).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: Tensor) -> Tensor:
        """Forward pass with masked bias.

        Args:
            input (Tensor): Input tensor of shape [..., in_features]

        Returns:
            Tensor: Output tensor of shape [..., out_features] with masked bias applied
        """
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    """Multi-head self-attention module with optional RoPE support.

    Implements standard multi-head self-attention with support for:
    - Rotary position embeddings (RoPE)
    - Optional K (key) bias masking
    - Batch processing of multiple inputs with different shapes

    Args:
        dim (int): Input/output dimension (must be divisible by num_heads)
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool): Whether to use bias in QKV projection. Default: False
        proj_bias (bool): Whether to use bias in output projection. Default: True
        attn_drop (float): Dropout rate for attention weights. Default: 0.0
        proj_drop (float): Dropout rate for output projection. Default: 0.0
        mask_k_bias (bool): Whether to mask the bias for K (key) component. Default: False
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """Apply rotary position embeddings to query and key tensors.

        Handles cases where q/k may have a prefix that doesn't receive RoPE.
        All operations are performed in rope's dtype, then cast back to original dtypes.

        Args:
            q (Tensor): Query tensor of shape [..., num_heads, N, head_dim] where ... is any batch dims
            k (Tensor): Key tensor of shape [..., num_heads, N, head_dim] where ... is any batch dims
            rope (Tuple[Tensor, Tensor]): Tuple of (sin, cos) tensors for RoPE application

        Returns:
            Tuple[Tensor, Tensor]: Tuple of (q, k) with RoPE applied, in their original dtypes
        """
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[..., :prefix, :]
        q = rope_apply(q[..., prefix:, :], sin, cos)  # [..., head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [..., head, N, D//head]
        k_prefix = k[..., :prefix, :]
        k = rope_apply(k[..., prefix:, :], sin, cos)  # [..., head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [..., head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(
        self,
        x: Tensor,
        attn_bias: Optional[Tensor] = None,
        rope: Optional[Tuple[Tensor, Tensor]] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Forward pass for self-attention.

        Args:
            x (Tensor): Input tensor of shape [..., N, dim] where ... is any number of batch dims
            attn_bias (Optional[Tensor]): Attention bias (currently not supported, must be None)
            rope (Optional[Tuple[Tensor, Tensor]]): Optional RoPE embeddings as (sin, cos) tuple
            is_causal (bool): Whether to apply causal masking. Default: False, so full self-attention is applied
        Returns:
            Tensor: Output tensor of shape [..., N, dim]
        """
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope, is_causal=is_causal)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def forward_list(
        self,
        x_list: List[Tensor],
        attn_bias: Optional[Tensor] = None,
        rope_list: Optional[List[Optional[Tuple[Tensor, Tensor]]]] = None,
        is_causal: bool = False,
    ) -> List[Tensor]:
        """Forward pass for a list of inputs with different shapes.

        Efficiently processes multiple inputs by batching the QKV projection,
        then computing attention separately for each input (as they may have
        different shapes and RoPE embeddings).

        Args:
            x_list (List[Tensor]): List of input tensors, each with shape [..., N_i, dim]
            attn_bias (Optional[Tensor]): Attention bias (currently not supported, must be None)
            rope_list (Optional[List[Optional[Tuple[Tensor, Tensor]]]]): List of RoPE embeddings, one per input
            is_causal (bool): Whether to apply causal masking. Default: False, so full self-attention is applied

        Returns:
            List[Tensor]: List of output tensors with same shapes as inputs
        """
        assert len(x_list) == len(rope_list)  # should be enforced by the Block
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope, is_causal=is_causal))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        x_flat = self.proj_drop(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(
        self,
        qkv: Tensor,
        attn_bias: Optional[Tensor] = None,
        rope: Optional[Tuple[Tensor, Tensor]] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Compute multi-head attention from QKV tensor.

        Args:
            qkv (Tensor): Combined query-key-value tensor of shape [..., N, dim * 3] where ... is any batch dims
            attn_bias (Optional[Tensor]): Attention bias (currently not supported, must be None)
            rope (Optional[Tuple[Tensor, Tensor]]): Optional RoPE embeddings as (sin, cos) tuple
            is_causal (bool): Whether to apply causal masking. Default: False, so full self-attention is applied

        Returns:
            Tensor: Attention output tensor of shape [..., N, dim]
        """
        assert attn_bias is None
        *batch_dims, N, _ = qkv.shape
        C = self.qkv.in_features
        head_dim = C // self.num_heads

        # [..., N, dim * 3] -> [..., N, 3, num_heads, head_dim]
        qkv = qkv.reshape(*batch_dims, N, 3, self.num_heads, head_dim)
        # Unbind on the "3" dimension to get q, k, v each of shape [..., N, num_heads, head_dim]
        q, k, v = torch.unbind(qkv, dim=-3)
        # Move num_heads before N: [..., N, num_heads, head_dim] -> [..., num_heads, N, head_dim]
        q, k, v = [t.transpose(-3, -2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop if self.training else 0, is_causal=is_causal)
        # [..., num_heads, N, head_dim] -> [..., N, num_heads, head_dim]
        x = x.transpose(-3, -2)
        return x.reshape(*batch_dims, N, C)


class CausalSelfAttention(SelfAttention):
    """Causal multi-head self-attention for autoregressive models.

    Implements causal (masked) self-attention where each position can only attend
    to positions at or before it. This is commonly used in decoder-only transformers
    and autoregressive language models.

    Args:
        dim (int): Input/output dimension (must be divisible by num_heads)
        num_heads (int): Number of attention heads. Default: 8
        qkv_bias (bool): Whether to use bias in QKV projection. Default: False
        proj_bias (bool): Whether to use bias in output projection. Default: True
        attn_drop (float): Dropout rate for attention weights. Default: 0.0
        proj_drop (float): Dropout rate for output projection. Default: 0.0
    """
    def forward(
        self,
        x: Tensor,
        attn_bias: Optional[Tensor] = None,
        rope: Optional[Tuple[Tensor, Tensor]] = None,
        is_causal: bool = True,
    ) -> Tensor:
        """Forward pass for causal self-attention.

        Args:
            x (Tensor): Input tensor of shape [..., N, dim] where ... is any number of batch dims
            attn_bias (Optional[Tensor]): Attention bias (currently not supported, must be None)
            rope (Optional[Tuple[Tensor, Tensor]]): Optional RoPE embeddings as (sin, cos) tuple
            is_causal (bool): Whether to apply causal masking. Default: True, so causal self-attention is applied
        Returns:
            Tensor: Output tensor of shape [..., N, dim]
        """
        return super().forward(x, attn_bias=attn_bias, rope=rope, is_causal=is_causal)

    def forward_list(
        self,
        x_list: List[Tensor],
        attn_bias: Optional[Tensor] = None,
        rope_list: Optional[List[Optional[Tuple[Tensor, Tensor]]]] = None,
        is_causal: bool = True,
    ) -> List[Tensor]:
        """Forward pass for a list of inputs with different shapes.

        Efficiently processes multiple inputs by batching the QKV projection,
        then computing attention separately for each input (as they may have
        different shapes and RoPE embeddings).

        Args:
            x_list (List[Tensor]): List of input tensors, each with shape [..., N_i, dim]
            attn_bias (Optional[Tensor]): Attention bias (currently not supported, must be None)
            rope_list (Optional[List[Optional[Tuple[Tensor, Tensor]]]]): List of RoPE embeddings, one per input
            is_causal (bool): Whether to apply causal masking. Default: True, so causal self-attention is applied

        Returns:
            List[Tensor]: List of output tensors with same shapes as inputs
        """
        return super().forward_list(x_list, attn_bias=attn_bias, rope_list=rope_list, is_causal=is_causal)


class BlockCausalSelfAttention(SelfAttention):
    """Block-causal multi-head self-attention for sequences with block structure.

    Implements block-causal attention where the sequence is divided into blocks:
    - Within each block, full bidirectional attention is allowed
    - Across blocks, causal attention is enforced (can only attend to previous blocks)

    This is useful for models that process sequences with natural block structure,
    such as video frames, document paragraphs, or multi-frame observations.

    For a sequence divided into blocks of size `block_size` with optional `prefix_block_size`:
    - If prefix_block_size == 0 (default):
        - Block 0: positions [0, block_size)
        - Block 1: positions [block_size, 2*block_size)
        - etc.
    - If prefix_block_size > 0:
        - Block 0: positions [0, prefix_block_size)  (prefix block)
        - Block 1: positions [prefix_block_size, prefix_block_size + block_size)
        - Block 2: positions [prefix_block_size + block_size, prefix_block_size + 2*block_size)
        - etc.

    A token at position i in block b_i can attend to a token at position j in block b_j if:
    - b_i > b_j (previous block), or
    - b_i == b_j (same block, full attention within block)

    Args:
        dim (int): Input/output dimension (must be divisible by num_heads)
        num_heads (int): Number of attention heads. Default: 8
        block_size (int): Size of each attention block. Default: 1 (equivalent to standard causal)
        prefix_block_size (int): Size of the first (prefix) block. If 0, all blocks have size
            `block_size`. Default: 0
        qkv_bias (bool): Whether to use bias in QKV projection. Default: False
        proj_bias (bool): Whether to use bias in output projection. Default: True
        attn_drop (float): Dropout rate for attention weights. Default: 0.0
        proj_drop (float): Dropout rate for output projection. Default: 0.0
        mask_k_bias (bool): Whether to mask the bias for K (key) component. Default: False
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        block_size: int = 1,
        prefix_block_size: int = 0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
    ) -> None:
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            mask_k_bias=mask_k_bias,
        )
        assert block_size >= 1, "block_size must be at least 1"
        assert prefix_block_size >= 0, "prefix_block_size must be non-negative"
        self.block_size = block_size
        self.prefix_block_size = prefix_block_size

    def _create_block_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Create a block-causal attention mask.

        The mask allows:
        - Full attention within each block
        - Causal attention across blocks (can only attend to previous blocks)

        If prefix_block_size > 0, the first block has a different size than subsequent blocks.

        Args:
            seq_len (int): Sequence length
            device (torch.device): Device to create the mask on

        Returns:
            Tensor: Boolean mask of shape [seq_len, seq_len] where True indicates
                positions that should NOT be attended to (masked out)
        """
        # Compute block indices for each position
        positions = torch.arange(seq_len, device=device)

        if self.prefix_block_size == 0:
            # Regular block structure: all blocks have the same size
            block_indices = positions // self.block_size  # [seq_len]
        else:
            # Prefix block structure: first block has prefix_block_size, rest have block_size
            # Positions in prefix block (0 to prefix_block_size-1) -> block 0
            # Positions after prefix -> block 1 + (pos - prefix_block_size) // block_size
            block_indices = torch.where(
                positions < self.prefix_block_size,
                torch.zeros_like(positions),
                1 + (positions - self.prefix_block_size) // self.block_size,
            )

        # Create mask: query can attend to key if query_block >= key_block
        # query_block_indices: [seq_len, 1], key_block_indices: [1, seq_len]
        query_blocks = block_indices.unsqueeze(1)  # [seq_len, 1]
        key_blocks = block_indices.unsqueeze(0)  # [1, seq_len]

        # Mask is True where attention should be BLOCKED (query_block < key_block)
        mask = query_blocks < key_blocks  # [seq_len, seq_len]
        return mask

    def compute_attention(
        self,
        qkv: Tensor,
        attn_bias: Optional[Tensor] = None,
        rope: Optional[Tuple[Tensor, Tensor]] = None,
        is_causal: bool = False,
    ) -> Tensor:
        """Compute multi-head attention with block-causal masking.

        Args:
            qkv (Tensor): Combined query-key-value tensor of shape [..., N, dim * 3] where ... is any batch dims
            attn_bias (Optional[Tensor]): Attention bias (currently not supported, must be None)
            rope (Optional[Tuple[Tensor, Tensor]]): Optional RoPE embeddings as (sin, cos) tuple
            is_causal (bool): Ignored, block-causal masking is always applied

        Returns:
            Tensor: Attention output tensor of shape [..., N, dim]
        """
        assert attn_bias is None
        *batch_dims, N, _ = qkv.shape
        C = self.qkv.in_features
        head_dim = C // self.num_heads

        # [..., N, dim * 3] -> [..., N, 3, num_heads, head_dim]
        qkv = qkv.reshape(*batch_dims, N, 3, self.num_heads, head_dim)
        # Unbind on the "3" dimension to get q, k, v each of shape [..., N, num_heads, head_dim]
        q, k, v = torch.unbind(qkv, dim=-3)
        # Move num_heads before N: [..., N, num_heads, head_dim] -> [..., num_heads, N, head_dim]
        q, k, v = [t.transpose(-3, -2) for t in [q, k, v]]

        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        # If block_size is 1 and no prefix, this is equivalent to standard causal attention
        if self.block_size == 1 and self.prefix_block_size == 0:
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=self.attn_drop if self.training else 0, is_causal=True
            )
        else:
            # Create block-causal mask and apply attention
            # The mask needs to be [..., num_heads, N, N] or broadcastable
            attn_mask = self._create_block_causal_mask(N, q.device)
            # Convert boolean mask to float mask: True (blocked) -> -inf, False (allowed) -> 0
            attn_mask = attn_mask.float().masked_fill(attn_mask, float("-inf"))
            # Expand for batch and heads: [1, 1, N, N] - broadcasts with any leading batch dims
            attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop if self.training else 0
            )

        # [..., num_heads, N, head_dim] -> [..., N, num_heads, head_dim]
        x = x.transpose(-3, -2)
        return x.reshape(*batch_dims, N, C)
