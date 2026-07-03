import math
from typing import Literal, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn


class RopePositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for 1D sequences.

    Implements standard RoPE positional embeddings for sequential data (e.g., text, time series).
    This is a non-learnable positional encoding that uses sinusoidal functions based on
    rotation matrices.

    RoPE was introduced in "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    and provides an elegant way to encode relative position information through rotation
    of embedding vectors.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float = 10000.0,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize the RoPE position embedding module for 1D sequences.

        Args:
            embed_dim (int): Total embedding dimension (must be divisible by 2 * num_heads).
            num_heads (int): Number of attention heads.
            base (float): Base for geometric frequency sequence. Defaults to 10000.0 (standard for NLP).
                Base dictates the maximum context length of the sequence, which is base * 2 * pi.
            dtype (Optional[torch.dtype]): Data type for the inverse frequencies buffer. Defaults to None.

        Raises:
            AssertionError: If embed_dim is not divisible by 2 * num_heads.
        """
        super().__init__()
        assert embed_dim % (2 * num_heads) == 0

        D_head = embed_dim // num_heads
        self.D_head = D_head
        self.dtype = dtype

        # Compute inverse frequencies: 1 / (base^(2i/d)) for i in [0, d/2)
        i = torch.arange(0, D_head, 2, dtype=dtype)
        inv_freq = 1.0 / (base ** (i / D_head))

        # Register as buffer for proper device handling and state_dict
        self.register_buffer("inv_freq", inv_freq, persistent=True)

    def forward(self, *, L: int) -> Tuple[Tensor, Tensor]:
        """Generate sine and cosine positional embeddings for a 1D sequence.

        Computes standard RoPE embeddings for all positions in a sequence of length L.

        Args:
            L (int): Length of the sequence (number of tokens/positions).

        Returns:
            Tuple[Tensor, Tensor]: A tuple of (sin, cos) tensors, each of shape [L, D_head],
                where D_head is the per-head embedding dimension. These are used to apply
                rotary position encoding to queries and keys in attention.
        """
        inv_freq: Tensor = self.inv_freq  # type: ignore[assignment]

        # Create position indices [0, 1, 2, ..., L-1]
        positions = torch.arange(L, dtype=self.dtype, device=inv_freq.device)  # [L]

        # Compute angles: position * inv_freq
        # positions: [L], inv_freq: [D_head//2] -> angles: [L, D_head//2]
        angles = torch.outer(positions, inv_freq)  # [L, D_head//2]

        # Duplicate for interleaved sin/cos pattern
        angles = angles.repeat_interleave(2, dim=1)  # [L, D_head]

        # Compute sin and cos
        sin = torch.sin(angles)  # [L, D_head]
        cos = torch.cos(angles)  # [L, D_head]

        return (sin, cos)  # 2 * [L, D_head]


class ImageRopePositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for 2D images without coordinate mixing.

    Implements axial RoPE positional embeddings for vision tasks, where x and y
    coordinates are encoded separately without mixing. This is a non-learnable
    positional encoding that uses sinusoidal functions based on rotation matrices.

    RoPE was introduced in "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    and has been adapted here for 2D spatial inputs (images). The implementation supports
    two parametrizations:
    1. Base parametrization: frequencies determined by a geometric sequence
    2. Period parametrization: explicit min/max periods for frequency range

    The encoding supports various augmentation strategies including coordinate shifting,
    jittering, and rescaling for improved robustness.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: Optional[float] = 100.0,
        min_period: Optional[float] = None,
        max_period: Optional[float] = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: Optional[float] = None,
        jitter_coords: Optional[float] = None,
        rescale_coords: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        """Initialize the RoPE position embedding module.

        Args:
            embed_dim (int): Total embedding dimension (must be divisible by 4 * num_heads).
            num_heads (int): Number of attention heads.
            base (Optional[float]): Base for geometric frequency sequence. Must be provided
                if min_period and max_period are not. Defaults to 100.0.
            min_period (Optional[float]): Minimum period for frequency range. Must be
                provided together with max_period if base is None. Defaults to None.
            max_period (Optional[float]): Maximum period for frequency range. Must be
                provided together with min_period if base is None. Defaults to None.
            normalize_coords (Literal["min", "max", "separate"]): Coordinate normalization
                strategy. "separate" normalizes height and width independently, "max"
                normalizes by the larger dimension, "min" by the smaller. Defaults to "separate".
            shift_coords (Optional[float]): If provided and training, randomly shift
                coordinates by uniform value in [-shift_coords, shift_coords]. Defaults to None.
            jitter_coords (Optional[float]): If provided and training, randomly scale
                coordinates by log-uniform factor in [1/jitter_coords, jitter_coords].
                Defaults to None.
            rescale_coords (Optional[float]): If provided and training, randomly rescale
                all coordinates by log-uniform factor in [1/rescale_coords, rescale_coords].
                Defaults to None.
            dtype (Optional[torch.dtype]): Data type for the periods buffer. Defaults to None.

        Raises:
            AssertionError: If embed_dim is not divisible by 4 * num_heads.
            ValueError: If neither base nor (min_period, max_period) pair is provided,
                or if both are provided.
        """
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        # Needs persistent=True because we do teacher.load_state_dict(student.state_dict()) to initialize the teacher
        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, dtype=self.dtype),
            persistent=True,  # Will be saved in the state_dict
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> Tuple[Tensor, Tensor]:
        """Generate sine and cosine positional embeddings for a 2D spatial grid.

        Computes axial RoPE embeddings for all positions in an H×W grid. During training,
        applies optional coordinate augmentations (shifting, jittering, rescaling).

        Args:
            H (int): Height of the spatial grid (number of patches/tokens vertically).
            W (int): Width of the spatial grid (number of patches/tokens horizontally).

        Returns:
            Tuple[Tensor, Tensor]: A tuple of (sin, cos) tensors, each of shape [H*W, D_head],
                where D_head is the per-head embedding dimension. These are used to apply
                rotary position encoding to queries and keys in attention.
        """
        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, dtype=self.dtype, device=self.periods.device) / max_HW  # [H]
            coords_w = torch.arange(0.5, W, dtype=self.dtype, device=self.periods.device) / max_HW  # [W]
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, dtype=self.dtype, device=self.periods.device) / min_HW  # [H]
            coords_w = torch.arange(0.5, W, dtype=self.dtype, device=self.periods.device) / min_HW  # [W]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, dtype=self.dtype, device=self.periods.device) / H  # [H]
            coords_w = torch.arange(0.5, W, dtype=self.dtype, device=self.periods.device) / W  # [W]
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, dtype=self.dtype).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_hw[None, :]

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, dtype=self.dtype).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, dtype=self.dtype).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_hw

        # Prepare angles and sin/cos
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # type: ignore[index]  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def _init_weights(self) -> None:
        """Initialize the frequency periods for RoPE.

        Computes and stores the periods (inverse frequencies) for the sinusoidal
        position encoding. Uses either the base parametrization (geometric sequence)
        or the explicit min/max period parametrization.

        The periods determine the wavelengths of the sinusoidal functions used for
        position encoding, with different periods capturing position information at
        different scales.
        """
        if self.base is not None:
            # base^(2i/d)
            # We encode two halfs of D_head for h and w, so instead of i in [0, d/2],
            # we have i in [0, d/4]. We divide d also by 2 to get the correct periods.
            i = torch.arange(self.D_head // 4, dtype=self.dtype)
            d = self.D_head // 2
            periods = self.base ** (2 * i / d)
        else:
            # Both max_period and min_period are guaranteed to be non-None by __init__ validation
            assert self.max_period is not None and self.min_period is not None
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, dtype=self.dtype)  # type: ignore[arg-type]  # [D//4] range [0, 1]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]
        self.periods.data = periods
