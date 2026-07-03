from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class VectorQuantizerEMA(nn.Module):
    """VQ-VAE layer with exponential moving average update of codes.

    Implements a slightly modified version of the algorithm presented in
    'Neural Discrete Representation Learning' by van den Oord et al.
    https://arxiv.org/abs/1711.00937

    The difference between VectorQuantizerEMA and VectorQuantizer is that
    this module uses exponential moving averages to update the embedding vectors
    instead of an auxiliary loss. This has the advantage that the embedding
    updates are independent of the choice of optimizer (SGD, RMSProp, Adam, K-Fac,
    ...) used for the encoder, decoder and other parts of the architecture. For
    most experiments the EMA version trains faster than the non-EMA version.

    Input any tensor to be quantized. Last dimension will be used as space in
    which to quantize. All other dimensions will be flattened and will be seen
    as different examples to quantize.

    The output tensor will have the same shape as the input.

    For example a tensor with shape [16, 32, 32, 64] will be reshaped into
    [16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
    independently.

    Reference: https://github.com/google-deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
               https://github.com/mazpie/choreographer/blob/main/agent/skill_rep.py#L162
    """

    def __init__(
        self,
        num_codes: int,
        code_dim: int,
        num_codebooks: int = 1,
        beta: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ) -> None:
        """Initializes a VQ-VAE module.

        Args:
            num_codes (int): Number of codes to use in the codebook.
            code_dim (int): Dimensionality of the latent codes.
            num_codebooks (int): Number of codebooks to use. If greater than 1, the latent will be split into num_codebooks parts.
            beta (float): Scalar which controls the weighting of the commitment loss (see equation 4 in the paper).
            decay (float): Controls the speed of the exponential moving average [0, 1.0].
            epsilon (float): Small constant to aid numerical stability.
        """
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.num_codebooks = num_codebooks
        self.beta = beta
        self.decay = decay
        self.epsilon = epsilon

        if not 0 <= self.decay <= 1:
            raise ValueError(f"Decay must be in range [0, 1], but is {self.decay}.")

        if num_codebooks > 1:
            assert (
                code_dim % num_codebooks == 0
            ), f"Code dimension must be divisible by the number of codebooks, but is {code_dim} % {num_codebooks}."
            self.code_dim = code_dim // num_codebooks

        # Disables gradients to update codebook with exponential moving average (EMA).
        codebook = torch.zeros(self.num_codebooks, self.num_codes, self.code_dim)  # [N, K, D]
        codebook.uniform_(
            -1 / self.num_codes * 5, 1 / self.num_codes * 5
        )  # TODO: Check if we really need to multiply with 5.

        # Initialize EMA.
        self.register_buffer("codebook", codebook)
        self.register_buffer("ema_count", torch.zeros(self.num_codebooks, self.num_codes))
        self.register_buffer("ema_weight", self.codebook.clone())

    def update_codebook(self, x: Tensor, code_assignment: Tensor) -> None:
        """Update the codebook using EMA.

        Args:
            x (Tensor): Latent to quantize. Shape: [N, B*H, D].
            code_assignment (Tensor): Assignment of codes to latent. Shape: [N, K].
        """
        # Calculate EMA for the number of embeddings per code: N_i^{(t)}.
        # N_i^{(t)} = N_i^{(t-1)} * \gamma + n_i^{(t)} * (1 - \gamma)
        ema_count = self.decay * self.ema_count + (1 - self.decay) * code_assignment.sum(dim=1)  # [N, K]

        # Smooth code cluster size to prevent division by zero with Laplace smoothing.
        n = torch.sum(ema_count, dim=-1, keepdim=True)
        ema_count = (ema_count + self.epsilon) / (n + self.num_codes * self.epsilon) * n  # [N, K]

        # Calculate EMA of embeddings: m_i^{(t)}.
        embedding_sum = torch.bmm(code_assignment.transpose(1, 2), x)  # [N, K, B*H] @ [N, B*H, D] -> [N, K, D]
        # m_i^{(t)} = m_i^{(t-1)} * \gamma + \sum_j z_{i, j}^{(t)} (1 - \gamma)
        ema_weight = self.decay * self.ema_weight + (1 - self.decay) * embedding_sum  # [N, K, D]

        # Normalize code weight. e_i^{(t)} = \frac{m_i^{(t)}}{N_i^{(t)}}
        codebook = ema_weight / ema_count.unsqueeze(-1)  # [N, K, D] / [N, K, 1] -> [N, K, D]

        # Update buffers in-place using copy_() which is compile-friendly
        self.ema_count.copy_(ema_count)
        self.ema_weight.copy_(ema_weight)
        self.codebook.copy_(codebook)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass.

        Args:
            x (Tensor): Latent to quantize. Shape: [B, C, H], where B is batch size,
            C is the code dimension and H is the number of separate latents to quantize.

        Returns:
            Tensor: Quantized latent, loss, perplexity.
        """
        assert x.dim() == 3, f"Input tensor must have 3 dimensions, but has {x.dim()} dimensions."
        b, c, h = x.shape  # [B, C, H]

        # Reshape the latent to split into num_codebooks parts.
        assert (
            c % self.num_codebooks == 0
        ), f"Code dimension must be divisible by the number of codebooks, but is {c} % {self.num_codebooks}."
        x = x.view(-1, self.num_codebooks, self.code_dim, h)  # [B, N, D, H]
        x = x.permute(1, 0, 3, 2)  # [B, N, H, D] -> [N, B, H, D]
        x = x.reshape(self.num_codebooks, -1, self.code_dim)  # [N, B, H, D] -> [N, B*H, D]

        with torch.no_grad():
            # Compute L2 distance between latent and codes.
            dist = torch.baddbmm(
                torch.sum(self.codebook**2, dim=-1).unsqueeze(1)  # [N, K, D] -> [N, 1, K]
                + torch.sum(x**2, dim=-1, keepdim=True),  # [N, B*H, D] -> [N, B*H, 1],
                x,
                self.codebook.transpose(1, 2),  # [N, K, D] -> [N, D, K],
                alpha=-2.0,
                beta=1.0,
            )  # [N, B*H, K]

            # Get the code that has min distance to latent
            code_idx = torch.argmin(dist, dim=-1)  # [N, B*H]

            # Quantize the latent
            code = torch.gather(self.codebook, 1, code_idx.unsqueeze(-1).expand(-1, -1, self.code_dim))  # [N, B*H, D]
            code_assignment = F.one_hot(code_idx, self.num_codes).float()  # [N, B*H, K]

            if self.training:
                self.update_codebook(x, code_assignment)

            # Calculate perplexity to measure how well codes are used.
            code_probs = code_assignment.mean(1)  # [C, K]
            perplexity = torch.exp(-torch.sum(code_probs * torch.log(code_probs + 1e-10), dim=-1)).sum()

        # Compute the VQ Losses
        commitment_loss = F.mse_loss(code.detach(), x)
        vq_loss = self.beta * commitment_loss

        # Straight-trough gradient estimation. Add the residue back to the latent.
        code = x + (code - x).detach()

        # Reshape the code back to original shape. [N, B*H, D] -> [B, C, H]
        code = code.view(self.num_codebooks, b, h, self.code_dim)
        code = code.permute(1, 0, 2, 3).reshape(b, c, h)  # [N, B, H, D] -> [B, N, H, D] -> [B, C, H]

        return code, vq_loss, perplexity


class FiniteScalarQuantizer(nn.Module):
    """Finite Scalar Quantization Layer.

    Implements the algorithm presented in
    'Finite Scalar Quantization: VQ-VAE Made Simple' by Mentzer et al.

    NOTE: According to FSQ Paper, dimensions of implicit codebooks need to be reduced strongly
    compared to VQ VAE. For the standard configuration of 256 (2 x 128) codes with dimension 128,
    the paper recommends only a FSQ dimension of d = 3 with levels = [8,5,6]
    """

    def __init__(self, levels: List[int] = [8, 6, 5], code_dim: int = 3, eps: float = 1e-3) -> None:
        """Initializes a Finite Scalar Quantization module.

        Args:
            levels (List[int]): Levels that each input dimension is quantized to.
            code_dim (int): Dimensionality of the latent codes.
            eps (float): Small constant to aid numerical stability.
        """
        super().__init__()
        assert len(levels) == code_dim, f" The number of levels {levels} must be equal to the code dim {code_dim}."
        self._eps = eps
        self.code_dim = code_dim
        self.register_buffer("_levels", torch.as_tensor(levels, dtype=torch.int32))
        self.register_buffer("_half_width", (self._levels // 2))
        self.register_buffer(
            "_basis", torch.concatenate((torch.ones((1,), dtype=torch.int32), torch.cumprod(self._levels[:-1], 0)))
        )
        self.register_buffer(
            "_implicit_codebook", self.index_to_code(torch.arange(self.codebook_size, dtype=torch.int32))
        )  # [D, C]

    @property
    def codebook_size(self) -> int:
        """Returns the size of the codebook.

        Returns:
            int: Size of the codebook.
        """
        return int(torch.prod(self._levels).item())

    def _bound(self, x: Tensor) -> Tensor:
        """Bounds the passed input tensor along each dimension according to `self._levels`

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Bounded input tensor

        Example:
            Suppose self._levels = [2, 5], so x has shape (..., 2).
            For the first dimension (levels=2):
                half_l ≈ 0.5, offset=0.5
                The output is bounded to [-1, 0] because tanh(x + shift) ∈ [-1, 1]
            For the second dimension (levels=5):
                half_l ≈ 2, offset=0
                The output is bounded to [-2, 2], so quantization will select from {-2, -1, 0, 1, 2}.
        """
        # Calculate half width and offset for each dimension
        half_l = (self._levels - 1) * (1 - self._eps) / 2
        offset = torch.where(self._levels % 2 == 1, 0.0, 0.5)
        shift = torch.tan(offset / half_l)
        return torch.tanh(x + shift) * half_l - offset

    def _round_ste(self, z: Tensor) -> Tensor:
        """Round with straight through gradients.

        Args:
            z (Tensor): Input tensor.

        Returns:
            Tensor: Rounded tensor.
        """
        zhat = torch.round(z)
        return z + (zhat - z).detach()

    def index_to_code(self, index: Tensor) -> Tensor:
        """
        Converts codebook indices to their corresponding quantized code vectors.

        Args:
            index (Tensor): Tensor of indices, shape (...,).

        Returns:
            Tensor: Quantized code vectors corresponding to the indices, shape (..., code_dim).

        Example:
            To obtain the entire codebook, use:
                codes = self.index_to_code(torch.arange(self.codebook_size))
        """
        # Returns all codes non centered e.g. [[0,0], [0,1], [0,2], [1,0], [1,1], [1,2]] for levels = [2,3]
        codes_non_centered = torch.remainder(torch.floor_divide(index[:, None], self._basis), self._levels)
        # Returns quantized codes [[-1, -1], [-1, 0], [-1,1], [0,-1], [0,0], [0,1]] for levels = [2,3]
        quantized_codes = (codes_non_centered - self._half_width) / self._half_width
        return quantized_codes

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Quantizes the input tensor using finite scalar quantization.

        Each dimension of the input tensor is first bounded to a specific interval using a scaled and shifted tanh,
        according to the number of quantization levels for that dimension. For example, if a dimension has 5 levels,
        its values are bounded to [-2, 2] and quantized to the nearest integer in {-2, -1, 0, 1, 2}.

        The quantization is performed with a straight-through estimator, allowing gradients to flow through the
        rounding operation.

        Args:
            x (Tensor): Input tensor to be quantized, of shape (batch, code_dim, num_latents).

        Returns:
            Tuple[Tensor, Tensor]:
                - Quantized tensor of the same shape as input (after renormalization to [-1, 1] per dimension).
                - Placeholder loss (always 0). Only done for API consistency.
                - Perplexity of code usage, indicating how evenly the quantization codes are utilized.
        """
        assert x.dim() == 3, f"Input tensor must have 3 dimensions, but has {x.dim()} dimensions."
        b, c, h = x.shape  # [B, C, H]
        x = x.permute(0, 2, 1)  # [B, C, H] -> [B, H, C]

        bounded_x = self._bound(x)
        quantized_x = self._round_ste(bounded_x)
        half_width = self._levels // 2

        # Renormalize each dimension to [-1,1]
        renormalized_quantized_x = quantized_x / half_width  # [B, H, C]

        # Get usage of each code
        with torch.no_grad():
            code_book = self._implicit_codebook[None]  # [1, D, C]
            used_codes = renormalized_quantized_x.view(-1, 1, c)  # [B*H, 1, C]
            code_assignment = torch.all(code_book == used_codes, dim=-1).float()  # [B*H, D]
            code_probs = code_assignment.mean(0)  # [D]
            perplexity = torch.exp(-torch.sum(code_probs * torch.log(code_probs + 1e-10), dim=-1)).sum()

        # Reshape back to original format
        renormalized_quantized_x = renormalized_quantized_x.permute(0, 2, 1)  # [B, H, C] -> [B, C, H]

        return renormalized_quantized_x, torch.zeros_like(perplexity), perplexity


class IdentityQuantizer(nn.Module):
    """Identity Quantization Layer that performs no quantization.

    This quantizer simply passes through the input unchanged, returning
    zero loss and unit perplexity. Useful as a baseline or for ablation studies.
    """

    def __init__(self) -> None:
        """Initializes an Identity Quantization module."""
        super().__init__()

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass that returns input unchanged.

        Args:
            x (Tensor): Input tensor to pass through. Shape: [B, C, H].

        Returns:
            Tuple[Tensor, Tensor, Tensor]:
                - Input tensor unchanged
                - Zero loss
                - Unit perplexity (1.0)
        """
        device = x.device
        zero_loss = torch.zeros(1, device=device)
        unit_perplexity = torch.ones(1, device=device)

        return x, zero_loss, unit_perplexity
