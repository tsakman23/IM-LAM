from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class VectorQuantizerEMA(nn.Module):
    """VQ-VAE with exponential moving average codebook update.

    Reference: https://arxiv.org/abs/1711.00937
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
        super().__init__()
        self.num_codes = num_codes
        self.code_dim = code_dim
        self.num_codebooks = num_codebooks
        self.beta = beta
        self.decay = decay
        self.epsilon = epsilon

        if num_codebooks > 1:
            assert code_dim % num_codebooks == 0
            self.code_dim = code_dim // num_codebooks

        codebook = torch.zeros(self.num_codebooks, self.num_codes, self.code_dim)
        codebook.uniform_(-1 / self.num_codes * 5, 1 / self.num_codes * 5)

        self.register_buffer("codebook", codebook)
        self.register_buffer("ema_count", torch.zeros(self.num_codebooks, self.num_codes))
        self.register_buffer("ema_weight", self.codebook.clone())

    def update_codebook(self, x: Tensor, code_assignment: Tensor) -> None:
        ema_count = self.decay * self.ema_count + (1 - self.decay) * code_assignment.sum(dim=1)
        n = torch.sum(ema_count, dim=-1, keepdim=True)
        ema_count = (ema_count + self.epsilon) / (n + self.num_codes * self.epsilon) * n

        embedding_sum = torch.bmm(code_assignment.transpose(1, 2), x)
        ema_weight = self.decay * self.ema_weight + (1 - self.decay) * embedding_sum
        codebook = ema_weight / ema_count.unsqueeze(-1)

        self.ema_count.copy_(ema_count)
        self.ema_weight.copy_(ema_weight)
        self.codebook.copy_(codebook)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Quantize input.

        Args:
            x: (B, C, H) where C is code_dim, H is num_latents.

        Returns:
            Quantized tensor, VQ loss, perplexity.
        """
        assert x.dim() == 3
        b, c, h = x.shape

        x = x.view(-1, self.num_codebooks, self.code_dim, h)
        x = x.permute(1, 0, 3, 2).reshape(self.num_codebooks, -1, self.code_dim)

        with torch.no_grad():
            dist = torch.baddbmm(
                torch.sum(self.codebook**2, dim=-1).unsqueeze(1)
                + torch.sum(x**2, dim=-1, keepdim=True),
                x,
                self.codebook.transpose(1, 2),
                alpha=-2.0,
                beta=1.0,
            )
            code_idx = torch.argmin(dist, dim=-1)
            code = torch.gather(self.codebook, 1, code_idx.unsqueeze(-1).expand(-1, -1, self.code_dim))
            code_assignment = F.one_hot(code_idx, self.num_codes).float()

            if self.training:
                self.update_codebook(x, code_assignment)

            code_probs = code_assignment.mean(1)
            perplexity = torch.exp(-torch.sum(code_probs * torch.log(code_probs + 1e-10), dim=-1)).sum()

        commitment_loss = F.mse_loss(code.detach(), x)
        vq_loss = self.beta * commitment_loss

        # Straight-through
        code = x + (code - x).detach()

        code = code.view(self.num_codebooks, b, h, self.code_dim)
        code = code.permute(1, 0, 2, 3).reshape(b, c, h)

        return code, vq_loss, perplexity


class FiniteScalarQuantizer(nn.Module):
    """Finite Scalar Quantization.

    Reference: https://arxiv.org/abs/2309.15505
    """

    def __init__(self, levels: List[int] = [8, 6, 5], code_dim: int = 3, eps: float = 1e-3) -> None:
        super().__init__()
        assert len(levels) == code_dim
        self._eps = eps
        self.code_dim = code_dim
        self.register_buffer("_levels", torch.as_tensor(levels, dtype=torch.int32))
        self.register_buffer("_half_width", (self._levels // 2))
        self.register_buffer(
            "_basis",
            torch.concatenate((torch.ones((1,), dtype=torch.int32), torch.cumprod(self._levels[:-1], 0))),
        )
        self.register_buffer(
            "_implicit_codebook",
            self.index_to_code(torch.arange(self.codebook_size, dtype=torch.int32)),
        )

    @property
    def codebook_size(self) -> int:
        return int(torch.prod(self._levels).item())

    def _bound(self, x: Tensor) -> Tensor:
        half_l = (self._levels - 1) * (1 - self._eps) / 2
        offset = torch.where(self._levels % 2 == 1, 0.0, 0.5)
        shift = torch.tan(offset / half_l)
        return torch.tanh(x + shift) * half_l - offset

    def _round_ste(self, z: Tensor) -> Tensor:
        zhat = torch.round(z)
        return z + (zhat - z).detach()

    def index_to_code(self, index: Tensor) -> Tensor:
        codes_non_centered = torch.remainder(torch.floor_divide(index[:, None], self._basis), self._levels)
        return (codes_non_centered - self._half_width) / self._half_width

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        assert x.dim() == 3
        b, c, h = x.shape
        x = x.permute(0, 2, 1)

        bounded_x = self._bound(x)
        quantized_x = self._round_ste(bounded_x)
        half_width = self._levels // 2
        renormalized = quantized_x / half_width

        with torch.no_grad():
            code_book = self._implicit_codebook[None]
            used_codes = renormalized.view(-1, 1, c)
            code_assignment = torch.all(code_book == used_codes, dim=-1).float()
            code_probs = code_assignment.mean(0)
            perplexity = torch.exp(-torch.sum(code_probs * torch.log(code_probs + 1e-10), dim=-1)).sum()

        renormalized = renormalized.permute(0, 2, 1)
        return renormalized, torch.zeros_like(perplexity), perplexity


class IdentityQuantizer(nn.Module):
    """Pass-through quantizer (no quantization)."""

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        device = x.device
        return x, torch.zeros(1, device=device), torch.ones(1, device=device)
