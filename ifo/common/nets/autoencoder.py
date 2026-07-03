"""
Autoencoder implementation from Black Forest Labs FLUX 1.

Source: https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/autoencoder.py

FLUX is a suite of text-to-image generation models developed by Black Forest Labs.
"""
from typing import List, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


class AttentionBlock(nn.Module):
    """Attention block using self-attention."""

    def __init__(self, in_channels: int) -> None:
        """Initialize AttentionBlock.

        Args:
            in_channels (int): Number of input channels.
        """
        super().__init__()
        self.in_channels = in_channels

        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        """Compute attention.

        Args:
            h_ (Tensor): Input tensor.

        Returns:
            Tensor: Attention output.
        """
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = rearrange(q, "b c h w -> b 1 (h w) c").contiguous()
        k = rearrange(k, "b c h w -> b 1 (h w) c").contiguous()
        v = rearrange(v, "b c h w -> b 1 (h w) c").contiguous()
        h_ = F.scaled_dot_product_attention(q, k, v)

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor with residual connection.
        """
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    """ResNet block with GroupNorm and Swish activation."""

    def __init__(self, in_channels: int, out_channels: Optional[int] = None) -> None:
        """Initialize ResnetBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int, optional): Number of output channels. Defaults to in_channels.
        """
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.silu = nn.SiLU()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        h = x
        h = self.norm1(h)
        h = self.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.silu(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)

        return x + h


class Downsample(nn.Module):
    """Downsampling layer using convolution."""

    def __init__(self, in_channels: int) -> None:
        """Initialize Downsample.

        Args:
            in_channels (int): Number of input channels.
        """
        super().__init__()
        # no asymmetric padding in torch conv, must do it ourselves
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Downsampled tensor.
        """
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        x = self.conv(x)
        return x


class Upsample(nn.Module):
    """Upsampling layer using interpolation and convolution."""

    def __init__(self, in_channels: int) -> None:
        """Initialize Upsample.

        Args:
            in_channels (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Upsampled tensor.
        """
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class Encoder(nn.Module):
    """Encoder part of the AutoEncoder."""

    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: List[int],
        num_res_blocks: int,
        z_channels: int,
    ) -> None:
        """Initialize Encoder.

        Args:
            resolution (int): Input resolution.
            in_channels (int): Number of input channels.
            ch (int): Base channel count.
            ch_mult (List[int]): Channel multipliers for each level.
            num_res_blocks (int): Number of residual blocks per level.
            z_channels (int): Number of latent channels.
        """
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.silu = nn.SiLU()

        # downsampling
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.ModuleDict()
            down["block"] = block
            down["attn"] = attn
            if i_level != self.num_resolutions - 1:
                down["downsample"] = Downsample(block_in)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.ModuleDict()
        self.mid["block_1"] = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid["attn_1"] = AttentionBlock(block_in)
        self.mid["block_2"] = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Encoded tensor.
        """
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            # Explicitly cast to ModuleDict to satisfy static analysis if needed,
            # but usually runtime works.
            down_module = cast(nn.ModuleDict, self.down[i_level])
            for i_block in range(self.num_res_blocks):
                # Accessing ModuleList from ModuleDict
                block_list = cast(nn.ModuleList, down_module["block"])
                attn_list = cast(nn.ModuleList, down_module["attn"])

                h = block_list[i_block](hs[-1])
                if len(attn_list) > 0:
                    h = attn_list[i_block](h)
                hs.append(h)
            if "downsample" in down_module:
                hs.append(down_module["downsample"](hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid["block_1"](h)
        h = self.mid["attn_1"](h)
        h = self.mid["block_2"](h)
        # end
        h = self.norm_out(h)
        h = self.silu(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    """Decoder part of the AutoEncoder."""

    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: List[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ) -> None:
        """Initialize Decoder.

        Args:
            ch (int): Base channel count.
            out_ch (int): Number of output channels.
            ch_mult (List[int]): Channel multipliers for each level.
            num_res_blocks (int): Number of residual blocks per level.
            in_channels (int): Number of input channels.
            resolution (int): Target resolution.
            z_channels (int): Number of latent channels.
        """
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)
        self.silu = nn.SiLU()

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        # z to block_in
        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.ModuleDict()
        self.mid["block_1"] = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid["attn_1"] = AttentionBlock(block_in)
        self.mid["block_2"] = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.ModuleDict()
            up["block"] = block
            up["attn"] = attn
            if i_level != 0:
                up["upsample"] = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        """Forward pass.

        Args:
            z (Tensor): Latent tensor.

        Returns:
            Tensor: Reconstructed tensor.
        """
        # get dtype for proper tracing
        upscale_dtype = next(self.up.parameters()).dtype

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid["block_1"](h)
        h = self.mid["attn_1"](h)
        h = self.mid["block_2"](h)

        # cast to proper dtype
        h = h.to(upscale_dtype)
        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                up_module = cast(nn.ModuleDict, self.up[i_level])
                block_list = cast(nn.ModuleList, up_module["block"])
                attn_list = cast(nn.ModuleList, up_module["attn"])

                h = block_list[i_block](h)
                if len(attn_list) > 0:
                    h = attn_list[i_block](h)

            up_module = cast(nn.ModuleDict, self.up[i_level])
            if "upsample" in up_module:
                h = up_module["upsample"](h)

        # end
        h = self.norm_out(h)
        h = self.silu(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussian(nn.Module):
    """Diagonal Gaussian distribution."""

    def __init__(self, sample: bool = True, chunk_dim: int = 1) -> None:
        """Initialize DiagonalGaussian.

        Args:
            sample (bool, optional): Whether to sample from the distribution. Defaults to True.
            chunk_dim (int, optional): Dimension to chunk. Defaults to 1.
        """
        super().__init__()
        self.sample = sample
        self.chunk_dim = chunk_dim

    def forward(self, z: Tensor) -> Tensor:
        """Forward pass.

        Args:
            z (Tensor): Input tensor.

        Returns:
            Tensor: Sampled tensor or mean.
        """
        mean, logvar = torch.chunk(z, 2, dim=self.chunk_dim)
        if self.sample:
            std = torch.exp(0.5 * logvar)
            return mean + std * torch.randn_like(mean)
        else:
            return mean


class AutoEncoder(nn.Module):
    """AutoEncoder model."""

    def __init__(
        self,
        resolution: int = 256,
        in_channels: int = 3,
        base_channels: int = 128,
        out_channels: int = 3,
        channel_mult: List[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        z_channels: int = 16,
        scale_factor: float = 1.0,
        shift_factor: float = 0.0,
        sample_z: bool = False,
    ) -> None:
        """Initialize AutoEncoder.

        Per default, we downsample the spatial dimension of the image by (HxW) -> (H/8xW/8).

        Args:
            resolution (int): Input resolution.
            in_channels (int): Number of input channels.
            base_channels (int): Base channel count.
            out_channels (int): Number of output channels.
            channel_mult (List[int]): Channel multipliers for each level.
            num_res_blocks (int): Number of residual blocks per level.
            z_channels (int): Number of latent channels.
            scale_factor (float): Scaling factor for latent space.
            shift_factor (float): Shift factor for latent space.
            sample_z (bool, optional): Whether to sample from the latent distribution. Defaults to False.
        """
        super().__init__()
        self.encoder = Encoder(
            resolution=resolution,
            in_channels=in_channels,
            ch=base_channels,
            ch_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
        )
        self.decoder = Decoder(
            resolution=resolution,
            in_channels=in_channels,
            ch=base_channels,
            out_ch=out_channels,
            ch_mult=channel_mult,
            num_res_blocks=num_res_blocks,
            z_channels=z_channels,
        )
        self.reg = DiagonalGaussian(sample=sample_z)

        self.scale_factor = scale_factor
        self.shift_factor = shift_factor

    def encode(self, x: Tensor) -> Tensor:
        """Encode input.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Latent representation.
        """
        z = self.reg(self.encoder(x))
        z = self.scale_factor * (z - self.shift_factor)
        return z

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent representation.

        Args:
            z (Tensor): Latent representation.

        Returns:
            Tensor: Reconstructed tensor.
        """
        z = z / self.scale_factor + self.shift_factor
        return self.decoder(z)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Reconstructed tensor.
        """
        return self.decode(self.encode(x))
