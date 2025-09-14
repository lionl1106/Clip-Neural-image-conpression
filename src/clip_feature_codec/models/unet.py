"""
FiLM-conditioned U-Net for diffusion decoding.

`CLIPCondUNet` takes a noisy image `x_t`, a CLIP embedding `z_clip`, and a discrete timestep
`t`, and predicts the noise ε at that timestep. The conditioning is formed by
summing a learned projection of the sinusoidal timestep embedding and a linear
projection of the CLIP embedding; this combined vector modulates each residual
block via FiLM. The architecture is deliberately compact to run on
consumer GPUs, but can be scaled via `base` and `ch_mult` parameters.
"""

from __future__ import annotations
import math
from typing import Tuple, List
import torch
from torch import nn
import torch.nn.functional as F

from .blocks import ResBlock, FiLM


def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Compute sinusoidal timestep embeddings.

    Args:
        t: (B,) integer timesteps in [0, T-1]
        dim: embedding dimension
        max_period: frequency scaling factor

    Returns:
        (B, dim) embedding vectors
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, device=t.device) / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class CLIPCondUNet(nn.Module):
    """A FiLM-conditioned U-Net for diffusion decoding."""

    def __init__(self, z_dim: int = 512, base: int = 128, ch_mult: Tuple[int, ...] = (1, 2, 2), time_dim: int = 256, img_ch: int = 3) -> None:
        super().__init__()
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4), nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        self.z_proj = nn.Sequential(
            nn.Linear(z_dim, time_dim), nn.SiLU()
        )
        cond_dim = time_dim
        self.in_conv = nn.Conv2d(img_ch, base, 3, padding=1)
        # Downsampling blocks
        downs = []
        ch = base
        self.down_chs: List[int] = [ch]
        for m in ch_mult:
            downs.append(ResBlock(ch, cond_dim))
            downs.append(ResBlock(ch, cond_dim))
            downs.append(nn.Conv2d(ch, ch * m, 3, stride=2, padding=1))
            ch = ch * m
            self.down_chs.append(ch)
        self.down = nn.ModuleList(downs)
        # Middle blocks
        self.mid1 = ResBlock(ch, cond_dim)
        self.mid2 = ResBlock(ch, cond_dim)
        # Upsampling blocks
        ups = []
        for m in reversed(ch_mult):
            ups.append(ResBlock(ch, cond_dim))
            ups.append(ResBlock(ch, cond_dim))
            ups.append(nn.ConvTranspose2d(ch, ch // m, 4, stride=2, padding=1))
            ch = ch // m
        self.up = nn.ModuleList(ups)
        self.out_norm = nn.GroupNorm(8, ch)
        self.out = nn.Conv2d(ch, img_ch, 3, padding=1)

    def forward(self, x_t: torch.Tensor, z_clip: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Compute conditioning vector (timestep embedding + z embedding)
        temb = timestep_embedding(t, self.time_proj[0].in_features).to(x_t.dtype)
        temb = self.time_proj(temb)
        zemb = self.z_proj(z_clip)
        h = temb + zemb
        # Down path
        x = self.in_conv(x_t)
        skips = []
        for i in range(0, len(self.down), 3):
            x = self.down[i](x, h)
            x = self.down[i + 1](x, h)
            skips.append(x)
            x = self.down[i + 2](x)
        # Middle
        x = self.mid1(x, h)
        x = self.mid2(x, h)
        # Up path
        for i in range(0, len(self.up), 3):
            x = self.up[i](x, h)
            x = self.up[i + 1](x, h)
            x = self.up[i + 2](x)
            if skips:
                x = x + skips.pop()
        x = self.out(self.out_norm(x).to(x.dtype))
        return x