"""
Neural building blocks used throughout the diffusion decoder.

Contains FiLM conditioning, residual blocks, attention blocks, and depthwise separable
convolution blocks.
"""

import math
import torch
from torch import nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """Feature-wise linear modulation."""

    def __init__(self, c: int, cond_dim: int) -> None:
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, c)
        self.to_shift = nn.Linear(cond_dim, c)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        s = self.to_scale(h).unsqueeze(-1).unsqueeze(-1)
        b = self.to_shift(h).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + s) + b


class ResBlock(nn.Module):
    """Residual block with FiLM conditioning."""

    def __init__(self, c: int, cond_dim: int, groups: int = 8) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, c), c)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(groups, c), c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.film = FiLM(c, cond_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        y = self.conv1(self.act(self.norm1(x)))
        y = self.film(y, h)
        y = self.conv2(self.act(self.norm2(y)))
        return x + y


class AttnBlock(nn.Module):
    """Self-attention block conditioned on a global vector.

    Key and value vectors are derived from the conditioning vector.
    """

    def __init__(self, c: int, cond_dim: int, heads: int = 4) -> None:
        super().__init__()
        self.q = nn.Conv2d(c, c, 1)
        self.kv = nn.Linear(cond_dim, 2 * c)
        self.proj = nn.Conv2d(c, c, 1)
        self.heads = heads

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.q(x).reshape(B, self.heads, C // self.heads, H * W).transpose(-1, -2)
        kv = self.kv(h).view(B, 2, self.heads, C // self.heads).transpose(0, 1)
        k, v = kv[0], kv[1]
        attn = (q @ k.unsqueeze(-2)) / (k.size(-1) ** 0.5)
        out = (attn.softmax(dim=-2) * v.unsqueeze(-2)).transpose(-1, -2)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


class DWConvBlock(nn.Module):
    """Depthwise separable convolution block."""

    def __init__(self, cin: int, cout: int, max_groups: int = 8) -> None:
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        g = math.gcd(cout, max_groups) or 1
        self.gn = nn.GroupNorm(g, cout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.gn(self.pw(self.dw(x))))