
import math
import torch
from torch import nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self, c, cond_dim):
        super().__init__()
        self.to_scale = nn.Linear(cond_dim, c)
        self.to_shift = nn.Linear(cond_dim, c)
    def forward(self, x, h):      # x:[B,C,H,W], h:[B,cond_dim]
        s = self.to_scale(h).unsqueeze(-1).unsqueeze(-1)
        b = self.to_shift(h).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + s) + b

class ResBlock(nn.Module):
    def __init__(self, c, cond_dim, groups=8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups,c), c)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(groups,c), c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.film = FiLM(c, cond_dim)
        self.act = nn.SiLU()
    def forward(self, x, h):
        y = self.conv1(self.act(self.norm1(x)))
        y = self.film(y, h)
        y = self.conv2(self.act(self.norm2(y)))
        return x + y

class AttnBlock(nn.Module):
    def __init__(self, c, cond_dim, heads=4):
        super().__init__()
        self.q = nn.Conv2d(c, c, 1); self.kv = nn.Linear(cond_dim, 2*c)
        self.proj = nn.Conv2d(c, c, 1)
        self.heads = heads
    def forward(self, x, h):  # x:[B,C,H,W], h:[B,cond_dim]
        B,C,H,W = x.shape
        q = self.q(x).reshape(B,self.heads,C//self.heads,H*W).transpose(-1,-2)   # [B,h,HW,C//h]
        kv = self.kv(h).view(B,2,self.heads,C//self.heads).transpose(0,1)        # [2,B,h,C//h]
        k, v = kv[0], kv[1]                                                      # [B,h,C//h]
        attn = (q @ k.unsqueeze(-2)) / (k.size(-1)**0.5)                         # [B,h,HW,1]
        out = (attn.softmax(dim=-2) * v.unsqueeze(-2)).transpose(-1,-2)          # [B,h,C//h,HW]
        out = out.reshape(B,C,H,W)
        return x + self.proj(out)

class DWConvBlock(nn.Module):
    def __init__(self, cin, cout, max_groups=8):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, padding=1, groups=cin, bias=False)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        g = math.gcd(cout, max_groups) or 1  # GroupNorm 的分組需整除通道數
        self.gn = nn.GroupNorm(g, cout)
        self.act = nn.GELU()
    def forward(self, x):
        # depthwise 需滿足 groups 與 in_channels 對齊
        return self.act(self.gn(self.pw(self.dw(x))))
