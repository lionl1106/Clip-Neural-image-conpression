
from torch import nn
import torch
import torch.nn.functional as F
from .blocks import DWConvBlock

class CLIPCondDecoder(nn.Module):
    """單一路徑上採樣；僅以 CLIP 向量作條件，與你的訓練/重建流程相容。"""
    def __init__(self, in_dim=512, base=192, out_size=512):
        super().__init__()
        self.out_size = int(out_size)
        self.fc = nn.Sequential(nn.Linear(in_dim, base*8*8), nn.GELU())
        stages, c = [], base
        while (8 * (2 ** len(stages))) < out_size:
            nxt = max(c // 2, 32)
            stages += [
                DWConvBlock(c, c),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                DWConvBlock(c, nxt)
            ]
            c = nxt
        self.up = nn.Sequential(*stages)
        self.to_img = nn.Sequential(nn.Conv2d(c, 3, 3, padding=1), nn.Tanh())
    def forward(self, z_clip):             # <── 單一參數
        b = z_clip.shape[0]
        x = self.fc(z_clip).view(b, -1, 8, 8)
        x = self.up(x)
        if x.shape[-1] != self.out_size:
            x = F.interpolate(x, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        return self.to_img(x)

class FeatureToImageDecoderLite(nn.Module):
    """
    Progressive upsampling decoder that maps a 1D CLIP feature vector to an
    image. Uses depthwise separable convs and GroupNorm.
    """
    def __init__(self, in_dim=512, base=256, out_size=64):
        super().__init__()
        self.out_size = out_size
        h = out_size // 8
        c = base
        self.fc = nn.Sequential(nn.Linear(in_dim, c*h*h), nn.GELU())

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.GroupNorm(8, cout),
                nn.GELU(),
                nn.Conv2d(cout, cout, 3, padding=1),
                nn.GroupNorm(8, cout),
                nn.GELU(),
            )
        self.up1 = block(c, c)
        self.up2 = block(c, c//2)
        self.up3 = block(c//2, c//4)
        self.to_img = nn.Sequential(nn.Conv2d(c//4, 3, 3, padding=1), nn.Tanh())

    def forward(self, z):
        B = z.shape[0]
        h = self.out_size // 8
        x = self.fc(z).view(B, -1, h, h)
        x = self.up1(x); x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up2(x); x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.up3(x); x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        img = self.to_img(x)
        return img
