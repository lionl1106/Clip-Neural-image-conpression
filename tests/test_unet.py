"""Unit tests for CLIPCondUNet."""

import torch
from clip_feature_codec.models.unet import CLIPCondUNet


def test_unet_forward() -> None:
    net = CLIPCondUNet(z_dim=512, base=64, ch_mult=(1, 2), img_ch=3)
    x = torch.randn(2, 3, 64, 64)
    z = torch.randn(2, 512)
    t = torch.randint(0, 1000, (2,))
    y = net(x, z, t)
    assert y.shape == x.shape