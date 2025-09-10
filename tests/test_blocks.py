
from clip_feature_codec.models.blocks import FiLM, ResBlock, AttnBlock, DWConvBlock
import torch

def test_film_shapes():
    x = torch.randn(2, 16, 8, 8)
    h = torch.randn(2, 32)
    film = FiLM(16, 32)
    y = film(x, h)
    assert y.shape == x.shape
