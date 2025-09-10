"""
Per-channel int8 quantizer for CLIP embeddings.

The quantizer learns a linear scale and offset for each embedding dimension
so that float32 vectors can be quantized to unsigned 8-bit integers.
"""

from __future__ import annotations
import numpy as np
import torch


class PerChannelAffineQuantizer:
    """Affine per-channel quantizer."""

    def __init__(self, num_bits: int = 8, eps: float = 1e-8) -> None:
        self.num_bits = num_bits
        self.eps = eps
        self.scale: torch.Tensor | None = None
        self.zero: torch.Tensor | None = None

    def fit(self, X: torch.Tensor) -> "PerChannelAffineQuantizer":
        xmin = X.min(dim=0).values
        xmax = X.max(dim=0).values
        self.scale = (xmax - xmin).clamp_min(self.eps) / (2 ** self.num_bits - 1)
        self.zero = xmin
        return self

    def encode(self, x: torch.Tensor) -> np.ndarray:
        if self.scale is None or self.zero is None:
            raise RuntimeError("Quantizer has not been fitted.")
        q = torch.round((x - self.zero) / self.scale).clamp(0, 2 ** self.num_bits - 1)
        return q.to(torch.uint8).cpu().numpy()

    def decode(self, q: np.ndarray) -> np.ndarray:
        if self.scale is None or self.zero is None:
            raise RuntimeError("Quantizer has not been fitted.")
        qf = torch.from_numpy(q.astype(np.float32))
        x = qf * self.scale + self.zero
        return x.cpu().numpy()