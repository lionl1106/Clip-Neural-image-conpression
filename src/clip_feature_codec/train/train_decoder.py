
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn

from ..codecs.quantizer import PerChannelAffineQuantizer
from ..io.bitstream import write_bitstream, read_bitstream
from ..models.decoders import CLIPCondDecoder, FeatureToImageDecoderLite

def l2_normalize_np(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)

def total_variation(x: torch.Tensor) -> torch.Tensor:
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w

def reconstruct_image_from_bitstream(bit_path: Path, store_dir: Path, decoder: nn.Module, device: str, out_size: int = 64) -> Image.Image:
    meta = np.load(store_dir / 'codec_meta.npz')
    scale = meta['scale'].astype('float32')
    zero  = meta['zero'].astype('float32')
    q = read_bitstream(bit_path)
    z = q.astype(np.float32) * scale + zero
    z = l2_normalize_np(z[None, :]).astype(np.float32)
    with torch.no_grad():
        y = decoder(torch.from_numpy(z).to(device))
    arr = y[0].clamp(-1, 1).cpu().numpy()
    img_np = ((arr + 1.0) * 127.5).astype(np.uint8).transpose(1, 2, 0)
    return Image.fromarray(img_np)
