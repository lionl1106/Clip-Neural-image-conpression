"""
Evaluation metrics for image reconstruction quality.

Provides functions to compute PSNR, SSIM (via scikit-image if available), LPIPS
distance (if the `lpips` package is installed), and CLIP similarity between two
images. All functions operate on numpy arrays in range [-1, 1] with shape
(C, H, W) or (H, W, C).
"""

from __future__ import annotations
from typing import Tuple
import numpy as np
import torch


def _to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert [-1,1] float image to uint8 [0,255]."""
    x = ((img + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return x


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute PSNR between two images in range [-1,1]."""
    x1 = _to_uint8(img1)
    x2 = _to_uint8(img2)
    mse = np.mean((x1.astype(np.float32) - x2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return 20.0 * np.log10(255.0 / np.sqrt(mse))


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute SSIM using scikit-image, if available; otherwise return NaN."""
    try:
        from skimage.metrics import structural_similarity
    except Exception:
        return float('nan')
    x1 = _to_uint8(img1)
    x2 = _to_uint8(img2)
    # Convert to H,W,C
    if x1.ndim == 3 and x1.shape[0] in (1, 3):
        x1 = x1.transpose(1, 2, 0)
        x2 = x2.transpose(1, 2, 0)
    multichannel = True if x1.ndim == 3 and x1.shape[2] > 1 else False
    val = structural_similarity(x1, x2, data_range=255, channel_axis=-1 if multichannel else None)
    return float(val)


def lpips_distance(img1: np.ndarray, img2: np.ndarray, device: str = 'cpu') -> float:
    """Compute LPIPS distance; returns NaN if lpips is not installed."""
    try:
        import lpips
    except Exception:
        return float('nan')
    # Convert to tensor, normalized to [-1,1]
    t1 = torch.from_numpy(img1).float().unsqueeze(0)
    t2 = torch.from_numpy(img2).float().unsqueeze(0)
    if t1.shape[1] != 3:
        raise ValueError('LPIPS expects 3-channel images')
    loss_fn = lpips.LPIPS(net='vgg').to(device)
    d = loss_fn(t1.to(device), t2.to(device)).item()
    return float(d)


def clip_similarity(img1: np.ndarray, img2: np.ndarray, device: str = 'cpu') -> float:
    """Compute cosine similarity between CLIP image embeddings of two images."""
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    model = model.to(device).eval()
    with torch.no_grad():
        def _embed(img: np.ndarray) -> torch.Tensor:
            # Accept either (C,H,W) or (H,W,C) in [-1,1], convert to PIL then to clip preprocess
            if img.ndim == 3 and img.shape[0] in (1, 3):
                img_vis = ((_to_uint8(img)).transpose(1, 2, 0)).astype('uint8')
            else:
                img_vis = _to_uint8(img)
            from PIL import Image as _Image
            pil = _Image.fromarray(img_vis)
            tensor = preprocess(pil).unsqueeze(0).to(device)
            feat = model.encode_image(tensor).float()
            return feat / feat.norm(dim=-1, keepdim=True)
        f1 = _embed(img1)
        f2 = _embed(img2)
        sim = (f1 * f2).sum().item()
    return float(sim)