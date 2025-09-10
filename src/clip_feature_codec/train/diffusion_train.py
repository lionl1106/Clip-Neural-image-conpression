"""
Training loop for the diffusion-based decoder.

This script trains `CLIPCondUNet` to predict the noise ε given a noisy image `x_t`,
CLIP embedding `z_clip`, and timestep `t`. The training data is loaded from
the store of quantized CLIP features and their corresponding images; each
sample is augmented by sampling a random timestep and adding Gaussian noise.
Optional reconstruction, TV, and CLIP alignment losses can be included.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Optional
import json
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import open_clip

from ..io.bitstream import read_bitstream
from ..codecs.quantizer import PerChannelAffineQuantizer
from ..diffusion.scheduler import NoiseScheduler
from ..models.unet import CLIPCondUNet


def l2_normalize_np(x: np.ndarray, axis: int = -1, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


class StoreDataset(Dataset):
    """Dataset loading images and corresponding CLIP embeddings from a store."""

    def __init__(self, store_dir: Path, out_size: int = 256) -> None:
        self.store_dir = Path(store_dir)
        self.manifest = json.loads((self.store_dir / 'manifest.json').read_text(encoding='utf-8'))
        meta = np.load(self.store_dir / 'codec_meta.npz')
        self.scale = meta['scale'].astype('float32')
        self.zero = meta['zero'].astype('float32')
        self.out_size = out_size

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, i: int):
        rec = self.manifest[i]
        q = read_bitstream(Path(rec['bitstream']))
        z = q.astype(np.float32) * self.scale + self.zero
        z = l2_normalize_np(z[None, :]).astype(np.float32).squeeze(0)
        img = Image.open(rec['image']).convert('RGB').resize((self.out_size, self.out_size), Image.BICUBIC)
        arr = (np.array(img).astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)
        return torch.from_numpy(arr), torch.from_numpy(z)


def total_variation(x: torch.Tensor) -> torch.Tensor:
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w


def train_diffusion(
    store_dir: Path,
    out_size: int = 256,
    epochs: int = 40,
    batch_size: int = 8,
    lr: float = 2e-4,
    timesteps: int = 1000,
    schedule: str = 'cosine',
    recon_w: float = 0.05,
    clip_w: float = 0.1,
    tv_w: float = 1e-4,
    device: str = 'cuda',
    save_dir: Optional[Path] = None,
) -> Path:
    """Train the diffusion decoder on all samples from the store.

    Args:
        store_dir: Directory containing `manifest.json` and bitstreams.
        out_size: Training/resizing resolution.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate for AdamW.
        timesteps: Number of diffusion timesteps.
        schedule: Beta schedule ('cosine' or 'linear').
        recon_w: Weight of reconstruction L1 loss.
        clip_w: Weight of CLIP alignment loss (computed every other epoch).
        tv_w: Weight of total variation regularization.
        device: Device string ('cuda' or 'cpu').
        save_dir: Optional directory to save checkpoints (defaults to store_dir).

    Returns:
        Path to the final model checkpoint.
    """
    save_dir = Path(save_dir or store_dir)
    ds = StoreDataset(store_dir, out_size=out_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    z_dim = ds[0][1].numel()
    net = CLIPCondUNet(z_dim=z_dim, base=128, ch_mult=(1, 2, 2), img_ch=3).to(device)
    sch = NoiseScheduler(timesteps=timesteps, schedule=schedule, device=device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))
    autocast_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    torch.backends.cuda.matmul.allow_tf32 = True
    net.train()
    # CLIP model for alignment loss
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    clip_model = clip_model.to(device).eval()
    for ep in range(epochs):
        running = 0.0
        for x0, z in tqdm(dl, desc=f'epoch {ep+1}/{epochs}'):
            x0 = x0.to(device)
            z = z.to(device)
            b = x0.size(0)
            t = torch.randint(0, timesteps, (b,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            with torch.autocast(device_type='cuda' if device == 'cuda' else 'cpu', dtype=autocast_dtype):
                x_t = sch.q_sample(x0, t, noise)
                eps_hat = net(x_t, z, t)
                loss = F.mse_loss(eps_hat, noise)
                x0_pred = sch.predict_x0_from_eps(x_t, t, eps_hat).clamp(-1, 1)
                if recon_w > 0:
                    loss = loss + recon_w * F.l1_loss(x0_pred, x0)
                if tv_w > 0:
                    loss = loss + tv_w * total_variation(x0_pred)
                if clip_w > 0 and (ep % 2 == 0):
                    with torch.no_grad():
                        y_clip = clip_model.encode_image(F.interpolate(x0_pred.float(), size=224, mode='bilinear', align_corners=False))
                        y_clip = y_clip / y_clip.norm(dim=-1, keepdim=True)
                        z_clip = z / z.norm(dim=-1, keepdim=True)
                        z_clip = z_clip.to(y_clip.dtype)
                    loss = loss + clip_w * (1.0 - torch.cosine_similarity(y_clip.float(), z_clip.float()).mean())
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            running += float(loss.detach().cpu()) * b
        # Save intermediate checkpoint
        ckpt_path = save_dir / f'diffusion_unet_ep{ep+1}.pt'
        torch.save(net.state_dict(), ckpt_path)
        print(f'[train] epoch {ep+1}/{epochs} loss={running/len(ds):.4f}')
    final_path = save_dir / 'diffusion_unet_final.pt'
    torch.save(net.state_dict(), final_path)
    return final_path