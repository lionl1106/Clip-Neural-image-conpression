"""
Training loop for the Stable Diffusion–based decoder.

This script trains the ``StableDiffusionDecoder`` to predict the noise
present in latent representations of images given a CLIP embedding and
a diffusion timestep.  The model architecture is defined in
``clip_feature_codec.models.sd_decoder`` and consists of a frozen
variational autoencoder and UNet taken from a pre‑trained Stable
Diffusion checkpoint, together with a small trainable adapter that
maps CLIP image embeddings into the UNet’s cross‑attention space.

Compared to ``diffusion_train.py``, which operates in pixel space,
training in latent space dramatically reduces memory requirements and
leverages the powerful prior learned by Stable Diffusion.  The loss
function comprises a mean squared error on the predicted noise plus
optional reconstruction, total variation and CLIP alignment terms.  Only
the adapter parameters are updated during training; all other weights
remain fixed.

Usage example::

    python -m clip_feature_codec.train.sd_diffusion_train \
      --store_dir path/to/store \
      --out_size 256 \
      --epochs 20 \
      --batch_size 4 \
      --model_name runwayml/stable-diffusion-v1-5

"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import timm
from timm.data import resolve_model_data_config, create_transform

from ..models.sd_decoder import StableDiffusionDecoder
from ..io.bitstream import read_bitstream  # local import to avoid cycle

class StoreDataset(Dataset):
    def __init__(self, store_dir: Path):
        self.store = Path(store_dir)
        self.manifest = json.loads((self.store / "manifest_latents.json").read_text())
        meta = np.load(self.store / "codec_meta.npz")
        self.scale = meta["scale"].astype("float32")
        self.zero  = meta["zero"].astype("float32")
        self.dim   = int(meta["dim"]) if "dim" in meta.files else int(self.scale.shape[0])
    def __len__(self): return len(self.manifest)
    def __getitem__(self, i: int):
        rec = self.manifest[i]
        q = read_bitstream(Path(rec["bitstream"]))
        if q.shape[0] != self.dim: raise ValueError("dim mismatch")
        z = torch.from_numpy((q.astype("float32")*self.scale+self.zero)).float()
        z = z / (z.norm(dim=-1, keepdim=True)+1e-9)
        lat = torch.from_numpy(np.load(rec["latent"])["lat"]).float()  # shape (4, H/8, W/8)
        return z, lat


def total_variation(x: torch.Tensor) -> torch.Tensor:
    """Compute isotropic total variation for a batch of images."""
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return tv_h + tv_w


def train_sd_diffusion(
    store_dir: Path,
    out_size: int = 256,
    epochs: int = 20,
    batch_size: int = 4,
    lr: float = 1e-4,
    timesteps: int = 1000,
    model_name: str = "runwayml/stable-diffusion-v1-5",
    recon_w: float = 0.05,
    clip_w: float = 0.1,
    tv_w: float = 1e-4,
    device: str = 'cuda',
    save_dir: Optional[Path] = None,
) -> Path:
    """Train the latent diffusion decoder on all samples from the store.

    Args:
        store_dir: Directory containing ``manifest.json`` and bitstreams.
        out_size: Training/resizing resolution.
        epochs: Number of training epochs.
        batch_size: Mini‑batch size.
        lr: Learning rate for AdamW.
        timesteps: Number of diffusion timesteps.  Must match the
            scheduler’s ``num_train_timesteps``.
        model_name: Hugging Face name or path of the Stable Diffusion
            checkpoint to load.
        recon_w: Weight of reconstruction L1 loss (in pixel space).
        clip_w: Weight of CLIP alignment loss (computed every other
            epoch).
        tv_w: Weight of total variation regularization (in pixel space).
        device: Device string (``'cuda'`` or ``'cpu'``).
        save_dir: Optional directory to save checkpoints; defaults to
            ``store_dir``.

    Returns:
        Path to the final adapter checkpoint.
    """
    save_dir = Path(save_dir or store_dir)
    ds = StoreDataset(store_dir)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    # Instantiate decoder
    clip_dim = int(ds.zero.shape[0])
    dec = StableDiffusionDecoder(model_name=model_name, device=device, clip_dim=clip_dim).to(device)
    # Ensure scheduler timesteps match requested training timesteps
    # dec.scheduler.set_timesteps(timesteps)
    sched = dec.noise_scheduler
    
    # Only train the adapter parameters
    opt = torch.optim.AdamW(dec.adapter.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))
    autocast_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    # ----------------------------------------------------------------------
    # DINOv2 model for perceptual alignment
    #
    # In place of the OpenAI CLIP model used previously, we load a
    # pre‑trained DINOv2 vision transformer from the timm library.  The
    # ``vit_base_patch14_dinov2.lvd142m`` checkpoint provides a 768‑dimensional
    # embedding that captures robust visual features without supervision.
    # We configure the model with ``num_classes=0`` to obtain the pooled
    # embeddings directly.  The associated data configuration supplies the
    # mean, standard deviation and input size for normalising images.
    dino_model = timm.create_model(
        'vit_base_patch14_dinov2.lvd142m',
        pretrained=True,
        num_classes=0
    ).to(device).eval()
    # Resolve data config for pre‑processing
    dino_cfg = resolve_model_data_config(dino_model)
    # Convert mean and std to tensors for broadcasting during normalisation
    dino_mean = torch.tensor(dino_cfg['mean'], device=device).view(1, 3, 1, 1)
    dino_std = torch.tensor(dino_cfg['std'], device=device).view(1, 3, 1, 1)
    # Input spatial size expected by DINOv2 (e.g. (224, 224) for base models)
    dino_size = (dino_cfg['input_size'][-2], dino_cfg['input_size'][-1])
    # Precompute alpha_bar schedule for latent diffusion
    al_bar = sched.alphas_cumprod.to(dec.device)
    # al_bar = dec.scheduler.alphas_cumprod.to(device)  # shape (T,)
    for ep in range(epochs):
        for a, b in tqdm(dl, desc=f'epoch {ep+1}/{epochs}'):
            # ---- 1) 判斷誰是 z、誰是 影像/潛向量 ----
            def _is_z(x):   # DINO/CLIP 特徵通常 (B, D)（2 維）
                return x.dim() == 2

            if _is_z(a) and not _is_z(b):
                z, img_or_lat = a, b
            elif _is_z(b) and not _is_z(a):
                z, img_or_lat = b, a
            else:
                raise ValueError(
                    f"Dataset batch ambiguous: got shapes a={tuple(a.shape)} b={tuple(b.shape)}; "
                    "expect one 2D feature (B,D) and one 4D image/latent (B,C,H,W)."
                )

            z = z.to(device, non_blocking=True)

            # ---- 2) 取得 lat0：若輸入是影像 C=3 則 encode；若 C=4 視為已是潛向量 ----
            if img_or_lat.dim() != 4:
                raise ValueError(f"Expected BCHW tensor for image/latent, got {tuple(img_or_lat.shape)}")

            C = img_or_lat.size(1)
            if C == 4:
                lat0 = img_or_lat.to(device, non_blocking=True)                    # 已是潛向量
            elif C == 3:
                with torch.no_grad():                                              # 影像 → 潛向量
                    lat0 = dec.encode(img_or_lat.to(device, non_blocking=True))
            else:
                raise ValueError(f"Channel C must be 3 (image) or 4 (latent), got C={C}")

            bsz = lat0.size(0)

            # ---- 3) 前向加噪 ----
            t = torch.randint(0, timesteps, (bsz,), device=device, dtype=torch.long)
            noise = torch.randn_like(lat0)
            sqrt_al = al_bar[t].sqrt().view(-1,1,1,1)
            sqrt_one_minus_al = (1.0 - al_bar[t]).sqrt().view(-1,1,1,1)
            lat_t = sqrt_al * lat0 + sqrt_one_minus_al * noise

            # ---- 4) 預測噪聲 + loss（MSE / TV / DINO 語義）----
            with torch.autocast(device_type=('cuda' if device=='cuda' else 'cpu'), dtype=autocast_dtype):
                eps_hat = dec(lat_t, z, t)                         # (B,4,H/8,W/8)

                loss = F.mse_loss(eps_hat, noise)                  # ε MSE（Ho et al.）

                if recon_w > 0 or tv_w > 0 or clip_w > 0:
                    lat_x0 = (lat_t - sqrt_one_minus_al * eps_hat) / sqrt_al
                    x0_pred = dec.decode(lat_x0).clamp(-1, 1)      # [-1,1]

                if recon_w > 0:
                    with torch.no_grad():
                        x0_ref = dec.decode(lat0).clamp(-1, 1)     # 以 lat0 解碼當 GT，不需原圖
                    loss = loss + recon_w * F.l1_loss(x0_pred, x0_ref)

                if tv_w > 0:
                    loss = loss + tv_w * total_variation(x0_pred)

                if clip_w > 0 and (ep % 2 == 0):
                    x_dino = (x0_pred.float() + 1.0) / 2.0
                    x_dino = F.interpolate(x_dino, size=dino_size, mode='bilinear', align_corners=False)
                    x_dino = (x_dino - dino_mean) / dino_std
                    y_dino = dino_model(x_dino).float()
                    y_dino = y_dino / (y_dino.norm(dim=-1, keepdim=True) + 1e-9)
                    z_norm = z / (z.norm(dim=-1, keepdim=True) + 1e-9)
                    z_norm = z_norm.to(y_dino.dtype)
                    loss = loss + clip_w * (1.0 - torch.cosine_similarity(y_dino, z_norm).mean())

            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)

        # Save checkpoint at the end of each epoch
        ckpt_path = save_dir / f'sd_adapter_ep{ep+1}.pt'
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'adapter': dec.adapter.state_dict()}, ckpt_path)
    # Save final adapter weights
    final_path = save_dir / 'sd_adapter_final.pt'
    torch.save({'adapter': dec.adapter.state_dict()}, final_path)
    return final_path


if __name__ == '__main__':  # pragma: no cover
    import argparse
    ap = argparse.ArgumentParser(description="Train StableDiffusionDecoder with CLIP conditioning.")
    ap.add_argument("--store_dir", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--out_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--recon_w", type=float, default=0.05)
    ap.add_argument("--clip_w", type=float, default=0.1)
    ap.add_argument("--tv_w", type=float, default=1e-4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_dir", type=str, default=None)
    args = ap.parse_args()
    train_sd_diffusion(
        store_dir=Path(args.store_dir),
        out_size=args.out_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        timesteps=args.timesteps,
        model_name=args.model_name,
        recon_w=args.recon_w,
        clip_w=args.clip_w,
        tv_w=args.tv_w,
        device=args.device,
        save_dir=Path(args.save_dir) if args.save_dir is not None else None,
    )