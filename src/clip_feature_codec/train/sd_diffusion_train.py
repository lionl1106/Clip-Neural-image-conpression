# Training loop for the SD decoder (minimal skeleton).
from __future__ import annotations
import json, math, random, os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from PIL import Image

from .models.sd_decoder import StableDiffusionDecoder  # type: ignore

def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def load_codec_meta(store_dir: Path):
    meta = np.load(store_dir / "codec_meta.npz")
    return meta["scale"].astype("float32"), meta["zero"].astype("float32"), int(meta["dim"])

def read_bitstream(p: Path) -> np.ndarray:
    raw = p.read_bytes()
    # your real repo already has zstd framing; here we assume raw uint8 for brevity
    return np.frombuffer(raw, dtype=np.uint8)

def total_variation(x: torch.Tensor) -> torch.Tensor:
    return (x[...,1:,:]-x[...,:-1,:]).abs().mean() + (x[...,:,1:]-x[...,:,:-1]).abs().mean()

class StoreDataset(torch.utils.data.Dataset):
    def __init__(self, store_dir: Path, size: int = 256):
        self.store_dir = Path(store_dir)
        self.manifest = json.loads((self.store_dir / "manifest.json").read_text(encoding="utf-8"))
        self.scale, self.zero, self.dim = load_codec_meta(self.store_dir)
        self.size = size
    def __len__(self): return len(self.manifest)
    def __getitem__(self, i: int):
        rec = self.manifest[i]
        q = read_bitstream(Path(rec["bitstream"]))
        z = q.astype("float32") * self.scale + self.zero
        z = torch.from_numpy(z).float()
        z = l2_normalize(z, dim=-1)
        img = Image.open(rec["image"]).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        x = torch.from_numpy((np.array(img).astype("float32")/127.5-1.0)).permute(2,0,1)
        return z, x

def train(store_dir: str, out: str = "sd_adapter_final.pt", steps: int = 20000, batch: int = 2, lr: float = 3e-4, size: int = 256, model_name: str = "runwayml/stable-diffusion-v1-5", device: str = "cuda"):
    ds = StoreDataset(Path(store_dir), size=size)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, num_workers=4, drop_last=True)
    dec = StableDiffusionDecoder(model_name=model_name, device=device)
    opt = torch.optim.AdamW(dec.adapter.parameters(), lr=lr)
    noise_scheduler = dec.noise_scheduler

    gstep = 0
    for epoch in range(9999):
        for z, x in dl:
            gstep += 1
            x = x.to(dec.device)
            with torch.no_grad():
                latents_0 = dec.encode(x)  # (B,4,H/8,W/8)
            # sample t
            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (x.size(0),), device=dec.device).long()
            noise = torch.randn_like(latents_0)
            alphas = noise_scheduler.alphas_cumprod.to(dec.device)[t].view(-1,1,1,1)
            latents_t = (alphas.sqrt() * latents_0 + (1 - alphas).sqrt() * noise)
            eps_hat = dec(latents_t, z.to(dec.device), t)
            loss = F.mse_loss(eps_hat, noise)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if gstep % 250 == 0:
                print(f"[train] step={gstep} loss={loss.item():.4f}")
            if gstep % 5000 == 0:
                torch.save(dec.adapter.state_dict(), out.replace(".pt", f"_{gstep}.pt"))
            if gstep >= steps:
                torch.save(dec.adapter.state_dict(), out)
                return out
