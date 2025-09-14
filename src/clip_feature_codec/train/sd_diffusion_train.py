# src/clip_feature_codec/train/sd_diffusion_train.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from ..models.sd_decoder import StableDiffusionDecoder
from ..io.bitstream import read_bitstream

def _l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-9) -> torch.Tensor:
    return x / (x.norm(dim=dim, keepdim=True) + eps)

def _total_variation(x: torch.Tensor) -> torch.Tensor:
    return (x[...,1:,:]-x[...,:-1,:]).abs().mean() + (x[...,:,1:]-x[...,:,:-1]).abs().mean()

def _load_meta(store_dir: Path):
    meta = np.load(store_dir / "codec_meta.npz")
    scale = meta["scale"].astype("float32")
    zero  = meta["zero"].astype("float32")
    dim   = int(meta["dim"]) if "dim" in meta.files else int(scale.shape[0])
    return scale, zero, dim

class StoreDataset(Dataset):
    def __init__(self, store_dir: Path, size: int = 256):
        self.store_dir = Path(store_dir)
        self.manifest = json.loads((self.store_dir / "manifest.json").read_text(encoding="utf-8"))
        self.scale, self.zero, self.dim = _load_meta(self.store_dir)
        self.size = int(size)
    def __len__(self): return len(self.manifest)
    def __getitem__(self, i: int):
        rec = self.manifest[i]
        q = read_bitstream(Path(rec["bitstream"]))
        if q.shape[0] != self.dim:
            raise ValueError(f"bitstream dim {q.shape[0]} != meta dim {self.dim}")
        z = q.astype("float32") * self.scale + self.zero  # (dim,)
        z = torch.from_numpy(z).float()
        z = _l2_normalize(z, dim=-1)
        img = Image.open(rec["image"]).convert("RGB").resize((self.size, self.size), Image.BICUBIC)
        x = torch.from_numpy((np.array(img).astype("float32")/127.5-1.0)).permute(2,0,1)  # [-1,1]
        return z, x

def train_sd_diffusion(
    store_dir: Path,
    out_size: int = 256,
    steps: int = 20000,
    batch_size: int = 2,
    lr: float = 3e-4,
    model_name: str = "runwayml/stable-diffusion-v1-5",
    tv_w: float = 0.0,
    device: str = "cuda",
    save_path: Optional[Path] = None,
) -> Path:
    ds = StoreDataset(store_dir, size=out_size)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True, pin_memory=True)

    # 以 meta 的維度初始化 adapter，使 cross-attn 對齊
    dec = StableDiffusionDecoder(model_name=model_name, device=device, clip_dim=ds.dim)
    sched = dec.noise_scheduler
    opt = torch.optim.AdamW(dec.adapter.parameters(), lr=lr)

    gstep = 0
    save_path = Path(save_path or "sd_adapter_final.pt")
    while gstep < steps:
        for z, x in dl:
            gstep += 1
            x = x.to(dec.device)
            z = z.to(dec.device)
            with torch.no_grad():
                lat0 = dec.encode(x)                                  # (B,4,H/8,W/8)
            t = torch.randint(0, sched.config.num_train_timesteps, (x.size(0),), device=dec.device).long()
            noise = torch.randn_like(lat0)
            alphas = sched.alphas_cumprod.to(dec.device)[t].view(-1,1,1,1)   # DDIM/DDPM 共同用法
            lat_t = alphas.sqrt()*lat0 + (1 - alphas).sqrt()*noise
            eps_hat = dec(lat_t, z, t)
            loss = F.mse_loss(eps_hat, noise)
            if tv_w > 0:
                with torch.no_grad():
                    lat_x0 = (lat_t - (1 - alphas).sqrt()*eps_hat) / alphas.sqrt()
                    x0_pred = dec.decode(lat_x0).clamp(-1,1)
                loss = loss + tv_w * _total_variation(x0_pred)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if gstep % 100 == 0:
                print(f"[train] step={gstep}/{steps} loss={loss.item():.4f}")
            if gstep % 5000 == 0:
                tmp = save_path.with_name(save_path.stem + f"_{gstep}.pt")
                torch.save(dec.adapter.state_dict(), tmp)
            if gstep >= steps:
                break

    torch.save(dec.adapter.state_dict(), save_path)
    print("Saved adapter to", save_path)
    return save_path

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Train SD-UNet + CLIP-Adapter decoder (latent diffusion).")
    ap.add_argument("--store_dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tv_w", type=float, default=0.0)
    ap.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=Path, default=Path("sd_adapter_final.pt"))
    args = ap.parse_args()
    train_sd_diffusion(
        store_dir=args.store_dir,
        out_size=args.size,
        steps=args.steps,
        batch_size=args.batch,
        lr=args.lr,
        model_name=args.model_name,
        tv_w=args.tv_w,
        device=args.device,
        save_path=args.out,
    )
