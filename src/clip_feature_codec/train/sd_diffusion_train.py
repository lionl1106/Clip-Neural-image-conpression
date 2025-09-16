# src/clip_feature_codec/train/sd_diffusion_train.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
import numpy as np
import torch, torch.nn.functional as F
import open_clip
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
    
def _clip_preprocess_torch(x: torch.Tensor, size: int = 224) -> torch.Tensor:
    """
    將 [-1,1] 影像張量轉成 CLIP 預期的正規化（0~1 → normalize）。
    不用 PIL，直接在 GPU 上做 resize + normalize。
    """
    x = (x.clamp(-1, 1) + 1.0) / 2.0                     # [0,1]
    x = F.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x.device).view(1, 3, 1, 1)
    x = (x - mean) / std
    return x

def train_sd_diffusion(
    store_dir: Path,
    out_size: int = 512,
    steps: int = 20000,
    batch_size: int = 2,
    lr: float = 3e-4,
    model_name: str = "runwayml/stable-diffusion-v1-5",
    tv_w: float = 1e-5,
    clip_w: float = 0.1,
    mse_w: float = 1.0,
    device: str = "cuda",
    save_path: Path | None = None,
) -> Path:
    ds = StoreDataset(store_dir)
    dl = DataLoader(StoreDataset(store_dir), batch_size=batch_size, shuffle=True, drop_last=True)
    dec = StableDiffusionDecoder(model_name=model_name, device=device, clip_dim=ds.dim)
    sched = dec.noise_scheduler
    opt = torch.optim.AdamW(dec.adapter.parameters(), lr=lr)

    for z, lat0 in dl:
        z = z.to(dec.device); lat0 = lat0.to(dec.device)
        t = torch.randint(0, sched.config.num_train_timesteps, (z.size(0),), device=dec.device).long()
        noise = torch.randn_like(lat0)
        a = sched.alphas_cumprod.to(dec.device)[t].view(-1,1,1,1)
        lat_t = a.sqrt()*lat0 + (1-a).sqrt()*noise
        eps_hat = dec(lat_t, z, t)
        loss = F.mse_loss(eps_hat, noise)

    # === CLIP 僅作特徵、也不更新權重（仍需保留反傳路徑，故不要 no_grad 包住） ===
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
    clip_model = clip_model.to(dec.device).eval()
    clip_model.requires_grad_(False)

    gstep = 0
    save_path = Path(save_path or "sd_adapter_final.pt")
    use_amp = (str(device).startswith("cuda") and torch.cuda.is_available())

    while gstep < steps:
        for z, lat0 in dl:                                     # ← DataLoader 直接給 (z, lat0)
            gstep += 1
            z    = z.to(dec.device, non_blocking=True)
            lat0 = lat0.to(dec.device, non_blocking=True)      # (B, 4, H/8, W/8) 並且已乘 scaling_factor

            bs = lat0.size(0)
            t = torch.randint(0, sched.config.num_train_timesteps, (bs,), device=dec.device).long()
            noise  = torch.randn_like(lat0)
            alphas = sched.alphas_cumprod.to(dec.device)[t].view(-1,1,1,1)

            lat_t  = alphas.sqrt() * lat0 + (1 - alphas).sqrt() * noise

            with torch.autocast(device_type="cuda", enabled=use_amp):
                eps_hat = dec(lat_t, z, t)
                loss_mse = F.mse_loss(eps_hat, noise)

                # 從 eps_hat 反推 x0 的 latent → 解碼到像素空間（供 TV / CLIP 用）
                lat_x0  = (lat_t - (1 - alphas).sqrt() * eps_hat) / alphas.sqrt()
                x0_pred = dec.decode(lat_x0)                    # 解碼時會自動 / scaling_factor

                loss_tv = _total_variation(x0_pred) if tv_w > 0 else x0_pred.new_tensor(0.0)

                loss_clip = x0_pred.new_tensor(0.0)
                if clip_w > 0:
                    img_in = _clip_preprocess_torch(x0_pred)    # 224 & CLIP normalize
                    y_clip = clip_model.encode_image(img_in)
                    y_clip = y_clip / (y_clip.norm(dim=-1, keepdim=True) + 1e-9)
                    z_clip = z / (z.norm(dim=-1, keepdim=True) + 1e-9)
                    loss_clip = (1.0 - torch.cosine_similarity(y_clip.float(), z_clip.float()).mean())

                loss = mse_w * loss_mse + clip_w * loss_clip + tv_w * loss_tv

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if gstep % 100 == 0:
                print(f"[train] step={gstep}/{steps} "
                      f"total={loss.item():.4f}  mse={loss_mse.item():.4f}  "
                      f"clip={loss_clip.item():.4f}  tv={loss_tv.item():.6f}")
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
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tv_w", type=float, default=1e-5)
    ap.add_argument("--clip_w", type=float, default=0.1)
    ap.add_argument("--mse_w", type=float, default=1.0)
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
        clip_w=args.clip_w,
        mse_w=args.mse_w,
        device=args.device,
        save_path=args.out,
    )
