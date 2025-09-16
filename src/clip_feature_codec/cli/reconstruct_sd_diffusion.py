# src/clip_feature_codec/cli/reconstruct_sd_diffusion.py
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, torch
from PIL import Image

from ..models.sd_decoder import StableDiffusionDecoder  # type: ignore
from ..io.bitstream import read_bitstream

# ---- CLIP preprocessing（OpenAI CLIP 的 mean/std）----
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def load_codec_meta(store_dir: Path):
    meta = np.load(store_dir / "codec_meta.npz")
    scale = meta["scale"].astype("float32")
    zero  = meta["zero"].astype("float32")
    dim   = int(meta["dim"]) if "dim" in meta.files else int(scale.shape[0])
    return scale, zero, dim

def l2_normalize_np(x, axis=-1, eps=1e-9):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)

def _clip_preprocess_torch(x: torch.Tensor, size: int = 224) -> torch.Tensor:
    # x: [-1,1] BCHW → CLIP normalize
    x = (x.clamp(-1,1) + 1.0) / 2.0
    x = torch.nn.functional.interpolate(x, size=(size, size), mode='bilinear', align_corners=False)
    mean = torch.tensor(_CLIP_MEAN, device=x.device).view(1,3,1,1)
    std  = torch.tensor(_CLIP_STD,  device=x.device).view(1,3,1,1)
    return (x - mean) / std

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store_dir", type=Path, required=True)
    ap.add_argument("--bitstream", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True, help="trained adapter .pt")
    ap.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--out", type=Path, default=Path("recon.png"))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--guidance", type=float, default=5.0)  # CFG scale
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda")

    # ---- 新增：inversion / CLIP 引導的選項 ----
    ap.add_argument("--inv_weight", type=float, default=1.0,
                    help="CLIP 特徵引導的步長（0 代表不啟用 inversion 引導）")
    ap.add_argument("--inv_every", type=int, default=1,
                    help="每隔多少個 timestep 做一次 CLIP 引導（預設每步都做）")
    ap.add_argument("--inv_clip_arch", type=str, default="ViT-B-32",
                    help="open_clip 影像編碼器架構（需和 bitstream 的 CLIP 一致或相近）")
    ap.add_argument("--inv_clip_ckpt", type=str, default="openai",
                    help="open_clip 的 pretrained 名稱（如 openai / laion2b_s34b_b79k 等）")
    args = ap.parse_args()

    # 1) 讀 meta + 正確解碼 bitstream
    scale, zero, dim = load_codec_meta(args.store_dir)
    q = read_bitstream(args.bitstream)
    if q.shape[0] != dim:
        raise ValueError(f"Bitstream dim {q.shape[0]} != meta dim {dim} "
                         f"(scale.shape={scale.shape}, zero.shape={zero.shape})")

    # 2) 反量化 + L2 正規化
    z = q.astype("float32") * scale + zero
    z = l2_normalize_np(z[None, :]).astype("float32")   # (1, dim)

    # 3) 建立 decoder（VAE/UNet 凍結）、載入 adapter
    dec = StableDiffusionDecoder(model_name=args.model_name, device=str(args.device), clip_dim=dim)
    dec.adapter.load_state_dict(torch.load(args.adapter, map_location=dec.device))
    unet, vae, scheduler = dec.unet, dec.vae, dec.noise_scheduler
    unet.eval(); vae.eval()

    # 4) 準備 CLIP 編碼器（只在 inv_weight>0 才建）
    inv_use = args.inv_weight > 0
    if inv_use:
        import open_clip
        clip_model, _, _ = open_clip.create_model_and_transforms(args.inv_clip_arch, pretrained=args.inv_clip_ckpt)
        clip_model = clip_model.to(dec.device).eval()

        def _clip_encode_img(x_img: torch.Tensor) -> torch.Tensor:
            x_in = _clip_preprocess_torch(x_img)             # BCHW
            y = clip_model.encode_image(x_in)
            return y / (y.norm(dim=-1, keepdim=True) + 1e-9)

        z_tgt = torch.from_numpy(z).to(dec.device)           # (1, dim)
        z_tgt = z_tgt / (z_tgt.norm(dim=-1, keepdim=True) + 1e-9)

    # 5) 展開 DDIM 抽樣 + CFG +（可選）CLIP 引導反饋
    B, C, H, W = 1, 4, args.size // 8, args.size // 8
    scheduler.set_timesteps(args.steps, device=dec.device)
    lat = torch.randn((B, C, H, W), device=dec.device)

    cond   = dec.adapter(torch.from_numpy(z).to(dec.device))                 # (1, S, D)
    uncond = dec.adapter(torch.zeros_like(torch.from_numpy(z)).to(dec.device))

    with torch.set_grad_enabled(inv_use):
        for i, t in enumerate(scheduler.timesteps):
            lat.requires_grad_(inv_use)

            # (a) 噪聲預測 + CFG（不需要梯度）
            with torch.no_grad():
                eps_u = unet(lat, t, encoder_hidden_states=uncond).sample
                eps_c = unet(lat, t, encoder_hidden_states=cond).sample
                eps   = eps_u + args.guidance * (eps_c - eps_u)  # classifier-free guidance【Dhariwal & Nichol + Ho 等】 

            # (b) 可選：用當前 \hat{x}_0 做 CLIP 特徵引導（“instance-level codec guidance”）
            if inv_use and (i % max(1, args.inv_every) == 0):
                # 先用 Tweedie 公式/標準反推得到 \hat{x}_0 的 latent，再經 VAE decode 成像【DDIM 理論】
                a_t = scheduler.alphas_cumprod.to(dec.device)[t].view(1,1,1,1)
                lat_x0 = (lat - (1 - a_t).sqrt() * eps.detach()) / a_t.sqrt()
                x0_img = vae.decode((lat_x0 / dec.scaling_factor)).sample.clamp(-1,1)

                # CLIP 相似度作為“分類器”→ 以梯度上升最大化 cos 相似（這就是 CLIP-guided diffusion 的精神）
                y_clip = _clip_encode_img(x0_img)
                clip_loss = (1.0 - torch.cosine_similarity(y_clip.float(), z_tgt.float()).mean())
                g = torch.autograd.grad(clip_loss, lat, retain_graph=False)[0]   # dL/d(lat)
                # 步長用 inv_weight，並做單位化避免爆炸
                lat = (lat - args.inv_weight * g / (g.norm() + 1e-8)).detach()

            # (c) DDIM 前進一步
            lat = scheduler.step(eps, t, lat, eta=args.eta).prev_sample

    # 6) 輸出影像
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(str(args.device).startswith("cuda") and torch.cuda.is_available())):
        img = vae.decode((lat / dec.scaling_factor)).sample
    out = (img[0].clamp(-1,1).add(1).mul(127.5).to(torch.uint8).permute(1,2,0).cpu().numpy())
    Image.fromarray(out).save(args.out)
    print("Saved to", args.out)

if __name__ == "__main__":
    main()
