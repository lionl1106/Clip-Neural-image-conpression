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

def _load_adapter_state(path, device):
    ckpt = torch.load(path, map_location=device)  # 可選: weights_only=True（PyTorch 2.0+）
    # 1) 解包常見容器
    if isinstance(ckpt, dict) and 'adapter' in ckpt and isinstance(ckpt['adapter'], (dict, torch.nn.modules.container.OrderedDict)):
        sd = ckpt['adapter']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt and isinstance(ckpt['state_dict'], (dict, torch.nn.modules.container.OrderedDict)):
        sd = ckpt['state_dict']
    else:
        sd = ckpt

    # 2) 移除常見前綴（DDP/包裝）
    def strip_prefix(d, pfx):
        if any(k.startswith(pfx) for k in d.keys()):
            return { (k[len(pfx):] if k.startswith(pfx) else k): v for k, v in d.items() }
        return d
    sd = strip_prefix(sd, 'module.')
    sd = strip_prefix(sd, 'adapter.')

    return sd

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
    ap.add_argument("--inv_backend", type=str, default="auto",
                choices=["auto", "dino", "clip"],
                help="inversion 特徵後端：auto 會依 bitstream 維度自動選擇")
    ap.add_argument("--inv_dino_model", type=str, default="vit_base_patch14_dinov2.lvd142m",
                help="timm 的 DINOv2 模型名，用於 inversion（若選 dino 或 auto→dino）")
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
    sd = _load_adapter_state(args.adapter, dec.device)
    missing, unexpected = dec.adapter.load_state_dict(sd, strict=False)  # 如需嚴格可改 True
    if missing or unexpected:
        print(f"[warn] load_state_dict: missing={missing}, unexpected={unexpected}")

    unet, vae, scheduler = dec.unet, dec.vae, dec.noise_scheduler
    unet.eval(); vae.eval()

    # 4) 準備 CLIP 編碼器（只在 inv_weight>0 才建）
    inv_use = args.inv_weight > 0
    if inv_use:
        backend = args.inv_backend
        if backend == "auto":
            # 512 → CLIP；384/768/1024/1536 → DINOv2
            backend = "clip" if dim == 512 else "dino"

        if backend == "clip":
            if dim != 512:
                raise ValueError(f"inv_backend=clip 但 bitstream 維度是 {dim}，請改用 --inv_backend dino（或 auto）")
            import open_clip
            clip_model, _, _ = open_clip.create_model_and_transforms(
                args.inv_clip_arch, pretrained=args.inv_clip_ckpt
            )
            clip_model = clip_model.to(dec.device).eval()

            def _encode_img(x_img: torch.Tensor) -> torch.Tensor:
                x_in = _clip_preprocess_torch(x_img)                      # BCHW → CLIP 正規化
                y = clip_model.encode_image(x_in)                         # (B, 512)
                return torch.nn.functional.normalize(y.float(), dim=-1)   # L2

        elif backend == "dino":
            import timm
            from timm.data import resolve_model_data_config
            dino = timm.create_model(args.inv_dino_model, pretrained=True, num_classes=0).to(dec.device).eval()
            cfg = resolve_model_data_config(dino)
            mean = torch.tensor(cfg["mean"], device=dec.device).view(1,3,1,1)
            std  = torch.tensor(cfg["std"],  device=dec.device).view(1,3,1,1)
            size = (cfg["input_size"][-2], cfg["input_size"][-1])

            def _encode_img(x_img: torch.Tensor) -> torch.Tensor:
                x_in = (x_img.clamp(-1,1) + 1.0) / 2.0
                x_in = torch.nn.functional.interpolate(x_in, size=size, mode="bilinear", align_corners=False)
                x_in = (x_in - mean) / std
                y = dino(x_in)                                            # (B, 768/1024/1536)
                return torch.nn.functional.normalize(y.float(), dim=-1)   # L2
        else:
            raise ValueError(f"Unknown inv_backend: {backend}")

        # 目標特徵（由 bitstream 反量化而來），與上面 encoder 尺寸對齊
        z_tgt = torch.from_numpy(z).to(dec.device)                         # (1, dim)
        z_tgt = torch.nn.functional.normalize(z_tgt, dim=-1)
        
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
                y_feat = _encode_img(x0_img)                      # (B, dim)
                z_b = z_tgt.expand(y_feat.size(0), -1)            # (B, dim)
                feat_loss = (1.0 - torch.cosine_similarity(y_feat, z_b, dim=-1).mean())
                g = torch.autograd.grad(feat_loss, lat, retain_graph=False)[0]
                lat = (lat - args.inv_weight * g / (g.norm() + 1e-8)).detach()

            # (c) DDIM 前進一步
            lat = scheduler.step(eps, t, lat, eta=args.eta).prev_sample

    # 6) 輸出影像
    with torch.no_grad(), torch.autocast(device_type="cuda", enabled=(str(args.device).startswith("cuda") and torch.cuda.is_available())):
        img = vae.decode((lat / dec.scaling_factor)).sample
    out = (img[0].clamp(-1,1).add(1).mul(127.5).to(torch.uint8).permute(1,2,0).cpu().numpy())

    def _fmt_num(x: float) -> str:
      return f"{x:g}"
    
    if args.out == Path("recon.png"):
        stem = args.bitstream.stem            # 例如 0001.clp → stem = "0001"
        out_name = f"{stem}-{args.steps}-{_fmt_num(args.guidance)}-{_fmt_num(args.inv_weight)}.png"
        out_path = args.bitstream.with_name(out_name)  # 存在 bitstream 同資料夾
    else:
        out_path = args.out

    Image.fromarray(out).save(out_path)
    print("Saved to", out_path)

if __name__ == "__main__":
    main()
