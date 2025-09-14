# CLI: reconstruct image from CLIP bitstream using SD decoder
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, torch
from PIL import Image

from ..models.sd_decoder import StableDiffusionDecoder
from ..io.bitstream import read_bitstream

def load_codec_meta(store_dir: Path):
    meta = np.load(store_dir / "codec_meta.npz")
    scale = meta["scale"].astype("float32")
    zero  = meta["zero"].astype("float32")
    dim   = int(meta["dim"]) if "dim" in meta.files else int(scale.shape[0])
    return scale, zero, dim

def l2_normalize_np(x, axis=-1, eps=1e-9):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store_dir", type=Path, required=True)
    ap.add_argument("--bitstream", type=Path, required=True)
    ap.add_argument("--adapter", type=Path, required=True, help="trained adapter .pt")
    ap.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5")
    ap.add_argument("--out", type=Path, default=Path("recon.png"))
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--guidance", type=float, default=5.0)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    # 1) 讀 meta + 正確解碼 bitstream
    scale, zero, dim = load_codec_meta(args.store_dir)
    q = read_bitstream(args.bitstream)           # ← 正確的 .clp 解碼
    if q.shape[0] != dim:
        raise ValueError(f"Bitstream dim {q.shape[0]} != meta dim {dim} "
                         f"(scale.shape={scale.shape}, zero.shape={zero.shape})")

    # 2) 反量化 + 正規化
    z = q.astype("float32") * scale + zero       # 逐元素對應，形狀一致
    z = l2_normalize_np(z[None, :]).astype("float32")  # (1, dim)

    # 3) 以正確的 clip_dim 初始化 decoder
    dec = StableDiffusionDecoder(
        model_name=args.model_name,
        device=str(args.device),
        clip_dim=dim,          # ← 依據 meta 維度
    )
    dec.adapter.load_state_dict(torch.load(args.adapter, map_location=dec.device))

    # 4) 采樣（DDIM + guidance）
    lat_hw = args.size // 8
    with torch.autocast(device_type="cuda",
                        enabled=(str(args.device).startswith("cuda") and torch.cuda.is_available())):
        y = dec.sample(torch.from_numpy(z).to(dec.device),
                       (1, 4, lat_hw, lat_hw),
                       steps=args.steps, eta=args.eta, guidance_scale=args.guidance)

    img = (y[0].clamp(-1,1).add(1).mul(127.5).to(torch.uint8).permute(1,2,0).cpu().numpy())
    Image.fromarray(img).save(args.out)
    print("Saved to", args.out)

if __name__ == "__main__":
    main()
