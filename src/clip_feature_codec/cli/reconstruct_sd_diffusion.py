# CLI: reconstruct image from CLIP bitstream using SD decoder
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from PIL import Image

from ..models.sd_decoder import StableDiffusionDecoder  # type: ignore

def load_codec_meta(store_dir: Path):
    meta = np.load(store_dir / "codec_meta.npz")
    return meta["scale"].astype("float32"), meta["zero"].astype("float32"), int(meta["dim"])

def read_bitstream(p: Path) -> np.ndarray:
    raw = p.read_bytes()
    return np.frombuffer(raw, dtype=np.uint8)

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

    scale, zero, dim = load_codec_meta(args.store_dir)
    q = read_bitstream(args.bitstream)
    z = q.astype("float32") * scale + zero
    z = l2_normalize_np(z[None, :]).astype("float32")
    dec = StableDiffusionDecoder(model_name=args.model_name, device=args.device)
    dec.adapter.load_state_dict(torch.load(args.adapter, map_location=dec.device))
    lat_hw = args.size // 8
    with torch.autocast(device_type="cuda", enabled=(args.device.startswith("cuda") and torch.cuda.is_available())):
        y = dec.sample(torch.from_numpy(z).to(dec.device), (1, 4, lat_hw, lat_hw), steps=args.steps, eta=args.eta, guidance_scale=args.guidance)
    img = (y[0].clamp(-1,1).add(1).mul(127.5).to(torch.uint8).permute(1,2,0).cpu().numpy())
    Image.fromarray(img).save(args.out)

if __name__ == "__main__":
    main()
