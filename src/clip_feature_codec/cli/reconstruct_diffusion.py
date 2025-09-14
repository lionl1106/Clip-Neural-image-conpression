"""
Reconstruct an image from a quantized CLIP bitstream via DDIM sampling.

Given a `.clp` bitstream and trained diffusion decoder weights, this script
reconstructs the corresponding image by sampling from the diffusion model
conditioned on the CLIP embedding. Supports adjustable DDIM steps and η.
"""

import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from clip_feature_codec.io.bitstream import read_bitstream
from clip_feature_codec.diffusion.scheduler import NoiseScheduler
from clip_feature_codec.diffusion.ddim import DDIMSampler
from clip_feature_codec.models.unet import CLIPCondUNet


def l2_normalize_np(x, axis=-1, eps=1e-9):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def main() -> None:
    ap = argparse.ArgumentParser(description="Reconstruct an image from a .clp bitstream via DDIM sampling.")
    ap.add_argument("--store_dir", type=str, required=True)
    ap.add_argument("--bitstream", type=str, required=True)
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--out", type=str, default="recon.png")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device
    # Load codec metadata
    meta = np.load(Path(args.store_dir) / 'codec_meta.npz')
    scale, zero = meta['scale'].astype('float32'), meta['zero'].astype('float32')
    # Decode bitstream to CLIP embedding
    q = read_bitstream(Path(args.bitstream))
    z = q.astype(np.float32) * scale + zero
    z = l2_normalize_np(z[None, :]).astype(np.float32)
    z_t = torch.from_numpy(z).to(device)
    # Load model
    net = CLIPCondUNet(z_dim=z.shape[1], base=128, ch_mult=(1, 2, 2), img_ch=3).to(device)
    net.load_state_dict(torch.load(args.weights, map_location=device), strict=True)
    net.eval()
    # Sampling
    sch = NoiseScheduler(timesteps=1000, schedule='cosine', device=device)
    sampler = DDIMSampler(sch, eta=args.eta)
    with torch.no_grad():
        x = sampler.sample(net, z_t, shape=(1, 3, args.size, args.size), steps=args.steps)
    img = x[0].clamp(-1, 1).cpu().numpy().transpose(1, 2, 0)
    img = ((img + 1.0) * 127.5).astype(np.uint8)
    Image.fromarray(img).save(args.out)
    print(f"Saved to {args.out}")


if __name__ == '__main__':
    main()