"""
Evaluate reconstructed images from a store against original images.

This script loads a store directory with `manifest.json` and `.clp` bitstreams,
reconstructs each image using a trained diffusion decoder, and computes
PSNR, SSIM, LPIPS, and CLIP similarity metrics. Results are aggregated and
printed; per-image metrics can optionally be saved to a JSON file.

Usage:
    python -m clip_feature_codec.cli.eval --store_dir store --weights ckpt.pt --size 256 --steps 50 --eta 0.0
"""

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch

from clip_feature_codec.io.bitstream import read_bitstream
from clip_feature_codec.diffusion.scheduler import NoiseScheduler
from clip_feature_codec.diffusion.ddim import DDIMSampler
from clip_feature_codec.models.unet import CLIPCondUNet
from clip_feature_codec.eval.metrics import psnr, ssim, lpips_distance, clip_similarity, _to_uint8


def l2_normalize_np(x, axis=-1, eps=1e-9):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate reconstruction quality on a store of images.")
    ap.add_argument("--store_dir", type=str, required=True)
    ap.add_argument("--weights", type=str, required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()
    store_dir = Path(args.store_dir)
    manifest = json.loads((store_dir / 'manifest.json').read_text(encoding='utf-8'))
    meta = np.load(store_dir / 'codec_meta.npz')
    scale, zero = meta['scale'].astype('float32'), meta['zero'].astype('float32')
    device = args.device
    # Load model
    z_dim = scale.shape[0]
    net = CLIPCondUNet(z_dim=z_dim, base=128, ch_mult=(1, 2, 2), img_ch=3).to(device)
    net.load_state_dict(torch.load(args.weights, map_location=device), strict=True)
    net.eval()
    sch = NoiseScheduler(timesteps=1000, schedule='cosine', device=device)
    sampler = DDIMSampler(sch, eta=args.eta)
    metrics = []
    for rec in tqdm(manifest, desc='eval'):
        q = read_bitstream(Path(rec['bitstream']))
        z = q.astype(np.float32) * scale + zero
        z = l2_normalize_np(z[None, :]).astype(np.float32)
        z_t = torch.from_numpy(z).to(device)
        # Reconstruct
        with torch.no_grad():
            x = sampler.sample(net, z_t, shape=(1, 3, args.size, args.size), steps=args.steps)
        img_recon = x[0].clamp(-1, 1).cpu().numpy()
        # Load original (resized)
        img0 = Image.open(rec['image']).convert('RGB').resize((args.size, args.size), Image.BICUBIC)
        img0_np = (np.array(img0).astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)
        m = {
            'image': rec['image'],
            'psnr': psnr(img0_np, img_recon),
            'ssim': ssim(img0_np, img_recon),
            'lpips': lpips_distance(img0_np, img_recon, device=device),
            'clip_sim': clip_similarity(img0_np, img_recon, device=device),
        }
        metrics.append(m)
    # Aggregate metrics
    def _agg(key):
        vals = [m[key] for m in metrics if not np.isnan(m[key])]
        return float(np.mean(vals)) if vals else float('nan')
    print(f"Average PSNR: {_agg('psnr'):.2f} dB")
    print(f"Average SSIM: {_agg('ssim'):.4f}")
    print(f"Average LPIPS: {_agg('lpips'):.4f}")
    print(f"Average CLIP similarity: {_agg('clip_sim'):.4f}")
    if args.out_json:
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()