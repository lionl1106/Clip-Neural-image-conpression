from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import timm
from timm.data import resolve_model_data_config

# ★ 新增：使用專案既有的位元流寫檔器，會自動寫入檔頭（magic）
from ..io.bitstream import write_bitstream


def compute_dino_embeddings(
    img_paths: List[Path],
    model_name: str = "vit_base_patch14_dinov2.lvd142m",
    device: str = "cuda",
) -> Tuple[np.ndarray, dict]:
    """Compute DINOv2 embeddings for a list of images."""
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device).eval()
    cfg = resolve_model_data_config(model)
    mean = torch.tensor(cfg['mean'], device=device).view(1, 3, 1, 1)
    std = torch.tensor(cfg['std'], device=device).view(1, 3, 1, 1)
    size = (cfg['input_size'][-2], cfg['input_size'][-1])

    embs = []
    for path in img_paths:
        img = Image.open(path).convert('RGB')
        x = torch.from_numpy(np.array(img).astype('float32') / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        x = torch.nn.functional.interpolate(x, size=size, mode='bilinear', align_corners=False)
        x = (x - mean) / std
        with torch.no_grad():
            y = model(x).squeeze(0).cpu().numpy()
        y_norm = y / (np.linalg.norm(y, keepdims=True) + 1e-9)
        embs.append(y_norm.astype('float32'))
    return np.stack(embs, axis=0), cfg


def quantise_vectors(z: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quantise a batch of feature vectors to uint8."""
    z_min = z.min(axis=0)
    z_max = z.max(axis=0)
    denom = np.maximum(z_max - z_min, 1e-6)
    scale = denom / 255.0
    zero = z_min
    q = np.round((z - zero) / scale).astype('uint8')
    return q, scale.astype('float32'), zero.astype('float32')


def main() -> None:
    ap = argparse.ArgumentParser(description="Encode images into DINOv2 feature bitstreams.")
    ap.add_argument("--img_dir", type=Path, required=True, help="Directory of input images")
    ap.add_argument("--out_dir", type=Path, required=True, help="Directory to write bitstreams and metadata")
    ap.add_argument("--model_name", type=str, default="vit_base_patch14_dinov2.lvd142m",
                    help="Name of timm DINOv2 model (e.g. vit_base_patch14_dinov2.lvd142m)")
    ap.add_argument("--device", type=str, default="cuda", help="Computation device")
    args = ap.parse_args()

    img_dir = args.img_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    img_paths = [p for p in sorted(img_dir.iterdir()) if p.suffix.lower() in exts]
    if not img_paths:
        raise ValueError(f"No supported image files found in {img_dir}")

    # 1) 取 DINOv2 嵌入
    z, cfg = compute_dino_embeddings(img_paths, model_name=args.model_name, device=args.device)

    # 2) 量化
    q, scale, zero = quantise_vectors(z)

    # 3) 寫出 bitstream（使用 write_bitstream，自動含 magic/header）
    manifest = []
    for i, path in enumerate(img_paths):
        bitstream_path = out_dir / (path.stem + '.clp')
        q_i = q[i].astype('uint8')
        write_bitstream(q_i.tobytes(), int(z.shape[1]), bitstream_path)
        manifest.append({
            'image': str(path),
            'bitstream': str(bitstream_path)
        })

    # 4) 寫 meta
    np.savez(out_dir / 'codec_meta.npz',
             scale=scale, zero=zero,
             dim=np.array(z.shape[1], dtype=np.int64))  # 寫成純量，reader 更穩

    with open(out_dir / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

    print(f"Encoded {len(img_paths)} images to {out_dir}")


if __name__ == '__main__':
    main()
