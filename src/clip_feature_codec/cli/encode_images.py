<<<<<<< HEAD

import argparse, json, os
=======
"""
Encode a directory of images into CLIP embeddings and quantize them.

This script processes all images in a given directory, encodes them with a
pretrained CLIP model from `open_clip_torch`, normalizes them to unit length,
fits a per‑channel 8‑bit quantizer, and writes each embedding to a `.clp`
bitstream. The manifest and quantizer parameters are also saved to disk.
"""

import argparse
import json
>>>>>>> origin/master
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import open_clip

from clip_feature_codec.codecs.quantizer import PerChannelAffineQuantizer
from clip_feature_codec.io.bitstream import write_bitstream

<<<<<<< HEAD
=======

>>>>>>> origin/master
def l2_normalize_np(x, axis=-1, eps=1e-9):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)

<<<<<<< HEAD
@torch.no_grad()
def encode_images_to_clip(paths: List[str], model, preprocess, device: str, batch_size: int = 64) -> Tuple[np.ndarray, List[str]]:
    zs = []; kept = []
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    for i in tqdm(range(0, len(paths), batch_size), desc='Encode CLIP'):
        batch = []; bpaths = []
        for p in paths[i:i+batch_size]:
            try:
                im = Image.open(p).convert('RGB')
=======

@torch.no_grad()
def encode_images_to_clip(paths: List[str], model, preprocess, device: str, batch_size: int = 64) -> Tuple[np.ndarray, List[str]]:
    zs: List[torch.Tensor] = []
    kept: List[str] = []
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    for i in tqdm(range(0, len(paths), batch_size), desc="Encode CLIP"):
        batch: List[torch.Tensor] = []
        bpaths: List[str] = []
        for p in paths[i : i + batch_size]:
            try:
                im = Image.open(p).convert("RGB")
>>>>>>> origin/master
                batch.append(preprocess(im))
                bpaths.append(p)
            except Exception:
                pass
<<<<<<< HEAD
        if not batch: continue
        x = torch.stack(batch).to(device)
        with torch.autocast(device_type='cuda' if device=='cuda' else 'cpu', dtype=dtype):
=======
        if not batch:
            continue
        x = torch.stack(batch).to(device)
        with torch.autocast(device_type="cuda" if device == "cuda" else "cpu", dtype=dtype):
>>>>>>> origin/master
            z = model.encode_image(x).float()
            z = z / z.norm(dim=-1, keepdim=True)
        zs.append(z.cpu())
        kept.extend(bpaths)
    if not zs:
<<<<<<< HEAD
        return np.zeros((0, model.text_projection.shape[1]), dtype='float32'), []
    Z = torch.cat(zs, dim=0).numpy().astype('float32')
    return Z, kept

def main():
    ap = argparse.ArgumentParser(description="Encode images to CLIP and save per‑vector bitstreams.")
=======
        return np.zeros((0, model.text_projection.shape[1]), dtype="float32"), []
    Z = torch.cat(zs, dim=0).numpy().astype("float32")
    return Z, kept


def main() -> None:
    ap = argparse.ArgumentParser(description="Encode images to CLIP and save per-vector bitstreams.")
>>>>>>> origin/master
    ap.add_argument("--img_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--model", type=str, default="ViT-B-32")
    ap.add_argument("--pretrained", type=str, default="openai")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
<<<<<<< HEAD

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)  # kept for symmetry; unused
    model = model.to(args.device).eval()

    paths = [str(p) for p in Path(args.img_dir).rglob("*") if p.suffix.lower() in {".jpg",".jpeg",".png",".webp",".bmp"}]
    feats, kept = encode_images_to_clip(paths, model, preprocess, args.device, args.batch_size)
    if feats.size == 0:
        raise SystemExit("No images encoded.")

    qzr = PerChannelAffineQuantizer(8).fit(torch.from_numpy(feats))
    np.savez(out / 'codec_meta.npz', scale=qzr.scale.cpu().numpy().astype(np.float32), zero=qzr.zero.cpu().numpy().astype(np.float32), dim=np.int32(feats.shape[1]))
    manifest = []
    for p, z in tqdm(list(zip(kept, feats)), total=len(kept), desc="Write bitstreams"):
        q = qzr.encode(torch.from_numpy(z))
        out_path = out / (Path(p).stem + ".clp")
        write_bitstream(q.tobytes(), feats.shape[1], out_path)
        manifest.append({"image": p, "bitstream": str(out_path)})
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Stored {len(manifest)} vectors in {out}")

if __name__ == "__main__":
    main()
=======
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(args.device).eval()
    paths = [str(p) for p in Path(args.img_dir).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}]
    feats, kept = encode_images_to_clip(paths, model, preprocess, args.device, args.batch_size)
    if feats.size == 0:
        raise SystemExit("No images encoded.")
    D = feats.shape[1]
    qzr = PerChannelAffineQuantizer(8).fit(torch.from_numpy(feats))
    np.savez(out / "codec_meta.npz", scale=qzr.scale.cpu().numpy().astype("float32"), zero=qzr.zero.cpu().numpy().astype("float32"), dim=np.int32(D))
    manifest: List[dict] = []
    for p, z in tqdm(list(zip(kept, feats)), total=len(kept), desc="Write bitstreams"):
        q = qzr.encode(torch.from_numpy(z))
        out_path = out / (Path(p).stem + ".clp")
        write_bitstream(q.tobytes(), D, out_path)
        manifest.append({"image": p, "bitstream": str(out_path)})
    with open(out / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Done. Stored {len(manifest)} vectors in {out}")


if __name__ == "__main__":
    main()
>>>>>>> origin/master
