
import argparse, json
from pathlib import Path
from typing import List, Tuple
import numpy as np
import torch
import open_clip
import faiss

from clip_feature_codec.index.faiss_index import build_index, search_index

def l2_normalize_np(x, axis=-1, eps=1e-9):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)

@torch.no_grad()
def encode_text_to_vec(query: str, model, tokenizer, device: str) -> np.ndarray:
    tokens = tokenizer([query]).to(device)
    z = model.encode_text(tokens).float()
    z = z / z.norm(dim=-1, keepdim=True)
    return z.cpu().numpy().astype('float32')[0]

@torch.no_grad()
def encode_image_to_vec(img_path: str, model, preprocess, device: str) -> np.ndarray:
    x = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
    z = model.encode_image(x).float()
    z = z / z.norm(dim=-1, keepdim=True)
    return z.cpu().numpy().astype('float32')[0]

def load_features(store_dir: Path):
    manifest = json.loads((store_dir / 'manifest.json').read_text(encoding='utf-8'))
    meta = np.load(store_dir / 'codec_meta.npz')
    scale, zero = meta['scale'].astype('float32'), meta['zero'].astype('float32')
    feats = []; paths = []
    for rec in manifest:
        q = np.frombuffer(open(rec["bitstream"], "rb").read()[-int.from_bytes(open(rec["bitstream"], "rb").read()[4:8], "little"):], dtype=np.uint8)  # not used; placeholder
        # In practice use io.bitstream.read_bitstream; here we rebuild from manifest later in Python API examples.
        paths.append(rec["image"])
    # Decode using the Python API (see examples notebook). For CLI searching, we read pre-decoded vectors below:
    decoded = np.load(store_dir / "decoded.npy") if (store_dir / "decoded.npy").exists() else None
    if decoded is None:
        raise SystemExit("Please run the Python API to generate decoded.npy before CLI search.")
    return decoded, paths

def main():
    ap = argparse.ArgumentParser(description="Search images with a text query against a FAISS IP index.")
    ap.add_argument("--store_dir", type=str, required=True)
    ap.add_argument("--query", type=str, required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--use_gpu", action="store_true")
    args = ap.parse_args()

    store_dir = Path(args.store_dir)
    feats = np.load(store_dir / "decoded.npy")
    with open(store_dir / "manifest.json", "r", encoding="utf-8") as f:
        paths = [rec["image"] for rec in json.load(f)]

    idx = build_index(feats, use_gpu=args.use_gpu)
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-B-32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    qvec = encode_text_to_vec(args.query, model, tok, device)
    results = search_index(qvec, idx, paths, k=args.k)
    for p, s in results:
        print(f"{s:.4f}\t{p}")

if __name__ == "__main__":
    main()
