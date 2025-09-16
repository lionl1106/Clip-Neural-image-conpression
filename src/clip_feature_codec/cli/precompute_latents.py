# scripts/precompute_latents.py
from pathlib import Path
import json, numpy as np, torch
import argparse
from diffusers import AutoencoderKL
from PIL import Image

def vae_encode_img(vae, img_path, size=512, device="cuda", scaling=0.18215):
    x = Image.open(img_path).convert("RGB").resize((size, size), Image.BICUBIC)
    x = torch.from_numpy((np.array(x).astype("float32")/127.5-1.0)).permute(2,0,1).unsqueeze(0).to(device)
    with torch.no_grad():
        lat = vae.encode(x).latent_dist.sample() * scaling
    return lat.squeeze(0).cpu().half().numpy()  # fp16 存

def main(store="store", model_name="runwayml/stable-diffusion-v1-5"):
    ap = argparse.ArgumentParser()
    ap.add_argument("--store_dir", type=Path, required=True)
    args = ap.parse_args()
    
    store = args.store_dir
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(dev).eval()
    meta = json.loads(Path(store, "manifest.json").read_text())
    out_dir = Path(store, "latents"); out_dir.mkdir(parents=True, exist_ok=True)
    for rec in meta:
        img = rec["image"]
        lat = vae_encode_img(vae, img, size=512, device=dev)
        lat_path = out_dir / (Path(img).stem + ".npz")
        np.savez_compressed(lat_path, lat=lat)  # 存 (4,64,64)
        rec["latent"] = str(lat_path)
    Path(store, "manifest_latents.json").write_text(json.dumps(meta, indent=2))

if __name__ == "__main__": main()
