# clip-feature-codec (Diffusion + Evaluation edition)

This repository contains a **high-quality image codec** built on CLIP features and a diffusion-based decoder. It includes:

* A **U-Net-based diffusion decoder** (`CLIPCondUNet`) conditioned on CLIP feature vectors and sinusoidal timesteps.
* Training utilities implementing the DDPM objective (predicting ε) with optional reconstruction, total-variation and CLIP alignment losses.
* A **DDIM sampler** for deterministic or stochastic sampling from noise conditioned on a CLIP embedding.
* Command-line tools to encode images into CLIP features, quantize them to per‑channel int8 bitstreams, train the diffusion decoder, reconstruct images via DDIM, and **evaluate reconstruction quality** (PSNR, SSIM, LPIPS, CLIP similarity).

## Features

### Compression
Images are encoded into CLIP embeddings using `open_clip_torch`, L2‑normalized, and quantized with an 8‑bit affine per‑channel quantizer. Encoded vectors are saved as `.clp` files with a simple zstd-based bitstream format alongside a manifest and codec metadata.

### Diffusion Decoder
`CLIPCondUNet` predicts noise ε given a noisy image `x_t`, CLIP embedding `z_clip`, and discrete timestep `t`. Sinusoidal timestep embeddings are projected and combined with a linear projection of the CLIP embedding via FiLM conditioning in each residual block. A cosine or linear noise schedule is implemented in `NoiseScheduler`.

### Training
`train/diffusion_train.py` provides a training loop that:

* Samples a random timestep `t` for each ground truth image `x0`.
* Generates `x_t` via the scheduler’s `q_sample` using random noise.
* Predicts ε with the UNet and computes MSE between predicted and true noise.
* Optionally adds small weights of reconstruction loss (L1), total variation, and CLIP alignment loss to encourage high fidelity.
* Uses `bfloat16` autocast and TF32 where available for efficient training on modern GPUs.

### Reconstruction
`cli/reconstruct_diffusion.py` reconstructs an image from its `.clp` bitstream by sampling from the diffusion model using **DDIM**. Starting from Gaussian noise, the sampler iteratively denoises conditioned on the CLIP embedding and the current timestep.

### Evaluation
`cli/eval.py` loads a store of images and their bitstreams, reconstructs each image using a trained decoder, and computes quantitative metrics:

* **PSNR** (peak signal‑to‑noise ratio)
* **SSIM** (structural similarity index) – requires `scikit-image`
* **LPIPS** perceptual distance – requires `lpips`
* **CLIP similarity** – cosine similarity between CLIP image embeddings of the original and reconstructed images

Results are aggregated and printed, with per-image metrics optionally dumped to JSON.

## Quickstart (Colab / A100)

```bash
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install open_clip_torch pillow numpy tqdm zstandard faiss-cpu lpips scikit-image
pip install -e .

# Step 1: Encode images and build store
python -m clip_feature_codec.cli.encode_images --img_dir /path/to/images --out_dir store

# Step 2: Train diffusion decoder (adjust hyperparameters for quality)
python - <<'PY'
from pathlib import Path
from clip_feature_codec.train.diffusion_train import train_diffusion
ckpt = train_diffusion(Path('store'), out_size=256, epochs=40, batch_size=8, timesteps=1000,
                       recon_w=0.05, clip_w=0.1, tv_w=1e-4, device='cuda')
print('saved model:', ckpt)
PY

# Step 3: Reconstruct an image via DDIM
python -m clip_feature_codec.cli.reconstruct_diffusion \
  --store_dir store \
  --bitstream store/<name>.clp \
  --weights store/diffusion_unet_final.pt \
  --steps 50 --eta 0.0 --size 256 --out recon.png

# Step 4: Evaluate the codec on all images in the store
python -m clip_feature_codec.cli.eval \
  --store_dir store \
  --weights store/diffusion_unet_final.pt \
  --steps 50 --eta 0.0 --size 256 --device cuda
```

## VRAM & Speed Considerations
* **Batch size and resolution**: Default UNet (`base=128, ch_mult=(1,2,2)`) handles 256px images with batch size 8 on a 24GB card (A100/4090). Reduce `base` or `ch_mult` for lower memory usage.
* **Mixed precision**: The training loop uses `torch.autocast` with `bfloat16` by default and TF32 for matmuls. Adjust if unsupported.
* **Sampling steps**: DDIM with 50 steps provides a good trade‑off between quality and speed. Increase to 100–250 for higher fidelity.

## Licence

This project is released under the MIT License.