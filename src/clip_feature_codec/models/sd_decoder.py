# Copyright (c) 2025
# Stable‑Diffusion–based decoder with CLIP adapter (minimal skeleton).
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F

try:
    from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
except Exception as e:
    raise RuntimeError("Please `pip install diffusers[torch] transformers accelerate safetensors`") from e

class SDClipAdapter(nn.Module):
    """
    Maps a single CLIP image embedding z_clip (B,512) to the UNet cross-attention space.
    Produces encoder_hidden_states of shape (B, S, D), where S is a small token count (default 1).
    """
    def __init__(self, in_dim: int = 512, out_dim: int = 768, seq_len: int = 1):
        super().__init__()
        self.seq_len = seq_len
        self.proj = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim), nn.SiLU())
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.proj(z)                         # (B, D)
        h = h.unsqueeze(1).expand(-1, self.seq_len, -1)  # (B, S, D)
        return h

class StableDiffusionDecoder(nn.Module):
    """
    Wraps a pretrained SD VAE + UNet. Only trains the CLIP adapter; VAE/UNet are frozen.
    """
    def __init__(self, model_name: str = "runwayml/stable-diffusion-v1-5", clip_dim: int = 512, seq_len: int = 1, device: str = "cuda"):
        super().__init__()
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else "cpu")
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae").to(self.device).eval()
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").to(self.device).eval()
        for p in self.vae.parameters():  p.requires_grad_(False)
        for p in self.unet.parameters(): p.requires_grad_(False)
        self.scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)
        cross_dim = getattr(self.unet.config, "cross_attention_dim", 768)
        self.adapter = SDClipAdapter(in_dim=clip_dim, out_dim=cross_dim, seq_len=seq_len).to(self.device)
        self.noise_scheduler = DDIMScheduler.from_pretrained(model_name, subfolder="scheduler")

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        latents = self.vae.encode(x).latent_dist.sample() * self.scaling_factor
        return latents

    @torch.no_grad()
    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        latents = (latents / self.scaling_factor).to(self.device)
        imgs = self.vae.decode(latents).sample
        return imgs

    def forward(self, latents_t: torch.Tensor, z_clip: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        cond = self.adapter(z_clip.to(self.device))
        return self.unet(latents_t.to(self.device), timesteps.to(self.device), encoder_hidden_states=cond).sample

    @torch.no_grad()
    def sample(self, z_clip: torch.Tensor, shape: tuple[int,int,int,int], steps: int = 30, eta: float = 0.0, guidance_scale: float = 5.0) -> torch.Tensor:
        B, C, H, W = shape
        self.noise_scheduler.set_timesteps(steps, device=self.device)
        latents = torch.randn((B, C, H, W), device=self.device)
        cond = self.adapter(z_clip.to(self.device))
        uncond = self.adapter(torch.zeros_like(z_clip).to(self.device))  # classifier-free guidance via null
        for i, t in enumerate(self.noise_scheduler.timesteps):
            # Predict eps for cond/uncond
            eps_uncond = self.unet(latents, t, encoder_hidden_states=uncond).sample
            eps_cond   = self.unet(latents, t, encoder_hidden_states=cond).sample
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            # DDIM step
            latents = self.noise_scheduler.step(eps, t, latents, eta=eta).prev_sample
        return self.decode(latents)
