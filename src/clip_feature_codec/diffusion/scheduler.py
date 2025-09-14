"""
DDPM noise schedule and helper functions.

This module implements a classic DDPM scheduler that precomputes beta values
(either linear or cosine), alpha products, and convenience terms such as
`sqrt_alphas_cumprod`, which are used both during training (`q_sample`) and
in sampling (DDPM or DDIM). The scheduler also provides functions for
predicting the original image from noise and computing posterior means and
variances.
"""

from __future__ import annotations
import math
import torch
from typing import Optional


class NoiseScheduler:
    """Basic DDPM scheduler with linear or cosine beta schedules."""

    def __init__(self, timesteps: int = 1000, schedule: str = "cosine", device: str = "cuda") -> None:
        self.timesteps = timesteps
        self.schedule = schedule
        self.device = device
        if schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        elif schedule == "cosine":
            s = 0.008
            steps = timesteps + 1
            t = torch.linspace(0, timesteps, steps, device=device) / timesteps
            alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = betas.clamp(0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule {schedule}")
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), self.alphas_cumprod[:-1]], dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Sample x_t by adding noise to x0 according to the schedule."""
        return (self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x0 +
                self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * noise)

    def predict_x0_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps_hat: torch.Tensor) -> torch.Tensor:
        """Predict the clean image x0 from x_t and predicted noise eps_hat."""
        return (
            x_t - self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * eps_hat
        ) / self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)

    def p_mean_variance(self, model, x_t: torch.Tensor, z_clip: torch.Tensor, t: torch.Tensor):
        """Compute mean and variance of the posterior distribution p(x_{t-1} | x_t)."""
        eps = model(x_t, z_clip, t)
        x0_pred = self.predict_x0_from_eps(x_t, t, eps).clamp(-1, 1)
        al_t = self.alphas[t]
        al_bar_t = self.alphas_cumprod[t]
        al_bar_prev = self.alphas_cumprod_prev[t]
        coef1 = (torch.sqrt(al_bar_prev) * (1 - al_t)) / (1 - al_bar_t)
        coef2 = (torch.sqrt(al_t) * (1 - al_bar_prev)) / (1 - al_bar_t)
        mean = coef1.view(-1, 1, 1, 1) * x0_pred + coef2.view(-1, 1, 1, 1) * x_t
        var = self.posterior_variance[t].view(-1, 1, 1, 1)
        return mean, var, x0_pred