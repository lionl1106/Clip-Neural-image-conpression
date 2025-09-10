"""
DDIM sampler for CLIP-conditioned diffusion models.

`DDIMSampler` implements the DDIM sampling scheme (deterministic if η=0) for
denoising diffusion models. It takes a noise scheduler and allows sampling
with a specified number of steps from a starting noise or random noise.
"""

from __future__ import annotations
from typing import Optional
import torch


class DDIMSampler:
    """Deterministic/stochastic DDIM sampler."""

    def __init__(self, scheduler, eta: float = 0.0) -> None:
        self.sch = scheduler
        self.eta = eta

    @torch.no_grad()
    def sample(self, model, z_clip: torch.Tensor, shape: tuple, steps: int = 50, cfg_scale: float = 1.0, x_T: Optional[torch.Tensor] = None):
        device = z_clip.device
        T = self.sch.timesteps
        ts = torch.linspace(T - 1, 0, steps, device=device).long()
        if x_T is None:
            x = torch.randn(shape, device=device)
        else:
            x = x_T
        for i in range(steps):
            t = ts[i]
            t_b = torch.full((shape[0],), int(t.item()), device=device, dtype=torch.long)
            eps = model(x, z_clip, t_b)
            al_bar_t = self.sch.alphas_cumprod[t]
            al_bar_prev = self.sch.alphas_cumprod_prev[t] if i < steps - 1 else torch.tensor(1.0, device=device)
            sqrt_al_bar_t = torch.sqrt(al_bar_t)
            sqrt_one_minus_al_bar_t = torch.sqrt(1 - al_bar_t)
            x0_pred = (x - sqrt_one_minus_al_bar_t * eps) / sqrt_al_bar_t
            x0_pred = x0_pred.clamp(-1, 1)
            al_bar_s = al_bar_prev
            sigma_t = self.eta * torch.sqrt((1 - al_bar_s) / (1 - al_bar_t) * (1 - al_bar_t / al_bar_s)) if al_bar_s != 0 else 0.0
            dir_xt = torch.sqrt(al_bar_s - sigma_t ** 2) * eps
            x = torch.sqrt(al_bar_s) * x0_pred + dir_xt
            if self.eta > 0 and sigma_t > 0:
                x = x + sigma_t * torch.randn_like(x)
        return x