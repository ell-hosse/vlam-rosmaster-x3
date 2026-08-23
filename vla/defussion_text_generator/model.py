"""The VLA model.

    image  --frozen CNN-->  f  ---.
                                  |--> cond --> FLOW HEAD --> z   (text embedding)
    telemetry ------------->  t  --'                          |
                                                              v
                                       cond + z --> TRAJECTORY + ACTION heads

Only the flow head is generative. It is trained with rectified flow
(flow matching): the path from noise to data is a straight line, so a
handful of Euler steps — often one — reproduces what DDPM needs ~100 for.

Two deliberate optimisations:
  * the FiLM conditioning is computed ONCE per frame and reused across every
    sampling step, because it does not depend on the step;
  * the latent is small (k <= 8 here), so a step costs ~0.1 MFLOP.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def timestep_embedding(s: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding of the scalar flow time s in [0, 1]. (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(1000.0) * torch.arange(half, device=s.device, dtype=torch.float32) / half
    )
    ang = s[:, None] * freqs[None] * 1000.0
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class FiLMResBlock(nn.Module):
    """y = x + W2 . act( FiLM( W1 . x ) ) — conditioning enters as a scale and shift."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
        h = self.norm(self.fc1(x))
        h = h * (1 + scale) + shift
        return x + self.fc2(self.act(h))


class FlowTextHead(nn.Module):
    """Generates the text latent z, conditioned on the scene + telemetry.

    The FiLM parameters are the sum of two terms:

        film = film_cond(cond)  +  film_time(s)

    The first depends only on the frame, so `sample()` computes it ONCE and
    reuses it across every Euler step. Only the small time term and the
    residual blocks re-run per step. That is the single largest saving
    available in a head this size, and it is why the per-step cost in the
    step-sweep table is so much lower than the first-step cost.
    """

    def __init__(self, latent_dim: int, cond_dim: int, hidden: int, n_blocks: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_dim = 32
        self.hidden, self.n_blocks = hidden, n_blocks
        n_film = 2 * hidden * n_blocks

        self.inp = nn.Linear(latent_dim, hidden)
        self.blocks = nn.ModuleList(FiLMResBlock(hidden) for _ in range(n_blocks))
        self.out = nn.Linear(hidden, latent_dim)

        # per-frame: the expensive branch, hoisted out of the sampling loop
        self.film_cond = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, n_film),
        )
        # per-step: a single small projection off the time embedding
        self.film_time = nn.Linear(self.time_dim, n_film)

    def cond_film(self, cond: torch.Tensor) -> torch.Tensor:
        """(B, cond_dim) -> (B, 2*hidden*n_blocks). Compute once per frame."""
        return self.film_cond(cond)

    def _split(self, film: torch.Tensor):
        p = film.view(-1, self.n_blocks, 2, self.hidden)
        return p[:, :, 0], p[:, :, 1]                       # scale, shift

    def velocity(self, z_s: torch.Tensor, s: torch.Tensor, cond_film: torch.Tensor):
        film = cond_film + self.film_time(timestep_embedding(s, self.time_dim))
        scale, shift = self._split(film)
        h = self.inp(z_s)
        for i, blk in enumerate(self.blocks):
            h = blk(h, scale[:, i], shift[:, i])
        return self.out(h)

    def forward(self, z_s: torch.Tensor, s: torch.Tensor, cond: torch.Tensor):
        """Training path: conditioning is fresh every call anyway."""
        return self.velocity(z_s, s, self.cond_film(cond))

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, steps: int, generator=None) -> torch.Tensor:
        """Euler-integrate the straight path from noise (s=0) to data (s=1)."""
        b = cond.shape[0]
        cond_film = self.cond_film(cond)          # <-- once, not once per step
        z = torch.randn(b, self.latent_dim, device=cond.device, generator=generator)
        dt = 1.0 / steps
        for i in range(steps):
            s = torch.full((b,), i * dt, device=cond.device)
            z = z + dt * self.velocity(z, s, cond_film)
        return z


class TrajActionHead(nn.Module):
    """cond + generated text latent -> future waypoints and the action class."""

    def __init__(self, cond_dim: int, latent_dim: int, hidden: int,
                 traj_dim: int, n_actions: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(cond_dim + latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.traj = nn.Linear(hidden, traj_dim)
        self.action = nn.Linear(hidden, n_actions)

    def forward(self, cond: torch.Tensor, z: torch.Tensor):
        h = self.trunk(torch.cat([cond, z], dim=-1))
        return self.traj(h), self.action(h)


class VLAModel(nn.Module):
    def __init__(self, img_dim: int, tel_dim: int, latent_dim: int, cfg):
        super().__init__()
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, cfg.img_proj_dim), nn.SiLU(),
        )
        self.tel_proj = nn.Sequential(
            nn.Linear(tel_dim, cfg.tel_proj_dim), nn.SiLU(),
        )
        cond_dim = cfg.img_proj_dim + cfg.tel_proj_dim

        self.flow = FlowTextHead(latent_dim, cond_dim, cfg.hidden_dim, cfg.n_blocks)
        self.head = TrajActionHead(
            cond_dim, latent_dim, cfg.hidden_dim, cfg.traj_dim, cfg.n_actions
        )
        self.latent_dim = latent_dim

    def condition(self, img: torch.Tensor, tel: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.img_proj(img), self.tel_proj(tel)], dim=-1)

    def forward(self, img, tel, z):
        return self.head(self.condition(img, tel), z)

    @torch.no_grad()
    def predict(self, img, tel, steps: int, generator=None):
        cond = self.condition(img, tel)
        z = self.flow.sample(cond, steps, generator=generator)
        traj, logits = self.head(cond, z)
        return z, traj, logits


def parameter_report(model: nn.Module) -> dict:
    """Trainable parameter count and on-disk size."""
    per_part = {}
    for name, module in model.named_children():
        per_part[name] = sum(p.numel() for p in module.parameters())
    total = sum(p.numel() for p in model.parameters())
    return {
        "total_params": total,
        "by_module": per_part,
        "size_fp32_mb": round(total * 4 / 1e6, 3),
        "size_fp16_mb": round(total * 2 / 1e6, 3),
    }
