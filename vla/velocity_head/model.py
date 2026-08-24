"""The velocity-head VLA.

    image  --frozen CNN-->  f ---.
                                  |--> cond --> FLOW HEAD --> z   (action_text latent)
    u[t-H..t-1] ----------->  t --'                           |
                                                              v
                                     cond + z --> VELOCITY head  -> u[t..t+F-1]
                                                  ACTION  head    -> class of u[t]

The flow head is unchanged from ../defussion_text_generator: rectified flow,
FiLM conditioning hoisted out of the sampling loop. What changed is the
downstream head -- it emits a command chunk instead of waypoints, and the
chunk is split into a linear and an angular branch so the two losses stay
separable all the way through.

Waypoints are NOT produced here. They come from the frozen 4-parameter plant
model in ../trajectory_predictor, which is not trained and has no gradient.
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
    """y = x + W2 . act( FiLM( W1 . x ) ) — conditioning enters as scale and shift."""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.SiLU()

    def forward(self, x, scale, shift):
        h = self.norm(self.fc1(x))
        h = h * (1 + scale) + shift
        return x + self.fc2(self.act(h))


class FlowTextHead(nn.Module):
    """Generates the action_text latent z, conditioned on scene + command history."""

    def __init__(self, latent_dim: int, cond_dim: int, hidden: int, n_blocks: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.time_dim = 32
        self.hidden, self.n_blocks = hidden, n_blocks
        n_film = 2 * hidden * n_blocks

        self.inp = nn.Linear(latent_dim, hidden)
        self.blocks = nn.ModuleList(FiLMResBlock(hidden) for _ in range(n_blocks))
        self.out = nn.Linear(hidden, latent_dim)

        self.film_cond = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(), nn.Linear(hidden, n_film),
        )
        self.film_time = nn.Linear(self.time_dim, n_film)

    def cond_film(self, cond):
        return self.film_cond(cond)

    def _split(self, film):
        p = film.view(-1, self.n_blocks, 2, self.hidden)
        return p[:, :, 0], p[:, :, 1]

    def velocity(self, z_s, s, cond_film):
        film = cond_film + self.film_time(timestep_embedding(s, self.time_dim))
        scale, shift = self._split(film)
        h = self.inp(z_s)
        for i, blk in enumerate(self.blocks):
            h = blk(h, scale[:, i], shift[:, i])
        return self.out(h)

    def forward(self, z_s, s, cond):
        return self.velocity(z_s, s, self.cond_film(cond))

    @torch.no_grad()
    def sample(self, cond, steps: int, generator=None):
        """Euler-integrate the straight path from noise (s=0) to data (s=1)."""
        b = cond.shape[0]
        cond_film = self.cond_film(cond)          # once, not once per step
        z = torch.randn(b, self.latent_dim, device=cond.device, generator=generator)
        dt = 1.0 / steps
        for i in range(steps):
            s = torch.full((b,), i * dt, device=cond.device)
            z = z + dt * self.velocity(z, s, cond_film)
        return z


class VelocityHead(nn.Module):
    """cond + text latent -> (v chunk, w chunk, action logits).

    The two output branches are separate Linear layers on a shared trunk, so
    `loss_v` and `loss_w` have disjoint parameters at the last layer and can be
    weighted or reported independently.
    """

    def __init__(self, cond_dim, latent_dim, hidden, horizon, n_actions):
        super().__init__()
        self.horizon = horizon
        self.trunk = nn.Sequential(
            nn.Linear(cond_dim + latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.lin = nn.Linear(hidden, horizon)      # v[t .. t+F-1]
        self.ang = nn.Linear(hidden, horizon)      # w[t .. t+F-1]
        self.action = nn.Linear(hidden, n_actions)

    def forward(self, cond, z):
        h = self.trunk(torch.cat([cond, z], dim=-1))
        return self.lin(h), self.ang(h), self.action(h)


class VelocityVLA(nn.Module):
    def __init__(self, img_dim: int, tel_dim: int, latent_dim: int, cfg):
        super().__init__()
        self.img_proj = nn.Sequential(nn.Linear(img_dim, cfg.img_proj_dim), nn.SiLU())
        self.tel_proj = nn.Sequential(nn.Linear(tel_dim, cfg.tel_proj_dim), nn.SiLU())
        cond_dim = cfg.img_proj_dim + cfg.tel_proj_dim

        self.flow = FlowTextHead(latent_dim, cond_dim, cfg.hidden_dim, cfg.n_blocks)
        self.head = VelocityHead(
            cond_dim, latent_dim, cfg.hidden_dim, cfg.horizon, cfg.n_actions
        )
        self.latent_dim = latent_dim

    def condition(self, img, tel):
        return torch.cat([self.img_proj(img), self.tel_proj(tel)], dim=-1)

    def forward(self, img, tel, z):
        return self.head(self.condition(img, tel), z)

    @torch.no_grad()
    def predict(self, img, tel, steps: int, generator=None):
        """-> z, v_chunk, w_chunk, action_logits  (chunks are standardised)."""
        cond = self.condition(img, tel)
        z = self.flow.sample(cond, steps, generator=generator)
        v, w, logits = self.head(cond, z)
        return z, v, w, logits


def parameter_report(model: nn.Module) -> dict:
    per_part = {
        name: sum(p.numel() for p in module.parameters())
        for name, module in model.named_children()
    }
    total = sum(p.numel() for p in model.parameters())

    flow = getattr(model, "flow", None)
    hoisted = sum(p.numel() for p in flow.film_cond.parameters()) if flow else 0
    per_step = (sum(p.numel() for p in flow.parameters()) - hoisted) if flow else 0

    return {
        "total_params": total,
        "by_module": per_part,
        "size_fp32_mb": round(total * 4 / 1e6, 3),
        "size_fp16_mb": round(total * 2 / 1e6, 3),
        "flow_per_step_params": per_step,
        "flow_hoisted_params": hoisted,
    }
