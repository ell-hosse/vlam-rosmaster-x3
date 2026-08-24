"""The NO-TEXT model.

    image  --frozen CNN-->  f  ---.
                                  |--> cond --> TRAJECTORY + ACTION heads
    telemetry ------------->  t  --'

Identical to ../defussion_text_generator/model.py with the flow head deleted
and the `z` input removed from the head. Nothing else changes: same
projections, same trunk shape, same output layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TrajActionHead(nn.Module):
    """cond -> future waypoints and the action class. No text latent."""

    def __init__(self, cond_dim: int, hidden: int, traj_dim: int, n_actions: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.traj = nn.Linear(hidden, traj_dim)
        self.action = nn.Linear(hidden, n_actions)

    def forward(self, cond: torch.Tensor):
        h = self.trunk(cond)
        return self.traj(h), self.action(h)


class VLAModelNoText(nn.Module):
    def __init__(self, img_dim: int, tel_dim: int, cfg, hidden: int | None = None):
        super().__init__()
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, cfg.img_proj_dim), nn.SiLU(),
        )
        self.tel_proj = nn.Sequential(
            nn.Linear(tel_dim, cfg.tel_proj_dim), nn.SiLU(),
        )
        cond_dim = cfg.img_proj_dim + cfg.tel_proj_dim
        self.hidden = hidden or cfg.hidden_dim
        self.head = TrajActionHead(
            cond_dim, self.hidden, cfg.traj_dim, cfg.n_actions
        )

    def condition(self, img: torch.Tensor, tel: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.img_proj(img), self.tel_proj(tel)], dim=-1)

    def forward(self, img, tel):
        return self.head(self.condition(img, tel))

    @torch.no_grad()
    def predict(self, img, tel):
        """Deterministic: there is no sampler, so one call is the answer."""
        return self.forward(img, tel)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def solve_hidden_for(target: int, img_dim: int, tel_dim: int, cfg,
                     lo: int = 32, hi: int = 4096) -> int:
    """Smallest trunk width whose parameter count is closest to `target`.

    Used by --match-params so the ablation is not simply a smaller network:
    if it still loses with the same budget, capacity is not the explanation.
    """
    best, best_gap = cfg.hidden_dim, None
    for h in range(lo, hi + 1, 4):
        n = count_params(VLAModelNoText(img_dim, tel_dim, cfg, hidden=h))
        gap = abs(n - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = h, gap
        elif n > target and best_gap is not None:
            break
    return best


def parameter_report(model: nn.Module) -> dict:
    """Trainable parameter count and size. The frozen CNN is not counted."""
    per_part = {
        name: sum(p.numel() for p in module.parameters())
        for name, module in model.named_children()
    }
    total = sum(p.numel() for p in model.parameters())
    return {
        "total_params": total,
        "by_module": per_part,
        "size_fp32_mb": round(total * 4 / 1e6, 3),
        "size_fp16_mb": round(total * 2 / 1e6, 3),
        "trunk_hidden": getattr(model, "hidden", None),
    }
