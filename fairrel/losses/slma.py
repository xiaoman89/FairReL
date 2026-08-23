"""Subspace-Localised Mean Alignment (SLMA), Sec. 3.4 of the paper.

SLMA aligns subgroup means of the residual-only pooled feature z_Delta within
each real/fake class:

    mu_{y,g}  = mean of z_Delta over batch samples with label y and subgroup g,
    mu_bar_y  = (1/G) sum_g mu_{y,g},
    L_SLMA    = sum_y sum_g || mu_{y,g} - mu_bar_y ||_2^2            (Eqs. 7-8).

Class conditioning is essential: subgroups are aligned only among samples with
the same real/fake label, so demographic shifts in the residual pathway are
reduced without pulling real and fake representations together.

Following the implementation details in the supplementary material, (y, g)
cells with fewer than ``min_count`` samples in a batch fall back to an
exponential-moving-average estimate of their mean (decay 0.9).  The EMA value
enters the loss detached, providing a stable reference without a gradient.
"""

import torch
import torch.nn as nn


class SLMALoss(nn.Module):
    def __init__(self, num_groups: int, feat_dim: int, ema_decay: float = 0.9,
                 min_count: int = 2):
        super().__init__()
        self.num_groups = num_groups
        self.ema_decay = ema_decay
        self.min_count = min_count
        self.register_buffer("ema_mu", torch.zeros(2, num_groups, feat_dim))
        self.register_buffer("ema_valid", torch.zeros(2, num_groups, dtype=torch.bool))
        self.reset_stats()

    def reset_stats(self):
        """Reset the (y, g) cell counters; call once per epoch."""
        self.stats = {"cells_total": 0, "cells_batch": 0, "cells_ema": 0,
                      "cells_skipped": 0}

    def fallback_fraction(self) -> float:
        """Fraction of (y, g) cells that used the EMA fallback since the last
        reset (the supplementary reports this is under 1% of batches with the
        group-balanced sampler)."""
        return self.stats["cells_ema"] / max(self.stats["cells_total"], 1)

    @torch.no_grad()
    def _update_ema(self, y: int, g: int, mean: torch.Tensor):
        mean = mean.detach().to(self.ema_mu.dtype)
        if self.ema_valid[y, g]:
            self.ema_mu[y, g].mul_(self.ema_decay).add_(mean, alpha=1.0 - self.ema_decay)
        else:
            self.ema_mu[y, g].copy_(mean)
            self.ema_valid[y, g] = True

    def forward(self, z_delta: torch.Tensor, label: torch.Tensor,
                group: torch.Tensor) -> torch.Tensor:
        """Args:
            z_delta: (N, D) residual-only pooled features.
            label: (N,) binary real/fake labels (0 real, 1 fake).
            group: (N,) integer subgroup labels in [0, G).
        """
        loss = z_delta.new_zeros(())
        for y in (0, 1):
            cls_mask = label == y
            mus = []
            for g in range(self.num_groups):
                mask = cls_mask & (group == g)
                count = int(mask.sum())
                if count >= 1:
                    batch_mean = z_delta[mask].mean(dim=0)
                    self._update_ema(y, g, batch_mean)
                if self.training:
                    self.stats["cells_total"] += 1
                if count >= self.min_count:
                    mus.append(batch_mean)
                    if self.training:
                        self.stats["cells_batch"] += 1
                elif self.training and self.ema_valid[y, g]:
                    mus.append(self.ema_mu[y, g].to(z_delta.dtype))
                    self.stats["cells_ema"] += 1
                elif self.training:
                    self.stats["cells_skipped"] += 1
            if len(mus) < 2:
                continue
            mu = torch.stack(mus, dim=0)              # (G_avail, D)
            mu_bar = mu.mean(dim=0, keepdim=True)     # subgroup-neutral reference
            loss = loss + (mu - mu_bar).pow(2).sum()
        return loss
