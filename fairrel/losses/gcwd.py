"""Group-Conditional Wavelet Decorrelation (GCWD), Sec. 3.3 of the paper.

Pipeline (Fig. 3):
  1. Channel-wise 2-level db4 DWT of the spatial feature map F -> L = 7 bands.
  2. Per-band, per-subgroup mean energies E_l^g and the cross-subgroup
     discrepancy D_l = sum_g (E_l^g - E_bar_l)^2                    (Eq. 5).
  3. Adaptive band weights w_l = softmax(sg(D_l) / tau) and the loss
     L_GCWD = sum_l w_l * D_l                                       (Eq. 6).

The stop-gradient on the weighting path makes w_l act as an importance score
only, so the optimiser reduces the discrepancies instead of gaming the weights.
"""

import torch
import torch.nn as nn

from .dwt import dwt2


class GCWDLoss(nn.Module):
    """Args:
        num_groups: number of demographic subgroups G.
        levels: DWT depth J (paper default 2, giving 7 sub-bands).
        temperature: softmax temperature tau for the adaptive band weights.
        normalize_energy: divide every band energy by the input feature-map
            size (one constant shared by all bands).  The paper defines
            e_{i,l} = ||C_{i,l}||_F^2; a shared constant leaves the relative
            band discrepancies D_l identical to that definition while keeping
            the loss magnitude independent of feature-map size.  Note the
            global scale interacts with the softmax temperature tau and
            lambda_G, so disable this only together with re-tuning both.
    """

    def __init__(self, num_groups: int, levels: int = 2, temperature: float = 0.5,
                 normalize_energy: bool = True):
        super().__init__()
        self.num_groups = num_groups
        self.levels = levels
        self.temperature = temperature
        self.normalize_energy = normalize_energy

    def band_energies(self, feat_map: torch.Tensor) -> torch.Tensor:
        """Per-sample sub-band energies.

        Args:
            feat_map: (N, H, W, D) spatial feature map from the backbone.
        Returns:
            (N, L) energies, bands ordered coarse to fine.
        """
        x = feat_map.permute(0, 3, 1, 2)  # (N, D, H, W): channel-wise DWT
        bands, _ = dwt2(x, self.levels)
        scale = float(x[0].numel()) if self.normalize_energy else 1.0
        energies = [band.pow(2).flatten(1).sum(dim=1) / scale for band in bands]
        return torch.stack(energies, dim=1)

    def forward(self, feat_map: torch.Tensor, group: torch.Tensor) -> torch.Tensor:
        """Args:
            feat_map: (N, H, W, D) spatial feature map.
            group: (N,) integer subgroup labels in [0, G).
        """
        energies = self.band_energies(feat_map)  # (N, L)

        group_means = []
        for g in range(self.num_groups):
            mask = group == g
            if mask.any():
                group_means.append(energies[mask].mean(dim=0))
        if len(group_means) < 2:
            # Cross-group statistics are undefined with fewer than two groups
            # in the batch (does not occur with the group-balanced sampler).
            return energies.new_zeros(())

        E = torch.stack(group_means, dim=0)          # (G_present, L)
        D = (E - E.mean(dim=0, keepdim=True)).pow(2).sum(dim=0)  # (L,)

        w = torch.softmax(D.detach() / self.temperature, dim=0)  # stop-gradient
        return (w * D).sum()
