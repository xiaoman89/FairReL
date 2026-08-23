"""Unit tests for the FairReL building blocks (no CLIP download required).

Run with either:
    python tests/test_fairrel.py
    pytest tests/
"""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fairrel.data.sampler import GroupBalancedBatchSampler          # noqa: E402
from fairrel.losses import GCWDLoss, SLMALoss                       # noqa: E402
from fairrel.losses.dwt import dwt2                                 # noqa: E402
from fairrel.metrics import fairness_report, threshold_at_tpr       # noqa: E402
from fairrel.models import (SVDResidualLinear, merge_svd_layers,    # noqa: E402
                            residual_only)

torch.manual_seed(0)


def test_dwt_energy_preservation():
    """Zero-extension decimated orthogonal DWT preserves energy exactly."""
    x = torch.randn(2, 8, 16, 16, dtype=torch.float64)
    bands, names = dwt2(x, levels=2)
    assert names == ["LL2", "LH2", "HL2", "HH2", "LH1", "HL1", "HH1"]
    total = sum(b.pow(2).sum() for b in bands)
    assert torch.allclose(total, x.pow(2).sum(), rtol=1e-10)


def test_dwt_gradients_flow():
    x = torch.randn(1, 4, 16, 16, requires_grad=True)
    bands, _ = dwt2(x, levels=2)
    loss = sum(b.pow(2).sum() for b in bands)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_svd_linear_reconstruction_and_ranks():
    w = torch.randn(64, 48)
    b = torch.randn(64)
    layer = SVDResidualLinear(w, b, rank=1)
    assert layer.frozen_rank == 47 and layer.rank == 1
    # At init, W_p + Delta_W must reconstruct W.
    assert torch.allclose(layer.W_p + layer.delta_w(), w, atol=1e-4)
    # Forward matches the original linear layer at init.
    x = torch.randn(5, 48)
    ref = x @ w.t() + b
    assert torch.allclose(layer(x), ref, atol=1e-4)
    # Orthogonality loss is ~0 at init (SVD bases are orthonormal).
    assert layer.orthogonality_loss().item() < 1e-6
    # Only the residual factors carry gradients.
    trainable = {n for n, p in layer.named_parameters() if p.requires_grad}
    assert trainable == {"U_r", "S_r", "V_r"}


def test_residual_only_context():
    w = torch.randn(32, 32)
    layer = SVDResidualLinear(w, None, rank=1)
    x = torch.randn(3, 32)
    full = layer(x)
    with residual_only(layer):
        res = layer(x)
    assert torch.allclose(res, x @ layer.delta_w().t(), atol=1e-5)
    assert not torch.allclose(full, res)
    assert torch.allclose(layer(x), full, atol=1e-6)  # flag restored


def test_gcwd_zero_when_groups_identical():
    gcwd = GCWDLoss(num_groups=2, levels=1)
    base = torch.randn(4, 8, 8, 6)
    feat = torch.cat([base, base], dim=0)          # identical group statistics
    group = torch.tensor([0] * 4 + [1] * 4)
    assert gcwd(feat, group).item() < 1e-10


def test_gcwd_positive_under_imbalance_and_differentiable():
    gcwd = GCWDLoss(num_groups=2, levels=2)
    feat = torch.randn(8, 16, 16, 6, requires_grad=True)
    scaled = torch.cat([feat[:4], feat[4:] * 3.0], dim=0)  # group 1 has 9x energy
    group = torch.tensor([0] * 4 + [1] * 4)
    loss = gcwd(scaled, group)
    assert loss.item() > 0
    loss.backward()
    assert feat.grad is not None and torch.isfinite(feat.grad).all()


def test_slma_zero_when_aligned_positive_when_shifted():
    slma = SLMALoss(num_groups=2, feat_dim=4)
    slma.train()
    z = torch.randn(8, 4)
    z_aligned = z.clone()
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    g = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    # Force identical (y, g) cell means: make each cell's mean zero.
    for yy in (0, 1):
        for gg in (0, 1):
            m = (y == yy) & (g == gg)
            z_aligned[m] -= z_aligned[m].mean(dim=0)
    assert slma(z_aligned, y, g).item() < 1e-10
    z_shifted = z_aligned.clone()
    z_shifted[(y == 0) & (g == 1)] += 5.0
    assert slma(z_shifted, y, g).item() > 1.0


def test_slma_ema_fallback_for_singleton_cell():
    slma = SLMALoss(num_groups=2, feat_dim=4)
    slma.train()
    # Warm-up batch initialises the EMA for every cell.
    z = torch.randn(8, 4)
    y = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    g = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1])
    slma(z, y, g)
    assert bool(slma.ema_valid.all())
    # Batch where (y=0, g=1) has a single sample: EMA fallback keeps the
    # class-0 alignment term active (loss stays finite and well defined).
    z2 = torch.randn(5, 4)
    y2 = torch.tensor([0, 0, 0, 1, 1])
    g2 = torch.tensor([0, 0, 1, 0, 1])
    loss = slma(z2, y2, g2)
    assert torch.isfinite(loss)


def test_group_balanced_sampler_counts():
    groups = torch.randint(0, 6, (1000,))
    sampler = GroupBalancedBatchSampler(groups, batch_size=64, seed=0)
    for batch in sampler:
        assert len(batch) == 64
        counts = torch.bincount(groups[torch.tensor(batch)], minlength=6)
        assert counts.min() >= 10 and counts.max() <= 12


def test_metrics_hand_computed():
    # Two groups; threshold 0.5.
    #   group 0 real FPR = 0.5, group 1 real FPR = 0.0, overall FPR = 0.25.
    scores = [0.9, 0.2, 0.8, 0.7, 0.1, 0.3, 0.95, 0.85]
    labels = [1, 0, 0, 1, 0, 0, 1, 1]
    groups = [0, 0, 0, 0, 1, 1, 1, 1]
    r = fairness_report(scores, labels, groups, threshold=0.5)
    assert abs(r["fpr_overall"] - 25.0) < 1e-9
    assert abs(r["f_fpr"] - 50.0) < 1e-9          # |50-25| + |0-25|
    assert abs(r["f_meo"] - 50.0) < 1e-9          # FPR gap 50, TPR gap 0
    thr = threshold_at_tpr(scores, labels, 0.90)  # all 4 fakes above thr
    assert thr < min(0.9, 0.7, 0.95, 0.85) + 1e-9


def test_threshold_realises_target_tpr():
    """The realised TPR at the selected threshold must meet the target, even
    when fake scores tie exactly with the threshold value."""
    scores = [0.9, 0.2, 0.8, 0.7, 0.1, 0.3, 0.95, 0.85]
    labels = [1, 0, 0, 1, 0, 0, 1, 1]
    groups = [0, 0, 0, 0, 1, 1, 1, 1]
    thr = threshold_at_tpr(scores, labels, 0.90)   # lands exactly on 0.7
    r = fairness_report(scores, labels, groups, thr)
    assert r["tpr_overall"] >= 90.0 - 1e-9


def test_merge_for_inference_equivalence():
    torch.manual_seed(1)
    w = torch.randn(24, 24)
    layer = SVDResidualLinear(w, torch.randn(24), rank=2)
    # Perturb the residual factors to simulate a trained state.
    with torch.no_grad():
        layer.S_r.mul_(1.5)
        layer.U_r.add_(0.01 * torch.randn_like(layer.U_r))
    container = torch.nn.Sequential(layer)
    x = torch.randn(4, 24)
    ref = container(x)
    merge_svd_layers(container)
    assert not any(isinstance(m, SVDResidualLinear) for m in container.modules())
    assert isinstance(container[0], torch.nn.Linear)
    assert torch.allclose(container(x), ref, atol=1e-5)


def main():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    main()
