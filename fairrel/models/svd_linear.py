"""SVD-decomposed linear layer with a frozen principal part and a trainable residual.

Implements Eq. (2) of the paper: a pre-trained weight W (m x n) with
s = min(m, n) singular directions is split as

    W = W_p + Delta_W,
    W_p     = U[:, :s-r] diag(S[:s-r]) V[:, :s-r]^T   (frozen, top-(s-r))
    Delta_W = U[:, s-r:] diag(S[s-r:]) V[:, s-r:]^T   (trainable, rank r)

Only the residual factors (U_r, S_r, V_r) are trainable.  The layer also
provides the orthogonality regulariser of Eq. (3), which keeps the residual
basis orthogonal to the frozen principal basis during training.
"""

from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F


class SVDResidualLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` with an SVD principal/residual split."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor = None, rank: int = 1):
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        m, n = weight.shape
        s = min(m, n)
        if rank >= s:
            raise ValueError(f"rank ({rank}) must be smaller than min(m, n) = {s}")
        self.rank = rank
        self.frozen_rank = s - rank

        U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
        V = Vh.transpose(0, 1)  # (n, s)
        k = s - rank

        # Frozen principal part: merged W_p for the forward pass, plus the
        # principal bases needed by the orthogonality regulariser.
        self.register_buffer("W_p", (U[:, :k] * S[:k]) @ V[:, :k].transpose(0, 1))
        self.register_buffer("U_p", U[:, :k].contiguous())
        self.register_buffer("V_p", V[:, :k].contiguous())

        # Trainable residual factors (rank r).
        self.U_r = nn.Parameter(U[:, k:].contiguous())
        self.S_r = nn.Parameter(S[k:].contiguous())
        self.V_r = nn.Parameter(V[:, k:].contiguous())

        if bias is not None:
            self.register_buffer("bias", bias.detach().clone().float())
        else:
            self.bias = None

        # When True the layer computes x @ Delta_W^T only (W_p replaced by 0),
        # producing the residual-only path of Eq. (4).  Toggled by the
        # ``residual_only`` context manager below.
        self._residual_only = False

    def delta_w(self) -> torch.Tensor:
        return (self.U_r * self.S_r) @ self.V_r.transpose(0, 1)

    def effective_weight(self) -> torch.Tensor:
        dw = self.delta_w()
        if self._residual_only:
            return dw
        return self.W_p + dw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.effective_weight()
        bias = self.bias
        if w.dtype != x.dtype:
            w = w.to(x.dtype)
            bias = bias.to(x.dtype) if bias is not None else None
        return F.linear(x, w, bias)

    def to_linear(self) -> nn.Linear:
        """Fold W_p + Delta_W into a plain ``nn.Linear`` for deployment."""
        m, n = self.W_p.shape
        linear = nn.Linear(n, m, bias=self.bias is not None)
        with torch.no_grad():
            linear.weight.copy_(self.W_p + self.delta_w())
            if self.bias is not None:
                linear.bias.copy_(self.bias)
        return linear

    def orthogonality_loss(self) -> torch.Tensor:
        """Eq. (3): || U_hat^T U_hat - I ||_F^2 + || V_hat^T V_hat - I ||_F^2.

        With U_hat = [U_p, U_r] and U_p frozen and orthonormal, the loss reduces
        to the residual Gram deviation plus (twice) the principal-residual
        cross-overlap; the frozen U_p^T U_p - I block is exactly zero.
        """
        eye = torch.eye(self.rank, device=self.U_r.device, dtype=self.U_r.dtype)
        loss = torch.zeros((), device=self.U_r.device, dtype=self.U_r.dtype)
        for base_p, base_r in ((self.U_p, self.U_r), (self.V_p, self.V_r)):
            cross = base_p.transpose(0, 1) @ base_r          # (s-r, r)
            gram = base_r.transpose(0, 1) @ base_r - eye     # (r, r)
            loss = loss + 2.0 * cross.pow(2).sum() + gram.pow(2).sum()
        return loss


@contextmanager
def residual_only(module: nn.Module):
    """Run the module with W_p replaced by 0 in every SVDResidualLinear (Eq. (4))."""
    layers = [m for m in module.modules() if isinstance(m, SVDResidualLinear)]
    for layer in layers:
        layer._residual_only = True
    try:
        yield module
    finally:
        for layer in layers:
            layer._residual_only = False


def merge_svd_layers(module: nn.Module) -> nn.Module:
    """Replace every SVDResidualLinear in the module with a plain ``nn.Linear``
    holding the merged weight W_p + Delta_W.

    After merging, the model is a standard backbone with no extra buffers or
    per-forward reconstruction, so inference cost and memory match the
    undecomposed network exactly.  Training-only functionality (orthogonality
    loss, residual-only path) is no longer available.
    """
    for name, child in module.named_children():
        if isinstance(child, SVDResidualLinear):
            setattr(module, name, child.to_linear())
        else:
            merge_svd_layers(child)
    return module


def total_orthogonality_loss(module: nn.Module) -> torch.Tensor:
    losses = [m.orthogonality_loss() for m in module.modules() if isinstance(m, SVDResidualLinear)]
    if not losses:
        raise RuntimeError("no SVDResidualLinear layers found")
    return torch.stack(losses).sum()
