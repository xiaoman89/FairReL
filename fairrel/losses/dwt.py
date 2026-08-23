"""Differentiable channel-wise 2D Daubechies-4 discrete wavelet transform.

Implemented with strided convolutions so the transform is GPU-friendly and
fully differentiable.  Zero-extension ("full" convolution) boundary handling is
used, which makes the decimated orthogonal transform exactly energy-preserving:
the summed energy of the four sub-bands equals the energy of the input at every
level (verified in ``tests/test_fairrel.py``).

Band ordering for J levels follows the paper (coarse to fine):
    [LL_J, LH_J, HL_J, HH_J, LH_{J-1}, ..., LH_1, HL_1, HH_1]
giving L = 3J + 1 sub-bands (J = 2 -> 7 bands).
"""

from typing import List, Tuple

import torch
import torch.nn.functional as F

# Daubechies-4 decomposition filters (8 taps, PyWavelets convention).
DB4_DEC_LO = [
    -0.010597401784997278,
    0.032883011666982945,
    0.030841381835986965,
    -0.18703481171888114,
    -0.02798376941698385,
    0.6308807679295904,
    0.7148465705525415,
    0.23037781330885523,
]
DB4_DEC_HI = [
    -0.23037781330885523,
    0.7148465705525415,
    -0.6308807679295904,
    -0.02798376941698385,
    0.18703481171888114,
    0.030841381835986965,
    -0.032883011666982945,
    -0.010597401784997278,
]


def _dwt_kernels(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    lo = torch.tensor(DB4_DEC_LO, dtype=dtype, device=device)
    hi = torch.tensor(DB4_DEC_HI, dtype=dtype, device=device)
    # 2D separable kernels via outer products; first factor filters rows
    # (height), second filters columns (width).
    ll = torch.outer(lo, lo)
    lh = torch.outer(hi, lo)  # high-pass along height, low-pass along width
    hl = torch.outer(lo, hi)
    hh = torch.outer(hi, hi)
    kernels = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # (4, 1, 8, 8)
    # conv2d computes cross-correlation; flip so the operation is a true
    # convolution with the decomposition filters.
    return kernels.flip(-1).flip(-2)


def dwt2_level(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One level of the 2D DWT applied channel-wise.

    Args:
        x: (N, C, H, W) feature map.
    Returns:
        (LL, LH, HL, HH), each of shape (N, C, H', W').
    """
    n, c, h, w = x.shape
    kernels = _dwt_kernels(x.dtype, x.device)
    pad = kernels.shape[-1] - 1  # full convolution: capture every overlap
    y = F.conv2d(x.reshape(n * c, 1, h, w), kernels, stride=2, padding=pad)
    y = y.reshape(n, c, 4, y.shape[-2], y.shape[-1])
    return y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]


def dwt2(x: torch.Tensor, levels: int = 2) -> Tuple[List[torch.Tensor], List[str]]:
    """Multi-level channel-wise 2D db4 DWT.

    Args:
        x: (N, C, H, W) feature map.
        levels: decomposition depth J.
    Returns:
        (bands, names): 3J+1 coefficient tensors ordered coarse to fine and
        their names, e.g. ["LL2", "LH2", "HL2", "HH2", "LH1", "HL1", "HH1"].
    """
    if levels < 1:
        raise ValueError("levels must be >= 1")
    details = []  # fine to coarse while iterating
    ll = x
    for j in range(1, levels + 1):
        ll, lh, hl, hh = dwt2_level(ll)
        details.append((j, lh, hl, hh))
    bands = [ll]
    names = [f"LL{levels}"]
    for j, lh, hl, hh in reversed(details):
        bands.extend([lh, hl, hh])
        names.extend([f"LH{j}", f"HL{j}", f"HH{j}"])
    return bands, names
