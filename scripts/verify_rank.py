"""Print the frozen/trainable ranks of every SVD-decomposed projection.

For CLIP ViT-L/14 the attention projections are 1024 x 1024, so with the paper
default r = 1 every decomposed layer reports frozen rank 1023 and trainable
rank 1 (Eq. 2 of the paper).

Usage:
    python scripts/verify_rank.py                # random init, no download
    python scripts/verify_rank.py --pretrained   # OpenAI CLIP weights
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fairrel.models import FairReLDetector, SVDResidualLinear  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default="ViT-L-14")
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--pretrained", action="store_true",
                   help="load OpenAI CLIP weights (ranks are shape-determined, "
                        "so random init gives the same answer without a download)")
    args = p.parse_args()

    pretrained = "openai" if args.pretrained else ""
    model = FairReLDetector(args.backbone, pretrained, args.rank)

    rows = [(name, m) for name, m in model.named_modules()
            if isinstance(m, SVDResidualLinear)]
    print(f"decomposed layers: {len(rows)}")
    for name, m in rows[:4]:
        dw_rank = torch.linalg.matrix_rank(m.delta_w()).item()
        print(f"  {name}: W {m.W_p.shape[0]}x{m.V_p.shape[0]}  "
              f"frozen rank {m.frozen_rank}  trainable rank {m.rank} "
              f"(numerical rank of Delta_W: {dw_rank})")
    print("  ...")

    frozen = {m.frozen_rank for _, m in rows}
    trainable = {m.rank for _, m in rows}
    print(f"all layers: frozen rank(s) {sorted(frozen)}, "
          f"trainable rank(s) {sorted(trainable)}")
    if args.rank == 1 and frozen == {1023} and trainable == {1}:
        print("OK: frozen top-1023 / trainable rank-1, matching the paper (r = 1).")


if __name__ == "__main__":
    main()
