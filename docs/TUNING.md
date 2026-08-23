# Hyperparameter Tuning Protocol

This document records the full tuning protocol used for the paper, so every
reported number can be reproduced from a documented selection procedure.

## Selection rule

All hyperparameters are selected **on the FF++ validation split only**, using
a combined score of source-domain detection and fairness at the fixed
operating point:

- threshold: pinned on the FF++ validation split at source-domain TPR = 0.90
- selection score: validation AUC (%) minus validation F_FPR (%)
  (implemented in `train.py`; higher is better)

**Target-domain (Celeb-DF / DFD / DFDC) results are never used for model or
hyperparameter selection.**

Every configuration is trained with three random seeds; the paper reports the
mean, with per-run standard deviations in the supplementary material
(<= 0.4 on AUC metrics, <= 0.6 on fairness metrics).

## Default configuration (paper)

| Hyperparameter | Value | Notes |
|---|---|---|
| Backbone | CLIP ViT-L/14 (OpenAI) | SVD split on all self-attention projections |
| SVD residual rank r | 1 | frozen rank 1023 / trainable rank 1 per projection |
| Pooling f_p | CLS token (post-LN) | matches the Effort codebase; `mean` averages patch tokens |
| lambda_orth | 0.1 | re-tuned; differs from the value in Effort |
| lambda_G (GCWD) | 0.5 | tied with lambda_S in the default setting |
| lambda_S (SLMA) | 0.5 | |
| DWT | db4, J = 2 | L = 3J + 1 = 7 sub-bands |
| GCWD temperature tau | 0.5 | softmax over detached band discrepancies |
| Optimiser | AdamW, lr 1e-3, weight decay 4e-3 | constant learning rate |
| Batch | 64 at 224 x 224 | group-balanced sampler, ~10-12 per subgroup |
| SLMA EMA fallback | decay 0.9, min count 2 | invoked on under 1% of batches |
| Hardware | 1x NVIDIA H100 | |

## Search grids

Tuning sweeps one axis at a time around the default configuration with all
other values fixed, using the selection rule above. Recommended grids:

| Axis | Grid |
|---|---|
| lambda_orth | {0, 0.01, 0.1, 1.0, 10} |
| (r, lambda_G = lambda_S) | r in {1, 4, 16, 64} x lambda in {0.1, 0.3, 0.5, 0.7, 1.0} |
| DWT depth J | {1, 2, 3} |
| GCWD temperature tau | {0.05, 0.1, 0.5, 1, 2} |
| untied (lambda_G, lambda_S) | a few points around the tied optimum |

On lambda_orth, the paper's sensitivity analysis (supplementary material)
shows a clear optimum at 0.1 with both fairness losses active: a too-loose
value lets principal directions leak into the residual and dilutes SLMA,
while a too-tight one over-constrains the residual. Note that this value was
re-tuned under the fairness objective and differs from the value used in
Effort.

## Practical notes

- The group-balanced sampler is required for stable subgroup moments in the
  GCWD/SLMA losses; with a plain random sampler small subgroups frequently
  miss batches and both losses degrade to their EMA fallbacks.
- GCWD band energies are normalised by the input feature-map size in this
  implementation (`normalize_energy: true`). The constant is shared by all
  bands, so the relative band discrepancies match the paper's un-normalised
  definition; only the global loss scale changes, which is absorbed by
  lambda_G and tau. The DWT uses zero-extension ("full" convolution)
  boundary handling.
- When changing the backbone (e.g. ViT-B/16), reuse the hyperparameters
  above as the starting point before re-tuning any individual axis.
