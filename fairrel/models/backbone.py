"""CLIP ViT backbone with SVD-decomposed self-attention projections.

Every self-attention projection (q, k, v, out) in every transformer block is
replaced by an :class:`SVDResidualLinear` (Eq. 2).  For CLIP ViT-L/14 each
projection is a 1024 x 1024 matrix, so with the paper default r = 1 the frozen
principal part has rank 1023 and the trainable residual has rank 1
(``scripts/verify_rank.py`` prints this for every decomposed layer).

The wrapper exposes the quantities used by the paper's forward path (Eq. 1):
the spatial feature map F (patch-token grid of the last block, after the final
LayerNorm) and the pooled representation z = f_p(F) obtained by global average
pooling over patch tokens.  Running the backbone inside the
``residual_only`` context of :mod:`fairrel.models.svd_linear` produces the
residual-only view z_Delta (Eq. 4).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .svd_linear import SVDResidualLinear


class SVDAttention(nn.Module):
    """Drop-in replacement for ``nn.MultiheadAttention`` (self-attention only)
    with all four projections decomposed by :class:`SVDResidualLinear`."""

    def __init__(self, mha: nn.MultiheadAttention, rank: int = 1):
        super().__init__()
        if mha.in_proj_weight is None:
            raise ValueError("expected a fused in_proj_weight (q/k/v of equal dim)")
        d = mha.embed_dim
        self.num_heads = mha.num_heads
        self.head_dim = d // self.num_heads

        w = mha.in_proj_weight.data
        b = mha.in_proj_bias.data if mha.in_proj_bias is not None else None
        self.q_proj = SVDResidualLinear(w[:d], b[:d] if b is not None else None, rank)
        self.k_proj = SVDResidualLinear(w[d:2 * d], b[d:2 * d] if b is not None else None, rank)
        self.v_proj = SVDResidualLinear(w[2 * d:], b[2 * d:] if b is not None else None, rank)
        self.out_proj = SVDResidualLinear(
            mha.out_proj.weight.data,
            mha.out_proj.bias.data if mha.out_proj.bias is not None else None,
            rank,
        )

    def forward(self, query, key=None, value=None, need_weights: bool = False,
                attn_mask=None, **kwargs):
        # Same call signature as nn.MultiheadAttention with batch_first=False:
        # inputs are (L, N, D); self-attention, so key/value are ignored.
        x = query
        L, N, D = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        def split_heads(t):
            return t.reshape(L, N * self.num_heads, self.head_dim).transpose(0, 1)

        q, k, v = split_heads(q), split_heads(k), split_heads(v)  # (N*H, L, hd)
        mask = attn_mask.to(q.dtype) if attn_mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        out = out.transpose(0, 1).reshape(L, N, D)
        out = self.out_proj(out)
        return out, None


class CLIPBackbone(nn.Module):
    """OpenAI-pretrained CLIP vision transformer with decomposed attention.

    ``pooling`` selects the pooled representation z:
      - "cls": the class-token output after the final LayerNorm, matching the
        pooler output used by the official Effort codebase (default);
      - "mean": global average pooling over the patch tokens of the spatial
        feature map, the literal reading of z = f_p(F) in Eq. (1).
    """

    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai",
                 rank: int = 1, pooling: str = "cls"):
        super().__init__()
        if pooling not in ("cls", "mean"):
            raise ValueError("pooling must be 'cls' or 'mean'")
        self.pooling = pooling
        import open_clip

        clip_model = open_clip.create_model(model_name, pretrained=pretrained)
        visual = clip_model.visual
        for attr in ("conv1", "class_embedding", "positional_embedding",
                     "ln_pre", "transformer", "ln_post"):
            if not hasattr(visual, attr):
                raise RuntimeError(
                    f"unexpected open_clip VisionTransformer layout (missing {attr}); "
                    "use the open_clip_torch version pinned in requirements.txt"
                )
        self.visual = visual
        self.feat_dim = visual.transformer.width
        self.grid_size = visual.grid_size if isinstance(visual.grid_size, tuple) \
            else (visual.grid_size, visual.grid_size)

        for block in self.visual.transformer.resblocks:
            block.attn = SVDAttention(block.attn, rank=rank)

    def forward(self, x: torch.Tensor):
        """Returns (F, z): the spatial feature map (N, h, w, D) and the pooled
        representation (N, D)."""
        v = self.visual
        x = v.conv1(x)                                     # (N, D, h, w)
        n, d, h, w = x.shape
        x = x.reshape(n, d, h * w).permute(0, 2, 1)        # (N, hw, D)
        cls = v.class_embedding.to(x.dtype).expand(n, 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x)

        x = x.permute(1, 0, 2)                             # NLD -> LND
        x = v.transformer(x)
        x = x.permute(1, 0, 2)                             # LND -> NLD

        tokens = v.ln_post(x)
        patch = tokens[:, 1:, :]
        feat_map = patch.reshape(n, h, w, d)
        if self.pooling == "cls":
            z = tokens[:, 0, :]
        else:
            z = patch.mean(dim=1)
        return feat_map, z


def build_backbone(model_name: str = "ViT-L-14", pretrained: str = "openai",
                   rank: int = 1, pooling: str = "cls") -> CLIPBackbone:
    backbone = CLIPBackbone(model_name, pretrained, rank, pooling)
    # Freeze everything except the SVD residual factors.
    for p in backbone.parameters():
        p.requires_grad_(False)
    for m in backbone.modules():
        if isinstance(m, SVDResidualLinear):
            m.U_r.requires_grad_(True)
            m.S_r.requires_grad_(True)
            m.V_r.requires_grad_(True)
    return backbone
