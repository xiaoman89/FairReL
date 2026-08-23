"""FairReL detector: SVD-decomposed CLIP backbone + binary classifier head.

Training uses two forward passes per batch (Sec. 3.2):
  - the standard path B_{p+Delta} produces the spatial feature map F, the
    pooled feature z, and the real/fake logit;
  - the residual-only path B_Delta (W_p replaced by 0 in every decomposed
    projection) produces z_Delta for the SLMA loss.

At inference only the standard path and the classifier are kept, so FairReL
adds no inference-time overhead over the underlying backbone.
"""

import torch
import torch.nn as nn

from .backbone import build_backbone
from .svd_linear import merge_svd_layers, residual_only, total_orthogonality_loss


class FairReLDetector(nn.Module):
    def __init__(self, model_name: str = "ViT-L-14", pretrained: str = "openai",
                 rank: int = 1, pooling: str = "cls"):
        super().__init__()
        self.backbone = build_backbone(model_name, pretrained, rank, pooling)
        self.classifier = nn.Linear(self.backbone.feat_dim, 1)

    def forward(self, x: torch.Tensor):
        """Standard path.  Returns (feat_map, z, logit)."""
        feat_map, z = self.backbone(x)
        logit = self.classifier(z).squeeze(-1)
        return feat_map, z, logit

    def forward_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Residual-only path (Eq. 4).  Returns z_Delta of shape (N, D)."""
        with residual_only(self.backbone):
            _, z_delta = self.backbone(x)
        return z_delta

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Fake probability, inference path only."""
        _, _, logit = self.forward(x)
        return torch.sigmoid(logit)

    def merge_for_inference(self) -> "FairReLDetector":
        """Fold the SVD decomposition into plain linear layers for deployment.

        The merged model produces identical predictions with the memory and
        speed of the undecomposed backbone; call after loading a trained
        checkpoint.  Training-only members (orthogonality loss, residual-only
        path) stop working afterwards.
        """
        merge_svd_layers(self.backbone)
        self.eval()
        return self

    def orthogonality_loss(self) -> torch.Tensor:
        return total_orthogonality_loss(self.backbone)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
