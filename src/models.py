"""
Cover encoder: backbone -> GeM -> L2-normalised embedding, plus an ArcFace head.

1) ArcFace (Deng et al., 2019). Plain softmax learns to separate classes;
   retrieval needs a space where distance is meaningful, including for
   classes never seen in training. ArcFace adds an angular margin m to the
   logit of the correct class:

       logit_y = s * cos(theta_y + m)      (other classes: s * cos(theta_j))

   With normalised features and weights the logit is a pure cosine, and the
   margin forces the angle to the correct class centre to be smaller than the
   angle to any other centre by at least m: compact, well separated clusters
   on the hypersphere, which is what nearest-neighbour search needs. The scale
   `s` keeps the logits large enough for cross-entropy not to saturate.

2) GeM pooling (Radenovic et al., 2018) generalises average and max pooling:

       f = ( mean(x^p) )^(1/p)

   p=1 is average pooling, p->inf is max pooling; p is learned (starts at 3).

3) Sub-centres (K > 1): each class gets K centres and uses the closest one.
   This absorbs intra-class variation; here, the same game at very different
   degradation levels behaves almost like two sub-populations.
"""

from __future__ import annotations

import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class GeM(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1e-6, learn_p: bool = True):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p) if learn_p else torch.tensor(p)
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        p = self.p.to(x.device) if isinstance(self.p, torch.Tensor) else self.p
        x = x.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, 1)
        return x.pow(1.0 / p).flatten(1)


class CoverEncoder(nn.Module):
    """Image -> L2-normalised embedding of size `embed_dim`."""

    def __init__(
        self,
        backbone: str = "convnext_base.fb_in22k_ft_in1k",
        embed_dim: int = 512,
        pretrained: bool = True,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        # num_classes=0, global_pool='' -> returns the feature map (B, C, H, W)
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool=""
        )
        feat_dim = self.backbone.num_features
        self.pool = GeM()
        self.neck = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feat_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        if f.ndim == 3:  # ViT: (B, N, C) -> approximate (B, C, H, W)
            b, n, c = f.shape
            h = int(math.sqrt(n))
            f = f[:, n - h * h:, :].transpose(1, 2).reshape(b, c, h, h)
        f = self.pool(f)
        f = self.neck(f)
        return F.normalize(f, dim=-1)


class ArcFace(nn.Module):
    """ArcFace head with sub-centres."""

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        scale: float = 30.0,
        margin: float = 0.30,
        k: int = 2,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes * k, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        self.num_classes = num_classes
        self.k = k
        self.scale = scale
        self.margin = margin

    def set_margin(self, m: float) -> None:
        """Margin warm-up: the full margin from the first step tends to stall
        training (very large initial gradients), so it is ramped up."""
        self.margin = m

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        w = F.normalize(self.weight, dim=-1)
        cos = emb @ w.t()                                  # (B, C*k)
        if self.k > 1:
            cos = cos.view(-1, self.num_classes, self.k).max(dim=2).values
        cos = cos.clamp(-1 + 1e-7, 1 - 1e-7)

        theta = torch.acos(cos)
        target = torch.zeros_like(cos)
        target.scatter_(1, labels.view(-1, 1), 1.0)
        # cos(theta + m) on the correct class only
        logits = torch.cos(theta + self.margin * target) * self.scale
        return logits


def cosine_distill_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """1 - cos(degraded_student, clean_teacher.detach()).

    Pulls the feature of the degraded image towards the feature of the clean
    image of the same game. Unlike ArcFace, this term does not depend on the
    set of classes, so it also applies to games outside the training set
    (open-set generalisation).
    """
    return (1.0 - (student * teacher.detach()).sum(dim=-1)).mean()
