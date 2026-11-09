"""
Search index, retrieval metrics and the no-training baseline.

Inference does not use the ArcFace head, only the encoder:

    gallery = [ encoder(clean_cover) for each game ]     # computed once
    query   = encoder(degraded_cover)
    guess   = argmax_j  cos(query, gallery_j)

Adding a new game only means embedding its cover and appending it to the
gallery; no retraining.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def embed_loader(model, loader, device="cuda", amp=True) -> torch.Tensor:
    model.eval()
    outs = []
    for batch in loader:
        x = batch[0].to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], enabled=amp):
            e = model(x)
        outs.append(e.float().cpu())
    return F.normalize(torch.cat(outs), dim=-1)


class CoverIndex:
    """Exact cosine search. Up to ~100k covers this takes milliseconds on a GPU,
    so no approximate index (FAISS) is needed."""

    def __init__(self, gallery: torch.Tensor, names: list[str], device: str = "cuda"):
        self.gallery = F.normalize(gallery, dim=-1).to(device)
        self.names = names
        self.device = device

    @torch.no_grad()
    def search(self, queries: torch.Tensor, topk: int = 10):
        q = F.normalize(queries, dim=-1).to(self.device)
        sims = q @ self.gallery.t()
        scores, idx = sims.topk(topk, dim=-1)
        return scores.cpu(), idx.cpu()

    def guess(self, query: torch.Tensor, topk: int = 5) -> list[tuple[str, float]]:
        scores, idx = self.search(query.unsqueeze(0) if query.ndim == 1 else query, topk)
        return [(self.names[i], float(s)) for s, i in zip(scores[0], idx[0])]


def recall_at_k(pred_idx: torch.Tensor, true_idx: torch.Tensor, ks=(1, 5, 10)) -> dict:
    """pred_idx: (N, K) sorted by score. true_idx: (N,)."""
    hits = pred_idx == true_idx.view(-1, 1)
    return {f"R@{k}": hits[:, :k].any(dim=1).float().mean().item() for k in ks}


def mean_reciprocal_rank(pred_idx: torch.Tensor, true_idx: torch.Tensor) -> float:
    hits = (pred_idx == true_idx.view(-1, 1)).float()
    ranks = hits.argmax(dim=1) + 1
    found = hits.any(dim=1).float()
    return float((found / ranks).mean())


# --------------------------------------------------------------------------
# Baseline without training
# --------------------------------------------------------------------------

def thumbnail_descriptor(img, size: int = 8) -> np.ndarray:
    """8x8 RGB thumbnail, mean-centred and L2-normalised (192 numbers).

    It captures the low-frequency colour layout of the cover, which is most of
    what the mosaic preserves. It sets the floor any trained model has to beat.
    """
    from PIL import Image
    a = np.asarray(img.convert("RGB").resize((size, size), Image.BOX), dtype=np.float32) / 255.0
    v = a.reshape(-1)
    v = v - v.mean()
    n = np.linalg.norm(v) + 1e-8
    return v / n
