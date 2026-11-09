"""
Screenshots: simulate them (for evaluation) and crop them (for inference).

The site serves each attempt as an image file, and the measurement and the
main evaluation use those files. A player, however, usually has a screenshot:
the cover cut by hand, rescaled by the screen and re-encoded. These helpers
cover both directions.
"""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageEnhance


def screenshot_like(img: Image.Image, seed: int, max_crop: float = 0.06) -> Image.Image:
    """Deterministic "screenshot" of an image: imprecise crop (up to `max_crop`
    of each side), on-screen rescaling, brightness change and JPEG re-encoding.

    None of these transformations was used in training, so this measures
    robustness outside the training distribution.
    """
    rng = random.Random(seed)
    img = img.convert("RGB")
    w, h = img.size
    left, top, right, bottom = (rng.uniform(0, max_crop) for _ in range(4))
    img = img.crop((round(left * w), round(top * h), round(w - right * w), round(h - bottom * h)))
    scale = rng.uniform(0.6, 1.5)
    img = img.resize((max(32, round(img.width * scale)), max(32, round(img.height * scale))), Image.BILINEAR)
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.9, 1.1))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=rng.randint(60, 90))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _differs_from_background(img: Image.Image, tol: int = 40) -> np.ndarray:
    """Pixels that differ from the background colour (the typical colour of the image border)."""
    a = np.asarray(img.convert("RGB")).astype(np.int16)
    background = np.median(np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]), axis=0)
    return np.abs(a - background).sum(-1) > tol


def trim_borders(img: Image.Image, frac: float = 0.02) -> Image.Image:
    """Cuts strips along the edges that have the background colour."""
    differs = _differs_from_background(img)
    rows = np.where(differs.mean(1) > frac)[0]
    cols = np.where(differs.mean(0) > frac)[0]
    if len(rows) < 32 or len(cols) < 24:
        return img
    return img.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))


def cover_shaped_regions(img: Image.Image, max_regions: int = 4) -> list[Image.Image]:
    """Finds page blocks that are not background (cover, bars, buttons...) and
    keeps the largest ones with a cover-like aspect ratio (width/height 0.4-1.2)."""
    from scipy import ndimage
    differs = ndimage.binary_closing(_differs_from_background(img), iterations=3)
    lab, _ = ndimage.label(differs)
    total = differs.size
    boxes = []
    for ys, xs in ndimage.find_objects(lab):
        w, h = xs.stop - xs.start, ys.stop - ys.start
        if w * h >= 0.01 * total and 0.4 <= w / h <= 1.2:
            boxes.append((w * h, (xs.start, ys.start, xs.stop, ys.stop)))
    return [img.crop(box) for _, box in sorted(boxes, reverse=True)[:max_regions]]


def crop_candidates(img: Image.Image) -> list[Image.Image]:
    """Candidate crops of a screenshot: the whole image, the image without
    borders, the cover-shaped regions, and 3:4 windows (the covers' aspect
    ratio) at 3 scales x 9 positions."""
    if max(img.size) > 1600:
        img = img.copy()
        img.thumbnail((1600, 1600))
    base = trim_borders(img)
    cands = [img, base] + cover_shaped_regions(img)
    W, H = base.size
    for scale in (1.0, 0.8, 0.6):
        h = min(H, W * 4 / 3) * scale
        w = h * 3 / 4
        for fy in (0.0, 0.5, 1.0):
            for fx in (0.0, 0.5, 1.0):
                x0, y0 = (W - w) * fx, (H - h) * fy
                cands.append(base.crop((round(x0), round(y0), round(x0 + w), round(y0 + h))))
    return cands
