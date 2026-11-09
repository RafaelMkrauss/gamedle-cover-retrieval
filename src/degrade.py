"""
Simulation of the visual degradation used by Gamedle (classic mode).

WHAT WAS MEASURED (pixel by pixel, on the files served by the site)
--------------------------------------------------------------------
The site does not blur anything in the browser: it serves pre-rendered files,
`<image_id>_p0.webp` ... `<image_id>_p5.webp` (540x720), one per attempt, plus
the final cover `<image_id>.webp`.

  - Background: a MOSAIC of 50x50 px square blocks on a fixed grid starting at
    the top-left corner (blocks on the right and bottom edges are cut). The
    colour of each block is the block's TOP-LEFT PIXEL, not its mean: on the
    first 3 games measured, the corner model has an error of 2-4 (RGB) versus
    ~120 for the block mean. 50 px blocks were found in all 21 games measured.
  - Revealed regions: tiles of 1/3 x 1/3 of the cover (3x3 grid) at full
    resolution on top of the mosaic, one more per attempt. The ORDER changes
    from game to game.

Taking one pixel per block without filtering first is subsampling with
aliasing: a block's colour can come from a tiny detail (an outline, a
highlight) rather than the dominant colour of the area. It is not a low-pass
filter like a blur.

METHOD
------
Several earlier assumptions were wrong: a uniform blur; a fixed tile order seen
in one game; a cursor magnifier in a screenshot read as a game effect; and a
mosaic modelled as the block mean. The last one passed every synthetic test,
because the fake "site" in those tests used the same mosaic operator as the
code (circular validation). Only real data exposed it. Hence: no screenshots
(download the site's files) and no assumed shapes:

  1. MEASURE (`measure_gamedle`): each pixel of the site's image is compared
     with the clean cover and with the mosaic. The result is a MASK per attempt
     with the exact shape of what was revealed, plus a map of what neither
     hypothesis explains.
  2. REUSE (`MaskBank`): masks do not depend on the cover's content. Masks
     measured on N real games form a bank that training applies to any cover,
     so the distribution of positions, sizes and orders comes from data.
  3. While the bank is empty, `random_reveal_masks` generates an explicit guess.
"""

from __future__ import annotations

import io
import math
import random
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Resolution at which masks are stored: 1/4 of the site's. Regions are large,
# so 4 px of precision is plenty.
BANK_SIZE = (135, 180)  # (width, height)
SITE_WIDTH = 540        # width of the site's images; block sizes are measured at this width
BLOCK_PX = 50           # mosaic block on the site (all 21 measured games)


# --------------------------------------------------------------------------
# Primitive operators
# --------------------------------------------------------------------------

def pixelate(img: Image.Image, block_px: float = BLOCK_PX, sampling: str = "corner",
             ref_width: int = SITE_WIDTH) -> Image.Image:
    """Mosaic as rendered by the site: a grid of square `block_px` blocks
    (measured on an image `ref_width` wide), starting at the top-left corner;
    blocks on the right and bottom edges are cut.

    sampling="corner": block colour = top-left pixel of the block (the site).
    sampling="box":    block colour = mean of the block's pixels.

    At other resolutions the block is scaled (50 px at 540 -> ~24.4 px at 264)
    so it covers the same fraction of the cover.
    """
    a = np.asarray(img.convert("RGB"))
    h, w = a.shape[:2]
    b = block_px * w / ref_width
    bx = np.floor(np.arange(w) / b).astype(int)  # block index of each column
    by = np.floor(np.arange(h) / b).astype(int)
    if sampling == "corner":
        cx = np.minimum(w - 1, np.floor(bx * b).astype(int))
        cy = np.minimum(h - 1, np.floor(by * b).astype(int))
        out = a[cy][:, cx]
    elif sampling == "box":
        nbx = int(bx[-1]) + 1
        ids = (by[:, None] * nbx + bx[None, :]).ravel()
        cnt = np.bincount(ids)
        means = np.stack([np.bincount(ids, a[..., k].ravel().astype(np.float64)) / np.maximum(cnt, 1)
                          for k in range(3)], axis=-1)
        out = np.rint(means[ids]).reshape(h, w, 3).astype(np.uint8)
    else:
        raise ValueError(f"unknown sampling: {sampling}")
    return Image.fromarray(np.ascontiguousarray(out))


def compose(img: Image.Image, mask: np.ndarray, block_px: float = BLOCK_PX,
            sampling: str = "corner") -> Image.Image:
    """Clean cover where mask=True, mosaic elsewhere. The mask is resized to the image."""
    mosaic = pixelate(img, block_px, sampling)
    m = Image.fromarray(mask.astype(np.uint8) * 255).resize(img.size, Image.NEAREST)
    return Image.composite(img.convert("RGB"), mosaic, m)


def compress(img: Image.Image, quality: int, fmt: str = "WEBP") -> Image.Image:
    """Re-encodes the image to reproduce the artefacts of the file served by the site (WebP)."""
    buf = io.BytesIO()
    try:
        img.save(buf, format=fmt, quality=int(quality))
    except (OSError, KeyError):  # Pillow without WebP support
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _box(a: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)x(2r+1) window using cumulative sums: O(pixels) for any r."""
    if r <= 0:
        return a
    k = 2 * r + 1
    c = np.pad(np.pad(a.astype(np.float64), r, mode="edge").cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def _arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32) / 255


# --------------------------------------------------------------------------
# Masks: empirical bank + random fallback
# --------------------------------------------------------------------------

class MaskBank:
    """Reveal masks measured on real games. One .npz file per game:
    `masks` [attempts, H, W] and `blocks` [attempts] (mosaic block, px at 540)."""

    def __init__(self, directory: str | Path | None = None):
        self.items: list[tuple[list[np.ndarray], list[int]]] = []
        if directory and Path(directory).is_dir():
            for f in sorted(Path(directory).glob("*.npz")):
                d = np.load(f)
                self.items.append(([m.astype(bool) for m in d["masks"]], [int(b) for b in d["blocks"]]))

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def save(path: str | Path, masks: list[np.ndarray], blocks: list[int]) -> None:
        small = np.stack([
            np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(BANK_SIZE, Image.NEAREST)) > 127
            for m in masks
        ])
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, masks=small.astype(np.uint8), blocks=np.asarray(blocks, dtype=np.int16))


def random_reveal_masks(n_levels: int, rng, size: tuple[int, int] = BANK_SIZE) -> list[np.ndarray]:
    """Fallback guess, used only while the bank is empty. Each attempt adds one
    region to the previous ones:
      - 70%: tiles of a 3x3 grid in random order (what the measured games show);
      - 30%: rectangles of random size and position (covers other shapes).
    """
    w, h = size
    acc = np.zeros((h, w), dtype=bool)
    out = []
    if rng.random() < 0.7:
        tiles = [(r, c) for r in range(3) for c in range(3)]
        rng.shuffle(tiles)
        for r, c in tiles[:n_levels]:
            acc[round(r * h / 3):round((r + 1) * h / 3), round(c * w / 3):round((c + 1) * w / 3)] = True
            out.append(acc.copy())
    else:
        for _ in range(n_levels):
            rw, rh = rng.uniform(0.2, 0.4) * w, rng.uniform(0.2, 0.4) * h
            x0, y0 = rng.uniform(0, w - rw), rng.uniform(0, h - rh)
            acc[int(y0):int(y0 + rh), int(x0):int(x0 + rw)] = True
            out.append(acc.copy())
    while len(out) < n_levels:  # more attempts than grid tiles
        out.append(acc.copy())
    return out


# --------------------------------------------------------------------------
# Degrader
# --------------------------------------------------------------------------

@dataclass
class GamedleParams:
    n_levels: int = 6
    block_px: int = BLOCK_PX   # mosaic block when the bank is empty
    block_jitter: int = 10     # training: 20% of samples vary the block by +-10 px
    p_bank: float = 0.8        # with a bank, fraction of training samples that use a real mask


class GamedleDegrader:
    """Level 0 = first attempt (hardest) ... level 5 = sixth attempt.

    mask_dir: folder with the measured masks. Without it, random shapes are used.
    base_size: (264, 352) is IGDB's `cover_big`, the resolution of the dataset.
    """

    def __init__(self, params: GamedleParams | None = None, base_size: tuple[int, int] = (264, 352),
                 mask_dir: str | Path | None = None):
        self.params = params or GamedleParams()
        self.base_size = base_size
        self.bank = MaskBank(mask_dir)

    def __len__(self) -> int:
        return self.params.n_levels

    @staticmethod
    def _at(seq, level: int):
        return seq[min(level, len(seq) - 1)]

    def _plan(self, rng, use_bank: bool) -> tuple[list[np.ndarray], list[int]]:
        """Draws a "match": the mask and mosaic block size for each attempt."""
        if use_bank and len(self.bank):
            return rng.choice(self.bank.items)
        return random_reveal_masks(self.params.n_levels, rng), [self.params.block_px] * self.params.n_levels

    def apply(self, img: Image.Image, level: int) -> Image.Image:
        """DETERMINISTIC degradation: a given cover always gets the same match
        (the seed is derived from the image content). Used for evaluation, so
        that metrics are comparable across runs and methods."""
        img = img.convert("RGB").resize(self.base_size, Image.LANCZOS)
        rng = random.Random(zlib.crc32(img.resize((16, 16)).tobytes()))
        masks, blocks = self._plan(rng, use_bank=True)
        return compose(img, self._at(masks, level), self._at(blocks, level), "corner")

    def apply_random(
        self,
        img: Image.Image,
        min_level: float = 0.0,
        max_level: float | None = None,
        rng: random.Random | None = None,
    ) -> tuple[Image.Image, float]:
        """Training version. Draws an integer level in [min_level, max_level].

        The level is drawn among integers: drawing a real number and rounding
        down would make the top level almost never appear.

        Augmentation on top of the drawn match:
          - horizontal flip of the mask (doubles the positions in the bank);
          - 20%: mosaic block +-block_jitter px; 15%: block mean instead of the
            corner pixel (robustness in case the site changes);
          - WebP re-compression with variable quality.
        """
        rng = rng or random
        top = len(self) - 1
        lo = max(0, math.ceil(min_level))
        hi = min(top, math.floor(top if max_level is None else max_level))
        if lo <= hi:
            level = rng.randint(lo, hi)
        else:
            level = min(top, max(0, round(top if max_level is None else max_level)))

        masks, blocks = self._plan(rng, use_bank=rng.random() < self.params.p_bank)
        mask = self._at(masks, level)
        if rng.random() < 0.5:
            mask = np.ascontiguousarray(mask[:, ::-1])
        block = self._at(blocks, level)
        if rng.random() < 0.2:
            j = self.params.block_jitter
            block = max(8, block + rng.randint(-j, j))
        sampling = "corner" if rng.random() < 0.85 else "box"

        img = img.convert("RGB").resize(self.base_size, Image.LANCZOS)
        img = compose(img, mask, block, sampling)
        if rng.random() < 0.8:
            img = compress(img, rng.randint(60, 95))
        return img, float(level)


# --------------------------------------------------------------------------
# Measurement against the site
# --------------------------------------------------------------------------

def _regions(new: np.ndarray, min_frac: float = 0.005) -> list[dict]:
    """Describes each connected region: area, bounding box and likely shape.

    fill = area / bounding-box area (circle ~0.785, rectangle 1.0).
    corners = how much of the box CORNERS (outside the inscribed ellipse) is
    filled: ~1 for a rectangle, ~0 for a circle. This is what separates the
    two, and it tolerates small mask defects that would throw `fill` off.
    Regions cut by the image border distort both.
    """
    try:
        from scipy import ndimage
    except ImportError:
        return []
    h, w = new.shape
    # remove thin strips: edges that move a few pixels between attempts show
    # up as "new region" glued to the old ones
    new = ndimage.binary_opening(new, iterations=4)
    lab, n = ndimage.label(new)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        area = int((lab[sl] == i).sum())
        if area < min_frac * h * w:
            continue
        ys, xs = sl
        region = lab[sl] == i
        bh, bw = region.shape
        fill = area / (bh * bw)
        yy, xx = np.mgrid[0:bh, 0:bw]
        outside_ellipse = ((xx + 0.5 - bw / 2) / (bw / 2)) ** 2 + ((yy + 0.5 - bh / 2) / (bh / 2)) ** 2 > 1
        corners = float(region[outside_ellipse].mean()) if outside_ellipse.any() else 1.0
        if corners > 0.5 and fill > 0.8:
            shape = "rectangle?"
        elif corners < 0.2 and 0.70 <= fill <= 0.86:
            shape = "circle?"
        else:
            shape = "irregular"
        out.append({
            "area": area / (h * w),
            "bbox": (xs.start / w, ys.start / h, xs.stop / w, ys.stop / h),
            "fill": fill,
            "corners": corners,
            "shape": shape,
        })
    return out


def measure_gamedle(levels: list[Image.Image], clean: Image.Image, block_range=range(20, 81),
                    qualities=(40, 60, 80, 95)) -> dict:
    """Measures the degradation in the site's real images, assuming no shape.

    levels: `_p0`, `_p1`, ... in order. clean: clean cover of the same game (the
    site's own copy, `<id>.webp`, or IGDB's `t_720p`; both 540x720).

    1. Mosaic: block size and sampling (corner vs. mean) that minimise, pixel
       by pixel, the smaller of the two errors (vs. clean cover, vs. mosaic).
       Searched on the first attempt, which has the most mosaic. The mosaic
       does not change between attempts; it only gets covered.
    2. Codec: the site serves WebP. Comparing that image with uncompressed
       hypotheses confuses "WebP erased the texture" with "region not
       revealed": a smooth grainy area, once compressed, looks like a mosaic
       block. So both hypotheses go through the SAME codec before comparison.
       The site's quality setting is unknown; the one that best explains the
       images is kept.
    3. Mask: a pixel is revealed if it is closer to the (compressed) clean
       cover than to the (compressed) mosaic. Where the hypotheses tie (smooth
       areas, where clean ~ mosaic) the neighbourhood decides.
    4. Unexplained: error far above what compression alone would cause at that
       point. A large patch here means an effect the model does not cover, or a
       clean cover different from the one the site used (other edition, crop).
    """
    size = levels[0].size
    clean = clean.convert("RGB").resize(size, Image.LANCZOS)
    reals = [l.convert("RGB").resize(size) for l in levels]

    R_arrs = [_arr(r) for r in reals]

    # ---- 1. mosaic ---------------------------------------------------------
    C = _arr(clean)
    sq_c0 = ((R_arrs[0] - C) ** 2).mean(-1)

    def mosaic_error(key):
        b, s = key
        M = _arr(pixelate(clean, b, s, ref_width=size[0]))
        return float(np.minimum(sq_c0, ((R_arrs[0] - M) ** 2).mean(-1)).mean())

    block_px, sampling = min(((b, s) for s in ("corner", "box") for b in block_range), key=mosaic_error)
    block_site = block_px * SITE_WIDTH / size[0]  # always stored at the 540 px scale
    mosaic = pixelate(clean, block_site, sampling)
    M = _arr(mosaic)

    # ---- 2. codec quality --------------------------------------------------

    def unexplained_by(q):
        Cq, Mq = _arr(compress(clean, q)), _arr(compress(mosaic, q))
        return sum(float(np.median(np.minimum(((R - Cq) ** 2).mean(-1), ((R - Mq) ** 2).mean(-1))))
                   for R in R_arrs)

    q = min(qualities, key=unexplained_by)
    Cq, Mq = _arr(compress(clean, q)), _arr(compress(mosaic, q))

    # ---- 3 and 4. mask and unexplained, per attempt --------------------------
    per_level, masks, unexplained = [], [], []
    for R in R_arrs:
        ec = _box(((R - Cq) ** 2).mean(-1), 3)
        em = _box(((R - Mq) ** 2).mean(-1), 3)
        best = np.minimum(ec, em)
        floor = float(np.median(best))  # most pixels are explained: the median ~ noise
        tie = np.abs(ec - em) < 3 * floor + 1e-5
        vote = np.where(tie, 0.5, (ec < em).astype(np.float64))
        mask = _box(vote, 6) > 0.5

        # Error that compression alone would cause at each point. It varies
        # within the image (high on edges and textures, ~0 on flat blocks), so
        # no single threshold works; the observed/expected ratio is compared
        # over a larger window, where isolated bad pixels are diluted.
        hyp = compose(clean, mask, block_site, sampling)
        H = _arr(hyp)
        expected = _box(((_arr(compress(hyp, q)) - H) ** 2).mean(-1), 3)
        odd = _box((_box(best, 5) > 6 * _box(expected, 5) + 1e-4).astype(np.float64), 6) > 0.5
        mask &= ~odd
        # CLOSED holes inside a revealed region become revealed: in every
        # measured game the regions are solid, and such holes come from
        # differences between the clean cover used and the site's source (e.g.
        # JPEG artefacts), not from the game. `odd` still records where the
        # measurement was doubtful.
        try:
            from scipy import ndimage
            mask = ndimage.binary_fill_holes(mask)
        except ImportError:
            pass

        masks.append(mask)
        unexplained.append(odd)
        mosaic_area = ~mask & ~odd
        sq_c, sq_m = ((R - C) ** 2).mean(-1), ((R - M) ** 2).mean(-1)
        per_level.append({
            "block_px": round(block_site),
            "sampling": sampling,
            "webp_quality": q,
            "revealed_frac": float(mask.mean()),
            "unexplained_frac": float(odd.mean()),
            "mse_revealed": float(sq_c[mask].mean()) if mask.any() else float("nan"),
            "mse_mosaic": float(sq_m[mosaic_area].mean()) if mosaic_area.any() else float("nan"),
        })

    for k, info in enumerate(per_level):
        prev = masks[k - 1] if k else np.zeros_like(masks[k])
        info["new_regions"] = _regions(masks[k] & ~prev)
        # fraction of the revealed area that DISAPPEARED in the next attempt.
        # Edges move a few pixels (up to ~5%); much larger values mean a whole
        # region vanished
        info["lost_next"] = (float((masks[k] & ~masks[k + 1]).sum() / max(1, masks[k].sum()))
                             if k + 1 < len(masks) else 0.0)

    return {"levels": per_level, "masks": masks, "unexplained": unexplained}
