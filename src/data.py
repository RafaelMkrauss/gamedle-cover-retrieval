"""
Dataset collection (IGDB) and PyTorch datasets.

IGDB has an official cover for each game, a free API (Twitch credentials) and an
`image_id` field that builds the cover URL at several resolutions.

API v4 limits: 4 requests/second, at most 8 open requests.
Docs: https://api-docs.igdb.com/
"""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import torch
from PIL import Image
from torch.utils.data import Dataset

from .degrade import GamedleDegrader
from .screenshot import screenshot_like

IGDB_IMG = "https://images.igdb.com/igdb/image/upload/t_{size}/{image_id}.jpg"


# --------------------------------------------------------------------------
# 1. Collection
# --------------------------------------------------------------------------

def get_twitch_token(client_id: str, client_secret: str) -> str:
    r = requests.post(
        "https://id.twitch.tv/oauth2/token",
        params={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "client_credentials",
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def _igdb_post(endpoint: str, headers: dict, query: str, retries: int = 5) -> list[dict]:
    """POST to IGDB. Waits and retries on 429 (rate limit), up to `retries` times."""
    for _ in range(retries):
        r = requests.post(
            f"https://api.igdb.com/v4/{endpoint}", headers=headers, data=query, timeout=60
        )
        if r.status_code == 429:
            time.sleep(2.0)
            continue
        if not r.ok:
            raise RuntimeError(f"IGDB /{endpoint} returned HTTP {r.status_code}: {r.text[:500]}")
        return r.json()
    raise RuntimeError(f"IGDB /{endpoint}: rate limit (429) persisted after {retries} attempts")


def main_game_type_id(headers: dict) -> int:
    """Id of the "Main Game" type, queried from the API instead of hard-coded.

    IGDB deprecated `games.category` (where MAIN_GAME was 0) in favour of
    `games.game_type`. Filtering on the deprecated field does not raise an
    error: the API answers 200 with an empty list. That is what made the first
    version of this pipeline download zero games.
    """
    types = _igdb_post("game_types", headers, "fields id,type; limit 100;")
    for t in types:
        if t.get("type", "").lower().replace("_", "").replace(" ", "") == "maingame":
            return int(t["id"])
    raise RuntimeError(f"'Main Game' type not found in /game_types. Received: {types}")


def fetch_games(
    client_id: str,
    token: str,
    n_games: int = 8000,
    min_rating_count: int = 20,
    page_size: int = 500,
    sleep: float = 0.3,
) -> list[dict]:
    """Downloads metadata for the most-rated games that have a cover.

    `min_rating_count` filters out obscure games, whose near-identical covers
    (asset flips, shovelware) would raise the error rate without the model
    being any worse.

    `game_type = Main Game` excludes DLC, bundles and ports, which usually reuse
    the base game's cover: the same image under different labels.
    """
    headers = {"Client-ID": client_id, "Authorization": f"Bearer {token}"}
    main_id = main_game_type_id(headers)
    where = f"cover != null & rating_count >= {min_rating_count} & game_type = {main_id}"

    out: list[dict] = []
    offset = 0
    while len(out) < n_games:
        query = (
            "fields id,name,first_release_date,cover.image_id,genres.name,platforms.abbreviation;"
            f" where {where}; sort rating_count desc; limit {page_size}; offset {offset};"
        )
        batch = _igdb_post("games", headers, query)
        if not batch:
            break
        out.extend(batch)
        offset += page_size
        time.sleep(sleep)  # stay under 4 req/s

    if not out:
        raise RuntimeError(
            "The API answered, but no game passed the filter:\n"
            f"    where {where};\n"
            "Relax one condition at a time (e.g. min_rating_count=0) to find the one that empties the result."
        )
    return out[:n_games]


def download_covers(
    games: list[dict],
    out_dir: str | Path,
    size: str = "cover_big",
    workers: int = 8,
) -> list[dict]:
    """Downloads the covers in parallel. Returns the records that succeeded."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def one(g: dict) -> dict | None:
        image_id = g.get("cover", {}).get("image_id")
        if not image_id:
            return None
        path = out_dir / f"{g['id']}.jpg"
        if not path.exists():
            url = IGDB_IMG.format(size=size, image_id=image_id)
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                path.write_bytes(resp.content)
            except Exception:
                return None
        return {
            "igdb_id": g["id"],
            "name": g["name"],
            "path": str(path),
            "genres": [x["name"] for x in g.get("genres", [])],
            "platforms": [x.get("abbreviation", "") for x in g.get("platforms", [])],
            "year": (
                time.gmtime(g["first_release_date"]).tm_year
                if g.get("first_release_date") else None
            ),
        }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        records = list(ex.map(one, games))
    return [r for r in records if r]


def save_manifest(records: list[dict], path: str | Path) -> None:
    Path(path).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# 2. Datasets
# --------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def to_tensor(img: Image.Image, size: int = 224) -> torch.Tensor:
    import torchvision.transforms.functional as TF
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    x = TF.to_tensor(img)
    return TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)


class CoverPairDataset(Dataset):
    """Returns (degraded_view, clean_view, label, level).

    The clean view is used for distillation: the teacher sees the sharp cover,
    the student sees the degraded one, and the two embeddings are pulled
    together. Same idea as low-resolution face recognition: align the degraded
    feature with the clean feature of the same identity instead of only
    classifying.
    """

    def __init__(
        self,
        records: list[dict],
        degrader: GamedleDegrader | None = None,
        img_size: int = 224,
        train: bool = True,
        min_level: float = 0.0,
        max_level: float | None = None,
        return_clean: bool = True,
    ):
        self.records = records
        self.degrader = degrader or GamedleDegrader()
        self.img_size = img_size
        self.train = train
        self.min_level = min_level
        self.max_level = max_level
        self.return_clean = return_clean
        # label = contiguous index; this map translates model output -> game.
        self.id2label = {r["igdb_id"]: i for i, r in enumerate(records)}

    def __len__(self) -> int:
        return len(self.records)

    def set_curriculum(self, min_level: float, max_level: float | None = None) -> None:
        """Called once per epoch to tighten the curriculum."""
        self.min_level = min_level
        self.max_level = max_level

    def _photometric(self, img: Image.Image) -> Image.Image:
        """Light augmentation applied after the degradation (screen/capture variation)."""
        from PIL import ImageEnhance
        if random.random() < 0.5:
            img = ImageEnhance.Brightness(img).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.5:
            img = ImageEnhance.Color(img).enhance(random.uniform(0.85, 1.15))
        return img

    def __getitem__(self, i: int):
        rec = self.records[i]
        clean = Image.open(rec["path"]).convert("RGB")
        label = self.id2label[rec["igdb_id"]]

        if self.train:
            deg, level = self.degrader.apply_random(
                clean, min_level=self.min_level, max_level=self.max_level
            )
            deg = self._photometric(deg)
        else:
            level = float(self.min_level)
            deg = self.degrader.apply(clean, int(level))

        x_deg = to_tensor(deg, self.img_size)
        if not self.return_clean:
            return x_deg, label, level
        x_clean = to_tensor(clean, self.img_size)
        return x_deg, x_clean, label, level


class GalleryDataset(Dataset):
    """Clean covers only: the index that queries are searched against."""

    def __init__(self, records: list[dict], img_size: int = 224):
        self.records = records
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        rec = self.records[i]
        img = Image.open(rec["path"]).convert("RGB")
        return to_tensor(img, self.img_size), i


class QueryDataset(Dataset):
    """Evaluation queries: attempt `level` of each cover (deterministic), optionally
    passed through a simulated screenshot. `make` is also used by the
    thumbnail baseline, so every method sees exactly the same queries."""

    def __init__(self, records: list[dict], degrader: GamedleDegrader, level: int,
                 screenshot: bool = False, img_size: int = 224):
        self.records = records
        self.degrader = degrader
        self.level = level
        self.screenshot = screenshot
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.records)

    def make(self, img: Image.Image, i: int) -> Image.Image:
        out = self.degrader.apply(img, self.level)
        if self.screenshot:
            out = screenshot_like(out, seed=i)
        return out

    def __getitem__(self, i: int):
        img = Image.open(self.records[i]["path"]).convert("RGB")
        return to_tensor(self.make(img, i), self.img_size), i
