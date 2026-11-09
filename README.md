# Gamedle cover retrieval

A model that plays [Gamedle](https://gamedle.wtf/classic): the site shows a degraded game cover and gives six attempts to name the game, revealing more of the cover each time. The model recognises the game from the degraded cover by searching a gallery of clean covers.

## Problem

There is one official cover per game and new games keep appearing, so this is treated as **retrieval**, not classification. An encoder maps an image to a vector so that

```
embedding(degraded cover of game X)  ≈  embedding(clean cover of game X)
```

and the guess is the nearest clean cover. Adding a game means embedding one image, with no retraining. The open question is how well this works at the first attempts, when only a small part of the cover is visible.

## Data

- **Covers:** 5,512 game covers from the [IGDB API](https://api-docs.igdb.com/) (`cover_big`, 264×352), filtered to main games (no DLC, bundles or ports) with at least 20 ratings; 5,510 after removing near-duplicates.
- **Degradation masks:** measured on the files of 21 real Gamedle rounds (see below).

Neither covers nor masks are included in this repository (covers are copyrighted). The notebook downloads them: create a free Twitch application at <https://dev.twitch.tv/console/apps> and add `TWITCH_ID` and `TWITCH_SECRET` to Colab *Secrets*.

## Approach

### 1. Measuring the degradation instead of guessing it

The first simulations were wrong in four different ways (a uniform blur; a fixed tile order taken from one game; a cursor magnifier in a screenshot read as a game effect; a mosaic coloured by the block mean, which passed every synthetic test because the tests used the same operator as the code). So the degradation is measured from the site's own files:

- the site serves pre-rendered images, one per attempt (`<id>_p0.webp` … `<id>_p5.webp`, 540×720);
- `measure_gamedle` compares every pixel with the clean cover and with a mosaic, after passing both through the same WebP codec, and extracts a mask of what was revealed, plus a map of pixels neither hypothesis explains;
- measurements with more than 15% unexplained pixels are rejected.

Result on 21 games: the background is a **mosaic of 50 px blocks coloured by each block's top-left pixel** (not the mean), and each attempt reveals **one more tile of a 3×3 grid** (~11% of the cover) in an order that changes from game to game. 20 of the 21 games follow this pattern exactly. The measured masks form a bank that training applies to any cover.

### 2. Model

- **Encoder:** ConvNeXt-Base (ImageNet-22k → 1k, `timm`) + GeM pooling + 512-d L2-normalised embedding.
- **Loss:** ArcFace with sub-centres (s=30, m=0.30, K=2) on the degraded and the clean view, plus a **clean→degraded distillation** term `1 − cos(f_degraded, stopgrad(f_clean))`, which does not depend on the training classes and so carries over to unseen games.
- **Curriculum:** from the easiest attempt to the hardest over 10 epochs; ArcFace margin warm-up over 5.
- **Augmentation:** masks from the measured bank (80%) or random tiles, mask flips, mosaic block jitter, WebP re-compression, brightness and saturation.
- **Training:** 4,010 games, 30 epochs, batch 48, AdamW with discriminative learning rates and cosine schedule, mixed precision; 53 minutes on a Colab T4.

### 3. Evaluation protocol

- **Split by game:** 1,500 test games never seen in training. The gallery holds their 1,500 clean covers.
- **Same queries for every method:** the evaluation degradation is deterministic (seeded by the image), and every method sees exactly the same queries.
- **Two conditions:** the attempts exactly as the site renders them, and a *simulated screenshot* of each attempt (crop up to 6% per side, rescaling, brightness, JPEG), which was not used in training.
- **Baselines:** an 8×8 colour thumbnail with no learning, and a frozen DINOv2 ViT-B/14.

## Results

Recall@1 on the 1,500 test games, per attempt (1 = least revealed):

| Method | Condition | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| 8×8 thumbnail (no learning) | game files | 99.0% | 99.3% | 99.7% | 99.9% | 99.9% | 99.9% |
| DINOv2 (frozen) | game files | 14.5% | 42.9% | 65.1% | 82.2% | 91.1% | 96.2% |
| **ConvNeXt + ArcFace + distillation** | game files | **99.7%** | **99.9%** | **99.9%** | **100%** | **100%** | **100%** |
| 8×8 thumbnail (no learning) | simulated screenshot | 97.7% | 98.6% | 99.3% | 99.7% | 99.9% | 99.9% |
| DINOv2 (frozen) | simulated screenshot | 11.5% | 36.2% | 58.1% | 76.0% | 87.5% | 94.2% |
| **ConvNeXt + ArcFace + distillation** | simulated screenshot | **98.7%** | **99.7%** | **99.9%** | **100%** | **100%** | **100%** |

- **The thumbnail is a strong baseline.** The site renders each attempt from the same IGDB image that is in the gallery, and the corner-sampled mosaic keeps the cover's colour layout, so on the site's files the task is close to near-duplicate retrieval. The trained model still cuts the errors at attempt 1 by about two thirds (1.0% → 0.3%).
- **A general-purpose model is not enough.** Frozen DINOv2 reaches only 14.5% at attempt 1: its features rely on texture that the mosaic destroys. Training on the measured degradation is what closes that gap.
- **Robustness.** With simulated screenshots every method drops, and the trained model keeps the lead (98.7% vs 97.7% at attempt 1; 1.3% vs 2.3% errors). The perturbation is mild (crops of up to 6%), so stronger misalignment is the next stress test.

**Errors.** At attempt 1 the model misses 5 of the 1,500 test games, and the correct game is ranked second in 4 of them (e.g. *The Suffering: Ties That Bind* confused with *Blacklight: Retribution*, *Toy Story 2* with *Dicey Dungeons*). The fifth, *Pro Evolution Soccer 2012*, was confused with two *Xenoblade Chronicles* covers that share its colour layout (blue and white above, green below).

**Real round.** On a real Gamedle round, the model ranked *Super Mario 3D Land* first from attempt 1, with similarity rising from 0.49 to 0.87 by attempt 6. The full gallery (5,510 games, training games included) was used, since the game of the day can be any of them.

## How to run

The notebook is meant for Google Colab with a GPU.

1. Clone this repository to Google Drive (or upload the folder) so that `data/` persists between sessions.
2. Add `TWITCH_ID` and `TWITCH_SECRET` to Colab *Secrets*.
3. Open `gamedle.ipynb`, select a T4 GPU and run all cells. The first run downloads the covers, measures the masks and trains (~1 h); later runs reuse `data/manifest.json`, `data/masks/` and `data/ckpt.pt`.

On Kaggle: upload `src/` and `data/` as a private dataset, attach it to the notebook, enable a T4 GPU and internet, and use *Save & Run All*; the notebook copies the dataset to the working directory.

Locally: `pip install -r requirements.txt` and run the notebook with a CUDA GPU.

```
src/
  data.py        IGDB collection, datasets (training pairs, gallery, evaluation queries)
  degrade.py     mosaic and reveal simulation, measurement against the site's files, mask bank
  models.py      encoder (ConvNeXt + GeM), ArcFace head, distillation loss
  train.py       training loop with curriculum and margin warm-up
  retrieval.py   cosine index, Recall@k and MRR, thumbnail baseline
  screenshot.py  simulated screenshots (evaluation) and crop candidates (inference)
```

## Limitations and next steps

- **Close to near-duplicate retrieval on the site's files.** The site renders attempts from the same IGDB image that is in the gallery, which is why a colour thumbnail is a strong baseline. Harder and more realistic tests: screenshots, and different editions or regional covers of the same game.
- **21 measured rounds.** The mask bank covers 21 games; one of them shows a different reveal pattern, so other variants may exist.
- **One cover per game.** Real intra-class variation (several covers per game from IGDB) would be more useful than synthetic augmentation.
- **Hints are not used.** Gamedle reveals attributes such as genre and platform after a wrong guess; filtering the gallery with them is a cheap gain.
- **Next:** train with screenshot-like augmentation if the screenshot condition shows a drop, add a re-ranking stage, and publish the encoder on Hugging Face.
