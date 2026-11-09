"""
Training loop: ArcFace + clean->degraded distillation + degradation curriculum.

The loss has three terms:

    L = L_arc(degraded) + w_clean * L_arc(clean) + w_distill * (1 - cos(f_deg, sg(f_clean)))

  - L_arc(degraded): the task itself.
  - L_arc(clean): anchor. Without it the space can drift towards a
    "degraded-only" regime, and the gallery (which holds clean covers) ends up
    misaligned with the queries.
  - distillation: explicitly aligns the two views; it matters most at the
    hardest levels.

Curriculum: training starts at the easiest level (most of the cover revealed)
and opens up to the hardest ones over `curriculum_epochs`. Early on the model
has no useful features, and mosaic-only inputs give mostly noisy gradients.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models import cosine_distill_loss


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 64
    lr_backbone: float = 1e-4      # pretrained backbone: low LR
    lr_head: float = 1e-3          # neck + ArcFace: higher LR (trained from scratch)
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    margin_final: float = 0.30
    margin_warmup_epochs: int = 5
    w_clean: float = 0.5
    w_distill: float = 1.0
    label_smoothing: float = 0.1
    grad_clip: float = 5.0
    amp: bool = True
    curriculum_epochs: int = 10    # epochs until the hardest level is included


def build_optimizer(model, head, cfg: TrainConfig):
    """Discriminative learning rates: pretrained weights move slowly, new layers fast."""
    return torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": cfg.lr_backbone},
            {"params": list(model.pool.parameters()) + list(model.neck.parameters()),
             "lr": cfg.lr_head},
            {"params": head.parameters(), "lr": cfg.lr_head},
        ],
        weight_decay=cfg.weight_decay,
    )


def cosine_schedule_with_warmup(optimizer, cfg: TrainConfig, steps_per_epoch: int):
    total = cfg.epochs * steps_per_epoch
    warm = cfg.warmup_epochs * steps_per_epoch

    def fn(step: int) -> float:
        if step < warm:
            return step / max(1, warm)
        p = (step - warm) / max(1, total - warm)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)


def train(
    model,
    head,
    dataset,
    cfg: TrainConfig | None = None,
    device: str = "cuda",
    num_workers: int = 2,
    log_every: int = 50,
    on_epoch_end=None,
):
    cfg = cfg or TrainConfig()
    model.to(device); head.to(device)

    loader = DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0,
    )
    opt = build_optimizer(model, head, cfg)
    sched = cosine_schedule_with_warmup(opt, cfg, len(loader))
    scaler = torch.amp.GradScaler(device.split(":")[0], enabled=cfg.amp)
    ce = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    n_levels = len(dataset.degrader) - 1
    step = 0
    history = []

    for epoch in range(cfg.epochs):
        # curriculum: from "easy only" to "all levels"
        prog = min(1.0, epoch / max(1, cfg.curriculum_epochs))
        dataset.set_curriculum(min_level=(1.0 - prog) * n_levels, max_level=n_levels)

        # margin warm-up
        head.set_margin(cfg.margin_final * min(1.0, (epoch + 1) / cfg.margin_warmup_epochs))

        model.train(); head.train()
        agg = {"loss": 0.0, "arc": 0.0, "dst": 0.0, "acc": 0.0, "n": 0}
        t0 = time.time()

        for x_deg, x_clean, y, _lvl in loader:
            x_deg = x_deg.to(device, non_blocking=True)
            x_clean = x_clean.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast(device_type=device.split(":")[0], enabled=cfg.amp):
                # a single forward pass over both views: faster, and BatchNorm
                # statistics stay consistent between them.
                f = model(torch.cat([x_deg, x_clean], dim=0))
                f_deg, f_clean = f.chunk(2, dim=0)

                loss_arc = ce(head(f_deg, y), y)
                loss_clean = ce(head(f_clean, y), y)
                loss_dst = cosine_distill_loss(f_deg, f_clean)
                loss = loss_arc + cfg.w_clean * loss_clean + cfg.w_distill * loss_dst

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), cfg.grad_clip
            )
            scaler.step(opt); scaler.update(); sched.step()

            with torch.no_grad():
                acc = (head(f_deg.float(), y).argmax(1) == y).float().mean().item()
            bs = y.size(0)
            agg["loss"] += loss.item() * bs; agg["arc"] += loss_arc.item() * bs
            agg["dst"] += loss_dst.item() * bs; agg["acc"] += acc * bs; agg["n"] += bs

            step += 1
            if step % log_every == 0:
                n = agg["n"]
                print(f"  step {step:6d} | loss {agg['loss']/n:.4f} "
                      f"arc {agg['arc']/n:.4f} dst {agg['dst']/n:.4f} acc {agg['acc']/n:.3f}")

        n = max(1, agg["n"])
        row = {
            "epoch": epoch,
            "loss": agg["loss"] / n, "arc": agg["arc"] / n,
            "distill": agg["dst"] / n, "train_acc": agg["acc"] / n,
            "margin": head.margin, "min_level": dataset.min_level,
            "secs": time.time() - t0,
        }
        history.append(row)
        print(f"[epoch {epoch:3d}] loss={row['loss']:.4f} acc={row['train_acc']:.3f} "
              f"margin={row['margin']:.3f} min_level={row['min_level']:.2f} "
              f"({row['secs']:.0f}s)")

        if on_epoch_end is not None:
            on_epoch_end(epoch, model, head, history)

    return history


def save_checkpoint(path, model, head, records, cfg: TrainConfig, history):
    torch.save(
        {
            "model": model.state_dict(),
            "head": head.state_dict(),
            "records": records,
            "config": asdict(cfg),
            "history": history,
        },
        path,
    )
