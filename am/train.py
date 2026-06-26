"""
PROJECT AM v10 — Training Loop
Trains TRM on ARC tasks with deep supervision, EMA, and LR scheduling.
This is the "fuel" that was always missing from previous versions.
"""

from __future__ import annotations

import time
import json
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):
        return iterable

from am import TRMConfig, CHECKPOINT_DIR, LOG_DIR, PAD_TOKEN, NUM_TOKENS
from am.trm import TRM, ARCTRMDataset, EMA
from am.data import (
    load_arc_dataset, encode_task_for_trm,
    augment_dataset, generate_synthetic_tasks, ARCTask,
)


def prepare_training_data(
    tasks: List[ARCTask],
    augment: bool = True,
    n_augments: int = 3,
    synthetic: int = 0,
    max_x: int = 1024,
    max_y: int = 900,
) -> ARCTRMDataset:
    """Prepare encoded dataset from ARC tasks."""
    all_tasks = list(tasks)

    if synthetic > 0:
        print(f"  Generating {synthetic} synthetic tasks...")
        all_tasks.extend(generate_synthetic_tasks(synthetic))

    if augment:
        print(f"  Augmenting {len(all_tasks)} tasks (x{n_augments + 1})...")
        all_tasks = augment_dataset(all_tasks, n_augments=n_augments)

    print(f"  Encoding {len(all_tasks)} tasks for TRM...")
    encoded = []
    skipped = 0
    for task in all_tasks:
        for test_idx in range(len(task.test_pairs)):
            try:
                enc = encode_task_for_trm(task, pair_idx=test_idx)
                if len(enc["x"]) <= max_x and len(enc["y_target"]) <= max_y:
                    encoded.append(enc)
                else:
                    skipped += 1
            except Exception:
                skipped += 1

    if skipped > 0:
        print(f"  Skipped {skipped} tasks (too long)")
    print(f"  Dataset ready: {len(encoded)} examples")
    return ARCTRMDataset(encoded, max_x=max_x, max_y=max_y)


def train(
    config: TRMConfig,
    dataset_version: str = "arc1",
    device: str = "auto",
    resume_from: Optional[str] = None,
    synthetic: int = 200,
) -> Dict[str, Any]:
    """
    Full training pipeline.

    1. Load ARC data
    2. Prepare dataset (encode + augment + synthetic)
    3. Initialize TRM
    4. Train with deep supervision + EMA
    5. Save checkpoints
    6. Return training stats
    """
    # Device selection
    if device == "auto":
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            print(f"  Using GPU: {torch.cuda.get_device_name()}")
        else:
            dev = torch.device("cpu")
            print(f"  Using CPU (training will be slower)")
    else:
        dev = torch.device(device)

    # Load data
    print(f"\n{'='*60}")
    print(f"  TRAINING TRM on {dataset_version.upper()}")
    print(f"{'='*60}")
    print(f"\n  Loading {dataset_version} training data...")
    train_tasks = load_arc_dataset(dataset_version, split="training")

    eval_tasks = None
    try:
        eval_tasks = load_arc_dataset(dataset_version, split="evaluation")
        print(f"  Loaded {len(eval_tasks)} evaluation tasks")
    except FileNotFoundError:
        print("  No evaluation split found (will use train for validation)")

    # Prepare datasets
    print("\n  Preparing training data...")
    train_dataset = prepare_training_data(
        train_tasks,
        augment=True,
        n_augments=3,
        synthetic=synthetic,
        max_x=config.max_x_len,
        max_y=config.max_y_len,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    eval_dataset = None
    eval_loader = None
    if eval_tasks:
        eval_dataset = prepare_training_data(
            eval_tasks[:50],  # use subset for fast eval
            augment=False,
            synthetic=0,
            max_x=config.max_x_len,
            max_y=config.max_y_len,
        )
        eval_loader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize model
    print("\n  Initializing TRM...")
    model = TRM(config).to(dev)

    if resume_from and Path(resume_from).exists():
        print(f"  Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=dev, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        start_epoch = checkpoint.get("epoch", 0)
    else:
        start_epoch = 0

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.lr * 0.01
    )

    # EMA
    ema = EMA(model, decay=config.ema_decay)

    # Training state
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    train_log = []
    best_eval_acc = 0.0
    best_train_loss = float("inf")

    print(f"\n  Training for {config.epochs} epochs...")
    print(f"  Batch size: {config.batch_size}, LR: {config.lr}")
    print(f"  Recursions per forward: {config.total_recursions}")
    print(f"  Dataset size: {len(train_dataset)} examples")
    print(f"  Steps per epoch: {len(train_loader)}")
    print()

    for epoch in range(start_epoch + 1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch in train_loader:
            x = batch["x"].to(dev)
            y_init = batch["y_init"].to(dev)
            y_target = batch["y_target"].to(dev)

            optimizer.zero_grad()
            result = model(x, y_init, targets=y_target)
            loss = result["total_loss"]
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            ema.update(model)

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        # Log
        entry = {
            "epoch": epoch,
            "loss": round(avg_loss, 6),
            "lr": round(scheduler.get_last_lr()[0], 8),
            "time": round(elapsed, 1),
        }

        # Periodic evaluation
        if eval_loader and epoch % 100 == 0:
            eval_acc = evaluate_during_training(model, ema, eval_loader, dev)
            entry["eval_acc"] = round(eval_acc, 4)
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                save_checkpoint(model, ema, optimizer, epoch, config,
                                CHECKPOINT_DIR / "best_eval.pt")
                entry["best"] = True

        # Print progress
        if epoch % 50 == 0 or epoch <= 5:
            parts = [f"Epoch {epoch:4d}/{config.epochs}",
                     f"loss={avg_loss:.4f}",
                     f"lr={scheduler.get_last_lr()[0]:.2e}",
                     f"{elapsed:.1f}s"]
            if "eval_acc" in entry:
                parts.append(f"eval={entry['eval_acc']:.1%}")
                if entry.get("best"):
                    parts.append("★ BEST")
            print(f"  {'  '.join(parts)}")

        train_log.append(entry)

        # ---------------------------------------------------------
        # MADDY'S FIX: SAVE EVERY EPOCH
        # ---------------------------------------------------------
        save_checkpoint(model, ema, optimizer, epoch, config,
                        CHECKPOINT_DIR / "last.pt")

        # Save periodic checkpoints
        if epoch % 500 == 0:
            save_checkpoint(model, ema, optimizer, epoch, config,
                            CHECKPOINT_DIR / f"epoch_{epoch}.pt")

        # Track best loss
        if avg_loss < best_train_loss:
            best_train_loss = avg_loss

    # Final save
    save_checkpoint(model, ema, optimizer, config.epochs, config,
                    CHECKPOINT_DIR / "final.pt")

    # Save log
    log_path = LOG_DIR / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2)

    print(f"\n  Training complete!")
    print(f"  Best train loss: {best_train_loss:.4f}")
    if best_eval_acc > 0:
        print(f"  Best eval accuracy: {best_eval_acc:.1%}")
    print(f"  Checkpoints saved to {CHECKPOINT_DIR}")
    print(f"  Log saved to {log_path}")

    return {
        "best_loss": best_train_loss,
        "best_eval_acc": best_eval_acc,
        "epochs_trained": config.epochs,
        "log": train_log,
    }


def evaluate_during_training(
    model: TRM, ema: EMA, loader: DataLoader, device: torch.device
) -> float:
    """Quick evaluation using EMA weights. Returns fraction of correct tokens."""
    ema.apply(model)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y_init = batch["y_init"].to(device)
            y_target = batch["y_target"].to(device)

            pred = model.predict(x, y_init)

            # Count correct non-padding tokens
            mask = y_target != PAD_TOKEN
            correct += (pred[mask] == y_target[mask]).sum().item()
            total += mask.sum().item()

    ema.restore(model)
    return correct / max(total, 1)


def save_checkpoint(
    model: TRM, ema: EMA, optimizer, epoch: int, config: TRMConfig, path: Path
):
    """Save model checkpoint."""
    torch.save({
        "model_state": model.state_dict(),
        "ema_state": ema.shadow,
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "config": config,
    }, path)


def load_checkpoint(path: str, device: str = "auto") -> tuple:
    """Load model from checkpoint. Returns (model, config)."""
    if device == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)

    checkpoint = torch.load(path, map_location=dev, weights_only=False)
    config = checkpoint["config"]
    model = TRM(config).to(dev)
    model.load_state_dict(checkpoint["model_state"])

    # Load EMA weights for evaluation
    if "ema_state" in checkpoint:
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in checkpoint["ema_state"]:
                    p.copy_(checkpoint["ema_state"][name])

    return model, config
