"""
Train a ResNet18 binary classifier (benign vs malignant) on BreakHis.

Usage:
    python -m src.training.train --epochs 15 --batch-size 32 --lr 1e-4

Notes on evaluation metric choice:
    Plain accuracy is a poor primary metric here. The classes are imbalanced
    (more malignant than benign in this dataset), and more importantly, in
    a cancer-detection context a false negative (calling a malignant sample
    benign) is far worse than a false positive. So alongside accuracy, we
    track recall (sensitivity) on the malignant class specifically, and use
    it — not raw accuracy — to decide which checkpoint is "best".
"""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from src.data.dataset import BreakHisDataset, LABEL_MAP
from src.data.transforms import get_train_transforms, get_eval_transforms
from src.models.model import build_model

SPLITS_PATH = Path("data/processed/splits.csv")
CHECKPOINT_DIR = Path("outputs/checkpoints")
LOG_PATH = Path("outputs/logs/train_log.csv")

MALIGNANT_LABEL = LABEL_MAP["malignant"]

DEFAULT_SEED = 42


def set_seed(seed: int):
    """
    Make training as reproducible as possible: same seed + same code +
    same data should give the same result. Covers Python's random module,
    NumPy, and PyTorch (CPU and CUDA).

    Note: cudnn.deterministic=True can slow training slightly in exchange
    for reproducibility — an acceptable trade-off while you're developing
    and comparing runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_splits():
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(
            f"{SPLITS_PATH} not found. Run `python -m src.data.build_splits` first."
        )
    df = pd.read_csv(SPLITS_PATH)
    return df[df.split == "train"], df[df.split == "val"], df[df.split == "test"]


def compute_class_weights(train_df: pd.DataFrame, device) -> torch.Tensor:
    """
    Inverse-frequency class weights, so the loss penalizes mistakes on the
    minority class (benign) proportionally more.
    """
    counts = train_df["label"].value_counts().sort_index()  # index 0, 1
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights.values, dtype=torch.float32, device=device)


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "malignant_recall": recall_score(all_labels, all_preds, pos_label=MALIGNANT_LABEL, zero_division=0),
        "malignant_precision": precision_score(all_labels, all_preds, pos_label=MALIGNANT_LABEL, zero_division=0),
        "f1": f1_score(all_labels, all_preds, pos_label=MALIGNANT_LABEL, zero_division=0),
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Random seed set to {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_df, val_df, _ = load_splits()

    train_ds = BreakHisDataset(train_df, transform=get_train_transforms())
    val_ds = BreakHisDataset(val_df, transform=get_eval_transforms())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    model = build_model(num_classes=2, freeze_backbone=args.freeze_backbone).to(device)

    class_weights = compute_class_weights(train_df, device)
    print(f"Class weights (benign, malignant): {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_malignant_recall = -1.0
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        elapsed = time.time() - start
        print(
            f"Epoch {epoch}/{args.epochs} ({elapsed:.1f}s) | "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_malignant_recall={val_metrics['malignant_recall']:.4f} "
            f"val_f1={val_metrics['f1']:.4f}"
        )

        log_rows.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })
        pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)

        if val_metrics["malignant_recall"] > best_malignant_recall:
            best_malignant_recall = val_metrics["malignant_recall"]
            checkpoint_path = CHECKPOINT_DIR / "best_model.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_metrics": val_metrics,
            }, checkpoint_path)
            print(f"  -> New best malignant recall ({best_malignant_recall:.4f}), saved checkpoint.")

    print(f"\nTraining complete. Best val malignant recall: {best_malignant_recall:.4f}")
    print(f"Best checkpoint: {CHECKPOINT_DIR / 'best_model.pth'}")
    print(f"Training log: {LOG_PATH}")


if __name__ == "__main__":
    main()
