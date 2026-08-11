"""
Evaluate a trained checkpoint on the held-out BreakHis test set.

Usage:
    python -m src.training.evaluate
    python -m src.training.evaluate --checkpoint outputs/checkpoints/best_model.pth

Reports accuracy, precision/recall/F1 on the malignant class, and a
confusion matrix. Also saves the confusion matrix as a PNG.

Why we look at more than accuracy:
    A model that just predicts "malignant" for everything can still score
    high accuracy on an imbalanced dataset while being clinically useless.
    The confusion matrix and per-class recall/precision show the actual
    error pattern — specifically, how many malignant cases get missed
    (false negatives), which is the mistake that matters most here.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.dataset import BreakHisDataset, LABEL_MAP
from src.data.transforms import get_eval_transforms
from src.models.model import build_model

SPLITS_PATH = Path("data/processed/splits.csv")
DEFAULT_CHECKPOINT = Path("outputs/checkpoints/best_model.pth")
CONFUSION_MATRIX_PATH = Path("outputs/logs/test_confusion_matrix.png")

LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}  # {0: "benign", 1: "malignant"}
MALIGNANT_LABEL = LABEL_MAP["malignant"]


def evaluate(checkpoint_path: Path, batch_size: int = 32, num_workers: int = 2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not SPLITS_PATH.exists():
        raise FileNotFoundError(f"{SPLITS_PATH} not found. Run build_splits.py first.")
    df = pd.read_csv(SPLITS_PATH)
    test_df = df[df.split == "test"]
    print(f"Test set: {len(test_df)} images from {test_df['patient_id'].nunique()} patients.")

    test_ds = BreakHisDataset(test_df, transform=get_eval_transforms())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    model = build_model(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}")

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    malignant_recall = recall_score(all_labels, all_preds, pos_label=MALIGNANT_LABEL)
    malignant_precision = precision_score(all_labels, all_preds, pos_label=MALIGNANT_LABEL)
    f1 = f1_score(all_labels, all_preds, pos_label=MALIGNANT_LABEL)

    print("\n=== Test Set Results ===")
    print(f"Accuracy:              {acc:.4f}")
    print(f"Malignant recall:      {malignant_recall:.4f}  "
          f"(fraction of actual malignant cases correctly caught)")
    print(f"Malignant precision:   {malignant_precision:.4f}  "
          f"(fraction of malignant predictions that were correct)")
    print(f"Malignant F1:          {f1:.4f}")

    print("\nFull classification report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=[LABEL_NAMES[0], LABEL_NAMES[1]],
    ))

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion matrix (rows = actual, cols = predicted):")
    print(f"                 pred_benign  pred_malignant")
    print(f"actual_benign    {cm[0][0]:<12} {cm[0][1]}")
    print(f"actual_malignant {cm[1][0]:<12} {cm[1][1]}")

    n_false_negatives = cm[1][0]
    if n_false_negatives > 0:
        print(
            f"\nNote: {n_false_negatives} actual malignant case(s) were "
            f"predicted as benign (false negatives) — the clinically "
            f"costliest error type."
        )

    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["benign", "malignant"],
                yticklabels=["benign", "malignant"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Test Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    print(f"\nConfusion matrix plot saved to {CONFUSION_MATRIX_PATH.resolve()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {args.checkpoint}. Train a model first "
            f"with `python -m src.training.train`."
        )

    evaluate(args.checkpoint, args.batch_size, args.num_workers)


if __name__ == "__main__":
    main()
