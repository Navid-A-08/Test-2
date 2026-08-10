"""
Build patient-level train/val/test splits for BreakHis.

Why patient-level, not image-level:
Each patient contributes many images (different magnifications, adjacent
crops). If the same patient's images appear in both train and validation,
the model can partially memorize per-patient tissue appearance rather than
learning general benign/malignant features — this inflates validation
accuracy in a misleading way. Splitting by patient_id prevents that leakage.

Usage:
    python -m src.data.build_splits
Writes:
    data/processed/splits.csv  (path, label, label_name, tumor_type,
                                 patient_id, magnification, split)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import index_breakhis

OUTPUT_PATH = Path("data/processed/splits.csv")

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# remaining 0.15 goes to test
RANDOM_SEED = 42


def build_patient_level_split(df: pd.DataFrame, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # One row per patient, keeping their label (a patient's images are all
    # the same benign/malignant class, so this is safe/unambiguous).
    patients = df[["patient_id", "label"]].drop_duplicates()

    split_assignment = {}
    # Stratify by label so train/val/test each keep a similar
    # benign/malignant ratio.
    for label, group in patients.groupby("label"):
        ids = group["patient_id"].to_numpy()
        rng.shuffle(ids)

        n = len(ids)
        n_train = int(round(n * TRAIN_FRAC))
        n_val = int(round(n * VAL_FRAC))

        for pid in ids[:n_train]:
            split_assignment[pid] = "train"
        for pid in ids[n_train:n_train + n_val]:
            split_assignment[pid] = "val"
        for pid in ids[n_train + n_val:]:
            split_assignment[pid] = "test"

    df = df.copy()
    df["split"] = df["patient_id"].map(split_assignment)
    return df


def main():
    df = index_breakhis()
    df = build_patient_level_split(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Indexed {len(df)} images from {df['patient_id'].nunique()} patients.")
    print(df.groupby("split")["label_name"].value_counts())
    print(f"\nSaved splits to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
