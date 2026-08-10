"""
PyTorch Dataset for the BreakHis histopathology dataset (binary: benign vs malignant).

Expected directory layout (as extracted from the Kaggle mirror):
data/raw/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/
    benign/SOB/{tumor_type}/{patient_id}/{magnification}X/*.png
    malignant/SOB/{tumor_type}/{patient_id}/{magnification}X/*.png
"""

import re
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# Root of the extracted BreakHis dataset, relative to project root.
BREAKHIS_ROOT = Path("data/raw/BreaKHis_v1/BreaKHis_v1/histology_slides/breast")

LABEL_MAP = {"benign": 0, "malignant": 1}


@dataclass
class BreakHisRecord:
    path: str
    label: int
    label_name: str
    tumor_type: str
    patient_id: str
    magnification: str


def _extract_patient_id(folder_name: str) -> str:
    """
    Folder names look like 'SOB_B_A_14-22549AB'. The patient ID is
    the token after the last underscore-separated segment, e.g. '14-22549AB'.
    We keep the raw folder name as the patient ID to guarantee uniqueness
    (simpler and safer than trying to parse out just the numeric part).
    """
    return folder_name


def index_breakhis(root: Path = BREAKHIS_ROOT) -> pd.DataFrame:
    """
    Walk the BreakHis directory tree and build a DataFrame with one row
    per image: path, label, tumor_type, patient_id, magnification.
    """
    if not root.exists():
        raise FileNotFoundError(
            f"BreakHis root not found at {root.resolve()}. "
            "Make sure the dataset is extracted into data/raw/ "
            "(see data/README.md)."
        )

    records = []
    for label_name in ("benign", "malignant"):
        label_dir = root / label_name / "SOB"
        if not label_dir.exists():
            continue
        for tumor_type_dir in label_dir.iterdir():
            if not tumor_type_dir.is_dir():
                continue
            for patient_dir in tumor_type_dir.iterdir():
                if not patient_dir.is_dir():
                    continue
                patient_id = _extract_patient_id(patient_dir.name)
                for mag_dir in patient_dir.iterdir():
                    if not mag_dir.is_dir():
                        continue
                    magnification = mag_dir.name  # e.g. "40X", "100X"
                    for img_path in mag_dir.glob("*.png"):
                        records.append(
                            BreakHisRecord(
                                path=str(img_path),
                                label=LABEL_MAP[label_name],
                                label_name=label_name,
                                tumor_type=tumor_type_dir.name,
                                patient_id=patient_id,
                                magnification=magnification,
                            )
                        )

    if not records:
        raise RuntimeError(
            f"No images found under {root.resolve()}. Check the folder structure "
            "matches what's described in data/README.md."
        )

    return pd.DataFrame(records)


class BreakHisDataset(Dataset):
    """
    A PyTorch Dataset over a subset of BreakHis images, defined by a
    DataFrame (as produced by index_breakhis, typically pre-filtered to
    a train/val/test split — see src/data/build_splits.py).
    """

    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, row["label"]
