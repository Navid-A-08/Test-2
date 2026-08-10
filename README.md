# Breast Cancer Detection (Histopathology, Deep Learning)

Binary classification (benign vs. malignant) of breast tissue histopathology
images using a CNN in PyTorch, trained on the BreakHis dataset.

## Status
🚧 Early setup — dataset not yet downloaded, no trained model yet.

## Dataset
[BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
— 7,909 microscopic images of breast tumor tissue (benign/malignant) at
40x, 100x, 200x, and 400x magnification.

Not included in this repo (too large for git). Download instructions in
`data/README.md`.

## Project structure
```
Test-2/
├── data/               # raw + processed data (gitignored, not committed)
│   └── README.md       # how to download/place the dataset
├── notebooks/           # exploratory notebooks
├── src/
│   ├── data/            # dataset loading, splitting, transforms
│   ├── models/           # model architecture definitions
│   ├── training/          # train/eval loops
│   └── utils/             # helpers (metrics, visualization, etc.)
├── outputs/
│   ├── checkpoints/       # saved model weights (gitignored)
│   └── logs/              # training logs (gitignored)
├── requirements.txt
└── README.md
```

## Setup
```bash
python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Scope / current decisions
- Task: binary classification (benign vs. malignant), not 8-class subtype
  classification — chosen to avoid severe class imbalance issues on a
  first pass.
- Framework: PyTorch.
- No stain normalization or WSI-scale tiling needed — BreakHis images are
  already pre-cropped patches, not whole-slide images.

## Disclaimer
This is a learning/research project. It is not a validated medical device
and must not be used for actual clinical diagnosis.
