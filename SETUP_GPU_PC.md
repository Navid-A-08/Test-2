# GPU PC Setup Checklist

Steps to get from a fresh clone to a running training job on a machine
with an NVIDIA GPU.

## 1. Clone the repo
```
git clone https://github.com/Navid-ab-08/Test-2.git
cd Test-2
```
(If already cloned, just `git pull` to get the latest.)

## 2. Create and activate a virtual environment
```
python -m venv venv
venv\Scripts\activate
```

## 3. Install CUDA-enabled PyTorch
Do NOT just run `pip install -r requirements.txt` first — that installs the
CPU-only build of torch/torchvision. Install the CUDA build explicitly:
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
(cu121 = CUDA 12.1. If your GPU driver is older/newer, check
https://pytorch.org/get-started/locally/ for the matching index URL.)

Then install the rest:
```
pip install -r requirements.txt
```

## 4. Verify the GPU is actually detected
```
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
Must print `True` and your GPU's name. If it prints `False`, stop and fix
this before training — do not proceed on CPU.

## 5. Download the BreakHis dataset
Not stored in git (too large, gitignored). Download from:
https://www.kaggle.com/datasets/ambarish/breakhis

Extract so the path looks like:
```
data/raw/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/benign/...
data/raw/BreaKHis_v1/BreaKHis_v1/histology_slides/breast/malignant/...
```

## 6. Build the train/val/test splits
```
python -m src.data.build_splits
```
Expected output: ~7,909 images indexed from 82 patients, split roughly
70/15/15 by patient (not by image — see src/data/build_splits.py for why).

## 7. Run training
```
python -m src.training.train --epochs 15 --batch-size 32 --lr 1e-4
```
- Watch GPU usage (Task Manager > Performance > GPU, or `nvidia-smi` in a
  separate terminal) to confirm it's actually being used, not just detected.
- Checkpoints save to `outputs/checkpoints/best_model.pth`
  (best by validation malignant recall, not accuracy).
- Training log saves to `outputs/logs/train_log.csv`.

## 8. Evaluate on the test set
```
python -m src.training.evaluate
```
Reports accuracy, malignant recall/precision/F1, full classification
report, and saves a confusion matrix plot to
`outputs/logs/test_confusion_matrix.png`.

## 9. Commit results (checkpoints and logs are gitignored by default)
The trained model weights (`.pth`) and logs are excluded from git via
`.gitignore` — they're large binary/output files, not source code. If you
want to keep a record of a specific run's results, copy the printed metrics
into a note or a markdown file and commit that instead of the raw weights.
