"""
Generate Grad-CAM overlays for sample test images — including specifically
some the model got wrong, since misclassified examples are often more
informative than correct ones (they show what the model is confusing, or
what part of the image misled it).

Usage:
    python -m src.utils.visualize_gradcam
    python -m src.utils.visualize_gradcam --checkpoint outputs/checkpoints/best_model.pth --num-samples 8
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.data.dataset import LABEL_MAP
from src.data.transforms import get_eval_transforms
from src.models.model import build_model
from src.utils.gradcam import GradCAM

SPLITS_PATH = Path("data/processed/splits.csv")
DEFAULT_CHECKPOINT = Path("outputs/checkpoints/best_model.pth")
OUTPUT_DIR = Path("outputs/logs/gradcam")

LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


def overlay_heatmap(pil_image: Image.Image, cam: np.ndarray, alpha: float = 0.45):
    """Blend a Grad-CAM heatmap (values in [0,1]) over the original image."""
    cmap = plt.get_cmap("jet")
    heatmap = cmap(cam)[:, :, :3]  # drop alpha channel from colormap
    heatmap = (heatmap * 255).astype(np.uint8)

    base = np.array(pil_image.resize(cam.shape[::-1])).astype(np.uint8)
    if base.ndim == 2:  # grayscale safety net
        base = np.stack([base] * 3, axis=-1)

    blended = (alpha * heatmap + (1 - alpha) * base).astype(np.uint8)
    return blended


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--num-samples", type=int, default=8,
                         help="Total images to visualize (split between correct and incorrect predictions).")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}. Train a model first.")
    if not SPLITS_PATH.exists():
        raise FileNotFoundError(f"{SPLITS_PATH} not found. Run build_splits.py first.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(num_classes=2)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    gradcam = GradCAM(model, target_layer=model.layer4[-1])
    transform = get_eval_transforms()

    df = pd.read_csv(SPLITS_PATH)
    test_df = df[df.split == "test"].sample(frac=1, random_state=0).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    correct_saved, incorrect_saved = 0, 0
    target_each = args.num_samples // 2

    for _, row in test_df.iterrows():
        if correct_saved >= target_each and incorrect_saved >= target_each:
            break

        pil_image = Image.open(row["path"]).convert("RGB")
        input_tensor = transform(pil_image).unsqueeze(0).to(device)
        true_label = row["label"]

        cam, predicted_class = gradcam.generate(input_tensor)
        is_correct = (predicted_class == true_label)

        if is_correct and correct_saved >= target_each:
            continue
        if not is_correct and incorrect_saved >= target_each:
            continue

        blended = overlay_heatmap(pil_image, cam)

        status = "correct" if is_correct else "WRONG"
        true_name = LABEL_NAMES[true_label]
        pred_name = LABEL_NAMES[predicted_class]

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(pil_image)
        axes[0].set_title(f"Original\ntrue: {true_name}")
        axes[0].axis("off")
        axes[1].imshow(blended)
        axes[1].set_title(f"Grad-CAM ({status})\npred: {pred_name}")
        axes[1].axis("off")
        plt.tight_layout()

        idx = correct_saved if is_correct else incorrect_saved
        filename = f"{status.lower()}_{idx}_{Path(row['path']).stem}.png"
        out_path = OUTPUT_DIR / filename
        plt.savefig(out_path, dpi=120)
        plt.close(fig)

        print(f"Saved {out_path.name} | true={true_name} pred={pred_name}")

        if is_correct:
            correct_saved += 1
        else:
            incorrect_saved += 1

    gradcam.remove_hooks()
    print(f"\nSaved {correct_saved} correct and {incorrect_saved} incorrect examples to {OUTPUT_DIR.resolve()}")

    if incorrect_saved == 0:
        print(
            "Note: no misclassified examples found in the sampled portion of the test set — "
            "either the model is doing very well, or you may need more samples to find one "
            "(increase --num-samples)."
        )


if __name__ == "__main__":
    main()
