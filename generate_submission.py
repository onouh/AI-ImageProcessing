from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError

import data_module as dm
from data_module import get_transforms
from model import DesignStyleModel

CONFIG = {
    # Put your BEST checkpoint here
    "checkpoint_path": r"C:\Users\Dell\Downloads\files (1)\cse281-scene-classification\8go2iciy\checkpoints\best-epoch=24-val_acc=0.5025.ckpt",

    # Kaggle sample submission file
    "sample_submission_path": r"C:\Users\Dell\.cache\kagglehub\competitions\cse-281-spring-26-scene-style-classification\sample_submission.csv",

    # Kaggle test images folder
    "test_dir": r"C:\Users\Dell\.cache\kagglehub\competitions\cse-281-spring-26-scene-style-classification\StyleClassificationIndoors\StyleClassificationIndoors\test",

    # Output CSV file
    "output_path": r"C:\Users\Dell\Downloads\files (1)\submission.csv",

    # Same ImageNet normalization used during training
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std":  [0.229, 0.224, 0.225],
}

CLASS_NAMES = [
    "asian",
    "boho",
    "coastal",
    "contemporary",
    "craftsman",
    "eclectic",
    "farmhouse",
    "french-country",
    "industrial",
    "mediterranean",
    "minimalist",
    "modern",
    "scandinavian",
    "shabby-chic-style",
    "southwestern",
    "tropical",
    "victorian",
]


def apply_imagenet_normalization():
    dm.CUSTOM_MEAN = CONFIG["normalization_mean"]
    dm.CUSTOM_STD = CONFIG["normalization_std"]


def find_image_path(test_dir: str, image_id) -> str:
    """
    Finds the test image path.

    Handles cases like:
        123.jpg
        123
        image_123.png
    """
    test_dir = Path(test_dir)
    image_id = str(image_id)

    # Case 1: image_id already includes extension
    direct_path = test_dir / image_id
    if direct_path.exists():
        return str(direct_path)

    # Case 2: image_id needs extension
    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        candidate = test_dir / f"{image_id}{ext}"
        if candidate.exists():
            return str(candidate)

    # Case 3: search recursively as a fallback
    matches = list(test_dir.rglob(image_id))
    if matches:
        return str(matches[0])

    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        matches = list(test_dir.rglob(f"{image_id}{ext}"))
        if matches:
            return str(matches[0])

    raise FileNotFoundError(f"Image not found for ID: {image_id}")

def predict_with_tta(model, tensor):

    logits_original = model(tensor)

    flipped_tensor = torch.flip(tensor, dims=[-1])
    logits_flipped = model(flipped_tensor)

    avg_logits = (logits_original + logits_flipped) / 2.0

    return avg_logits

def main():
    apply_imagenet_normalization()

    if not os.path.exists(CONFIG["checkpoint_path"]):
        raise FileNotFoundError(f"Checkpoint not found: {CONFIG['checkpoint_path']}")

    if not os.path.exists(CONFIG["sample_submission_path"]):
        raise FileNotFoundError(f"sample_submission.csv not found: {CONFIG['sample_submission_path']}")

    if not os.path.exists(CONFIG["test_dir"]):
        raise FileNotFoundError(f"Test directory not found: {CONFIG['test_dir']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[submission] Device: {device}")

    print("[submission] Loading checkpoint:")
    print(CONFIG["checkpoint_path"])

    model = DesignStyleModel.load_from_checkpoint(CONFIG["checkpoint_path"])
    model.eval()
    model.to(device)

    transform = get_transforms("val")

    submission_df = pd.read_csv(CONFIG["sample_submission_path"])

    id_col = submission_df.columns[0]
    label_col = submission_df.columns[1]

    print(f"[submission] ID column: {id_col}")
    print(f"[submission] Label column: {label_col}")
    print(f"[submission] Number of test images: {len(submission_df)}")
    print(f"[submission] Class names: {CLASS_NAMES}")

    predictions = []

    with torch.no_grad():
        for image_id in submission_df[id_col]:
            try:
                image_path = find_image_path(CONFIG["test_dir"], image_id)

                image = Image.open(image_path).convert("RGB")
                tensor = transform(image).unsqueeze(0).to(device)

                logits = predict_with_tta(model, tensor)
                pred_idx = torch.argmax(logits, dim=1).item()



                pred_label = CLASS_NAMES[pred_idx]

                predictions.append(pred_idx)

            except (FileNotFoundError, UnidentifiedImageError) as e:
                print(f"[submission] Warning: {e}")
                predictions.append(0)

    submission_df[label_col] = predictions

    submission_df.to_csv(CONFIG["output_path"], index=False)

    print("\n[submission] Saved CSV:")
    print(CONFIG["output_path"])

    print("\nFirst 10 rows:")
    print(submission_df.head(10))

    print("\nNumber of different predicted classes:")
    print(submission_df[label_col].nunique())


if __name__ == "__main__":
    main()