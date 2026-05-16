from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError

import data_module as dm
from data_module import get_transforms
from model import DesignStyleModel
import torch.nn as nn
from model import build_backbone, get_feature_dim
from StyleAwareLoss import StyleAwareLoss
import torchmetrics
import pytorch_lightning as pl
from torchvision.transforms import v2

class OldHeadDesignStyleModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int = 17,
        texture_weight: float = 0.0,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-4,
        pretrained_backbone: bool = False,
        pretrained_path=None,
        freeze_backbone: bool = False,
        dropout: float = 0.30,
        label_smoothing: float = 0.05,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.backbone = build_backbone(
            model_name=model_name,
            pretrained_backbone=pretrained_backbone,
            pretrained_path=pretrained_path,
        )

        self.is_sequence = getattr(self.backbone, "is_sequence", False)
        feature_dim = get_feature_dim(self.backbone)

        if self.is_sequence:
            self.pool = lambda x: x.mean(dim=1)
        else:
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )

        hidden_dim = max(256, feature_dim // 2)

        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout / 2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)
        pooled = self.pool(features)
        logits = self.head(pooled)
        return logits

CONFIG = {
    "sample_submission_path": r"C:\Users\Dell\.cache\kagglehub\competitions\cse-281-spring-26-scene-style-classification\sample_submission.csv",

    "test_dir": r"C:\Users\Dell\.cache\kagglehub\competitions\cse-281-spring-26-scene-style-classification\StyleClassificationIndoors\StyleClassificationIndoors\test",

    "output_path": r"C:\Users\Dell\Downloads\files (1)\ensemble_submission.csv",

    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std":  [0.229, 0.224, 0.225],

    "num_classes": 17,
}

ENSEMBLE_MODELS = [
    # {
    #     "model_name": "convnext_base",
    #     "checkpoint_path": r"C:\Users\Dell\Downloads\files (1)\cse281-scene-classification\d2par44t\checkpoints\best-epoch=14-val_acc=0.4929.ckpt",
    #     "weight": 0.10,
    #     "dropout": 0.30,
    #     "label_smoothing": 0.05,
    #     "head_type": "old",
    # },
    # {
    #     "model_name": "convnext_small",
    #     "checkpoint_path": r"C:\Users\Dell\Downloads\files (1)\cse281-scene-classification\bbet6ws9\checkpoints\best-epoch=23-val_acc=0.5148.ckpt",
    #     "weight": 0.22,
    #     "dropout": 0.30,
    #     "label_smoothing": 0.05,
    #     "head_type": "new",
    # },
    # {
    #     "model_name": "resnet50",
    #     "checkpoint_path": r"C:\Users\Dell\Downloads\files (1)\cse281-scene-classification\nsqa81wt\checkpoints\best-epoch=20-val_acc=0.4517.ckpt",
    #     "weight": 0.14,
    #     "head_type": "new",
    # },
    {
        "model_name": "swin_small",
        "checkpoint_path": r"C:\Users\Dell\Downloads\files (1)\cse281-scene-classification\mdt8u54e\checkpoints\best-epoch=09-val_acc=0.5478.ckpt",
        "weight": 0.35,
        "dropout": 0.30,
        "label_smoothing": 0.1,
        "head_type": "new",
    },
    {
        "model_name": "convnext_base",
        "checkpoint_path": r"C:\Users\Dell\Downloads\files (1)\cse281-scene-classification\3l15ao8m\checkpoints\best-epoch=06-val_acc=0.6033.ckpt",
        "weight": 0.65,
        "dropout": 0.30,
        "label_smoothing": 0.10,
        "head_type": "new",
    },
]


def apply_imagenet_normalization():
    dm.CUSTOM_MEAN = CONFIG["normalization_mean"]
    dm.CUSTOM_STD = CONFIG["normalization_std"]


def find_image_path(test_dir: str, image_id) -> str:
    test_dir = Path(test_dir)
    image_id = str(image_id)

    direct_path = test_dir / image_id
    if direct_path.exists():
        return str(direct_path)

    for ext in [".jpg", ".jpeg", ".png", ".webp"]:
        candidate = test_dir / f"{image_id}{ext}"
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(f"Image not found for ID: {image_id}")


def load_model_from_checkpoint(model_info: dict, device: torch.device):

    ckpt_path = model_info["checkpoint_path"]

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"\n[ensemble] Loading {model_info['model_name']}")
    print(f"[ensemble] Checkpoint: {ckpt_path}")

    head_type = model_info.get("head_type", "new")

    if head_type == "old":
        model = OldHeadDesignStyleModel(
            model_name=model_info["model_name"],
            num_classes=CONFIG["num_classes"],
            texture_weight=0.0,
            label_smoothing=model_info.get("label_smoothing", 0.05),
            backbone_lr=1e-5,
            head_lr=1e-4,
            pretrained_backbone=False,
            pretrained_path=None,
            freeze_backbone=False,
            dropout=model_info.get("dropout", 0.30),
        )
    else:
        model = DesignStyleModel(
            model_name=model_info["model_name"],
            num_classes=CONFIG["num_classes"],
            texture_weight=0.0,
            label_smoothing=model_info.get("label_smoothing", 0.05),
            backbone_lr=1e-5,
            head_lr=1e-4,
            pretrained_backbone=False,
            pretrained_path=None,
            freeze_backbone=False,
            dropout=model_info.get("dropout", 0.30),
        )

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"[ensemble] Missing keys: {missing[:10]}")
    if unexpected:
        print(f"[ensemble] Unexpected keys: {unexpected[:10]}")

    model.eval()
    model.to(device)

    return model


def predict_with_tta(model, image, device, base_transform, tta_transform, tta_steps=5):

    probs_sum = None

    for i in range(tta_steps):
        if i == 0:
            tensor = base_transform(image).unsqueeze(0).to(device)
        else:
            tensor = tta_transform(image).unsqueeze(0).to(device)

        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

        if probs_sum is None:
            probs_sum = probs
        else:
            probs_sum += probs

    return probs_sum / tta_steps


def main():
    apply_imagenet_normalization()

    if not os.path.exists(CONFIG["sample_submission_path"]):
        raise FileNotFoundError(f"sample_submission.csv not found: {CONFIG['sample_submission_path']}")

    if not os.path.exists(CONFIG["test_dir"]):
        raise FileNotFoundError(f"Test directory not found: {CONFIG['test_dir']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ensemble] Device: {device}")

    base_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

    tta_transform = v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(
            size=(224, 224),
            scale=(0.85, 1.0),
            ratio=(0.75, 1.33),
            antialias=True,
        ),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Load all models
    models = []
    total_weight = 0.0

    for model_info in ENSEMBLE_MODELS:
        model = load_model_from_checkpoint(model_info, device)
        models.append((model, float(model_info["weight"])))
        total_weight += float(model_info["weight"])

    print(f"\n[ensemble] Loaded {len(models)} models.")
    print(f"[ensemble] Total ensemble weight: {total_weight}")

    submission_df = pd.read_csv(CONFIG["sample_submission_path"])

    id_col = submission_df.columns[0]
    label_col = submission_df.columns[1]

    print(f"[ensemble] ID column: {id_col}")
    print(f"[ensemble] Label column: {label_col}")
    print(f"[ensemble] Number of test images: {len(submission_df)}")

    predictions = []

    with torch.no_grad():
        for idx, image_id in enumerate(submission_df[id_col]):
            try:
                image_path = find_image_path(CONFIG["test_dir"], image_id)

                image = Image.open(image_path).convert("RGB")

                final_probs = None

                for model, weight in models:
                    probs = predict_with_tta(
                        model=model,
                        image=image,
                        device=device,
                        base_transform=base_transform,
                        tta_transform=tta_transform,
                        tta_steps=7,
                    )

                    if final_probs is None:
                        final_probs = weight * probs
                    else:
                        final_probs += weight * probs

                final_probs = final_probs / total_weight
                pred_idx = torch.argmax(final_probs, dim=1).item()

                # Kaggle sample_submission expects numeric ClassLabel
                predictions.append(pred_idx)

            except (FileNotFoundError, UnidentifiedImageError) as e:
                print(f"[ensemble] Warning: {e}")
                predictions.append(0)

            if (idx + 1) % 100 == 0:
                print(f"[ensemble] Processed {idx + 1}/{len(submission_df)} images")

    submission_df[label_col] = predictions
    submission_df.to_csv(CONFIG["output_path"], index=False)

    print("\n[ensemble] Saved submission CSV:")
    print(CONFIG["output_path"])

    print("\nFirst 10 rows:")
    print(submission_df.head(10))

    print("\nNumber of different predicted classes:")
    print(submission_df[label_col].nunique())


if __name__ == "__main__":
    main()