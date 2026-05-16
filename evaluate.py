from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import wandb

import data_module as dm
from data_module import SceneDataModule
from model import DesignStyleModel

# ==========================================
# CONFIGURATION
# ==========================================
EVAL_CONFIG = {
    "data_dir":   r"D:\cse-281-spring-26-scene-style-classification\StyleClassificationIndoors\StyleClassificationIndoors\train",
    "batch_size": 32,
    "num_workers": 4,
    "normalization_mean": [0.485, 0.456, 0.406],
    "normalization_std":  [0.229, 0.224, 0.225],
    "project":    "cse281-scene-classification",
    "output_dir": "eval_outputs",
    "seed": 42,
}


def apply_imagenet_normalization(config: dict) -> None:
    if hasattr(dm, "CUSTOM_MEAN"):
        dm.CUSTOM_MEAN = config["normalization_mean"]
    if hasattr(dm, "CUSTOM_STD"):
        dm.CUSTOM_STD = config["normalization_std"]


def run_inference(
    model: DesignStyleModel,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run full inference loop. Returns (all_preds, all_labels) as numpy arrays."""
    model.eval()
    model.to(device)

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits = model(images)
            preds  = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


def plot_confusion_matrix(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    output_path: str,
) -> None:
    """Plot and save a normalised confusion matrix."""
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.2), max(5, n * 1.1)))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0, vmax=1,
        linewidths=0.5,
    )
    ax.set_title("Confusion Matrix (row-normalised)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate.py] Confusion matrix saved: {output_path}")


def print_classification_report(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> dict:
    """Print sklearn classification report and return it as a dict."""
    from sklearn.metrics import classification_report, accuracy_score, f1_score

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds, average="macro")

    print("\n" + "=" * 60)
    print(f"  Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Macro F1      : {f1:.4f}")
    print("=" * 60)
    print(classification_report(labels, preds, target_names=class_names, digits=4))

    return {"test_acc": acc, "test_macro_f1": f1}


def main(ckpt_path: str, data_dir: str | None = None):
    pl.seed_everything(EVAL_CONFIG["seed"])

    if data_dir:
        EVAL_CONFIG["data_dir"] = data_dir

    if not os.path.exists(EVAL_CONFIG["data_dir"]):
        raise FileNotFoundError(
            f"Data directory not found: {EVAL_CONFIG['data_dir']}"
        )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    os.makedirs(EVAL_CONFIG["output_dir"], exist_ok=True)

    # ── normalization ──────────────────────────────────────────────────────────
    apply_imagenet_normalization(EVAL_CONFIG)

    # ── data ──────────────────────────────────────────────────────────────────
    datamodule = SceneDataModule(
        data_dir=EVAL_CONFIG["data_dir"],
        batch_size=EVAL_CONFIG["batch_size"],
        num_workers=EVAL_CONFIG["num_workers"],
        seed=EVAL_CONFIG["seed"],
        mixup_alpha=0.0,   # never MixUp during eval
    )
    datamodule.setup()

    try:
        class_names = datamodule.train_ds.features["label"].names
    except Exception:
        n = len(set(datamodule.test_ds["label"]))
        class_names = [f"class_{i}" for i in range(n)]

    print(f"[evaluate.py] Classes: {class_names}")

    # ── model ─────────────────────────────────────────────────────────────────
    print(f"[evaluate.py] Loading checkpoint: {ckpt_path}")
    model = DesignStyleModel.load_from_checkpoint(ckpt_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate.py] Device: {device}")

    # ── inference ─────────────────────────────────────────────────────────────
    preds, labels = run_inference(model, datamodule.test_dataloader(), device)

    # ── report ────────────────────────────────────────────────────────────────
    metrics = print_classification_report(preds, labels, class_names)

    # ── confusion matrix ──────────────────────────────────────────────────────
    cm_path = os.path.join(EVAL_CONFIG["output_dir"], "confusion_matrix.png")
    plot_confusion_matrix(preds, labels, class_names, cm_path)

    # ── W&B logging ───────────────────────────────────────────────────────────
    run_name = Path(ckpt_path).stem + "_eval"
    wandb.init(
        project=EVAL_CONFIG["project"],
        name=run_name,
        job_type="evaluation",
    )
    wandb.log(metrics)
    wandb.log({
        "Confusion Matrix": wandb.Image(cm_path, caption="Row-normalised confusion matrix")
    })

    # W&B table for per-class metrics
    from sklearn.metrics import precision_recall_fscore_support
    prec, rec, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(class_names)))
    )
    table = wandb.Table(columns=["Class", "Precision", "Recall", "F1", "Support"])
    for i, name in enumerate(class_names):
        table.add_data(name, f"{prec[i]:.4f}", f"{rec[i]:.4f}", f"{f1[i]:.4f}", int(support[i]))
    wandb.log({"Per-Class Metrics": table})

    wandb.finish()
    print(f"\n[evaluate.py] ✓ Evaluation complete. Results in: {EVAL_CONFIG['output_dir']}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a scene classification checkpoint.")
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to the best .ckpt file produced by train.py",
    )
    parser.add_argument(
        "--data_dir",
        default=None,
        help="Override the data directory (optional if EVAL_CONFIG is already correct)",
    )
    args = parser.parse_args()
    main(ckpt_path=args.ckpt, data_dir=args.data_dir)
