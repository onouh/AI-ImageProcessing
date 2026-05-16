
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from StyleAwareLoss import StyleAwareLoss

from model1 import ResNet50
from model2 import EfficientNetB4
from model3 import ConvNeXtSmall
#from model4 import ViTB16
from model5 import SwinTiny
from model6 import ConvNeXtTiny
from model7 import ConvNeXtBase
from model8 import SwinSmall
from model9 import ConvNeXtLarge


SUPPORTED_BACKBONES = {
    "convnext_tiny":   ConvNeXtTiny , 
    "convnext_small":  ConvNeXtSmall,
    "resnet50":        ResNet50,
    "efficientnet_b4": EfficientNetB4,
    "convnext_base": ConvNeXtBase,
    "swin_tiny":       SwinTiny,
    "swin_small":      SwinSmall,
    "convnext_large": ConvNeXtLarge,
}


CLASSIFIER_PREFIXES = (
    "fc.",
    "classifier.",
    "head.",
    "heads.",
)


def _strip_common_prefix(key: str) -> str:
    """Remove common checkpoint prefixes from Lightning/DataParallel checkpoints."""
    for prefix in ("model.", "backbone.", "module."):
        if key.startswith(prefix):
            key = key[len(prefix):]
    return key


def load_backbone_weights_only(backbone: nn.Module, weights_path: str) -> None:
    """
    Loads pretrained weights into the backbone only.

    - Classification-head weights are skipped.
    - Only matching keys with matching tensor shapes are loaded.
    """
    checkpoint = torch.load(weights_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        if "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        elif "model" in checkpoint:
            checkpoint = checkpoint["model"]

    current_state = backbone.state_dict()
    filtered_state = {}
    skipped_classifier = 0
    skipped_mismatch = 0

    for raw_key, value in checkpoint.items():
        key = _strip_common_prefix(raw_key)

        if key.startswith(CLASSIFIER_PREFIXES):
            skipped_classifier += 1
            continue

        if key in current_state and current_state[key].shape == value.shape:
            filtered_state[key] = value
        else:
            skipped_mismatch += 1

    current_state.update(filtered_state)
    backbone.load_state_dict(current_state, strict=True)

    print(f"[model.py] Loaded backbone tensors: {len(filtered_state)}")
    print(f"[model.py] Skipped classifier tensors: {skipped_classifier}")
    print(f"[model.py] Skipped non-matching tensors: {skipped_mismatch}")


def build_backbone(
    model_name: str,
    pretrained_backbone: bool = True,
    pretrained_path: Optional[str] = None,
) -> nn.Module:
    if model_name not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"Unknown model_name='{model_name}'. "
            f"Choose from: {list(SUPPORTED_BACKBONES.keys())}"
        )

    backbone_cls = SUPPORTED_BACKBONES[model_name]

    if pretrained_backbone and pretrained_path is None:
        return backbone_cls(pretrained=True)

    backbone = backbone_cls(pretrained=False)

    if pretrained_backbone and pretrained_path is not None:
        load_backbone_weights_only(backbone, pretrained_path)

    return backbone


def get_feature_dim(backbone: nn.Module) -> int:
    """Support both previous naming styles: num_features and feature_dim."""
    if hasattr(backbone, "num_features"):
        return int(backbone.num_features)
    if hasattr(backbone, "feature_dim"):
        return int(backbone.feature_dim)
    raise AttributeError(
        "Backbone must define either `num_features` or `feature_dim`."
    )


class DesignStyleModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str = "convnext_tiny",
        num_classes: int = 6,
        texture_weight: float = 0.0,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-3,
        pretrained_backbone: bool = True,
        pretrained_path: Optional[str] = None,
        freeze_backbone: bool = True,
        dropout: float = 0.3,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── backbone ──────────────────────────────────────────────────────────
        self.backbone = build_backbone(
            model_name=model_name,
            pretrained_backbone=pretrained_backbone,
            pretrained_path=pretrained_path,
        )

        self.is_sequence = getattr(self.backbone, "is_sequence", False)
        feature_dim = get_feature_dim(self.backbone)
        if self.is_sequence:
            # For ViT/Swin-style models: (B, N, D) → (B, D)
            self.pool = lambda x: x.mean(dim=1)
            pooled_dim = feature_dim
        else:
            # For CNN models: (B, C, H, W) → (B, C)
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            pooled_dim = feature_dim

        hidden_dim = max(512, pooled_dim // 2)

        self.head = nn.Sequential(
            nn.LayerNorm(pooled_dim),

            nn.Dropout(p=dropout),
            nn.Linear(pooled_dim, hidden_dim),
            nn.GELU(),

            nn.Dropout(p=dropout / 2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),

            nn.Dropout(p=dropout / 2),
            nn.Linear(hidden_dim // 2, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

        self.criterion = StyleAwareLoss(
            texture_weight=texture_weight,
            label_smoothing=label_smoothing,
        )

        metric_kwargs = dict(task="multiclass", num_classes=num_classes)
        self.train_acc  = torchmetrics.Accuracy(**metric_kwargs)
        self.val_acc    = torchmetrics.Accuracy(**metric_kwargs)
        self.test_acc   = torchmetrics.Accuracy(**metric_kwargs)
        self.val_f1     = torchmetrics.F1Score(**metric_kwargs, average="macro")
        self.test_f1    = torchmetrics.F1Score(**metric_kwargs, average="macro")


    def freeze_backbone(self) -> None:
        """Freeze pretrained feature extractor; only the new head trains."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.head.parameters():
            param.requires_grad = True
        print("[model.py] Backbone frozen. Training custom head only.")

    def unfreeze_backbone(self) -> None:
        """Unfreeze backbone for careful fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("[model.py] Backbone unfrozen. Fine-tuning full model.")


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        pooled   = self.pool(features)
        logits   = self.head(pooled)
        return logits

    def _shared_step(self, batch: tuple, stage: str):
        images, labels = batch

        features = self.backbone.forward_features(images)
        pooled   = self.pool(features)
        logits   = self.head(pooled)

        if self.hparams.texture_weight > 0.0:
            if stage == "train":
                images_aug = torch.flip(images, dims=[-1])
                with torch.no_grad():
                    target_features = self.backbone.forward_features(images_aug)
            else:
                target_features = features.detach()

            loss, loss_dict = self.criterion(
                logits=logits,
                labels=labels,
                features=features,
                target_features=target_features,
                is_sequence=self.is_sequence,
            )
        else:
            loss, loss_dict = self.criterion(logits=logits, labels=labels)

        for key, value in loss_dict.items():
            self.log(
                f"{stage}/{key}",
                value,
                on_step=(stage == "train"),
                on_epoch=True,
                prog_bar=False,
            )

        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, labels = self._shared_step(batch, "train")
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)
        self.log("train_loss", loss,            on_step=True,  on_epoch=True, prog_bar=True)
        self.log("train_acc",  self.train_acc,  on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, labels = self._shared_step(batch, "val")
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)
        self.log("val_loss", loss,          on_epoch=True, prog_bar=True)
        self.log("val_acc",  self.val_acc,  on_epoch=True, prog_bar=True)
        self.log("val_f1",   self.val_f1,   on_epoch=True, prog_bar=False)

    def test_step(self, batch, batch_idx):
        loss, logits, labels = self._shared_step(batch, "test")
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, labels)
        self.test_f1(preds, labels)
        self.log("test_loss", loss,          on_epoch=True, prog_bar=True)
        self.log("test_acc",  self.test_acc, on_epoch=True, prog_bar=True)
        self.log("test_f1",   self.test_f1,  on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.backbone.parameters(),
                    "lr": self.hparams.backbone_lr,
                    "name": "backbone",
                },
                {
                    "params": self.head.parameters(),
                    "lr": self.hparams.head_lr,
                    "name": "custom_head",
                },
            ],
            weight_decay=1e-5,
        )

        warmup_epochs = 2
        max_epochs = 15

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)

            progress = float(epoch - warmup_epochs) / float(max_epochs - warmup_epochs)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.1415926535))).item()

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambda,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }