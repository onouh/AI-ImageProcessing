from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GramMatrix(nn.Module):

    def forward(self, features: torch.Tensor, is_sequence: bool = False) -> torch.Tensor:
        if is_sequence:
            if features.ndim != 3:
                raise ValueError(
                    f"Expected sequence features (B, N, D), got {tuple(features.shape)}"
                )
            batch_size, num_tokens, feature_dim = features.shape
            flattened = features.transpose(1, 2)             # (B, D, N)
            gram = torch.bmm(flattened, flattened.transpose(1, 2))
            gram = gram / (feature_dim * num_tokens)
            return gram

        if features.ndim != 4:
            raise ValueError(
                f"Expected CNN features (B, C, H, W), got {tuple(features.shape)}"
            )
        batch_size, channels, height, width = features.shape
        flattened = features.reshape(batch_size, channels, height * width)
        gram = torch.bmm(flattened, flattened.transpose(1, 2))
        gram = gram / (channels * height * width)
        return gram


class StyleAwareLoss(nn.Module):

    def __init__(
        self,
        texture_weight: float = 0.0,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.texture_weight  = float(texture_weight)
        self.label_smoothing = float(label_smoothing)
        self.gram    = GramMatrix()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def gram_loss(
        self,
        pred_features: torch.Tensor,
        target_features: torch.Tensor,
        is_sequence: bool = False,
    ) -> torch.Tensor:
        gram_pred   = self.gram(pred_features,   is_sequence=is_sequence)
        gram_target = self.gram(target_features, is_sequence=is_sequence)
        return F.mse_loss(gram_pred, gram_target)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor | None = None,
        target_features: torch.Tensor | None = None,
        is_sequence: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ce = self.ce_loss(logits, labels)

        # CrossEntropy-only mode
        if self.texture_weight == 0.0:
            zero = ce.detach().new_tensor(0.0)
            return ce, {
                "loss/cross_entropy": ce.detach(),
                "loss/texture_gram":  zero,
                "loss/total":         ce.detach(),
            }

        if features is None or target_features is None:
            raise ValueError(
                "features and target_features are required when texture_weight > 0."
            )

        texture = self.gram_loss(features, target_features, is_sequence=is_sequence)
        total   = ce + self.texture_weight * texture

        return total, {
            "loss/cross_entropy": ce.detach(),
            "loss/texture_gram":  texture.detach(),
            "loss/total":         total.detach(),
        }
