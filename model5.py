
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import swin_t, Swin_T_Weights


class SwinTiny(nn.Module):

    num_features: int = 768
    is_sequence: bool = True

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        if pretrained:
            weights = Swin_T_Weights.IMAGENET1K_V1
            print("[SwinTiny] Loading torchvision pretrained Swin-Tiny backbone.")
        else:
            weights = None
            print("[SwinTiny] Using Swin-Tiny without pretrained weights.")

        model = swin_t(weights=weights)

        self.features = model.features
        self.norm = model.norm
        self.head = nn.Identity()

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:

        x = self.features(x)      # (B, H, W, C)
        x = self.norm(x)          # (B, H, W, C)

        b, h, w, c = x.shape
        x = x.reshape(b, h * w, c)  # (B, N, C)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)