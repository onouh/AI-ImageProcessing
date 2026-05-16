
from __future__ import annotations

import torch
import torch.nn as nn
import timm


class SwinSmall(nn.Module):
    num_features: int = 768
    is_sequence: bool = True

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        model_name = "swin_small_patch4_window7_224.ms_in22k_ft_in1k"

        print(f"[SwinSmall] Loading timm model: {model_name}")
        print(f"[SwinSmall] pretrained={pretrained}")

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,    
            global_pool="",     
        )

        self.num_features = int(getattr(self.model, "num_features", 768))

        print(f"[SwinSmall] num_features = {self.num_features}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:

        x = self.model.forward_features(x)

        if x.ndim == 4:
            # (B, H, W, C) -> (B, N, C)
            b, h, w, c = x.shape
            x = x.reshape(b, h * w, c)

        elif x.ndim == 3:
            # Already (B, N, C)
            pass

        else:
            raise RuntimeError(f"Unexpected Swin feature shape: {x.shape}")

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)