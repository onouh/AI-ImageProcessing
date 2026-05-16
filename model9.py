
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url


WEIGHTS_URL_22K = "https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth"


class LayerNorm(nn.Module):

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        data_format: str = "channels_last",
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError("data_format must be 'channels_last' or 'channels_first'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            )

        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]

        return x


class DropPath(nn.Module):

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device,
        )
        random_tensor.floor_()

        return x.div(keep_prob) * random_tensor


class Block(nn.Module):

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()

        self.dwconv = nn.Conv2d(
            dim,
            dim,
            kernel_size=7,
            padding=3,
            groups=dim,
        )

        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")

        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones(dim),
                requires_grad=True,
            )
        else:
            self.gamma = None

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x

        x = self.dwconv(x)

        # (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)

        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = self.gamma * x

        # (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)

        x = shortcut + self.drop_path(x)

        return x


class ConvNeXtLarge(nn.Module):

    num_features: int = 1536

    DEPTHS = [3, 3, 27, 3]
    DIMS = [192, 384, 768, 1536]

    def __init__(
        self,
        pretrained: bool = True,
        in_22k: bool = True,
        drop_path_rate: float = 0.3,
        layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv2d(
                3,
                self.DIMS[0],
                kernel_size=4,
                stride=4,
            ),
            LayerNorm(
                self.DIMS[0],
                eps=1e-6,
                data_format="channels_first",
            ),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(
                    self.DIMS[i],
                    eps=1e-6,
                    data_format="channels_first",
                ),
                nn.Conv2d(
                    self.DIMS[i],
                    self.DIMS[i + 1],
                    kernel_size=2,
                    stride=2,
                ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()

        dp_rates = [
            x.item()
            for x in torch.linspace(
                0,
                drop_path_rate,
                sum(self.DEPTHS),
            )
        ]

        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[
                    Block(
                        dim=self.DIMS[i],
                        drop_path=dp_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(self.DEPTHS[i])
                ]
            )
            self.stages.append(stage)
            cur += self.DEPTHS[i]

        self.norm = nn.LayerNorm(self.DIMS[-1], eps=1e-6)

        self._init_weights()

        if pretrained:
            if in_22k:
                self._load_pretrained_22k()
            else:
                raise ValueError(
                    "This ConvNeXtLarge class is intended for ImageNet-22K weights. "
                    "Use in_22k=True."
                )

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _load_pretrained_22k(self) -> None:
        print("[ConvNeXtLarge] Loading ImageNet-22K pretrained weights.")
        print(f"[ConvNeXtLarge] URL: {WEIGHTS_URL_22K}")

        checkpoint = load_state_dict_from_url(
            WEIGHTS_URL_22K,
            map_location="cpu",
            progress=True,
        )

        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state = checkpoint["model"]
        else:
            state = checkpoint

        current_state = self.state_dict()
        filtered_state = {}

        skipped_head = 0
        skipped_mismatch = 0

        for key, value in state.items():
            # Remove ImageNet-22K classification head.
            if key.startswith("head."):
                skipped_head += 1
                continue

            if key in current_state and current_state[key].shape == value.shape:
                filtered_state[key] = value
            else:
                skipped_mismatch += 1

        current_state.update(filtered_state)
        self.load_state_dict(current_state, strict=True)

        print(f"[ConvNeXtLarge] Loaded tensors: {len(filtered_state)}")
        print(f"[ConvNeXtLarge] Skipped head tensors: {skipped_head}")
        print(f"[ConvNeXtLarge] Skipped mismatched tensors: {skipped_mismatch}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:

        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)