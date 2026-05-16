
import math
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

WEIGHTS_URL = (
    "https://download.pytorch.org/models/efficientnet_b4_rwightman-7eb33cd5.pth"
)

# ─────────────────────────── scaling helpers ──────────────────────────────────

WIDTH_MULT  = 1.4
DEPTH_MULT  = 1.8
_DIVISOR    = 8


def _round_filters(filters: int) -> int:
    scaled = filters * WIDTH_MULT
    new_f  = max(_DIVISOR, (int(scaled + _DIVISOR / 2) // _DIVISOR) * _DIVISOR)
    if new_f < 0.9 * scaled:
        new_f += _DIVISOR
    return int(new_f)


def _round_repeats(repeats: int) -> int:
    return int(math.ceil(DEPTH_MULT * repeats))


class ConvBNAct(nn.Sequential):

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        bias: bool = False,
    ) -> None:
        if activation_layer is None:
            activation_layer = nn.SiLU
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size,
                      stride=stride, padding=padding,
                      groups=groups, bias=bias),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
            activation_layer(inplace=True),
        )


class SqueezeExcitation(nn.Module):

    def __init__(self, in_ch: int, squeeze_ch: int) -> None:
        super().__init__()
        self._se_reduce = nn.Conv2d(in_ch, squeeze_ch, kernel_size=1)
        self._se_expand = nn.Conv2d(squeeze_ch, in_ch, kernel_size=1)
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=(-2, -1), keepdim=True)       
        scale = self.activation(self._se_reduce(scale))
        scale = torch.sigmoid(self._se_expand(scale))
        return x * scale


@dataclass
class MBConvConfig:
    expand_ratio:   int
    kernel:         int
    stride:         int
    in_channels:    int   
    out_channels:   int    
    num_layers:     int    


class MBConv(nn.Module):

    def __init__(
        self,
        cfg: MBConvConfig,
        stochastic_depth_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.use_res_connect = (
            cfg.stride == 1 and cfg.in_channels == cfg.out_channels
        )

        expanded = cfg.in_channels * cfg.expand_ratio
        se_ch    = max(1, int(cfg.in_channels * 0.25))

        layers: List[nn.Module] = []

        if cfg.expand_ratio != 1:                         
            layers.append(
                ConvBNAct(cfg.in_channels, expanded, kernel_size=1)
            )

        layers += [
            ConvBNAct(                                    
                expanded, expanded,
                kernel_size=cfg.kernel,
                stride=cfg.stride,
                groups=expanded,
            ),
            SqueezeExcitation(expanded, se_ch),              
            nn.Sequential(                                  
                nn.Conv2d(expanded, cfg.out_channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(cfg.out_channels,
                               eps=1e-3, momentum=0.01),
            ),
        ]

        self.block = nn.Sequential(*layers)

        self.stochastic_depth_prob = stochastic_depth_prob

    def _drop_path(
        self, x: torch.Tensor, p: float, training: bool
    ) -> torch.Tensor:
        if p == 0.0 or not training:
            return x
        survival = 1.0 - p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.empty(shape, dtype=x.dtype, device=x.device)
        noise.bernoulli_(survival).div_(survival)
        return x * noise

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result = self._drop_path(
                result, self.stochastic_depth_prob, self.training
            )
            result += x
        return result


class EfficientNetB4(nn.Module):
    num_features: int = 1792

    _STAGE_CFGS = [
        (1, 3, 1,  48,  24, _round_repeats(1)),
        (6, 3, 2,  24,  32, _round_repeats(2)),
        (6, 5, 2,  32,  56, _round_repeats(2)),
        (6, 3, 2,  56, 112, _round_repeats(3)),
        (6, 5, 1, 112, 160, _round_repeats(3)),
        (6, 5, 2, 160, 272, _round_repeats(4)),
        (6, 3, 1, 272, 448, _round_repeats(1)),
    ]

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        total_blocks = sum(c[-1] for c in self._STAGE_CFGS)
        sd_prob      = 0.3      

        stem = ConvBNAct(3, _round_filters(32), kernel_size=3, stride=2)

        stages: List[nn.Sequential] = []
        block_idx = 0
        for er, k, s, in_c, out_c, n in self._STAGE_CFGS:
            stage_blocks: List[nn.Module] = []
            for i in range(n):
                sd = sd_prob * block_idx / (total_blocks - 1)
                cfg = MBConvConfig(
                    expand_ratio=er, kernel=k,
                    stride=s if i == 0 else 1,
                    in_channels=in_c if i == 0 else out_c,
                    out_channels=out_c, num_layers=n,
                )
                stage_blocks.append(MBConv(cfg, stochastic_depth_prob=sd))
                block_idx += 1
            stages.append(nn.Sequential(*stage_blocks))
        head = ConvBNAct(_round_filters(320), self.num_features, kernel_size=1)
        self.features = nn.Sequential(stem, *stages, head)

        self._init_weights()
        if pretrained:
            self._load_pretrained()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    def _load_pretrained(self) -> None:
        state = load_state_dict_from_url(WEIGHTS_URL, progress=True)
        # Drop classifier head
        for key in list(state.keys()):
            if key.startswith("classifier"):
                del state[key]
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            print(f"[EfficientNetB4] Missing keys: {missing}")
        if unexpected:
            print(f"[EfficientNetB4] Unexpected keys (ignored): {unexpected}")


    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return spatial feature map **(B, 1792, H/32, W/32)**."""
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)
