
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url

WEIGHTS_URL = "https://download.pytorch.org/models/convnext_tiny-983f1562.pth"


# ─────────────────────────── building blocks ──────────────────────────────────

class LayerNorm2d(nn.LayerNorm):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)       # BCHW → BHWC
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)    # BHWC → BCHW


class CNBlock(nn.Module):

    def __init__(
        self,
        dim: int,
        layer_scale: float = 1e-6,
        stochastic_depth_prob: float = 0.0,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            _Permute(0, 2, 3, 1),
            nn.LayerNorm(dim, eps=1e-6),
            nn.Linear(dim, 4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(4 * dim, dim, bias=True),
            _Permute(0, 3, 1, 2),
        )

        self.layer_scale = nn.Parameter(
            torch.ones(dim, 1, 1) * layer_scale
        )
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
        result = self.layer_scale * self.block(x)
        result = self._drop_path(result, self.stochastic_depth_prob, self.training)
        return x + result


class _Permute(nn.Module):

    def __init__(self, *dims: int) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims).contiguous()


class ConvNeXtTiny(nn.Module):

    num_features: int = 768

    _STAGE_DEPTHS   = [3, 3, 9, 3]         
    _STAGE_CHANNELS = [96, 192, 384, 768]

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        total_blocks = sum(self._STAGE_DEPTHS)
        sd_prob_max  = 0.1         
        block_idx    = 0

        layers = []
        layers.append(nn.Sequential(
            nn.Conv2d(3, self._STAGE_CHANNELS[0],
                      kernel_size=4, stride=4, padding=0, bias=True),
            LayerNorm2d(self._STAGE_CHANNELS[0], eps=1e-6),
        ))

        for stage_idx, (depth, dim) in enumerate(
            zip(self._STAGE_DEPTHS, self._STAGE_CHANNELS)
        ):
            stage_blocks = []
            for _ in range(depth):
                sd = sd_prob_max * block_idx / max(total_blocks - 1, 1)
                stage_blocks.append(CNBlock(dim, stochastic_depth_prob=sd))
                block_idx += 1
            layers.append(nn.Sequential(*stage_blocks))

            if stage_idx < len(self._STAGE_CHANNELS) - 1:
                next_dim = self._STAGE_CHANNELS[stage_idx + 1]
                layers.append(nn.Sequential(
                    LayerNorm2d(dim, eps=1e-6),
                    nn.Conv2d(dim, next_dim,
                              kernel_size=2, stride=2, padding=0, bias=True),
                ))

        self.features = nn.Sequential(*layers)

        self._init_weights()
        if pretrained:
            self._load_pretrained()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _load_pretrained(self) -> None:
        state = load_state_dict_from_url(WEIGHTS_URL, progress=True)
        for key in list(state.keys()):
            if key.startswith("classifier"):
                del state[key]
        missing, unexpected = self.load_state_dict(state, strict=False)
        if missing:
            print(f"[ConvNeXtTiny] Missing keys: {missing}")
        if unexpected:
            print(f"[ConvNeXtTiny] Unexpected keys (ignored): {unexpected}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return spatial feature map **(B, 768, H/32, W/32)**."""
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)
