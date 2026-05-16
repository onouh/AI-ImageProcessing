
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

WEIGHTS_URL_RESNET = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"

class Bottleneck(nn.Module):
    expansion: int = 4
    def __init__(self, in_channels: int, planes: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class ResNet50(nn.Module):
    num_features: int = 2048

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self._in_channels = 64
        self.conv1   = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1     = nn.BatchNorm2d(64)
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(planes=64,  blocks=3)
        self.layer2 = self._make_layer(planes=128, blocks=4, stride=2)
        self.layer3 = self._make_layer(planes=256, blocks=6, stride=2)
        self.layer4 = self._make_layer(planes=512, blocks=3, stride=2)

        self._init_weights()
        if pretrained:
            self._load_pretrained()

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        downsample = None
        out_channels = planes * Bottleneck.expansion
        if stride != 1 or self._in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self._in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [Bottleneck(self._in_channels, planes, stride, downsample)]
        self._in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._in_channels, planes))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _load_pretrained(self) -> None:
        state = load_state_dict_from_url(WEIGHTS_URL_RESNET, progress=True)
        for key in ("fc.weight", "fc.bias"):
            state.pop(key, None)
        missing, unexpected = self.load_state_dict(state, strict=False)

        print(f"[ResNet50] Missing keys count: {len(missing)}")
        print(f"[ResNet50] Unexpected keys count: {len(unexpected)}")

        if missing:
            print(f"[ResNet50] First missing keys: {missing[:20]}")

        if unexpected:
            print(f"[ResNet50] First unexpected keys: {unexpected[:20]}")

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)