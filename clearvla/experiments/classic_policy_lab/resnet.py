from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Callable

import torch
from torch import nn


class FrozenBatchNorm2d(nn.Module):
    """Frozen affine BatchNorm used by the official ACT DETR backbone."""

    def __init__(self, num_features: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        state_dict.pop(prefix + "num_batches_tracked", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        scale = weight * (running_var + self.eps).rsqrt()
        return x * scale + (bias - running_mean * scale)


def group_norm_32(num_channels: int) -> nn.GroupNorm:
    # DP replaces BatchNorm with GroupNorm. Keep group count valid for all stages.
    groups = max(1, min(32, num_channels // 16))
    while num_channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, num_channels)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = int(stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet18(nn.Module):
    """Self-contained torchvision-compatible ResNet-18.

    torchvision cannot be assumed to import cleanly in every robotics runtime.
    This implementation keeps the standard module names so torchvision ResNet-18
    state dictionaries can be loaded after removing an optional ``fc`` prefix.
    """

    def __init__(self, *, norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d) -> None:
        super().__init__()
        self._norm_layer = norm_layer
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1000)
        self._reset_parameters()

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride=stride, bias=False),
                norm_layer(planes),
            )
        layers = [
            BasicBlock(
                self.inplanes, planes, stride=stride, downsample=downsample, norm_layer=norm_layer
            )
        ]
        self.inplanes = planes
        layers.extend(
            BasicBlock(self.inplanes, planes, norm_layer=norm_layer) for _ in range(1, blocks)
        )
        return nn.Sequential(*layers)

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def load_resnet18_weights(model: ResNet18, path: Path | None) -> tuple[list[str], list[str]]:
    if path is None:
        return [], []
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a state dict in {path}, got {type(payload)!r}")
    cleaned: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in payload.items():
        key = str(key)
        for prefix in ("module.", "model.", "backbone."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        cleaned[key] = value
    incompatible = model.load_state_dict(cleaned, strict=False)
    # fc weights are allowed to be absent because policy encoders discard them.
    missing = [key for key in incompatible.missing_keys if not key.startswith("fc.")]
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("fc.")]
    if missing or unexpected:
        raise RuntimeError(
            f"Incompatible ResNet-18 weights: missing={missing}, unexpected={unexpected}"
        )
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)


class ResNet18FeatureMap(nn.Module):
    def __init__(self, *, frozen_batch_norm: bool = True, weights: Path | None = None) -> None:
        super().__init__()
        norm = FrozenBatchNorm2d if frozen_batch_norm else nn.BatchNorm2d
        self.backbone = ResNet18(norm_layer=norm)
        load_resnet18_weights(self.backbone, weights)
        self.num_channels = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone.forward_features(x)


class ResNet18GlobalPool(nn.Module):
    def __init__(self, *, group_norm: bool = True, weights: Path | None = None) -> None:
        super().__init__()
        norm = group_norm_32 if group_norm else nn.BatchNorm2d
        self.backbone = ResNet18(norm_layer=norm)
        load_resnet18_weights(self.backbone, weights)
        self.backbone.fc = nn.Identity()
        self.output_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)
        return torch.flatten(self.backbone.avgpool(features), 1)
