from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, layers: int = 2) -> None:
        super().__init__()
        modules: list[nn.Module] = []
        current = in_channels
        for _ in range(layers):
            modules.extend(
                [
                    nn.Conv2d(current, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            current = out_channels
        self.block = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableBottleneck(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ConvBlock(in_channels, out_channels, layers=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class EnhancementUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.enc1 = ConvBlock(3, 32, layers=1)
        self.enc2 = ConvBlock(32, 64, layers=2)
        self.enc3 = ConvBlock(64, 128, layers=2)
        self.enc4 = ConvBlock(128, 256, layers=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DepthwiseSeparableBottleneck(256, 512)
        self.up4 = DecoderBlock(256 + 256, 128)
        self.up3 = DecoderBlock(128 + 128, 64)
        self.up2 = DecoderBlock(64 + 64, 32)
        self.up1 = DecoderBlock(32 + 32, 32)
        self.head = nn.Sequential(nn.Conv2d(32, 3, kernel_size=1), nn.Tanh())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        bottleneck = self.bottleneck(self.pool(enc4))

        up4 = torch.nn.functional.interpolate(
            bottleneck,
            size=enc4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        up4 = self.up4(torch.cat([up4, enc4], dim=1))

        up3 = torch.nn.functional.interpolate(
            up4,
            size=enc3.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        up3 = self.up3(torch.cat([up3, enc3], dim=1))

        up2 = torch.nn.functional.interpolate(
            up3,
            size=enc2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        up2 = self.up2(torch.cat([up2, enc2], dim=1))

        up1 = torch.nn.functional.interpolate(
            up2,
            size=enc1.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        up1 = self.up1(torch.cat([up1, enc1], dim=1))
        return self.head(up1)
