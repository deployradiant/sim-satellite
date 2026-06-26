"""Native VAE building blocks for satellite image latents."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class SatelliteVAEConfig:
    image_channels: int = 3
    latent_channels: int = 8
    hidden_channels: int = 32
    downsample_factor: int = 4


class ResBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SatelliteVAE(nn.Module):
    """Small convolutional VAE for 4x satellite image compression."""

    def __init__(self, config: SatelliteVAEConfig) -> None:
        super().__init__()
        if config.downsample_factor != 4:
            raise ValueError("SatelliteVAE currently supports downsample_factor=4")
        if config.image_channels != 3:
            raise ValueError("SatelliteVAE currently supports RGB images only")
        if config.hidden_channels % 8 != 0:
            raise ValueError("hidden_channels must be divisible by 8")

        self.config = config
        h = config.hidden_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(config.image_channels, h, 3, padding=1),
            ResBlock(h),
            nn.Conv2d(h, h * 2, 4, stride=2, padding=1),
            ResBlock(h * 2),
            nn.Conv2d(h * 2, h * 4, 4, stride=2, padding=1),
            ResBlock(h * 4),
            nn.GroupNorm(8, h * 4),
            nn.SiLU(),
            nn.Conv2d(h * 4, config.latent_channels * 2, 3, padding=1),
        )
        self.decoder_in = nn.Conv2d(config.latent_channels, h * 4, 3, padding=1)
        self.decoder = nn.Sequential(
            ResBlock(h * 4),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(h * 4, h * 2, 3, padding=1),
            ResBlock(h * 2),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(h * 2, h, 3, padding=1),
            ResBlock(h),
            nn.GroupNorm(8, h),
            nn.SiLU(),
            nn.Conv2d(h, config.image_channels, 3, padding=1),
        )

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.encoder(images).chunk(2, dim=1)
        return mu, logvar.clamp(-8.0, 8.0)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.decoder_in(latents)).tanh()

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(images)
        std = torch.exp(0.5 * logvar)
        recon = self.decode(mu + torch.randn_like(std) * std)
        return recon, mu, logvar

    def reconstruct(self, images: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(images)
        return self.decode(mu).clamp(-1.0, 1.0)
