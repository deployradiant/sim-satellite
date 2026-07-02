"""Segmentation model architectures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from sim_satellite.segmentation.config import (
    LEGACY_SEGMENTATION_MODEL_NAMES,
    SMP_UNET_EFFICIENTNET_B4_SCSE_MODEL_NAME,
    VAE_ASPP_UNET_MODEL_NAME,
    SegmentationTrainingConfig,
)


def build_segmentation_model(
    *,
    num_classes: int,
    input_channels: int = 3,
    vae_latent_channels: int,
    model_name: str = VAE_ASPP_UNET_MODEL_NAME,
) -> nn.Module:
    if model_name == SMP_UNET_EFFICIENTNET_B4_SCSE_MODEL_NAME:
        return SegmentationModelsPyTorchUNet(
            encoder_name="efficientnet-b4",
            decoder_attention_type="scse",
            input_channels=input_channels,
            num_classes=num_classes,
        )
    if model_name not in (*LEGACY_SEGMENTATION_MODEL_NAMES, VAE_ASPP_UNET_MODEL_NAME):
        raise ValueError(f"Unsupported segmentation model: {model_name}")
    return VAEASPPUNetSegmentationModel(
        num_classes=num_classes,
        input_channels=input_channels,
        vae_latent_channels=vae_latent_channels,
    )


def _segmentation_input_channels(config: SegmentationTrainingConfig | Mapping[str, Any]) -> int:
    image_size = int(_config_value(config, "image_size", 0))
    context_size = int(_config_value(config, "context_size", 0))
    return 6 if context_size > image_size else 3


def _config_value(
    config: SegmentationTrainingConfig | Mapping[str, Any],
    name: str,
    default: Any,
) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _forward_segmentation_model(
    model: nn.Module,
    images: torch.Tensor,
    *,
    vae_latents: torch.Tensor | None,
) -> torch.Tensor:
    if isinstance(model, VAEASPPUNetSegmentationModel):
        if vae_latents is None:
            raise RuntimeError("VAE-fused segmentation requires VAE features")
        return model(images, vae_latents)
    if isinstance(model, SegmentationModelsPyTorchUNet):
        return model(images)
    raise RuntimeError(f"Unsupported segmentation model type: {type(model).__name__}")


class SegmentationModelsPyTorchUNet(nn.Module):
    _rgb_mean: torch.Tensor
    _rgb_std: torch.Tensor

    def __init__(
        self,
        *,
        encoder_name: str,
        decoder_attention_type: str | None,
        input_channels: int,
        num_classes: int,
    ) -> None:
        super().__init__()
        import segmentation_models_pytorch as smp

        if input_channels < 1:
            raise ValueError("input_channels must be positive")
        if input_channels % 3 != 0:
            raise ValueError("SMP segmenters expect RGB channel groups")

        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=input_channels,
            classes=num_classes,
            decoder_attention_type=decoder_attention_type,
        )
        mean = (0.0, 0.0, 0.0)
        std = (1.0, 1.0, 1.0)
        channel_groups = input_channels // 3
        self.input_channels = input_channels
        self.register_buffer(
            "_rgb_mean",
            torch.tensor(mean * channel_groups, dtype=torch.float32).view(
                1,
                input_channels,
                1,
                1,
            ),
            persistent=False,
        )
        self.register_buffer(
            "_rgb_std",
            torch.tensor(std * channel_groups, dtype=torch.float32).view(
                1,
                input_channels,
                1,
                1,
            ),
            persistent=False,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] != self.input_channels:
            raise RuntimeError(f"SMP segmenters expect {self.input_channels} input channels")
        return self.model((image - self._rgb_mean) / self._rgb_std)


class VAEASPPUNetSegmentationModel(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        input_channels: int = 3,
        vae_latent_channels: int,
    ) -> None:
        super().__init__()
        if vae_latent_channels < 1:
            raise ValueError("vae_latent_channels must be positive")
        self.stem = _conv_block(input_channels, 48)
        self.down1 = _down_block(48, 96)
        self.down2 = _down_block(96, 192)
        self.vae_projection = nn.Sequential(
            nn.Conv2d(vae_latent_channels, 192, kernel_size=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        self.down3 = _down_block(192, 256)
        self.context = _AtrousSpatialPyramid(256, 256)
        self.decode2 = _conv_block(448, 192)
        self.decode1 = _conv_block(288, 128)
        self.decode0 = _conv_block(176, 96)
        self.head = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(
        self,
        image: torch.Tensor,
        vae_latents: torch.Tensor,
    ) -> torch.Tensor:
        level0 = self.stem(image)
        level1 = self.down1(level0)
        level2 = self.down2(level1)
        level2 = level2 + _resize_like(self.vae_projection(vae_latents), level2)
        latent = self.context(self.down3(level2))

        decoded2 = self.decode2(
            torch.cat((_resize_like(latent, level2), level2), dim=1),
        )
        decoded1 = self.decode1(
            torch.cat((_resize_like(decoded2, level1), level1), dim=1),
        )
        decoded0 = self.decode0(
            torch.cat((_resize_like(decoded1, level0), level0), dim=1),
        )
        return self.head(decoded0)


def _resize_like(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        image,
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _down_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


# ASPP follows the parallel atrous-convolution context idea from DeepLab
# (Chen et al., arXiv:1606.00915) and DeepLabv3 (Chen et al., arXiv:1706.05587).
class _AtrousSpatialPyramid(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        branch_channels = out_channels // 4
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=2, dilation=2),
                nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=4, dilation=4),
                nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=6, dilation=6),
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(image) for branch in self.branches], dim=1))
