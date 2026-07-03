"""Segmentation model architectures."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from sim_satellite.segmentation.config import (
    SMP_UNET_EFFICIENTNET_B4_SCSE_MODEL_NAME,
    SegmentationTrainingConfig,
)


def build_segmentation_model(
    *,
    num_classes: int,
    input_channels: int = 3,
    model_name: str = SMP_UNET_EFFICIENTNET_B4_SCSE_MODEL_NAME,
) -> nn.Module:
    if model_name != SMP_UNET_EFFICIENTNET_B4_SCSE_MODEL_NAME:
        raise ValueError(f"Unsupported segmentation model: {model_name}")
    return SegmentationModelsPyTorchUNet(
        encoder_name="efficientnet-b4",
        decoder_attention_type="scse",
        input_channels=input_channels,
        num_classes=num_classes,
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
) -> torch.Tensor:
    return model(images)


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
        channel_groups = input_channels // 3
        self.input_channels = input_channels
        self.register_buffer(
            "_rgb_mean",
            torch.zeros((1, input_channels, 1, 1), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_rgb_std",
            torch.ones((1, 3, 1, 1), dtype=torch.float32).repeat(1, channel_groups, 1, 1),
            persistent=False,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] != self.input_channels:
            raise RuntimeError(f"SMP segmenters expect {self.input_channels} input channels")
        return self.model((image - self._rgb_mean) / self._rgb_std)
