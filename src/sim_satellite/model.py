"""Tiny conditional diffusion model primitives.

This is intentionally small: it gives the project an end-to-end PyTorch
surface without pretending to be the final satellite model architecture.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 32
    image_channels: int = 3
    mask_channels: int = 9
    hidden_channels: int = 32
    timesteps: int = 16
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.image_size < 8:
            raise ValueError("image_size must be at least 8")
        if self.image_channels < 1:
            raise ValueError("image_channels must be positive")
        if self.mask_channels < 1:
            raise ValueError("mask_channels must be positive")
        if self.hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")
        if self.timesteps < 1:
            raise ValueError("timesteps must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ModelConfig:
        names = {field.name for field in fields(cls)}
        return cls(**{name: values[name] for name in names if name in values})


def require_torch() -> tuple[Any, Any]:
    try:
        import torch
        from torch import nn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "PyTorch is required for model execution. Install the project "
            "runtime dependencies or run this through the Modal target."
        ) from exc

    return torch, nn


def build_model(config: ModelConfig | None = None) -> Any:
    torch, nn = require_torch()
    config = config or ModelConfig()

    class TinyConditionalDenoiser(nn.Module):
        def __init__(self, model_config: ModelConfig) -> None:
            super().__init__()
            self.config = model_config
            input_channels = model_config.image_channels + model_config.mask_channels + 1
            self.net = nn.Sequential(
                nn.Conv2d(input_channels, model_config.hidden_channels, 3, padding=1),
                nn.SiLU(),
                nn.Conv2d(
                    model_config.hidden_channels,
                    model_config.hidden_channels,
                    3,
                    padding=1,
                ),
                nn.SiLU(),
                nn.Conv2d(
                    model_config.hidden_channels,
                    model_config.image_channels,
                    3,
                    padding=1,
                ),
            )

        def forward(self, image: Any, semantic_mask: Any, timestep: Any) -> Any:
            batch_size, _, height, width = image.shape
            scaled_timestep = timestep.float().view(batch_size, 1, 1, 1)
            scaled_timestep = scaled_timestep / max(1, self.config.timesteps - 1)
            timestep_plane = scaled_timestep.expand(batch_size, 1, height, width)
            model_input = torch.cat([image, semantic_mask, timestep_plane], dim=1)
            return self.net(model_input)

    return TinyConditionalDenoiser(config)


def add_noise(clean_image: Any, noise: Any, timestep: Any, timesteps: int) -> Any:
    scale = (timestep.float().view(-1, 1, 1, 1) + 1.0) / float(timesteps)
    return clean_image + noise * scale
