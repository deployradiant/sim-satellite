"""Native latent-space tensor transforms for flow-matching experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from sim_satellite.model import ModelConfig

TRAINING_SPACE_PIXEL = "pixel"
TRAINING_SPACE_LATENT = "latent"
LATENT_CODEC_AVERAGE_POOL_RGB = "average_pool_rgb"
LATENT_CODEC_PIXEL_UNSHUFFLE_RGB = "pixel_unshuffle_rgb"
NATIVE_LATENT_CODECS = frozenset(
    {
        LATENT_CODEC_AVERAGE_POOL_RGB,
        LATENT_CODEC_PIXEL_UNSHUFFLE_RGB,
    }
)


@dataclass(frozen=True)
class NativeLatentCodecConfig:
    codec: str
    downsample_factor: int
    image_channels: int


class NativeLatentCodec:
    def __init__(self, config: NativeLatentCodecConfig, *, torch: Any) -> None:
        if config.codec not in NATIVE_LATENT_CODECS:
            raise ValueError(f"Unknown latent codec: {config.codec}")
        if config.downsample_factor <= 0:
            raise ValueError("latent_downsample_factor must be positive")
        if config.image_channels <= 0:
            raise ValueError("image_channels must be positive")
        self.config = config
        self.torch = torch

    @property
    def channels(self) -> int:
        return native_latent_channels(
            self.config.codec,
            self.config.downsample_factor,
            self.config.image_channels,
        )

    def encode(self, images: Any) -> Any:
        _validate_nchw(images, "images")
        self._validate_spatial_divisible(images)
        if int(images.shape[1]) != self.config.image_channels:
            raise ValueError(
                f"Expected {self.config.image_channels} image channels, got {int(images.shape[1])}"
            )

        factor = self.config.downsample_factor
        if factor == 1:
            return images
        if self.config.codec == LATENT_CODEC_AVERAGE_POOL_RGB:
            return self.torch.nn.functional.avg_pool2d(images, kernel_size=factor, stride=factor)
        if self.config.codec == LATENT_CODEC_PIXEL_UNSHUFFLE_RGB:
            return self.torch.nn.functional.pixel_unshuffle(images, factor)
        raise AssertionError(f"Unhandled latent codec: {self.config.codec}")

    def decode(self, latents: Any) -> Any:
        _validate_nchw(latents, "latents")
        if int(latents.shape[1]) != self.channels:
            raise ValueError(
                f"Expected {self.channels} latent channels, got {int(latents.shape[1])}"
            )

        factor = self.config.downsample_factor
        if factor == 1:
            return latents
        if self.config.codec == LATENT_CODEC_AVERAGE_POOL_RGB:
            return self.torch.nn.functional.interpolate(
                latents,
                scale_factor=factor,
                mode="bilinear",
                align_corners=False,
            )
        if self.config.codec == LATENT_CODEC_PIXEL_UNSHUFFLE_RGB:
            return self.torch.nn.functional.pixel_shuffle(latents, factor)
        raise AssertionError(f"Unhandled latent codec: {self.config.codec}")

    def semantic_mask_for_model(self, semantic_mask: Any) -> Any:
        _validate_nchw(semantic_mask, "semantic_mask")
        self._validate_spatial_divisible(semantic_mask)
        factor = self.config.downsample_factor
        if factor == 1:
            return semantic_mask
        return self.torch.nn.functional.interpolate(
            semantic_mask.float(),
            size=(
                int(semantic_mask.shape[-2]) // factor,
                int(semantic_mask.shape[-1]) // factor,
            ),
            mode="area",
        )

    def _validate_spatial_divisible(self, tensor: Any) -> None:
        factor = self.config.downsample_factor
        height = int(tensor.shape[-2])
        width = int(tensor.shape[-1])
        if height % factor != 0 or width % factor != 0:
            raise ValueError(
                f"Spatial size {(height, width)} must be divisible by latent_downsample_factor "
                f"{factor}"
            )


def native_latent_channels(codec: str, downsample_factor: int, image_channels: int) -> int:
    if codec == LATENT_CODEC_AVERAGE_POOL_RGB:
        return image_channels
    if codec == LATENT_CODEC_PIXEL_UNSHUFFLE_RGB:
        return image_channels * downsample_factor * downsample_factor
    raise ValueError(f"Unknown latent codec: {codec}")


def native_latent_size(image_size: int, downsample_factor: int) -> int:
    return image_size // downsample_factor


def config_uses_latents(config: Any) -> bool:
    return getattr(config, "training_space", TRAINING_SPACE_PIXEL) == TRAINING_SPACE_LATENT


def latent_codec_config(config: Any) -> NativeLatentCodecConfig:
    return NativeLatentCodecConfig(
        codec=str(config.latent_codec),
        downsample_factor=int(config.latent_downsample_factor),
        image_channels=int(config.image_channels),
    )


def load_latent_codec(config: Any, *, torch: Any) -> NativeLatentCodec | None:
    if not config_uses_latents(config):
        return None
    return NativeLatentCodec(latent_codec_config(config), torch=torch)


def checkpoint_uses_latents(checkpoint: Mapping[str, Any]) -> bool:
    training_config = checkpoint.get("training_config")
    return (
        isinstance(training_config, Mapping)
        and training_config.get("training_space") == TRAINING_SPACE_LATENT
    )


def load_checkpoint_latent_codec(
    checkpoint: Mapping[str, Any],
    *,
    torch: Any,
) -> NativeLatentCodec | None:
    if not checkpoint_uses_latents(checkpoint):
        return None
    training_config = _checkpoint_training_config(checkpoint)
    return NativeLatentCodec(
        NativeLatentCodecConfig(
            codec=str(training_config["latent_codec"]),
            downsample_factor=int(training_config["latent_downsample_factor"]),
            image_channels=int(training_config["image_channels"]),
        ),
        torch=torch,
    )


def checkpoint_pixel_model_config(
    checkpoint: Mapping[str, Any],
    model_config: ModelConfig,
) -> ModelConfig:
    if not checkpoint_uses_latents(checkpoint):
        return model_config
    training_config = _checkpoint_training_config(checkpoint)
    return ModelConfig(
        image_size=int(training_config["image_size"]),
        image_channels=int(training_config["image_channels"]),
        mask_channels=model_config.mask_channels,
        hidden_channels=model_config.hidden_channels,
        seed=model_config.seed,
    )


def _checkpoint_training_config(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    training_config = checkpoint.get("training_config")
    if not isinstance(training_config, Mapping):
        raise RuntimeError("Latent checkpoint is missing training_config metadata")
    missing = [
        name
        for name in (
            "image_size",
            "image_channels",
            "latent_codec",
            "latent_downsample_factor",
        )
        if name not in training_config
    ]
    if missing:
        raise RuntimeError(
            "Latent checkpoint training_config is missing "
            + ", ".join(sorted(missing))
        )
    return training_config


def _validate_nchw(tensor: Any, name: str) -> None:
    if len(tensor.shape) != 4:
        raise ValueError(f"Expected {name} with shape (N, C, H, W), got {tuple(tensor.shape)}")
