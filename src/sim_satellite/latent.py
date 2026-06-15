"""Fixed latent encode/decode helpers for flow-matching experiments."""

from __future__ import annotations

from typing import Any


def validate_latent_downsample_factor(image_size: int, factor: int) -> None:
    if factor < 1:
        raise ValueError("latent_downsample_factor must be positive")
    if image_size % factor != 0:
        raise ValueError("image_size must be divisible by latent_downsample_factor")
    if image_size // factor < 32:
        raise ValueError("latent image_size must be at least 32")


def encode_image_latent(image: Any, *, factor: int, torch: Any) -> Any:
    if factor == 1:
        return image
    return torch.nn.functional.avg_pool2d(image, kernel_size=factor, stride=factor)


def encode_mask_latent(semantic_mask: Any, *, size: int, torch: Any) -> Any:
    if semantic_mask.shape[-2:] == (size, size):
        return semantic_mask
    return torch.nn.functional.interpolate(semantic_mask, size=(size, size), mode="nearest")


def decode_image_latent(image: Any, *, factor: int, output_size: int, torch: Any) -> Any:
    if factor == 1 and image.shape[-2:] == (output_size, output_size):
        return image
    return torch.nn.functional.interpolate(
        image,
        size=(output_size, output_size),
        mode="bilinear",
        align_corners=False,
    )
