"""Lossless spatial latent encode/decode helpers for flow-matching experiments."""

from __future__ import annotations

from typing import Any


def validate_latent_downsample_factor(image_size: int, factor: int) -> None:
    if factor < 1:
        raise ValueError("latent_downsample_factor must be positive")
    if image_size % factor != 0:
        raise ValueError("image_size must be divisible by latent_downsample_factor")
    if image_size // factor < 32:
        raise ValueError("latent image_size must be at least 32")


def latent_image_channels(image_channels: int, *, factor: int) -> int:
    return image_channels * factor * factor


def encode_image_latent(image: Any, *, factor: int, torch: Any) -> Any:
    if factor == 1:
        return image
    return torch.nn.functional.pixel_unshuffle(image, downscale_factor=factor)


def encode_mask_latent(semantic_mask: Any, *, size: int, torch: Any) -> Any:
    if semantic_mask.shape[-2:] == (size, size):
        return semantic_mask
    return torch.nn.functional.interpolate(semantic_mask, size=(size, size), mode="nearest")


def decode_image_latent(image: Any, *, factor: int, output_size: int, torch: Any) -> Any:
    if factor == 1 and image.shape[-2:] == (output_size, output_size):
        return image
    decoded = torch.nn.functional.pixel_shuffle(image, upscale_factor=factor)
    if decoded.shape[-2:] != (output_size, output_size):
        raise ValueError(
            f"Decoded latent image has size {decoded.shape[-2:]}, expected "
            f"{(output_size, output_size)}"
        )
    return decoded
