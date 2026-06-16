"""Small color and chroma helpers for RGB images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageColorStats:
    width: int
    height: int
    rgb_mean: tuple[float, float, float]
    rgb_std: tuple[float, float, float]
    chroma_mean: float
    chroma_std: float
    saturation_mean: float
    saturation_std: float


def image_color_stats(image: torch.Tensor) -> ImageColorStats:
    """Return simple RGB, chroma, and saturation stats for an image tensor."""
    image = rgb_image_tensor(image)
    height, width, _ = image.shape
    flat = image.reshape(-1, 3)
    chroma = image_chroma(image)
    saturation = image_saturation(image)
    return ImageColorStats(
        width=width,
        height=height,
        rgb_mean=_float_tuple(flat.mean(dim=0)),
        rgb_std=_float_tuple(flat.std(dim=0, unbiased=False)),
        chroma_mean=_mean(chroma),
        chroma_std=_std(chroma),
        saturation_mean=_mean(saturation),
        saturation_std=_std(saturation),
    )


def image_chroma(image: torch.Tensor) -> torch.Tensor:
    """Return per-pixel RGB chroma, max(channel) - min(channel), in 0..1 space."""
    image = rgb_image_tensor(image)
    return image.amax(dim=-1) - image.amin(dim=-1)


def image_saturation(image: torch.Tensor) -> torch.Tensor:
    """Return per-pixel HSV-style saturation, chroma / max(channel), in 0..1 space."""
    image = rgb_image_tensor(image)
    maximum = image.amax(dim=-1)
    chroma = image.amax(dim=-1) - image.amin(dim=-1)
    return torch.where(maximum > 0.0, chroma / maximum.clamp_min(1e-12), torch.zeros_like(maximum))


def load_rgb_image(path: Path) -> torch.Tensor:
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        raise ValueError(f"Unsupported image file suffix {path.suffix!r}: {path}")

    with Image.open(path) as image:
        image = image.convert("RGB")
        width, height = image.size
        values = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return values.view(height, width, 3).float() / 255.0


def rgb_image_tensor(image: torch.Tensor) -> torch.Tensor:
    tensor = image.detach().cpu()
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError(f"Expected a single image tensor, got shape {tuple(tensor.shape)}")
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dimensions, got {tuple(tensor.shape)}")
    if tensor.shape[-1] == 3:
        pass
    elif tensor.shape[0] == 3:
        tensor = tensor.permute(1, 2, 0)
    else:
        raise ValueError(f"Expected RGB image tensor, got shape {tuple(tensor.shape)}")

    if tensor.dtype == torch.uint8:
        return tensor.float() / 255.0

    tensor = tensor.float()
    minimum = float(tensor.min().item()) if tensor.numel() else 0.0
    maximum = float(tensor.max().item()) if tensor.numel() else 0.0
    if minimum < 0.0:
        return ((tensor.clamp(-1.0, 1.0) + 1.0) / 2.0).contiguous()
    if maximum > 1.0:
        return (tensor.clamp(0.0, 255.0) / 255.0).contiguous()
    return tensor.clamp(0.0, 1.0).contiguous()


def _mean(values: torch.Tensor) -> float:
    return float(values.mean().item())


def _std(values: torch.Tensor) -> float:
    return float(values.std(unbiased=False).item())


def _float_tuple(values: torch.Tensor) -> tuple[float, float, float]:
    red, green, blue = values.detach().cpu().tolist()
    return (float(red), float(green), float(blue))
