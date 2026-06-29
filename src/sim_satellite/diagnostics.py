"""Small color and chroma helpers for RGB images."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ImageColorStats:
    image_count: int
    width: int
    height: int
    rgb_mean: tuple[float, float, float]
    rgb_std: tuple[float, float, float]
    chroma_mean: float
    chroma_std: float
    saturation_mean: float
    saturation_std: float


def image_batch_color_stats(images: torch.Tensor) -> ImageColorStats:
    """Return aggregate RGB, chroma, and saturation stats for an image batch."""
    images = rgb_image_batch_tensor(images)
    image_count, height, width, _ = images.shape
    flat = images.reshape(-1, 3)
    chroma = image_chroma(images)
    saturation = image_saturation(images)
    return ImageColorStats(
        image_count=image_count,
        width=width,
        height=height,
        rgb_mean=_float_tuple(flat.mean(dim=0)),
        rgb_std=_float_tuple(flat.std(dim=0, unbiased=False)),
        chroma_mean=_mean(chroma),
        chroma_std=_std(chroma),
        saturation_mean=_mean(saturation),
        saturation_std=_std(saturation),
    )


def color_stats_payload(stats: ImageColorStats) -> dict[str, float | int]:
    return {
        "color_image_count": stats.image_count,
        "color_width": stats.width,
        "color_height": stats.height,
        "color_rgb_mean_r": stats.rgb_mean[0],
        "color_rgb_mean_g": stats.rgb_mean[1],
        "color_rgb_mean_b": stats.rgb_mean[2],
        "color_rgb_std_r": stats.rgb_std[0],
        "color_rgb_std_g": stats.rgb_std[1],
        "color_rgb_std_b": stats.rgb_std[2],
        "color_chroma_mean": stats.chroma_mean,
        "color_chroma_std": stats.chroma_std,
        "color_saturation_mean": stats.saturation_mean,
        "color_saturation_std": stats.saturation_std,
    }


def image_chroma(image: torch.Tensor) -> torch.Tensor:
    """Return per-pixel RGB chroma, max(channel) - min(channel), in 0..1 space."""
    image = rgb_image_or_batch_tensor(image)
    return image.amax(dim=-1) - image.amin(dim=-1)


def image_saturation(image: torch.Tensor) -> torch.Tensor:
    """Return per-pixel HSV-style saturation, chroma / max(channel), in 0..1 space."""
    image = rgb_image_or_batch_tensor(image)
    maximum = image.amax(dim=-1)
    chroma = image.amax(dim=-1) - image.amin(dim=-1)
    return torch.where(maximum > 0.0, chroma / maximum.clamp_min(1e-12), torch.zeros_like(maximum))


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

    return _normalize_rgb_tensor(tensor)


def rgb_image_batch_tensor(images: torch.Tensor) -> torch.Tensor:
    tensor = images.detach().cpu()
    if tensor.ndim == 3:
        return rgb_image_tensor(tensor).unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(
            f"Expected image batch tensor with 4 dimensions, got {tuple(tensor.shape)}"
        )
    if tensor.shape[-1] == 3:
        pass
    elif tensor.shape[1] == 3:
        tensor = tensor.permute(0, 2, 3, 1)
    else:
        raise ValueError(f"Expected RGB image batch tensor, got shape {tuple(tensor.shape)}")
    return _normalize_rgb_tensor(tensor)


def rgb_image_or_batch_tensor(image: torch.Tensor) -> torch.Tensor:
    if image.ndim == 4:
        return rgb_image_batch_tensor(image)
    return rgb_image_tensor(image)


def _normalize_rgb_tensor(tensor: torch.Tensor) -> torch.Tensor:
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
