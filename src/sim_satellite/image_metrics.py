"""Shared image comparison helpers."""

from __future__ import annotations

from typing import Any


def structural_similarity(image_a: Any, image_b: Any, *, torch: Any) -> Any:
    image_a = _unit_image_batch(image_a)
    image_b = _unit_image_batch(image_b)
    if image_a.shape != image_b.shape:
        raise ValueError(
            "SSIM image batches must have the same shape: "
            f"{tuple(image_a.shape)} != {tuple(image_b.shape)}"
        )

    kernel_size = _ssim_kernel_size(int(image_a.shape[-2]), int(image_a.shape[-1]))
    padding = kernel_size // 2

    mu_a = torch.nn.functional.avg_pool2d(image_a, kernel_size, stride=1, padding=padding)
    mu_b = torch.nn.functional.avg_pool2d(image_b, kernel_size, stride=1, padding=padding)
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a = (
        torch.nn.functional.avg_pool2d(
            image_a * image_a,
            kernel_size,
            stride=1,
            padding=padding,
        )
        - mu_a_sq
    )
    sigma_b = (
        torch.nn.functional.avg_pool2d(
            image_b * image_b,
            kernel_size,
            stride=1,
            padding=padding,
        )
        - mu_b_sq
    )
    sigma_ab = (
        torch.nn.functional.avg_pool2d(
            image_a * image_b,
            kernel_size,
            stride=1,
            padding=padding,
        )
        - mu_ab
    )

    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a + sigma_b + c2)
    ssim = numerator / denominator.clamp_min(1e-12)
    return ssim.mean(dim=(1, 2, 3)).clamp(-1.0, 1.0)


def _unit_image_batch(image: Any) -> Any:
    if image.ndim != 4:
        raise ValueError(f"Expected image batches with shape (N, C, H, W), got {image.ndim}D")
    return image.float().clamp(0.0, 1.0)


def _ssim_kernel_size(height: int, width: int) -> int:
    size = min(height, width, 11)
    if size < 1:
        raise ValueError(f"Invalid image size for SSIM: {(width, height)}")
    if size % 2 == 0:
        size -= 1
    return max(size, 1)
