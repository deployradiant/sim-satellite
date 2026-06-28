"""Reconstruction metrics for satellite VAE audits."""

from __future__ import annotations

from typing import Any

from sim_satellite.image_metrics import structural_similarity


def reconstruction_metrics(real: Any, reconstructed: Any, *, torch: Any) -> dict[str, float]:
    """Compare paired image batches in [0, 1] NCHW format."""
    real = _unit_image_batch(real)
    reconstructed = _unit_image_batch(reconstructed)
    if real.shape != reconstructed.shape:
        raise ValueError(
            "Reconstruction image batches must have the same shape: "
            f"{tuple(real.shape)} != {tuple(reconstructed.shape)}"
        )

    mse = (real - reconstructed).square().mean(dim=(1, 2, 3))
    psnr = -10.0 * torch.log10(mse.clamp_min(1e-12))
    edge_mae = (
        (edge_map(real, torch=torch) - edge_map(reconstructed, torch=torch))
        .abs()
        .mean(dim=(1, 2, 3))
    )
    mean_rgb_mae = (real.mean(dim=(2, 3)) - reconstructed.mean(dim=(2, 3))).abs().mean(dim=1)
    return {
        "mse": float(mse.mean().item()),
        "psnr": float(psnr.mean().item()),
        "ssim": float(structural_similarity(real, reconstructed, torch=torch).mean().item()),
        "edge_mae": float(edge_mae.mean().item()),
        "mean_rgb_mae": float(mean_rgb_mae.mean().item()),
    }


def edge_map(image: Any, *, torch: Any) -> Any:
    image = _unit_image_batch(image)
    dx = image[:, :, :, 1:] - image[:, :, :, :-1]
    dy = image[:, :, 1:, :] - image[:, :, :-1, :]
    return torch.nn.functional.pad(dx.abs(), (0, 1, 0, 0)) + torch.nn.functional.pad(
        dy.abs(),
        (0, 0, 0, 1),
    )


def _unit_image_batch(image: Any) -> Any:
    if image.ndim != 4:
        raise ValueError(f"Expected image batches with shape (N, C, H, W), got {image.ndim}D")
    return image.float().clamp(0.0, 1.0)
