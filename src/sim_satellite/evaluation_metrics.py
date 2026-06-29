"""Generation metric helpers for validation image comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sim_satellite.image_metrics import structural_similarity


@dataclass
class MaskMetricState:
    pixel_count: Any
    rgb_generated_sum: Any
    rgb_real_sum: Any
    gradient_generated_sum: Any
    gradient_real_sum: Any
    sample_count: int = 0
    boundary_pixel_count: int = 0
    boundary_gradient_generated_sum: float = 0.0
    boundary_gradient_real_sum: float = 0.0

    @classmethod
    def create(
        cls,
        *,
        class_count: int,
        image_channels: int,
        torch: Any,
    ) -> MaskMetricState:
        return cls(
            pixel_count=torch.zeros(class_count, dtype=torch.float64),
            rgb_generated_sum=torch.zeros(class_count, image_channels, dtype=torch.float64),
            rgb_real_sum=torch.zeros(class_count, image_channels, dtype=torch.float64),
            gradient_generated_sum=torch.zeros(class_count, dtype=torch.float64),
            gradient_real_sum=torch.zeros(class_count, dtype=torch.float64),
        )

    def payload(self) -> dict[str, Any]:
        pixel_count = self.pixel_count
        valid = pixel_count > 0.0
        safe_count = pixel_count.clamp_min(1.0).view(-1, 1)
        rgb_generated = self.rgb_generated_sum / safe_count
        rgb_real = self.rgb_real_sum / safe_count
        rgb_mae = (rgb_generated - rgb_real).abs().mean(dim=1)

        gradient_generated = self.gradient_generated_sum / pixel_count.clamp_min(1.0)
        gradient_real = self.gradient_real_sum / pixel_count.clamp_min(1.0)
        gradient_mae = (gradient_generated - gradient_real).abs()

        boundary_generated = (
            self.boundary_gradient_generated_sum / self.boundary_pixel_count
            if self.boundary_pixel_count > 0
            else None
        )
        boundary_real = (
            self.boundary_gradient_real_sum / self.boundary_pixel_count
            if self.boundary_pixel_count > 0
            else None
        )
        boundary_delta = (
            abs(boundary_generated - boundary_real)
            if boundary_generated is not None and boundary_real is not None
            else None
        )

        return {
            "mask_sample_count": self.sample_count,
            "mask_class_pixel_counts": [int(value) for value in pixel_count.tolist()],
            "mask_rgb_mean_mae_macro": _macro_average(rgb_mae, valid),
            "mask_rgb_mean_mae_weighted": _weighted_average(rgb_mae, pixel_count),
            "mask_rgb_mean_mae_per_class": _per_class_values(rgb_mae, valid),
            "mask_gradient_mae_macro": _macro_average(gradient_mae, valid),
            "mask_gradient_mae_weighted": _weighted_average(gradient_mae, pixel_count),
            "mask_gradient_mae_per_class": _per_class_values(gradient_mae, valid),
            "mask_boundary_pixel_count": self.boundary_pixel_count,
            "mask_boundary_gradient_generated": boundary_generated,
            "mask_boundary_gradient_real": boundary_real,
            "mask_boundary_gradient_delta": boundary_delta,
        }


class GenerationMetricAccumulator:
    """Accumulate paired generated-vs-real image metrics over batches."""

    def __init__(
        self,
        *,
        class_count: int,
        image_channels: int,
        torch: Any,
    ) -> None:
        self.class_count = class_count
        self.image_channels = image_channels
        self.sample_count = 0
        self._mse_sum = 0.0
        self._psnr_sum = 0.0
        self._psnr_count = 0
        self._ssim_sum = 0.0
        self._mask_state = MaskMetricState.create(
            class_count=class_count,
            image_channels=image_channels,
            torch=torch,
        )

    def update(self, generated: Any, real: Any, *, semantic_mask: Any, torch: Any) -> None:
        generated = generated.detach().float().clamp(0.0, 1.0)
        real = real.detach().float().clamp(0.0, 1.0)
        batch_mse = torch.mean((generated - real) ** 2, dim=(1, 2, 3))
        finite_psnr = batch_mse > 0.0
        batch_ssim = structural_similarity(generated, real, torch=torch)

        self.sample_count += int(generated.shape[0])
        self._mse_sum += float(batch_mse.sum().detach().cpu().item())
        if bool(finite_psnr.any().detach().cpu().item()):
            psnr = -10.0 * torch.log10(batch_mse[finite_psnr])
            self._psnr_sum += float(psnr.sum().detach().cpu().item())
            self._psnr_count += int(finite_psnr.sum().detach().cpu().item())
        self._ssim_sum += float(batch_ssim.sum().detach().cpu().item())
        self._update_mask_metrics(generated, real, semantic_mask, torch=torch)

    def _update_mask_metrics(
        self,
        generated: Any,
        real: Any,
        semantic_mask: Any,
        *,
        torch: Any,
    ) -> None:
        semantic_mask = semantic_mask.detach().float()
        state = self._mask_state
        class_count = self.class_count
        labels = semantic_mask.argmax(dim=1).long()
        flat_labels = labels.reshape(-1)
        pixel_count = torch.bincount(flat_labels, minlength=class_count).detach().cpu().double()
        state.pixel_count += pixel_count
        state.sample_count += int(generated.shape[0])

        for channel in range(self.image_channels):
            generated_values = generated[:, channel, :, :].reshape(-1)
            real_values = real[:, channel, :, :].reshape(-1)
            state.rgb_generated_sum[:, channel] += (
                torch.bincount(flat_labels, weights=generated_values, minlength=class_count)
                .detach()
                .cpu()
                .double()
            )
            state.rgb_real_sum[:, channel] += (
                torch.bincount(flat_labels, weights=real_values, minlength=class_count)
                .detach()
                .cpu()
                .double()
            )
        generated_gradient = _sobel_gradient_magnitude(generated, torch=torch).reshape(-1)
        real_gradient = _sobel_gradient_magnitude(real, torch=torch).reshape(-1)
        state.gradient_generated_sum += (
            torch.bincount(flat_labels, weights=generated_gradient, minlength=class_count)
            .detach()
            .cpu()
            .double()
        )
        state.gradient_real_sum += (
            torch.bincount(flat_labels, weights=real_gradient, minlength=class_count)
            .detach()
            .cpu()
            .double()
        )

        boundary_mask = _semantic_boundary_mask(labels, torch=torch).reshape(-1)
        boundary_count = int(boundary_mask.sum().detach().cpu().item())
        if boundary_count > 0:
            state.boundary_pixel_count += boundary_count
            state.boundary_gradient_generated_sum += float(
                generated_gradient[boundary_mask].sum().detach().cpu().item()
            )
            state.boundary_gradient_real_sum += float(
                real_gradient[boundary_mask].sum().detach().cpu().item()
            )

    def payload(
        self,
        *,
        requested_sample_count: int,
        generation_steps: int,
        guidance_scale: float,
    ) -> dict[str, Any]:
        payload = {
            "enabled": True,
            "status": (
                "computed" if self.sample_count == requested_sample_count else "computed_partial"
            ),
            "sample_count": self.sample_count,
            "requested_sample_count": requested_sample_count,
            "generation_steps": generation_steps,
            "guidance_scale": guidance_scale,
            "mse": self._mse_sum / self.sample_count,
            "psnr": self._psnr_sum / self._psnr_count if self._psnr_count > 0 else None,
            "psnr_finite_sample_count": self._psnr_count,
            "psnr_exact_match_count": self.sample_count - self._psnr_count,
            "ssim": self._ssim_sum / self.sample_count,
        }
        payload.update(self._mask_state.payload())
        return payload


def disabled_generation_metrics_payload() -> dict[str, Any]:
    return {
        "enabled": False,
        "status": "disabled: set generation_metrics_samples > 0",
        "sample_count": 0,
        "mse": None,
        "psnr": None,
        "ssim": None,
    }


def _sobel_gradient_magnitude(images: Any, *, torch: Any) -> Any:
    channels = int(images.shape[1])
    kernel_x = images.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(
        1, 1, 3, 3
    )
    kernel_y = images.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(
        1, 1, 3, 3
    )
    kernel_x = kernel_x.repeat(channels, 1, 1, 1)
    kernel_y = kernel_y.repeat(channels, 1, 1, 1)
    grad_x = torch.nn.functional.conv2d(images, kernel_x, padding=1, groups=channels)
    grad_y = torch.nn.functional.conv2d(images, kernel_y, padding=1, groups=channels)
    return torch.sqrt((grad_x * grad_x + grad_y * grad_y).clamp_min(1e-12)).mean(
        dim=1,
        keepdim=True,
    )


def _semantic_boundary_mask(labels: Any, *, torch: Any) -> Any:
    boundary = torch.zeros_like(labels, dtype=torch.bool)
    boundary[:, :, 1:] |= labels[:, :, 1:] != labels[:, :, :-1]
    boundary[:, :, :-1] |= labels[:, :, 1:] != labels[:, :, :-1]
    boundary[:, 1:, :] |= labels[:, 1:, :] != labels[:, :-1, :]
    boundary[:, :-1, :] |= labels[:, 1:, :] != labels[:, :-1, :]
    return (
        torch.nn.functional.max_pool2d(
            boundary.float().unsqueeze(1),
            kernel_size=3,
            stride=1,
            padding=1,
        ).squeeze(1)
        > 0.0
    )


def _macro_average(values: Any, valid: Any) -> float | None:
    if not bool(valid.any().item()):
        return None
    return float(values[valid].mean().item())


def _weighted_average(values: Any, weights: Any) -> float | None:
    total = float(weights.sum().item())
    if total <= 0.0:
        return None
    return float(((values * weights).sum() / weights.sum().clamp_min(1.0)).item())


def _per_class_values(values: Any, valid: Any) -> list[float | None]:
    return [
        float(value) if bool(is_valid) else None
        for value, is_valid in zip(values.tolist(), valid.tolist(), strict=True)
    ]
