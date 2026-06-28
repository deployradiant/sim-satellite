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
    hist_generated_count: Any
    hist_real_count: Any
    gradient_generated_sum: Any
    gradient_real_sum: Any
    histogram_bins: int
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
        histogram_bins: int,
        torch: Any,
    ) -> MaskMetricState:
        return cls(
            pixel_count=torch.zeros(class_count, dtype=torch.float64),
            rgb_generated_sum=torch.zeros(class_count, image_channels, dtype=torch.float64),
            rgb_real_sum=torch.zeros(class_count, image_channels, dtype=torch.float64),
            hist_generated_count=torch.zeros(
                class_count,
                image_channels,
                histogram_bins,
                dtype=torch.float64,
            ),
            hist_real_count=torch.zeros(
                class_count,
                image_channels,
                histogram_bins,
                dtype=torch.float64,
            ),
            gradient_generated_sum=torch.zeros(class_count, dtype=torch.float64),
            gradient_real_sum=torch.zeros(class_count, dtype=torch.float64),
            histogram_bins=histogram_bins,
        )

    def validate_shape(self, *, class_count: int, image_channels: int) -> None:
        if int(self.pixel_count.shape[0]) != class_count:
            raise ValueError(
                "All semantic mask batches must have the same class count: "
                f"{int(self.pixel_count.shape[0])} != {class_count}"
            )
        if int(self.rgb_generated_sum.shape[1]) != image_channels:
            raise ValueError(
                "All image batches must have the same channel count for mask metrics: "
                f"{int(self.rgb_generated_sum.shape[1])} != {image_channels}"
            )

    def add_histogram_counts(
        self,
        values: Any,
        labels: Any,
        *,
        class_count: int,
        channel: int,
        torch: Any,
        target: str,
    ) -> None:
        bin_ids = (values * self.histogram_bins).long().clamp(0, self.histogram_bins - 1)
        combined = labels * self.histogram_bins + bin_ids
        counts = (
            torch.bincount(
                combined,
                minlength=class_count * self.histogram_bins,
            )
            .view(class_count, self.histogram_bins)
            .detach()
            .cpu()
            .double()
        )
        if target == "generated":
            self.hist_generated_count[:, channel, :] += counts
        else:
            self.hist_real_count[:, channel, :] += counts

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

        hist_js = _histogram_jensen_shannon(
            self.hist_generated_count,
            self.hist_real_count,
        )
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
            "mask_rgb_hist_js_macro": _macro_average(hist_js, valid),
            "mask_rgb_hist_js_weighted": _weighted_average(hist_js, pixel_count),
            "mask_rgb_hist_js_per_class": _per_class_values(hist_js, valid),
            "mask_gradient_mae_macro": _macro_average(gradient_mae, valid),
            "mask_gradient_mae_weighted": _weighted_average(gradient_mae, pixel_count),
            "mask_gradient_mae_per_class": _per_class_values(gradient_mae, valid),
            "mask_boundary_pixel_count": self.boundary_pixel_count,
            "mask_boundary_gradient_generated": boundary_generated,
            "mask_boundary_gradient_real": boundary_real,
            "mask_boundary_gradient_delta": boundary_delta,
        }


@dataclass
class GenerationMetricAccumulator:
    """Accumulate paired generated-vs-real image metrics over batches."""

    sample_count: int = 0
    mask_histogram_bins: int = 16
    _mse_sum: float = 0.0
    _psnr_sum: float = 0.0
    _psnr_count: int = 0
    _ssim_sum: float = 0.0
    _mask_state: MaskMetricState | None = None

    def update(self, generated: Any, real: Any, *, torch: Any, semantic_mask: Any = None) -> None:
        generated = generated.detach().float().clamp(0.0, 1.0)
        real = real.detach().float().clamp(0.0, 1.0)
        if generated.shape != real.shape:
            raise ValueError(
                "Generated and real image batches must have the same shape: "
                f"{tuple(generated.shape)} != {tuple(real.shape)}"
            )
        if generated.ndim != 4:
            raise ValueError(
                f"Expected image batches with shape (N, C, H, W), got {generated.ndim}D"
            )

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
        if semantic_mask is not None:
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
        if semantic_mask.ndim != 4:
            raise ValueError(
                f"Expected semantic masks with shape (N, K, H, W), got {semantic_mask.ndim}D"
            )
        shape_mismatch = (
            semantic_mask.shape[0] != generated.shape[0]
            or semantic_mask.shape[-2:] != generated.shape[-2:]
        )
        if shape_mismatch:
            raise ValueError(
                "Semantic masks must match generated image batch and spatial shape: "
                f"{tuple(semantic_mask.shape)} vs {tuple(generated.shape)}"
            )
        if semantic_mask.shape[1] < 1:
            raise ValueError("Semantic masks must contain at least one class channel")

        class_count = int(semantic_mask.shape[1])
        image_channels = int(generated.shape[1])
        state = self._mask_state_for(class_count, image_channels, torch=torch)

        labels = semantic_mask.argmax(dim=1).long()
        flat_labels = labels.reshape(-1)
        pixel_count = torch.bincount(flat_labels, minlength=class_count).detach().cpu().double()
        state.pixel_count += pixel_count
        state.sample_count += int(generated.shape[0])

        for channel in range(image_channels):
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
            state.add_histogram_counts(
                generated_values,
                flat_labels,
                class_count=class_count,
                channel=channel,
                torch=torch,
                target="generated",
            )
            state.add_histogram_counts(
                real_values,
                flat_labels,
                class_count=class_count,
                channel=channel,
                torch=torch,
                target="real",
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

    def _mask_state_for(
        self,
        class_count: int,
        image_channels: int,
        *,
        torch: Any,
    ) -> MaskMetricState:
        if self._mask_state is None:
            self._mask_state = MaskMetricState.create(
                class_count=class_count,
                image_channels=image_channels,
                histogram_bins=self.mask_histogram_bins,
                torch=torch,
            )
        else:
            self._mask_state.validate_shape(
                class_count=class_count,
                image_channels=image_channels,
            )
        return self._mask_state

    def payload(
        self,
        *,
        requested_sample_count: int,
        generation_steps: int,
        guidance_scale: float,
    ) -> dict[str, Any]:
        if self.sample_count == 0:
            return {
                "enabled": False,
                "status": "skipped: no validation samples were available",
                "sample_count": 0,
                "requested_sample_count": requested_sample_count,
            }

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
        payload.update(self._mask_payload())
        return payload

    def _mask_payload(self) -> dict[str, Any]:
        if self._mask_state is None:
            return {}
        return self._mask_state.payload()


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


def _histogram_jensen_shannon(generated_counts: Any, real_counts: Any) -> Any:
    generated = generated_counts.double()
    real = real_counts.double()
    generated = generated / generated.sum(dim=2, keepdim=True).clamp_min(1.0)
    real = real / real.sum(dim=2, keepdim=True).clamp_min(1.0)
    midpoint = 0.5 * (generated + real)
    epsilon = 1e-12
    generated_kl = (
        generated * (generated.clamp_min(epsilon) / midpoint.clamp_min(epsilon)).log()
    ).sum(dim=2)
    real_kl = (real * (real.clamp_min(epsilon) / midpoint.clamp_min(epsilon)).log()).sum(dim=2)
    return (0.5 * (generated_kl + real_kl)).mean(dim=1)


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
