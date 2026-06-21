"""Generation metric helpers for validation image comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

REFERENCE_METRIC_STATUS = {
    "fid": "not computed: no pretrained image feature extractor is configured",
    "kid": "not computed: no pretrained image feature extractor is configured",
    "lpips": "not computed: no pretrained perceptual-distance model is configured",
    "dinov2_fid": "not computed: no DINOv2 feature extractor is configured",
    "satclip_fid": "not computed: no SatCLIP feature extractor is configured",
}


@dataclass
class GenerationMetricAccumulator:
    """Accumulate paired generated-vs-real image metrics over batches."""

    sample_count: int = 0
    _mse_sum: float = 0.0
    _psnr_sum: float = 0.0
    _psnr_count: int = 0
    _ssim_sum: float = 0.0

    def update(self, generated: Any, real: Any, *, torch: Any) -> None:
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

        return {
            "enabled": True,
            "status": (
                "computed"
                if self.sample_count == requested_sample_count
                else "computed_partial"
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
            "fid": None,
            "kid": None,
            "lpips": None,
            "dinov2_fid": None,
            "satclip_fid": None,
            "reference_metric_status": dict(REFERENCE_METRIC_STATUS),
        }


def disabled_generation_metrics_payload() -> dict[str, Any]:
    return {
        "enabled": False,
        "status": "disabled: set generation_metrics_samples > 0",
        "sample_count": 0,
        "fid": None,
        "kid": None,
        "lpips": None,
        "mse": None,
        "psnr": None,
        "ssim": None,
        "dinov2_fid": None,
        "satclip_fid": None,
        "reference_metric_status": dict(REFERENCE_METRIC_STATUS),
    }


def structural_similarity(image_a: Any, image_b: Any, *, torch: Any) -> Any:
    if image_a.shape != image_b.shape:
        raise ValueError(
            "SSIM image batches must have the same shape: "
            f"{tuple(image_a.shape)} != {tuple(image_b.shape)}"
        )
    if image_a.ndim != 4:
        raise ValueError(f"Expected image batches with shape (N, C, H, W), got {image_a.ndim}D")

    image_a = image_a.float().clamp(0.0, 1.0)
    image_b = image_b.float().clamp(0.0, 1.0)
    kernel_size = _ssim_kernel_size(int(image_a.shape[-2]), int(image_a.shape[-1]))
    padding = kernel_size // 2

    mu_a = torch.nn.functional.avg_pool2d(image_a, kernel_size, stride=1, padding=padding)
    mu_b = torch.nn.functional.avg_pool2d(image_b, kernel_size, stride=1, padding=padding)
    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a = torch.nn.functional.avg_pool2d(
        image_a * image_a,
        kernel_size,
        stride=1,
        padding=padding,
    ) - mu_a_sq
    sigma_b = torch.nn.functional.avg_pool2d(
        image_b * image_b,
        kernel_size,
        stride=1,
        padding=padding,
    ) - mu_b_sq
    sigma_ab = torch.nn.functional.avg_pool2d(
        image_a * image_b,
        kernel_size,
        stride=1,
        padding=padding,
    ) - mu_ab

    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a + sigma_b + c2)
    ssim = numerator / denominator.clamp_min(1e-12)
    return ssim.mean(dim=(1, 2, 3)).clamp(-1.0, 1.0)


def _ssim_kernel_size(height: int, width: int) -> int:
    size = min(height, width, 11)
    if size < 1:
        raise ValueError(f"Invalid image size for SSIM: {(width, height)}")
    if size % 2 == 0:
        size -= 1
    return max(size, 1)
