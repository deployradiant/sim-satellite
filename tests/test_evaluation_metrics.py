from __future__ import annotations

import pytest
import torch

from sim_satellite.evaluation_metrics import (
    GenerationMetricAccumulator,
    disabled_generation_metrics_payload,
    structural_similarity,
)


def test_generation_metric_accumulator_reports_paired_image_metrics() -> None:
    real = torch.zeros(2, 3, 8, 8)
    generated = real.clone()
    generated[1] = 0.5

    accumulator = GenerationMetricAccumulator()
    accumulator.update(generated, real, torch=torch)
    payload = accumulator.payload(
        requested_sample_count=2,
        generation_steps=4,
        guidance_scale=1.5,
    )

    assert payload["enabled"] is True
    assert payload["sample_count"] == 2
    assert payload["generation_steps"] == 4
    assert payload["guidance_scale"] == 1.5
    assert payload["mse"] == pytest.approx(0.125)
    assert payload["psnr"] == pytest.approx(6.0205999)
    assert payload["psnr_finite_sample_count"] == 1
    assert payload["psnr_exact_match_count"] == 1
    assert 0.0 <= payload["ssim"] < 1.0
    assert "fid" not in payload
    assert "kid" not in payload
    assert "lpips" not in payload


def test_generation_metric_accumulator_marks_psnr_unbounded_for_exact_matches() -> None:
    image = torch.rand(1, 3, 8, 8)
    accumulator = GenerationMetricAccumulator()
    accumulator.update(image, image, torch=torch)
    payload = accumulator.payload(
        requested_sample_count=1,
        generation_steps=2,
        guidance_scale=1.0,
    )

    assert payload["mse"] == 0.0
    assert payload["psnr"] is None
    assert payload["psnr_finite_sample_count"] == 0
    assert payload["psnr_exact_match_count"] == 1
    assert payload["ssim"] == pytest.approx(1.0)


def test_structural_similarity_rejects_mismatched_batches() -> None:
    with pytest.raises(ValueError, match="same shape"):
        structural_similarity(
            torch.zeros(1, 3, 8, 8),
            torch.zeros(2, 3, 8, 8),
            torch=torch,
        )


def test_disabled_generation_metrics_payload_only_names_computable_metrics() -> None:
    payload = disabled_generation_metrics_payload()

    assert payload["enabled"] is False
    assert payload["mse"] is None
    assert payload["ssim"] is None
    assert payload["psnr"] is None
    assert "fid" not in payload
    assert "kid" not in payload
    assert "lpips" not in payload
