from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sim_satellite.vae_recon_eval import (
    latent_stats_payload,
    reconstruction_color_metrics_from_batches,
    reconstruction_gate_payload,
    select_reconstruction_samples,
)


def test_select_reconstruction_samples_filters_holdout_patterns(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                '{"image": "extracted/OpenEarthMap_wo_xBD/abancay/images/a.tif", '
                '"mask": "extracted/OpenEarthMap_wo_xBD/abancay/labels/a.tif", '
                '"prepared": "prepared/abancay/a.pt", '
                '"split": "train"}',
                '{"image": "extracted/OpenEarthMap_wo_xBD/tokyo/images/b.tif", '
                '"mask": "extracted/OpenEarthMap_wo_xBD/tokyo/labels/b.tif", '
                '"prepared": "prepared/tokyo/b.pt", '
                '"split": "train"}',
                '{"image": "extracted/OpenEarthMap_wo_xBD/abancay/images/c.tif", '
                '"mask": "extracted/OpenEarthMap_wo_xBD/abancay/labels/c.tif", '
                '"prepared": "prepared/abancay/c.pt", '
                '"split": "train"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    samples = select_reconstruction_samples(
        manifest_path=str(manifest_path),
        split="train",
        validation_holdout_patterns="abancay",
        sample_count=1,
    )

    assert [sample.image_path.name for sample in samples] == ["a.tif"]


def test_reconstruction_gate_payload_passes_color_faithful_reconstruction() -> None:
    original = torch.full((2, 3, 8, 8), 0.4)
    reconstructed = original + 0.001
    metrics = {"psnr": 32.0, "ssim": 0.91}
    color_metrics = reconstruction_color_metrics_from_batches(original, reconstructed)
    latent_values = torch.tensor([-1.0, 1.0]).repeat(64).view(2, 4, 4, 4)
    latent_stats = latent_stats_payload(latent_values)

    gates = reconstruction_gate_payload(metrics, color_metrics, latent_stats)

    assert gates["passed"] is True
    assert all(gates["checks"].values())


def test_reconstruction_gate_payload_flags_muted_color_and_bad_latents() -> None:
    original = torch.zeros(1, 3, 4, 4)
    original[:, 0] = 0.8
    reconstructed = original * 0.5
    metrics = {"psnr": 25.0, "ssim": 0.79}
    color_metrics = reconstruction_color_metrics_from_batches(original, reconstructed)
    latent_stats = {
        "shape": [1, 4, 4, 4],
        "channel_mean": [0.2, 0.0, 0.0, 0.0],
        "channel_std": [0.7, 1.0, 1.0, 1.3],
        "mean_abs_max": 0.2,
        "std_min": 0.7,
        "std_max": 1.3,
    }

    gates = reconstruction_gate_payload(metrics, color_metrics, latent_stats)

    assert gates["passed"] is False
    assert gates["checks"]["psnr"] is False
    assert gates["checks"]["ssim"] is False
    assert gates["checks"]["rgb_mean_abs_delta"] is False
    assert gates["checks"]["latent_mean_abs"] is False
    assert gates["checks"]["latent_std_range"] is False


def test_latent_stats_payload_requires_latent_batches() -> None:
    with pytest.raises(ValueError, match="Expected latents"):
        latent_stats_payload(torch.zeros(4))
