from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sim_satellite.vae_compare import (
    VaeCodecSpec,
    VaeReconstructionAccumulator,
    comparison_payload,
    parse_codec_specs,
    select_reconstruction_samples,
)


def test_parse_codec_specs_requires_labeled_unique_codecs() -> None:
    specs = parse_codec_specs(
        (
            "sd=stabilityai/sd-vae-ft-mse",
            "custom=artifacts/custom-vae",
        )
    )

    assert specs == (
        VaeCodecSpec("sd", "stabilityai/sd-vae-ft-mse"),
        VaeCodecSpec("custom", "artifacts/custom-vae"),
    )

    with pytest.raises(ValueError, match="LABEL=MODEL_OR_PATH"):
        parse_codec_specs(("sd",))
    with pytest.raises(ValueError, match="unique"):
        parse_codec_specs(("sd=a", "sd=b"))


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


def test_vae_reconstruction_accumulator_reports_color_and_latent_stats() -> None:
    original = torch.zeros(2, 3, 8, 8)
    reconstructed = original.clone()
    reconstructed[:, 0] = 0.1
    latents = torch.tensor([-1.0, 1.0]).repeat(64).view(2, 4, 4, 4)

    accumulator = VaeReconstructionAccumulator()
    accumulator.update(original=original, reconstructed=reconstructed, latents=latents)
    payload = accumulator.payload(sample_count=2)

    assert payload["metrics"]["sample_count"] == 2
    assert payload["metrics"]["mse"] > 0.0
    assert payload["color"]["rgb_mean_abs_delta_max"] == pytest.approx(0.05)
    assert payload["color"]["chroma_mean_delta"] > 0.0
    assert payload["latents"]["shape"] == [2, 4, 4, 4]
    assert payload["latents"]["mean_abs_max"] == pytest.approx(0.0)
    assert payload["latents"]["std_min"] == pytest.approx(1.0)
    assert payload["latents"]["std_max"] == pytest.approx(1.0)


def test_comparison_payload_uses_first_codec_as_baseline() -> None:
    payload = comparison_payload(
        {
            "sd": {
                "metrics": {"mse": 0.2, "psnr": 10.0, "ssim": 0.5},
                "color": {
                    "rgb_mean_abs_delta_max": 0.03,
                    "chroma_mean_delta": -0.02,
                    "saturation_mean_delta": -0.04,
                },
            },
            "custom": {
                "metrics": {"mse": 0.1, "psnr": 12.0, "ssim": 0.7},
                "color": {
                    "rgb_mean_abs_delta_max": 0.01,
                    "chroma_mean_delta": -0.01,
                    "saturation_mean_delta": -0.02,
                },
            },
        }
    )

    assert payload["baseline"] == "sd"
    assert payload["deltas"]["custom"]["mse_delta"] == pytest.approx(-0.1)
    assert payload["deltas"]["custom"]["psnr_delta"] == pytest.approx(2.0)
    assert payload["deltas"]["custom"]["ssim_delta"] == pytest.approx(0.2)
    assert payload["deltas"]["custom"]["rgb_mean_abs_delta_max_delta"] == pytest.approx(-0.02)
    assert payload["deltas"]["custom"]["chroma_mean_delta_abs_delta"] == pytest.approx(-0.01)
    assert payload["deltas"]["custom"]["saturation_mean_delta_abs_delta"] == pytest.approx(-0.02)
