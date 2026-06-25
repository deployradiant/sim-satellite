from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from sim_satellite.data.loader import SatelliteSample, write_manifest
from sim_satellite.data.prepared import PREPARED_DATA_VERSION
from sim_satellite.vae import SatelliteVAE, SatelliteVAEConfig
from sim_satellite.vae_training import (
    VAETrainingConfig,
    load_vae_training_config_values,
    reconstruction_metrics,
    run_vae_training,
    split_vae_samples,
)


def test_satellite_vae_encode_decode_shapes() -> None:
    model = SatelliteVAE(SatelliteVAEConfig(latent_channels=4, hidden_channels=8))
    images = torch.randn(2, 3, 32, 32)

    mu, logvar = model.encode(images)
    reconstructed = model.reconstruct(images)

    assert tuple(mu.shape) == (2, 4, 8, 8)
    assert tuple(logvar.shape) == (2, 4, 8, 8)
    assert tuple(reconstructed.shape) == tuple(images.shape)
    assert torch.all(reconstructed >= -1.0)
    assert torch.all(reconstructed <= 1.0)


def test_vae_training_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError, match="Unknown VAE training config fields: nope"):
        load_vae_training_config_values({"nope": 1})


def test_vae_training_config_validates_split_bounds() -> None:
    with pytest.raises(ValueError, match="train_fraction"):
        load_vae_training_config_values({"train_fraction": 1.0})


def test_split_vae_samples_is_deterministic(tmp_path: Path) -> None:
    manifest_path = _write_manifest_with_prepared_samples(tmp_path, count=10)
    config = VAETrainingConfig(
        manifest_path=str(manifest_path),
        train_fraction=0.5,
        validation_samples=2,
        seed=123,
    )

    first_train, first_validation = split_vae_samples(config)
    second_train, second_validation = split_vae_samples(config)

    assert [sample.prepared_path for sample in first_train] == [
        sample.prepared_path for sample in second_train
    ]
    assert [sample.prepared_path for sample in first_validation] == [
        sample.prepared_path for sample in second_validation
    ]
    assert len(first_train) == 5
    assert len(first_validation) == 2


def test_reconstruction_metrics_identical_images_are_strong() -> None:
    images = torch.rand(2, 3, 16, 16) * 2.0 - 1.0

    metrics = reconstruction_metrics(images, images)

    assert metrics["mse"] == pytest.approx(0.0)
    assert metrics["psnr"] > 100.0
    assert metrics["ssim"] == pytest.approx(1.0)
    assert metrics["edge_mae"] == pytest.approx(0.0)


def test_run_vae_training_writes_checkpoint_metrics_and_grid(tmp_path: Path) -> None:
    manifest_path = _write_manifest_with_prepared_samples(tmp_path, count=6, image_size=16)
    config = VAETrainingConfig(
        manifest_path=str(manifest_path),
        train_fraction=0.5,
        validation_samples=2,
        image_size=16,
        eval_size=16,
        batch_size=1,
        steps=1,
        latent_channels=4,
        hidden_channels=8,
        device="cpu",
        cache_images=2,
        log_every=1,
        grid_samples=2,
        compare_sd_vae=False,
        checkpoint_path=str(tmp_path / "custom-vae.pt"),
        metrics_path=str(tmp_path / "metrics.json"),
        grid_path=str(tmp_path / "grid.png"),
    )

    result = run_vae_training(config)

    assert Path(result.checkpoint_path).exists()
    assert Path(result.metrics_path).exists()
    assert Path(result.grid_path).exists()
    payload = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    assert payload["train_image_count"] == 3
    assert payload["validation_image_count"] == 2
    assert payload["sd_vae"] is None
    checkpoint = torch.load(result.checkpoint_path, map_location="cpu")
    assert checkpoint["vae_config"]["latent_channels"] == 4


def _write_manifest_with_prepared_samples(
    tmp_path: Path,
    *,
    count: int,
    image_size: int = 32,
) -> Path:
    samples: list[SatelliteSample] = []
    for index in range(count):
        prepared_path = tmp_path / "prepared" / f"sample-{index}.pt"
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        image = torch.full((image_size, image_size, 3), index * 5, dtype=torch.uint8)
        mask = torch.zeros(image_size, image_size, dtype=torch.uint8)
        torch.save(
            {
                "version": PREPARED_DATA_VERSION,
                "image": image,
                "mask": mask,
                "size": (image_size, image_size),
                "coordinates": None,
            },
            prepared_path,
        )
        samples.append(
            SatelliteSample(
                image_path=tmp_path / "images" / f"sample-{index}.png",
                mask_path=tmp_path / "masks" / f"sample-{index}.png",
                prepared_path=prepared_path,
                split="train",
            )
        )

    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(samples, manifest_path)
    return manifest_path
