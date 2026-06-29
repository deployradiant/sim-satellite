from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import sim_satellite.vae.training as vae_training
from sim_satellite.data.loader import SatelliteSample, write_manifest
from sim_satellite.data.prepared import PREPARED_DATA_VERSION
from sim_satellite.vae.metrics import reconstruction_metrics, reconstruction_metrics_by_class
from sim_satellite.vae.model import SatelliteVAE, SatelliteVAEConfig
from sim_satellite.vae.training import (
    VAETrainingConfig,
    load_vae_training_config_values,
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
    with pytest.raises(ValueError, match="validation_split"):
        load_vae_training_config_values({"validation_split": "zipcode"})


def test_vae_training_config_validates_exponential_reconstruction_settings() -> None:
    with pytest.raises(ValueError, match="reconstruction_exponential_weight"):
        load_vae_training_config_values({"reconstruction_exponential_weight": -0.1})
    with pytest.raises(ValueError, match="reconstruction_exponential_scale"):
        load_vae_training_config_values({"reconstruction_exponential_scale": 0.0})


def test_exponential_reconstruction_loss_rewards_near_exact_reconstruction() -> None:
    real = torch.zeros(2, 3, 8, 8)
    one_percent_error = torch.full_like(real[:1], 0.01)
    five_percent_error = torch.full_like(real[1:], 0.05)

    small = vae_training._reconstruction_exponential_loss(
        real[:1],
        one_percent_error,
        scale=12.0,
    )
    large = vae_training._reconstruction_exponential_loss(
        real[1:],
        five_percent_error,
        scale=12.0,
    )

    assert large > small * 5.0


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


def test_city_split_holds_out_validation_groups(tmp_path: Path) -> None:
    manifest_path = _write_manifest_with_prepared_samples(
        tmp_path,
        count=12,
        group_count=4,
    )
    config = VAETrainingConfig(
        manifest_path=str(manifest_path),
        train_fraction=0.5,
        validation_split="city",
        validation_samples=3,
        seed=123,
    )

    train_samples, validation_samples = split_vae_samples(config)
    train_groups = {vae_training._sample_validation_group(sample) for sample in train_samples}
    validation_groups = {
        vae_training._sample_validation_group(sample) for sample in validation_samples
    }

    assert train_samples
    assert len(validation_samples) == 3
    assert train_groups.isdisjoint(validation_groups)


def test_reconstruction_metrics_identical_images_are_strong() -> None:
    images = torch.rand(2, 3, 16, 16) * 2.0 - 1.0

    metrics = reconstruction_metrics(images, images, torch=torch)

    assert metrics["mse"] == pytest.approx(0.0)
    assert metrics["psnr"] > 100.0
    assert metrics["ssim"] == pytest.approx(1.0)
    assert metrics["edge_mae"] == pytest.approx(0.0)


def test_reconstruction_metrics_by_class_reports_absent_classes() -> None:
    real = torch.zeros(1, 3, 4, 4)
    reconstructed = real.clone()
    reconstructed[:, :, :, 2:] = 0.5
    labels = torch.zeros(1, 4, 4, dtype=torch.long)
    labels[:, :, 2:] = 1

    metrics = reconstruction_metrics_by_class(
        real,
        reconstructed,
        labels,
        class_names=("background", "changed", "absent"),
        torch=torch,
    )

    assert metrics["background"]["pixels"] == 8
    assert metrics["background"]["mae"] == pytest.approx(0.0)
    assert metrics["changed"]["pixels"] == 8
    assert metrics["changed"]["mae"] == pytest.approx(0.5)
    assert metrics["absent"] == {
        "pixels": 0,
        "mae": None,
        "mse": None,
        "edge_mae": None,
    }


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
    assert "custom_vae_by_class" in payload
    assert payload["custom_vae_by_class"]["unlabeled"]["pixels"] == 512
    assert payload["latent_stats"]["sample_count"] == 3
    assert len(payload["latent_stats"]["mean"]) == 4
    checkpoint = torch.load(result.checkpoint_path, map_location="cpu")
    assert checkpoint["vae_config"]["latent_channels"] == 4
    assert checkpoint["latent_stats"]["sample_count"] == 3


def _write_manifest_with_prepared_samples(
    tmp_path: Path,
    *,
    count: int,
    image_size: int = 32,
    group_count: int | None = None,
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
                image_path=_fixture_path(
                    tmp_path,
                    "images",
                    index=index,
                    group_count=group_count,
                ),
                mask_path=_fixture_path(
                    tmp_path,
                    "masks",
                    index=index,
                    group_count=group_count,
                ),
                prepared_path=prepared_path,
                split="train",
                source="fixture",
            )
        )

    manifest_path = tmp_path / "manifest.jsonl"
    write_manifest(samples, manifest_path)
    return manifest_path


def _fixture_path(
    tmp_path: Path,
    root: str,
    *,
    index: int,
    group_count: int | None,
) -> Path:
    filename = f"sample-{index}.png"
    if group_count is None:
        return tmp_path / root / filename
    return tmp_path / root / f"city-{index % group_count}" / filename
