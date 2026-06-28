from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from sim_satellite.latent import (
    LatentCodecConfig,
    SatelliteVAELatentCodec,
    checkpoint_uses_latents,
    semantic_mask_for_model,
)
from sim_satellite.vae.model import SatelliteVAE, SatelliteVAEConfig


def test_satellite_vae_latent_codec_uses_checkpoint_stats(tmp_path: Path) -> None:
    checkpoint_path = _write_vae_checkpoint(tmp_path, latent_channels=4)
    codec = SatelliteVAELatentCodec(
        LatentCodecConfig(
            checkpoint_path=str(checkpoint_path),
            downsample_factor=4,
            channels=4,
        ),
        torch=torch,
        device=torch.device("cpu"),
    )

    images = torch.randn(2, 3, 32, 32).clamp(-1.0, 1.0)
    latents = codec.encode(images)
    decoded = codec.decode(latents)

    assert tuple(latents.shape) == (2, 4, 8, 8)
    assert tuple(decoded.shape) == tuple(images.shape)
    assert torch.all(decoded >= -1.0)
    assert torch.all(decoded <= 1.0)


def test_semantic_mask_for_model_downsamples_with_area() -> None:
    semantic_mask = torch.zeros(1, 2, 4, 4)
    semantic_mask[:, 0, :, :2] = 1.0
    semantic_mask[:, 1, :, 2:] = 1.0

    resized = semantic_mask_for_model(semantic_mask, model_size=(2, 2))

    assert tuple(resized.shape) == (1, 2, 2, 2)
    assert torch.allclose(resized[:, :, :, 0], torch.tensor([[[1.0, 1.0], [0.0, 0.0]]]))
    assert torch.allclose(resized[:, :, :, 1], torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]))


def test_checkpoint_uses_latents_reads_training_config_metadata() -> None:
    assert checkpoint_uses_latents({"training_config": {"training_space": "latent"}})
    assert not checkpoint_uses_latents({"training_config": {"training_space": "pixel"}})
    assert not checkpoint_uses_latents({})


def _write_vae_checkpoint(tmp_path: Path, *, latent_channels: int) -> Path:
    config = SatelliteVAEConfig(latent_channels=latent_channels, hidden_channels=8)
    model = SatelliteVAE(config)
    checkpoint_path = tmp_path / "vae.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "vae_config": asdict(config),
            "latent_stats": {
                "mean": [0.1] * latent_channels,
                "std": [2.0] * latent_channels,
            },
        },
        checkpoint_path,
    )
    return checkpoint_path
