from __future__ import annotations

import pytest
import torch

from sim_satellite.latent import (
    LATENT_CODEC_AVERAGE_POOL_RGB,
    LATENT_CODEC_PIXEL_UNSHUFFLE_RGB,
    NativeLatentCodec,
    NativeLatentCodecConfig,
    checkpoint_pixel_model_config,
    checkpoint_uses_latents,
    load_checkpoint_latent_codec,
    native_latent_channels,
)
from sim_satellite.model import ModelConfig


def test_average_pool_latent_codec_encodes_and_decodes_rgb() -> None:
    codec = NativeLatentCodec(
        NativeLatentCodecConfig(
            codec=LATENT_CODEC_AVERAGE_POOL_RGB,
            downsample_factor=2,
            image_channels=3,
        ),
        torch=torch,
    )
    images = torch.arange(3 * 4 * 4, dtype=torch.float32).view(1, 3, 4, 4)

    latents = codec.encode(images)
    decoded = codec.decode(latents)

    assert latents.shape == (1, 3, 2, 2)
    assert decoded.shape == images.shape
    assert torch.allclose(latents[:, :, 0, 0], images[:, :, :2, :2].mean(dim=(2, 3)))


def test_pixel_unshuffle_latent_codec_round_trips_exactly() -> None:
    codec = NativeLatentCodec(
        NativeLatentCodecConfig(
            codec=LATENT_CODEC_PIXEL_UNSHUFFLE_RGB,
            downsample_factor=2,
            image_channels=3,
        ),
        torch=torch,
    )
    images = torch.randn(2, 3, 8, 8)

    latents = codec.encode(images)
    decoded = codec.decode(latents)

    assert latents.shape == (2, 12, 4, 4)
    assert torch.allclose(decoded, images)
    assert native_latent_channels(LATENT_CODEC_PIXEL_UNSHUFFLE_RGB, 2, 3) == 12


def test_latent_codec_downsamples_semantic_mask_with_area() -> None:
    codec = NativeLatentCodec(
        NativeLatentCodecConfig(
            codec=LATENT_CODEC_AVERAGE_POOL_RGB,
            downsample_factor=2,
            image_channels=3,
        ),
        torch=torch,
    )
    semantic_mask = torch.zeros(1, 2, 4, 4)
    semantic_mask[:, 0, :, :2] = 1.0
    semantic_mask[:, 1, :, 2:] = 1.0

    resized = codec.semantic_mask_for_model(semantic_mask)

    assert resized.shape == (1, 2, 2, 2)
    assert torch.allclose(resized[:, :, :, 0], torch.tensor([[[1.0, 1.0], [0.0, 0.0]]]))
    assert torch.allclose(resized[:, :, :, 1], torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]))


def test_latent_codec_rejects_non_divisible_spatial_size() -> None:
    codec = NativeLatentCodec(
        NativeLatentCodecConfig(
            codec=LATENT_CODEC_AVERAGE_POOL_RGB,
            downsample_factor=8,
            image_channels=3,
        ),
        torch=torch,
    )

    with pytest.raises(ValueError, match="must be divisible"):
        codec.encode(torch.zeros(1, 3, 10, 10))


def test_checkpoint_helpers_read_latent_training_metadata() -> None:
    model_config = ModelConfig(image_size=32, image_channels=3, mask_channels=2, hidden_channels=4)
    checkpoint = {
        "model_config": {
            "image_size": 32,
            "image_channels": 3,
            "mask_channels": 2,
            "hidden_channels": 4,
            "seed": 2026,
        },
        "training_config": {
            "training_space": "latent",
            "latent_codec": "average_pool_rgb",
            "latent_downsample_factor": 8,
            "image_size": 256,
            "image_channels": 3,
        },
    }

    codec = load_checkpoint_latent_codec(checkpoint, torch=torch)
    pixel_config = checkpoint_pixel_model_config(checkpoint, model_config)

    assert checkpoint_uses_latents(checkpoint)
    assert codec is not None
    assert codec.channels == 3
    assert pixel_config.image_size == 256
    assert pixel_config.image_channels == 3
