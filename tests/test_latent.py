from __future__ import annotations

import pytest
import torch

from sim_satellite.latent import (
    decode_image_latent,
    encode_image_latent,
    encode_mask_latent,
    latent_image_channels,
    validate_latent_downsample_factor,
)


def test_spatial_latent_codec_round_trips_losslessly() -> None:
    image = torch.arange(48, dtype=torch.float32).view(1, 3, 4, 4)

    latent = encode_image_latent(image, factor=2, torch=torch)
    decoded = decode_image_latent(latent, factor=2, output_size=4, torch=torch)

    assert latent.shape == (1, 12, 2, 2)
    assert torch.equal(decoded, image)


def test_latent_image_channels_expand_by_patch_area() -> None:
    assert latent_image_channels(3, factor=4) == 48


def test_fixed_latent_mask_codec_uses_nearest_resize() -> None:
    mask = torch.zeros(1, 2, 4, 4)
    mask[:, 1, 2:, 2:] = 1.0

    latent = encode_mask_latent(mask, size=2, torch=torch)

    assert latent.shape == (1, 2, 2, 2)
    assert latent[0, 1].tolist() == [[0.0, 0.0], [0.0, 1.0]]


def test_latent_downsample_factor_validation() -> None:
    validate_latent_downsample_factor(256, 4)

    with pytest.raises(ValueError, match="positive"):
        validate_latent_downsample_factor(256, 0)
    with pytest.raises(ValueError, match="divisible"):
        validate_latent_downsample_factor(250, 4)
    with pytest.raises(ValueError, match="at least 32"):
        validate_latent_downsample_factor(64, 4)
