from __future__ import annotations

import pytest
import torch

from sim_satellite.model import ModelConfig, _semantic_mask_at, build_model


def test_model_rejects_image_size_that_makes_full_attention_too_large() -> None:
    with pytest.raises(ValueError, match="4096 tokens to avoid attention OOM"):
        build_model(ModelConfig(image_size=512))


def test_conditional_unet_forward_shape_is_deterministic() -> None:
    torch.manual_seed(123)
    config = ModelConfig(image_size=32, mask_channels=4, hidden_channels=8)
    model = build_model(config)
    model.eval()

    image = torch.randn(2, config.image_channels, config.image_size, config.image_size)
    semantic_mask = torch.zeros(2, config.mask_channels, config.image_size, config.image_size)
    semantic_mask[:, 1] = 1.0
    timestep = torch.tensor([0.25, 0.75])
    coordinates = torch.tensor([[0.0, 0.0], [12.0, 34.0]])

    with torch.no_grad():
        output = model(image, semantic_mask, timestep, coordinates)

    assert output.shape == image.shape
    assert torch.isfinite(output).all()


def test_model_output_changes_with_timestep() -> None:
    torch.manual_seed(123)
    config = ModelConfig(image_size=32, mask_channels=4, hidden_channels=8)
    model = build_model(config)
    model.eval()

    image = torch.randn(2, config.image_channels, config.image_size, config.image_size)
    semantic_mask = torch.zeros(2, config.mask_channels, config.image_size, config.image_size)
    semantic_mask[:, 1] = 1.0
    coordinates = torch.tensor([[0.0, 0.0], [12.0, 34.0]])

    with torch.no_grad():
        early_output = model(image, semantic_mask, torch.full((2,), 0.1), coordinates)
        late_output = model(image, semantic_mask, torch.full((2,), 0.9), coordinates)

    assert early_output.shape == image.shape
    assert late_output.shape == image.shape
    assert torch.isfinite(early_output).all()
    assert torch.isfinite(late_output).all()
    assert not torch.allclose(early_output, late_output)


def test_model_output_changes_with_coordinates() -> None:
    torch.manual_seed(123)
    config = ModelConfig(image_size=32, mask_channels=4, hidden_channels=8)
    model = build_model(config)
    model.eval()

    image = torch.randn(2, config.image_channels, config.image_size, config.image_size)
    semantic_mask = torch.zeros(2, config.mask_channels, config.image_size, config.image_size)
    semantic_mask[:, 1] = 1.0
    timestep = torch.full((2,), 0.5)

    with torch.no_grad():
        origin_output = model(image, semantic_mask, timestep, torch.zeros(2, 2))
        located_output = model(
            image,
            semantic_mask,
            timestep,
            torch.tensor([[6.0, 50.0], [7.0, 51.0]]),
        )

    assert origin_output.shape == image.shape
    assert located_output.shape == image.shape
    assert not torch.allclose(origin_output, located_output)


def test_model_rejects_coordinate_batch_mismatch() -> None:
    config = ModelConfig(image_size=32, mask_channels=4, hidden_channels=8)
    model = build_model(config)
    image = torch.randn(2, config.image_channels, config.image_size, config.image_size)
    semantic_mask = torch.zeros(2, config.mask_channels, config.image_size, config.image_size)

    with pytest.raises(ValueError, match="Coordinate batch size"):
        model(image, semantic_mask, torch.full((2,), 0.5), torch.zeros(1, 2))


def test_semantic_mask_pyramid_downsamples_one_hot_with_area() -> None:
    semantic_mask = torch.zeros(1, 2, 4, 4)
    semantic_mask[:, 0, :, :1] = 1.0
    semantic_mask[:, 1, :, 1:] = 1.0
    mask_pyramid = {(4, 4): semantic_mask}

    resized = _semantic_mask_at(semantic_mask, (2, 2), mask_pyramid)

    assert resized.shape == (1, 2, 2, 2)
    assert torch.allclose(resized[:, :, :, 0], torch.tensor([[[0.5, 0.5], [0.5, 0.5]]]))
    assert torch.allclose(resized[:, :, :, 1], torch.tensor([[[0.0, 0.0], [1.0, 1.0]]]))
    assert mask_pyramid[(2, 2)] is resized
