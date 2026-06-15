from __future__ import annotations

from dataclasses import asdict

import pytest
import torch

from sim_satellite.model import ModelConfig, build_model


def test_model_config_rejects_unknown_checkpoint_fields() -> None:
    config = asdict(ModelConfig(image_size=64))

    with pytest.raises(ValueError, match="Unknown model config fields: timesteps"):
        ModelConfig.from_dict({**config, "timesteps": 1000})


def test_conditional_unet_forward_shape_is_deterministic() -> None:
    torch.manual_seed(123)
    config = ModelConfig(image_size=32, mask_channels=4, hidden_channels=8)
    model = build_model(config)
    model.eval()

    image = torch.randn(2, config.image_channels, config.image_size, config.image_size)
    semantic_mask = torch.zeros(2, config.mask_channels, config.image_size, config.image_size)
    semantic_mask[:, 1] = 1.0
    timestep = torch.tensor([0.25, 0.75])

    with torch.no_grad():
        output = model(image, semantic_mask, timestep)

    assert output.shape == image.shape
    assert torch.isfinite(output).all()


def test_model_uses_learned_timestep_embedding_and_deep_attention() -> None:
    config = ModelConfig(image_size=32, mask_channels=4, hidden_channels=8)
    model = build_model(config)
    state = model.state_dict()

    assert state["time_embedding.1.weight"].shape == (32, 8)
    assert state["time_embedding.3.weight"].shape == (32, 32)
    assert state["down_blocks.0.0.time_projection.weight"].shape == (16, 32)
    assert "deep_down_attention.qkv.weight" in state

    image = torch.randn(2, config.image_channels, config.image_size, config.image_size)
    semantic_mask = torch.zeros(2, config.mask_channels, config.image_size, config.image_size)
    timestep = torch.tensor([0.25, 0.75])

    with torch.no_grad():
        output = model(image, semantic_mask, timestep)

    assert output.shape == image.shape
    assert torch.isfinite(output).all()
