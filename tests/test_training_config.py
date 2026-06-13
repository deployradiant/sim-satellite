from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sim_satellite.train import DEFAULT_TARGET, TrainingConfig  # noqa: E402


def test_training_config_defaults_are_deterministic() -> None:
    config = TrainingConfig()

    assert DEFAULT_TARGET == "modal"
    assert config.seed == 1337
    assert config.steps == 8
    assert config.batch_size == 4
    assert config.device == "cpu"
    assert config.manifest_path == "data/openearthmap/manifest.jsonl"
    assert config.checkpoint_path == "artifacts/tiny-sim-satellite.pt"
    assert config.mask_channels == 9


def test_training_config_round_trips_known_fields() -> None:
    config = TrainingConfig(steps=3, batch_size=2)

    assert TrainingConfig.from_dict(config.to_dict()) == config
    assert (
        TrainingConfig.from_dict(
            {
                **config.to_dict(),
                "future_field_from_modal": "ignored",
            }
        )
        == config
    )


def test_training_config_builds_matching_model_config() -> None:
    config = TrainingConfig(
        seed=7,
        image_size=16,
        image_channels=4,
        mask_channels=3,
        hidden_channels=8,
        timesteps=5,
    )

    model_config = config.model_config()

    assert model_config.seed == config.seed
    assert model_config.image_size == config.image_size
    assert model_config.image_channels == config.image_channels
    assert model_config.mask_channels == config.mask_channels
    assert model_config.hidden_channels == config.hidden_channels
    assert model_config.timesteps == config.timesteps


def test_training_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(steps=0)
    with pytest.raises(ValueError):
        TrainingConfig(batch_size=0)
    with pytest.raises(ValueError):
        TrainingConfig(learning_rate=0.0)
