from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sim_satellite.data.loader import SatelliteSample  # noqa: E402
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASS_IDS  # noqa: E402
from sim_satellite.train import (  # noqa: E402
    DEFAULT_TARGET,
    TrainingConfig,
    _random_square_crop_box,
)


def test_training_config_defaults_are_deterministic() -> None:
    config = TrainingConfig()

    assert DEFAULT_TARGET == "modal"
    assert config.seed == 1337
    assert config.steps == 8
    assert config.batch_size == 4
    assert config.device == "cpu"
    assert config.manifest_path == "data/openearthmap/manifest.jsonl"
    assert config.checkpoint_path == "artifacts/tiny-sim-satellite.pt"
    assert config.mask_channels == len(OPENEARTHMAP_CLASS_IDS)


def test_training_config_round_trips_known_fields() -> None:
    config = TrainingConfig(steps=3, batch_size=2)

    assert TrainingConfig.from_dict(config.to_dict()) == config


def test_training_config_rejects_unknown_fields() -> None:
    config = TrainingConfig(steps=3, batch_size=2)

    with pytest.raises(ValueError, match="Unknown training config fields: future_field"):
        TrainingConfig.from_dict({**config.to_dict(), "future_field": "rejected"})


def test_training_config_builds_matching_model_config() -> None:
    config = TrainingConfig(
        seed=7,
        image_size=16,
        image_channels=4,
        mask_channels=3,
        hidden_channels=8,
    )

    model_config = config.model_config()

    assert model_config.seed == config.seed
    assert model_config.image_size == config.image_size
    assert model_config.image_channels == config.image_channels
    assert model_config.mask_channels == config.mask_channels
    assert model_config.hidden_channels == config.hidden_channels


def test_training_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(steps=0)
    with pytest.raises(ValueError):
        TrainingConfig(batch_size=0)
    with pytest.raises(ValueError):
        TrainingConfig(learning_rate=0.0)


def test_random_square_crop_box_is_aligned_and_deterministic(tmp_path: Path) -> None:
    from PIL import Image

    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    Image.new("RGB", (10, 8)).save(image_path)
    Image.new("L", (10, 8)).save(mask_path)
    sample = SatelliteSample(image_path=image_path, mask_path=mask_path, split="train")

    crop_box = _random_square_crop_box(
        sample,
        crop_size=4,
        torch=_FakeTorch([3, 2]),
        generator=object(),
    )

    assert crop_box == (3, 2, 7, 6)


class _FakeTorch:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def randint(
        self,
        low: int,
        high: int,
        shape: tuple[int],
        *,
        generator: object,
        device: str,
    ) -> _FakeTensor:
        assert low == 0
        assert shape == (1,)
        assert device == "cpu"
        value = self.values.pop(0)
        assert value < high
        return _FakeTensor(value)


class _FakeTensor:
    def __init__(self, value: int) -> None:
        self.value = value

    def item(self) -> int:
        return self.value
