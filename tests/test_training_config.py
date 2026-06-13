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
    _load_training_samples,
    _load_validation_batches,
    _random_square_crop_box,
    _snapshot_checkpoint_path,
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
    assert config.resume_from == ""
    assert config.save_every == 0
    assert config.plot_every == 0
    assert config.validation_split == ""
    assert config.validation_holdout_patterns == ""
    assert config.validation_every == 0
    assert config.validation_batches == 8
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
    with pytest.raises(ValueError):
        TrainingConfig(save_every=-1)
    with pytest.raises(ValueError):
        TrainingConfig(plot_every=-1)
    with pytest.raises(ValueError):
        TrainingConfig(validation_every=-1)
    with pytest.raises(ValueError):
        TrainingConfig(validation_batches=0)


def test_snapshot_checkpoint_path_includes_step() -> None:
    path = _snapshot_checkpoint_path(Path("artifacts/model.pt"), 50)

    assert path == Path("artifacts/model-step-50.pt")


def test_validation_holdout_patterns_are_excluded_from_training(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                '{"image": "extracted/OpenEarthMap_wo_xBD/abancay/images/a.tif", '
                '"mask": "extracted/OpenEarthMap_wo_xBD/abancay/labels/a.tif", '
                '"split": "train"}',
                '{"image": "extracted/OpenEarthMap_wo_xBD/tokyo/images/b.tif", '
                '"mask": "extracted/OpenEarthMap_wo_xBD/tokyo/labels/b.tif", '
                '"split": "train"}',
                '{"image": "extracted/OpenEarthMap_wo_xBD/maputo/images/c.tif", '
                '"mask": "extracted/OpenEarthMap_wo_xBD/maputo/labels/c.tif", '
                '"split": "train"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    samples = _load_training_samples(
        TrainingConfig(
            manifest_path=str(manifest_path),
            validation_holdout_patterns="abancay, maputo",
        )
    )

    assert [sample.image_path.name for sample in samples] == ["b.tif"]


def test_validation_batches_use_holdout_patterns(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(
            [
                '{"image": "extracted/OpenEarthMap_wo_xBD/abancay/images/a.tif", '
                '"mask": "extracted/OpenEarthMap_wo_xBD/abancay/labels/a.tif", '
                '"split": "train"}',
                '{"image": "extracted/OpenEarthMap_wo_xBD/tokyo/images/b.tif", '
                '"mask": "extracted/OpenEarthMap_wo_xBD/tokyo/labels/b.tif", '
                '"split": "train"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    batches = _load_validation_batches(
        TrainingConfig(
            manifest_path=str(manifest_path),
            validation_every=10,
            validation_holdout_patterns="abancay",
            batch_size=1,
        )
    )

    assert len(batches) == 1
    assert batches[0][0].image_path.name == "a.tif"


def test_validation_requires_split_or_holdout_patterns(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    manifest_path.write_text(
        '{"image": "images/a.tif", "mask": "labels/a.tif", "split": "train"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="validation_split or validation_holdout_patterns"):
        _load_validation_batches(
            TrainingConfig(manifest_path=str(manifest_path), validation_every=1)
        )


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
