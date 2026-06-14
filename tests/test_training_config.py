from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from sim_satellite.data.loader import SatelliteSample
from sim_satellite.train import (
    TrainingConfig,
    _load_training_samples,
    _load_validation_batches,
    _random_square_crop_box,
)


def test_training_config_rejects_unknown_fields() -> None:
    config = TrainingConfig(steps=3, batch_size=2)

    with pytest.raises(ValueError, match="Unknown training config fields: future_field"):
        TrainingConfig.from_dict({**asdict(config), "future_field": "rejected"})


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
