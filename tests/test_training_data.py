from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from sim_satellite.data.loader import SatelliteSample
from sim_satellite.data.prepared import write_prepared_sample
from sim_satellite.model import ModelConfig
from sim_satellite.training_data import (
    load_batch,
    preload_samples,
    random_square_crop_box,
)


def test_preloaded_batch_loading_uses_aligned_crop(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    prepared_path = tmp_path / "prepared.pt"
    Image.new("RGB", (16, 12), (255, 0, 0)).save(image_path)
    Image.new("L", (16, 12), 1).save(mask_path)
    config = ModelConfig(image_size=8, mask_channels=2)
    sample = SatelliteSample(
        image_path=image_path,
        mask_path=mask_path,
        prepared_path=prepared_path,
        split="train",
    )
    write_prepared_sample(sample, torch=torch, config=config)

    preloaded = preload_samples((sample,))
    crop_box = random_square_crop_box(
        sample,
        crop_size=8,
        torch=_FakeTorch([3, 2]),
        generator=object(),
        preloaded_samples=preloaded,
    )
    images, masks = load_batch(
        (sample,),
        [crop_box],
        torch=torch,
        model_config=config,
        device=torch.device("cpu"),
        preloaded_samples=preloaded,
    )

    assert crop_box == (3, 2, 11, 10)
    assert images.shape == (1, 3, 8, 8)
    assert masks.shape == (1, 2, 8, 8)
    assert torch.allclose(images[0, :, 0, 0], torch.tensor([1.0, -1.0, -1.0]))
    assert torch.all(masks[:, 1] == 1.0)


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
