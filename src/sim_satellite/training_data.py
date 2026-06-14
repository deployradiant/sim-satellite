"""Training-time crop, preload, and batch tensor helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from PIL import Image

from sim_satellite.data.loader import SatelliteSample
from sim_satellite.data.tensors import (
    image_to_tensor,
    load_image_tensor,
    load_mask_tensor,
    mask_to_tensor,
)
from sim_satellite.model import ModelConfig


@dataclass(frozen=True)
class PreloadedSample:
    image: Image.Image
    mask: Image.Image
    size: tuple[int, int]


def unique_samples(
    training_samples: tuple[SatelliteSample, ...],
    validation_batches: tuple[tuple[SatelliteSample, ...], ...],
) -> tuple[SatelliteSample, ...]:
    samples: list[SatelliteSample] = []
    seen: set[SatelliteSample] = set()
    for sample in training_samples:
        if sample not in seen:
            seen.add(sample)
            samples.append(sample)
    for batch in validation_batches:
        for sample in batch:
            if sample not in seen:
                seen.add(sample)
                samples.append(sample)
    return tuple(samples)


def preload_samples(samples: tuple[SatelliteSample, ...]) -> dict[SatelliteSample, PreloadedSample]:
    preloaded: dict[SatelliteSample, PreloadedSample] = {}
    for sample in samples:
        with Image.open(sample.image_path) as image:
            loaded_image = image.convert("RGB").copy()
        with Image.open(sample.mask_path) as mask:
            loaded_mask = mask.copy()
        if loaded_image.size != loaded_mask.size:
            raise ValueError(
                f"Image and mask sizes must match for aligned crops: "
                f"{sample.image_path} is {loaded_image.size}, "
                f"{sample.mask_path} is {loaded_mask.size}"
            )
        preloaded[sample] = PreloadedSample(
            image=loaded_image,
            mask=loaded_mask,
            size=loaded_image.size,
        )
    return preloaded


def load_batch(
    samples: tuple[SatelliteSample, ...],
    crop_boxes: list[tuple[int, int, int, int]],
    *,
    torch: Any,
    model_config: ModelConfig,
    device: Any,
    preloaded_samples: Mapping[SatelliteSample, PreloadedSample] | None = None,
) -> tuple[Any, Any]:
    preloaded_samples = preloaded_samples or {}
    images = [
        _load_image_sample_tensor(
            sample,
            crop_box,
            torch=torch,
            model_config=model_config,
            preloaded_samples=preloaded_samples,
        )
        for sample, crop_box in zip(samples, crop_boxes, strict=True)
    ]
    masks = [
        _load_mask_sample_tensor(
            sample,
            crop_box,
            torch=torch,
            model_config=model_config,
            preloaded_samples=preloaded_samples,
        )
        for sample, crop_box in zip(samples, crop_boxes, strict=True)
    ]
    return torch.stack(images).to(device), torch.stack(masks).to(device)


def random_square_crop_box(
    sample: SatelliteSample,
    *,
    crop_size: int,
    torch: Any,
    generator: Any,
    preloaded_samples: Mapping[SatelliteSample, PreloadedSample] | None = None,
) -> tuple[int, int, int, int]:
    width, height = _sample_image_size(sample, preloaded_samples or {})
    size = min(width, height, crop_size)
    if size < 1:
        raise ValueError(f"Sample has invalid dimensions: {sample.image_path} is {(width, height)}")

    max_left = width - size
    max_top = height - size
    left = (
        0
        if max_left <= 0
        else int(torch.randint(0, max_left + 1, (1,), generator=generator, device="cpu").item())
    )
    top = (
        0
        if max_top <= 0
        else int(torch.randint(0, max_top + 1, (1,), generator=generator, device="cpu").item())
    )
    return left, top, left + size, top + size


def center_square_crop_box(
    sample: SatelliteSample,
    *,
    crop_size: int,
    preloaded_samples: Mapping[SatelliteSample, PreloadedSample] | None = None,
) -> tuple[int, int, int, int]:
    width, height = _sample_image_size(sample, preloaded_samples or {})
    size = min(width, height, crop_size)
    if size < 1:
        raise ValueError(f"Sample has invalid dimensions: {sample.image_path} is {(width, height)}")
    left = (width - size) // 2
    top = (height - size) // 2
    return left, top, left + size, top + size


def _load_image_sample_tensor(
    sample: SatelliteSample,
    crop_box: tuple[int, int, int, int],
    *,
    torch: Any,
    model_config: ModelConfig,
    preloaded_samples: Mapping[SatelliteSample, PreloadedSample],
) -> Any:
    preloaded = preloaded_samples.get(sample)
    if preloaded is not None:
        return image_to_tensor(
            preloaded.image,
            torch=torch,
            config=model_config,
            crop_box=crop_box,
        )
    return load_image_tensor(sample.image_path, torch=torch, config=model_config, crop_box=crop_box)


def _load_mask_sample_tensor(
    sample: SatelliteSample,
    crop_box: tuple[int, int, int, int],
    *,
    torch: Any,
    model_config: ModelConfig,
    preloaded_samples: Mapping[SatelliteSample, PreloadedSample],
) -> Any:
    preloaded = preloaded_samples.get(sample)
    if preloaded is not None:
        return mask_to_tensor(
            preloaded.mask,
            torch=torch,
            config=model_config,
            crop_box=crop_box,
        )
    return load_mask_tensor(sample.mask_path, torch=torch, config=model_config, crop_box=crop_box)


def _sample_image_size(
    sample: SatelliteSample,
    preloaded_samples: Mapping[SatelliteSample, PreloadedSample],
) -> tuple[int, int]:
    preloaded = preloaded_samples.get(sample)
    if preloaded is not None:
        return preloaded.size
    return _aligned_image_size(sample)


def _aligned_image_size(sample: SatelliteSample) -> tuple[int, int]:
    with Image.open(sample.image_path) as image:
        image_size = image.size
    with Image.open(sample.mask_path) as mask:
        mask_size = mask.size
    if image_size != mask_size:
        raise ValueError(
            f"Image and mask sizes must match for aligned crops: "
            f"{sample.image_path} is {image_size}, {sample.mask_path} is {mask_size}"
        )
    return image_size
