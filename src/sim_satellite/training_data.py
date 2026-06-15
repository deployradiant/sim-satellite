"""Training-time crop, preload, and batch tensor helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sim_satellite.data.loader import SatelliteSample
from sim_satellite.data.prepared import PreparedSample, load_prepared_sample
from sim_satellite.data.tensors import labels_to_one_hot
from sim_satellite.model import ModelConfig


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


def preload_samples(samples: tuple[SatelliteSample, ...]) -> dict[SatelliteSample, PreparedSample]:
    import torch

    preloaded: dict[SatelliteSample, PreparedSample] = {}
    for sample in samples:
        preloaded[sample] = load_prepared_sample(sample.prepared_path, torch=torch)
    return preloaded


def load_batch(
    samples: tuple[SatelliteSample, ...],
    crop_boxes: list[tuple[int, int, int, int]],
    *,
    torch: Any,
    model_config: ModelConfig,
    device: Any,
    preloaded_samples: Mapping[SatelliteSample, PreparedSample],
) -> tuple[Any, Any]:
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
    mask_labels = [
        _load_mask_sample_tensor(
            sample,
            crop_box,
            torch=torch,
            model_config=model_config,
            preloaded_samples=preloaded_samples,
        )
        for sample, crop_box in zip(samples, crop_boxes, strict=True)
    ]
    image_batch = torch.stack(images).to(device).float() / 127.5 - 1.0
    label_batch = torch.stack(mask_labels).to(device)
    return image_batch, labels_to_one_hot(label_batch, torch=torch, config=model_config)


def random_square_crop_box(
    sample: SatelliteSample,
    *,
    crop_size: int,
    torch: Any,
    generator: Any,
    preloaded_samples: Mapping[SatelliteSample, PreparedSample],
) -> tuple[int, int, int, int]:
    width, height = _sample_image_size(sample, preloaded_samples)
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
    preloaded_samples: Mapping[SatelliteSample, PreparedSample],
) -> tuple[int, int, int, int]:
    width, height = _sample_image_size(sample, preloaded_samples)
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
    preloaded_samples: Mapping[SatelliteSample, PreparedSample],
) -> Any:
    prepared = _prepared_sample(sample, torch=torch, preloaded_samples=preloaded_samples)
    left, top, right, bottom = crop_box
    image = prepared.image[top:bottom, left:right, :].permute(2, 0, 1)
    if image.shape[-2:] == (model_config.image_size, model_config.image_size):
        return image.contiguous()
    resized = torch.nn.functional.interpolate(
        image.float().unsqueeze(0),
        size=(model_config.image_size, model_config.image_size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    return resized.clamp(0, 255).round().to(torch.uint8)


def _load_mask_sample_tensor(
    sample: SatelliteSample,
    crop_box: tuple[int, int, int, int],
    *,
    torch: Any,
    model_config: ModelConfig,
    preloaded_samples: Mapping[SatelliteSample, PreparedSample],
) -> Any:
    prepared = _prepared_sample(sample, torch=torch, preloaded_samples=preloaded_samples)
    left, top, right, bottom = crop_box
    mask = prepared.mask[top:bottom, left:right]
    if mask.shape[-2:] == (model_config.image_size, model_config.image_size):
        return mask.long().contiguous()
    return (
        torch.nn.functional.interpolate(
            mask.float().view(1, 1, mask.shape[0], mask.shape[1]),
            size=(model_config.image_size, model_config.image_size),
            mode="nearest",
        )
        .view(model_config.image_size, model_config.image_size)
        .long()
    )


def _sample_image_size(
    sample: SatelliteSample,
    preloaded_samples: Mapping[SatelliteSample, PreparedSample],
) -> tuple[int, int]:
    return _prepared_sample(sample, torch=None, preloaded_samples=preloaded_samples).size


def _prepared_sample(
    sample: SatelliteSample,
    *,
    torch: Any | None,
    preloaded_samples: Mapping[SatelliteSample, PreparedSample],
) -> PreparedSample:
    prepared = preloaded_samples.get(sample)
    if prepared is not None:
        return prepared
    raise KeyError(f"Prepared sample was not preloaded: {sample.prepared_path}")
