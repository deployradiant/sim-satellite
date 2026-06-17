"""Training-time crop, preload, and batch tensor helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
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


def preload_samples(
    samples: tuple[SatelliteSample, ...],
    *,
    progress: Callable[[int, int], None] | None = None,
) -> dict[SatelliteSample, PreparedSample]:
    import torch

    preloaded: dict[SatelliteSample, PreparedSample] = {}
    for index, sample in enumerate(samples, start=1):
        preloaded[sample] = load_prepared_sample(sample.prepared_path, torch=torch)
        if progress is not None and _should_report(index, len(samples)):
            progress(index, len(samples))
    return preloaded


def _should_report(current: int, total: int) -> bool:
    if total <= 0:
        return False
    if current == 1 or current == total or total <= 10:
        return True
    step = max(1, total // 10)
    return current % step == 0


def load_batch(
    samples: tuple[SatelliteSample, ...],
    crop_boxes: list[tuple[int, int, int, int]],
    *,
    torch: Any,
    model_config: ModelConfig,
    device: Any,
    preloaded_samples: Mapping[SatelliteSample, PreparedSample],
) -> tuple[Any, Any, Any | None]:
    missing_samples = tuple(sample for sample in samples if sample not in preloaded_samples)
    assert not missing_samples, "Samples must be preloaded prepared data: " + ", ".join(
        str(sample.prepared_path) for sample in missing_samples
    )

    images: list[Any] = []
    mask_labels: list[Any] = []
    coordinates: list[Any] = []
    for sample, crop_box in zip(samples, crop_boxes, strict=True):
        prepared = preloaded_samples[sample]
        left, top, right, bottom = crop_box
        if prepared.coordinates is not None:
            coordinates.append(prepared.coordinates.float())

        image = prepared.image[top:bottom, left:right, :].permute(2, 0, 1)
        if image.shape[-2:] == (model_config.image_size, model_config.image_size):
            images.append(image.contiguous())
        else:
            resized = torch.nn.functional.interpolate(
                image.float().unsqueeze(0),
                size=(model_config.image_size, model_config.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            images.append(resized.clamp(0, 255).round().to(torch.uint8))

        mask = prepared.mask[top:bottom, left:right]
        if mask.shape[-2:] == (model_config.image_size, model_config.image_size):
            mask_labels.append(mask.long().contiguous())
        else:
            mask_labels.append(
                torch.nn.functional.interpolate(
                    mask.float().view(1, 1, mask.shape[0], mask.shape[1]),
                    size=(model_config.image_size, model_config.image_size),
                    mode="nearest",
                )
                .view(model_config.image_size, model_config.image_size)
                .long()
            )

    image_batch = torch.stack(images).to(device).float() / 127.5 - 1.0
    label_batch = torch.stack(mask_labels).to(device)
    if coordinates and len(coordinates) != len(samples):
        raise RuntimeError("Batch mixes samples with and without coordinates")
    coordinate_batch = torch.stack(coordinates).to(device) if coordinates else None
    return (
        image_batch,
        labels_to_one_hot(label_batch, torch=torch, config=model_config),
        coordinate_batch,
    )


def random_square_crop_box(
    sample: SatelliteSample,
    *,
    crop_size: int,
    torch: Any,
    generator: Any,
    preloaded_samples: Mapping[SatelliteSample, PreparedSample],
) -> tuple[int, int, int, int]:
    assert sample in preloaded_samples, (
        f"Sample must be preloaded prepared data: {sample.prepared_path}"
    )
    width, height = preloaded_samples[sample].size
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
    assert sample in preloaded_samples, (
        f"Sample must be preloaded prepared data: {sample.prepared_path}"
    )
    width, height = preloaded_samples[sample].size
    size = min(width, height, crop_size)
    if size < 1:
        raise ValueError(f"Sample has invalid dimensions: {sample.image_path} is {(width, height)}")
    left = (width - size) // 2
    top = (height - size) // 2
    return left, top, left + size, top + size
