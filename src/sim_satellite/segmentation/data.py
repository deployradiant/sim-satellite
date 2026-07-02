"""Data preparation, splitting, cropping, and augmentation for segmentation."""

from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F

from sim_satellite.data.loader import SatelliteSample, read_manifest
from sim_satellite.data.prepared import PreparedSample, load_prepared_sample
from sim_satellite.segmentation.config import SegmentationTrainingConfig


class _PreparedSampleCache:
    def __init__(self, *, max_samples: int) -> None:
        self.max_samples = max_samples
        self.samples: dict[SatelliteSample, PreparedSample] = {}
        self.order: list[SatelliteSample] = []

    def get(self, sample: SatelliteSample) -> PreparedSample:
        cached = self.samples.get(sample)
        if cached is not None:
            return cached
        if not sample.prepared_path.exists():
            raise FileNotFoundError(
                f"Prepared sample data not found: {sample.prepared_path}. "
                "Run `sim-satellite openearthmap --force-prepared` first."
            )
        prepared = load_prepared_sample(sample.prepared_path, torch=torch)
        if self.max_samples > 0:
            self.samples[sample] = prepared
            self.order.append(sample)
            while len(self.order) > self.max_samples:
                evicted = self.order.pop(0)
                self.samples.pop(evicted, None)
        return prepared


def _split_training_samples(
    config: SegmentationTrainingConfig,
) -> tuple[tuple[SatelliteSample, ...], tuple[SatelliteSample, ...]]:
    manifest_path = Path(config.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Segmentation manifest not found: {manifest_path}")

    train_candidates = _samples_for_split(manifest_path, config.split)
    if not train_candidates:
        raise RuntimeError(f"No samples for split `{config.split}` in {manifest_path}")

    if config.validation_split:
        validation_samples = _samples_for_split(manifest_path, config.validation_split)
        train_samples = train_candidates
    else:
        patterns = _validation_holdout_patterns(config.validation_holdout_patterns)
        if patterns:
            validation_samples = tuple(
                sample for sample in train_candidates if _sample_matches_any(sample, patterns)
            )
            train_samples = tuple(
                sample for sample in train_candidates if not _sample_matches_any(sample, patterns)
            )
        else:
            shuffled = list(train_candidates)
            random.Random(config.seed).shuffle(shuffled)
            validation_count = int(round(len(shuffled) * config.validation_fraction))
            validation_count = min(max(1, validation_count), len(shuffled) - 1)
            validation_samples = tuple(shuffled[:validation_count])
            train_samples = tuple(shuffled[validation_count:])

    train_samples = _shuffled_limit_samples(
        train_samples,
        max_samples=config.max_train_samples,
        seed=config.seed,
    )
    validation_samples = _shuffled_limit_samples(
        validation_samples,
        max_samples=config.max_validation_samples,
        seed=config.seed + 1,
    )
    if not train_samples:
        raise RuntimeError("No segmentation training samples selected")
    if not validation_samples:
        raise RuntimeError("No segmentation validation samples selected")
    return train_samples, validation_samples


def _samples_for_split(
    manifest_path: str | Path,
    split: str,
) -> tuple[SatelliteSample, ...]:
    split_names = _split_names(split)
    samples = tuple(read_manifest(manifest_path))
    if split_names == ("*",):
        return samples
    return tuple(sample for sample in samples if sample.split in split_names)


def _split_names(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if len(names) == 1 and names[0].lower() in {"*", "all"}:
        return ("*",)
    return names


def _limit_samples(
    samples: tuple[SatelliteSample, ...],
    max_samples: int,
) -> tuple[SatelliteSample, ...]:
    if max_samples <= 0 or len(samples) <= max_samples:
        return samples
    return samples[:max_samples]


def _shuffled_limit_samples(
    samples: tuple[SatelliteSample, ...],
    *,
    max_samples: int,
    seed: int,
) -> tuple[SatelliteSample, ...]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    return _limit_samples(tuple(shuffled), max_samples)


def _validation_holdout_patterns(value: str) -> tuple[str, ...]:
    return tuple(pattern.strip() for pattern in value.split(",") if pattern.strip())


def _sample_matches_any(sample: SatelliteSample, patterns: Sequence[str]) -> bool:
    image_path = sample.image_path.as_posix()
    mask_path = sample.mask_path.as_posix()
    return any(pattern in image_path or pattern in mask_path for pattern in patterns)


def _load_segmentation_batch(
    samples: Sequence[SatelliteSample],
    crop_boxes: Sequence[tuple[int, int, int, int]],
    *,
    cache: _PreparedSampleCache,
    image_size: int,
    device: torch.device,
    context_size: int = 0,
    augmentation_config: SegmentationTrainingConfig | None = None,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    images: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for sample, crop_box in zip(samples, crop_boxes, strict=True):
        prepared = cache.get(sample)
        image, label = _prepared_crop(
            prepared,
            crop_box,
            image_size=image_size,
            context_size=context_size,
        )
        if augmentation_config is not None and augmentation_config.augment:
            if rng is None:
                raise RuntimeError("augmentation requires a random generator")
            image, label = _augment_segmentation_sample(image, label, augmentation_config, rng)
        images.append(image)
        labels.append(label)
    return torch.stack(images).to(device), torch.stack(labels).to(device)


def _prepared_crop(
    prepared: PreparedSample,
    crop_box: tuple[int, int, int, int],
    *,
    image_size: int,
    context_size: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    left, top, right, bottom = crop_box
    image = _prepared_image_crop(prepared, crop_box, image_size=image_size)
    label = prepared.mask[top:bottom, left:right]
    if context_size > image_size:
        context_box = _context_crop_box(prepared.size, crop_box, context_size)
        context_image = _prepared_image_crop(prepared, context_box, image_size=image_size)
        image = torch.cat((image, context_image), dim=0)

    if label.shape[-2:] != (image_size, image_size):
        label = (
            F.interpolate(
                label.float().view(1, 1, label.shape[0], label.shape[1]),
                size=(image_size, image_size),
                mode="nearest",
            )
            .view(image_size, image_size)
            .long()
        )
    else:
        label = label.long().contiguous()
    return image.float() / 255.0, label


def _prepared_image_crop(
    prepared: PreparedSample,
    crop_box: tuple[int, int, int, int],
    *,
    image_size: int,
) -> torch.Tensor:
    left, top, right, bottom = crop_box
    image = prepared.image[top:bottom, left:right, :].permute(2, 0, 1)
    if image.shape[-2:] == (image_size, image_size):
        return image.contiguous()
    return (
        F.interpolate(
            image.float().unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .clamp(0, 255)
        .round()
        .to(torch.uint8)
    )


def _context_crop_box(
    image_size: tuple[int, int],
    target_box: tuple[int, int, int, int],
    context_size: int,
) -> tuple[int, int, int, int]:
    width, height = image_size
    left, top, right, bottom = target_box
    target_center_x = (left + right) // 2
    target_center_y = (top + bottom) // 2

    context_width = min(width, context_size)
    context_height = min(height, context_size)
    context_left = _clamped_start(target_center_x - context_width // 2, width, context_width)
    context_top = _clamped_start(target_center_y - context_height // 2, height, context_height)
    return (
        context_left,
        context_top,
        context_left + context_width,
        context_top + context_height,
    )


def _clamped_start(start: int, extent: int, crop_extent: int) -> int:
    return min(max(0, start), max(0, extent - crop_extent))


def _augment_segmentation_sample(
    image: torch.Tensor,
    label: torch.Tensor,
    config: SegmentationTrainingConfig,
    rng: random.Random,
) -> tuple[torch.Tensor, torch.Tensor]:
    if rng.random() < config.flip_probability:
        image = torch.flip(image, dims=(2,))
        label = torch.flip(label, dims=(1,))
    if rng.random() < config.flip_probability:
        image = torch.flip(image, dims=(1,))
        label = torch.flip(label, dims=(0,))
    if rng.random() < config.rotate_probability:
        rotations = rng.randrange(4)
        image = torch.rot90(image, rotations, dims=(1, 2))
        label = torch.rot90(label, rotations, dims=(0, 1))
    if config.color_jitter_strength > 0.0:
        image = _color_jitter(image, strength=config.color_jitter_strength, rng=rng)
    if config.blur_probability > 0.0 and rng.random() < config.blur_probability:
        image = _blur_image(image)
    return image.clamp(0.0, 1.0), label.contiguous()


def _color_jitter(
    image: torch.Tensor,
    *,
    strength: float,
    rng: random.Random,
) -> torch.Tensor:
    brightness = rng.uniform(1.0 - strength, 1.0 + strength)
    contrast = rng.uniform(1.0 - strength, 1.0 + strength)
    saturation = rng.uniform(1.0 - strength, 1.0 + strength)

    image = image * brightness
    channel_mean = image.mean(dim=(1, 2), keepdim=True)
    image = (image - channel_mean) * contrast + channel_mean
    grayscale = image.mean(dim=0, keepdim=True)
    image = (image - grayscale) * saturation + grayscale
    return image


def _blur_image(image: torch.Tensor) -> torch.Tensor:
    return F.avg_pool2d(
        image.unsqueeze(0),
        kernel_size=3,
        stride=1,
        padding=1,
        count_include_pad=False,
    ).squeeze(0)


def _random_square_crop_box(
    prepared: PreparedSample,
    crop_size: int,
    rng: random.Random,
) -> tuple[int, int, int, int]:
    width, height = prepared.size
    size = min(width, height, crop_size)
    if size < 1:
        raise ValueError(f"Prepared sample has invalid size: {prepared.size}")
    left = 0 if width == size else rng.randrange(0, width - size + 1)
    top = 0 if height == size else rng.randrange(0, height - size + 1)
    return left, top, left + size, top + size


def _tile_windows(
    image_size: tuple[int, int],
    tile_size: int,
    *,
    stride: int = 0,
) -> tuple[tuple[int, int, int, int], ...]:
    width, height = image_size
    if width < 1 or height < 1:
        raise ValueError(f"Invalid image size: {image_size}")
    if tile_size < 1:
        raise ValueError("tile_size must be positive")
    if stride < 0:
        raise ValueError("stride must be non-negative")
    if stride == 0:
        stride = tile_size

    windows: list[tuple[int, int, int, int]] = []
    for top in _tile_starts(height, tile_size, stride):
        bottom = min(height, top + tile_size)
        for left in _tile_starts(width, tile_size, stride):
            right = min(width, left + tile_size)
            windows.append((left, top, right, bottom))
    return tuple(windows)


def _tile_starts(extent: int, tile_size: int, stride: int) -> tuple[int, ...]:
    if extent <= tile_size:
        return (0,)
    if stride >= tile_size:
        return tuple(range(0, extent, tile_size))

    starts = list(range(0, extent - tile_size + 1, stride))
    final_start = extent - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return tuple(starts)
