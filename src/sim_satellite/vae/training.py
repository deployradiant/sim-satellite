"""Training and reconstruction audits for the native satellite VAE."""

from __future__ import annotations

import json
import random
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import click
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn
from torch.nn import functional as F

from sim_satellite.data.loader import SatelliteDataset, SatelliteSample
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES
from sim_satellite.data.prepared import load_prepared_sample
from sim_satellite.training_config import DEFAULT_MANIFEST_PATH
from sim_satellite.vae.metrics import (
    edge_map,
    reconstruction_metrics,
    reconstruction_metrics_by_class,
)
from sim_satellite.vae.model import SatelliteVAE, SatelliteVAEConfig


@dataclass(frozen=True)
class VAETrainingConfig:
    seed: int = 2026
    manifest_path: str = DEFAULT_MANIFEST_PATH
    split: str = "train"
    train_fraction: float = 0.5
    validation_split: str = "sample"
    validation_samples: int = 32
    image_size: int = 256
    eval_size: int = 256
    batch_size: int = 4
    steps: int = 5000
    learning_rate: float = 1.5e-4
    latent_channels: int = 24
    hidden_channels: int = 32
    downsample_factor: int = 4
    kl_weight: float = 1e-6
    mse_weight: float = 0.2
    reconstruction_exponential_weight: float = 0.5
    reconstruction_exponential_scale: float = 12.0
    edge_weight: float = 0.0
    class_balanced_l1_weight: float = 0.0
    class_balanced_edge_weight: float = 0.0
    include_unlabeled_in_class_loss: bool = False
    class_balanced_crop_probability: float = 0.0
    include_unlabeled_in_class_crop: bool = False
    device: str = "auto"
    cache_images: int = 128
    log_every: int = 250
    grid_samples: int = 8
    checkpoint_path: str = "artifacts/openearthmap-custom-vae-f4-c24-half.pt"
    metrics_path: str = "artifacts/openearthmap-custom-vae-f4-c24-half-metrics.json"
    grid_path: str = "artifacts/openearthmap-custom-vae-f4-c24-half-grid.png"

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> VAETrainingConfig:
        names = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - names)
        if unknown:
            raise ValueError(f"Unknown VAE training config fields: {', '.join(unknown)}")
        return cls(**{name: values[name] for name in names if name in values})

    def vae_config(self) -> SatelliteVAEConfig:
        return SatelliteVAEConfig(
            latent_channels=self.latent_channels,
            hidden_channels=self.hidden_channels,
            downsample_factor=self.downsample_factor,
        )


@dataclass(frozen=True)
class VAETrainingResult:
    checkpoint_path: str
    metrics_path: str
    grid_path: str
    train_image_count: int
    validation_image_count: int
    final_loss: float
    custom_vae: dict[str, float]
    custom_vae_by_class: dict[str, Any]
    latent_stats: dict[str, Any]


class TensorCache:
    def __init__(self, max_items: int) -> None:
        self.max_items = max_items
        self._items: OrderedDict[Path, PreparedTrainingTensor] = OrderedDict()

    def get(self, sample: SatelliteSample) -> PreparedTrainingTensor:
        cached = self._items.get(sample.prepared_path)
        if cached is not None:
            self._items.move_to_end(sample.prepared_path)
            return cached

        prepared = _load_prepared_training_tensor(sample)
        self._items[sample.prepared_path] = prepared
        self._items.move_to_end(sample.prepared_path)
        while len(self._items) > self.max_items:
            self._items.popitem(last=False)
        return prepared


@dataclass(frozen=True)
class PreparedTrainingTensor:
    image: torch.Tensor
    labels: torch.Tensor


def load_vae_training_config_file(config_path: Path) -> VAETrainingConfig:
    values = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError("VAE training config JSON must be an object")
    return load_vae_training_config_values(values)


def load_vae_training_config_values(values: Mapping[str, Any]) -> VAETrainingConfig:
    config = VAETrainingConfig.from_dict(values)
    _validate_config(config)
    return config


def split_vae_samples(
    config: VAETrainingConfig,
) -> tuple[tuple[SatelliteSample, ...], tuple[SatelliteSample, ...]]:
    split = None if config.split == "" else config.split
    samples = list(SatelliteDataset(config.manifest_path, split=split))
    if not samples:
        raise ValueError("VAE training found no samples")

    if config.validation_split == "city":
        return _split_vae_samples_by_group(samples, config)
    return _split_vae_samples_by_sample(samples, config)


def _split_vae_samples_by_sample(
    samples: Sequence[SatelliteSample],
    config: VAETrainingConfig,
) -> tuple[tuple[SatelliteSample, ...], tuple[SatelliteSample, ...]]:
    samples = list(samples)
    random.Random(config.seed).shuffle(samples)
    train_count = int(len(samples) * config.train_fraction)
    train_samples = tuple(samples[:train_count])
    validation_samples = tuple(samples[train_count : train_count + config.validation_samples])
    if not train_samples:
        raise ValueError("VAE training split produced no training samples")
    if len(validation_samples) != config.validation_samples:
        raise ValueError(
            "VAE training split does not leave enough validation samples after train_fraction"
        )
    return train_samples, validation_samples


def _split_vae_samples_by_group(
    samples: Sequence[SatelliteSample],
    config: VAETrainingConfig,
) -> tuple[tuple[SatelliteSample, ...], tuple[SatelliteSample, ...]]:
    grouped: dict[str, list[SatelliteSample]] = {}
    for sample in samples:
        grouped.setdefault(_sample_validation_group(sample), []).append(sample)
    if len(grouped) < 2:
        raise ValueError("city validation split requires at least two sample groups")

    rng = random.Random(config.seed)
    group_names = sorted(grouped)
    rng.shuffle(group_names)
    train_target = int(len(samples) * config.train_fraction)
    train_groups: list[str] = []
    for group_name in group_names:
        remaining = [name for name in group_names if name not in train_groups]
        remaining_samples = sum(len(grouped[name]) for name in remaining)
        if remaining_samples <= config.validation_samples:
            break
        train_groups.append(group_name)
        train_count = sum(len(grouped[name]) for name in train_groups)
        if train_count >= train_target:
            break

    validation_groups = [name for name in group_names if name not in train_groups]
    train_samples = tuple(sample for name in train_groups for sample in grouped[name])
    validation_pool = [sample for name in validation_groups for sample in grouped[name]]
    rng.shuffle(validation_pool)
    validation_samples = tuple(validation_pool[: config.validation_samples])

    if not train_samples:
        raise ValueError("VAE training city split produced no training samples")
    if len(validation_samples) != config.validation_samples:
        raise ValueError(
            "VAE training city split does not leave enough validation samples after train_fraction"
        )
    return train_samples, validation_samples


def run_vae_training(config: VAETrainingConfig) -> VAETrainingResult:
    device = _training_device(config)
    torch.manual_seed(config.seed)
    rng = random.Random(config.seed)
    train_samples, validation_samples = split_vae_samples(config)
    cache = TensorCache(config.cache_images)
    model = SatelliteVAE(config.vae_config()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    history: list[dict[str, float]] = []

    start_time = time.time()
    model.train()
    final_loss = 0.0
    for step in range(1, config.steps + 1):
        batch_crops = (
            _random_training_crop(
                cache.get(rng.choice(train_samples)),
                config.image_size,
                rng,
                config=config,
            )
            for _ in range(config.batch_size)
        )
        images, labels = zip(*batch_crops, strict=True)
        batch = torch.stack(images).to(device)
        batch_labels = torch.stack(labels).to(device)
        recon, mu, logvar = model(batch)
        losses = _vae_losses(
            batch,
            recon,
            mu,
            logvar,
            config,
            labels=batch_labels,
        )
        loss = losses["loss"]
        final_loss = float(loss.detach().cpu().item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % config.log_every == 0 or step == config.steps:
            record = {
                "step": float(step),
                **{name: float(value.detach().cpu().item()) for name, value in losses.items()},
                "elapsed_seconds": round(time.time() - start_time, 1),
            }
            history.append(record)
            click.echo(json.dumps(record), err=True)

    validation_crops = [
        _load_prepared_eval_crop(sample, config.eval_size) for sample in validation_samples
    ]
    validation_real = torch.stack([image for image, _labels in validation_crops])
    validation_labels = torch.stack([labels for _image, labels in validation_crops])
    model.eval()
    custom_recon = _custom_reconstruct(model, validation_real, device=device)
    custom_metrics = reconstruction_metrics(
        _to_unit(validation_real),
        _to_unit(custom_recon),
        torch=torch,
    )
    custom_metrics_by_class = reconstruction_metrics_by_class(
        _to_unit(validation_real),
        _to_unit(custom_recon),
        validation_labels,
        class_names=OPENEARTHMAP_CLASSES,
        torch=torch,
    )
    latent_stats = _latent_stats(model, train_samples, config=config, device=device)

    checkpoint_path = Path(config.checkpoint_path)
    metrics_path = Path(config.metrics_path)
    grid_path = Path(config.grid_path)
    _write_checkpoint(
        checkpoint_path,
        model=model,
        config=config,
        train_samples=train_samples,
        validation_samples=validation_samples,
        history=history,
        latent_stats=latent_stats,
    )
    _write_grid(
        grid_path,
        real=validation_real[: config.grid_samples],
        custom=custom_recon[: config.grid_samples],
    )
    _write_metrics(
        metrics_path,
        config=config,
        device=device,
        train_samples=train_samples,
        validation_samples=validation_samples,
        custom_metrics=custom_metrics,
        custom_metrics_by_class=custom_metrics_by_class,
        latent_stats=latent_stats,
        checkpoint_path=checkpoint_path,
        grid_path=grid_path,
        history=history,
    )

    return VAETrainingResult(
        checkpoint_path=str(checkpoint_path),
        metrics_path=str(metrics_path),
        grid_path=str(grid_path),
        train_image_count=len(train_samples),
        validation_image_count=len(validation_samples),
        final_loss=final_loss,
        custom_vae=custom_metrics,
        custom_vae_by_class=custom_metrics_by_class,
        latent_stats=latent_stats,
    )


def _validate_config(config: VAETrainingConfig) -> None:
    for name in (
        "validation_samples",
        "image_size",
        "eval_size",
        "batch_size",
        "steps",
        "learning_rate",
        "latent_channels",
        "hidden_channels",
        "downsample_factor",
        "cache_images",
        "log_every",
        "grid_samples",
    ):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if config.train_fraction <= 0.0 or config.train_fraction >= 1.0:
        raise ValueError("train_fraction must be greater than 0 and less than 1")
    if config.validation_split not in {"sample", "city"}:
        raise ValueError("validation_split must be one of: sample, city")
    for name in (
        "kl_weight",
        "mse_weight",
        "reconstruction_exponential_weight",
        "edge_weight",
        "class_balanced_l1_weight",
        "class_balanced_edge_weight",
    ):
        if getattr(config, name) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if config.reconstruction_exponential_scale <= 0.0:
        raise ValueError("reconstruction_exponential_scale must be positive")
    if config.class_balanced_crop_probability < 0.0 or config.class_balanced_crop_probability > 1.0:
        raise ValueError("class_balanced_crop_probability must be between 0 and 1")
    if config.downsample_factor != 4:
        raise ValueError("downsample_factor must be 4")
    if config.image_size % config.downsample_factor != 0:
        raise ValueError("image_size must be divisible by downsample_factor")
    if config.eval_size % config.downsample_factor != 0:
        raise ValueError("eval_size must be divisible by downsample_factor")
    if config.hidden_channels % 8 != 0:
        raise ValueError("hidden_channels must be divisible by 8")
    if config.device not in {"auto", "cpu", "mps", "cuda"} and not config.device.startswith(
        "cuda:"
    ):
        raise ValueError("device must be one of: auto, cpu, mps, cuda, cuda:N")
    for name in ("manifest_path", "checkpoint_path", "metrics_path", "grid_path"):
        if not getattr(config, name):
            raise ValueError(f"{name} is required")


def _training_device(config: VAETrainingConfig) -> torch.device:
    if config.device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if (
        config.device == "cuda" or config.device.startswith("cuda:")
    ) and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device is unavailable: {config.device}")
    if config.device == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("Requested device is unavailable: mps")
    return torch.device(config.device)


def _load_prepared_image(sample: SatelliteSample) -> torch.Tensor:
    prepared = load_prepared_sample(sample.prepared_path, torch=torch)
    image = prepared.image
    return _prepared_image_to_float(image, sample=sample)


def _load_prepared_training_tensor(sample: SatelliteSample) -> PreparedTrainingTensor:
    prepared = load_prepared_sample(sample.prepared_path, torch=torch)
    return PreparedTrainingTensor(
        image=_prepared_image_to_float(prepared.image, sample=sample),
        labels=prepared.mask.long(),
    )


def _useful_class_ids(*, include_unlabeled: bool) -> tuple[int, ...]:
    start_class = 0 if include_unlabeled else 1
    return tuple(range(start_class, len(OPENEARTHMAP_CLASSES)))


def _prepared_image_to_float(image: torch.Tensor, *, sample: SatelliteSample) -> torch.Tensor:
    if image.ndim != 3:
        raise RuntimeError(f"Prepared image must have shape HWC or CHW: {sample.prepared_path}")
    if image.shape[-1] == 3:
        image = image.permute(2, 0, 1)
    elif image.shape[0] != 3:
        raise RuntimeError(f"Prepared image must have 3 RGB channels: {sample.prepared_path}")
    return image.float() / 127.5 - 1.0


def _load_prepared_eval_crop(
    sample: SatelliteSample,
    size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prepared = load_prepared_sample(sample.prepared_path, torch=torch)
    image = _prepared_image_to_float(prepared.image, sample=sample)
    labels = prepared.mask.long()
    return _center_crop_pair(image, labels, size)


def _random_crop(image: torch.Tensor, size: int, rng: random.Random) -> torch.Tensor:
    _, height, width = image.shape
    if height < size or width < size:
        return F.interpolate(
            image.unsqueeze(0),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    top = rng.randint(0, height - size)
    left = rng.randint(0, width - size)
    return image[:, top : top + size, left : left + size]


def _random_training_crop(
    prepared: PreparedTrainingTensor,
    size: int,
    rng: random.Random,
    *,
    config: VAETrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    image = prepared.image
    labels = prepared.labels
    _, height, width = image.shape
    if height < size or width < size:
        return _resize_training_pair(image, labels, size)
    if rng.random() < config.class_balanced_crop_probability:
        crop_window = _class_balanced_crop_window(
            labels,
            size=size,
            rng=rng,
            include_unlabeled=config.include_unlabeled_in_class_crop,
        )
        if crop_window is not None:
            top, left = crop_window
            return (
                image[:, top : top + size, left : left + size],
                labels[top : top + size, left : left + size].contiguous(),
            )
    top = rng.randint(0, height - size)
    left = rng.randint(0, width - size)
    return (
        image[:, top : top + size, left : left + size],
        labels[top : top + size, left : left + size].contiguous(),
    )


def _class_balanced_crop_window(
    labels: torch.Tensor,
    *,
    size: int,
    rng: random.Random,
    include_unlabeled: bool,
) -> tuple[int, int] | None:
    counts = torch.bincount(
        labels.reshape(-1),
        minlength=len(OPENEARTHMAP_CLASSES),
    )[: len(OPENEARTHMAP_CLASSES)]
    present_classes = [
        class_id
        for class_id in _useful_class_ids(include_unlabeled=include_unlabeled)
        if int(counts[class_id].item()) > 0
    ]
    if not present_classes:
        return None
    class_id = rng.choice(present_classes)
    pixels = torch.nonzero(labels == class_id, as_tuple=False)
    if pixels.numel() == 0:
        return None
    y, x = pixels[rng.randrange(pixels.shape[0])].tolist()
    height, width = labels.shape
    top_min = max(0, int(y) - size + 1)
    top_max = min(int(y), height - size)
    left_min = max(0, int(x) - size + 1)
    left_max = min(int(x), width - size)
    if top_min > top_max or left_min > left_max:
        return None
    return rng.randint(top_min, top_max), rng.randint(left_min, left_max)


def _center_crop(image: torch.Tensor, size: int) -> torch.Tensor:
    _, height, width = image.shape
    if height < size or width < size:
        return F.interpolate(
            image.unsqueeze(0),
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    top = (height - size) // 2
    left = (width - size) // 2
    return image[:, top : top + size, left : left + size]


def _center_crop_pair(
    image: torch.Tensor,
    labels: torch.Tensor,
    size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    _, height, width = image.shape
    if height < size or width < size:
        return _resize_training_pair(image, labels, size)
    top = (height - size) // 2
    left = (width - size) // 2
    return (
        image[:, top : top + size, left : left + size],
        labels[top : top + size, left : left + size].contiguous(),
    )


def _resize_training_pair(
    image: torch.Tensor,
    labels: torch.Tensor,
    size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    resized_image = F.interpolate(
        image.unsqueeze(0),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)
    resized_labels = (
        F.interpolate(
            labels.float().view(1, 1, labels.shape[0], labels.shape[1]),
            size=(size, size),
            mode="nearest",
        )
        .view(size, size)
        .long()
    )
    return resized_image, resized_labels


def _vae_losses(
    real: torch.Tensor,
    recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    config: VAETrainingConfig,
    *,
    labels: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    l1 = F.l1_loss(recon, real)
    mse = F.mse_loss(recon, real)
    recon_unit = _to_unit(recon)
    real_unit = _to_unit(real)
    reconstruction_exponential = _reconstruction_exponential_loss(
        real_unit,
        recon_unit,
        scale=config.reconstruction_exponential_scale,
    )
    edge = _edge_reconstruction_loss(real_unit, recon_unit)
    class_balanced_l1 = _optional_class_balanced_loss(
        real_unit,
        recon_unit,
        labels=labels,
        config=config,
        metric="l1",
    )
    class_balanced_edge = _optional_class_balanced_loss(
        real_unit,
        recon_unit,
        labels=labels,
        config=config,
        metric="edge",
    )
    kl = -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp())
    loss = (
        l1
        + config.mse_weight * mse
        + config.reconstruction_exponential_weight * reconstruction_exponential
        + config.edge_weight * edge
        + config.class_balanced_l1_weight * class_balanced_l1
        + config.class_balanced_edge_weight * class_balanced_edge
        + config.kl_weight * kl
    )
    return {
        "loss": loss,
        "l1": l1,
        "mse": mse,
        "reconstruction_exponential": reconstruction_exponential,
        "edge": edge,
        "class_balanced_l1": class_balanced_l1,
        "class_balanced_edge": class_balanced_edge,
        "kl": kl,
    }


def _reconstruction_exponential_loss(
    real_unit: torch.Tensor,
    recon_unit: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    per_sample_mae = (real_unit - recon_unit).abs().mean(dim=(1, 2, 3))
    return torch.expm1(per_sample_mae * scale).mean() / scale


def _edge_reconstruction_loss(real_unit: torch.Tensor, recon_unit: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(edge_map(real_unit, torch=torch), edge_map(recon_unit, torch=torch))


def _optional_class_balanced_loss(
    real_unit: torch.Tensor,
    recon_unit: torch.Tensor,
    *,
    labels: torch.Tensor | None,
    config: VAETrainingConfig,
    metric: str,
) -> torch.Tensor:
    if metric == "l1":
        enabled = config.class_balanced_l1_weight > 0.0
        per_pixel_error = (real_unit - recon_unit).abs().mean(dim=1)
    elif metric == "edge":
        enabled = config.class_balanced_edge_weight > 0.0
        per_pixel_error = (
            (edge_map(real_unit, torch=torch) - edge_map(recon_unit, torch=torch)).abs().mean(dim=1)
        )
    else:
        raise ValueError(f"Unknown class-balanced loss metric: {metric}")

    if not enabled:
        return real_unit.new_tensor(0.0)
    if labels is None:
        raise ValueError("Class-balanced VAE losses require labels")
    return _class_balanced_pixel_loss(
        per_pixel_error,
        labels,
        include_unlabeled=config.include_unlabeled_in_class_loss,
    )


def _class_balanced_pixel_loss(
    per_pixel_error: torch.Tensor,
    labels: torch.Tensor,
    *,
    include_unlabeled: bool,
) -> torch.Tensor:
    if labels.ndim != 3:
        raise ValueError(f"Expected class labels with shape (N, H, W), got {labels.ndim}D")
    if labels.shape[0] != per_pixel_error.shape[0]:
        raise ValueError(
            "Class labels must have the same batch size as per-pixel errors: "
            f"{tuple(labels.shape)} != {tuple(per_pixel_error.shape)}"
        )
    if tuple(labels.shape[-2:]) != tuple(per_pixel_error.shape[-2:]):
        labels = (
            F.interpolate(
                labels.float().unsqueeze(1),
                size=per_pixel_error.shape[-2:],
                mode="nearest",
            )
            .squeeze(1)
            .long()
        )

    start_class = 0 if include_unlabeled else 1
    class_losses = []
    for class_id in range(start_class, len(OPENEARTHMAP_CLASSES)):
        mask = labels == class_id
        if bool(mask.any().item()):
            class_losses.append(per_pixel_error[mask].mean())
    if not class_losses:
        return per_pixel_error.mean()
    return torch.stack(class_losses).mean()


def _custom_reconstruct(
    model: SatelliteVAE,
    images: torch.Tensor,
    *,
    device: torch.device,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, images.shape[0], 4):
            outputs.append(model.reconstruct(images[start : start + 4].to(device)).detach().cpu())
    return torch.cat(outputs, dim=0)


def _latent_stats(
    model: SatelliteVAE,
    samples: Sequence[SatelliteSample],
    *,
    config: VAETrainingConfig,
    device: torch.device,
) -> dict[str, Any]:
    latent_sum = torch.zeros(config.latent_channels, dtype=torch.float64)
    latent_square_sum = torch.zeros(config.latent_channels, dtype=torch.float64)
    logvar_sum = torch.zeros(config.latent_channels, dtype=torch.float64)
    element_count = 0

    with torch.inference_mode():
        for start in range(0, len(samples), config.batch_size):
            batch_samples = samples[start : start + config.batch_size]
            batch = torch.stack(
                [
                    _center_crop(_load_prepared_image(sample), config.eval_size)
                    for sample in batch_samples
                ]
            ).to(device)
            mu, logvar = model.encode(batch)
            latent_sum += mu.sum(dim=(0, 2, 3)).detach().double().cpu()
            latent_square_sum += mu.square().sum(dim=(0, 2, 3)).detach().double().cpu()
            logvar_sum += logvar.sum(dim=(0, 2, 3)).detach().double().cpu()
            element_count += int(mu.shape[0] * mu.shape[2] * mu.shape[3])

    mean = latent_sum / element_count
    variance = (latent_square_sum / element_count - mean.square()).clamp_min(0.0)
    std = variance.sqrt()
    logvar_mean = logvar_sum / element_count
    return {
        "source": "train_mu",
        "sample_count": len(samples),
        "element_count": element_count,
        "mean": [float(value) for value in mean.tolist()],
        "std": [float(value) for value in std.tolist()],
        "global_mean": float(mean.mean().item()),
        "global_std": float(std.mean().item()),
        "logvar_mean": [float(value) for value in logvar_mean.tolist()],
        "global_logvar_mean": float(logvar_mean.mean().item()),
    }


def _write_checkpoint(
    checkpoint_path: Path,
    *,
    model: nn.Module,
    config: VAETrainingConfig,
    train_samples: Sequence[SatelliteSample],
    validation_samples: Sequence[SatelliteSample],
    history: Sequence[Mapping[str, float]],
    latent_stats: Mapping[str, Any],
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "vae_config": asdict(config.vae_config()),
            "training_config": asdict(config),
            "train_samples": [str(sample.prepared_path) for sample in train_samples],
            "validation_samples": [str(sample.prepared_path) for sample in validation_samples],
            "history": list(history),
            "latent_stats": dict(latent_stats),
        },
        checkpoint_path,
    )


def _write_metrics(
    metrics_path: Path,
    *,
    config: VAETrainingConfig,
    device: torch.device,
    train_samples: Sequence[SatelliteSample],
    validation_samples: Sequence[SatelliteSample],
    custom_metrics: Mapping[str, float],
    custom_metrics_by_class: Mapping[str, Any],
    latent_stats: Mapping[str, Any],
    checkpoint_path: Path,
    grid_path: Path,
    history: Sequence[Mapping[str, float]],
) -> None:
    payload = {
        "config": asdict(config),
        "device": str(device),
        "train_image_count": len(train_samples),
        "validation_image_count": len(validation_samples),
        "custom_vae": dict(custom_metrics),
        "custom_vae_by_class": dict(custom_metrics_by_class),
        "latent_stats": dict(latent_stats),
        "checkpoint_path": str(checkpoint_path),
        "metrics_path": str(metrics_path),
        "grid_path": str(grid_path),
        "history": list(history),
        "train_samples": [str(sample.prepared_path) for sample in train_samples],
        "validation_samples": [str(sample.prepared_path) for sample in validation_samples],
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_grid(
    output_path: Path,
    *,
    real: torch.Tensor,
    custom: torch.Tensor,
) -> None:
    labels = ["Original", "Custom VAE", "Custom error x4"]
    rows = real.shape[0]
    cell = int(real.shape[-1])
    label_height = 24
    canvas = Image.new("RGB", (cell * len(labels), label_height + rows * cell), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for col, label in enumerate(labels):
        draw.text((col * cell + 5, 5), label, fill=(0, 0, 0), font=font)

    for row in range(rows):
        panels = [
            _tensor_to_pil(real[row]),
            _tensor_to_pil(custom[row]),
            _error_to_pil(real[row], custom[row]),
        ]
        for col, panel in enumerate(panels):
            canvas.paste(panel, (col * cell, label_height + row * cell))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _to_unit(image: torch.Tensor) -> torch.Tensor:
    return ((image.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = (
        (_to_unit(image.detach().cpu()).permute(1, 2, 0).numpy() * 255.0)
        .round()
        .clip(0, 255)
        .astype("uint8")
    )
    return Image.fromarray(array)


def _error_to_pil(real: torch.Tensor, reconstructed: torch.Tensor) -> Image.Image:
    error = (_to_unit(real) - _to_unit(reconstructed)).abs().mul(4.0).clamp(0.0, 1.0)
    array = (
        (error.detach().cpu().permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype("uint8")
    )
    return Image.fromarray(array)


def _sample_validation_group(sample: SatelliteSample) -> str:
    parts = sample.image_path.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "images":
            tail = parts[index + 1 :]
            if len(tail) > 1:
                return tail[0]
            if tail:
                return _stem_group(Path(tail[0]).stem)
    parent = sample.image_path.parent.name
    if parent and parent != "images":
        return parent
    return _stem_group(sample.image_path.stem)


def _stem_group(stem: str) -> str:
    for separator in ("_", "-"):
        if separator in stem:
            return stem.split(separator, 1)[0]
    return stem


@click.command("train-vae", help="Train the native satellite VAE from a JSON config file.")
@click.option(
    "--target",
    type=click.Choice(["local", "modal"]),
    default="local",
    show_default=True,
    help="Execution target.",
)
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
def cli(target: str, config_path: Path) -> None:
    try:
        config = load_vae_training_config_file(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise click.ClickException(str(error)) from error
    if target == "modal":
        from sim_satellite.modal_app import run_vae_training_target

        result = run_vae_training_target(config)
    else:
        result = asdict(run_vae_training(config))
    click.echo(json.dumps(result, indent=2))
