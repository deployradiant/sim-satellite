"""Training loop for VAE-fused semantic segmentation classifiers."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from sim_satellite.data.loader import SatelliteSample
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES
from sim_satellite.segmentation.checkpoint import _save_checkpoint
from sim_satellite.segmentation.config import (
    SegmentationTrainingConfig,
    SegmentationTrainingResult,
    _validate_training_config,
    segmentation_model_uses_vae_features,
)
from sim_satellite.segmentation.data import (
    _class_balanced_crop_box,
    _load_segmentation_batch,
    _PreparedSampleCache,
    _random_square_crop_box,
    _split_training_samples,
)
from sim_satellite.segmentation.inference import _evaluate_samples
from sim_satellite.segmentation.metrics import (
    _class_weights,
    _focal_cross_entropy_loss,
    _lovasz_softmax_loss,
    _multiclass_dice_loss,
)
from sim_satellite.segmentation.models import (
    _forward_segmentation_model,
    _segmentation_input_channels,
    build_segmentation_model,
)
from sim_satellite.segmentation.utils import ProgressReporter, _report
from sim_satellite.segmentation.vae import _encode_vae_features, _load_vae_feature_encoder


def train_segmentation_model(
    config: SegmentationTrainingConfig,
    *,
    progress: ProgressReporter | None = None,
    after_save: Callable[[], None] | None = None,
) -> SegmentationTrainingResult:
    _validate_training_config(config)
    _seed_everything(config.seed)

    device = torch.device(config.device)
    train_samples, validation_samples = _split_training_samples(config)
    cache = _PreparedSampleCache(max_samples=config.cache_samples)
    checkpoint_config = asdict(config)
    vae_encoder = None
    vae_latent_channels = 0
    if segmentation_model_uses_vae_features(config.model_name):
        vae_encoder = _load_vae_feature_encoder(checkpoint_config, device=device)
        vae_latent_channels = vae_encoder.latent_channels
        checkpoint_config["vae_latent_channels"] = vae_encoder.latent_channels
    model = build_segmentation_model(
        num_classes=len(OPENEARTHMAP_CLASSES),
        input_channels=_segmentation_input_channels(config),
        vae_latent_channels=vae_latent_channels,
        model_name=config.model_name,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    _report(progress, f"Scanning class balance across {len(train_samples)} samples")
    class_weights, class_sample_index = _scan_class_balance_and_crop_index(
        train_samples,
        cache=cache,
        power=config.class_weight_power,
        unlabeled_weight=config.unlabeled_weight,
        min_crop_pixels=config.class_balanced_crop_min_pixels,
        build_crop_index=config.class_balanced_crop_probability > 0.0,
        progress=progress,
    )
    class_weights = _apply_class_weight_multipliers(
        class_weights,
        config.class_weight_multipliers,
    ).to(device)

    class_crop_sampling_weights = _class_name_weight_map(config.class_balanced_crop_weights)
    rng = random.Random(config.seed)
    losses: list[float] = []
    best_metrics: dict[str, Any] | None = None
    best_step = 0
    checkpoint_path = Path(config.checkpoint_path)
    metrics_path = checkpoint_path.with_suffix(".json")

    _report(
        progress,
        f"Training {config.model_name} segmenter on {len(train_samples)} samples; "
        f"validating on {len(validation_samples)} samples",
    )
    model.train()
    for step in range(1, config.steps + 1):
        _set_optimizer_learning_rate(
            optimizer,
            _segmentation_learning_rate(config, step=step),
        )
        batch_samples, crop_boxes = _sample_training_batch(
            train_samples,
            cache=cache,
            config=config,
            rng=rng,
            class_sample_index=class_sample_index,
            class_sampling_weights=class_crop_sampling_weights,
        )
        images, labels = _load_segmentation_batch(
            batch_samples,
            crop_boxes,
            cache=cache,
            image_size=config.image_size,
            context_size=config.context_size,
            device=device,
            augmentation_config=config,
            rng=rng,
        )
        optimizer.zero_grad(set_to_none=True)
        logits = _forward_segmentation_model(
            model,
            images,
            vae_latents=_encode_vae_features(vae_encoder, images) if vae_encoder else None,
        )
        loss = _segmentation_training_loss(
            logits,
            labels,
            class_weights=class_weights,
            dice_weight=config.dice_weight,
            focal_weight=config.focal_weight,
            focal_gamma=config.focal_gamma,
            lovasz_weight=config.lovasz_weight,
            lovasz_max_pixels=config.lovasz_max_pixels,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

        should_eval = step == config.steps or step % config.eval_every == 0
        should_report = step == 1 or should_eval or step % _segmentation_log_every(config) == 0
        if should_report:
            _report(progress, f"step={step} loss={losses[-1]:.4f}")
        if should_eval:
            metrics = _evaluate_samples(
                model,
                validation_samples,
                image_size=config.image_size,
                context_size=config.context_size,
                tile_stride=config.evaluation_stride,
                device=device,
                cache=cache,
                vae_encoder=vae_encoder,
                feather_blending=config.evaluation_feather_blending,
            )
            _report(
                progress,
                f"step={step} val_mean_iou={metrics['mean_iou']:.4f} "
                f"val_pixel_accuracy={metrics['pixel_accuracy']:.4f}",
            )
            if best_metrics is None or metrics["mean_iou"] > best_metrics["mean_iou"]:
                best_metrics = metrics
                best_step = step
                _save_checkpoint(
                    checkpoint_path,
                    model,
                    config=checkpoint_config,
                    metrics=metrics,
                    step=step,
                )
                if after_save is not None:
                    after_save()
            model.train()

    final_metrics = _evaluate_samples(
        model,
        validation_samples,
        image_size=config.image_size,
        context_size=config.context_size,
        tile_stride=config.evaluation_stride,
        device=device,
        cache=cache,
        vae_encoder=vae_encoder,
        feather_blending=config.evaluation_feather_blending,
    )
    if best_metrics is None:
        best_metrics = final_metrics
        best_step = config.steps
        _save_checkpoint(
            checkpoint_path,
            model,
            config=checkpoint_config,
            metrics=best_metrics,
            step=config.steps,
        )
        if after_save is not None:
            after_save()

    recent_losses = losses[-min(len(losses), 100) :]
    result = SegmentationTrainingResult(
        checkpoint_path=str(checkpoint_path),
        metrics_path=str(metrics_path),
        train_samples=len(train_samples),
        validation_samples=len(validation_samples),
        steps=config.steps,
        parameters=_count_parameters(model),
        final_train_loss=sum(recent_losses) / len(recent_losses),
        best_step=best_step,
        best_mean_iou=float(best_metrics["mean_iou"]),
        best_pixel_accuracy=float(best_metrics["pixel_accuracy"]),
        best_class_iou=best_metrics["class_iou"],
        final_mean_iou=float(final_metrics["mean_iou"]),
        final_pixel_accuracy=float(final_metrics["pixel_accuracy"]),
        class_iou=final_metrics["class_iou"],
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "config": checkpoint_config,
                "result": asdict(result),
                "class_weights": {
                    class_name: float(class_weights[index].detach().cpu().item())
                    for index, class_name in enumerate(OPENEARTHMAP_CLASSES)
                },
                "class_balanced_crop_samples": {
                    OPENEARTHMAP_CLASSES[class_id]: len(samples)
                    for class_id, samples in class_sample_index.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def _segmentation_training_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_weights: torch.Tensor,
    dice_weight: float,
    focal_weight: float,
    focal_gamma: float,
    lovasz_weight: float,
    lovasz_max_pixels: int,
) -> torch.Tensor:
    ce_loss = F.cross_entropy(logits, labels, weight=class_weights)
    dice_loss = _multiclass_dice_loss(
        logits,
        labels,
        include_unlabeled=False,
    )
    loss = ce_loss + dice_weight * dice_loss
    if focal_weight > 0.0:
        loss = loss + focal_weight * _focal_cross_entropy_loss(
            logits,
            labels,
            class_weights=class_weights,
            gamma=focal_gamma,
        )
    if lovasz_weight > 0.0:
        loss = loss + lovasz_weight * _lovasz_softmax_loss(
            logits,
            labels,
            include_unlabeled=False,
            max_pixels=lovasz_max_pixels,
        )
    return loss


def _count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _segmentation_learning_rate(config: SegmentationTrainingConfig, *, step: int) -> float:
    if config.lr_schedule == "constant" or config.steps <= 1:
        return config.learning_rate
    progress = min(1.0, max(0.0, (step - 1) / (config.steps - 1)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine


def _set_optimizer_learning_rate(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = value


def _apply_class_weight_multipliers(
    weights: torch.Tensor,
    multipliers: dict[str, float],
) -> torch.Tensor:
    if not multipliers:
        return weights
    adjusted = weights.clone()
    for class_name, multiplier in multipliers.items():
        adjusted[OPENEARTHMAP_CLASSES.index(class_name)] *= float(multiplier)
    nonzero = adjusted > 0
    if bool(nonzero.any().item()):
        adjusted[nonzero] = adjusted[nonzero] / adjusted[nonzero].mean()
    return adjusted


def _scan_class_balance_and_crop_index(
    samples: Sequence[SatelliteSample],
    *,
    cache: _PreparedSampleCache,
    power: float,
    unlabeled_weight: float,
    min_crop_pixels: int,
    build_crop_index: bool,
    progress: ProgressReporter | None,
) -> tuple[torch.Tensor, dict[int, tuple[SatelliteSample, ...]]]:
    if not build_crop_index:
        return (
            _class_weights(
                samples,
                cache=cache,
                power=power,
                unlabeled_weight=unlabeled_weight,
                progress=progress,
            ),
            {},
        )

    counts = torch.zeros(len(OPENEARTHMAP_CLASSES), dtype=torch.long)
    indexed: dict[int, list[SatelliteSample]] = {}
    if build_crop_index:
        indexed = {class_id: [] for class_id in range(1, len(OPENEARTHMAP_CLASSES))}

    for index, sample in enumerate(samples, start=1):
        labels = cache.get(sample).mask.long()
        sample_counts = torch.bincount(
            labels.reshape(-1),
            minlength=len(OPENEARTHMAP_CLASSES),
        )[: len(OPENEARTHMAP_CLASSES)]
        counts += sample_counts
        for class_id in indexed:
            if int(sample_counts[class_id].item()) >= min_crop_pixels:
                indexed[class_id].append(sample)
        if index == len(samples) or index % max(1, len(samples) // 10) == 0:
            _report(progress, f"Scanned class balance {index}/{len(samples)} samples")

    return (
        _class_weights_from_counts(
            counts,
            power=power,
            unlabeled_weight=unlabeled_weight,
        ),
        {class_id: tuple(class_samples) for class_id, class_samples in indexed.items()},
    )


def _class_weights_from_counts(
    counts: torch.Tensor,
    *,
    power: float,
    unlabeled_weight: float,
) -> torch.Tensor:
    counts = counts.float()
    weights = torch.zeros_like(counts)
    present = counts > 0
    if not bool(present.any().item()):
        return weights
    median = counts[present].median()
    weights[present] = (median / counts[present]).pow(power)
    weights[0] *= unlabeled_weight
    nonzero = weights > 0
    weights[nonzero] = weights[nonzero] / weights[nonzero].mean()
    return weights


def _sample_training_batch(
    train_samples: Sequence[SatelliteSample],
    *,
    cache: _PreparedSampleCache,
    config: SegmentationTrainingConfig,
    rng: random.Random,
    class_sample_index: Mapping[int, Sequence[SatelliteSample]],
    class_sampling_weights: Mapping[int, float],
) -> tuple[tuple[SatelliteSample, ...], list[tuple[int, int, int, int]]]:
    batch_samples: list[SatelliteSample] = []
    crop_boxes: list[tuple[int, int, int, int]] = []
    for _ in range(config.batch_size):
        selected = None
        crop_size = _segmentation_training_crop_size(config, rng)
        if (
            class_sample_index
            and config.class_balanced_crop_probability > 0.0
            and rng.random() < config.class_balanced_crop_probability
        ):
            selected = _class_balanced_training_crop(
                class_sample_index,
                cache=cache,
                crop_size=crop_size,
                rng=rng,
                class_sampling_weights=class_sampling_weights,
                min_target_pixels=config.class_balanced_crop_min_pixels,
            )
        if selected is None:
            sample = train_samples[rng.randrange(len(train_samples))]
            selected = (
                sample,
                _random_square_crop_box(cache.get(sample), crop_size, rng),
            )
        sample, crop_box = selected
        batch_samples.append(sample)
        crop_boxes.append(crop_box)
    return tuple(batch_samples), crop_boxes


def _class_balanced_training_crop(
    class_sample_index: Mapping[int, Sequence[SatelliteSample]],
    *,
    cache: _PreparedSampleCache,
    crop_size: int,
    rng: random.Random,
    class_sampling_weights: Mapping[int, float],
    min_target_pixels: int,
) -> tuple[SatelliteSample, tuple[int, int, int, int]] | None:
    class_ids = tuple(class_id for class_id, samples in class_sample_index.items() if samples)
    if not class_ids:
        return None
    class_id = _weighted_choice(
        class_ids,
        weights={class_id: class_sampling_weights.get(class_id, 1.0) for class_id in class_ids},
        rng=rng,
    )
    samples = class_sample_index[class_id]
    for _ in range(min(8, max(1, len(samples)))):
        sample = samples[rng.randrange(len(samples))]
        crop_box = _class_balanced_crop_box(
            cache.get(sample),
            crop_size,
            rng,
            class_id=class_id,
            min_target_pixels=min_target_pixels,
        )
        if crop_box is not None:
            return sample, crop_box
    return None


def _class_name_weight_map(weights: Mapping[str, float]) -> dict[int, float]:
    return {
        OPENEARTHMAP_CLASSES.index(class_name): float(weight)
        for class_name, weight in weights.items()
    }


def _weighted_choice(
    values: Sequence[int],
    *,
    weights: Mapping[int, float],
    rng: random.Random,
) -> int:
    total = sum(max(0.0, float(weights.get(value, 1.0))) for value in values)
    if total <= 0.0:
        return values[rng.randrange(len(values))]
    target = rng.random() * total
    cumulative = 0.0
    for value in values:
        cumulative += max(0.0, float(weights.get(value, 1.0)))
        if cumulative >= target:
            return value
    return values[-1]


def _segmentation_training_crop_size(
    config: SegmentationTrainingConfig,
    rng: random.Random,
) -> int:
    if config.crop_scale_min == 1.0 and config.crop_scale_max == 1.0:
        return config.image_size
    scale = rng.uniform(config.crop_scale_min, config.crop_scale_max)
    return max(1, int(round(config.image_size * scale)))


def _segmentation_log_every(config: SegmentationTrainingConfig) -> int:
    return config.log_every or max(1, config.steps // 10)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda"):
        torch.cuda.manual_seed_all(seed)
