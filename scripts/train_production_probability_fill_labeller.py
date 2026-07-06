"""Train a production-probability fill labeller for OpenEarthMap.

The first step is the frozen production OpenEarth segmenter. It labels only
pixels above a confidence threshold. The second step is a small residual fill
model trained only on the labelled pixels the production pass leaves unlabelled.
The fill model sees the RGB image, the full production probability map, the
high-confidence labels, and confidence/known masks. It predicts a residual over
the production log-probabilities so dense production remains the fallback.

Run on Modal:

    uv run modal run scripts/train_production_probability_fill_labeller.py

Useful smoke-run overrides:

    SIM_SATELLITE_PRODUCTION_PROBABILITY_FILL_STEPS=20 \
    SIM_SATELLITE_PRODUCTION_PROBABILITY_FILL_MAX_TRAIN_SAMPLES=128 \
    SIM_SATELLITE_PRODUCTION_PROBABILITY_FILL_MAX_VALIDATION_SAMPLES=16 \
    uv run modal run scripts/train_production_probability_fill_labeller.py
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from sim_satellite.data.loader import SatelliteSample
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES
from sim_satellite.modal_app import (
    MODAL_EPHEMERAL_DISK_MIB,
    MODAL_TRAINING_GPU,
    MODAL_TRAINING_TIMEOUT,
    _ensure_modal_training_data,
    _modal_artifact_path,
    _modal_data_path,
    app,
    artifact_volume,
    image,
    modal_volumes,
)
from sim_satellite.segmentation.checkpoint import (
    SegmentationCheckpoint,
    _load_segmentation_checkpoint,
)
from sim_satellite.segmentation.cli import (
    PRODUCTION_OPENEARTH_CHECKPOINT_PATH,
    PRODUCTION_OPENEARTH_CLASS_LOGIT_BIASES,
    PRODUCTION_OPENEARTH_TILE_SCALES,
    PRODUCTION_OPENEARTH_TILE_STRIDE,
)
from sim_satellite.segmentation.config import SegmentationTrainingConfig
from sim_satellite.segmentation.data import (
    _load_segmentation_batch,
    _PreparedSampleCache,
    _samples_for_split,
    _tile_windows,
)
from sim_satellite.segmentation.inference import (
    _accumulate_resized_tile_logits,
    _apply_class_logit_biases,
    _predict_prepared_sample_logits,
    _resized_tile_logits,
    _scaled_tile_windows_config,
)
from sim_satellite.segmentation.metrics import _confusion_matrix, _segmentation_metrics
from sim_satellite.segmentation.models import _forward_segmentation_model
from sim_satellite.segmentation.training import (
    _apply_class_weight_multipliers,
    _class_name_weight_map,
    _sample_training_batch,
    _scan_class_balance_and_crop_index,
    _segmentation_learning_rate,
    _set_optimizer_learning_rate,
)
from sim_satellite.segmentation.utils import ProgressReporter, _report, _should_report

DEFAULT_CHECKPOINT_PATH = (
    "artifacts/segmentation-production-probability-fill-v1/"
    "production-probability-fill-s2000-t080.pt"
)
MODEL_NAME = "production-probability-fill-v1"


@dataclass(frozen=True)
class ProductionProbabilityFillConfig:
    manifest_path: str = "data/openearthmap/manifest.jsonl"
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH
    metrics_path: str = ""
    production_checkpoint_path: str = PRODUCTION_OPENEARTH_CHECKPOINT_PATH
    split: str = "train"
    validation_split: str = "val"
    max_train_samples: int = 0
    max_validation_samples: int = 128
    production_confidence_threshold: float = 0.8
    production_class_logit_biases: dict[str, float] = field(
        default_factory=lambda: dict(PRODUCTION_OPENEARTH_CLASS_LOGIT_BIASES)
    )
    production_tile_stride: int = PRODUCTION_OPENEARTH_TILE_STRIDE
    production_tile_scales: tuple[float, ...] = PRODUCTION_OPENEARTH_TILE_SCALES
    production_feather_blending: bool = True
    steps: int = 2000
    batch_size: int = 2
    image_size: int = 512
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    lr_schedule: str = "cosine"
    weight_decay: float = 1e-4
    class_weight_power: float = 0.35
    class_weight_multipliers: dict[str, float] = field(
        default_factory=lambda: {
            "building": 1.25,
            "developed space": 1.8,
            "rangeland": 1.5,
            "road": 1.55,
            "tree": 1.35,
        }
    )
    eval_every: int = 250
    evaluation_stride: int = 256
    evaluation_tile_scales: tuple[float, ...] = ()
    evaluation_feather_blending: bool = True
    log_every: int = 25
    device: str = "cpu"
    seed: int = 20260705
    cache_samples: int = 512
    augment: bool = True
    flip_probability: float = 0.5
    rotate_probability: float = 0.45
    color_jitter_strength: float = 0.03
    blur_probability: float = 0.02
    crop_scale_min: float = 0.85
    crop_scale_max: float = 1.15
    class_balanced_crop_probability: float = 0.75
    class_balanced_crop_weights: dict[str, float] = field(
        default_factory=lambda: {
            "building": 1.25,
            "developed space": 1.8,
            "rangeland": 1.5,
            "road": 1.55,
            "tree": 1.35,
        }
    )
    class_balanced_crop_min_pixels: int = 64

    @classmethod
    def from_dict(
        cls,
        values: Mapping[str, Any],
    ) -> ProductionProbabilityFillConfig:
        names = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - names)
        if unknown:
            raise ValueError(
                f"Unknown production-probability fill config fields: {', '.join(unknown)}"
            )
        return cls(**{name: values[name] for name in names if name in values})


@dataclass(frozen=True)
class ProductionProbabilityFillResult:
    checkpoint_path: str
    metrics_path: str
    train_samples: int
    validation_samples: int
    steps: int
    parameters: int
    final_train_loss: float
    skipped_batches: int
    best_step: int
    best_mean_iou: float
    best_labeled_pixel_accuracy: float
    final_mean_iou: float
    final_labeled_pixel_accuracy: float
    production_dense_mean_iou: float
    production_dense_labeled_pixel_accuracy: float
    production_selected_mean_iou: float
    production_selected_labeled_pixel_accuracy: float
    production_selected_labeled_pixel_coverage: float
    fill_region_mean_iou: float
    fill_region_labeled_pixel_accuracy: float
    fill_region_labeled_pixel_coverage: float


def train_production_probability_fill(
    config: ProductionProbabilityFillConfig,
    *,
    progress: ProgressReporter | None = None,
    after_save: Callable[[], None] | None = None,
) -> ProductionProbabilityFillResult:
    _validate_config(config)
    _seed_everything(config.seed)

    device = torch.device(config.device)
    train_samples = _shuffled_limit_samples(
        _samples_for_split(config.manifest_path, config.split),
        max_samples=config.max_train_samples,
        seed=config.seed,
    )
    validation_samples = _shuffled_limit_samples(
        _samples_for_split(config.manifest_path, config.validation_split),
        max_samples=config.max_validation_samples,
        seed=config.seed + 1,
    )
    if not train_samples:
        raise RuntimeError(f"No training samples for split {config.split!r}")
    if not validation_samples:
        raise RuntimeError(f"No validation samples for split {config.validation_split!r}")

    cache = _PreparedSampleCache(max_samples=config.cache_samples)
    production = _load_segmentation_checkpoint(
        Path(config.production_checkpoint_path),
        device=config.device,
    )
    _validate_production_checkpoint(production)
    for parameter in production.model.parameters():
        parameter.requires_grad_(False)
    model = ProductionProbabilityFillModel(
        num_classes=len(OPENEARTHMAP_CLASSES),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    sampling_config = _sampling_config(config)
    _report(progress, f"Scanning class balance across {len(train_samples)} samples")
    class_weights, class_sample_index = _scan_class_balance_and_crop_index(
        train_samples,
        cache=cache,
        power=config.class_weight_power,
        unlabeled_weight=0.0,
        min_crop_pixels=config.class_balanced_crop_min_pixels,
        build_crop_index=config.class_balanced_crop_probability > 0.0,
        progress=progress,
    )
    class_weights = _apply_class_weight_multipliers(
        class_weights,
        config.class_weight_multipliers,
    ).to(device)
    class_crop_sampling_weights = _class_name_weight_map(config.class_balanced_crop_weights)

    checkpoint_config = asdict(config)
    checkpoint_path = Path(config.checkpoint_path)
    metrics_path = (
        Path(config.metrics_path)
        if config.metrics_path
        else checkpoint_path.with_suffix(".json")
    )
    rng = random.Random(config.seed)
    losses: list[float] = []
    best_metrics: dict[str, Any] | None = None
    best_step = 0
    skipped_batches = 0

    _report(
        progress,
        f"Training production-probability fill labeller on {len(train_samples)} samples; "
        f"validating on {len(validation_samples)} samples; "
        f"production threshold={config.production_confidence_threshold:.2f}",
    )
    model.train()
    step = 0
    empty_batch_streak = 0
    while step < config.steps:
        batch_samples, crop_boxes = _sample_training_batch(
            train_samples,
            cache=cache,
            config=sampling_config,
            rng=rng,
            class_sample_index=class_sample_index,
            class_sampling_weights=class_crop_sampling_weights,
        )
        images, labels = _load_segmentation_batch(
            batch_samples,
            crop_boxes,
            cache=cache,
            image_size=config.image_size,
            context_size=0,
            device=device,
            augmentation_config=sampling_config,
            rng=rng,
        )
        with torch.no_grad():
            condition = _production_condition_for_batch(
                production,
                images,
                confidence_threshold=config.production_confidence_threshold,
                class_logit_biases=config.production_class_logit_biases,
            )

        logits = model(images, condition.features, condition.base_logits)
        loss = _unknown_region_cross_entropy(
            logits,
            labels,
            known_mask=condition.known_mask,
            class_weights=class_weights,
        )
        if loss is None:
            skipped_batches += 1
            empty_batch_streak += 1
            if empty_batch_streak > 100:
                raise RuntimeError(
                    "No unknown labelled pixels found in 100 consecutive batches. "
                    "Raise the production confidence threshold or lower batch/crop filtering."
                )
            continue
        empty_batch_streak = 0
        step += 1
        _set_optimizer_learning_rate(
            optimizer,
            _fill_learning_rate(config, step=step),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

        should_eval = step == config.steps or step % config.eval_every == 0
        should_report = step == 1 or should_eval or step % config.log_every == 0
        if should_report:
            _report(
                progress,
                f"step={step} loss={losses[-1]:.4f} skipped_batches={skipped_batches}",
            )
        if should_eval:
            metrics = evaluate_production_probability_fill(
                model,
                production=production,
                samples=validation_samples,
                config=config,
                cache=cache,
                progress=progress,
            )
            _report(
                progress,
                f"step={step} two_step_mIoU={metrics['two_step_dense']['mean_iou']:.4f} "
                f"fill_mIoU={metrics['fill_region']['mean_iou']:.4f}",
            )
            if (
                best_metrics is None
                or metrics["two_step_dense"]["mean_iou"]
                > best_metrics["two_step_dense"]["mean_iou"]
            ):
                best_metrics = metrics
                best_step = step
                _save_two_step_checkpoint(
                    checkpoint_path,
                    model,
                    config=checkpoint_config,
                    metrics=metrics,
                    step=step,
                )
                if after_save is not None:
                    after_save()
            model.train()

    final_metrics = evaluate_production_probability_fill(
        model,
        production=production,
        samples=validation_samples,
        config=config,
        cache=cache,
        progress=progress,
    )
    if best_metrics is None:
        best_metrics = final_metrics
        best_step = config.steps
        _save_two_step_checkpoint(
            checkpoint_path,
            model,
            config=checkpoint_config,
            metrics=best_metrics,
            step=config.steps,
        )
        if after_save is not None:
            after_save()

    recent_losses = losses[-min(len(losses), 100) :]
    result = ProductionProbabilityFillResult(
        checkpoint_path=str(checkpoint_path),
        metrics_path=str(metrics_path),
        train_samples=len(train_samples),
        validation_samples=len(validation_samples),
        steps=config.steps,
        parameters=_count_parameters(model),
        final_train_loss=sum(recent_losses) / len(recent_losses),
        skipped_batches=skipped_batches,
        best_step=best_step,
        best_mean_iou=float(best_metrics["two_step_dense"]["mean_iou"]),
        best_labeled_pixel_accuracy=float(
            best_metrics["two_step_dense"]["labeled_pixel_accuracy"]
        ),
        final_mean_iou=float(final_metrics["two_step_dense"]["mean_iou"]),
        final_labeled_pixel_accuracy=float(
            final_metrics["two_step_dense"]["labeled_pixel_accuracy"]
        ),
        production_dense_mean_iou=float(final_metrics["production_dense"]["mean_iou"]),
        production_dense_labeled_pixel_accuracy=float(
            final_metrics["production_dense"]["labeled_pixel_accuracy"]
        ),
        production_selected_mean_iou=float(final_metrics["production_selected"]["mean_iou"]),
        production_selected_labeled_pixel_accuracy=float(
            final_metrics["production_selected"]["labeled_pixel_accuracy"]
        ),
        production_selected_labeled_pixel_coverage=float(
            final_metrics["production_selected_labeled_pixel_coverage"]
        ),
        fill_region_mean_iou=float(final_metrics["fill_region"]["mean_iou"]),
        fill_region_labeled_pixel_accuracy=float(
            final_metrics["fill_region"]["labeled_pixel_accuracy"]
        ),
        fill_region_labeled_pixel_coverage=float(
            final_metrics["fill_region_labeled_pixel_coverage"]
        ),
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "config": checkpoint_config,
                "result": asdict(result),
                "final_metrics": final_metrics,
                "class_weights": {
                    class_name: float(class_weights[index].detach().cpu().item())
                    for index, class_name in enumerate(OPENEARTHMAP_CLASSES)
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def evaluate_production_probability_fill(
    model: nn.Module,
    *,
    production: SegmentationCheckpoint,
    samples: Sequence[SatelliteSample],
    config: ProductionProbabilityFillConfig,
    cache: _PreparedSampleCache,
    progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    device = torch.device(config.device)
    class_count = len(OPENEARTHMAP_CLASSES)
    model.eval()
    production.model.eval()
    production_dense_confusion = torch.zeros((class_count, class_count), dtype=torch.long)
    production_selected_confusion = torch.zeros_like(production_dense_confusion)
    fill_region_confusion = torch.zeros_like(production_dense_confusion)
    two_step_confusion = torch.zeros_like(production_dense_confusion)

    total_pixels = 0
    labeled_pixels = 0
    selected_pixels = 0
    selected_labeled_pixels = 0
    fill_pixels = 0
    fill_labeled_pixels = 0
    production_tiles = 0
    fill_tiles = 0
    production_confidence_sum = 0.0
    selected_confidence_sum = 0.0

    with torch.inference_mode():
        for index, sample in enumerate(samples, start=1):
            prepared = cache.get(sample)
            labels = prepared.mask.long().cpu()
            labeled_mask = labels > 0
            production_logits, sample_production_tiles = _predict_prepared_sample_logits(
                production.model,
                sample,
                image_size=int(production.config["image_size"]),
                context_size=int(production.config.get("context_size", 0)),
                tile_stride=config.production_tile_stride,
                tile_scales=config.production_tile_scales,
                device=device,
                cache=cache,
                output_class_names=production.classes,
                feather_blending=config.production_feather_blending,
            )
            production_logits = _apply_class_logit_biases(
                production_logits,
                config.production_class_logit_biases,
                class_names=production.classes,
            )
            production_probabilities = production_logits.softmax(dim=0)
            production_confidence, production_predictions = production_probabilities.max(dim=0)
            production_known_mask = production_confidence >= config.production_confidence_threshold

            fill_logits, sample_fill_tiles = _predict_fill_prepared_sample_logits(
                model,
                sample,
                production_predictions=production_predictions,
                production_probabilities=production_probabilities,
                production_confidence=production_confidence,
                production_known_mask=production_known_mask,
                image_size=config.image_size,
                tile_stride=config.evaluation_stride,
                tile_scales=config.evaluation_tile_scales,
                device=device,
                cache=cache,
                feather_blending=config.evaluation_feather_blending,
            )
            fill_predictions = fill_logits.argmax(dim=0)
            two_step_predictions = torch.where(
                production_known_mask,
                production_predictions,
                fill_predictions,
            )

            production_predictions_cpu = production_predictions.cpu()
            production_known_cpu = production_known_mask.cpu()
            fill_predictions_cpu = fill_predictions.cpu()
            two_step_predictions_cpu = two_step_predictions.cpu()
            production_confidence_cpu = production_confidence.cpu()
            fill_mask_cpu = ~production_known_cpu

            production_dense_confusion += _confusion_matrix(
                production_predictions_cpu,
                labels,
                num_classes=class_count,
            )
            production_selected_confusion += _confusion_matrix(
                production_predictions_cpu[production_known_cpu],
                labels[production_known_cpu],
                num_classes=class_count,
            )
            fill_region_confusion += _confusion_matrix(
                fill_predictions_cpu[fill_mask_cpu],
                labels[fill_mask_cpu],
                num_classes=class_count,
            )
            two_step_confusion += _confusion_matrix(
                two_step_predictions_cpu,
                labels,
                num_classes=class_count,
            )

            total_pixels += int(labels.numel())
            labeled_pixels += int(labeled_mask.sum().item())
            selected_pixels += int(production_known_cpu.sum().item())
            selected_labeled_pixels += int((production_known_cpu & labeled_mask).sum().item())
            fill_pixels += int(fill_mask_cpu.sum().item())
            fill_labeled_pixels += int((fill_mask_cpu & labeled_mask).sum().item())
            production_tiles += sample_production_tiles
            fill_tiles += sample_fill_tiles
            production_confidence_sum += float(production_confidence_cpu.sum().item())
            selected_confidence_sum += float(
                production_confidence_cpu[production_known_cpu].sum().item()
            )

            if _should_report(index, len(samples)):
                _report(progress, f"Evaluated {index}/{len(samples)} fill-labeller samples")

    production_dense = _segmentation_metrics(
        production_dense_confusion,
        include_unlabeled_iou=False,
    )
    production_selected = _segmentation_metrics(
        production_selected_confusion,
        include_unlabeled_iou=False,
    )
    fill_region = _segmentation_metrics(
        fill_region_confusion,
        include_unlabeled_iou=False,
    )
    two_step_dense = _segmentation_metrics(two_step_confusion, include_unlabeled_iou=False)
    return {
        "samples": len(samples),
        "production_tiles": production_tiles,
        "fill_tiles": fill_tiles,
        "production_confidence_threshold": config.production_confidence_threshold,
        "production_mean_confidence": (
            production_confidence_sum / total_pixels if total_pixels else 0.0
        ),
        "production_selected_mean_confidence": (
            selected_confidence_sum / selected_pixels if selected_pixels else 0.0
        ),
        "production_selected_pixel_coverage": (
            selected_pixels / total_pixels if total_pixels else 0.0
        ),
        "production_selected_labeled_pixel_coverage": (
            selected_labeled_pixels / labeled_pixels if labeled_pixels else 0.0
        ),
        "fill_region_pixel_coverage": fill_pixels / total_pixels if total_pixels else 0.0,
        "fill_region_labeled_pixel_coverage": (
            fill_labeled_pixels / labeled_pixels if labeled_pixels else 0.0
        ),
        "production_dense": production_dense,
        "production_selected": production_selected,
        "fill_region": fill_region,
        "two_step_dense": two_step_dense,
    }


@dataclass(frozen=True)
class ProductionCondition:
    features: torch.Tensor
    known_mask: torch.Tensor
    confidence: torch.Tensor
    predictions: torch.Tensor
    probabilities: torch.Tensor
    base_logits: torch.Tensor


class ProductionProbabilityFillModel(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
    ) -> None:
        super().__init__()
        input_channels = 3 + (num_classes * 2) + 2
        self.stem = _conv_block(input_channels, 48)
        self.down1 = _down_block(48, 96)
        self.down2 = _down_block(96, 192)
        self.down3 = _down_block(192, 256)
        self.context = _AtrousSpatialPyramid(256, 256)
        self.decode2 = _conv_block(448, 192)
        self.decode1 = _conv_block(288, 128)
        self.decode0 = _conv_block(176, 96)
        self.head = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(
        self,
        images: torch.Tensor,
        production_condition: torch.Tensor,
        production_base_logits: torch.Tensor,
    ) -> torch.Tensor:
        inputs = torch.cat((images, production_condition), dim=1)
        level0 = self.stem(inputs)
        level1 = self.down1(level0)
        level2 = self.down2(level1)
        latent = self.context(self.down3(level2))
        decoded2 = self.decode2(torch.cat((_resize_like(latent, level2), level2), dim=1))
        decoded1 = self.decode1(torch.cat((_resize_like(decoded2, level1), level1), dim=1))
        decoded0 = self.decode0(torch.cat((_resize_like(decoded1, level0), level0), dim=1))
        return production_base_logits + self.head(decoded0)


class _AtrousSpatialPyramid(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        branch_channels = out_channels // 4
        self.branches = nn.ModuleList(
            (
                nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=2, dilation=2),
                nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=4, dilation=4),
                nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=6, dilation=6),
            )
        )
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(image) for branch in self.branches], dim=1))


def _production_condition_for_batch(
    production: SegmentationCheckpoint,
    images: torch.Tensor,
    *,
    confidence_threshold: float,
    class_logit_biases: Mapping[str, float],
) -> ProductionCondition:
    logits = _forward_segmentation_model(production.model, images)
    logits = _apply_class_logit_biases(
        logits,
        class_logit_biases,
        class_names=production.classes,
    )
    probabilities = logits.softmax(dim=1)
    confidence, predictions = probabilities.max(dim=1)
    known_mask = confidence >= confidence_threshold
    features = _production_condition_features(
        probabilities,
        predictions,
        confidence,
        known_mask,
        num_classes=len(production.classes),
    )
    return ProductionCondition(
        features=features,
        known_mask=known_mask,
        confidence=confidence,
        predictions=predictions,
        probabilities=probabilities,
        base_logits=probabilities.clamp_min(1e-6).log(),
    )


def _production_condition_features(
    probabilities: torch.Tensor,
    predictions: torch.Tensor,
    confidence: torch.Tensor,
    known_mask: torch.Tensor,
    *,
    num_classes: int,
) -> torch.Tensor:
    one_hot = F.one_hot(predictions.long(), num_classes=num_classes)
    one_hot = one_hot.permute(0, 3, 1, 2).float()
    known = known_mask.unsqueeze(1).float()
    one_hot = one_hot * known
    return torch.cat(
        (probabilities.float(), one_hot, known, confidence.unsqueeze(1).float()),
        dim=1,
    )


def _unknown_region_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    known_mask: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor | None:
    target_mask = (labels > 0) & (~known_mask)
    if not bool(target_mask.any().item()):
        return None
    logits_by_pixel = logits.permute(0, 2, 3, 1)[target_mask]
    labels_by_pixel = labels[target_mask]
    return F.cross_entropy(logits_by_pixel, labels_by_pixel, weight=class_weights)


def _predict_fill_prepared_sample_logits(
    model: nn.Module,
    sample: SatelliteSample,
    *,
    production_predictions: torch.Tensor,
    production_probabilities: torch.Tensor,
    production_confidence: torch.Tensor,
    production_known_mask: torch.Tensor,
    image_size: int,
    tile_stride: int,
    tile_scales: Sequence[float],
    device: torch.device,
    cache: _PreparedSampleCache,
    feather_blending: bool,
) -> tuple[torch.Tensor, int]:
    prepared = cache.get(sample)
    height, width = prepared.mask.shape
    logit_sum = torch.zeros(
        (len(OPENEARTHMAP_CLASSES), height, width),
        dtype=torch.float32,
        device=device,
    )
    counts = torch.zeros((1, height, width), dtype=torch.float32, device=device)
    tiles = 0
    for tile_size, stride in _scaled_tile_windows_config(
        image_size=image_size,
        tile_stride=tile_stride,
        tile_scales=tile_scales,
    ):
        for crop_box in _tile_windows(prepared.size, tile_size, stride=stride):
            images, _labels = _load_segmentation_batch(
                (sample,),
                (crop_box,),
                cache=cache,
                image_size=image_size,
                context_size=0,
                device=device,
            )
            condition, base_logits = _production_condition_for_crop(
                production_predictions,
                production_probabilities,
                production_confidence,
                production_known_mask,
                crop_box,
                image_size=image_size,
                num_classes=len(OPENEARTHMAP_CLASSES),
                device=device,
            )
            logits = model(images, condition, base_logits)
            resized = _resized_tile_logits(logits, crop_box)
            _accumulate_resized_tile_logits(
                logit_sum,
                counts,
                resized,
                crop_box,
                tile_size=tile_size,
                tile_stride=stride,
                feather_blending=feather_blending,
            )
            tiles += 1
    if not bool((counts > 0).all().item()):
        raise RuntimeError(f"Fill-labeller tiling left uncovered pixels for {sample.image_path}")
    return logit_sum / counts, tiles


def _production_condition_for_crop(
    production_predictions: torch.Tensor,
    production_probabilities: torch.Tensor,
    production_confidence: torch.Tensor,
    production_known_mask: torch.Tensor,
    crop_box: tuple[int, int, int, int],
    *,
    image_size: int,
    num_classes: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    left, top, right, bottom = crop_box
    probabilities = _resize_probability_map(
        production_probabilities[:, top:bottom, left:right],
        image_size=image_size,
        device=device,
    )
    predictions = _resize_label_map(
        production_predictions[top:bottom, left:right],
        image_size=image_size,
        device=device,
    )
    confidence = _resize_float_map(
        production_confidence[top:bottom, left:right],
        image_size=image_size,
        device=device,
    )
    known = _resize_bool_map(
        production_known_mask[top:bottom, left:right],
        image_size=image_size,
        device=device,
    )
    features = _production_condition_features(
        probabilities.unsqueeze(0),
        predictions.unsqueeze(0),
        confidence.unsqueeze(0),
        known.unsqueeze(0),
        num_classes=num_classes,
    )
    return features, probabilities.clamp_min(1e-6).log().unsqueeze(0)


def _resize_probability_map(
    values: torch.Tensor,
    *,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    values = values.to(device=device).float()
    if values.shape[-2:] == (image_size, image_size):
        resized = values
    else:
        resized = F.interpolate(
            values.unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return resized / resized.sum(dim=0, keepdim=True).clamp_min(1e-6)


def _resize_label_map(
    values: torch.Tensor,
    *,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    values = values.to(device=device).long()
    if values.shape == (image_size, image_size):
        return values
    return (
        F.interpolate(
            values.float().view(1, 1, values.shape[0], values.shape[1]),
            size=(image_size, image_size),
            mode="nearest",
        )
        .view(image_size, image_size)
        .long()
    )


def _resize_float_map(
    values: torch.Tensor,
    *,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    values = values.to(device=device).float()
    if values.shape == (image_size, image_size):
        return values
    return (
        F.interpolate(
            values.view(1, 1, values.shape[0], values.shape[1]),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        .view(image_size, image_size)
        .float()
    )


def _resize_bool_map(
    values: torch.Tensor,
    *,
    image_size: int,
    device: torch.device,
) -> torch.Tensor:
    values = values.to(device=device).bool()
    if values.shape == (image_size, image_size):
        return values
    return (
        F.interpolate(
            values.float().view(1, 1, values.shape[0], values.shape[1]),
            size=(image_size, image_size),
            mode="nearest",
        )
        .view(image_size, image_size)
        .bool()
    )


def _sampling_config(config: ProductionProbabilityFillConfig) -> SegmentationTrainingConfig:
    return SegmentationTrainingConfig(
        manifest_path=config.manifest_path,
        checkpoint_path=config.checkpoint_path,
        split=config.split,
        validation_split=config.validation_split,
        max_train_samples=config.max_train_samples,
        max_validation_samples=config.max_validation_samples,
        steps=config.steps,
        batch_size=config.batch_size,
        image_size=config.image_size,
        context_size=0,
        learning_rate=config.learning_rate,
        min_learning_rate=config.min_learning_rate,
        lr_schedule=config.lr_schedule,
        weight_decay=config.weight_decay,
        class_weight_power=config.class_weight_power,
        unlabeled_weight=0.0,
        eval_every=config.eval_every,
        evaluation_stride=config.evaluation_stride,
        evaluation_feather_blending=config.evaluation_feather_blending,
        log_every=config.log_every,
        device=config.device,
        seed=config.seed,
        cache_samples=config.cache_samples,
        augment=config.augment,
        flip_probability=config.flip_probability,
        rotate_probability=config.rotate_probability,
        color_jitter_strength=config.color_jitter_strength,
        blur_probability=config.blur_probability,
        crop_scale_min=config.crop_scale_min,
        crop_scale_max=config.crop_scale_max,
        class_balanced_crop_probability=config.class_balanced_crop_probability,
        class_balanced_crop_weights=dict(config.class_balanced_crop_weights),
        class_balanced_crop_min_pixels=config.class_balanced_crop_min_pixels,
    )


def _fill_learning_rate(
    config: ProductionProbabilityFillConfig,
    *,
    step: int,
) -> float:
    return _segmentation_learning_rate(_sampling_config(config), step=step)


def _save_two_step_checkpoint(
    path: Path,
    model: nn.Module,
    *,
    config: dict[str, Any],
    metrics: dict[str, Any],
    step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": MODEL_NAME,
            "model_state": model.state_dict(),
            "classes": list(OPENEARTHMAP_CLASSES),
            "config": config,
            "metrics": metrics,
            "step": step,
        },
        path,
    )


def _validate_config(config: ProductionProbabilityFillConfig) -> None:
    if not config.manifest_path:
        raise ValueError("manifest_path is required")
    if not config.checkpoint_path:
        raise ValueError("checkpoint_path is required")
    if not config.production_checkpoint_path:
        raise ValueError("production_checkpoint_path is required")
    if (
        config.production_confidence_threshold <= 0.0
        or config.production_confidence_threshold > 1.0
    ):
        raise ValueError("production_confidence_threshold must be in (0, 1]")
    for name in (
        "steps",
        "batch_size",
        "image_size",
        "eval_every",
        "log_every",
        "class_balanced_crop_min_pixels",
    ):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in (
        "max_train_samples",
        "max_validation_samples",
        "cache_samples",
        "evaluation_stride",
        "production_tile_stride",
    ):
        if getattr(config, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    for name in (
        "learning_rate",
        "weight_decay",
        "class_weight_power",
        "min_learning_rate",
    ):
        if getattr(config, name) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if config.min_learning_rate > config.learning_rate:
        raise ValueError("min_learning_rate must be less than or equal to learning_rate")
    if config.lr_schedule not in {"constant", "cosine"}:
        raise ValueError("lr_schedule must be constant or cosine")
    for scale in (*config.production_tile_scales, *config.evaluation_tile_scales):
        if scale <= 0.0:
            raise ValueError("tile scales must be positive")


def _validate_production_checkpoint(production: SegmentationCheckpoint) -> None:
    input_channels = getattr(production.model, "input_channels", 3)
    if not isinstance(input_channels, int) or input_channels != 3:
        raise RuntimeError(
            "The production-probability fill trainer expects a production segmenter with RGB inputs"
        )


def _shuffled_limit_samples(
    samples: Sequence[SatelliteSample],
    *,
    max_samples: int,
    seed: int,
) -> tuple[SatelliteSample, ...]:
    shuffled = list(samples)
    random.Random(seed).shuffle(shuffled)
    if max_samples > 0:
        shuffled = shuffled[:max_samples]
    return tuple(shuffled)


def _resize_like(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        image,
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _down_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


def _count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda"):
        torch.cuda.manual_seed_all(seed)


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_tuple_float(name: str, default: Sequence[float]) -> tuple[float, ...]:
    value = os.environ.get(name)
    if not value:
        return tuple(default)
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _config() -> dict[str, Any]:
    prefix = "SIM_SATELLITE_PRODUCTION_PROBABILITY_FILL"
    steps = _env_int(f"{prefix}_STEPS", 2000)
    threshold = _env_float(f"{prefix}_PRODUCTION_THRESHOLD", 0.8)
    threshold_label = f"{int(round(threshold * 100)):03d}"
    checkpoint_path = os.environ.get(
        f"{prefix}_CHECKPOINT_PATH",
        (
            "artifacts/segmentation-production-probability-fill-v1/"
            f"production-probability-fill-s{steps}-t{threshold_label}.pt"
        ),
    )
    metrics_path = os.environ.get(f"{prefix}_METRICS_PATH", "")
    return {
        "manifest_path": _modal_data_path(
            os.environ.get(
                f"{prefix}_MANIFEST_PATH",
                "data/openearthmap/manifest.jsonl",
            ),
            field_name="manifest_path",
        ),
        "checkpoint_path": _modal_artifact_path(checkpoint_path),
        "metrics_path": _modal_artifact_path(metrics_path) if metrics_path else "",
        "production_checkpoint_path": _modal_artifact_path(
            os.environ.get(
                f"{prefix}_PRODUCTION_CHECKPOINT_PATH",
                PRODUCTION_OPENEARTH_CHECKPOINT_PATH,
            )
        ),
        "split": os.environ.get(f"{prefix}_SPLIT", "train"),
        "validation_split": os.environ.get(f"{prefix}_VALIDATION_SPLIT", "val"),
        "max_train_samples": _env_int(f"{prefix}_MAX_TRAIN_SAMPLES", 0),
        "max_validation_samples": _env_int(
            f"{prefix}_MAX_VALIDATION_SAMPLES",
            128,
        ),
        "production_confidence_threshold": threshold,
        "production_class_logit_biases": dict(PRODUCTION_OPENEARTH_CLASS_LOGIT_BIASES),
        "production_tile_stride": _env_int(
            f"{prefix}_PRODUCTION_TILE_STRIDE",
            PRODUCTION_OPENEARTH_TILE_STRIDE,
        ),
        "production_tile_scales": _env_tuple_float(
            f"{prefix}_PRODUCTION_TILE_SCALES",
            PRODUCTION_OPENEARTH_TILE_SCALES,
        ),
        "production_feather_blending": _env_bool(
            f"{prefix}_PRODUCTION_FEATHER_BLENDING",
            True,
        ),
        "steps": steps,
        "batch_size": _env_int(f"{prefix}_BATCH_SIZE", 2),
        "image_size": _env_int(f"{prefix}_IMAGE_SIZE", 512),
        "learning_rate": _env_float(f"{prefix}_LR", 3e-4),
        "min_learning_rate": _env_float(f"{prefix}_MIN_LR", 3e-5),
        "eval_every": _env_int(f"{prefix}_EVAL_EVERY", 250),
        "evaluation_stride": _env_int(f"{prefix}_EVAL_STRIDE", 256),
        "evaluation_tile_scales": _env_tuple_float(
            f"{prefix}_EVAL_TILE_SCALES",
            (),
        ),
        "evaluation_feather_blending": _env_bool(
            f"{prefix}_EVAL_FEATHER_BLENDING",
            True,
        ),
        "log_every": _env_int(f"{prefix}_LOG_EVERY", 25),
        "device": "cuda",
        "seed": _env_int(f"{prefix}_SEED", 20260705),
        "cache_samples": _env_int(f"{prefix}_CACHE_SAMPLES", 512),
        "augment": _env_bool(f"{prefix}_AUGMENT", True),
    }


@app.function(
    image=image,
    timeout=MODAL_TRAINING_TIMEOUT,
    gpu=MODAL_TRAINING_GPU,
    ephemeral_disk=MODAL_EPHEMERAL_DISK_MIB,
    volumes=modal_volumes,
)
def train_production_probability_fill_remote(config: dict[str, Any]) -> dict[str, Any]:
    _ensure_modal_training_data(config["manifest_path"])
    artifact_volume.reload()
    result = asdict(
        train_production_probability_fill(
            ProductionProbabilityFillConfig.from_dict(config),
            progress=lambda message: _print_modal_event(
                "modal_production_probability_fill_training_progress",
                message=message,
            ),
            after_save=artifact_volume.commit,
        )
    )
    artifact_volume.commit()
    result["execution_target"] = "modal"
    return result


def _print_modal_event(event: str, **payload: object) -> None:
    print(json.dumps({"event": event, **payload}, sort_keys=True))


@app.local_entrypoint()
def production_probability_fill_main() -> None:
    result = train_production_probability_fill_remote.spawn(_config()).get()
    print(json.dumps(result, indent=2, sort_keys=True))
