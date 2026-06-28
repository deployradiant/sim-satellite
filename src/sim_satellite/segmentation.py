"""Direct semantic segmentation baseline for satellite images."""

from __future__ import annotations

import json
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from importlib import import_module
from pathlib import Path
from typing import Any, cast

import click
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from sim_satellite.data.loader import SatelliteSample, read_manifest, write_manifest
from sim_satellite.data.openearthmap import (
    IMAGE_SUFFIXES,
    OPENEARTHMAP_CLASSES,
    OPENEARTHMAP_RGB,
)
from sim_satellite.data.prepared import PreparedSample, load_prepared_sample, write_prepared_sample
from sim_satellite.data.tensors import image_to_uint8_tensor
from sim_satellite.model import ModelConfig
from sim_satellite.training_config import DEFAULT_MANIFEST_PATH

ProgressReporter = Callable[[str], None]
DEFAULT_SEGMENTATION_CHECKPOINT_PATH = "artifacts/openearthmap-direct-segmenter.pt"
DEFAULT_SEGMENTATION_PREDICTION_DIR = "artifacts/segmentation-predictions"
FPN_LITE_MODEL_NAME = "fpn-lite-v1"
DEEPLAB_UNET_MODEL_NAME = "deeplab-unet-v1"
SEGMENTATION_MODEL_NAMES = (
    FPN_LITE_MODEL_NAME,
    DEEPLAB_UNET_MODEL_NAME,
)
DEFAULT_SEGMENTATION_MODEL_NAME = DEEPLAB_UNET_MODEL_NAME


@dataclass(frozen=True)
class SegmentationTrainingConfig:
    manifest_path: str = DEFAULT_MANIFEST_PATH
    checkpoint_path: str = DEFAULT_SEGMENTATION_CHECKPOINT_PATH
    model: str = DEFAULT_SEGMENTATION_MODEL_NAME
    split: str = "train"
    validation_split: str = ""
    validation_holdout_patterns: str = ""
    validation_fraction: float = 0.2
    max_train_samples: int = 0
    max_validation_samples: int = 128
    steps: int = 2000
    batch_size: int = 8
    image_size: int = 128
    context_size: int = 0
    vae_checkpoint_path: str = ""
    vae_normalize_latents: bool = True
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dice_weight: float = 0.5
    class_weight_power: float = 0.5
    unlabeled_weight: float = 0.25
    eval_every: int = 250
    device: str = "cpu"
    seed: int = 2026
    cache_samples: int = 128
    augment: bool = True
    flip_probability: float = 0.5
    rotate_probability: float = 0.75
    color_jitter_strength: float = 0.15
    blur_probability: float = 0.1

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> SegmentationTrainingConfig:
        names = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - names)
        if unknown:
            raise ValueError(f"Unknown segmentation training config fields: {', '.join(unknown)}")
        return cls(**{name: values[name] for name in names if name in values})


@dataclass(frozen=True)
class SegmentationEvaluationConfig:
    checkpoint_path: str = DEFAULT_SEGMENTATION_CHECKPOINT_PATH
    manifest_path: str = DEFAULT_MANIFEST_PATH
    split: str = "train"
    max_samples: int = 128
    output_path: str = ""
    device: str = "cpu"
    cache_samples: int = 64


@dataclass(frozen=True)
class SegmentationPredictionConfig:
    checkpoint_path: str = DEFAULT_SEGMENTATION_CHECKPOINT_PATH
    output_dir: str = DEFAULT_SEGMENTATION_PREDICTION_DIR
    images_manifest_path: str = ""
    images_dir: str = ""
    max_images: int = 0
    device: str = "cpu"
    save_color_masks: bool = True


@dataclass(frozen=True)
class SegmentationPreparationConfig:
    raw_manifest_path: str
    output_manifest_path: str
    prepared_dir: str
    source: str = "custom"
    force_prepared: bool = False


@dataclass(frozen=True)
class SegmentationTrainingResult:
    checkpoint_path: str
    metrics_path: str
    train_samples: int
    validation_samples: int
    steps: int
    parameters: int
    final_train_loss: float
    best_step: int
    best_mean_iou: float
    best_pixel_accuracy: float
    best_class_iou: dict[str, float | None]
    final_mean_iou: float
    final_pixel_accuracy: float
    class_iou: dict[str, float | None]


@dataclass(frozen=True)
class SegmentationEvaluationResult:
    checkpoint_path: str
    samples: int
    tiles: int
    mean_iou: float
    pixel_accuracy: float
    class_iou: dict[str, float | None]


@dataclass(frozen=True)
class SegmentationPredictionResult:
    checkpoint_path: str
    output_dir: str
    manifest_path: str
    images: int


@dataclass(frozen=True)
class SegmentationPreparationResult:
    raw_manifest_path: str
    output_manifest_path: str
    prepared_dir: str
    samples: int
    splits: dict[str, int]
    sources: dict[str, int]


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    source: str
    split: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RawSegmentationRecord:
    image_path: Path
    mask_path: Path
    split: str
    source: str


@dataclass(frozen=True)
class VAEFeatureEncoder:
    model: nn.Module
    latent_channels: int
    mean: torch.Tensor | None
    std: torch.Tensor | None
    normalize: bool


def prepare_segmentation_manifest(
    config: SegmentationPreparationConfig,
    *,
    progress: ProgressReporter | None = None,
) -> SegmentationPreparationResult:
    _validate_preparation_config(config)
    raw_manifest_path = Path(config.raw_manifest_path)
    output_manifest_path = Path(config.output_manifest_path)
    prepared_dir = Path(config.prepared_dir)
    records = _read_raw_segmentation_records(raw_manifest_path, default_source=config.source)
    if not records:
        raise RuntimeError(f"No raw segmentation records found in {raw_manifest_path}")

    model_config = ModelConfig(mask_channels=len(OPENEARTHMAP_CLASSES))
    samples: list[SatelliteSample] = []
    split_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for index, record in enumerate(records, start=1):
        sample = SatelliteSample(
            image_path=record.image_path,
            mask_path=record.mask_path,
            prepared_path=_prepared_path_for_raw_record(record, prepared_dir, index=index),
            split=record.split,
            source=record.source,
        )
        write_prepared_sample(
            sample,
            torch=torch,
            config=model_config,
            force=config.force_prepared,
        )
        samples.append(sample)
        split_counts[sample.split] = split_counts.get(sample.split, 0) + 1
        source_counts[sample.source] = source_counts.get(sample.source, 0) + 1
        if _should_report(index, len(records)):
            _report(progress, f"Prepared segmentation sample {index}/{len(records)}")

    write_manifest(samples, output_manifest_path)
    return SegmentationPreparationResult(
        raw_manifest_path=str(raw_manifest_path),
        output_manifest_path=str(output_manifest_path),
        prepared_dir=str(prepared_dir),
        samples=len(samples),
        splits=split_counts,
        sources=source_counts,
    )


def train_segmentation_model(
    config: SegmentationTrainingConfig,
    *,
    progress: ProgressReporter | None = None,
) -> SegmentationTrainingResult:
    _validate_training_config(config)
    _seed_everything(config.seed)

    device = torch.device(config.device)
    train_samples, validation_samples = _split_training_samples(config)
    cache = _PreparedSampleCache(max_samples=config.cache_samples)
    checkpoint_config = asdict(config)
    vae_encoder = _load_vae_feature_encoder(checkpoint_config, device=device)
    if vae_encoder is not None:
        checkpoint_config["vae_latent_channels"] = vae_encoder.latent_channels
    model = build_segmentation_model(
        config.model,
        num_classes=len(OPENEARTHMAP_CLASSES),
        input_channels=_segmentation_input_channels(config),
        vae_latent_channels=vae_encoder.latent_channels if vae_encoder is not None else 0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    _report(progress, f"Scanning class balance across {len(train_samples)} samples")
    class_weights = _class_weights(
        train_samples,
        cache=cache,
        power=config.class_weight_power,
        unlabeled_weight=config.unlabeled_weight,
        progress=progress,
    ).to(device)

    rng = random.Random(config.seed)
    losses: list[float] = []
    best_metrics: dict[str, Any] | None = None
    best_step = 0
    checkpoint_path = Path(config.checkpoint_path)
    metrics_path = checkpoint_path.with_suffix(".json")

    _report(
        progress,
        f"Training direct segmenter on {len(train_samples)} samples; "
        f"validating on {len(validation_samples)} samples",
    )
    model.train()
    for step in range(1, config.steps + 1):
        batch_samples = tuple(
            train_samples[rng.randrange(len(train_samples))] for _ in range(config.batch_size)
        )
        crop_boxes = [
            _random_square_crop_box(cache.get(sample), config.image_size, rng)
            for sample in batch_samples
        ]
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
            vae_latents=_encode_vae_features(vae_encoder, images),
        )
        ce_loss = F.cross_entropy(logits, labels, weight=class_weights)
        dice_loss = _multiclass_dice_loss(
            logits,
            labels,
            include_unlabeled=False,
        )
        loss = ce_loss + config.dice_weight * dice_loss
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

        should_eval = step == config.steps or step % config.eval_every == 0
        should_report = step == 1 or should_eval or step % max(1, config.steps // 10) == 0
        if should_report:
            _report(progress, f"step={step} loss={losses[-1]:.4f}")
        if should_eval:
            metrics = _evaluate_samples(
                model,
                validation_samples,
                image_size=config.image_size,
                context_size=config.context_size,
                device=device,
                cache=cache,
                vae_encoder=vae_encoder,
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
            model.train()

    final_metrics = _evaluate_samples(
        model,
        validation_samples,
        image_size=config.image_size,
        context_size=config.context_size,
        device=device,
        cache=cache,
        vae_encoder=vae_encoder,
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
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def evaluate_segmentation_model(
    config: SegmentationEvaluationConfig,
) -> SegmentationEvaluationResult:
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint = _load_segmentation_checkpoint(checkpoint_path, device=config.device)
    model = checkpoint["model"]
    model_config = checkpoint["config"]
    vae_encoder = checkpoint["vae_encoder"]
    image_size = int(model_config["image_size"])
    context_size = int(model_config.get("context_size", 0))

    samples = _samples_for_split(config.manifest_path, config.split)
    if config.max_samples > 0:
        samples = samples[: config.max_samples]
    if not samples:
        raise RuntimeError(f"No samples for split `{config.split}` in {config.manifest_path}")

    cache = _PreparedSampleCache(max_samples=config.cache_samples)
    metrics = _evaluate_samples(
        model,
        samples,
        image_size=image_size,
        context_size=context_size,
        device=torch.device(config.device),
        cache=cache,
        vae_encoder=vae_encoder,
    )
    result = SegmentationEvaluationResult(
        checkpoint_path=str(checkpoint_path),
        samples=len(samples),
        tiles=int(metrics["tiles"]),
        mean_iou=float(metrics["mean_iou"]),
        pixel_accuracy=float(metrics["pixel_accuracy"]),
        class_iou=metrics["class_iou"],
    )
    if config.output_path:
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
    return result


def predict_segmentation_masks(
    config: SegmentationPredictionConfig,
    *,
    progress: ProgressReporter | None = None,
) -> SegmentationPredictionResult:
    _validate_prediction_config(config)
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint = _load_segmentation_checkpoint(checkpoint_path, device=config.device)
    model = checkpoint["model"]
    model_config = checkpoint["config"]
    vae_encoder = checkpoint["vae_encoder"]
    image_size = int(model_config["image_size"])
    context_size = int(model_config.get("context_size", 0))
    device = torch.device(config.device)

    image_records = _prediction_image_records(config)
    if config.max_images > 0:
        image_records = image_records[: config.max_images]
    if not image_records:
        raise RuntimeError("No prediction images found")

    output_dir = Path(config.output_dir)
    mask_dir = output_dir / "masks"
    color_dir = output_dir / "color-masks"
    manifest_path = output_dir / "manifest.jsonl"
    records: list[dict[str, Any]] = []
    for index, record in enumerate(image_records, start=1):
        prediction = _predict_image_mask(
            model,
            record.image_path,
            image_size=image_size,
            context_size=context_size,
            device=device,
            vae_encoder=vae_encoder,
        )
        mask_path = mask_dir / _prediction_mask_name(index, record.image_path)
        _write_label_mask(mask_path, prediction["labels"])
        color_mask_path = ""
        if config.save_color_masks:
            color_path = color_dir / _prediction_mask_name(index, record.image_path)
            _write_color_mask(color_path, prediction["labels"])
            color_mask_path = str(color_path)

        payload = {
            **record.metadata,
            "image": str(record.image_path),
            "mask": str(mask_path),
            "color_mask": color_mask_path,
            "source": record.source,
            "split": record.split,
            "prediction_source": "direct-segmentation-baseline",
            "checkpoint": str(checkpoint_path),
            "mean_confidence": prediction["mean_confidence"],
            "class_counts": prediction["class_counts"],
        }
        records.append(payload)
        if _should_report(index, len(image_records)):
            _report(progress, f"Predicted {index}/{len(image_records)} masks")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")

    return SegmentationPredictionResult(
        checkpoint_path=str(checkpoint_path),
        output_dir=str(output_dir),
        manifest_path=str(manifest_path),
        images=len(records),
    )


def build_segmentation_model(
    model_name: str,
    *,
    num_classes: int,
    input_channels: int = 3,
    vae_latent_channels: int = 0,
) -> nn.Module:
    if model_name == FPN_LITE_MODEL_NAME:
        return DirectSegmentationModel(num_classes=num_classes, input_channels=input_channels)
    if model_name == DEEPLAB_UNET_MODEL_NAME:
        return DeepLabUNetSegmentationModel(
            num_classes=num_classes,
            input_channels=input_channels,
            vae_latent_channels=vae_latent_channels,
        )
    raise ValueError(f"Unsupported segmentation model: {model_name}")


def _segmentation_input_channels(config: SegmentationTrainingConfig | Mapping[str, Any]) -> int:
    image_size = int(_config_value(config, "image_size", 0))
    context_size = int(_config_value(config, "context_size", 0))
    return 6 if context_size > image_size else 3


def _config_value(
    config: SegmentationTrainingConfig | Mapping[str, Any],
    name: str,
    default: Any,
) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _load_vae_feature_encoder(
    config: Mapping[str, Any],
    *,
    device: torch.device,
) -> VAEFeatureEncoder | None:
    checkpoint_value = config.get("vae_checkpoint_path", "")
    if not checkpoint_value:
        return None
    checkpoint_path = Path(str(checkpoint_value))
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"VAE checkpoint not found: {checkpoint_path}")

    try:
        vae_model_module = import_module("sim_satellite.vae.model")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "VAE segmentation features require the VAE module from current main"
        ) from exc

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f"VAE checkpoint must be a mapping: {checkpoint_path}")
    vae_config_values = checkpoint.get("vae_config")
    if not isinstance(vae_config_values, Mapping):
        raise RuntimeError(f"VAE checkpoint is missing vae_config: {checkpoint_path}")
    model_state = checkpoint.get("model_state")
    if not isinstance(model_state, Mapping):
        raise RuntimeError(f"VAE checkpoint is missing model_state: {checkpoint_path}")

    vae_config = vae_model_module.SatelliteVAEConfig(**dict(vae_config_values))
    vae_model = vae_model_module.SatelliteVAE(vae_config).to(device)
    vae_model.load_state_dict(model_state)
    vae_model.eval()
    for parameter in vae_model.parameters():
        parameter.requires_grad_(False)

    latent_channels = int(vae_config_values.get("latent_channels", 0))
    mean: torch.Tensor | None = None
    std: torch.Tensor | None = None
    latent_stats = checkpoint.get("latent_stats")
    normalize = bool(config.get("vae_normalize_latents", True))
    if normalize and isinstance(latent_stats, Mapping):
        raw_mean = latent_stats.get("mean")
        raw_std = latent_stats.get("std")
        if isinstance(raw_mean, Sequence) and isinstance(raw_std, Sequence):
            mean = torch.tensor(raw_mean, dtype=torch.float32, device=device).view(1, -1, 1, 1)
            std = (
                torch.tensor(raw_std, dtype=torch.float32, device=device)
                .clamp_min(1e-6)
                .view(1, -1, 1, 1)
            )

    return VAEFeatureEncoder(
        model=vae_model,
        latent_channels=latent_channels,
        mean=mean,
        std=std,
        normalize=normalize,
    )


def _encode_vae_features(
    encoder: VAEFeatureEncoder | None,
    images: torch.Tensor,
) -> torch.Tensor | None:
    if encoder is None:
        return None
    vae_images = images[:, :3].mul(2.0).sub(1.0).clamp(-1.0, 1.0)
    vae_model = cast(Any, encoder.model)
    with torch.no_grad():
        mu, _logvar = vae_model.encode(vae_images)
    if encoder.normalize and encoder.mean is not None and encoder.std is not None:
        mu = (mu - encoder.mean) / encoder.std
    return mu.detach()


def _forward_segmentation_model(
    model: nn.Module,
    images: torch.Tensor,
    *,
    vae_latents: torch.Tensor | None,
) -> torch.Tensor:
    if vae_latents is None:
        return model(images)
    if not isinstance(model, DeepLabUNetSegmentationModel):
        raise RuntimeError("VAE features require deeplab-unet-v1")
    return model(images, vae_latents)


class DirectSegmentationModel(nn.Module):
    def __init__(self, *, num_classes: int, input_channels: int = 3) -> None:
        super().__init__()
        self.stem = _conv_block(input_channels, 32)
        self.down1 = _down_block(32, 64)
        self.down2 = _down_block(64, 128)
        self.down3 = _down_block(128, 192)
        self.context = _AtrousSpatialPyramid(192, 192)
        self.lateral3 = nn.Conv2d(192, 96, kernel_size=1)
        self.lateral2 = nn.Conv2d(128, 96, kernel_size=1)
        self.lateral1 = nn.Conv2d(64, 96, kernel_size=1)
        self.lateral0 = nn.Conv2d(32, 96, kernel_size=1)
        self.smooth3 = _conv_block(96, 96)
        self.smooth2 = _conv_block(96, 96)
        self.smooth1 = _conv_block(96, 96)
        self.smooth0 = _conv_block(96, 96)
        self.head = nn.Sequential(
            _conv_block(96, 64),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        level0 = self.stem(image)
        level1 = self.down1(level0)
        level2 = self.down2(level1)
        level3 = self.context(self.down3(level2))

        pyramid3 = self.smooth3(self.lateral3(level3))
        pyramid2 = self.smooth2(
            self.lateral2(level2)
            + F.interpolate(pyramid3, size=level2.shape[-2:], mode="bilinear", align_corners=False)
        )
        pyramid1 = self.smooth1(
            self.lateral1(level1)
            + F.interpolate(pyramid2, size=level1.shape[-2:], mode="bilinear", align_corners=False)
        )
        pyramid0 = self.smooth0(
            self.lateral0(level0)
            + F.interpolate(pyramid1, size=level0.shape[-2:], mode="bilinear", align_corners=False)
        )
        return self.head(pyramid0)


class DeepLabUNetSegmentationModel(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int,
        input_channels: int = 3,
        vae_latent_channels: int = 0,
    ) -> None:
        super().__init__()
        self.stem = _conv_block(input_channels, 48)
        self.down1 = _down_block(48, 96)
        self.down2 = _down_block(96, 192)
        self.vae_projection = (
            nn.Sequential(
                nn.Conv2d(vae_latent_channels, 192, kernel_size=1, bias=False),
                nn.BatchNorm2d(192),
                nn.ReLU(inplace=True),
            )
            if vae_latent_channels > 0
            else None
        )
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
        image: torch.Tensor,
        vae_latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        level0 = self.stem(image)
        level1 = self.down1(level0)
        level2 = self.down2(level1)
        if self.vae_projection is not None:
            if vae_latents is None:
                raise RuntimeError("VAE latents are required for this segmentation model")
            level2 = level2 + _resize_like(self.vae_projection(vae_latents), level2)
        latent = self.context(self.down3(level2))

        decoded2 = self.decode2(
            torch.cat((_resize_like(latent, level2), level2), dim=1),
        )
        decoded1 = self.decode1(
            torch.cat((_resize_like(decoded2, level1), level1), dim=1),
        )
        decoded0 = self.decode0(
            torch.cat((_resize_like(decoded1, level0), level0), dim=1),
        )
        return self.head(decoded0)


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


class _AtrousSpatialPyramid(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        branch_channels = out_channels // 4
        self.branches = nn.ModuleList(
            [
                nn.Conv2d(in_channels, branch_channels, kernel_size=1),
                nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=2, dilation=2),
                nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=4, dilation=4),
                nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=6, dilation=6),
            ]
        )
        self.project = nn.Sequential(
            nn.Conv2d(branch_channels * 4, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(image) for branch in self.branches], dim=1))


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


def _read_raw_segmentation_records(
    raw_manifest_path: Path,
    *,
    default_source: str,
) -> tuple[RawSegmentationRecord, ...]:
    root = raw_manifest_path.parent
    records: list[RawSegmentationRecord] = []
    with raw_manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{raw_manifest_path}:{line_number}: invalid JSON") from exc
            if not isinstance(payload, Mapping):
                raise ValueError(f"{raw_manifest_path}:{line_number}: record must be an object")
            image = _required_manifest_string(payload, "image", raw_manifest_path, line_number)
            mask = _required_manifest_string(payload, "mask", raw_manifest_path, line_number)
            split = str(payload.get("split", "train")).strip() or "train"
            source = str(payload.get("source", default_source)).strip() or default_source
            records.append(
                RawSegmentationRecord(
                    image_path=_resolve_manifest_path(root, image),
                    mask_path=_resolve_manifest_path(root, mask),
                    split=split,
                    source=source,
                )
            )
    return tuple(records)


def _required_manifest_string(
    payload: Mapping[str, Any],
    name: str,
    path: Path,
    line_number: int,
) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}:{line_number}: {name} is required")
    return value


def _resolve_manifest_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _prepared_path_for_raw_record(
    record: RawSegmentationRecord,
    prepared_dir: Path,
    *,
    index: int,
) -> Path:
    source = _safe_path_component(record.source)
    split = _safe_path_component(record.split)
    stem = _safe_path_component(record.image_path.stem)
    return prepared_dir / source / split / f"{index:06d}-{stem}.pt"


def _safe_path_component(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_." else "-" for character in value
    )
    return safe.strip(".-") or "sample"


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
) -> tuple[tuple[int, int, int, int], ...]:
    width, height = image_size
    if width < 1 or height < 1:
        raise ValueError(f"Invalid image size: {image_size}")
    windows: list[tuple[int, int, int, int]] = []
    for top in range(0, height, tile_size):
        bottom = min(height, top + tile_size)
        for left in range(0, width, tile_size):
            right = min(width, left + tile_size)
            windows.append((left, top, right, bottom))
    return tuple(windows)


def _evaluate_samples(
    model: nn.Module,
    samples: Sequence[SatelliteSample],
    *,
    image_size: int,
    device: torch.device,
    cache: _PreparedSampleCache,
    context_size: int = 0,
    vae_encoder: VAEFeatureEncoder | None = None,
) -> dict[str, Any]:
    model.eval()
    confusion = torch.zeros(
        (len(OPENEARTHMAP_CLASSES), len(OPENEARTHMAP_CLASSES)),
        dtype=torch.long,
    )
    tiles = 0
    with torch.no_grad():
        for sample in samples:
            prepared = cache.get(sample)
            for crop_box in _tile_windows(prepared.size, image_size):
                images, labels = _load_segmentation_batch(
                    (sample,),
                    (crop_box,),
                    cache=cache,
                    image_size=image_size,
                    context_size=context_size,
                    device=device,
                )
                predictions = (
                    _forward_segmentation_model(
                        model,
                        images,
                        vae_latents=_encode_vae_features(vae_encoder, images),
                    )
                    .argmax(dim=1)
                    .cpu()
                )
                confusion += _confusion_matrix(
                    predictions,
                    labels.cpu(),
                    num_classes=len(OPENEARTHMAP_CLASSES),
                )
                tiles += 1
    metrics = _segmentation_metrics(confusion, include_unlabeled_iou=False)
    metrics["tiles"] = tiles
    return metrics


def _class_weights(
    samples: Sequence[SatelliteSample],
    *,
    cache: _PreparedSampleCache,
    power: float,
    unlabeled_weight: float,
    progress: ProgressReporter | None,
) -> torch.Tensor:
    counts = torch.zeros(len(OPENEARTHMAP_CLASSES), dtype=torch.long)
    for index, sample in enumerate(samples, start=1):
        labels = cache.get(sample).mask.long()
        counts += torch.bincount(labels.flatten(), minlength=len(OPENEARTHMAP_CLASSES))
        if _should_report(index, len(samples)):
            _report(progress, f"Scanned class balance {index}/{len(samples)} samples")
    return _class_weights_from_counts(
        counts,
        power=power,
        unlabeled_weight=unlabeled_weight,
    )


def _class_weights_from_counts(
    counts: torch.Tensor,
    *,
    power: float,
    unlabeled_weight: float,
) -> torch.Tensor:
    if power < 0.0:
        raise ValueError("class_weight_power must be non-negative")
    if unlabeled_weight < 0.0:
        raise ValueError("unlabeled_weight must be non-negative")
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


def _multiclass_dice_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    include_unlabeled: bool,
) -> torch.Tensor:
    num_classes = logits.shape[1]
    probabilities = logits.softmax(dim=1)
    target = F.one_hot(labels.long(), num_classes=num_classes).permute(0, 3, 1, 2).float()
    start_class = 0 if include_unlabeled else 1
    probabilities = probabilities[:, start_class:]
    target = target[:, start_class:]
    intersection = (probabilities * target).sum(dim=(2, 3))
    denominator = probabilities.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()


def _confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_classes: int,
) -> torch.Tensor:
    valid = (labels >= 0) & (labels < num_classes)
    encoded = num_classes * labels[valid].long() + predictions[valid].long()
    return torch.bincount(encoded, minlength=num_classes * num_classes).reshape(
        num_classes,
        num_classes,
    )


def _segmentation_metrics(
    confusion: torch.Tensor,
    *,
    include_unlabeled_iou: bool,
) -> dict[str, Any]:
    true_positive = confusion.diag().float()
    predicted = confusion.sum(dim=0).float()
    actual = confusion.sum(dim=1).float()
    union = predicted + actual - true_positive
    class_iou_values: list[float] = []
    class_iou: dict[str, float | None] = {}
    for class_id, class_name in enumerate(OPENEARTHMAP_CLASSES):
        if union[class_id].item() == 0:
            class_iou[class_name] = None
            continue
        value = float((true_positive[class_id] / union[class_id]).item())
        class_iou[class_name] = value
        if include_unlabeled_iou or class_id != 0:
            class_iou_values.append(value)
    total = int(confusion.sum().item())
    pixel_accuracy = float(true_positive.sum().item() / total) if total else 0.0
    mean_iou = sum(class_iou_values) / len(class_iou_values) if class_iou_values else 0.0
    return {
        "mean_iou": mean_iou,
        "pixel_accuracy": pixel_accuracy,
        "class_iou": class_iou,
    }


def _save_checkpoint(
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
            "model": config["model"],
            "model_state": model.state_dict(),
            "classes": list(OPENEARTHMAP_CLASSES),
            "config": config,
            "metrics": metrics,
            "step": step,
        },
        path,
    )


def _load_segmentation_checkpoint(path: Path, *, device: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Segmentation checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f"Segmentation checkpoint must be a mapping: {path}")
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        raise RuntimeError(f"Segmentation checkpoint config must be a mapping: {path}")
    model_name = checkpoint.get("model")
    if not isinstance(model_name, str):
        raise RuntimeError(f"Segmentation checkpoint model must be a string: {path}")
    if model_name not in SEGMENTATION_MODEL_NAMES:
        raise RuntimeError(
            f"Unsupported segmentation checkpoint model {model_name!r}; "
            f"expected one of {', '.join(SEGMENTATION_MODEL_NAMES)}: {path}"
        )
    classes = checkpoint.get("classes")
    if not isinstance(classes, Sequence):
        raise RuntimeError(f"Segmentation checkpoint classes must be a sequence: {path}")
    if tuple(classes) != OPENEARTHMAP_CLASSES:
        raise RuntimeError(f"Segmentation checkpoint classes do not match OpenEarthMap: {path}")
    state = checkpoint.get("model_state")
    if not isinstance(state, Mapping):
        raise RuntimeError(f"Segmentation checkpoint is missing model_state: {path}")
    vae_encoder = _load_vae_feature_encoder(dict(config), device=torch.device(device))
    model = build_segmentation_model(
        model_name,
        num_classes=len(OPENEARTHMAP_CLASSES),
        input_channels=_segmentation_input_channels(config),
        vae_latent_channels=int(config.get("vae_latent_channels", 0)),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return {"model": model, "config": dict(config), "vae_encoder": vae_encoder}


def _predict_image_mask(
    model: nn.Module,
    image_path: Path,
    *,
    image_size: int,
    device: torch.device,
    context_size: int = 0,
    vae_encoder: VAEFeatureEncoder | None = None,
) -> dict[str, Any]:
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        output = torch.zeros((image.height, image.width), dtype=torch.uint8)
        confidence_sum = 0.0
        pixel_count = 0
        for crop_box in _tile_windows(image.size, image_size):
            crop = image.crop(crop_box)
            tensor = image_to_uint8_tensor(
                crop,
                torch=torch,
                size=(image_size, image_size),
            )
            if context_size > image_size:
                context_crop = image.crop(_context_crop_box(image.size, crop_box, context_size))
                context_tensor = image_to_uint8_tensor(
                    context_crop,
                    torch=torch,
                    size=(image_size, image_size),
                )
                tensor = torch.cat((tensor, context_tensor), dim=2)
            batch = tensor.permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            with torch.no_grad():
                probabilities = (
                    _forward_segmentation_model(
                        model,
                        batch,
                        vae_latents=_encode_vae_features(vae_encoder, batch),
                    )
                    .softmax(dim=1)
                    .cpu()
                )
            confidence, labels = probabilities.max(dim=1)
            left, top, right, bottom = crop_box
            target_size = (right - left, bottom - top)
            labels = (
                F.interpolate(
                    labels.float().unsqueeze(1),
                    size=(target_size[1], target_size[0]),
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .to(torch.uint8)
            )
            confidence = (
                F.interpolate(
                    confidence.unsqueeze(1),
                    size=(target_size[1], target_size[0]),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .squeeze(0)
            )
            output[top:bottom, left:right] = labels
            confidence_sum += float(confidence.sum().item())
            pixel_count += int(confidence.numel())
    counts = torch.bincount(output.long().flatten(), minlength=len(OPENEARTHMAP_CLASSES))
    return {
        "labels": output,
        "mean_confidence": confidence_sum / pixel_count if pixel_count else 0.0,
        "class_counts": {
            class_name: int(counts[index].item())
            for index, class_name in enumerate(OPENEARTHMAP_CLASSES)
            if counts[index].item()
        },
    }


def _prediction_image_records(config: SegmentationPredictionConfig) -> tuple[ImageRecord, ...]:
    if config.images_manifest_path:
        return _read_image_records(config.images_manifest_path)
    image_dir = Path(config.images_dir)
    return tuple(
        ImageRecord(
            image_path=path,
            source="image-directory",
            split="predict",
            metadata={},
        )
        for path in sorted(image_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _read_image_records(path: str | Path) -> tuple[ImageRecord, ...]:
    manifest_path = Path(path)
    root = manifest_path.parent
    records: list[ImageRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{manifest_path}:{line_number}: invalid JSON") from exc
            image = payload.get("image")
            if not isinstance(image, str) or not image:
                raise ValueError(f"{manifest_path}:{line_number}: image is required")
            image_path = Path(image)
            if not image_path.is_absolute():
                image_path = root / image_path
            metadata = dict(payload)
            records.append(
                ImageRecord(
                    image_path=image_path,
                    source=str(payload.get("source", "image-manifest")),
                    split=str(payload.get("split", "predict")),
                    metadata=metadata,
                )
            )
    return tuple(records)


def _write_label_mask(path: Path, labels: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = labels.shape
    values = labels.to(device="cpu", dtype=torch.uint8).contiguous()
    Image.frombytes("L", (width, height), values.numpy().tobytes()).save(path)


def _write_color_mask(path: Path, labels: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = labels.shape
    image = Image.frombytes(
        "L",
        (width, height),
        labels.to(device="cpu", dtype=torch.uint8).contiguous().numpy().tobytes(),
    )
    palette: list[int] = []
    for class_name in OPENEARTHMAP_CLASSES:
        palette.extend(OPENEARTHMAP_RGB[class_name])
    palette.extend([0] * (768 - len(palette)))
    image.putpalette(palette)
    image.convert("RGB").save(path)


def _prediction_mask_name(index: int, image_path: Path) -> str:
    return f"{index:06d}-{image_path.stem}.png"


def _count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda"):
        torch.cuda.manual_seed_all(seed)


def _validate_training_config(config: SegmentationTrainingConfig) -> None:
    if config.validation_fraction <= 0.0 or config.validation_fraction >= 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    for name in ("steps", "batch_size", "image_size", "eval_every"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in ("max_train_samples", "max_validation_samples", "cache_samples", "context_size"):
        if getattr(config, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if config.context_size > 0 and config.context_size <= config.image_size:
        raise ValueError("context_size must be 0 or greater than image_size")
    for name in ("learning_rate",):
        if getattr(config, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    for name in ("weight_decay", "dice_weight", "class_weight_power", "unlabeled_weight"):
        if getattr(config, name) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    for name in ("flip_probability", "rotate_probability", "blur_probability"):
        value = getattr(config, name)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if config.color_jitter_strength < 0.0:
        raise ValueError("color_jitter_strength must be non-negative")
    if config.model not in SEGMENTATION_MODEL_NAMES:
        raise ValueError(f"model must be one of: {', '.join(SEGMENTATION_MODEL_NAMES)}")
    if config.vae_checkpoint_path and config.model != DEEPLAB_UNET_MODEL_NAME:
        raise ValueError("vae_checkpoint_path requires model=deeplab-unet-v1")
    if not config.manifest_path:
        raise ValueError("manifest_path is required")
    if not config.checkpoint_path:
        raise ValueError("checkpoint_path is required")


def _validate_preparation_config(config: SegmentationPreparationConfig) -> None:
    if not config.raw_manifest_path:
        raise ValueError("raw_manifest_path is required")
    if not config.output_manifest_path:
        raise ValueError("output_manifest_path is required")
    if not config.prepared_dir:
        raise ValueError("prepared_dir is required")
    if not config.source:
        raise ValueError("source is required")


def _validate_prediction_config(config: SegmentationPredictionConfig) -> None:
    if not config.checkpoint_path:
        raise ValueError("checkpoint_path is required")
    if not config.output_dir:
        raise ValueError("output_dir is required")
    if bool(config.images_manifest_path) == bool(config.images_dir):
        raise ValueError("pass exactly one of images_manifest_path or images_dir")
    if config.max_images < 0:
        raise ValueError("max_images must be non-negative")


def _should_report(current: int, total: int) -> bool:
    if total <= 0:
        return False
    if current == 1 or current == total or total <= 10:
        return True
    step = max(1, total // 10)
    return current % step == 0


def _report(progress: ProgressReporter | None, message: str) -> None:
    if progress is not None:
        progress(message)


def load_segmentation_training_config_file(config_path: Path) -> SegmentationTrainingConfig:
    values = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError("Segmentation training config JSON must be an object")
    return SegmentationTrainingConfig.from_dict(values)


@click.command("segmentation-prepare", help="Prepare a raw segmentation manifest for training.")
@click.argument("raw_manifest_path", type=click.Path(path_type=Path))
@click.option(
    "--output-manifest",
    "output_manifest_path",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option("--prepared-dir", type=click.Path(path_type=Path), required=True)
@click.option("--source", default="custom", show_default=True)
@click.option("--force-prepared", is_flag=True, help="Rewrite prepared sample data files.")
def prepare_cli(
    raw_manifest_path: Path,
    output_manifest_path: Path,
    prepared_dir: Path,
    source: str,
    force_prepared: bool,
) -> None:
    try:
        result = prepare_segmentation_manifest(
            SegmentationPreparationConfig(
                raw_manifest_path=str(raw_manifest_path),
                output_manifest_path=str(output_manifest_path),
                prepared_dir=str(prepared_dir),
                source=source,
                force_prepared=force_prepared,
            ),
            progress=click.echo,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(asdict(result), indent=2, sort_keys=True))


@click.command("segmentation-train", help="Train a direct RGB-to-mask segmentation baseline.")
@click.argument("config_path", type=click.Path(path_type=Path))
def train_cli(config_path: Path) -> None:
    try:
        result = train_segmentation_model(
            load_segmentation_training_config_file(config_path),
            progress=click.echo,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(asdict(result), indent=2, sort_keys=True))


@click.command("segmentation-evaluate", help="Evaluate a direct segmentation checkpoint.")
@click.option("--checkpoint", "checkpoint_path", default=DEFAULT_SEGMENTATION_CHECKPOINT_PATH)
@click.option("--manifest", "manifest_path", default=DEFAULT_MANIFEST_PATH)
@click.option("--split", default="train")
@click.option("--max-samples", type=int, default=128, show_default=True)
@click.option("--output", "output_path", default="")
@click.option("--device", default="cpu", show_default=True)
def evaluate_cli(
    checkpoint_path: str,
    manifest_path: str,
    split: str,
    max_samples: int,
    output_path: str,
    device: str,
) -> None:
    try:
        result = evaluate_segmentation_model(
            SegmentationEvaluationConfig(
                checkpoint_path=checkpoint_path,
                manifest_path=manifest_path,
                split=split,
                max_samples=max_samples,
                output_path=output_path,
                device=device,
            )
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(asdict(result), indent=2, sort_keys=True))


@click.command("segmentation-predict", help="Predict semantic masks for unlabeled images.")
@click.option("--checkpoint", "checkpoint_path", default=DEFAULT_SEGMENTATION_CHECKPOINT_PATH)
@click.option("--output-dir", default=DEFAULT_SEGMENTATION_PREDICTION_DIR)
@click.option("--images-manifest", "images_manifest_path", default="")
@click.option("--images-dir", default="")
@click.option("--max-images", type=int, default=0, show_default=True)
@click.option("--device", default="cpu", show_default=True)
@click.option("--color-masks/--no-color-masks", default=True, show_default=True)
def predict_cli(
    checkpoint_path: str,
    output_dir: str,
    images_manifest_path: str,
    images_dir: str,
    max_images: int,
    device: str,
    color_masks: bool,
) -> None:
    try:
        result = predict_segmentation_masks(
            SegmentationPredictionConfig(
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                images_manifest_path=images_manifest_path,
                images_dir=images_dir,
                max_images=max_images,
                device=device,
                save_color_masks=color_masks,
            ),
            progress=click.echo,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(asdict(result), indent=2, sort_keys=True))
