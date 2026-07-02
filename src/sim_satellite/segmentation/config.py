"""Configuration and result payloads for semantic segmentation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from sim_satellite.training_config import DEFAULT_MANIFEST_PATH

DEFAULT_SEGMENTATION_CHECKPOINT_PATH = "artifacts/openearthmap-vae-segmenter.pt"
DEFAULT_SEGMENTATION_VAE_CHECKPOINT_PATH = (
    "artifacts/vae-research/balanced-vae-pool2048-f4-c32-h48-5k-t04-data20274737-train1006705.pt"
)
DEFAULT_SEGMENTATION_PREDICTION_DIR = "artifacts/segmentation-predictions"
DEFAULT_SEGMENTATION_PREDICTION_TILE_STRIDE = 256
VAE_ASPP_UNET_MODEL_NAME = "vae-aspp-unet-v1"
SMP_UNET_EFFICIENTNET_B4_SCSE_MODEL_NAME = "smp-unet-efficientnet-b4-scse"
LEGACY_SEGMENTATION_MODEL_NAMES = ("deeplab-unet-v1",)
SUPPORTED_SEGMENTATION_MODEL_NAMES = (
    VAE_ASPP_UNET_MODEL_NAME,
    SMP_UNET_EFFICIENTNET_B4_SCSE_MODEL_NAME,
    *LEGACY_SEGMENTATION_MODEL_NAMES,
)
SEGMENTATION_LR_SCHEDULES = ("constant", "cosine")


@dataclass(frozen=True)
class SegmentationTrainingConfig:
    manifest_path: str = DEFAULT_MANIFEST_PATH
    checkpoint_path: str = DEFAULT_SEGMENTATION_CHECKPOINT_PATH
    model_name: str = VAE_ASPP_UNET_MODEL_NAME
    split: str = "train"
    validation_split: str = ""
    validation_holdout_patterns: str = ""
    validation_fraction: float = 0.2
    max_train_samples: int = 0
    max_validation_samples: int = 128
    required_split_counts: dict[str, int] = field(default_factory=dict)
    steps: int = 2000
    batch_size: int = 8
    image_size: int = 128
    context_size: int = 0
    vae_checkpoint_path: str = DEFAULT_SEGMENTATION_VAE_CHECKPOINT_PATH
    vae_normalize_latents: bool = True
    learning_rate: float = 3e-4
    min_learning_rate: float = 0.0
    lr_schedule: str = "constant"
    weight_decay: float = 1e-4
    dice_weight: float = 0.5
    focal_weight: float = 0.0
    focal_gamma: float = 2.0
    lovasz_weight: float = 0.0
    lovasz_max_pixels: int = 0
    class_weight_power: float = 0.5
    class_weight_multipliers: dict[str, float] = field(default_factory=dict)
    unlabeled_weight: float = 0.25
    eval_every: int = 250
    evaluation_stride: int = 0
    evaluation_feather_blending: bool = False
    log_every: int = 0
    device: str = "cpu"
    seed: int = 2026
    cache_samples: int = 128
    augment: bool = True
    flip_probability: float = 0.5
    rotate_probability: float = 0.75
    color_jitter_strength: float = 0.15
    blur_probability: float = 0.1
    crop_scale_min: float = 1.0
    crop_scale_max: float = 1.0
    class_balanced_crop_probability: float = 0.0
    class_balanced_crop_weights: dict[str, float] = field(default_factory=dict)
    class_balanced_crop_min_pixels: int = 16

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
    class_logit_biases: dict[str, float] = field(default_factory=dict)
    manifest_path: str = DEFAULT_MANIFEST_PATH
    split: str = "train"
    max_samples: int = 128
    tile_stride: int = 0
    tile_scales: tuple[float, ...] = ()
    feather_blending: bool = False
    confidence_threshold: float = 0.0
    output_path: str = ""
    device: str = "cpu"
    cache_samples: int = 64


@dataclass(frozen=True)
class SegmentationPredictionConfig:
    checkpoint_path: str = DEFAULT_SEGMENTATION_CHECKPOINT_PATH
    class_logit_biases: dict[str, float] = field(default_factory=dict)
    output_dir: str = DEFAULT_SEGMENTATION_PREDICTION_DIR
    images_manifest_path: str = ""
    images_dir: str = ""
    max_images: int = 0
    tile_stride: int = DEFAULT_SEGMENTATION_PREDICTION_TILE_STRIDE
    tile_scales: tuple[float, ...] = ()
    feather_blending: bool = False
    device: str = "cpu"
    confidence_threshold: float = 0.0
    save_confidence_masks: bool = True
    save_color_masks: bool = True


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
    class_logit_biases: dict[str, float]
    confidence_threshold: float
    feather_blending: bool
    samples: int
    tiles: int
    mean_confidence: float
    mean_iou: float
    pixel_accuracy: float
    labeled_pixel_accuracy: float
    selected_mean_confidence: float
    selected_pixels: int
    selected_labeled_pixels: int
    selected_pixel_coverage: float
    selected_labeled_pixel_coverage: float
    selected_mean_iou: float
    selected_pixel_accuracy: float
    selected_labeled_pixel_accuracy: float
    selected_class_iou: dict[str, float | None]
    class_iou: dict[str, float | None]


@dataclass(frozen=True)
class SegmentationPredictionResult:
    checkpoint_path: str
    class_logit_biases: dict[str, float]
    confidence_threshold: float
    feather_blending: bool
    output_dir: str
    manifest_path: str
    images: int


def _validate_training_config(config: SegmentationTrainingConfig) -> None:
    if config.model_name not in SUPPORTED_SEGMENTATION_MODEL_NAMES:
        raise ValueError(
            "model_name must be one of: " + ", ".join(SUPPORTED_SEGMENTATION_MODEL_NAMES)
        )
    for name in (
        "evaluation_feather_blending",
    ):
        if not isinstance(getattr(config, name), bool):
            raise ValueError(f"{name} must be a boolean")
    if config.validation_fraction <= 0.0 or config.validation_fraction >= 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    for name in ("steps", "batch_size", "image_size", "eval_every"):
        if getattr(config, name) < 1:
            raise ValueError(f"{name} must be positive")
    for name in (
        "max_train_samples",
        "max_validation_samples",
        "cache_samples",
        "context_size",
        "evaluation_stride",
        "log_every",
        "lovasz_max_pixels",
    ):
        if getattr(config, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if config.context_size > 0 and config.context_size <= config.image_size:
        raise ValueError("context_size must be 0 or greater than image_size")
    if config.lr_schedule not in SEGMENTATION_LR_SCHEDULES:
        raise ValueError("lr_schedule must be one of: " + ", ".join(SEGMENTATION_LR_SCHEDULES))
    for name in ("learning_rate",):
        if getattr(config, name) <= 0.0:
            raise ValueError(f"{name} must be positive")
    if config.min_learning_rate < 0.0:
        raise ValueError("min_learning_rate must be non-negative")
    if config.min_learning_rate > config.learning_rate:
        raise ValueError("min_learning_rate must be less than or equal to learning_rate")
    for name in (
        "weight_decay",
        "dice_weight",
        "focal_weight",
        "focal_gamma",
        "lovasz_weight",
        "class_weight_power",
        "unlabeled_weight",
    ):
        if getattr(config, name) < 0.0:
            raise ValueError(f"{name} must be non-negative")
    _validate_class_weight_mapping(config.class_weight_multipliers, "class_weight_multipliers")
    _validate_class_weight_mapping(
        config.class_balanced_crop_weights,
        "class_balanced_crop_weights",
    )
    _validate_required_split_counts(config.required_split_counts)
    for name in ("flip_probability", "rotate_probability", "blur_probability"):
        value = getattr(config, name)
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if config.class_balanced_crop_probability < 0.0 or config.class_balanced_crop_probability > 1.0:
        raise ValueError("class_balanced_crop_probability must be between 0 and 1")
    if config.color_jitter_strength < 0.0:
        raise ValueError("color_jitter_strength must be non-negative")
    if config.crop_scale_min <= 0.0 or config.crop_scale_max <= 0.0:
        raise ValueError("crop scale bounds must be positive")
    if config.crop_scale_min > config.crop_scale_max:
        raise ValueError("crop_scale_min must be less than or equal to crop_scale_max")
    if config.class_balanced_crop_min_pixels < 1:
        raise ValueError("class_balanced_crop_min_pixels must be positive")
    if not config.manifest_path:
        raise ValueError("manifest_path is required")
    if not config.checkpoint_path:
        raise ValueError("checkpoint_path is required")
    if segmentation_model_uses_vae_features(config.model_name) and not config.vae_checkpoint_path:
        raise ValueError("vae_checkpoint_path is required")


def _validate_class_weight_mapping(value: object, name: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object mapping OpenEarthMap classes to weights")
    from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES

    valid_class_names = set(OPENEARTHMAP_CLASSES)
    for class_name, weight in value.items():
        if not isinstance(class_name, str) or class_name not in valid_class_names:
            raise ValueError(f"{name} contains unknown OpenEarthMap class name: {class_name!r}")
        if not isinstance(weight, int | float) or isinstance(weight, bool):
            raise ValueError(f"{name}[{class_name!r}] must be numeric")
        if float(weight) < 0.0:
            raise ValueError(f"{name}[{class_name!r}] must be non-negative")


def _validate_required_split_counts(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("required_split_counts must be an object mapping split names to counts")
    for split_name, count in value.items():
        if not isinstance(split_name, str) or not split_name.strip():
            raise ValueError("required_split_counts contains an invalid split name")
        if not isinstance(count, int) or isinstance(count, bool):
            raise ValueError(f"required_split_counts[{split_name!r}] must be an integer")
        if count < 0:
            raise ValueError(f"required_split_counts[{split_name!r}] must be non-negative")


def _validate_prediction_config(config: SegmentationPredictionConfig) -> None:
    if not config.checkpoint_path:
        raise ValueError("checkpoint_path is required")
    if not config.output_dir:
        raise ValueError("output_dir is required")
    if bool(config.images_manifest_path) == bool(config.images_dir):
        raise ValueError("pass exactly one of images_manifest_path or images_dir")
    if config.max_images < 0:
        raise ValueError("max_images must be non-negative")
    if config.tile_stride < 0:
        raise ValueError("tile_stride must be non-negative")
    if not isinstance(config.save_confidence_masks, bool):
        raise ValueError("save_confidence_masks must be a boolean")
    if not isinstance(config.save_color_masks, bool):
        raise ValueError("save_color_masks must be a boolean")
    _validate_confidence_threshold(config.confidence_threshold)
    _validate_class_logit_biases(config.class_logit_biases or {})


def _validate_evaluation_config(config: SegmentationEvaluationConfig) -> None:
    if not config.checkpoint_path:
        raise ValueError("checkpoint_path is required")
    if config.max_samples < 0:
        raise ValueError("max_samples must be non-negative")
    if config.tile_stride < 0:
        raise ValueError("tile_stride must be non-negative")
    _validate_confidence_threshold(config.confidence_threshold)
    _validate_class_logit_biases(config.class_logit_biases or {})


def _validate_confidence_threshold(value: float) -> None:
    if value < 0.0 or value > 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")


def _validate_class_logit_biases(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("class_logit_biases must be an object mapping class names to biases")
    from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASSES

    valid_class_names = set(OPENEARTHMAP_CLASSES)
    for class_name, bias in value.items():
        if not isinstance(class_name, str) or class_name not in valid_class_names:
            raise ValueError(f"class_logit_biases contains unknown class name: {class_name!r}")
        if not isinstance(bias, int | float) or isinstance(bias, bool):
            raise ValueError(f"class_logit_biases[{class_name!r}] must be numeric")


def load_segmentation_training_config_file(config_path: Path) -> SegmentationTrainingConfig:
    values = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError("Segmentation training config JSON must be an object")
    return SegmentationTrainingConfig.from_dict(values)


def segmentation_model_uses_vae_features(model_name: str) -> bool:
    return model_name in (*LEGACY_SEGMENTATION_MODEL_NAMES, VAE_ASPP_UNET_MODEL_NAME)
