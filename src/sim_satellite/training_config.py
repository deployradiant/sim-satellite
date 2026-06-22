"""Training configuration loading and validation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASS_IDS
from sim_satellite.model import ModelConfig

DEFAULT_MANIFEST_PATH = "data/openearthmap/manifest.jsonl"


@dataclass(frozen=True)
class TrainingConfig:
    seed: int = 2026
    steps: int = 8
    batch_size: int = 4
    learning_rate: float = 1e-3
    manifest_path: str = DEFAULT_MANIFEST_PATH
    split: str = "train"
    checkpoint_path: str = "artifacts/tiny-sim-satellite.pt"
    metrics_path: str = ""
    loss_plot_path: str = ""
    resume_from: str = ""
    save_every: int = 0
    plot_every: int = 0
    validation_split: str = ""
    validation_holdout_patterns: str = ""
    validation_every: int = 0
    validation_batches: int = 8
    generation_metrics_samples: int = 0
    generation_metrics_steps: int = 16
    generation_metrics_guidance_scale: float = 1.0
    device: str = "cuda"
    image_size: int = 32
    image_channels: int = 3
    mask_channels: int = len(OPENEARTHMAP_CLASS_IDS)
    hidden_channels: int = 64
    mask_drop_probability: float = 0.0
    max_train_samples: int = 0
    precision: str = "float32"
    timing_sample_every: int = 0
    color_stats_every: int = 0
    log_every: int = 10

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> TrainingConfig:
        names = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - names)
        if unknown:
            raise ValueError(f"Unknown training config fields: {', '.join(unknown)}")
        return cls(**{name: values[name] for name in names if name in values})

    def model_config(self) -> ModelConfig:
        return ModelConfig(
            image_size=self.image_size,
            image_channels=self.image_channels,
            mask_channels=self.mask_channels,
            hidden_channels=self.hidden_channels,
            seed=self.seed,
        )


def load_training_config_file(config_path: Path) -> TrainingConfig:
    values = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError("Training config JSON must be an object")
    return load_training_config_values(values)


def load_training_config_values(values: Mapping[str, Any]) -> TrainingConfig:
    config = TrainingConfig.from_dict(values)
    _validate_training_config(config)
    return config


def _validate_training_config(config: TrainingConfig) -> None:
    for name in (
        "steps",
        "batch_size",
        "log_every",
        "validation_batches",
        "generation_metrics_steps",
        "learning_rate",
    ):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")

    for name in (
        "save_every",
        "plot_every",
        "validation_every",
        "generation_metrics_samples",
        "timing_sample_every",
        "color_stats_every",
        "max_train_samples",
    ):
        if getattr(config, name) < 0:
            raise ValueError(f"{name} must be non-negative")

    if config.mask_drop_probability < 0.0 or config.mask_drop_probability > 1.0:
        raise ValueError("mask_drop_probability must be between 0 and 1")
    if config.generation_metrics_guidance_scale < 0.0:
        raise ValueError("generation_metrics_guidance_scale must be non-negative")
    if config.precision not in {"float32", "bf16"}:
        raise ValueError("precision must be one of: bf16, float32")
    if config.device != "cuda" and not config.device.startswith("cuda:"):
        raise ValueError("device must be a CUDA device")

    for name in ("manifest_path", "split"):
        if not getattr(config, name):
            raise ValueError(f"{name} is required")
