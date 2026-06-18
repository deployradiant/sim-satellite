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
    device: str = "cuda"
    image_size: int = 32
    image_channels: int = 3
    mask_channels: int = len(OPENEARTHMAP_CLASS_IDS)
    hidden_channels: int = 32
    mask_drop_probability: float = 0.0
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

    @property
    def checkpoint_file(self) -> Path:
        return Path(self.checkpoint_path)

    @property
    def metrics_file(self) -> Path:
        checkpoint_file = self.checkpoint_file
        return (
            Path(self.metrics_path)
            if self.metrics_path
            else checkpoint_file.with_suffix(".metrics.json")
        )

    @property
    def loss_plot_file(self) -> Path:
        checkpoint_file = self.checkpoint_file
        return (
            Path(self.loss_plot_path)
            if self.loss_plot_path
            else checkpoint_file.with_suffix(".loss.png")
        )

    @property
    def resume_checkpoint_file(self) -> Path | None:
        return Path(self.resume_from) if self.resume_from else None

    @property
    def validation_enabled(self) -> bool:
        return self.validation_every != 0

    def should_sample_step_timing(self, step_number: int) -> bool:
        return self._matches_endpoint_interval(self.timing_sample_every, step_number)

    def should_log_color_stats(self, step_number: int) -> bool:
        return self._matches_endpoint_interval(self.color_stats_every, step_number)

    def should_log_training_step(self, step_number: int) -> bool:
        return self._matches_endpoint_interval(self.log_every, step_number)

    def should_write_step_artifacts(self, step_number: int) -> bool:
        return self._matches_interval(self.save_every, step_number) or self._matches_interval(
            self.plot_every,
            step_number,
        )

    def should_validate_step(self, step_number: int) -> bool:
        return self._matches_interval(self.validation_every, step_number)

    def _matches_endpoint_interval(self, interval: int, step_number: int) -> bool:
        return interval != 0 and (
            step_number == 1 or step_number == self.steps or step_number % interval == 0
        )

    def _matches_interval(self, interval: int, step_number: int) -> bool:
        return interval != 0 and step_number % interval == 0


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
    _require_positive(
        {
            "steps": config.steps,
            "batch_size": config.batch_size,
            "log_every": config.log_every,
            "validation_batches": config.validation_batches,
            "learning_rate": config.learning_rate,
        }
    )
    _require_non_negative(
        {
            "save_every": config.save_every,
            "plot_every": config.plot_every,
            "validation_every": config.validation_every,
            "timing_sample_every": config.timing_sample_every,
            "color_stats_every": config.color_stats_every,
        }
    )
    _require_probability({"mask_drop_probability": config.mask_drop_probability})
    _require_choice("precision", config.precision, {"float32", "bf16"})
    _require_cuda_device(config.device)
    _require_present(
        {
            "manifest_path": config.manifest_path,
            "split": config.split,
        }
    )


def _require_positive(values: Mapping[str, int | float]) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive")


def _require_non_negative(values: Mapping[str, int | float]) -> None:
    for name, value in values.items():
        if value < 0:
            raise ValueError(f"{name} must be non-negative")


def _require_probability(values: Mapping[str, float]) -> None:
    for name, value in values.items():
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be between 0 and 1")


def _require_choice(name: str, value: str, choices: set[str]) -> None:
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"{name} must be one of: {expected}")


def _require_cuda_device(value: str) -> None:
    if value != "cuda" and not value.startswith("cuda:"):
        raise ValueError("device must be a CUDA device")


def _require_present(values: Mapping[str, str]) -> None:
    for name, value in values.items():
        if not value:
            raise ValueError(f"{name} is required")
