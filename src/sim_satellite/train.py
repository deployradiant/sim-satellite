"""Training orchestration for the minimal satellite diffusion scaffold."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from types import ModuleType
from typing import Any

import click
import torch

from sim_satellite import diagnostics, training_data, training_log
from sim_satellite.data.loader import SatelliteDataset, SatelliteSample, iter_batches
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASS_IDS
from sim_satellite.data.prepared import PreparedSample
from sim_satellite.model import (
    ModelConfig,
    build_model,
    flow_velocity,
    interpolate_flow_sample,
)

DEFAULT_TARGET = "modal"
DEFAULT_MANIFEST_PATH = "data/openearthmap/manifest.jsonl"
TIMESTEP_SAMPLING = "logit_normal_uniform_mix"
TIMESTEP_LOGIT_NORMAL_MEAN = 0.0
TIMESTEP_LOGIT_NORMAL_STD = 1.0
TIMESTEP_UNIFORM_MIX_PROBABILITY = 0.5


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


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: str
    final_loss: float
    steps: int
    device: str
    model_parameters: int
    metrics_path: str
    loss_plot_path: str


def run_training(
    config: TrainingConfig | None = None,
    *,
    after_save: Callable[[], None] | None = None,
) -> TrainingResult:
    config = config or TrainingConfig()
    started_at_epoch_utc = time.time()
    started_at_monotonic = time.monotonic()

    torch.manual_seed(config.seed)
    device = _training_device(config, torch=torch)
    model_config = config.model_config()
    model = build_model(model_config).to(device)
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    crop_generator = torch.Generator(device="cpu").manual_seed(config.seed)
    random_generator = torch.Generator(device=device).manual_seed(config.seed)
    training_samples = _load_training_samples(config)
    batches = tuple(iter_batches(training_samples, config.batch_size))
    validation_batches = _load_validation_batches(config)
    preload_started_at = time.monotonic()
    samples_to_preload = training_data.unique_samples(training_samples, validation_batches)
    train_sample_count = sum(len(batch) for batch in batches)
    training_log.print_preload_start(
        manifest_path=config.manifest_path,
        split=config.split,
        sample_count=len(samples_to_preload),
        train_samples=train_sample_count,
        validation_batches=len(validation_batches),
    )

    def report_preload_progress(completed: int, total: int) -> None:
        training_log.print_preload_progress(
            started_at_monotonic=preload_started_at,
            completed=completed,
            total=total,
        )

    preloaded_samples = training_data.preload_samples(
        samples_to_preload,
        progress=report_preload_progress,
    )
    preload_seconds = max(0.0, time.monotonic() - preload_started_at)

    final_loss = 0.0
    losses: list[dict[str, float | int]] = []
    validation_losses: list[dict[str, float | int]] = []
    start_step = 0
    resume_checkpoint_file = config.resume_checkpoint_file
    if resume_checkpoint_file is not None:
        start_step, final_loss, losses, validation_losses = _load_training_state(
            resume_checkpoint_file,
            model=model,
            optimizer=optimizer,
            generator=crop_generator,
            device_generator=random_generator,
            torch=torch,
            model_config=model_config,
            device=device,
        )

    checkpoint_path = config.checkpoint_file
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = config.metrics_file
    loss_plot_path = config.loss_plot_file

    training_log.print_train_start(
        started_at_epoch_utc=started_at_epoch_utc,
        start_step=start_step,
        steps=config.steps,
        train_samples=train_sample_count,
        train_batches=len(batches),
        batch_size=config.batch_size,
        timestep_sampling=TIMESTEP_SAMPLING,
        timestep_logit_normal_mean=TIMESTEP_LOGIT_NORMAL_MEAN,
        timestep_logit_normal_std=TIMESTEP_LOGIT_NORMAL_STD,
        timestep_uniform_mix_probability=TIMESTEP_UNIFORM_MIX_PROBABILITY,
        precision=config.precision,
        preload_sample_count=len(preloaded_samples),
        preload_seconds=preload_seconds,
        timing_sample_every=config.timing_sample_every,
        color_stats_every=config.color_stats_every,
        model_parameters=model_parameters,
        image_size=config.image_size,
        device=device,
        torch_device_payload=training_log.torch_device_payload(torch, device),
    )

    model.train()
    for step in range(start_step, config.steps):
        step_number = step + 1
        step_started_at = time.monotonic()
        should_log_color_stats = config.should_log_color_stats(step_number)
        loss, step_timing = _train_step(
            batches[step % len(batches)],
            model=model,
            optimizer=optimizer,
            torch=torch,
            model_config=model_config,
            device=device,
            crop_generator=crop_generator,
            random_generator=random_generator,
            preloaded_samples=preloaded_samples,
            mask_drop_probability=config.mask_drop_probability,
            precision=config.precision,
            profile_timing=config.should_sample_step_timing(step_number),
            log_color_stats=should_log_color_stats,
        )
        step_seconds = max(0.0, time.monotonic() - step_started_at)
        should_log = should_log_color_stats or config.should_log_training_step(step_number)
        should_write_artifacts = config.should_write_step_artifacts(step_number)
        loss_record = None
        if should_log or should_write_artifacts or step_timing:
            final_loss = training_log.loss_scalar(loss)
            loss_record = training_log.loss_record(
                step=step_number,
                loss=final_loss,
                step_seconds=step_seconds,
                batch_size=config.batch_size,
                step_timing=step_timing,
            )
            losses.append(loss_record)

        if validation_batches and config.should_validate_step(step_number):
            validation_loss, validation_color_stats = _evaluate_validation_loss(
                model,
                validation_batches,
                torch=torch,
                model_config=model_config,
                device=device,
                config=config,
                preloaded_samples=preloaded_samples,
                log_color_stats=should_log_color_stats,
            )
            validation_record: dict[str, float | int] = {
                "step": step_number,
                "loss": validation_loss,
            }
            validation_record.update(validation_color_stats)
            validation_losses.append(validation_record)
            training_log.print_validation_progress(
                started_at_epoch_utc=started_at_epoch_utc,
                started_at_monotonic=started_at_monotonic,
                start_step=start_step,
                step=step_number,
                steps=config.steps,
                batch_size=config.batch_size,
                loss=validation_loss,
                color_stats=validation_color_stats,
            )

        if should_log:
            if loss_record is None:
                final_loss = training_log.loss_scalar(loss)
                loss_record = training_log.loss_record(
                    step=step_number,
                    loss=final_loss,
                    step_seconds=step_seconds,
                    batch_size=config.batch_size,
                    step_timing=step_timing,
                )
                losses.append(loss_record)
            training_log.print_train_progress(
                started_at_epoch_utc=started_at_epoch_utc,
                started_at_monotonic=started_at_monotonic,
                start_step=start_step,
                step=step_number,
                steps=config.steps,
                batch_size=config.batch_size,
                loss=final_loss,
                loss_record=loss_record,
                step_timing=step_timing,
            )

        if should_write_artifacts:
            if loss_record is None:
                final_loss = training_log.loss_scalar(loss)
                loss_record = training_log.loss_record(
                    step=step_number,
                    loss=final_loss,
                    step_seconds=step_seconds,
                    batch_size=config.batch_size,
                    step_timing=step_timing,
                )
                losses.append(loss_record)
            training_log.write_training_artifacts(
                checkpoint_path.with_name(
                    f"{checkpoint_path.stem}-step-{step_number}{checkpoint_path.suffix}"
                ),
                metrics_path,
                loss_plot_path,
                model=model,
                optimizer=optimizer,
                generator=crop_generator,
                device_generator=random_generator,
                model_config=model_config,
                config=config,
                model_parameters=model_parameters,
                losses=losses,
                validation_losses=validation_losses,
                final_loss=final_loss,
                step=step_number,
                run_timing=training_log.run_timing_payload(
                    started_at_epoch_utc=started_at_epoch_utc,
                    started_at_monotonic=started_at_monotonic,
                    start_step=start_step,
                    step=step_number,
                    steps=config.steps,
                    batch_size=config.batch_size,
                ),
                torch=torch,
            )
            if after_save is not None:
                after_save()

    final_step = max(start_step, config.steps)
    if not losses and resume_checkpoint_file is not None:
        losses.append({"step": final_step, "loss": final_loss})
    training_log.write_training_artifacts(
        checkpoint_path,
        metrics_path,
        loss_plot_path,
        model=model,
        optimizer=optimizer,
        generator=crop_generator,
        device_generator=random_generator,
        model_config=model_config,
        config=config,
        model_parameters=model_parameters,
        losses=losses,
        validation_losses=validation_losses,
        final_loss=final_loss,
        step=final_step,
        run_timing=training_log.run_timing_payload(
            started_at_epoch_utc=started_at_epoch_utc,
            started_at_monotonic=started_at_monotonic,
            start_step=start_step,
            step=final_step,
            steps=config.steps,
            batch_size=config.batch_size,
        ),
        torch=torch,
    )
    if after_save is not None:
        after_save()

    return TrainingResult(
        checkpoint_path=str(checkpoint_path),
        final_loss=final_loss,
        steps=config.steps,
        device=str(device),
        model_parameters=model_parameters,
        metrics_path=str(metrics_path),
        loss_plot_path=str(loss_plot_path),
    )


def _training_device(config: TrainingConfig, *, torch: Any) -> Any:
    device = torch.device(config.device)
    if not torch.cuda.is_available():
        raise RuntimeError("training requires a CUDA-capable GPU")
    return device


def _autocast_context(torch: Any, precision: str) -> Any:
    if precision == "float32":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _synchronize_device(torch: Any, device: Any, enabled: bool) -> None:
    if not enabled:
        return
    torch.cuda.synchronize(device)


def _sample_flow_timesteps(
    batch_size: int,
    *,
    torch: Any,
    generator: Any,
    device: Any,
) -> Any:
    logits = torch.randn(batch_size, generator=generator, device=device)
    logits = logits * TIMESTEP_LOGIT_NORMAL_STD + TIMESTEP_LOGIT_NORMAL_MEAN
    logit_normal = torch.sigmoid(logits)
    uniform_timestep = torch.rand(batch_size, generator=generator, device=device)
    use_uniform = torch.rand(batch_size, generator=generator, device=device)
    return torch.where(
        use_uniform < TIMESTEP_UNIFORM_MIX_PROBABILITY,
        uniform_timestep,
        logit_normal,
    )


def _train_step(
    samples: tuple[SatelliteSample, ...],
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    torch: ModuleType,
    model_config: ModelConfig,
    device: torch.device,
    crop_generator: torch.Generator,
    random_generator: torch.Generator,
    preloaded_samples: Mapping[SatelliteSample, PreparedSample],
    mask_drop_probability: float,
    precision: str,
    profile_timing: bool,
    log_color_stats: bool,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    timing: dict[str, float | int] = {}

    _synchronize_device(torch, device, profile_timing)
    crop_started_at = time.monotonic()
    crop_boxes = [
        training_data.random_square_crop_box(
            sample,
            crop_size=model_config.image_size,
            torch=torch,
            generator=crop_generator,
            preloaded_samples=preloaded_samples,
        )
        for sample in samples
    ]
    _synchronize_device(torch, device, profile_timing)
    if profile_timing:
        timing["crop_seconds"] = max(0.0, time.monotonic() - crop_started_at)

    load_started_at = time.monotonic()
    clean, semantic_mask, coordinates = training_data.load_batch(
        samples,
        crop_boxes,
        torch=torch,
        model_config=model_config,
        device=device,
        preloaded_samples=preloaded_samples,
    )
    _synchronize_device(torch, device, profile_timing)
    if profile_timing:
        timing["load_seconds"] = max(0.0, time.monotonic() - load_started_at)
    if log_color_stats:
        timing.update(diagnostics.color_stats_payload(diagnostics.image_batch_color_stats(clean)))

    random_started_at = time.monotonic()
    noise = torch.randn(clean.shape, generator=random_generator, device=device)
    timestep = _sample_flow_timesteps(
        clean.shape[0],
        torch=torch,
        generator=random_generator,
        device=device,
    )
    noisy = interpolate_flow_sample(clean, noise, timestep)
    target = flow_velocity(clean, noise)

    if mask_drop_probability > 0.0:
        keep_mask = torch.rand(
            (semantic_mask.shape[0],),
            generator=random_generator,
            device=device,
        )
        keep_mask = (keep_mask >= mask_drop_probability).float().view(-1, 1, 1, 1)
        semantic_mask = semantic_mask * keep_mask
    _synchronize_device(torch, device, profile_timing)
    if profile_timing:
        timing["random_seconds"] = max(0.0, time.monotonic() - random_started_at)

    compute_started_at = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    with _autocast_context(torch, precision):
        prediction = model(noisy, semantic_mask, timestep, coordinates)
    loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
    loss.backward()
    optimizer.step()
    _synchronize_device(torch, device, profile_timing)
    if profile_timing:
        timing["train_compute_seconds"] = max(0.0, time.monotonic() - compute_started_at)
        timing["timing_sampled"] = 1
    return loss.detach(), timing


def _load_training_state(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    generator: Any,
    device_generator: Any | None,
    torch: Any,
    model_config: ModelConfig,
    device: Any,
) -> tuple[int, float, list[dict[str, float | int]], list[dict[str, float | int]]]:
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    checkpoint_model_config = checkpoint.get("model_config")
    if (
        not isinstance(checkpoint_model_config, Mapping)
        or ModelConfig.from_dict(checkpoint_model_config) != model_config
    ):
        raise ValueError("Resume checkpoint model_config does not match the requested model")
    model.load_state_dict(checkpoint["model_state"])
    optimizer_state = checkpoint.get("optimizer_state")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    generator_state = checkpoint.get("generator_state")
    if generator_state is not None:
        generator.set_state(generator_state)
    device_generator_state = checkpoint.get("device_generator_state")
    if device_generator is not None and device_generator_state is not None:
        device_generator.set_state(device_generator_state)
    step = int(checkpoint["step"])
    final_loss = float(checkpoint.get("final_loss", 0.0))
    losses = list(checkpoint.get("losses", []))
    validation_losses = list(checkpoint.get("validation_losses", []))
    return step, final_loss, losses, validation_losses


def _load_training_samples(config: TrainingConfig) -> tuple[SatelliteSample, ...]:
    manifest_path = Path(config.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Training manifest not found: {manifest_path}. Run `sim-satellite openearthmap` first."
        )

    dataset = SatelliteDataset(manifest_path, split=config.split, shuffle_seed=config.seed)
    if len(dataset) == 0:
        raise RuntimeError(f"No samples for split `{config.split}` in {manifest_path}")

    patterns = _validation_holdout_patterns(config)
    if not patterns:
        return dataset.samples

    samples = tuple(
        sample for sample in dataset.samples if not _sample_matches_any(sample, patterns)
    )
    if not samples:
        raise RuntimeError(
            "No training samples remain after applying validation holdout patterns: "
            f"{', '.join(patterns)}"
        )
    return samples


def _load_validation_batches(config: TrainingConfig) -> tuple[tuple[SatelliteSample, ...], ...]:
    if not config.validation_enabled:
        return ()

    manifest_path = Path(config.manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Training manifest not found: {manifest_path}. Run `sim-satellite openearthmap` first."
        )

    if config.validation_split:
        dataset = SatelliteDataset(
            manifest_path,
            split=config.validation_split,
            shuffle_seed=config.seed + 1,
        )
        samples = dataset.samples
    else:
        patterns = _validation_holdout_patterns(config)
        if not patterns:
            raise ValueError(
                "validation_every requires validation_split or validation_holdout_patterns"
            )
        dataset = SatelliteDataset(manifest_path, split=config.split, shuffle_seed=config.seed + 1)
        samples = tuple(
            sample for sample in dataset.samples if _sample_matches_any(sample, patterns)
        )

    if not samples:
        raise RuntimeError(f"No validation samples found in {manifest_path}")
    return tuple(iter_batches(samples, config.batch_size))


def _evaluate_validation_loss(
    model: Any,
    validation_batches: tuple[tuple[SatelliteSample, ...], ...],
    *,
    torch: Any,
    model_config: ModelConfig,
    device: Any,
    config: TrainingConfig,
    preloaded_samples: Mapping[SatelliteSample, PreparedSample],
    log_color_stats: bool,
) -> tuple[float, dict[str, float | int]]:
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(config.seed + 10_000)
    losses: list[Any] = []
    color_batches: list[Any] = []
    with torch.no_grad():
        for samples in validation_batches[: config.validation_batches]:
            crop_boxes = [
                training_data.center_square_crop_box(
                    sample,
                    crop_size=model_config.image_size,
                    preloaded_samples=preloaded_samples,
                )
                for sample in samples
            ]
            clean, semantic_mask, coordinates = training_data.load_batch(
                samples,
                crop_boxes,
                torch=torch,
                model_config=model_config,
                device=device,
                preloaded_samples=preloaded_samples,
            )
            if log_color_stats:
                color_batches.append(clean.detach().cpu())
            noise = torch.randn(clean.shape, generator=generator, device=device)
            timestep = _sample_flow_timesteps(
                clean.shape[0],
                torch=torch,
                generator=generator,
                device=device,
            )
            noisy = interpolate_flow_sample(clean, noise, timestep)
            target = flow_velocity(clean, noise)
            with _autocast_context(torch, config.precision):
                prediction = model(noisy, semantic_mask, timestep, coordinates)
            loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
            losses.append(loss.detach())

    if was_training:
        model.train()
    color_stats = {}
    if color_batches:
        color_stats = diagnostics.color_stats_payload(
            diagnostics.image_batch_color_stats(torch.cat(color_batches, dim=0))
        )
    return training_log.loss_scalar(torch.stack(losses).mean()), color_stats


def _validation_holdout_patterns(config: TrainingConfig) -> tuple[str, ...]:
    return tuple(
        pattern.strip()
        for pattern in config.validation_holdout_patterns.split(",")
        if pattern.strip()
    )


def _sample_matches_any(sample: SatelliteSample, patterns: tuple[str, ...]) -> bool:
    image_path = sample.image_path.as_posix()
    mask_path = sample.mask_path.as_posix()
    return any(pattern in image_path or pattern in mask_path for pattern in patterns)


def _run_with_target(config: TrainingConfig, target: str) -> dict[str, Any]:
    if target == "local":
        result = asdict(run_training(config))
        result["execution_target"] = "local"
        return result

    from sim_satellite.modal_app import run_training_target

    if target != "modal":
        raise ValueError(f"Unsupported training target: {target}")
    return run_training_target(config)


def load_training_config_values(values: Mapping[str, Any]) -> TrainingConfig:
    config = TrainingConfig.from_dict(values)
    _validate_training_config(config)
    return config


def _load_training_config(config_path: Path) -> TrainingConfig:
    values = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError("Training config JSON must be an object")
    return load_training_config_values(values)


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


@click.command("train", help="Run model training from a JSON config file.")
@click.argument(
    "config_path",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
)
@click.option(
    "--target",
    type=click.Choice(["modal", "local"]),
    default=DEFAULT_TARGET,
    show_default=True,
)
def cli(config_path: Path, target: str) -> None:
    try:
        config = _load_training_config(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(json.dumps(_run_with_target(config, target), indent=2))


if __name__ == "__main__":
    cli()
