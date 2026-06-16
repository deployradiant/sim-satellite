"""Training orchestration for the minimal satellite diffusion scaffold."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import click
import torch

from sim_satellite import training_data, training_log
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
TIMESTEP_SAMPLING = "logit_normal"
TIMESTEP_LOGIT_NORMAL_MEAN = 0.0
TIMESTEP_LOGIT_NORMAL_STD = 1.0


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
    chroma_loss_weight: float = 0.05
    precision: str = "float32"
    timing_sample_every: int = 0
    log_every: int = 10

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.log_every < 1:
            raise ValueError("log_every must be positive")
        if self.save_every < 0:
            raise ValueError("save_every must be non-negative")
        if self.plot_every < 0:
            raise ValueError("plot_every must be non-negative")
        if self.validation_every < 0:
            raise ValueError("validation_every must be non-negative")
        if self.validation_batches < 1:
            raise ValueError("validation_batches must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.mask_drop_probability < 0.0 or self.mask_drop_probability > 1.0:
            raise ValueError("mask_drop_probability must be between 0 and 1")
        if self.chroma_loss_weight < 0.0:
            raise ValueError("chroma_loss_weight must be non-negative")
        if self.precision not in {"float32", "bf16"}:
            raise ValueError("precision must be 'float32' or 'bf16'")
        if self.device != "cuda" and not self.device.startswith("cuda:"):
            raise ValueError("device must be a CUDA device")
        if self.timing_sample_every < 0:
            raise ValueError("timing_sample_every must be non-negative")
        if not self.manifest_path:
            raise ValueError("manifest_path is required")
        if not self.split:
            raise ValueError("split is required")

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
    if config.resume_from:
        start_step, final_loss, losses, validation_losses = _load_training_state(
            Path(config.resume_from),
            model=model,
            optimizer=optimizer,
            generator=crop_generator,
            device_generator=random_generator,
            torch=torch,
            model_config=model_config,
            device=device,
        )

    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = (
        Path(config.metrics_path)
        if config.metrics_path
        else checkpoint_path.with_suffix(".metrics.json")
    )
    loss_plot_path = (
        Path(config.loss_plot_path)
        if config.loss_plot_path
        else checkpoint_path.with_suffix(".loss.png")
    )

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
        precision=config.precision,
        preload_sample_count=len(preloaded_samples),
        preload_seconds=preload_seconds,
        timing_sample_every=config.timing_sample_every,
        model_parameters=model_parameters,
        image_size=config.image_size,
        device=device,
        torch_device_payload=training_log.torch_device_payload(torch, device),
    )

    model.train()
    for step in range(start_step, config.steps):
        step_number = step + 1
        step_started_at = time.monotonic()
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
            chroma_loss_weight=config.chroma_loss_weight,
            precision=config.precision,
            profile_timing=_should_sample_step_timing(config, step_number),
        )
        step_seconds = max(0.0, time.monotonic() - step_started_at)
        should_log = (
            step_number == 1 or step_number == config.steps or step_number % config.log_every == 0
        )
        should_write_artifacts = (
            config.save_every > 0 and step_number % config.save_every == 0
        ) or (config.plot_every > 0 and step_number % config.plot_every == 0)
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

        if (
            validation_batches
            and config.validation_every > 0
            and step_number % config.validation_every == 0
        ):
            validation_loss = _evaluate_validation_loss(
                model,
                validation_batches,
                torch=torch,
                model_config=model_config,
                device=device,
                config=config,
                preloaded_samples=preloaded_samples,
            )
            validation_losses.append({"step": step_number, "loss": validation_loss})
            training_log.print_validation_progress(
                started_at_epoch_utc=started_at_epoch_utc,
                started_at_monotonic=started_at_monotonic,
                start_step=start_step,
                step=step_number,
                steps=config.steps,
                batch_size=config.batch_size,
                loss=validation_loss,
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
    if not losses and config.resume_from:
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
    if device.type != "cuda":
        raise ValueError("training requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("training requires a CUDA-capable GPU")
    return device


def _should_sample_step_timing(config: TrainingConfig, step_number: int) -> bool:
    return config.timing_sample_every > 0 and (
        step_number == 1
        or step_number == config.steps
        or step_number % config.timing_sample_every == 0
    )


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
    return torch.sigmoid(logits)


def _flow_training_loss(
    prediction: Any,
    target: Any,
    noisy: Any,
    clean: Any,
    timestep: Any,
    *,
    chroma_loss_weight: float,
    torch: Any,
) -> Any:
    loss = torch.nn.functional.mse_loss(prediction.float(), target.float())
    if chroma_loss_weight <= 0.0:
        return loss

    predicted_clean = noisy.float() - timestep.float().view(-1, 1, 1, 1) * prediction.float()
    return loss + chroma_loss_weight * _chroma_reconstruction_loss(
        predicted_clean,
        clean.float(),
        torch=torch,
    )


def _chroma_reconstruction_loss(predicted_clean: Any, clean: Any, *, torch: Any) -> Any:
    predicted_chroma = predicted_clean - predicted_clean.mean(dim=1, keepdim=True)
    clean_chroma = clean - clean.mean(dim=1, keepdim=True)
    return torch.nn.functional.mse_loss(predicted_chroma, clean_chroma)


def _train_step(
    samples: tuple[SatelliteSample, ...],
    *,
    model: Any,
    optimizer: Any,
    torch: Any,
    model_config: ModelConfig,
    device: Any,
    crop_generator: Any,
    random_generator: Any,
    preloaded_samples: Mapping[SatelliteSample, PreparedSample],
    mask_drop_probability: float,
    chroma_loss_weight: float,
    precision: str,
    profile_timing: bool,
) -> tuple[Any, dict[str, float | int]]:
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
    clean, semantic_mask = training_data.load_batch(
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
        prediction = model(noisy, semantic_mask, timestep)
    loss = _flow_training_loss(
        prediction,
        target,
        noisy,
        clean,
        timestep,
        chroma_loss_weight=chroma_loss_weight,
        torch=torch,
    )
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
    if config.validation_every <= 0:
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
) -> float:
    was_training = model.training
    model.eval()
    generator = torch.Generator(device=device).manual_seed(config.seed + 10_000)
    losses: list[Any] = []
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
            clean, semantic_mask = training_data.load_batch(
                samples,
                crop_boxes,
                torch=torch,
                model_config=model_config,
                device=device,
                preloaded_samples=preloaded_samples,
            )
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
                prediction = model(noisy, semantic_mask, timestep)
            loss = _flow_training_loss(
                prediction,
                target,
                noisy,
                clean,
                timestep,
                chroma_loss_weight=config.chroma_loss_weight,
                torch=torch,
            )
            losses.append(loss.detach())

    if was_training:
        model.train()
    return training_log.loss_scalar(torch.stack(losses).mean())


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


def _load_training_config(config_path: Path) -> TrainingConfig:
    values = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(values, Mapping):
        raise ValueError("Training config JSON must be an object")
    return TrainingConfig.from_dict(values)


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
