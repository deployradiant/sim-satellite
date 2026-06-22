"""Training orchestration for the minimal satellite diffusion scaffold."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import click
import torch

from sim_satellite import diagnostics, evaluation_metrics, training_data, training_log
from sim_satellite.data.loader import SatelliteDataset, SatelliteSample, iter_batches
from sim_satellite.data.prepared import PreparedSample
from sim_satellite.generation import generate_conditioned_images
from sim_satellite.model import (
    ModelConfig,
    build_model,
    flow_velocity,
    interpolate_flow_sample,
)
from sim_satellite.training_config import TrainingConfig, load_training_config_file

DEFAULT_TARGET = "modal"
TIMESTEP_SAMPLING = "logit_normal_uniform_mix"
TIMESTEP_LOGIT_NORMAL_MEAN = 0.0
TIMESTEP_LOGIT_NORMAL_STD = 1.0
TIMESTEP_UNIFORM_MIX_PROBABILITY = 0.5
TRAINING_TASK_BASE = "base"


@dataclass(frozen=True)
class TrainingResult:
    checkpoint_path: str
    final_loss: float
    steps: int
    device: str
    model_parameters: int
    metrics_path: str
    loss_plot_path: str
    generation_metrics: dict[str, Any]


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
    losses: list[training_log.TrainingLossRecord] = []
    validation_losses: list[training_log.ValidationLossRecord] = []
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
        timestep_uniform_mix_probability=TIMESTEP_UNIFORM_MIX_PROBABILITY,
        precision=config.precision,
        preload_sample_count=len(preloaded_samples),
        preload_seconds=preload_seconds,
        timing_sample_every=config.timing_sample_every,
        color_stats_every=config.color_stats_every,
        generation_metrics_samples=config.generation_metrics_samples,
        generation_metrics_steps=config.generation_metrics_steps,
        generation_metrics_guidance_scale=config.generation_metrics_guidance_scale,
        model_parameters=model_parameters,
        image_size=config.image_size,
        device=device,
        torch_device_payload=training_log.torch_device_payload(torch, device),
    )

    model.train()
    for step in range(start_step, config.steps):
        step_number = step + 1
        step_started_at = time.monotonic()
        should_log_color_stats = _should_log_color_stats(config, step_number)
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
            profile_timing=_should_sample_step_timing(config, step_number),
            log_color_stats=should_log_color_stats,
        )
        step_seconds = max(0.0, time.monotonic() - step_started_at)
        should_log = (
            should_log_color_stats
            or step_number == 1
            or step_number == config.steps
            or step_number % config.log_every == 0
        )
        should_write_artifacts = (
            config.save_every > 0 and step_number % config.save_every == 0
        ) or (config.plot_every > 0 and step_number % config.plot_every == 0)
        loss_record: training_log.TrainingLossRecord | None = None
        if should_log or should_write_artifacts or step_timing:
            final_loss = training_log.loss_scalar(loss)
            loss_record = training_log.loss_record(
                step=step_number,
                loss=final_loss,
                step_seconds=step_seconds,
                batch_size=config.batch_size,
                step_timing=step_timing,
                task=TRAINING_TASK_BASE,
            )
            losses.append(loss_record)

        if (
            validation_batches
            and config.validation_every > 0
            and step_number % config.validation_every == 0
        ):
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
            validation_record: training_log.ValidationLossRecord = {
                "step": step_number,
                "loss": validation_loss,
                "task": TRAINING_TASK_BASE,
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
                task=TRAINING_TASK_BASE,
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
                    task=TRAINING_TASK_BASE,
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
                    task=TRAINING_TASK_BASE,
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
                generation_metrics=evaluation_metrics.disabled_generation_metrics_payload(),
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
        losses.append(
            {
                "step": final_step,
                "loss": final_loss,
                "task": TRAINING_TASK_BASE,
            }
        )
    generation_metrics = _evaluate_final_generation_metrics(
        model,
        validation_batches,
        torch=torch,
        model_config=model_config,
        device=device,
        config=config,
        preloaded_samples=preloaded_samples,
    )
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
        generation_metrics=generation_metrics,
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
        generation_metrics=generation_metrics,
    )


def _training_device(config: TrainingConfig, *, torch: Any) -> Any:
    device = torch.device(config.device)
    if not torch.cuda.is_available():
        raise RuntimeError("training requires a CUDA-capable GPU")
    return device


def _should_sample_step_timing(config: TrainingConfig, step_number: int) -> bool:
    return config.timing_sample_every > 0 and (
        step_number == 1
        or step_number == config.steps
        or step_number % config.timing_sample_every == 0
    )


def _should_log_color_stats(config: TrainingConfig, step_number: int) -> bool:
    return config.color_stats_every > 0 and (
        step_number == 1
        or step_number == config.steps
        or step_number % config.color_stats_every == 0
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
        prediction = model(noisy, semantic_mask, timestep)
    loss = _flow_prediction_loss(prediction, target, torch=torch)
    loss.backward()
    optimizer.step()
    _synchronize_device(torch, device, profile_timing)
    if profile_timing:
        timing["train_compute_seconds"] = max(0.0, time.monotonic() - compute_started_at)
        timing["timing_sampled"] = 1
    return loss.detach(), timing


def _flow_prediction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    torch: ModuleType,
) -> torch.Tensor:
    return torch.nn.functional.mse_loss(prediction.float(), target.float())


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
) -> tuple[
    int,
    float,
    list[training_log.TrainingLossRecord],
    list[training_log.ValidationLossRecord],
]:
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
    losses = cast(list[training_log.TrainingLossRecord], list(checkpoint.get("losses", [])))
    validation_losses = cast(
        list[training_log.ValidationLossRecord],
        list(checkpoint.get("validation_losses", [])),
    )
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
        return _limit_training_samples(dataset.samples, config.max_train_samples)

    samples = tuple(
        sample for sample in dataset.samples if not _sample_matches_any(sample, patterns)
    )
    if not samples:
        raise RuntimeError(
            "No training samples remain after applying validation holdout patterns: "
            f"{', '.join(patterns)}"
        )
    return _limit_training_samples(samples, config.max_train_samples)


def _limit_training_samples(
    samples: tuple[SatelliteSample, ...],
    max_train_samples: int,
) -> tuple[SatelliteSample, ...]:
    if max_train_samples <= 0 or len(samples) <= max_train_samples:
        return samples
    return samples[:max_train_samples]


def _load_validation_batches(config: TrainingConfig) -> tuple[tuple[SatelliteSample, ...], ...]:
    if config.validation_every <= 0 and config.generation_metrics_samples <= 0:
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
                "validation_every or generation_metrics_samples requires "
                "validation_split or validation_holdout_patterns"
            )
        dataset = SatelliteDataset(manifest_path, split=config.split, shuffle_seed=config.seed + 1)
        samples = tuple(
            sample for sample in dataset.samples if _sample_matches_any(sample, patterns)
        )

    if not samples:
        raise RuntimeError(f"No validation samples found in {manifest_path}")
    batches = tuple(iter_batches(samples, config.batch_size))
    return batches[: _validation_batch_limit(config)]


def _validation_batch_limit(config: TrainingConfig) -> int:
    limit = 0
    if config.validation_every > 0:
        limit = max(limit, config.validation_batches)
    if config.generation_metrics_samples > 0:
        limit = max(limit, math.ceil(config.generation_metrics_samples / config.batch_size))
    return limit


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
            clean, semantic_mask = training_data.load_batch(
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
                prediction = model(noisy, semantic_mask, timestep)
            loss = _flow_prediction_loss(prediction, target, torch=torch)
            losses.append(loss.detach())

    if was_training:
        model.train()
    color_stats = {}
    if color_batches:
        color_stats = diagnostics.color_stats_payload(
            diagnostics.image_batch_color_stats(torch.cat(color_batches, dim=0))
        )
    return training_log.loss_scalar(torch.stack(losses).mean()), color_stats


def _evaluate_final_generation_metrics(
    model: Any,
    validation_batches: tuple[tuple[SatelliteSample, ...], ...],
    *,
    torch: Any,
    model_config: ModelConfig,
    device: Any,
    config: TrainingConfig,
    preloaded_samples: Mapping[SatelliteSample, PreparedSample],
) -> dict[str, Any]:
    if config.generation_metrics_samples <= 0:
        return evaluation_metrics.disabled_generation_metrics_payload()

    if not validation_batches:
        raise RuntimeError("generation_metrics_samples requires validation samples")

    training_log.print_generation_metrics_start(
        sample_count=config.generation_metrics_samples,
        generation_steps=config.generation_metrics_steps,
        guidance_scale=config.generation_metrics_guidance_scale,
    )
    accumulator = evaluation_metrics.GenerationMetricAccumulator()
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch_index, samples in enumerate(validation_batches):
            remaining = config.generation_metrics_samples - accumulator.sample_count
            if remaining <= 0:
                break
            samples = samples[:remaining]
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

            generated = generate_conditioned_images(
                model,
                semantic_mask,
                torch=torch,
                model_config=model_config,
                device=device,
                seed=config.seed + 20_000 + batch_index,
                steps=config.generation_metrics_steps,
                guidance_scale=config.generation_metrics_guidance_scale,
                precision=config.precision,
            )
            real = (clean.float().clamp(-1.0, 1.0) + 1.0) / 2.0
            accumulator.update(generated, real, semantic_mask=semantic_mask, torch=torch)

    if was_training:
        model.train()

    payload = accumulator.payload(
        requested_sample_count=config.generation_metrics_samples,
        generation_steps=config.generation_metrics_steps,
        guidance_scale=config.generation_metrics_guidance_scale,
    )
    training_log.print_generation_metrics_done(payload)
    return payload


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
        config = load_training_config_file(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise click.ClickException(str(error)) from error

    click.echo(json.dumps(_run_with_target(config, target), indent=2))


if __name__ == "__main__":
    cli()
