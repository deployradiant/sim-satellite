"""Training orchestration for the minimal satellite diffusion scaffold."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import click
import torch

from sim_satellite.data.loader import SatelliteDataset, SatelliteSample, iter_batches
from sim_satellite.data.openearthmap import OPENEARTHMAP_CLASS_IDS
from sim_satellite.data.tensors import load_image_tensor, load_mask_tensor
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
    seed: int = 1337
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
    device: str = "cpu"
    image_size: int = 32
    image_channels: int = 3
    mask_channels: int = len(OPENEARTHMAP_CLASS_IDS)
    hidden_channels: int = 32
    mask_drop_probability: float = 0.0
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
    device = torch.device(config.device)
    model_config = config.model_config()
    model = build_model(model_config).to(device)
    model_parameters = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    batches = tuple(iter_batches(_load_training_samples(config), config.batch_size))
    validation_batches = _load_validation_batches(config)

    final_loss = 0.0
    losses: list[dict[str, float | int]] = []
    validation_losses: list[dict[str, float | int]] = []
    start_step = 0
    if config.resume_from:
        start_step, final_loss, losses, validation_losses = _load_training_state(
            Path(config.resume_from),
            model=model,
            optimizer=optimizer,
            generator=generator,
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

    print(
        json.dumps(
            {
                "event": "train_start",
                "started_at_epoch_utc": started_at_epoch_utc,
                "start_step": start_step,
                "steps": config.steps,
                "remaining_steps": max(config.steps - start_step, 0),
                "train_samples": sum(len(batch) for batch in batches),
                "train_batches": len(batches),
                "batch_size": config.batch_size,
                "total_images_target": config.steps * config.batch_size,
                "timestep_sampling": TIMESTEP_SAMPLING,
                "timestep_logit_normal_mean": TIMESTEP_LOGIT_NORMAL_MEAN,
                "timestep_logit_normal_std": TIMESTEP_LOGIT_NORMAL_STD,
                "model_parameters": model_parameters,
                "image_size": config.image_size,
                "device": str(device),
                **_torch_device_payload(torch, device),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    model.train()
    for step in range(start_step, config.steps):
        step_number = step + 1
        final_loss = _train_step(
            batches[step % len(batches)],
            model=model,
            optimizer=optimizer,
            torch=torch,
            model_config=model_config,
            device=device,
            generator=generator,
            mask_drop_probability=config.mask_drop_probability,
        )
        losses.append({"step": step_number, "loss": final_loss})

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
            )
            validation_losses.append({"step": step_number, "loss": validation_loss})
            print(
                json.dumps(
                    {
                        "event": "validation_progress",
                        "step": step_number,
                        "steps": config.steps,
                        "loss": validation_loss,
                        **_run_timing_payload(
                            started_at_epoch_utc=started_at_epoch_utc,
                            started_at_monotonic=started_at_monotonic,
                            start_step=start_step,
                            step=step_number,
                            steps=config.steps,
                            batch_size=config.batch_size,
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if step_number == 1 or step_number == config.steps or step_number % config.log_every == 0:
            print(
                json.dumps(
                    {
                        "event": "train_progress",
                        "step": step_number,
                        "steps": config.steps,
                        "loss": final_loss,
                        **_run_timing_payload(
                            started_at_epoch_utc=started_at_epoch_utc,
                            started_at_monotonic=started_at_monotonic,
                            start_step=start_step,
                            step=step_number,
                            steps=config.steps,
                            batch_size=config.batch_size,
                        ),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        if (config.save_every > 0 and step_number % config.save_every == 0) or (
            config.plot_every > 0 and step_number % config.plot_every == 0
        ):
            progress_timing = _run_timing_payload(
                started_at_epoch_utc=started_at_epoch_utc,
                started_at_monotonic=started_at_monotonic,
                start_step=start_step,
                step=step_number,
                steps=config.steps,
                batch_size=config.batch_size,
            )
            _write_training_metrics(
                metrics_path,
                model_config=model_config,
                config=config,
                model_parameters=model_parameters,
                losses=losses,
                validation_losses=validation_losses,
                final_loss=final_loss,
                step=step_number,
                run_timing=progress_timing,
            )
            _write_loss_plot(loss_plot_path, losses, validation_losses)
            _write_training_checkpoint(
                checkpoint_path.with_name(
                    f"{checkpoint_path.stem}-step-{step_number}{checkpoint_path.suffix}"
                ),
                model=model,
                optimizer=optimizer,
                generator=generator,
                model_config=model_config,
                config=config,
                losses=losses,
                validation_losses=validation_losses,
                final_loss=final_loss,
                step=step_number,
                metrics_path=metrics_path,
                loss_plot_path=loss_plot_path,
                torch=torch,
            )
            if after_save is not None:
                after_save()

    final_step = max(start_step, config.steps)
    if not losses and config.resume_from:
        losses.append({"step": final_step, "loss": final_loss})
    _write_training_metrics(
        metrics_path,
        model_config=model_config,
        config=config,
        model_parameters=model_parameters,
        losses=losses,
        validation_losses=validation_losses,
        final_loss=final_loss,
        step=final_step,
        run_timing=_run_timing_payload(
            started_at_epoch_utc=started_at_epoch_utc,
            started_at_monotonic=started_at_monotonic,
            start_step=start_step,
            step=final_step,
            steps=config.steps,
            batch_size=config.batch_size,
        ),
    )
    _write_loss_plot(loss_plot_path, losses, validation_losses)
    _write_training_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        generator=generator,
        model_config=model_config,
        config=config,
        losses=losses,
        validation_losses=validation_losses,
        final_loss=final_loss,
        step=final_step,
        metrics_path=metrics_path,
        loss_plot_path=loss_plot_path,
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


def _run_timing_payload(
    *,
    started_at_epoch_utc: float,
    started_at_monotonic: float,
    start_step: int,
    step: int,
    steps: int,
    batch_size: int,
    now_epoch_utc: float | None = None,
    now_monotonic: float | None = None,
) -> dict[str, float | int | None]:
    now_epoch_utc = time.time() if now_epoch_utc is None else now_epoch_utc
    now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
    elapsed_seconds = max(0.0, now_monotonic - started_at_monotonic)
    completed_steps_this_run = max(0, step - start_step)
    remaining_steps = max(0, steps - step)
    completed_images_this_run = completed_steps_this_run * batch_size
    remaining_images = remaining_steps * batch_size
    steps_per_second = (
        completed_steps_this_run / elapsed_seconds
        if elapsed_seconds > 0.0 and completed_steps_this_run > 0
        else 0.0
    )
    estimated_remaining_seconds = (
        remaining_steps / steps_per_second if steps_per_second > 0.0 else None
    )
    return {
        "started_at_epoch_utc": started_at_epoch_utc,
        "logged_at_epoch_utc": now_epoch_utc,
        "elapsed_seconds": elapsed_seconds,
        "completed_steps_this_run": completed_steps_this_run,
        "completed_images_this_run": completed_images_this_run,
        "remaining_steps": remaining_steps,
        "remaining_images": remaining_images,
        "progress_fraction": step / steps if steps > 0 else 0.0,
        "steps_per_second": steps_per_second,
        "images_per_second": steps_per_second * batch_size,
        "estimated_remaining_seconds": estimated_remaining_seconds,
        "estimated_total_seconds": (
            elapsed_seconds + estimated_remaining_seconds
            if estimated_remaining_seconds is not None
            else None
        ),
        "estimated_finish_epoch_utc": (
            now_epoch_utc + estimated_remaining_seconds
            if estimated_remaining_seconds is not None
            else None
        ),
    }


def _torch_device_payload(torch: Any, device: Any) -> dict[str, int | str | None]:
    if getattr(device, "type", str(device)) != "cuda":
        return {
            "torch_cuda_device_count": 0,
            "torch_device_name": None,
        }
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        return {
            "torch_cuda_device_count": 0,
            "torch_device_name": None,
        }
    return {
        "torch_cuda_device_count": int(cuda.device_count()),
        "torch_device_name": str(cuda.get_device_name(device)),
    }


def _sample_flow_timesteps(
    batch_size: int,
    *,
    torch: Any,
    generator: Any,
    device: Any,
) -> Any:
    logits = torch.randn(batch_size, generator=generator, device="cpu")
    logits = logits * TIMESTEP_LOGIT_NORMAL_STD + TIMESTEP_LOGIT_NORMAL_MEAN
    return torch.sigmoid(logits).to(device)


def _train_step(
    samples: tuple[SatelliteSample, ...],
    *,
    model: Any,
    optimizer: Any,
    torch: Any,
    model_config: ModelConfig,
    device: Any,
    generator: Any,
    mask_drop_probability: float,
) -> float:
    clean, semantic_mask = _load_batch(
        samples,
        torch=torch,
        model_config=model_config,
        device=device,
        random_crop=True,
        generator=generator,
    )
    noise = torch.randn(clean.shape, generator=generator, device="cpu").to(device)
    timestep = _sample_flow_timesteps(
        clean.shape[0],
        torch=torch,
        generator=generator,
        device=device,
    )
    noisy = interpolate_flow_sample(clean, noise, timestep)
    target = flow_velocity(clean, noise)

    if mask_drop_probability > 0.0:
        keep_mask = torch.rand(
            semantic_mask.shape[0],
            generator=generator,
            device="cpu",
        ).to(semantic_mask.device)
        keep_mask = (keep_mask >= mask_drop_probability).float().view(-1, 1, 1, 1)
        semantic_mask = semantic_mask * keep_mask

    optimizer.zero_grad(set_to_none=True)
    prediction = model(noisy, semantic_mask, timestep)
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu().item())


def _write_training_metrics(
    path: Path,
    *,
    model_config: ModelConfig,
    config: TrainingConfig,
    model_parameters: int,
    losses: list[dict[str, float | int]],
    validation_losses: list[dict[str, float | int]],
    final_loss: float,
    step: int,
    run_timing: Mapping[str, float | int | None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model_config": asdict(model_config),
                "training_config": asdict(config),
                "model_parameters": model_parameters,
                "step": step,
                "final_loss": final_loss,
                "run_timing": dict(run_timing),
                "losses": losses,
                "validation_losses": validation_losses,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_loss_plot(
    path: Path,
    losses: list[dict[str, float | int]],
    validation_losses: list[dict[str, float | int]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot

    steps = [int(item["step"]) for item in losses]
    values = [float(item["loss"]) for item in losses]
    smoothed = _exponential_moving_average(values, decay=0.95)

    figure, axis = pyplot.subplots(figsize=(8, 4.5), dpi=140)
    if steps:
        axis.plot(steps, values, color="#94a3b8", linewidth=1.0, alpha=0.55, label="train step")
        axis.plot(steps, smoothed, color="#0f766e", linewidth=2.0, label="train EMA 0.95")
    if validation_losses:
        validation_steps = [int(item["step"]) for item in validation_losses]
        validation_values = [float(item["loss"]) for item in validation_losses]
        axis.plot(
            validation_steps,
            validation_values,
            color="#dc2626",
            marker="o",
            markersize=3,
            linewidth=1.5,
            label="validation",
        )
    axis.set_title("Training Loss")
    axis.set_xlabel("Step")
    axis.set_ylabel("MSE")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path)
    pyplot.close(figure)


def _exponential_moving_average(values: list[float], *, decay: float) -> list[float]:
    if not values:
        return []
    average = values[0]
    smoothed = []
    for value in values:
        average = decay * average + (1.0 - decay) * value
        smoothed.append(average)
    return smoothed


def _write_training_checkpoint(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    generator: Any,
    model_config: ModelConfig,
    config: TrainingConfig,
    losses: list[dict[str, float | int]],
    validation_losses: list[dict[str, float | int]],
    final_loss: float,
    step: int,
    metrics_path: Path,
    loss_plot_path: Path,
    torch: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "generator_state": generator.get_state(),
            "model_config": asdict(model_config),
            "training_config": asdict(config),
            "step": step,
            "losses": losses,
            "validation_losses": validation_losses,
            "final_loss": final_loss,
            "metrics_path": str(metrics_path),
            "loss_plot_path": str(loss_plot_path),
        },
        path,
    )


def _load_training_state(
    path: Path,
    *,
    model: Any,
    optimizer: Any,
    generator: Any,
    torch: Any,
    model_config: ModelConfig,
    device: Any,
) -> tuple[int, float, list[dict[str, float | int]], list[dict[str, float | int]]]:
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    checkpoint_model_config = checkpoint.get("model_config")
    if checkpoint_model_config != asdict(model_config):
        raise ValueError("Resume checkpoint model_config does not match the requested model")
    model.load_state_dict(checkpoint["model_state"])
    optimizer_state = checkpoint.get("optimizer_state")
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    generator_state = checkpoint.get("generator_state")
    if generator_state is not None:
        generator.set_state(generator_state)
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
) -> float:
    was_training = model.training
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 10_000)
    losses: list[float] = []
    with torch.no_grad():
        for samples in validation_batches[: config.validation_batches]:
            clean, semantic_mask = _load_batch(
                samples,
                torch=torch,
                model_config=model_config,
                device=device,
                random_crop=False,
            )
            noise = torch.randn(clean.shape, generator=generator, device="cpu").to(device)
            timestep = _sample_flow_timesteps(
                clean.shape[0],
                torch=torch,
                generator=generator,
                device=device,
            )
            noisy = interpolate_flow_sample(clean, noise, timestep)
            target = flow_velocity(clean, noise)
            prediction = model(noisy, semantic_mask, timestep)
            losses.append(float(torch.nn.functional.mse_loss(prediction, target).cpu().item()))

    if was_training:
        model.train()
    return sum(losses) / len(losses)


def _load_batch(
    samples: tuple[SatelliteSample, ...],
    *,
    torch: Any,
    model_config: ModelConfig,
    device: Any,
    random_crop: bool,
    generator: Any | None = None,
) -> tuple[Any, Any]:
    crop_boxes: list[tuple[int, int, int, int]] = []
    for sample in samples:
        if random_crop:
            if generator is None:
                raise ValueError("generator is required for random crop loading")
            crop_boxes.append(
                _random_square_crop_box(
                    sample,
                    crop_size=model_config.image_size,
                    torch=torch,
                    generator=generator,
                )
            )
        else:
            crop_boxes.append(_center_square_crop_box(sample, crop_size=model_config.image_size))

    images = [
        load_image_tensor(sample.image_path, torch=torch, config=model_config, crop_box=crop_box)
        for sample, crop_box in zip(samples, crop_boxes, strict=True)
    ]
    masks = [
        load_mask_tensor(sample.mask_path, torch=torch, config=model_config, crop_box=crop_box)
        for sample, crop_box in zip(samples, crop_boxes, strict=True)
    ]
    return torch.stack(images).to(device), torch.stack(masks).to(device)


def _random_square_crop_box(
    sample: SatelliteSample,
    *,
    crop_size: int,
    torch: Any,
    generator: Any,
) -> tuple[int, int, int, int]:
    width, height = _aligned_image_size(sample)
    size = min(width, height, crop_size)
    if size < 1:
        raise ValueError(f"Sample has invalid dimensions: {sample.image_path} is {(width, height)}")

    max_left = width - size
    max_top = height - size
    left = (
        0
        if max_left <= 0
        else int(torch.randint(0, max_left + 1, (1,), generator=generator, device="cpu").item())
    )
    top = (
        0
        if max_top <= 0
        else int(torch.randint(0, max_top + 1, (1,), generator=generator, device="cpu").item())
    )
    return left, top, left + size, top + size


def _center_square_crop_box(
    sample: SatelliteSample,
    *,
    crop_size: int,
) -> tuple[int, int, int, int]:
    width, height = _aligned_image_size(sample)
    size = min(width, height, crop_size)
    if size < 1:
        raise ValueError(f"Sample has invalid dimensions: {sample.image_path} is {(width, height)}")
    left = (width - size) // 2
    top = (height - size) // 2
    return left, top, left + size, top + size


def _aligned_image_size(sample: SatelliteSample) -> tuple[int, int]:
    from PIL import Image

    with Image.open(sample.image_path) as image:
        image_size = image.size
    with Image.open(sample.mask_path) as mask:
        mask_size = mask.size
    if image_size != mask_size:
        raise ValueError(
            f"Image and mask sizes must match for aligned crops: "
            f"{sample.image_path} is {image_size}, {sample.mask_path} is {mask_size}"
        )
    return image_size


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
